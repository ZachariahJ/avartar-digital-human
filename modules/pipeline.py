"""Core orchestrator: streaming sentence-by-sentence processing with barge-in support."""

import os
import threading
import logging
import queue
import tempfile
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor

import numpy as np

import config
from modules import asr, clipcache, llm, tts, privacy
# modules.avatar is imported lazily inside render_into/_render_frames, never at
# module scope:
# importing it pulls in torch plus the whole sibling float/ repo (face_alignment,
# torchvision, librosa, the MuseTalk model classes). Voice-only mode must not fail to
# start, or pay that import, because of a repo it will never call.
from modules.privacy import phi, phi_keys
from modules.sbirt import crisis, runtime, state_view, templates
from modules.sbirt.instruments import BY_KEY, InvalidResponse, PRE_SCREEN

logger = logging.getLogger(__name__)


class Segment:
    """One utterance on its way to the browser.

    A segment is NOT a video file. It is a whole audio file plus a live stream
    of JPEG frames, because that is the only shape that gets the first frame out
    fast. MuseTalk needs the ENTIRE utterance's audio up front (whisper features
    are trimmed and padded against the full length, so audio cannot be fed in
    pieces), but it produces frames a batch at a time — so the audio is handed
    over as one URL and the frames arrive progressively.

    The browser plays `audio_url` on one continuous <audio> element and uses its
    currentTime as THE clock, drawing frame floor(t * fps) on a canvas. That is
    what removes the per-chunk seams: nothing is re-encoded, no container is
    restarted, and the audio never stops for the video to catch up. The URL
    points at bytes in clipcache, not a file — no part of an utterance is
    written to disk.

    Ownership: `frames` is filled by the render thread and drained by the state
    poller. The terminating None is mandatory — it is how the poller learns the
    utterance is complete rather than merely slow.
    """

    __slots__ = ("sentence", "audio_url", "fps", "frames", "started",
                 "cancelled", "_t_enqueue")

    def __init__(self, sentence: str = ""):
        self.sentence = sentence
        self.audio_url = None           # set before `started` fires
        self.fps = config.MUSETALK_FPS
        self.frames = queue.Queue()     # jpeg bytes ..., then None
        self.started = threading.Event()  # audio_url is valid; safe to announce
        self.cancelled = threading.Event()
        self._t_enqueue = 0.0

    def open(self, audio_url: str):
        self.audio_url = audio_url
        self.started.set()

    def close(self):
        self.frames.put(None)

    def cancel(self):
        """Barge-in: stop the render and stop forwarding whatever is buffered."""
        self.cancelled.set()
        self.frames.put(None)


# --------------- Fixed-clip keys ---------------
# A fixed utterance is identified by a stable KEY, not by a file path: the clip
# it names lives in RAM (modules/clipcache.py), never on disk. Keys are
# namespaced so a protocol key can never collide with the greeting's.

def crisis_clip_key(category: str) -> str:
    """Cache key for one crisis category's fixed response."""
    return f"crisis.{category}"


def protocol_clip_key(key: str) -> str:
    """Cache key for a fixed protocol utterance (runtime.Say key)."""
    return f"protocol.{key}"


def clip_stamp(text: str) -> str:
    """The full cache key for a rendered clip: the spoken text AND everything
    about how it was rendered. A clip is a function of all of them, so keying on
    text alone was a face-swap trap — repointing config.AVATAR_VIDEO left every
    cached clip replaying the OLD face forever, because no text had changed.

    MUSETALK_FPS is in the stamp because the cached frames ARE a fixed-rate
    sequence: the browser indexes them as floor(currentTime * fps), so replaying
    24fps frames under a 25fps clock desynchronises the whole clip.

    A voice-only clip has no portrait in it, so it is deliberately NOT stamped
    with the fingerprint — swapping the avatar video must not invalidate audio
    that cannot possibly show a face."""
    if not config.ENABLE_VIDEO_AVATAR:
        return f"audio-only\n{text}"
    return (f"avatar:{config.avatar_fingerprint()}\n"
            f"fps:{config.MUSETALK_FPS}\n{text}")


# --------------- Rendering ---------------

def _scratch_audio(audio: bytes) -> str:
    """Spill an utterance's wav to a scratch file, because MuseTalk needs a PATH.

    This is the only disk write left in the speaking path, and it is not a cache:
    whisper's feature extraction is get_audio_feature(audio_path), so the bytes
    have to exist as a file for exactly the length of one render. It goes to
    config.RENDER_SCRATCH_DIR — /dev/shm, i.e. RAM — and every caller deletes it
    in a finally. The janitor sweeps anything a crash leaks.
    """
    os.makedirs(config.RENDER_SCRATCH_DIR, exist_ok=True)
    fd, path = tempfile.mkstemp(suffix=".wav", dir=config.RENDER_SCRATCH_DIR)
    with os.fdopen(fd, "wb") as f:
        f.write(audio)
    return path


def render_into(seg: Segment, audio: bytes, abort=None, collect=None) -> int:
    """Drive `seg` from finished TTS audio. Returns the frame count.

    The audio is published to clipcache first and the segment opened on its URL,
    so the browser can start fetching and playing while MuseTalk is still on its
    first batch. Voice-only mode emits zero frames and the segment IS just the
    audio — the page shows the still portrait, exactly as before.

    `collect`, when given, accumulates the JPEGs as they stream so the caller can
    cache a fixed utterance it just had to render (see Pipeline._speak_dynamic).
    """
    seg.open(clipcache.publish_url(audio))
    if not config.ENABLE_VIDEO_AVATAR:
        seg.close()
        return 0
    from modules import avatar

    def _on_frame(idx, jpeg):
        if collect is not None:
            collect.append(jpeg)
        seg.frames.put(jpeg)

    path = _scratch_audio(audio)
    try:
        return avatar.stream_video(path, _on_frame,
                                   abort=abort or seg.cancelled.is_set)
    finally:
        seg.close()
        try:
            os.remove(path)
        except OSError:
            pass


def _render_frames(audio: bytes, abort=None) -> list:
    """Render an utterance to JPEG frames with no Segment and nobody listening —
    the pre-warm path. Returns the frames rendered so far, which is a PARTIAL
    list if `abort` fired; the caller checks that before caching."""
    from modules import avatar
    frames = []
    path = _scratch_audio(audio)
    try:
        avatar.stream_video(path, lambda idx, jpeg: frames.append(jpeg), abort=abort)
    finally:
        try:
            os.remove(path)
        except OSError:
            pass
    return frames


def fixed_segment(text: str, key: str) -> Segment | None:
    """A ready-to-play Segment for a fixed line, straight out of RAM.

    Returns None on a cache miss. Callers MUST fall back to rendering the line
    (Pipeline._speak_dynamic with cache_key=key), which streams it and fills the
    cache: the cache does not survive a restart, so a miss is an ordinary event
    on the first session after boot, not a failure.

    Frames are loaded into the segment up front rather than streamed: a cached
    clip has no render to wait for, so there is nothing to gain by dribbling
    them out, and a complete segment lets the client start on its first frame.
    """
    clip = clipcache.get_clip(key, clip_stamp(text))
    if clip is None:
        return None
    seg = Segment(text)
    for jpeg in clip.frames:
        seg.frames.put(jpeg)
    seg.open(clipcache.url_for(clip.audio_token))
    seg.close()
    return seg


# --------------- Conversation activity (the pre-warm's GPU yield) ---------------
# The pre-warm renders fixed clips on the SAME GPUs that answer people. These
# three functions are the whole arbitration: a turn marks the conversation live,
# and the pre-warm both waits on this before starting a clip and passes it as the
# render's abort hook — MuseTalk polls abort every batch, so a person who starts
# talking gets the GPU back within one batch instead of after the whole clip.

_activity_lock = threading.Lock()
_active_turns = 0
_last_active = 0.0


def _turn_begin():
    global _active_turns
    with _activity_lock:
        _active_turns += 1


def _turn_end():
    global _active_turns, _last_active
    with _activity_lock:
        _active_turns = max(0, _active_turns - 1)
        _last_active = time.monotonic()


def note_activity():
    """Mark the conversation live without owning a turn — used the moment the
    VAD hears speech, so the pre-warm lets go of the GPU before the turn that
    needs it has even been created."""
    global _last_active
    with _activity_lock:
        _last_active = time.monotonic()


def conversation_busy() -> bool:
    """True while any turn is in flight, or one ended within the idle grace
    period (so the pre-warm doesn't grab a GPU in the gap between two turns of
    the same exchange)."""
    with _activity_lock:
        if _active_turns > 0:
            return True
        idle_for = time.monotonic() - _last_active
    return idle_for < config.CLIP_PREWARM_IDLE_SEC


def _prewarm_abort() -> bool:
    return conversation_busy() or config.SHUTTING_DOWN.is_set()


# --------------- Fixed-clip pre-warm ---------------

def fixed_catalogue() -> list:
    """(clip key, text) for every fixed utterance the protocol can ever speak."""
    from modules.sbirt import templates
    items = [(config.GREETING_CLIP_KEY, config.GREETING_TEXT),
             (config.DECLINE_CLIP_KEY, config.DECLINE_TEXT)]
    items += [(crisis_clip_key(c), t) for c, t in crisis.RESPONSES.items()]
    items += [(protocol_clip_key(k), t)
              for k, t in templates.all_fixed_utterances().items()]
    return items


def _prewarm_one(key: str, text: str, stamp: str) -> str:
    """Render one fixed utterance into the cache. "cached", "preempted" (someone
    started talking — the GPU was handed back, try again later) or "failed"."""
    audio = tts.synthesize(text, config.SHUTTING_DOWN)
    if audio is None:
        return "failed"
    frames = []
    if config.ENABLE_VIDEO_AVATAR:
        frames = _render_frames(audio, abort=_prewarm_abort)
        if _prewarm_abort():
            return "preempted"      # frames are truncated; never cache those
        if not frames:
            return "failed"
    clipcache.put_clip(key, stamp, audio, frames)
    return "cached"


def _wait_until_idle() -> bool:
    """Block until no conversation is in flight. False if we're shutting down."""
    while not config.SHUTTING_DOWN.is_set():
        if not conversation_busy():
            return True
        config.SHUTTING_DOWN.wait(config.CLIP_PREWARM_POLL_SEC)
    return False


def prewarm_fixed_clips():
    """Render every fixed utterance into the in-RAM clip cache, at idle.

    Runs in a background thread for the life of the process. Unlike the old
    on-disk cache this has to happen on EVERY boot, so it is explicitly the
    lowest-priority user of the GPUs: it renders one clip at a time, only while
    conversation_busy() is false, and abandons a clip mid-render the moment
    someone speaks (re-queueing it for a later idle window). A clip that is
    needed before the pre-warm reaches it is simply rendered on demand by the
    turn that needs it, which caches it too — so this loop is an optimisation,
    never a correctness requirement.
    """
    if not config.CLIP_PREWARM:
        logger.info("[prewarm] disabled (CLIP_PREWARM=0)")
        return
    pending = deque(fixed_catalogue())
    attempts = {}
    cached = skipped = 0
    total = len(pending)
    logger.info("[prewarm] %d fixed utterances to render at idle", total)
    while pending and not config.SHUTTING_DOWN.is_set():
        if not _wait_until_idle():
            return
        key, text = pending.popleft()
        stamp = clip_stamp(text)
        if clipcache.has_clip(key, stamp):
            cached += 1
            continue
        try:
            result = _prewarm_one(key, text, stamp)
        except Exception:
            logger.exception("[prewarm] %s raised; skipping", key)
            result = "failed"
        if result == "cached":
            cached += 1
        elif result == "preempted":
            logger.info("[prewarm] yielded the GPU mid-render of %s; "
                        "re-queued for the next idle window", key)
            pending.appendleft((key, text))   # front: it was next in line
        else:
            attempts[key] = attempts.get(key, 0) + 1
            if attempts[key] < config.CLIP_PREWARM_MAX_ATTEMPTS:
                pending.append((key, text))   # back: let the others through
            else:
                skipped += 1
                logger.warning("[prewarm] giving up on %s after %d attempts "
                               "(it will be rendered on demand)",
                               key, attempts[key])
    if not config.SHUTTING_DOWN.is_set():
        logger.info("[prewarm] complete: %d/%d clips cached, %d skipped; %s",
                    cached, total, skipped, clipcache.stats())


class Pipeline:
    """State machine: idle → listening → processing → speaking → idle"""

    def __init__(self, audit_key: str = "default"):
        self.audit_key = audit_key  # pseudonymous session id for audit records
        self.state = "idle"  # idle, listening, processing, speaking
        self.cancel_event = threading.Event()
        self.video_queue = queue.Queue()
        # THE conversation record: {"role", "content"} dicts. The frontend
        # shows all of it; the LLM API gets a derived sliding-window suffix
        # (_api_window). Deterministic scores live in self.clinical, so window
        # trimming can never corrupt triage.
        self.chat_history = []
        # Structured patient profile (age, sex, substances, screening scores, ...),
        # injected into the prompt every turn so it survives history-window trimming
        # and the model never loses key clinical facts.
        self.patient = {}
        # Set when the user declines consent: the server then turns the mic off and
        # tells the client to stop, ending the session until Start is pressed again.
        self.ended = False
        # Dynamic (non-cached) TTS+MuseTalk renders this session — the generation
        # budget observable (T18). Fixed content contributes zero to THIS counter
        # however it was produced.
        self.dynamic_renders = 0
        # Fixed utterances this session had to render because the clip cache was
        # cold (the cache is RAM, so the first session after a restart pays for
        # whatever the pre-warm hasn't reached yet). Counted separately: these
        # are verbatim protocol lines, not generated content, and folding them
        # into dynamic_renders would make T18 unreadable.
        self.fixed_renders = 0
        # THE clinical state: protocol node, coded answers, deterministic
        # scores/zones, readiness. The machine (modules/sbirt/runtime.py)
        # decides every transition; the LLM never does.
        self.clinical = runtime.ClinicalSession()
        runtime.start(self.clinical)
        self._lock = threading.Lock()
        # Serializes protocol turns: coding -> machine advance -> delivery.
        # Two concurrent turns (voice + typed, or a barge-in racing a slow
        # coder) must never both advance the clinical machine; the stale turn
        # re-checks _aborted() under this lock and drops out.
        self._protocol_lock = threading.Lock()
        self._processing_thread = None
        # Monotonic turn id: each new utterance bumps it. A response only touches
        # shared state (enqueue video) while it still owns the current turn, so a
        # barged-in response can't leak stale segments into the next turn even
        # after cancel_event is cleared for the new turn.
        self._turn = 0
        # perf_counter() at the moment the user stopped speaking — the T0 for the
        # latency waterfall logged through the rest of the turn.
        self._t0 = 0.0
        # Pause-split continuation merge (voice): remember the in-flight
        # utterance's audio until the machine ACTS on its words; if new
        # speech supersedes the turn first, that audio is prepended to the
        # next speech_end so the whole sentence reaches ASR as one piece —
        # instead of the first half silently vanishing. Guarded by its own
        # lock: these ops run on the WS thread and must never block on
        # _protocol_lock (held for seconds during LLM calls).
        self._carry_lock = threading.Lock()
        self._pending_voice = None   # (turn id, audio) — unconsumed voice turn
        self._carry_audio = None     # carried first half awaiting the merge
        # Every Segment created for the CURRENT turn, so a barge-in can reach
        # into the ones already handed to the poller and cancel them too —
        # emptying video_queue alone would leave the segment being played right
        # now streaming happily into a browser that has moved on.
        self._live_lock = threading.Lock()
        self._live: list[Segment] = []

    def _new_segment(self, sentence: str) -> Segment:
        seg = Segment(sentence)
        with self._live_lock:
            self._live.append(seg)
        return seg

    def _enqueue(self, seg: Segment):
        """Hand a segment to the poller. Called BEFORE its render finishes: the
        poller forwards frames as they land, so delivery overlaps generation."""
        seg._t_enqueue = time.perf_counter()
        self.video_queue.put(seg)
        self.state = "speaking"

    def _flush(self):
        """Cascade flush — the whole response is abandoned, at every stage at once.

        Ordering matters. Cancelling the live segments FIRST makes every
        in-flight MuseTalk render see its abort flag on the next batch (and the
        next frame), and makes TTS drop its partial audio, so the GPU and the
        network stop producing before the queue is emptied. Draining first would
        leave the renderer busily filling queues nobody reads.

        cancel_event (set by the caller) is what stops the LLM stream; this stops
        everything downstream of it.
        """
        with self._live_lock:
            live, self._live = self._live, []
        for seg in live:
            seg.cancel()
        while True:
            try:
                item = self.video_queue.get_nowait()
            except queue.Empty:
                break
            if isinstance(item, Segment):
                item.cancel()

    def _aborted(self, turn):
        """True if this response was cancelled or superseded by a newer turn."""
        return self.cancel_event.is_set() or turn != self._turn

    def _consume_utterance(self, turn):
        """The machine is acting on this voice turn's words — no longer
        carryable (a later barge-in must never double-process them)."""
        with self._carry_lock:
            if self._pending_voice is not None and self._pending_voice[0] == turn:
                self._pending_voice = None

    def _clear_carry(self):
        """Abandon any pause-split fragment (Stop / typed input / fresh
        session): a stale first half must never prepend to a later utterance."""
        with self._carry_lock:
            self._pending_voice = self._carry_audio = None

    def on_speech_start(self):
        """Called by main.ws_audio when user starts speaking (barge-in)."""
        # Someone is talking: the clip pre-warm must let go of its GPU now, not
        # when the resulting turn is created a second or two from now.
        note_activity()
        if self.state in ("processing", "speaking"):
            logger.info("Barge-in detected! Cancelling current response.")
            # Continuation, not interruption: if the turn being cancelled is a
            # voice utterance whose words were never consumed (still in
            # ASR/NLU flight — the avatar hasn't acted on them), the person is
            # finishing their own sentence after a pause the VAD read as a
            # turn end. Carry that audio for the next speech_end.
            with self._carry_lock:
                pv = self._pending_voice
                if pv is not None and pv[0] == self._turn:
                    self._carry_audio, self._pending_voice = pv[1], None
                    logger.info("[continuation] pause-split: carrying the "
                                "unconsumed first half into the next utterance")
            self.cancel_event.set()
            self._turn += 1  # invalidate the in-flight response immediately
            self._flush()    # LLM stream, TTS, MuseTalk, frame queues — all of it
            # History is pipeline-owned; the in-flight producer's finally commits
            # whatever was generated so far, so no truncation is needed here.

        self.state = "listening"

    def cancel_response(self):
        """Stop any in-flight response immediately and end at idle (used by Stop and
        by text-send interruption). Drops queued clips; leaves histories intact."""
        if self.state in ("processing", "speaking"):
            self.cancel_event.set()
            self._turn += 1
            self._flush()
        # An explicit Stop abandons any pause-split fragment too.
        self._clear_carry()
        self.state = "idle"

    def on_speech_end(self, audio_array):
        """Called by main.ws_audio when user finishes speaking.
        audio_array: float32 numpy array at 16kHz.
        """
        self._t0 = time.perf_counter()
        # Pause-split continuation: prepend the carried first half (if any)
        # so ASR transcribes the whole utterance in one piece.
        with self._carry_lock:
            if self._carry_audio is not None:
                audio_array = np.concatenate([self._carry_audio, audio_array])
                self._carry_audio = None
                logger.info("[continuation] resumed after a pause: merged "
                            "utterance is now %.2fs", len(audio_array) / 16000)
        self.state = "processing"
        self.cancel_event.clear()
        self._turn += 1
        turn = self._turn
        with self._carry_lock:
            self._pending_voice = (turn, audio_array)

        # Run processing in background thread. The turn is marked live HERE,
        # not inside the thread, so the pre-warm stops taking new GPU work from
        # this instant rather than one scheduling delay later; the thread's
        # finally closes it.
        _turn_begin()
        self._processing_thread = threading.Thread(
            target=self._process_speech, args=(audio_array, turn), daemon=True
        )
        self._processing_thread.start()

    def on_speech_end_text(self, text):
        """Text input: skip ASR and feed text directly into pipeline."""
        self._t0 = time.perf_counter()
        # Typing supersedes any pause-split voice fragment: never prepend a
        # stale first half to a LATER voice utterance.
        self._clear_carry()
        self.state = "processing"
        self.cancel_event.clear()
        self._turn += 1
        turn = self._turn

        _turn_begin()
        self._processing_thread = threading.Thread(
            target=self._process_text, args=(text, turn), daemon=True
        )
        self._processing_thread.start()

    # ---------- Proactive greeting (counselor leads the conversation) ----------
    def start_greeting(self):
        """Kick off the counselor's opening turn with NO user input, so the avatar
        greets and asks the first SBIRT question instead of waiting to be spoken
        to. Runs the same synthesis path as a normal turn."""
        self._t0 = time.perf_counter()
        self._clear_carry()   # a fresh session never inherits a voice fragment
        self.state = "processing"
        self.cancel_event.clear()
        self._turn += 1
        turn = self._turn
        _turn_begin()
        self._processing_thread = threading.Thread(
            target=self._process_greeting, args=(turn,), daemon=True
        )
        self._processing_thread.start()

    def _process_greeting(self, turn):
        """Deliver the fixed opening from the cached clip — no LLM/TTS/MuseTalk. Records
        it as the assistant's first turn so the conversation flows straight into the
        user's yes/no consent reply."""
        try:
            if self._aborted(turn):
                self.state = "idle"
                return
            text = config.GREETING_TEXT
            key = config.GREETING_CLIP_KEY
            seg = fixed_segment(text, key)
            # Record the fixed opening as the assistant's first turn in both histories.
            self._history_set_assistant(text)
            # Fresh protocol run: the machine starts at the consent expectation.
            self.clinical = runtime.ClinicalSession()
            runtime.start(self.clinical)
            if seg and not self._aborted(turn):
                self._enqueue(seg)
                self.video_queue.put(None)
            elif not self._aborted(turn):
                # Cache miss — the pre-warm hasn't reached the greeting yet (or
                # the cache was cleared). Render it now, which caches it for
                # every later session; the first frame still streams out fast.
                if self._speak_dynamic(text, turn, cache_key=key) is not None:
                    self.video_queue.put(None)
                else:
                    # Render failed too — the text greeting still shows.
                    self.state = "idle"
            else:
                self.state = "idle"
        except Exception as e:
            logger.error(f"Greeting error: {e}", exc_info=True)
            self.state = "idle"
        finally:
            _turn_end()

    def _deliver_decline(self, user_text, turn):
        """User declined consent: record their reply + the FIXED thank-you line, play
        the cached decline clip, and end the turn. No screening, no dynamic LLM."""
        self._history_begin(user_text)          # record the user's "no"
        text = config.DECLINE_TEXT
        key = config.DECLINE_CLIP_KEY
        seg = fixed_segment(text, key)
        self._history_set_assistant(text)
        if seg and not self._aborted(turn):
            self._enqueue(seg)
            self.video_queue.put(None)
        elif not self._aborted(turn):
            # Cache miss: render the fixed line now (and cache it) rather than
            # end the session on silence.
            if self._speak_dynamic(text, turn, cache_key=key) is not None:
                self.video_queue.put(None)
            else:
                self.state = "idle"
        else:
            self.state = "idle"
        # Consent declined: end the session (the server turns the mic off + tells the
        # client to stop) now that the goodbye clip is queued for delivery.
        self.ended = True

    def _deliver_crisis(self, user_text, hit, turn):
        """Deterministic crisis net fired: speak the FIXED response for the
        category from its cached clip — no LLM anywhere on this path — and stay
        in the conversation (the counselor's crisis protocol owns later turns).
        Falls back to the normal TTS+MuseTalk render if the cached clip is missing,
        and to on-screen text if even that fails; the fixed TEXT always lands in
        both histories either way."""
        logger.warning("[crisis] deterministic net fired: category=%s pattern=%s",
                       hit.category, hit.pattern)  # no user text in the log
        self._consume_utterance(turn)   # acting on these words: no longer carryable
        self._history_begin(user_text)
        # Pause the clinical protocol permanently for this session; later
        # turns run the full counselor with the crisis protocol.
        runtime.enter_crisis(self.clinical)
        text = crisis.RESPONSES[hit.category]
        key = crisis_clip_key(hit.category)
        seg = fixed_segment(text, key)
        self._history_set_assistant(text)
        if seg and not self._aborted(turn):
            self._enqueue(seg)
            self.video_queue.put(None)
        elif not self._aborted(turn):
            # Cache miss (e.g. pre-warm hasn't reached it): render it now rather
            # than stay silent — and cache it for the rest of the process.
            if self._speak_dynamic(text, turn, cache_key=key) is not None:
                self.video_queue.put(None)
            else:
                self.state = "idle"
        else:
            self.state = "idle"

    # ---------- History management (ONE history; API view derived) ----------
    # chat_history is the single conversation record. What the LLM API sees is
    # derived from it on demand (_api_window): the most recent
    # LLM_HISTORY_MAX_MESSAGES entries, never starting on an assistant turn.
    # One list, no lockstep invariant to break on barge-in or LLM errors.
    def _api_window(self):
        """The sliding-window suffix of chat_history sent to the LLM API.
        Caller must hold self._lock."""
        win = [dict(m) for m in
               self.chat_history[-config.LLM_HISTORY_MAX_MESSAGES:]]
        if win and win[0]["role"] == "assistant":
            del win[0]   # never orphan a reply from its user prompt
        return win

    def _history_begin(self, user_text):
        """Append the user turn and return the API message list. If the
        previous turn's user message is still unanswered (e.g. speech split by
        a pause into two segments, so the first half hasn't been replied to
        yet), MERGE the new text into it — never drop it. This preserves
        everything the user said, reassembles the paused sentence into one
        turn, and still avoids sending two user messages in a row to the API."""
        with self._lock:
            hist = self.chat_history
            if hist and hist[-1]["role"] == "user":
                hist[-1] = {"role": "user",
                            "content": (hist[-1]["content"].rstrip()
                                        + " " + user_text).strip()}
            else:
                hist.append({"role": "user", "content": user_text})
            window = self._api_window()
            messages = llm.build_messages(window, dict(self.patient))
        # Fire-and-forget structured extraction to keep the patient profile current.
        # Runs off the hot path (never blocks TTS/display) and applies to the NEXT
        # turn's prompt, so age/sex/screening facts survive history-window trimming.
        threading.Thread(target=self._extract_patient, args=(window,),
                         name="patient-extract", daemon=True).start()
        return messages

    def _extract_patient(self, history_snapshot):
        """Merge any newly-extracted patient facts into the profile (background)."""
        facts = llm.extract_patient_facts(history_snapshot)
        if not facts:
            return
        with self._lock:
            for k, v in facts.items():
                if v in (None, "", [], {}):
                    continue
                self.patient[k] = v
        logger.info("[patient] profile now: %s", phi_keys(self.patient))

    def _history_set_assistant(self, text):
        """Create or update the current assistant turn (display and API memory
        are the same list, so they can never disagree)."""
        with self._lock:
            hist = self.chat_history
            if hist and hist[-1]["role"] == "assistant":
                hist[-1]["content"] = text
            else:
                hist.append({"role": "assistant", "content": text})

    def _process_speech(self, audio_array, turn):
        """Full pipeline: ASR → LLM stream → TTS+MuseTalk (pipelined) → video queue."""
        try:
            # Step 1: ASR
            if self._aborted(turn):
                self.state = "idle"
                return

            user_text = asr.transcribe_array(audio_array, sample_rate=16000)
            logger.info("[latency] ASR done at +%.2fs -> %s",
                        time.perf_counter() - self._t0, phi(user_text))

            if not user_text.strip():
                self.state = "idle"
                return

            if self._aborted(turn):
                self.state = "idle"
                return

            # Crisis safety net FIRST (overrides everything, incl. the consent
            # gate): deterministic patterns, unioned with the LLM's own protocol.
            hit = crisis.detect(user_text)
            if hit:
                self._deliver_crisis(user_text, hit, turn)
                return

            # Every other turn goes through the clinical state machine: the
            # coder maps the words onto the expected input, the machine decides
            # the transition, the LLM at most phrases bounded utterances.
            self._protocol_turn(user_text, turn)

        except Exception as e:
            logger.error(f"Pipeline error: {e}", exc_info=True)
            self.state = "idle"
        finally:
            _turn_end()

    def _process_text(self, user_text, turn):
        """Text-only pipeline: skip ASR, go straight to LLM → TTS+MuseTalk (pipelined)."""
        try:
            if self._aborted(turn):
                self.state = "idle"
                return

            # Crisis safety net FIRST (same as the voice path).
            hit = crisis.detect(user_text)
            if hit:
                self._deliver_crisis(user_text, hit, turn)
                return

            # Same protocol path as the voice turns.
            self._protocol_turn(user_text, turn)

        except Exception as e:
            logger.error(f"Pipeline error (text): {e}", exc_info=True)
            self.state = "idle"
        finally:
            _turn_end()

    # ---------- Protocol turns: coder -> state machine -> bounded rendering ----------

    def _current_question(self):
        """(question_text, options) for the machine's current option
        expectation, from the structured instrument data."""
        exp = self.clinical.expect
        if exp.kind != "option":
            return None, None
        if exp.instrument == "prescreen":
            item = PRE_SCREEN[exp.item_index].item
        else:
            item = BY_KEY[exp.instrument].items[exp.item_index]
        return item.text, item.options

    def _last_question_text(self):
        """The most recent question the avatar asked (context for the NLU
        turn call): last fixed Say of the current pause, else the assistant's
        last spoken text (compose asks), else the greeting's consent ask."""
        step = self.clinical.last_step
        if step:
            for utt in reversed(step.utterances):
                if isinstance(utt, runtime.Say):
                    return utt.text
        with self._lock:
            for m in reversed(self.chat_history):
                if m["role"] == "assistant" and m["content"].strip():
                    return m["content"]
        return "May I ask you some questions about your health?"

    def _turn_facts(self):
        """Grounding for the NLU turn's replies: ONLY what the machine knows
        deterministically. This is what lets a user's question ('have we
        discussed the standard drink?') be answered honestly from state
        instead of re-asking the machine's own question."""
        c = self.clinical
        facts = {
            "current_phase": c.node,
            "standard_drink_definition_discussed":
                "alcohol.edu.standard_drink" in c.covered,
            "drinking_limits_discussed": "alcohol.edu.limits" in c.covered,
            "permissions_declined_so_far": list(c.declined),
            "active_topic": c.arm,
        }
        # Content of the education actually delivered, so a "you spoke too
        # fast / say that again" turn can honestly re-give the key facts
        # instead of ignoring the request (the facts block is the reply's
        # ONLY permitted factual source).
        for unit_key, fact_key in (
                ("alcohol.edu.standard_drink", "standard_drink_definition"),
                ("alcohol.edu.limits", "recommended_drinking_limits")):
            if unit_key in c.covered:
                facts[fact_key] = templates.FIXED[unit_key]
        # Mid-instrument: what they already answered, so a correction turn
        # ("actually it's more like three times a week") can name its target
        # item (T21). Deterministic state only — item text + coded label.
        exp = c.expect
        if (exp.kind == "option" and exp.instrument
                and exp.instrument != "prescreen"):
            items = BY_KEY[exp.instrument].items
            facts["answers_already_given"] = [
                {"item": i, "question": items[i].text,
                 "answer": items[i].options[code].label}
                for i, code in sorted(
                    c.responses.get(exp.instrument, {}).items())]
        return facts

    def _protocol_turn(self, user_text, turn):
        """ONE NLU+voice call classifies the utterance relative to the current
        ask (answer / continuation / question / tangent / crisis / unclear),
        codes it if it is an answer, and produces this turn's bounded reply.
        Only a VALIDATED answer advances the deterministic machine; every
        other action holds position (T12). The whole turn is serialized under
        _protocol_lock so a superseded turn can never advance the machine
        after a newer one already has."""
        with self._protocol_lock:
            if self._aborted(turn):
                return
            clinical = self.clinical

            # In-crisis sessions: the protocol stays paused; every turn runs the
            # full counselor (its prompt carries the complete crisis protocol).
            if clinical.crisis:
                self._consume_utterance(turn)
                messages = self._history_begin(user_text)
                self.state = "processing"
                self._run_crisis_synthesis(messages, turn)
                return

            exp = clinical.expect
            if exp.kind == "end":
                self._consume_utterance(turn)
                # Session already closed/declined; stay warm, done.
                return self._deliver_step(user_text, runtime.Step(
                    clinical.node, (runtime.LLMSay(
                        "The screening session is already complete. In one warm "
                        "sentence, acknowledge what the person said and remind "
                        "them their provider will follow up with them."),),
                    clinical.expect), turn)

            with self._lock:
                history = self._api_window()
                patient = dict(self.patient)
            out = llm.turn(user_text, exp,
                           ask_text=self._last_question_text(),
                           history=history, patient=patient,
                           facts=self._turn_facts(),
                           interview_state=state_view.render_interview_state(
                               clinical))

            # The NLU may have taken a slow LLM round-trip; if a newer turn
            # started meanwhile (barge-in, typed message), don't touch the
            # machine — an unconsumed voice utterance stays carryable so a
            # pause-split continuation can merge it (modules/carry.py).
            if self._aborted(turn):
                return
            # Committing to act on these words: from here on, new speech is a
            # fresh turn (or a real barge-in), never a continuation-merge.
            self._consume_utterance(turn)

            if out.action == "crisis":
                # NLU-flagged crisis (union with the deterministic net, which
                # already ran in _process_speech/_process_text): pause the
                # protocol permanently; this and every later turn follow the
                # crisis protocol.
                logger.warning("[crisis] NLU flagged crisis at node %s",
                               clinical.node)
                runtime.enter_crisis(clinical)
                return self._deliver_step(
                    user_text, runtime.crisis_step(clinical), turn)

            if out.action == "abort":
                # T22: the user wants to stop the whole session. Close
                # gracefully with the fixed goodbye (no retention attempt),
                # keep everything coded so far, and end the session like a
                # consent decline (mic off via `ended`).
                step = runtime.enter_abort(clinical)
                self._deliver_step(user_text, step, turn)
                self.ended = True
                return

            if out.action == "correction":
                # T21: overwrite the earlier item, let skips/score re-derive,
                # re-pose the (possibly changed) current item. An inapplicable
                # target (not an answered item of the active instrument)
                # holds and clarifies instead of moving anything.
                try:
                    step = runtime.correct(clinical, out)
                except InvalidResponse:
                    logger.exception("correction failed; holding")
                    step = None
                if step is None:
                    return self._hold(user_text, out.reply, turn)
                return self._deliver_step(user_text, step, turn,
                                          ack=out.reply)

            if out.action == "answer":
                if exp.ask_key == "consent.opening":
                    # THE study consent (the greeting's ask) -> audit trail.
                    privacy.record_consent(
                        self.audit_key, "yes" if out.code == 1 else "no")
                if exp.kind == "confirm":
                    # T20: verdict on the read-back — yes commits the held
                    # code, no re-collects the same item.
                    step = runtime.resolve_confirm(clinical,
                                                   yes=(out.code == 1))
                    return self._deliver_step(user_text, step, turn,
                                              ack=out.reply)
                reason = runtime.confirm_reason(clinical, out)
                if reason is not None:
                    # T20/F5: the read-back is the exception — it fires only
                    # when a conversion assumption entered the coding, the
                    # value sat near a bucket edge, the answer contradicts an
                    # earlier one, or nothing deterministic vouches for a
                    # semantic code on a score-critical item.
                    step = runtime.request_confirm(clinical, out, reason)
                    return self._deliver_step(user_text, step, turn,
                                              ack=out.reply)
                try:
                    step = runtime.advance(clinical, out)
                except (runtime.ProtocolError, InvalidResponse):
                    # A wiring bug must not strand the user: log loudly, then
                    # re-ask instead of guessing or going silent.
                    logger.exception("protocol advance failed; re-asking")
                    return self._hold(user_text, "", turn)
                if clinical.node == "declined":
                    # Machine recorded the decline; the fixed decline path
                    # speaks it and ends the session (mic off via `ended`).
                    return self._deliver_decline(user_text, turn)
                return self._deliver_step(user_text, step, turn,
                                          ack=out.reply)

            if out.action == "continuation":
                # The two-breath answer: fold it into the previous capture,
                # keep the machine exactly where it is, and let the reply
                # re-pose the pending ask (never a duplicate re-question).
                runtime.absorb(clinical, out)
                return self._hold(user_text, out.reply, turn)

            if out.action == "dont_know":
                # T25/F1: probe ONCE with the manual's recall anchor, then
                # take the missing-data exit (F2) — never a re-ask loop.
                if runtime.note_stall(clinical) >= runtime.DONT_KNOW_LIMIT:
                    return self._deliver_missing(user_text, "dont_know", turn)
                return self._hold_probe(user_text, turn)

            if out.action == "unclear":
                # F2: clarifications are finite — at the limit the item is
                # recorded as unanswered and the protocol moves on.
                if runtime.note_stall(clinical) >= runtime.UNCLEAR_LIMIT:
                    return self._deliver_missing(user_text, "no_answer", turn)
                return self._hold(user_text, out.reply, turn)

            # question / tangent: machine holds position; the reply
            # answers/acknowledges and re-poses the current ask.
            return self._hold(user_text, out.reply, turn)

    def _deliver_missing(self, user_text, reason, turn):
        """F2's guaranteed terminus: mark the current pause missing and let
        the protocol continue (an unanswerable consent gate degrades to its
        decline path, which may end the session)."""
        step = runtime.mark_missing(self.clinical, reason)
        if self.clinical.node == "declined":
            return self._deliver_decline(user_text, turn)
        return self._deliver_step(user_text, step, turn)

    def _hold_probe(self, user_text, turn):
        """First dont_know at a pause: ONE assisted-recall attempt per the
        WHO manual (p.18 — help the person estimate, anchored to their
        heaviest period in the past year), always with the skip offer so
        declining again is easy. The machine does not move."""
        exp = self.clinical.expect
        if exp.kind == "option":
            q, _ = self._current_question()
            instruction = (
                "The person says they don't know or can't remember. In one "
                "or two gentle sentences, help them estimate: suggest "
                "thinking about the period in the past year when they were "
                f"drinking or using the most, briefly re-ask the substance "
                f"of {q!r} in fresh words, and mention that a rough guess "
                "is fine — or we can skip it and move on.")
        elif exp.kind == "number":
            instruction = (
                "The person says they don't know. In one gentle sentence, "
                "say it doesn't have to be exact and ask for whatever "
                "number from 0 to 10 feels closest — or offer to skip it.")
        else:
            # Gates and read-backs: re-pose the pending ask as-is.
            return self._deliver_step(
                user_text, runtime.repeat_step(self.clinical), turn)
        return self._deliver_step(
            user_text,
            runtime.Step(self.clinical.node,
                         (runtime.LLMSay(instruction),), exp),
            turn)

    def _hold(self, user_text, reply, turn):
        """Speak a bounded hold-turn WITHOUT touching the machine: the NLU's
        reply if it produced one, else a deterministic re-ask built from the
        current expectation (the guess-free fallback when the model failed)."""
        exp = self.clinical.expect
        if reply:
            utterance = runtime.Speak(reply)
        else:
            question = self._last_question_text()
            if exp.kind == "option":
                q, options = self._current_question()
                labels = "; ".join(o.label for o in options)
                instruction = (
                    "The person's answer didn't clearly match one of the "
                    f"answer choices. In one or two short sentences, gently "
                    f"re-ask the substance of {q!r} IN DIFFERENT WORDS than "
                    f"before — you may briefly mention the choices "
                    f"({labels}). Do not suggest which one to pick.")
            elif exp.kind == "number":
                instruction = ("In one short sentence, gently ask again for "
                               "a single number from 0 to 10.")
            else:
                instruction = (f"The person's answer to {question!r} wasn't "
                               "clear. In one short sentence, ask again in "
                               "different words than the question was asked "
                               "before.")
            utterance = runtime.LLMSay(instruction)
        self._deliver_step(
            user_text,
            runtime.Step(self.clinical.node, (utterance,), exp),
            turn)

    def _speak_dynamic(self, text, turn, cache_key=None):
        """TTS + streamed MuseTalk for an utterance the clip cache didn't have;
        the Segment, or None on failure/cancel.

        The segment is enqueued BEFORE the render runs, which is the whole
        latency win: the poller starts forwarding frames after the first UNet
        batch while this thread is still generating the rest. It then blocks
        until the render finishes, so utterances stay strictly in order.

        `cache_key` marks the text as FIXED — the caller found it missing from
        the cache. The frames are then collected as they stream and the finished
        clip is stored, so a fixed line costs at most ONE render per process even
        if the pre-warm never got to it. Only a render that ran to completion is
        cached: a barge-in leaves a truncated frame list, and replaying half an
        utterance forever is far worse than re-rendering it.

        Counts dynamic renders (T18): generated content increments
        dynamic_renders, a cold fixed line increments fixed_renders."""
        if cache_key is None:
            self.dynamic_renders += 1
            logger.info("[latency] dynamic render #%d this session",
                        self.dynamic_renders)
        else:
            self.fixed_renders += 1
            logger.info("[latency] fixed clip %s not cached; rendering it "
                        "(#%d cold fixed render this session)",
                        cache_key, self.fixed_renders)
        seg = self._new_segment(text)
        audio = tts.synthesize(text, self.cancel_event)
        if audio is None or self._aborted(turn):
            seg.cancel()
            return None
        self._enqueue(seg)
        frames = [] if cache_key else None
        render_into(seg, audio, abort=seg.cancelled.is_set, collect=frames)
        if (cache_key and not seg.cancelled.is_set() and not self._aborted(turn)
                and (frames or not config.ENABLE_VIDEO_AVATAR)):
            # Adopt the blob this segment is already playing from rather than
            # storing the same mp3 a second time.
            clipcache.put_clip(cache_key, clip_stamp(text), audio, frames,
                               token=clipcache.token_from_url(seg.audio_url))
        return seg

    def _deliver_step(self, user_text, step, turn, ack=""):
        """Speak one machine step: fixed utterances come from the shared clip
        cache (rendered once, reused across sessions); Speak utterances are
        pre-resolved dynamic text (the NLU turn's acknowledgment); LLMSay
        utterances are phrased by the bounded LLM then rendered. Enqueued
        strictly in order. `ack` (when the turn was an answer) is prepended
        as its own short Speak so the person hears they were heard BEFORE the
        next protocol content."""
        self._history_begin(user_text)
        self.state = "processing"
        utterances = step.utterances
        if ack:
            utterances = (runtime.Speak(ack),) + tuple(utterances)
        spoken = []
        for utt in utterances:
            if self._aborted(turn):
                return
            # `seg` is None for a cached line whose text still has to be spoken
            # by the dynamic path below; `pending` marks that case. Fixed lines
            # are already complete when fixed_segment() returns, dynamic ones
            # stream while _speak_dynamic blocks.
            seg, pending, clip_key = None, False, None
            if isinstance(utt, runtime.Say):
                text = utt.text
                clip_key = protocol_clip_key(utt.key)
                seg = fixed_segment(text, clip_key)
                pending = seg is None       # cache miss -> render it below
            elif isinstance(utt, runtime.Speak):
                text = utt.text
                if not text.strip():
                    continue
                pending = True
            else:
                with self._lock:
                    history = self._api_window()
                    patient = dict(self.patient)
                text = llm.phrase_utterance(utt.instruction, history, patient)
                if not text.strip():
                    continue          # bounded utterance failed -> skip, protocol continues
                pending = True
            if self._aborted(turn):
                return
            spoken.append(text)
            # Text lands in the chat even if this clip failed to render.
            self._history_set_assistant(" ".join(spoken))
            if seg is not None:
                self._enqueue(seg)
            elif pending:
                self._speak_dynamic(text, turn, cache_key=clip_key)
        if not self._aborted(turn):
            self.video_queue.put(None)
            self.state = "speaking"
            if step.expect.kind == "end":
                # Terminal node (flow.End): the close was just queued, so the
                # session is OVER — end it the same way a consent decline does
                # (server drops the mic, client stops capturing and resets to
                # Start once the goodbye finishes playing). Without this the
                # counselor said its goodbye and then kept listening forever,
                # and every further utterance burnt an LLM+TTS turn on the
                # "session is already complete" reply below.
                self.ended = True

    def _run_crisis_synthesis(self, messages, turn):
        """CRISIS turns only — the sole remaining full-LLM synthesis path (every
        normal turn goes through the clinical protocol instead). Streams
        sentences from the LLM (chat text appears live in
        <1s) AND render each sentence's TTS+MuseTalk concurrently across the whole
        GPU pool, while enqueuing strictly in sentence order.

        A producer thread pulls sentences off the LLM stream and submits each as a
        TTS->MuseTalk job to a pool sized to len(MUSETALK_GPUS); this consumer reads the
        resulting futures IN ORDER and enqueues the finished clips. So sentence 1
        starts playing after just 1 TTS + 1 MuseTalk render, while sentences 2..N are already
        rendering on the other GPUs -> no stall between segments.
        """
        futures_q = queue.Queue()
        SENTINEL = object()
        n_gpus = max(1, len(config.MUSETALK_GPUS))

        def _render(seg, idx):
            """Fill one segment on a pool thread. The consumer announces it the
            moment `seg.started` fires (audio ready), so frames stream out of
            here while this is still running."""
            if self._aborted(turn) or config.SHUTTING_DOWN.is_set():
                seg.cancel()
                return
            t_tts0 = time.perf_counter()
            audio = tts.synthesize(seg.sentence, self.cancel_event)
            if audio is None or self._aborted(turn):
                seg.cancel()
                return
            t_render0 = time.perf_counter()
            # The mp3 IS the segment's audio, not an intermediate: render_into
            # publishes it to clipcache and the browser fetches it from there.
            render_into(seg, audio, abort=seg.cancelled.is_set)
            logger.info("[latency] seg %d rendered: tts=%.2fs render=%.2fs",
                        idx, t_render0 - t_tts0, time.perf_counter() - t_render0)

        def _producer(executor):
            """Pull sentences off the LLM stream, submit renders, update chat live."""
            full_response = ""
            try:
                for idx, sentence in enumerate(
                    llm.chat_stream(messages, cancel_event=self.cancel_event)
                ):
                    if self._aborted(turn) or config.SHUTTING_DOWN.is_set():
                        break
                    if idx == 0:
                        logger.info("[latency] LLM first sentence at +%.2fs",
                                    time.perf_counter() - self._t0)
                    full_response += sentence
                    logger.info("LLM sentence: %s", phi(sentence))

                    # Live update BOTH histories so display and API memory agree.
                    self._history_set_assistant(full_response)

                    # The Segment is created HERE, on the ordering thread, so
                    # the consumer can hold it before the render has begun.
                    seg = self._new_segment(sentence)
                    futures_q.put((executor.submit(_render, seg, idx), seg))
            except Exception:
                logger.exception("streaming producer failed")
            finally:
                # Finalize history only if we still own the turn. If superseded
                # (barge-in, or a follow-on utterance after a pause), the newer turn
                # now owns the histories — don't touch them here, and NEVER delete
                # the user's message. A produced-nothing turn just leaves its user
                # message, which the next turn merges into (see _history_begin).
                if full_response.strip() and not self._aborted(turn):
                    self._history_set_assistant(full_response)
                futures_q.put(SENTINEL)

        first_seg = True
        with ThreadPoolExecutor(max_workers=n_gpus) as executor:
            producer = threading.Thread(
                target=_producer, args=(executor,), name="llm-producer", daemon=True
            )
            producer.start()

            # Consume render futures strictly in order (sentences i+1.. render in
            # parallel while we wait on sentence i).
            while True:
                item = futures_q.get()
                if item is SENTINEL:
                    break
                fut, seg = item
                if self._aborted(turn):
                    fut.cancel()
                    seg.cancel()
                    continue  # keep draining to SENTINEL so the producer finishes
                # Announce as soon as the AUDIO exists, not when the render is
                # done — that is what lets sentence i stream while i+1 renders on
                # another GPU. Waiting on `started` OR the future completing
                # covers the TTS-failed case, where `started` never fires.
                while not seg.started.wait(0.02):
                    if fut.done() or self._aborted(turn) or config.SHUTTING_DOWN.is_set():
                        break
                if not seg.started.is_set() or self._aborted(turn):
                    seg.cancel()
                    # A single sentence's TTS/MuseTalk failing must NOT abort the
                    # whole turn (which would skip the video_end sentinel below and
                    # freeze the avatar on its last frame). Skip it and continue.
                    continue
                self._enqueue(seg)
                if first_seg:
                    first_seg = False
                    logger.info("[latency] FIRST segment enqueued at +%.2fs",
                                time.perf_counter() - self._t0)

            producer.join(timeout=1.0)

        if not self._aborted(turn):
            self.video_queue.put(None)
            self.state = "speaking"

    def get_next_video(self):
        """Non-blocking: get next video from queue.

        Returns:
            a Segment, or None if the queue is empty, or False if the response
            is complete.
        """
        try:
            item = self.video_queue.get_nowait()
            if item is None:
                # End of response
                self.state = "idle"
                return False
            return item
        except queue.Empty:
            return None

    def mark_playback_done(self):
        """Called when frontend finishes playing all videos."""
        if self.video_queue.empty() and self.state == "speaking":
            self.state = "idle"

    def get_chat_history(self):
        """Return current chat history for display."""
        return list(self.chat_history)

    def reset(self):
        """Reset everything."""
        self.cancel_event.set()
        self._turn += 1  # invalidate any in-flight response
        time.sleep(0.1)
        self.cancel_event.clear()
        self.chat_history.clear()
        self.patient.clear()
        self.ended = False
        self.clinical = runtime.ClinicalSession()
        runtime.start(self.clinical)
        self._flush()
        self.state = "idle"
