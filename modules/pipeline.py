"""Turns what the user said into what the avatar says back.

One turn runs: transcribe, classify and code the utterance, let the clinical
state machine in modules/sbirt/ decide what comes next, then speak it. Each
sentence becomes a Segment — audio plus a live stream of frames — and segments
are delivered strictly in order even though several may be rendering at once.

Two properties shape most of the design here:

  * Nothing survives a barge-in. When the user starts talking, the LLM stream,
    the pending synthesis, every in-flight render and every queued segment are
    all abandoned together, and a turn id makes the abandoned work unable to
    write into the turn that replaced it.
  * The state machine decides, not the model. The LLM classifies an utterance
    and words individual lines; it never chooses the next question, a score, or
    where the session goes.
"""

import os
import threading
import logging
import queue
import tempfile
import time
from collections import deque

import numpy as np

import config
from modules import asr, clipcache, llm, tts, privacy
# modules.avatar is imported inside the render functions, never here. Importing
# it pulls in torch and the entire sibling MuseTalk checkout, which voice-only
# mode must neither pay for nor be able to fail on.
from modules.privacy import phi, phi_keys
from modules.sbirt import runtime, state_view, templates
from modules.sbirt.instruments import BY_KEY, InvalidResponse, PRE_SCREEN

logger = logging.getLogger(__name__)


class Segment:
    """One utterance travelling to the browser: complete audio, streaming frames.

    The asymmetry is forced by the renderer. It needs the whole utterance's
    audio before it can start, because features are padded against the total
    length, but it produces frames a batch at a time. So audio is handed over
    once as a URL while frames arrive progressively.

    The browser treats the audio element's currentTime as the clock and draws
    frame floor(t * fps) on a canvas. Nothing is re-encoded and no container is
    restarted between sentences, which is what removes the seams a per-sentence
    video file would have; the audio never pauses for the video to catch up.

    The render thread fills `frames` and the state poller drains it. The
    terminating None is mandatory: without it the poller cannot distinguish a
    finished utterance from a slow one.
    """

    __slots__ = ("sentence", "audio_url", "fps", "frames", "started",
                 "cancelled", "_t_enqueue")

    def __init__(self, sentence: str = ""):
        self.sentence = sentence
        self.audio_url = None           # valid once `started` is set
        self.fps = config.MUSETALK_FPS
        self.frames = queue.Queue()     # JPEG bytes, terminated by None
        self.started = threading.Event()  # audio exists; safe to announce
        self.cancelled = threading.Event()
        self._t_enqueue = 0.0           # for the latency log only

    def open(self, audio_url: str):
        self.audio_url = audio_url
        self.started.set()

    def close(self):
        self.frames.put(None)

    def cancel(self):
        """Abandon this utterance: stop its render and discard what is buffered.

        The sentinel is queued as well, so a poller already blocked on this
        segment wakes rather than waiting for a render that will never finish.
        """
        self.cancelled.set()
        self.frames.put(None)


# Fixed utterances are addressed by a stable key rather than a path, since the
# clip they name lives in memory. The prefixes keep the three namespaces from
# colliding.

def protocol_clip_key(key: str) -> str:
    """Cache key for a fixed protocol utterance."""
    return f"protocol.{key}"


def clip_stamp(text: str) -> str:
    """Everything a rendered clip depends on, as one cache-validity string.

    Keying on the text alone is a trap: repointing the driving video changes no
    text, so every cached clip would go on replaying the previous face
    indefinitely.

    The frame rate belongs here too, because cached frames are a fixed-rate
    sequence that the browser indexes by wall-clock time — replaying them under
    a different clock desynchronises the whole clip.

    Voice-only clips deliberately omit the avatar entirely: audio that can never
    show a face must not be invalidated by swapping the video.
    """
    if not config.ENABLE_VIDEO_AVATAR:
        return f"audio-only\n{text}"
    return (f"avatar:{config.avatar_fingerprint()}\n"
            f"fps:{config.MUSETALK_FPS}\n{text}")


def _scratch_audio(audio: bytes) -> str:
    """Write an utterance's audio to a temporary file and return its path.

    The only disk write left in the speaking path, and not a cache: the whisper
    feature extractor takes a filename, so the bytes must exist as a file for
    the duration of one render. The destination is normally RAM-backed, and
    every caller deletes the file in a finally.
    """
    os.makedirs(config.RENDER_SCRATCH_DIR, exist_ok=True)
    fd, path = tempfile.mkstemp(suffix=".mp3", dir=config.RENDER_SCRATCH_DIR)
    with os.fdopen(fd, "wb") as f:
        f.write(audio)
    return path


def render_into(seg: Segment, audio: bytes, abort=None, collect=None) -> int:
    """Fill `seg` from finished audio, streaming frames as they render.

    Args:
        seg: the segment to open and fill.
        audio: the complete utterance.
        abort: polled by the renderer; defaults to the segment's own cancelled
            flag.
        collect: if given, receives every JPEG as it is produced, so the caller
            can cache a fixed utterance it had to render itself.

    Returns:
        The number of frames rendered; zero in voice-only mode, where the audio
        alone is the whole segment.

    The audio is published and the segment opened before rendering starts, so
    the browser can fetch and begin playing while the first batch is still on
    the GPU.
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
    """Render an utterance to frames with no segment and no listener.

    Used by the pre-warm, which has nowhere to stream to. Returns whatever was
    rendered, which is a partial list if `abort` fired — the caller must check
    that before caching, or half an utterance gets cached permanently.
    """
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
    """A ready-to-play segment for a fixed line, or None if it is not cached.

    A miss is ordinary, not an error: the cache is memory and is empty after
    every restart, so callers must fall back to rendering the line themselves.
    With config.CLIP_CACHE off every call misses, which is how the whole cache
    is disabled without removing it.

    The frames are loaded up front rather than streamed. There is no render to
    wait for, so withholding them would only delay playback.
    """
    if not config.CLIP_CACHE:
        return None
    clip = clipcache.get_clip(key, clip_stamp(text))
    if clip is None:
        return None
    seg = Segment(text)
    for jpeg in clip.frames:
        seg.frames.put(jpeg)
    # A fresh blob per playback: the clip keeps the bytes, and the URL the
    # browser is handed expires on its own without the cache having to
    # track who is still playing what.
    seg.open(clipcache.publish_url(clip.audio))
    seg.close()
    return seg


# The pre-warm renders on the same GPUs that answer people, so it needs to know
# when a conversation is live. These few functions are the whole arbitration:
# the pre-warm waits on conversation_busy() before starting a clip and also
# passes it as the render's abort condition, so somebody who starts talking gets
# the GPU back within one batch rather than after the whole clip.

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
    """Mark the conversation live without claiming a turn.

    Called the instant the VAD hears speech, so the pre-warm releases its GPU
    before the turn that will need it has even been created.
    """
    global _last_active
    with _activity_lock:
        _last_active = time.monotonic()


def conversation_busy() -> bool:
    """True while a turn is in flight, or one ended within the grace period.

    The grace period stops the pre-warm from seizing a GPU in the pause between
    two turns of the same exchange.
    """
    with _activity_lock:
        if _active_turns > 0:
            return True
        idle_for = time.monotonic() - _last_active
    return idle_for < config.CLIP_PREWARM_IDLE_SEC


def _prewarm_abort() -> bool:
    """The pre-warm's abort condition: anyone talking, or the server stopping."""
    return conversation_busy() or config.SHUTTING_DOWN.is_set()


def fixed_catalogue() -> list:
    """(key, text) for every fixed utterance the protocol can ever speak."""
    # The preamble only: the consent question that used to end the greeting is
    # now protocol text, and arrives with the rest of templates below.
    items = [(protocol_clip_key(config.GREETING_CLIP_KEY),
              config.GREETING_PREAMBLE)]
    items += [(protocol_clip_key(k), t)
              for k, t in templates.all_fixed_utterances().items()]
    return items


def _prewarm_one(key: str, text: str, stamp: str) -> str:
    """Render one fixed utterance into the cache.

    Returns "cached", "preempted" when somebody started talking and the GPU was
    handed back, or "failed". A preempted render leaves truncated frames, which
    are discarded rather than cached.
    """
    audio = tts.synthesize(text, config.SHUTTING_DOWN)
    if audio is None:
        return "failed"
    frames = []
    if config.ENABLE_VIDEO_AVATAR:
        frames = _render_frames(audio, abort=_prewarm_abort)
        if _prewarm_abort():
            return "preempted"
        if not frames:
            return "failed"
    clipcache.put_clip(key, stamp, audio, frames)
    return "cached"


def _wait_until_idle() -> bool:
    """Block until no conversation is in flight. False if the server is stopping."""
    while not config.SHUTTING_DOWN.is_set():
        if not conversation_busy():
            return True
        config.SHUTTING_DOWN.wait(config.CLIP_PREWARM_POLL_SEC)
    return False


def prewarm_fixed_clips():
    """Fill the clip cache with every fixed utterance, whenever nobody is talking.

    Runs in a background thread for the life of the process. Because the cache
    is memory, this happens on every boot, which makes it the lowest-priority
    user of the GPUs: one clip at a time, only while idle, and abandoned
    mid-render as soon as somebody speaks. An abandoned clip is re-queued.

    Purely an optimisation. Any clip needed before this reaches it is rendered
    on demand by the turn that needs it, and cached the same way.
    """
    if not config.CLIP_CACHE:
        logger.info("[prewarm] disabled (CLIP_CACHE=0)")
        return
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
            pending.appendleft((key, text))   # retry first; it was next anyway
        else:
            attempts[key] = attempts.get(key, 0) + 1
            if attempts[key] < config.CLIP_PREWARM_MAX_ATTEMPTS:
                pending.append((key, text))   # retry last; do not block the rest
            else:
                skipped += 1
                logger.warning("[prewarm] giving up on %s after %d attempts "
                               "(it will be rendered on demand)",
                               key, attempts[key])
    if not config.SHUTTING_DOWN.is_set():
        logger.info("[prewarm] complete: %d/%d clips cached, %d skipped; %s",
                    cached, total, skipped, clipcache.stats())


class Pipeline:
    """One conversation: its history, its clinical state, and its turn machinery.

    Cycles through idle, listening, processing and speaking. One instance per
    session; nothing is shared between instances.
    """

    def __init__(self, audit_key: str = "default"):
        self.audit_key = audit_key  # pseudonymous session id, for audit records
        self.state = "idle"
        self.cancel_event = threading.Event()
        self.video_queue = queue.Queue()
        # The conversation, as {"role", "content"} dicts. The page shows all of
        # it; the model sees a trimmed window derived from it. Screening results
        # live in self.clinical instead, so trimming can never lose a score.
        self.chat_history = []
        # Patient facts extracted in the background and re-injected every turn,
        # so they outlive the history window.
        self.patient = {}
        # Set when the session is over — consent refused, aborted, or the
        # protocol finished. The server reads it, drops the mic and tells the
        # client to stop.
        self.ended = False
        # Renders of generated content this session, which is the observable the
        # study caps. Fixed protocol lines never count toward it, however they
        # were produced.
        self.dynamic_renders = 0
        # Fixed lines this session had to render because the cache was still
        # cold. Counted apart from dynamic_renders precisely so that they do not
        # inflate the number above: these are verbatim protocol, not generation.
        self.fixed_renders = 0
        # The clinical state: where the protocol is, what has been coded, the
        # derived scores and zones. modules/sbirt/runtime.py owns every
        # transition; the model never makes one.
        self.clinical = runtime.ClinicalSession()
        runtime.start(self.clinical)
        self._lock = threading.Lock()
        # Serializes a whole protocol turn, from coding through delivery. A
        # typed message racing a spoken one, or a barge-in racing a slow coder,
        # must not both advance the machine; the loser re-checks _aborted()
        # under this lock and withdraws.
        self._protocol_lock = threading.Lock()
        self._processing_thread = None
        # Incremented by every new utterance. A response may only touch shared
        # state while it still owns the current turn, which is what stops an
        # abandoned response from leaking segments into its replacement after
        # cancel_event has been cleared for the new turn.
        self._turn = 0
        self._t0 = 0.0               # start of the turn, for the latency log
        # Someone pausing mid-sentence trips the turn detector, and their second
        # half then arrives as a separate utterance while the first is still in
        # flight. Holding the first half here lets the two be merged so ASR sees
        # one sentence, instead of the first half vanishing.
        #
        # Its own lock, because these run on the websocket thread and must never
        # wait on _protocol_lock, which is held across LLM calls.
        self._carry_lock = threading.Lock()
        self._pending_voice = None   # (turn id, audio) not yet acted on
        self._carry_audio = None     # first half awaiting its continuation
        # Segments belonging to the current turn, including ones already handed
        # to the poller. Emptying video_queue alone would leave whichever
        # segment is playing right now streaming into a browser that has already
        # moved on.
        self._live_lock = threading.Lock()
        self._live: list[Segment] = []

    def _new_segment(self, sentence: str) -> Segment:
        seg = Segment(sentence)
        with self._live_lock:
            self._live.append(seg)
        return seg

    def _enqueue(self, seg: Segment):
        """Hand a segment to the poller, normally before its render has finished.

        The poller forwards frames as they land, so delivery overlaps
        generation rather than waiting for it.
        """
        seg._t_enqueue = time.perf_counter()
        self.video_queue.put(seg)
        self.state = "speaking"

    def _flush(self):
        """Abandon the whole response at every stage at once.

        The caller sets cancel_event, which stops the LLM stream; this stops
        everything downstream of it.

        Order matters. Cancelling the live segments first makes every in-flight
        render see its abort flag on the next batch, so the GPUs stop producing
        before the queue is drained. Draining first would leave the renderers
        busily filling queues nobody will ever read.

        One synthesis already in flight cannot be stopped; its audio is
        discarded when it arrives.
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
        """True if `turn` was cancelled or has been superseded by a newer one.

        Checked at every point where a response would touch shared state, so
        work from an abandoned turn cannot write into its replacement.
        """
        return self.cancel_event.is_set() or turn != self._turn

    def _consume_utterance(self, turn):
        """Mark this turn's audio as acted upon, so it can no longer be carried.

        Past this point a later interruption must start a fresh turn rather than
        merging these words into it a second time.
        """
        with self._carry_lock:
            if self._pending_voice is not None and self._pending_voice[0] == turn:
                self._pending_voice = None

    def _clear_carry(self):
        """Throw away any held sentence fragment.

        Used on Stop, typed input and a new session: a stale first half must
        never be prepended to an unrelated later utterance.
        """
        with self._carry_lock:
            self._pending_voice = self._carry_audio = None

    def on_speech_start(self):
        """The user has begun speaking: abandon whatever the avatar was saying."""
        # Release the pre-warm's GPU now, not when the resulting turn is created
        # a second or two from here.
        note_activity()
        if self.state in ("processing", "speaking"):
            logger.info("Barge-in detected! Cancelling current response.")
            # If the turn being cancelled was speech whose words have not been
            # acted on yet, this is most likely the same person finishing a
            # sentence after a pause the turn detector misread as an ending.
            # Keep that audio so the two halves can be rejoined.
            with self._carry_lock:
                pv = self._pending_voice
                if pv is not None and pv[0] == self._turn:
                    self._carry_audio, self._pending_voice = pv[1], None
                    logger.info("[continuation] pause-split: carrying the "
                                "unconsumed first half into the next utterance")
            self.cancel_event.set()
            self._turn += 1  # invalidates the in-flight response at once
            self._flush()
            # No history repair is needed: the abandoned producer's finally
            # commits whatever it generated before it stopped.

        self.state = "listening"

    def cancel_response(self):
        """Stop any response in flight and settle at idle.

        Used by Stop and by typed input. Queued clips are dropped; the
        conversation history is left intact.
        """
        if self.state in ("processing", "speaking"):
            self.cancel_event.set()
            self._turn += 1
            self._flush()
        # An explicit stop discards a held fragment as well: it is not going to
        # be continued.
        self._clear_carry()
        self.state = "idle"

    def on_speech_end(self, audio_array):
        """Start a turn from a finished utterance.

        Args:
            audio_array: float32 at 16kHz, the whole utterance.
        """
        self._t0 = time.perf_counter()
        # Rejoin a held first half, so ASR transcribes one continuous sentence
        # rather than two fragments that each read as nonsense.
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

        # The turn is marked live here rather than inside the thread, so the
        # pre-warm stops taking GPU work immediately instead of after a
        # scheduling delay. The thread's finally closes it.
        _turn_begin()
        self._processing_thread = threading.Thread(
            target=self._process_speech, args=(audio_array, turn), daemon=True
        )
        self._processing_thread.start()

    def on_speech_end_text(self, text):
        """Start a turn from typed text, skipping ASR."""
        self._t0 = time.perf_counter()
        # Typing supersedes a held spoken fragment; merging the two would
        # produce an utterance the person never made.
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

    def start_greeting(self):
        """Open the conversation with no user input, so the counselor leads.

        Runs the same delivery path as any other turn.
        """
        self._t0 = time.perf_counter()
        self._clear_carry()   # a new session inherits nothing from the last
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
        """Speak the preamble, then let the protocol ask for consent.

        Two utterances rather than one: the question belongs to the protocol,
        which is what lets it be re-asked later from the same place as every
        other question. The person hears the same words in the same order.

        Recording them as the assistant's first turn is what makes the user's
        reply read as an answer to it.
        """
        try:
            if self._aborted(turn):
                self.state = "idle"
                return
            # Greeting means a new run, so the machine restarts at consent.
            self.clinical = runtime.ClinicalSession()
            step = runtime.start(self.clinical)
            beats = (runtime.Say(config.GREETING_CLIP_KEY,
                                 config.GREETING_PREAMBLE),) + step.utterances
            if not self._speak_beats(beats, turn):
                self.state = "idle"
        except Exception as e:
            logger.error(f"Greeting error: {e}", exc_info=True)
            self.state = "idle"
        finally:
            _turn_end()

    # There is one conversation record, and what the model sees is derived from
    # it on demand. Keeping a second, parallel list for the API would introduce
    # an invariant that a barge-in or a failed call could break.
    def _api_window(self):
        """The trailing slice of the history to send to the model.

        Caller must hold self._lock.
        """
        win = [dict(m) for m in
               self.chat_history[-config.LLM_HISTORY_MAX_MESSAGES:]]
        if win and win[0]["role"] == "assistant":
            del win[0]   # a reply whose prompt was trimmed away misleads
        return win

    def _history_begin(self, user_text):
        """Record the user's turn.

        When the previous user message is still unanswered — speech split by a
        pause, so the first half never got a reply — the new text is merged into
        it. That preserves everything the person said, reassembles the sentence,
        and avoids sending two consecutive user messages to the API.
        """
        with self._lock:
            hist = self.chat_history
            if hist and hist[-1]["role"] == "user":
                hist[-1] = {"role": "user",
                            "content": (hist[-1]["content"].rstrip()
                                        + " " + user_text).strip()}
            else:
                hist.append({"role": "user", "content": user_text})
            window = self._api_window()
        # Deliberately not awaited: extraction costs an API call and only
        # affects the next turn's prompt, so it must never delay this one.
        threading.Thread(target=self._extract_patient, args=(window,),
                         name="patient-extract", daemon=True).start()

    def _extract_patient(self, history_snapshot):
        """Merge newly extracted patient facts into the profile. Runs in the background."""
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
        """Start or overwrite the assistant's current turn.

        Overwriting rather than appending is what lets a sentence-by-sentence
        reply grow in place as it streams, instead of arriving as fragments.
        """
        with self._lock:
            hist = self.chat_history
            if hist and hist[-1]["role"] == "assistant":
                hist[-1]["content"] = text
            else:
                hist.append({"role": "assistant", "content": text})

    def _process_speech(self, audio_array, turn):
        """Run one spoken turn end to end, in a background thread."""
        try:
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

            self._protocol_turn(user_text, turn)

        except Exception as e:
            logger.error(f"Pipeline error: {e}", exc_info=True)
            self.state = "idle"
        finally:
            _turn_end()

    def _process_text(self, user_text, turn):
        """Run one typed turn. Identical to _process_speech without the ASR step."""
        try:
            if self._aborted(turn):
                self.state = "idle"
                return

            self._protocol_turn(user_text, turn)

        except Exception as e:
            logger.error(f"Pipeline error (text): {e}", exc_info=True)
            self.state = "idle"
        finally:
            _turn_end()

    def _current_question(self):
        """(text, options) for the item currently being asked, or (None, None).

        Read from the instrument data rather than from anything spoken, so it is
        the question as authored.
        """
        exp = self.clinical.expect
        if exp.kind != "option":
            return None, None
        if exp.instrument == "prescreen":
            item = PRE_SCREEN[exp.item_index].item
        else:
            item = BY_KEY[exp.instrument].items[exp.item_index]
        return item.text, item.options

    def _last_question_text(self):
        """What the avatar most recently asked, as context for classifying a reply.

        Prefers the protocol's own fixed wording, falls back to whatever the
        assistant last said, and finally to the consent question — which is the
        only thing that can have been asked before any step exists.
        """
        step = self.clinical.last_step
        if step:
            for utt in reversed(step.utterances):
                if isinstance(utt, runtime.Say):
                    return utt.text
        with self._lock:
            for m in reversed(self.chat_history):
                if m["role"] == "assistant" and m["content"].strip():
                    return m["content"]
        return config.CONSENT_QUESTION

    def _turn_facts(self):
        """The only factual source the model may draw on when replying.

        Everything here is deterministic state, never model output. That is what
        lets a question like "have we covered what a standard drink is?" be
        answered honestly rather than guessed at or deflected.
        """
        c = self.clinical
        facts = {
            "current_phase": c.node,
            "standard_drink_definition_discussed":
                "alcohol.edu.standard_drink" in c.covered,
            "drinking_limits_discussed": "alcohol.edu.limits" in c.covered,
            "permissions_declined_so_far": list(c.declined),
            "active_topic": c.arm,
        }
        # The actual wording of any education already delivered, so that "say
        # that again" can repeat the real content instead of deflecting.
        for unit_key, fact_key in (
                ("alcohol.edu.standard_drink", "standard_drink_definition"),
                ("alcohol.edu.limits", "recommended_drinking_limits")):
            if unit_key in c.covered:
                facts[fact_key] = templates.FIXED[unit_key]
        # What has already been answered on the current instrument, so that a
        # correction can identify which item it refers to.
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
        """Classify one utterance, act on it, and speak the result.

        A single model call decides what the utterance is relative to the
        question on the table — an answer, a continuation, a question, an aside,
        discomfort, a correction, a refusal, a crisis, or unclear — and codes it
        if it is an answer. Only a validated answer moves the machine; every
        other outcome holds position and re-poses the ask, except discomfort or
        a second aside, which offer to stop instead.

        Serialized, so that a superseded turn cannot advance the machine after
        its replacement already has.
        """
        with self._protocol_lock:
            if self._aborted(turn):
                return
            clinical = self.clinical

            exp = clinical.expect
            if exp.kind == "end":
                self._consume_utterance(turn)
                # The session is over but the person is still talking; answer
                # warmly rather than ignoring them.
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

            # That call may have taken seconds, so a newer turn may have started
            # in the meantime. Returning here leaves the utterance unconsumed
            # and therefore still available to be merged with a continuation.
            if self._aborted(turn):
                return
            # Past this point the words have been acted on, so anything new is a
            # separate turn rather than the rest of this sentence.
            self._consume_utterance(turn)

            # Before any branch: what they volunteered is true regardless of
            # what this turn is classified as, and a crisis or an aside must
            # not throw it away.
            runtime.record_harvest(clinical, out)

            if out.action == "crisis":
                # Hand off and stop. The fixed close gives the emergency
                # numbers and says their provider will follow up; this system
                # does not stay in the conversation trying to keep somebody
                # safe, and the screening does not resume.
                logger.warning("[crisis] NLU flagged crisis at node %s",
                               clinical.node)
                step = runtime.enter_crisis(clinical)
                self._deliver_step(user_text, step, turn)
                self.ended = True
                return

            if out.action == "abort":
                # The person wants to stop entirely. Close with the fixed
                # goodbye and make no attempt to keep them; what was coded so
                # far is retained.
                step = runtime.enter_abort(clinical)
                self._deliver_step(user_text, step, turn)
                self.ended = True
                return

            if out.action == "correction":
                # Overwrite the earlier answer and let skips and scores
                # re-derive. A correction that does not name an answered item of
                # the current instrument moves nothing and asks which was meant.
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
                    # The study consent decision itself, which is the one thing
                    # that has to be recorded outside the session.
                    privacy.record_consent(
                        self.audit_key, "yes" if out.code == 1 else "no")
                if exp.ask_key == "pause.offer":
                    # Not a protocol gate, so it never reaches runtime.advance:
                    # either the pending question comes back or the session ends.
                    step = runtime.resolve_pause(
                        clinical, keep_going=(out.code == 1))
                    return self._deliver_step(user_text, step, turn,
                                              ack=out.reply)
                if exp.kind == "confirm":
                    # Their verdict on a read-back: yes commits the held code,
                    # no re-asks the same item.
                    step = runtime.resolve_confirm(clinical,
                                                   yes=(out.code == 1))
                    return self._deliver_step(user_text, step, turn,
                                              ack=out.reply)
                reason = runtime.confirm_reason(clinical, out)
                if reason is not None:
                    # Reading an answer back is the exception, not the rule: it
                    # happens only where a mistake would change a score — a unit
                    # conversion was assumed, the value sits on a bucket edge,
                    # it contradicts an earlier answer, or nothing deterministic
                    # vouches for the code.
                    step = runtime.request_confirm(clinical, out, reason)
                    return self._deliver_step(user_text, step, turn,
                                              ack=out.reply)
                try:
                    step = runtime.advance(clinical, out)
                except (runtime.ProtocolError, InvalidResponse):
                    # A bug in the protocol wiring must not strand the person
                    # mid-screening: log it loudly and re-ask.
                    logger.exception("protocol advance failed; re-asking")
                    return self._hold(user_text, "", turn)
                if clinical.node == "declined":
                    # A refusal is answered identically every time, so the
                    # fixed close is spoken on its own — no model-worded
                    # acknowledgment in front of it.
                    return self._deliver_step(user_text, step, turn)
                return self._deliver_step(user_text, step, turn,
                                          ack=out.reply)

            if out.action == "continuation":
                # An answer delivered in two breaths. Fold it into the previous
                # capture and hold position, so the person is not asked the same
                # thing twice.
                runtime.absorb(clinical, out)
                return self._hold(user_text, out.reply, turn)

            if out.action == "dont_know":
                # One assisted-recall attempt, then record the item as missing.
                # Bounded on purpose: repeating the question is worse than
                # having no answer for it.
                if runtime.note_stall(clinical) >= runtime.DONT_KNOW_LIMIT:
                    return self._deliver_missing(user_text, "dont_know", turn)
                return self._hold_probe(user_text, turn)

            if out.action == "unclear":
                # Clarification attempts are finite for the same reason: at the
                # limit the item is recorded unanswered and the protocol moves on.
                if runtime.note_stall(clinical) >= runtime.UNCLEAR_LIMIT:
                    return self._deliver_missing(user_text, "no_answer", turn)
                return self._hold(user_text, out.reply, turn,
                                  repose=not out.reply)

            if out.action == "discomfort":
                # Somebody who feels unwell should not have to say it twice
                # before the interview stops pressing.
                return self._deliver_step(
                    user_text, runtime.offer_pause(clinical), turn,
                    ack=out.reply)

            if out.action == "tangent":
                # Once is an aside; twice running means the question is not
                # what they want to talk about, so the choice goes back to them.
                if runtime.note_aside(clinical) >= runtime.ASIDE_LIMIT:
                    return self._deliver_step(
                        user_text, runtime.offer_pause(clinical), turn,
                        ack=out.reply)
                return self._hold(user_text, out.reply, turn)

            # Questions: answered from state, then the ask is re-posed.
            return self._hold(user_text, out.reply, turn)

    def _deliver_missing(self, user_text, reason, turn):
        """Record the current item as unanswered and carry on.

        This is what guarantees no question can loop forever. A consent gate
        that cannot be answered degrades to its refusal path, which may end the
        session.
        """
        step = runtime.mark_missing(self.clinical, reason)
        return self._deliver_step(user_text, step, turn)

    def _hold_probe(self, user_text, turn):
        """Help the person estimate an answer they say they do not know.

        One attempt only, following the WHO manual (p.18): anchor them to their
        heaviest period in the past year, and always offer to skip so that
        declining a second time is easy. The machine does not move.
        """
        exp = self.clinical.expect
        if exp.kind == "option":
            instruction = (
                "The person says they don't know or can't remember. In one "
                "or two gentle sentences, help them estimate: suggest "
                "thinking about the period in the past year when they were "
                "drinking or using the most, and mention that a rough guess "
                "is fine — or we can skip it and move on. Do NOT re-ask the "
                "question; it is repeated for you straight afterwards.")
        elif exp.kind == "number":
            instruction = (
                "The person says they don't know. In one gentle sentence, "
                "say it doesn't have to be exact and that whatever number "
                "feels closest is fine — or offer to skip it. Do NOT re-ask "
                "the question; it is repeated for you straight afterwards.")
        else:
            # A gate or a read-back has nothing to estimate; just ask again.
            return self._deliver_step(
                user_text, runtime.repeat_step(self.clinical), turn)
        beats = [runtime.LLMSay(instruction)]
        ask = runtime.current_ask(self.clinical)
        if ask is not None:
            beats.append(ask)
        return self._deliver_step(
            user_text,
            runtime.Step(self.clinical.node, tuple(beats), exp),
            turn)

    def _hold(self, user_text, reply, turn, repose=True):
        """Say the model's reply, then put the same question back as authored.

        The model answers the person; the protocol re-poses the ask. Splitting
        it this way is what lets the reply be genuinely responsive — explaining
        a word, answering a question, acknowledging an aside — without ever
        putting a reworded version of a validated item in front of somebody.

        Args:
            repose: False when the reply is itself a question about the item —
                a clarification. Re-reading the whole item after "did you mean
                five or seven?" is the same question twice in one breath.

        A turn that produced no reply still re-poses the ask, so a failed model
        call costs a repeat rather than silence.
        """
        exp = self.clinical.expect
        beats = []
        if reply:
            beats.append(runtime.Speak(reply))
        ask = runtime.current_ask(self.clinical) if repose else None
        if ask is not None:
            beats.append(ask)
        elif not beats:
            beats.append(runtime.LLMSay(
                "In one short sentence, gently ask again for an answer to "
                f"{self._last_question_text()!r}."))
        self._deliver_step(
            user_text,
            runtime.Step(self.clinical.node, tuple(beats), exp),
            turn)

    def _speak_dynamic(self, text, turn, cache_key=None):
        """Synthesize and render one utterance that was not already cached.

        Args:
            text: the sentence to speak.
            turn: the turn that owns this work.
            cache_key: set when the text is a fixed line the caller found
                missing from the cache. Its frames are then collected and
                stored, so a fixed line costs at most one render per process.
                It still marks the line as protocol rather than generation when
                config.CLIP_CACHE is off; only the storing stops.

        Returns:
            The Segment, or None if synthesis failed or the turn was abandoned.

        Enqueued before rendering rather than after, which is where the latency
        win comes from: the poller starts forwarding frames after the first
        batch while this thread is still producing the rest. It then blocks
        until the render ends, which is what keeps utterances in order.

        Only a completed render is cached. An interrupted one leaves truncated
        frames, and half an utterance replaying forever is far worse than
        rendering it again.
        """
        if cache_key is None:
            self.dynamic_renders += 1
            logger.info("[latency] dynamic render #%d this session",
                        self.dynamic_renders)
        else:
            self.fixed_renders += 1
            logger.info("[latency] fixed line %s rendered live (#%d this "
                        "session; cache %s)", cache_key, self.fixed_renders,
                        "cold" if config.CLIP_CACHE else "off")
        seg = self._new_segment(text)
        audio = tts.synthesize(text, self.cancel_event)
        if audio is None or self._aborted(turn):
            seg.cancel()
            return None
        self._enqueue(seg)
        keep = bool(cache_key) and config.CLIP_CACHE
        frames = [] if keep else None
        render_into(seg, audio, abort=seg.cancelled.is_set, collect=frames)
        if (keep and not seg.cancelled.is_set() and not self._aborted(turn)
                and (frames or not config.ENABLE_VIDEO_AVATAR)):
            clipcache.put_clip(cache_key, clip_stamp(text), audio, frames)
        return seg

    def _speak_beats(self, utterances, turn):
        """Say a run of utterances in order, and return whether it finished.

        Three kinds arrive here: fixed protocol lines, which come from the
        shared cache when it holds them; text the turn already produced; and
        instructions the model must word first. Shared by the greeting and by
        every step, so the opening is delivered by the same code as the rest of
        the conversation rather than a copy of it.
        """
        spoken = []
        for utt in utterances:
            if self._aborted(turn):
                return False
            # A fixed line arrives complete from the cache; everything else has
            # to be synthesized below, which `pending` marks.
            seg, pending, clip_key = None, False, None
            if isinstance(utt, runtime.Say):
                text = utt.text
                clip_key = protocol_clip_key(utt.key)
                seg = fixed_segment(text, clip_key)
                pending = seg is None
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
                    continue          # skip it; the protocol still advances
                pending = True
            if self._aborted(turn):
                return False
            spoken.append(text)
            # Written before it is spoken, so a failed render still leaves the
            # words on screen.
            self._history_set_assistant(" ".join(spoken))
            if seg is not None:
                self._enqueue(seg)
            elif pending:
                self._speak_dynamic(text, turn, cache_key=clip_key)
        if self._aborted(turn):
            return False
        self.video_queue.put(None)
        self.state = "speaking"
        return True

    def _deliver_step(self, user_text, step, turn, ack=""):
        """Speak everything one machine step calls for, in order.

        Args:
            user_text: what the person said, recorded before anything is spoken.
            step: the step to deliver.
            turn: the turn that owns this work.
            ack: a brief acknowledgment, spoken first so the person hears they
                were heard before the next question arrives.
        """
        self._history_begin(user_text)
        self.state = "processing"
        utterances = tuple(step.utterances)
        if ack:
            if utterances and isinstance(utterances[0], runtime.LLMSay):
                # One utterance, one author. Left as two beats, the second
                # model call restates the acknowledgment it can see in the
                # history and the person hears the same sentence twice.
                utterances = (runtime.LLMSay(
                    f"First acknowledge what the person just said, using these "
                    f"words or very close to them: {ack!r}. Then, in the same "
                    f"breath and without repeating yourself, {utterances[0].instruction}"),
                ) + utterances[1:]
            else:
                utterances = (runtime.Speak(ack),) + utterances
        # Deliberately not short-circuited on an unfinished delivery: the end
        # flag below is session state, not delivery state.
        self._speak_beats(utterances, turn)
        if step.expect.kind == "end":
            # Session state, not delivery state, so it is set even when a
            # barge-in cut the closing line short: the protocol reached its
            # end, and whether the goodbye finished playing does not change
            # that. Skipping it on an aborted turn is what let somebody talk
            # straight past a goodbye and keep the session alive forever.
            self.ended = True

    def get_next_video(self):
        """Take the next segment for delivery, without blocking.

        Returns:
            A Segment; None if nothing is ready yet; False once the whole
            response has been delivered. The three are distinct because "not
            yet" and "finished" mean opposite things to the poller.
        """
        try:
            item = self.video_queue.get_nowait()
            if item is None:
                self.state = "idle"
                return False
            return item
        except queue.Empty:
            return None

    def get_chat_history(self):
        """A snapshot of the conversation, safe to serialize while it mutates."""
        return list(self.chat_history)

    def reset(self):
        """Clear the conversation and the clinical state, back to a new session."""
        self.cancel_event.set()
        self._turn += 1
        # Give in-flight threads a moment to notice the cancellation before the
        # state they are reading is torn out from under them.
        time.sleep(0.1)
        self.cancel_event.clear()
        self.chat_history.clear()
        self.patient.clear()
        self.ended = False
        self.clinical = runtime.ClinicalSession()
        runtime.start(self.clinical)
        self._flush()
        self.state = "idle"
