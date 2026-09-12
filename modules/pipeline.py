
import os
import threading
import logging
import queue
import tempfile
import time
from collections import deque

import numpy as np

import config
from modules import asr, clipcache, dialog, llm, tts
from modules.privacy import phi, phi_keys
from modules.sbirt import runtime, select as sel, state_view, templates
from modules.sbirt.instruments import BY_KEY

logger = logging.getLogger(__name__)


class Segment:

    """One synthesized utterance and the frames that lip-sync it.

    `delivered` is set by the pump when it broadcasts segment_end, which is the
    only honest evidence that the audio left the server for a live socket. The
    voiced ledger is committed on that rather than on the render finishing,
    because a rendered segment still sitting in the queue is cancelled by a
    flush and was never heard.
    """

    __slots__ = ("sentence", "audio_url", "fps", "frames",
                 "cancelled", "delivered", "_t_enqueue")

    def __init__(self, sentence: str = ""):
        self.sentence = sentence
        self.audio_url = None
        self.fps = config.MUSETALK_FPS
        self.frames = queue.Queue()
        self.cancelled = threading.Event()
        self.delivered = threading.Event()
        self._t_enqueue = 0.0

    def open(self, audio_url: str):
        self.audio_url = audio_url

    def close(self):
        self.frames.put(None)

    def cancel(self):
        self.cancelled.set()
        self.delivered.set()
        self.frames.put(None)



def protocol_clip_key(key: str) -> str:
    return f"protocol.{key}"


def clip_stamp(text: str) -> str:
    if not config.ENABLE_VIDEO_AVATAR:
        return f"audio-only\n{text}"
    return (f"avatar:{config.avatar_fingerprint()}\n"
            f"fps:{config.MUSETALK_FPS}\n{text}")


def _scratch_audio(audio: bytes) -> str:
    os.makedirs(config.RENDER_SCRATCH_DIR, exist_ok=True)
    fd, path = tempfile.mkstemp(suffix=".mp3", dir=config.RENDER_SCRATCH_DIR)
    with os.fdopen(fd, "wb") as f:
        f.write(audio)
    return path


def render_into(seg: Segment, audio: bytes, abort=None, collect=None) -> int:
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
    if not config.CLIP_CACHE:
        return None
    clip = clipcache.get_clip(key, clip_stamp(text))
    if clip is None:
        return None
    seg = Segment(text)
    for jpeg in clip.frames:
        seg.frames.put(jpeg)
    seg.open(clipcache.publish_url(clip.audio))
    seg.close()
    return seg



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
    global _last_active
    with _activity_lock:
        _last_active = time.monotonic()


def conversation_busy() -> bool:
    with _activity_lock:
        if _active_turns > 0:
            return True
        idle_for = time.monotonic() - _last_active
    return idle_for < config.CLIP_PREWARM_IDLE_SEC


def _prewarm_abort() -> bool:
    return conversation_busy() or config.SHUTTING_DOWN.is_set()


def fixed_catalogue() -> list:
    items = [(protocol_clip_key(config.GREETING_CLIP_KEY),
              config.GREETING_PREAMBLE)]
    items += [(protocol_clip_key(k), t)
              for k, t in templates.all_fixed_utterances().items()]
    return items


def _prewarm_one(key: str, text: str, stamp: str) -> str:
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
    while not config.SHUTTING_DOWN.is_set():
        if not conversation_busy():
            return True
        config.SHUTTING_DOWN.wait(config.CLIP_PREWARM_POLL_SEC)
    return False


def prewarm_fixed_clips():
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
            pending.appendleft((key, text))
        else:
            attempts[key] = attempts.get(key, 0) + 1
            if attempts[key] < config.CLIP_PREWARM_MAX_ATTEMPTS:
                pending.append((key, text))
            else:
                skipped += 1
                logger.warning("[prewarm] giving up on %s after %d attempts "
                               "(it will be rendered on demand)",
                               key, attempts[key])
    if not config.SHUTTING_DOWN.is_set():
        logger.info("[prewarm] complete: %d/%d clips cached, %d skipped; %s",
                    cached, total, skipped, clipcache.stats())




class Pipeline:
    """Media, history and the browser-facing state. No clinical decisions.

    Everything about which question comes next now lives in DialogRunner, which
    owns the session on its own thread. That is what retired the protocol lock:
    it used to be held across the model call, the TTS round-trip and a
    multi-second render, so a second utterance queued behind a render instead of
    superseding it.
    """

    def __init__(self, audit_key: str = "default"):
        self.audit_key = audit_key
        self.state = "idle"
        self.cancel_event = threading.Event()
        self.video_queue = queue.Queue()
        self.chat_history = []
        self.patient = {}
        self.ended = False
        self.dynamic_renders = 0
        self.fixed_renders = 0
        self.clinical = runtime.ClinicalSession()
        self._lock = threading.Lock()
        self._epoch_lock = threading.Lock()
        self._epoch = 0
        self._t0 = 0.0
        self._carry_lock = threading.Lock()
        self._pending_voice = None
        self._carry_audio = None
        self._live_lock = threading.Lock()
        self._live: list[Segment] = []
        self.runner = dialog.DialogRunner(self, self.clinical)

    # --- segments -----------------------------------------------------------

    def _track(self, seg: Segment) -> Segment:
        """Register a segment so a barge-in can cancel it.

        Cached clips used to skip this and were only reachable through the queue
        drain, so one already being pumped survived the flush that was supposed
        to stop it.
        """
        with self._live_lock:
            self._live.append(seg)
        return seg

    def _new_segment(self, sentence: str) -> Segment:
        return self._track(Segment(sentence))

    def _enqueue(self, seg: Segment):
        seg._t_enqueue = time.perf_counter()
        self.video_queue.put(seg)
        self.state = "speaking"

    def _flush(self):
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

    def get_next_video(self):
        try:
            item = self.video_queue.get_nowait()
            if item is None:
                self.state = "idle"
                return False
            return item
        except queue.Empty:
            return None

    def end_response(self):
        """Close the current response so the browser stops waiting for frames.

        Emitted when the runner stops delivering rather than after each unit, so
        a question that follows a piece of education does not make the avatar
        blink back to idle in between.
        """
        self.video_queue.put(None)

    def session_ended(self):
        self.ended = True

    # --- epochs -------------------------------------------------------------

    def _bump_epoch(self) -> int:
        """Invalidate work in flight.

        Guarded because three threads reach it — the event loop on a barge-in,
        a worker on reset, and the HTTP handler on stop — and a bare += lost
        increments.
        """
        with self._epoch_lock:
            self._epoch += 1
            return self._epoch

    def _superseded(self, epoch) -> bool:
        return epoch != self._epoch

    def ready_to_speak(self) -> bool:
        """Delivery cannot succeed while a cancel stands, and trying spins
        the tick loop."""
        return not self.cancel_event.is_set()

    def floor_returned(self) -> None:
        """The runner owns the turn again, so no cancel may still stand.

        Clearing it here rather than at speech_end is what keeps the gap
        between the two closed: until the runner has the floor back, nothing it
        might deliver is wanted. It also bounds the cancel's life — a
        speech_start the near-field gate rejects produces no speech_end, and
        the cancel it set used to outlive the session.
        """
        self.cancel_event.clear()
        if self.state == "listening":
            self.state = "idle"

    # --- speech in ----------------------------------------------------------

    def on_speech_start(self):
        note_activity()
        self.runner.floor_taken()
        if self.state in ("processing", "speaking"):
            logger.info("Barge-in detected! Cancelling current response.")
            with self._carry_lock:
                pv = self._pending_voice
                if pv is not None and pv[0] == self._epoch:
                    self._carry_audio, self._pending_voice = pv[1], None
                    logger.info("[continuation] pause-split: carrying the "
                                "unconsumed first half into the next utterance")
            self.cancel_event.set()
            self._bump_epoch()
            self._flush()
            self.runner.interrupted()
        self.state = "listening"

    def cancel_response(self):
        if self.state in ("processing", "speaking"):
            self.cancel_event.set()
            self._bump_epoch()
            self._flush()
            self.runner.interrupted()
            self.cancel_event.clear()
        self._clear_carry()
        self.state = "idle"

    def on_speech_end(self, audio_array):
        with self._carry_lock:
            if self._carry_audio is not None:
                audio_array = np.concatenate([self._carry_audio, audio_array])
                self._carry_audio = None
                logger.info("[continuation] resumed after a pause: merged "
                            "utterance is now %.2fs", len(audio_array) / 16000)
        self._start_turn(audio_array)

    def _start_turn(self, audio_array):
        self._t0 = time.perf_counter()
        self.state = "processing"
        self.runner.floor_pending()
        epoch = self._bump_epoch()
        with self._carry_lock:
            self._pending_voice = (epoch, audio_array)
        _turn_begin()
        threading.Thread(target=self._transcribe, args=(audio_array, epoch),
                         name="asr", daemon=True).start()

    def on_speech_end_text(self, text):
        self._t0 = time.perf_counter()
        self._clear_carry()
        self.state = "processing"
        self._bump_epoch()
        self.runner.user_said(text)

    def _transcribe(self, audio_array, epoch):
        try:
            text = asr.transcribe_array(audio_array, sample_rate=16000)
            logger.info("[latency] ASR done at +%.2fs -> %s",
                        time.perf_counter() - self._t0, phi(text))
            if self._superseded(epoch):
                return
            self._consume_utterance(epoch)
            if not text.strip():
                # Nothing was actually said. The runner keeps whatever it still
                # owes, so a phantom barge-in costs a pause rather than a
                # question.
                self.state = "idle"
                self.runner.floor_released()
                return
            self.runner.user_said(text)
        except Exception:
            logger.exception("ASR failed")
            self.state = "idle"
            self.runner.floor_released()
        finally:
            _turn_end()

    def _consume_utterance(self, epoch):
        with self._carry_lock:
            if self._pending_voice is not None and self._pending_voice[0] == epoch:
                self._pending_voice = None

    def _clear_carry(self):
        with self._carry_lock:
            self._pending_voice = self._carry_audio = None

    def start_greeting(self):
        self._t0 = time.perf_counter()
        self._clear_carry()
        self.cancel_event.clear()
        self._bump_epoch()
        self.clinical = runtime.ClinicalSession()
        self.runner.restart(self.clinical)

    # --- history and NLU context -------------------------------------------

    def api_window(self):
        with self._lock:
            win = [dict(m) for m in
                   self.chat_history[-config.LLM_HISTORY_MAX_MESSAGES:]]
        if win and win[0]["role"] == "assistant":
            del win[0]
        return win

    def patient_facts(self):
        with self._lock:
            return dict(self.patient)

    def history_user(self, user_text):
        """Record the user's turn.

        A still-unanswered user message absorbs the new text: speech split by a
        pause reassembles into one sentence rather than reaching the model as two
        consecutive user messages.
        """
        with self._lock:
            hist = self.chat_history
            if hist and hist[-1]["role"] == "user":
                hist[-1] = {"role": "user",
                            "content": (hist[-1]["content"].rstrip()
                                        + " " + user_text).strip()}
            else:
                hist.append({"role": "user", "content": user_text})
        window = self.api_window()
        threading.Thread(target=self._extract_patient, args=(window,),
                         name="patient-extract", daemon=True).start()

    def history_assistant_append(self, text):
        """Grow the assistant's current turn in place as it is spoken."""
        with self._lock:
            hist = self.chat_history
            if hist and hist[-1]["role"] == "assistant":
                hist[-1]["content"] = (hist[-1]["content"] + " " + text).strip()
            else:
                hist.append({"role": "assistant", "content": text})

    def _extract_patient(self, history_snapshot):
        facts = llm.extract_patient_facts(history_snapshot)
        if not facts:
            return
        with self._lock:
            for k, v in facts.items():
                if v in (None, "", [], {}):
                    continue
                self.patient[k] = v
        logger.info("[patient] profile now: %s", phi_keys(self.patient))

    def turn_facts(self, session):
        """The only factual source the model may draw on when it replies.

        Every entry is deterministic state. What was DISCUSSED is read from the
        voiced ledger rather than from a flag set while composing, so a piece of
        education lost to a barge-in is no longer reported as delivered.
        """
        fld = sel.answerable(session)
        facts = {
            "current_phase": session.node,
            "standard_drink_definition_discussed":
                "alcohol.edu.standard_drink" in session.spoken,
            "drinking_limits_discussed":
                "alcohol.edu.limits" in session.spoken,
            "permissions_declined_so_far": list(session.declined),
            "active_topic": session.arm,
        }
        for unit_key, fact_key in (
                ("alcohol.edu.standard_drink", "standard_drink_definition"),
                ("alcohol.edu.limits", "recommended_drinking_limits")):
            if unit_key in session.spoken:
                facts[fact_key] = templates.FIXED[unit_key]
        if (fld is not None and fld.kind == "option" and fld.instrument
                and fld.instrument != "prescreen"):
            items = BY_KEY[fld.instrument].items
            facts["answers_already_given"] = [
                {"item": i, "question": items[i].text,
                 "answer": items[i].options[code].label}
                for i, code in sorted(
                    session.responses.get(fld.instrument, {}).items())]
        return facts

    def interview_state(self, session):
        return state_view.render_interview_state(session)

    # --- speech out ---------------------------------------------------------

    def speak(self, delivery) -> bool:
        """Voice one delivery unit, and report whether it reached the transcript.

        The return value is the ledger's only input, and the bar is the
        transcript rather than the audio: a line the person can read has been
        said, so a barge-in part-way through it is not worth hearing again.
        Only a cancel that lands before a beat is composed leaves the unit owed.
        """
        utterances = tuple(delivery.beats)
        if delivery.ack:
            if utterances and isinstance(utterances[0], runtime.LLMSay):
                # One utterance, one author. Left as two beats, the second model
                # call restates the acknowledgment it can see in the history and
                # the person hears the same sentence twice.
                utterances = (runtime.LLMSay(
                    f"First acknowledge what the person just said, using these "
                    f"words or very close to them: {delivery.ack!r}. Then, in "
                    f"the same breath and without repeating yourself, "
                    f"{utterances[0].instruction}"),) + utterances[1:]
            else:
                utterances = (runtime.Speak(delivery.ack),) + utterances

        for utt in utterances:
            if self.cancel_event.is_set():
                # Nothing of this beat was written, so the unit is still owed.
                return False
            self._voice_one(utt)
        return True

    def _voice_one(self, utt):
        clip_key = None
        if isinstance(utt, runtime.Say):
            text = utt.text
            clip_key = protocol_clip_key(utt.key)
            cached = fixed_segment(text, clip_key)
            if cached is not None:
                self.history_assistant_append(text)
                self._track(cached)
                self._enqueue(cached)
                return cached
        elif isinstance(utt, runtime.Speak):
            text = utt.text
        else:
            text = llm.phrase_utterance(utt.instruction, self.api_window(),
                                        self.patient_facts())
        if not text or not text.strip():
            return None
        self.history_assistant_append(text)
        return self._speak_dynamic(text, cache_key=clip_key)

    def _speak_dynamic(self, text, cache_key=None):
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
        if audio is None:
            seg.cancel()
            return None
        self._enqueue(seg)
        keep = bool(cache_key) and config.CLIP_CACHE
        frames = [] if keep else None
        render_into(seg, audio, abort=seg.cancelled.is_set, collect=frames)
        if (keep and not seg.cancelled.is_set()
                and (frames or not config.ENABLE_VIDEO_AVATAR)):
            clipcache.put_clip(cache_key, clip_stamp(text), audio, frames)
        return seg

    # --- lifecycle ----------------------------------------------------------

    def get_chat_history(self):
        return list(self.chat_history)

    def close(self):
        """A dropped session must fall silent and must not keep its tick
        thread alive."""
        self.cancel_event.set()
        self._bump_epoch()
        self._flush()
        self.runner.shutdown()
        self.state = "idle"
