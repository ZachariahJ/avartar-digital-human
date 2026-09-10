
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
from modules.privacy import phi, phi_keys
from modules.sbirt import runtime, state_view, templates
from modules.sbirt.instruments import BY_KEY, InvalidResponse, PRE_SCREEN

logger = logging.getLogger(__name__)


class Segment:

    __slots__ = ("sentence", "audio_url", "fps", "frames", "started",
                 "cancelled", "_t_enqueue")

    def __init__(self, sentence: str = ""):
        self.sentence = sentence
        self.audio_url = None
        self.fps = config.MUSETALK_FPS
        self.frames = queue.Queue()
        self.started = threading.Event()
        self.cancelled = threading.Event()
        self._t_enqueue = 0.0

    def open(self, audio_url: str):
        self.audio_url = audio_url
        self.started.set()

    def close(self):
        self.frames.put(None)

    def cancel(self):
        self.cancelled.set()
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
        runtime.start(self.clinical)
        self._lock = threading.Lock()
        self._protocol_lock = threading.Lock()
        self._processing_thread = None
        self._turn = 0
        self._t0 = 0.0
        self._carry_lock = threading.Lock()
        self._pending_voice = None
        self._carry_audio = None
        self._live_lock = threading.Lock()
        self._live: list[Segment] = []

    def _new_segment(self, sentence: str) -> Segment:
        seg = Segment(sentence)
        with self._live_lock:
            self._live.append(seg)
        return seg

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

    def _aborted(self, turn):
        return self.cancel_event.is_set() or turn != self._turn

    def _consume_utterance(self, turn):
        with self._carry_lock:
            if self._pending_voice is not None and self._pending_voice[0] == turn:
                self._pending_voice = None

    def _clear_carry(self):
        with self._carry_lock:
            self._pending_voice = self._carry_audio = None

    def on_speech_start(self):
        note_activity()
        if self.state in ("processing", "speaking"):
            logger.info("Barge-in detected! Cancelling current response.")
            with self._carry_lock:
                pv = self._pending_voice
                if pv is not None and pv[0] == self._turn:
                    self._carry_audio, self._pending_voice = pv[1], None
                    logger.info("[continuation] pause-split: carrying the "
                                "unconsumed first half into the next utterance")
            self.cancel_event.set()
            self._turn += 1
            self._flush()

        self.state = "listening"

    def cancel_response(self):
        if self.state in ("processing", "speaking"):
            self.cancel_event.set()
            self._turn += 1
            self._flush()
        self._clear_carry()
        self.state = "idle"

    def on_speech_end(self, audio_array):
        self._t0 = time.perf_counter()
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

        _turn_begin()
        self._processing_thread = threading.Thread(
            target=self._process_speech, args=(audio_array, turn), daemon=True
        )
        self._processing_thread.start()

    def on_speech_end_text(self, text):
        self._t0 = time.perf_counter()
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
        self._t0 = time.perf_counter()
        self._clear_carry()
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
        try:
            if self._aborted(turn):
                self.state = "idle"
                return
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

    def _api_window(self):
        win = [dict(m) for m in
               self.chat_history[-config.LLM_HISTORY_MAX_MESSAGES:]]
        if win and win[0]["role"] == "assistant":
            del win[0]
        return win

    def _history_begin(self, user_text):
        with self._lock:
            hist = self.chat_history
            if hist and hist[-1]["role"] == "user":
                hist[-1] = {"role": "user",
                            "content": (hist[-1]["content"].rstrip()
                                        + " " + user_text).strip()}
            else:
                hist.append({"role": "user", "content": user_text})
            window = self._api_window()
        threading.Thread(target=self._extract_patient, args=(window,),
                         name="patient-extract", daemon=True).start()

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

    def _history_set_assistant(self, text):
        with self._lock:
            hist = self.chat_history
            if hist and hist[-1]["role"] == "assistant":
                hist[-1]["content"] = text
            else:
                hist.append({"role": "assistant", "content": text})

    def _process_speech(self, audio_array, turn):
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
        exp = self.clinical.expect
        if exp.kind != "option":
            return None, None
        if exp.instrument == "prescreen":
            item = PRE_SCREEN[exp.item_index].item
        else:
            item = BY_KEY[exp.instrument].items[exp.item_index]
        return item.text, item.options

    def _last_question_text(self):
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
        c = self.clinical
        facts = {
            "current_phase": c.node,
            "standard_drink_definition_discussed":
                "alcohol.edu.standard_drink" in c.covered,
            "drinking_limits_discussed": "alcohol.edu.limits" in c.covered,
            "permissions_declined_so_far": list(c.declined),
            "active_topic": c.arm,
        }
        for unit_key, fact_key in (
                ("alcohol.edu.standard_drink", "standard_drink_definition"),
                ("alcohol.edu.limits", "recommended_drinking_limits")):
            if unit_key in c.covered:
                facts[fact_key] = templates.FIXED[unit_key]
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
        with self._protocol_lock:
            if self._aborted(turn):
                return
            clinical = self.clinical

            exp = clinical.expect
            if exp.kind == "end":
                self._consume_utterance(turn)
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

            if self._aborted(turn):
                return
            self._consume_utterance(turn)

            runtime.record_harvest(clinical, out)

            if out.action == "crisis":
                logger.warning("[crisis] NLU flagged crisis at node %s",
                               clinical.node)
                step = runtime.enter_crisis(clinical)
                self._deliver_step(user_text, step, turn)
                self.ended = True
                return

            if out.action == "abort":
                step = runtime.enter_abort(clinical)
                self._deliver_step(user_text, step, turn)
                self.ended = True
                return

            if out.action == "correction":
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
                    privacy.record_consent(
                        self.audit_key, "yes" if out.code == 1 else "no")
                if exp.ask_key == "pause.offer":
                    step = runtime.resolve_pause(
                        clinical, keep_going=(out.code == 1))
                    return self._deliver_step(user_text, step, turn,
                                              ack=out.reply)
                if exp.kind == "confirm":
                    step = runtime.resolve_confirm(clinical,
                                                   yes=(out.code == 1))
                    return self._deliver_step(user_text, step, turn,
                                              ack=out.reply)
                reason = runtime.confirm_reason(clinical, out)
                if reason is not None:
                    step = runtime.request_confirm(clinical, out, reason)
                    return self._deliver_step(user_text, step, turn,
                                              ack=out.reply)
                try:
                    step = runtime.advance(clinical, out)
                except (runtime.ProtocolError, InvalidResponse):
                    logger.exception("protocol advance failed; re-asking")
                    return self._hold(user_text, "", turn)
                if clinical.node == "declined":
                    return self._deliver_step(user_text, step, turn)
                return self._deliver_step(user_text, step, turn,
                                          ack=out.reply)

            if out.action == "continuation":
                runtime.absorb(clinical, out)
                return self._hold(user_text, out.reply, turn)

            if out.action == "dont_know":
                if runtime.note_stall(clinical) >= runtime.DONT_KNOW_LIMIT:
                    return self._deliver_missing(user_text, "dont_know", turn)
                return self._hold_probe(user_text, turn)

            if out.action == "unclear":
                if runtime.note_stall(clinical) >= runtime.UNCLEAR_LIMIT:
                    return self._deliver_missing(user_text, "no_answer", turn)
                return self._hold(user_text, out.reply, turn,
                                  repose=not out.reply)

            if out.action == "discomfort":
                return self._deliver_step(
                    user_text, runtime.offer_pause(clinical), turn,
                    ack=out.reply)

            if out.action == "tangent":
                if runtime.note_aside(clinical) >= runtime.ASIDE_LIMIT:
                    return self._deliver_step(
                        user_text, runtime.offer_pause(clinical), turn,
                        ack=out.reply)
                return self._hold(user_text, out.reply, turn)

            return self._hold(user_text, out.reply, turn)

    def _deliver_missing(self, user_text, reason, turn):
        step = runtime.mark_missing(self.clinical, reason)
        return self._deliver_step(user_text, step, turn)

    def _hold_probe(self, user_text, turn):
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
        spoken = []
        for utt in utterances:
            if self._aborted(turn):
                return False
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
                    continue
                pending = True
            if self._aborted(turn):
                return False
            spoken.append(text)
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
        self._history_begin(user_text)
        self.state = "processing"
        utterances = tuple(step.utterances)
        if ack:
            if utterances and isinstance(utterances[0], runtime.LLMSay):
                utterances = (runtime.LLMSay(
                    f"First acknowledge what the person just said, using these "
                    f"words or very close to them: {ack!r}. Then, in the same "
                    f"breath and without repeating yourself, {utterances[0].instruction}"),
                ) + utterances[1:]
            else:
                utterances = (runtime.Speak(ack),) + utterances
        self._speak_beats(utterances, turn)
        if step.expect.kind == "end":
            self.ended = True

    def get_next_video(self):
        try:
            item = self.video_queue.get_nowait()
            if item is None:
                self.state = "idle"
                return False
            return item
        except queue.Empty:
            return None

    def get_chat_history(self):
        return list(self.chat_history)

    def reset(self):
        self.cancel_event.set()
        self._turn += 1
        time.sleep(0.1)
        self.cancel_event.clear()
        self.chat_history.clear()
        self.patient.clear()
        self.ended = False
        self.clinical = runtime.ClinicalSession()
        runtime.start(self.clinical)
        self._flush()
        self.state = "idle"
