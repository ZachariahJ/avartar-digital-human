"""Core orchestrator: streaming sentence-by-sentence processing with barge-in support."""

import os
import threading
import logging
import queue
import time
from concurrent.futures import ThreadPoolExecutor, Future

import config
from modules import asr, llm, tts, avatar

logger = logging.getLogger(__name__)

_fixed_clip_lock = threading.Lock()


def ensure_fixed_clip(text, path):
    """Render a FIXED line (`text`) to a cached clip at `path` ONCE and reuse it —
    no per-session LLM/TTS/FLOAT, plays instantly. Regenerates only if the text
    changed (tracked via a sidecar .txt). Returns the cached path, or None on failure.
    Used for the always-identical greeting and consent-decline clips."""
    sidecar = path + ".txt"
    with _fixed_clip_lock:
        try:
            if os.path.exists(path) and os.path.exists(sidecar):
                with open(sidecar, encoding="utf-8") as f:
                    if f.read() == text:
                        return path  # cached and up to date
        except OSError:
            pass
        try:
            tts_path = tts.synthesize(text, None, None)
            if not tts_path:
                return None
            os.makedirs(os.path.dirname(path), exist_ok=True)
            result = avatar.generate_video(tts_path, output_path=path)
            try:
                os.remove(tts_path)
            except OSError:
                pass
            if result is None:
                return None
            with open(sidecar, "w", encoding="utf-8") as f:
                f.write(text)
            logger.info("Fixed clip cached at %s", path)
            return path
        except Exception as e:
            logger.warning("Fixed clip generation failed for %s: %s", path, e)
            return None


def prewarm_fixed_clips():
    """Pre-render the fixed greeting + decline clips once (called at startup warmup)
    so both play instantly on first use."""
    ensure_fixed_clip(config.GREETING_TEXT, config.GREETING_VIDEO_PATH)
    ensure_fixed_clip(config.DECLINE_TEXT, config.DECLINE_VIDEO_PATH)


class Pipeline:
    """State machine: idle → listening → processing → speaking → idle"""

    def __init__(self):
        self.state = "idle"  # idle, listening, processing, speaking
        self.cancel_event = threading.Event()
        self.video_queue = queue.Queue()
        self.chat_history = []  # List of {"role": ..., "content": ...} (frontend display)
        self.llm_history = []   # Conversation sent to the LLM API (this session only)
        # Structured patient profile (age, sex, substances, screening scores, ...),
        # injected into the prompt every turn so it survives history-window trimming
        # and the model never loses key clinical facts.
        self.patient = {}
        # True from when the fixed greeting is delivered until the user's consent
        # reply is handled — that one turn is routed through classify_consent so a
        # "no" gets the fixed decline clip instead of the full LLM.
        self.awaiting_consent = False
        # Set when the user declines consent: the server then turns the mic off and
        # tells the client to stop, ending the session until Start is pressed again.
        self.ended = False
        self._lock = threading.Lock()
        self._processing_thread = None
        # Track which sentences were actually played
        self._pending_assistant_text = ""
        self._played_sentences = []
        # Monotonic turn id: each new utterance bumps it. A response only touches
        # shared state (enqueue video) while it still owns the current turn, so a
        # barged-in response can't leak stale segments into the next turn even
        # after cancel_event is cleared for the new turn.
        self._turn = 0
        # perf_counter() at the moment the user stopped speaking — the T0 for the
        # latency waterfall logged through the rest of the turn.
        self._t0 = 0.0

    def _aborted(self, turn):
        """True if this response was cancelled or superseded by a newer turn."""
        return self.cancel_event.is_set() or turn != self._turn

    def on_speech_start(self):
        """Called by audio_server when user starts speaking (barge-in)."""
        if self.state in ("processing", "speaking"):
            logger.info("Barge-in detected! Cancelling current response.")
            self.cancel_event.set()
            self._turn += 1  # invalidate the in-flight response immediately
            # Clear video queue
            while not self.video_queue.empty():
                try:
                    self.video_queue.get_nowait()
                except queue.Empty:
                    break
            # Histories are pipeline-owned and committed together by the in-flight
            # producer's finally (whatever was generated so far lands in BOTH
            # chat_history and llm_history), so the two stay consistent without any
            # truncation here.
            self._pending_assistant_text = ""
            self._played_sentences = []

        self.state = "listening"

    def cancel_response(self):
        """Stop any in-flight response immediately and end at idle (used by Stop and
        by text-send interruption). Drops queued clips; leaves histories intact."""
        if self.state in ("processing", "speaking"):
            self.cancel_event.set()
            self._turn += 1
            while not self.video_queue.empty():
                try:
                    self.video_queue.get_nowait()
                except queue.Empty:
                    break
        self.state = "idle"

    def on_speech_end(self, audio_array):
        """Called by audio_server when user finishes speaking.
        audio_array: float32 numpy array at 16kHz.
        """
        self._t0 = time.perf_counter()
        self.state = "processing"
        self.cancel_event.clear()
        self._pending_assistant_text = ""
        self._played_sentences = []
        self._turn += 1
        turn = self._turn

        # Run processing in background thread
        self._processing_thread = threading.Thread(
            target=self._process_speech, args=(audio_array, turn), daemon=True
        )
        self._processing_thread.start()

    def on_speech_end_text(self, text):
        """Text input: skip ASR and feed text directly into pipeline."""
        self._t0 = time.perf_counter()
        self.state = "processing"
        self.cancel_event.clear()
        self._pending_assistant_text = ""
        self._played_sentences = []
        self._turn += 1
        turn = self._turn

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
        self.state = "processing"
        self.cancel_event.clear()
        self._turn += 1
        turn = self._turn
        self._processing_thread = threading.Thread(
            target=self._process_greeting, args=(turn,), daemon=True
        )
        self._processing_thread.start()

    def _process_greeting(self, turn):
        """Deliver the fixed opening from the cached clip — no LLM/TTS/FLOAT. Records
        it as the assistant's first turn so the conversation flows straight into the
        user's yes/no consent reply."""
        try:
            if self._aborted(turn):
                self.state = "idle"
                return
            text = config.GREETING_TEXT
            video = ensure_fixed_clip(config.GREETING_TEXT, config.GREETING_VIDEO_PATH)
            # Record the fixed opening as the assistant's first turn in both histories.
            self._history_set_assistant(text)
            # The next user turn is the consent answer — route it through the gate.
            self.awaiting_consent = True
            if video and not self._aborted(turn):
                self.video_queue.put({
                    "video": video,
                    "sentence": text,
                    "_t_enqueue": time.perf_counter(),
                })
                self.state = "speaking"
                self.video_queue.put(None)
            else:
                # No clip (generation failed / not ready) — the text greeting still shows.
                self.state = "idle"
        except Exception as e:
            logger.error(f"Greeting error: {e}", exc_info=True)
            self.state = "idle"

    def _deliver_decline(self, user_text, turn):
        """User declined consent: record their reply + the FIXED thank-you line, play
        the cached decline clip, and end the turn. No screening, no dynamic LLM."""
        self._history_begin(user_text)          # record the user's "no" in both histories
        text = config.DECLINE_TEXT
        video = ensure_fixed_clip(text, config.DECLINE_VIDEO_PATH)
        self._history_set_assistant(text)
        if video and not self._aborted(turn):
            self.video_queue.put({
                "video": video,
                "sentence": text,
                "_t_enqueue": time.perf_counter(),
            })
            self.state = "speaking"
            self.video_queue.put(None)
        else:
            self.state = "idle"
        # Consent declined: end the session (the server turns the mic off + tells the
        # client to stop) now that the goodbye clip is queued for delivery.
        self.ended = True

    # ---------- Shared history management (both histories move together) ----------
    # chat_history (frontend display) and llm_history (sent to the API) are kept in
    # lockstep here so they can never diverge on barge-in or LLM errors. llm.py no
    # longer mutates history at all — the Pipeline is the single owner.
    def _history_begin(self, user_text):
        """Append the user turn to BOTH histories and return the API message list.
        If the previous turn's user message is still unanswered (e.g. speech split
        by a pause into two segments, so the first half hasn't been replied to yet),
        MERGE the new text into it — never drop it. This preserves everything the
        user said, reassembles the paused sentence into one turn, and still avoids
        sending two user messages in a row to the API."""
        with self._lock:
            for hist in (self.chat_history, self.llm_history):
                if hist and hist[-1]["role"] == "user":
                    hist[-1] = {"role": "user",
                                "content": (hist[-1]["content"].rstrip()
                                            + " " + user_text).strip()}
                else:
                    hist.append({"role": "user", "content": user_text})
            llm._trim_history(self.llm_history)
            messages = llm.build_messages(self.llm_history, dict(self.patient))
            history_snapshot = list(self.llm_history)
        # Fire-and-forget structured extraction to keep the patient profile current.
        # Runs off the hot path (never blocks TTS/display) and applies to the NEXT
        # turn's prompt, so age/sex/screening facts survive history-window trimming.
        threading.Thread(target=self._extract_patient, args=(history_snapshot,),
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
        logger.info("[patient] profile now: %s", self.patient)

    def _history_set_assistant(self, text):
        """Create or update the current assistant turn with the SAME text in both
        histories, so display and API memory always agree."""
        with self._lock:
            for hist in (self.chat_history, self.llm_history):
                if hist and hist[-1]["role"] == "assistant":
                    hist[-1]["content"] = text
                else:
                    hist.append({"role": "assistant", "content": text})

    def _process_speech(self, audio_array, turn):
        """Full pipeline: ASR → LLM stream → TTS+FLOAT (pipelined) → video queue."""
        try:
            # Step 1: ASR
            if self._aborted(turn):
                self.state = "idle"
                return

            user_text = asr.transcribe_array(audio_array, sample_rate=16000)
            logger.info(f"[latency] ASR done at +{time.perf_counter() - self._t0:.2f}s "
                        f"-> {user_text!r}")

            if not user_text.strip():
                self.state = "idle"
                return

            if self._aborted(turn):
                self.state = "idle"
                return

            # Consent gate: the turn right after the greeting is the yes/no answer.
            # A clear decline gets the fixed cached clip; yes/unclear/distress fall
            # through to the full counselor (which owns the crisis protocol).
            if self.awaiting_consent:
                self.awaiting_consent = False
                if llm.classify_consent(user_text) == "no":
                    self._deliver_decline(user_text, turn)
                    return

            # Step 2: append the user turn to BOTH histories, then synthesize.
            messages = self._history_begin(user_text)
            self.state = "processing"
            self._run_synthesis(messages, turn)

        except Exception as e:
            logger.error(f"Pipeline error: {e}", exc_info=True)
            self.state = "idle"

    def _process_text(self, user_text, turn):
        """Text-only pipeline: skip ASR, go straight to LLM → TTS+FLOAT (pipelined)."""
        try:
            if self._aborted(turn):
                self.state = "idle"
                return

            # Consent gate (same as the voice path): a clear decline -> fixed clip.
            if self.awaiting_consent:
                self.awaiting_consent = False
                if llm.classify_consent(user_text) == "no":
                    self._deliver_decline(user_text, turn)
                    return

            messages = self._history_begin(user_text)
            self.state = "processing"
            self._run_synthesis(messages, turn)

        except Exception as e:
            logger.error(f"Pipeline error (text): {e}", exc_info=True)
            self.state = "idle"

    def _run_synthesis(self, messages, turn):
        """Dispatch to the configured synthesis mode. `messages` is the full API
        message list (system + history + current user) built by the caller; the user
        turn is already in both histories."""
        mode = getattr(config, "SYNTHESIS_MODE", "stream_parallel")
        if mode == "batch":
            self._run_batch_synthesis(messages, turn)
        elif mode == "stream":
            self._run_pipelined_synthesis(messages, turn)
        elif mode == "hybrid":
            self._run_hybrid_synthesis(messages, turn)
        else:
            self._run_streaming_parallel_synthesis(messages, turn)

    def _run_streaming_parallel_synthesis(self, messages, turn):
        """Fastest mode: stream sentences from the LLM (chat text appears live in
        <1s) AND render each sentence's TTS+FLOAT concurrently across the whole
        GPU pool, while enqueuing strictly in sentence order.

        A producer thread pulls sentences off the LLM stream and submits each as a
        TTS->FLOAT job to a pool sized to len(FLOAT_GPUS); this consumer reads the
        resulting futures IN ORDER and enqueues the finished clips. So sentence 1
        starts playing after just 1 TTS + 1 FLOAT, while sentences 2..N are already
        rendering on the other GPUs -> no stall between segments.
        """
        futures_q = queue.Queue()
        SENTINEL = object()
        n_gpus = max(1, len(config.FLOAT_GPUS))

        def _render(sentence, idx):
            if self._aborted(turn) or config.SHUTTING_DOWN.is_set():
                return None
            # First sentence renders at a lower NFE to get the avatar talking sooner.
            nfe = config.FLOAT_NFE_FIRST if idx == 0 else config.FLOAT_NFE
            t_tts0 = time.perf_counter()
            tts_path = tts.synthesize(sentence, None, self.cancel_event)
            if tts_path is None or self._aborted(turn):
                return None
            t_float0 = time.perf_counter()
            video_path = avatar.generate_video(tts_path, nfe=nfe)
            # The wav is consumed by FLOAT; drop it now so tmp/ doesn't fill up.
            try:
                os.remove(tts_path)
            except OSError:
                pass
            logger.info("[latency] seg %d rendered: tts=%.2fs float=%.2fs (nfe=%d)",
                        idx, t_float0 - t_tts0, time.perf_counter() - t_float0, nfe)
            return video_path

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
                    logger.info(f"LLM sentence: {sentence}")

                    # Live update BOTH histories so display and API memory agree.
                    self._history_set_assistant(full_response)

                    futures_q.put((executor.submit(_render, sentence, idx), sentence))
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
                    self._pending_assistant_text = full_response
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
                fut, sentence = item
                if self._aborted(turn):
                    fut.cancel()
                    continue  # keep draining to SENTINEL so the producer finishes
                try:
                    video_path = fut.result()
                except Exception:
                    # A single sentence's TTS/FLOAT failing must NOT abort the whole
                    # turn (which would skip the video_end sentinel below and freeze
                    # the avatar on its last frame). Skip this clip and continue.
                    logger.exception("segment render failed; skipping this clip")
                    continue
                if video_path is None or self._aborted(turn):
                    continue
                self.video_queue.put({
                    "video": video_path,
                    "sentence": sentence,
                    "_t_enqueue": time.perf_counter(),
                })
                self.state = "speaking"
                if first_seg:
                    first_seg = False
                    logger.info("[latency] FIRST segment enqueued at +%.2fs",
                                time.perf_counter() - self._t0)

            producer.join(timeout=1.0)

        if not self._aborted(turn):
            self.video_queue.put(None)
            self.state = "speaking"

    @staticmethod
    def _split_sentences(text):
        """Split a full response into sentences (same boundaries as the LLM
        streamer): only on sentence-final punctuation, so each spoken clip is a
        whole sentence."""
        endings = {"。", "！", "？", ".", "!", "?", "\n"}
        sentences, buf = [], ""
        for ch in text:
            buf += ch
            if ch in endings:
                s = buf.strip()
                if s:
                    sentences.append(s)
                buf = ""
        if buf.strip():
            sentences.append(buf.strip())
        return sentences

    def _run_hybrid_synthesis(self, messages, turn):
        """Hybrid: get the FULL LLM answer first so the chat shows the complete,
        stable text at once (no live updates), then render per-sentence TTS+FLOAT
        (pipelined) so the avatar starts speaking quickly. The displayed answer
        text never changes while it plays.
        """
        if self._aborted(turn):
            self.state = "idle"
            return

        # 1. Full answer in one shot (no live, sentence-by-sentence chat updates).
        full_response = (llm.chat(messages) or "").strip()
        logger.info(f"LLM full response: {full_response}")
        if not full_response or self._aborted(turn):
            # Nothing to commit; leave the user message (merged by the next turn).
            self.state = "idle"
            return

        # 2. Commit the complete answer to BOTH histories (one stable chat entry).
        self._history_set_assistant(full_response)
        self._pending_assistant_text = full_response

        # 3. Render sentences CONCURRENTLY across the FLOAT GPU pool, but enqueue
        #    them strictly in order so the avatar still speaks them in sequence.
        #    avatar.generate_video() is thread-safe and hands each call a free GPU
        #    (0/1/2), so up to len(FLOAT_GPUS) sentences render in parallel,
        #    cutting total render time from ~N to ~ceil(N/len(GPUs)) clips.
        #    Note: first-segment latency is unchanged (still one TTS + one render);
        #    parallelism speeds up sentences 2..N so playback doesn't stall.
        sentences = self._split_sentences(full_response)

        def _render(sentence):
            if self._aborted(turn):
                return None
            tts_path = tts.synthesize(sentence, None, self.cancel_event)
            if tts_path is None or self._aborted(turn):
                return None
            return avatar.generate_video(tts_path)

        with ThreadPoolExecutor(max_workers=max(1, len(config.FLOAT_GPUS))) as ex:
            futures = [ex.submit(_render, s) for s in sentences]
            for i, fut in enumerate(futures):
                if self._aborted(turn):
                    for g in futures[i:]:
                        g.cancel()
                    break
                # Wait for sentence i in order (sentences i+1.. render in parallel).
                video_path = fut.result()
                if video_path is None or self._aborted(turn):
                    for g in futures[i:]:
                        g.cancel()
                    break
                self.video_queue.put({"video": video_path, "sentence": sentences[i]})
                self.state = "speaking"

        if not self._aborted(turn):
            self.video_queue.put(None)
            self.state = "speaking"

    def _run_batch_synthesis(self, messages, turn):
        """Batch mode: get the FULL LLM answer first, show it at once, then do a
        SINGLE TTS + FLOAT render. The chat no longer updates live and the avatar
        plays one continuous clip instead of changing per sentence.
        """
        if self._aborted(turn):
            self.state = "idle"
            return

        # 1. Full answer in one shot (no live, sentence-by-sentence updates).
        full_response = (llm.chat(messages) or "").strip()
        logger.info(f"LLM full response: {full_response}")
        if not full_response or self._aborted(turn):
            # Nothing to commit; leave the user message (merged by the next turn).
            self.state = "idle"
            return

        # 2. Commit the complete answer to BOTH histories (one stable chat entry).
        self._history_set_assistant(full_response)
        self._pending_assistant_text = full_response

        # 3. One TTS for the whole answer.
        tts_path = tts.synthesize(full_response, None, self.cancel_event)
        if tts_path is None or self._aborted(turn):
            self.state = "idle"
            return

        # 4. One FLOAT video for the whole answer.
        video_path = avatar.generate_video(tts_path)
        if video_path is None or self._aborted(turn):
            self.state = "idle"
            return

        # 5. Play it as a single segment, then signal completion.
        self.video_queue.put({"video": video_path, "sentence": full_response})
        self.state = "speaking"
        self.video_queue.put(None)

    def _run_pipelined_synthesis(self, messages, turn):
        """Pipelined LLM → TTS → FLOAT: overlap TTS(N+1) with FLOAT(N)."""
        full_response = ""
        pending_tts_future = None  # Future for TTS of next sentence
        pending_tts_sentence = None

        with ThreadPoolExecutor(max_workers=1) as tts_executor:
            for sentence in llm.chat_stream(messages, cancel_event=self.cancel_event):
                if self._aborted(turn):
                    break

                full_response += sentence
                logger.info(f"LLM sentence: {sentence}")

                # Progressively update BOTH histories so display and API memory agree.
                self._history_set_assistant(full_response)

                if self._aborted(turn):
                    break

                # If there's a pending TTS result from prev iteration, render its FLOAT now
                # while we kick off TTS for current sentence in parallel
                if pending_tts_future is not None:
                    # Start TTS for current sentence in background
                    current_tts_future = tts_executor.submit(
                        tts.synthesize, sentence, None, self.cancel_event
                    )
                    current_tts_sentence = sentence

                    # Wait for previous TTS and render FLOAT
                    prev_tts_path = pending_tts_future.result()
                    if prev_tts_path is None or self._aborted(turn):
                        pending_tts_future = None
                        break

                    video_path = avatar.generate_video(prev_tts_path)
                    if video_path is None or self._aborted(turn):
                        pending_tts_future = None
                        break

                    self.video_queue.put({
                        "video": video_path,
                        "sentence": pending_tts_sentence
                    })
                    self.state = "speaking"

                    # Current sentence's TTS is now our pending
                    pending_tts_future = current_tts_future
                    pending_tts_sentence = current_tts_sentence
                else:
                    # First sentence: just start TTS, no FLOAT to overlap with
                    pending_tts_future = tts_executor.submit(
                        tts.synthesize, sentence, None, self.cancel_event
                    )
                    pending_tts_sentence = sentence

            # Process the last pending TTS result
            if pending_tts_future is not None and not self._aborted(turn):
                tts_path = pending_tts_future.result()
                if tts_path is not None and not self._aborted(turn):
                    video_path = avatar.generate_video(tts_path)
                    if video_path is not None and not self._aborted(turn):
                        self.video_queue.put({
                            "video": video_path,
                            "sentence": pending_tts_sentence
                        })
                        self.state = "speaking"

        # Finalize history only if we still own the turn; never delete the user's
        # message (a produced-nothing turn is merged by the next turn).
        if full_response.strip() and not self._aborted(turn):
            self._history_set_assistant(full_response)
            self._pending_assistant_text = full_response
            self.video_queue.put(None)

        if not self._aborted(turn):
            self.state = "speaking"

    def get_next_video(self):
        """Non-blocking: get next video from queue.

        Returns:
            dict with "video" and "sentence" keys, or None if queue empty,
            or False if response is complete.
        """
        try:
            item = self.video_queue.get_nowait()
            if item is None:
                # End of response
                self.state = "idle"
                return False
            self._played_sentences.append(item["sentence"])
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
        self.awaiting_consent = False
        self.ended = False
        self._pending_assistant_text = ""
        self._played_sentences = []
        while not self.video_queue.empty():
            try:
                self.video_queue.get_nowait()
            except queue.Empty:
                break
        self.state = "idle"
        self.llm_history.clear()
