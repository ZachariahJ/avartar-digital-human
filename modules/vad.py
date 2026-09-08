"""Turn detection: when has the user started talking, and when have they stopped.

Silero answers only "is this chunk speech", which is short of what turn-taking
needs in two directions, and this module supplies both.

Whether the speech is even addressed to us: in a shared room the next table is
speech too, and Silero says so. The near-field gate below adds the dimension
that separates them — level above the room's own ambient — so background
conversation never reaches the buffer that barge-in and the ASR read.

Whether a silence means the turn is over: modules/eou.py judges whether the
sentence sounds finished, and _should_end_turn combines that with the silence
clock, falling back to pure silence timing whenever that model is unavailable.
"""

import logging
import time

import torch
import numpy as np

import config
from modules import eou

logger = logging.getLogger(__name__)

# How fast the ambient estimate follows the room, per 32ms chunk. Asymmetric on
# purpose: it drops toward a new quiet almost immediately, so a lull is
# recognised as the true floor, but climbs slowly (~6s to settle) so that one
# passing voice — or the user's own, if the gate ever misjudges it — cannot
# drag the bar up behind itself and deafen the detector.
_FLOOR_FALL = 0.30
_FLOOR_RISE = 0.005
# Silence in int16 is exactly zero, and log(0) is not a number. This is ~-140dB,
# far below anything a microphone produces.
_EPS = 1e-7


class VoiceActivityDetector:
    def __init__(
        self,
        threshold: float = config.VAD_THRESHOLD,
        silence_duration: float = config.VAD_SILENCE_DURATION,
        sample_rate: int = 16000,
    ):
        self.threshold = threshold
        self.sample_rate = sample_rate
        self.near_field_margin = config.VAD_NEAR_FIELD_MARGIN_DB
        self.near_field_release = config.VAD_NEAR_FIELD_RELEASE_DB
        self.min_level_db = config.VAD_MIN_LEVEL_DBFS
        # Ambient level of the room. Lives on the detector, not on a turn: it
        # describes where the session is sitting, so _reset() deliberately
        # leaves it alone and the estimate carries across utterances.
        self.noise_floor_db = config.VAD_NOISE_FLOOR_INIT_DBFS
        self._gate_logged_at = 0.0
        # Fixed by Silero: it consumes exactly 512 samples per call, 32ms at
        # 16kHz. Every duration below is converted into a count of these.
        self.chunk_size = 512
        self.silence_chunks_needed = int(silence_duration * sample_rate / self.chunk_size)

        chunk_sec = self.chunk_size / sample_rate
        self.use_eou = config.USE_EOU
        self.eou_threshold = config.EOU_THRESHOLD
        self.eou_confirm = max(1, int(config.EOU_CONFIRM_CONSULTS))
        self.eou_pause_chunks = max(1, round(config.EOU_PAUSE_DURATION / chunk_sec))
        self.eou_recheck_chunks = max(1, round(config.EOU_RECHECK_DURATION / chunk_sec))
        self.eou_max_silence_chunks = max(
            self.eou_pause_chunks + 1, round(config.EOU_MAX_SILENCE / chunk_sec)
        )

        self.model, _ = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            trust_repo=True,
        )
        self.model.eval()

        self._reset()

    def _reset(self):
        self.model.reset_states()
        self.is_speaking = False
        self.silent_chunks = 0
        self.speech_buffer = []
        self.eou_complete_streak = 0

    def _should_end_turn(self) -> bool:
        """Whether the trailing silence means the user's turn is over.

        Without the EOU model this is purely a silence threshold. With it, the
        model is consulted at the first brief pause and again at intervals while
        the silence continues, ending the turn once it judges the sentence
        finished.
        """
        if not self.use_eou:
            return self.silent_chunks >= self.silence_chunks_needed

        # Bounds the wait: without this, a model that never returns "complete"
        # would leave the user's turn open indefinitely.
        if self.silent_chunks >= self.eou_max_silence_chunks:
            logger.info("[eou] forced end at %.2fs silence (max cap)",
                        self.silent_chunks * self.chunk_size / self.sample_rate)
            return True

        # Inference is not free, so consult at the first pause and then only
        # once per recheck interval rather than on every chunk of silence.
        past_pause = self.silent_chunks - self.eou_pause_chunks
        if past_pause < 0 or (past_pause > 0 and past_pause % self.eou_recheck_chunks != 0):
            return False

        p = eou.predict_complete(np.concatenate(self.speech_buffer))
        if p is None:
            return self.silent_chunks >= self.silence_chunks_needed
        # Require consecutive agreeing verdicts. A clause boundary can produce
        # one high reading mid-sentence, and acting on it cuts the user off.
        if p >= self.eou_threshold:
            self.eou_complete_streak += 1
        else:
            self.eou_complete_streak = 0
        end = self.eou_complete_streak >= self.eou_confirm
        logger.info("[eou] P=%.2f silent=%.2fs streak=%d/%d -> %s", p,
                    self.silent_chunks * self.chunk_size / self.sample_rate,
                    self.eou_complete_streak, self.eou_confirm,
                    "END" if end else "hold")
        return end

    @staticmethod
    def _level_db(chunk: np.ndarray) -> float:
        """Chunk level in dBFS, where 0 is full scale."""
        rms = float(np.sqrt(np.mean(chunk * chunk)))
        return 20.0 * np.log10(max(rms, _EPS))

    def _track_ambient(self, level_db: float):
        """Fold one non-speech chunk into the running estimate of the room."""
        rate = _FLOOR_FALL if level_db < self.noise_floor_db else _FLOOR_RISE
        self.noise_floor_db += rate * (level_db - self.noise_floor_db)

    def _is_near_field(self, level_db: float) -> bool:
        """Whether a voice-like chunk is close enough to be addressed to us.

        The bar is the louder of two: a margin above the room's own level, which
        is what actually separates the microphone's owner from the next table,
        and an absolute minimum for a room so quiet the first bar sinks below
        anything a person at the microphone could produce.
        """
        margin = self.near_field_release if self.is_speaking else self.near_field_margin
        return level_db >= max(self.min_level_db, self.noise_floor_db + margin)

    def process_chunk(self, audio_chunk: np.ndarray):
        """Feed one chunk of microphone audio and report any turn boundary.

        Args:
            audio_chunk: int16 or float32 samples at 16kHz, any length.

        Returns:
            (event, audio) where event is "speech_start", "speech_end" or None.
            On "speech_end" the audio is the whole buffered utterance as float32;
            otherwise it is None.
        """
        if audio_chunk.dtype == np.int16:
            audio_f32 = audio_chunk.astype(np.float32) / 32768.0
        else:
            audio_f32 = audio_chunk.astype(np.float32)

        # Silero requires exactly chunk_size samples per call, so a caller's
        # chunk is split up and any remainder is zero-padded.
        for i in range(0, len(audio_f32), self.chunk_size):
            sub = audio_f32[i : i + self.chunk_size]
            # Measured before padding: the zeros are not part of the room, and
            # averaging them in would read a short trailing slice as quieter
            # than it was.
            level_db = self._level_db(sub)
            if len(sub) < self.chunk_size:
                sub = np.pad(sub, (0, self.chunk_size - len(sub)))

            tensor = torch.from_numpy(sub)
            prob = self.model(tensor, self.sample_rate).item()

            # Both questions, in order: voice-like, and near enough to be meant
            # for us. A chunk that fails either one is silence as far as
            # everything downstream is concerned, and it is also what the
            # ambient estimate is built from — background conversation raises
            # the floor it is measured against, which is what lets the gate
            # stay shut for as long as the room stays noisy.
            voice = prob >= self.threshold
            if voice and self.near_field_margin > 0 and not self._is_near_field(level_db):
                voice = False
                self._log_gated(level_db, prob)
            if not voice:
                self._track_ambient(level_db)

            if voice:
                self.silent_chunks = 0
                if not self.is_speaking:
                    self.is_speaking = True
                    self.speech_buffer = []
                self.speech_buffer.append(sub)
            else:
                if self.is_speaking:
                    self.speech_buffer.append(sub)
                    self.silent_chunks += 1
                    if self._should_end_turn():
                        full_audio = np.concatenate(self.speech_buffer)
                        self._reset()
                        return ("speech_end", full_audio)

        # Report the onset only on the call that first saw speech: past that,
        # the buffer holds more than this call could have contributed.
        if self.is_speaking and self.silent_chunks == 0 and len(self.speech_buffer) > 0:
            if len(self.speech_buffer) <= len(audio_f32) // self.chunk_size + 1:
                return ("speech_start", None)

        return (None, None)

    def _log_gated(self, level_db: float, prob: float):
        """Report gated speech at most once a second: it can fire 30x/s."""
        now = time.monotonic()
        if now - self._gate_logged_at < 1.0:
            return
        self._gate_logged_at = now
        logger.info("[vad] far-field speech ignored: %.0fdBFS vs floor %.0f+%.0fdB "
                    "(p=%.2f)", level_db, self.noise_floor_db,
                    self.near_field_release if self.is_speaking else self.near_field_margin,
                    prob)

    def pending_audio(self):
        """The utterance so far, or None if the user is not currently speaking.

        Lets barge-in transcribe speech that has not yet reached a turn boundary,
        instead of waiting for speech_end while the avatar talks over the user.
        """
        if self.is_speaking and self.speech_buffer:
            return np.concatenate(self.speech_buffer)
        return None


if __name__ == "__main__":
    # Smoke test: the Silero download and load are the parts that break.
    vad = VoiceActivityDetector()
    print("VAD loaded successfully.")
