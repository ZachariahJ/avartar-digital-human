
import logging
import time

import torch
import numpy as np

import config
from modules import eou

logger = logging.getLogger(__name__)

_FLOOR_FALL = 0.30
_FLOOR_RISE = 0.005
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
        self.noise_floor_db = config.VAD_NOISE_FLOOR_INIT_DBFS
        self._gate_logged_at = 0.0
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
        if not self.use_eou:
            return self.silent_chunks >= self.silence_chunks_needed

        if self.silent_chunks >= self.eou_max_silence_chunks:
            logger.info("[eou] forced end at %.2fs silence (max cap)",
                        self.silent_chunks * self.chunk_size / self.sample_rate)
            return True

        past_pause = self.silent_chunks - self.eou_pause_chunks
        if past_pause < 0 or (past_pause > 0 and past_pause % self.eou_recheck_chunks != 0):
            return False

        p = eou.predict_complete(np.concatenate(self.speech_buffer))
        if p is None:
            return self.silent_chunks >= self.silence_chunks_needed
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
        rms = float(np.sqrt(np.mean(chunk * chunk)))
        return 20.0 * np.log10(max(rms, _EPS))

    def _track_ambient(self, level_db: float):
        rate = _FLOOR_FALL if level_db < self.noise_floor_db else _FLOOR_RISE
        self.noise_floor_db += rate * (level_db - self.noise_floor_db)

    def _is_near_field(self, level_db: float) -> bool:
        margin = self.near_field_release if self.is_speaking else self.near_field_margin
        return level_db >= max(self.min_level_db, self.noise_floor_db + margin)

    def process_chunk(self, audio_chunk: np.ndarray):
        if audio_chunk.dtype == np.int16:
            audio_f32 = audio_chunk.astype(np.float32) / 32768.0
        else:
            audio_f32 = audio_chunk.astype(np.float32)

        for i in range(0, len(audio_f32), self.chunk_size):
            sub = audio_f32[i : i + self.chunk_size]
            level_db = self._level_db(sub)
            if len(sub) < self.chunk_size:
                sub = np.pad(sub, (0, self.chunk_size - len(sub)))

            tensor = torch.from_numpy(sub)
            prob = self.model(tensor, self.sample_rate).item()

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

        if self.is_speaking and self.silent_chunks == 0 and len(self.speech_buffer) > 0:
            if len(self.speech_buffer) <= len(audio_f32) // self.chunk_size + 1:
                return ("speech_start", None)

        return (None, None)

    def _log_gated(self, level_db: float, prob: float):
        now = time.monotonic()
        if now - self._gate_logged_at < 1.0:
            return
        self._gate_logged_at = now
        logger.info("[vad] far-field speech ignored: %.0fdBFS vs floor %.0f+%.0fdB "
                    "(p=%.2f)", level_db, self.noise_floor_db,
                    self.near_field_release if self.is_speaking else self.near_field_margin,
                    prob)

    def pending_audio(self):
        if self.is_speaking and self.speech_buffer:
            return np.concatenate(self.speech_buffer)
        return None


if __name__ == "__main__":
    vad = VoiceActivityDetector()
    print("VAD loaded successfully.")
