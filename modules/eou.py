"""Decides whether a pause means the speaker is finished.

A silence-duration VAD cannot tell a finished sentence from someone thinking
mid-sentence, so it cuts people off. This model reads intonation and filler
words straight from the waveform and predicts whether the turn is semantically
complete; modules/vad.py uses it to gate speech_end.

It is a strict add-on and can never break the pipeline: predict_complete()
returns None whenever the model or onnxruntime is unavailable, and the VAD then
falls back to plain silence timing.

Runs on CPU (~9MB model, ~28ms per consult), deliberately away from the MuseTalk
GPUs. Running it on GPU would need onnxruntime-gpu and a CUDA build matching the
pinned torch, which buys nothing at this size.
"""

import logging
import threading

import numpy as np

import config

logger = logging.getLogger(__name__)

SR = 16000
WINDOW_SEC = 8  # fixed by the model: it consumes exactly this much audio

_session = None
_feature_extractor = None
_input_name = None
_load_failed = False
_lock = threading.Lock()


def _load():
    """Build the ONNX session and its feature extractor. Raises on failure."""
    global _session, _feature_extractor, _input_name
    import onnxruntime as ort
    from transformers import WhisperFeatureExtractor

    so = ort.SessionOptions()
    so.inter_op_num_threads = 1
    so.intra_op_num_threads = config.EOU_ONNX_THREADS
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    sess = ort.InferenceSession(
        config.EOU_MODEL_PATH, sess_options=so, providers=["CPUExecutionProvider"]
    )
    _feature_extractor = WhisperFeatureExtractor(chunk_length=WINDOW_SEC)
    _input_name = sess.get_inputs()[0].name
    _session = sess


def get_model():
    """The ONNX session, loaded on first use, or None if it cannot be loaded.

    A failed load is remembered, so a missing model costs one warning rather
    than a failed load attempt on every pause. Double-checked locking, because
    the startup pre-warm and a first user request race here.
    """
    global _load_failed
    if _session is None and not _load_failed:
        with _lock:
            if _session is None and not _load_failed:
                try:
                    _load()
                    logger.info("EOU (smart-turn v3) loaded from %s", config.EOU_MODEL_PATH)
                except Exception as e:
                    _load_failed = True
                    logger.warning(
                        "EOU model load failed (%s); falling back to silence-only VAD", e
                    )
    return _session


def predict_complete(audio: np.ndarray) -> float | None:
    """Probability in [0,1] that the speaker has finished their turn.

    Args:
        audio: float32 mono at 16kHz — the speech so far, including the
            trailing pause being judged.

    Returns:
        The probability, or None when the model is unavailable or inference
        failed. Callers must read None as "no opinion" and fall back to silence
        timing rather than treating it as a low score.
    """
    sess = get_model()
    if sess is None:
        return None
    try:
        a = np.asarray(audio, dtype=np.float32)
        # Trim from the front: the cues that mark a turn ending are all at the
        # end of the utterance.
        if a.size > WINDOW_SEC * SR:
            a = a[-WINDOW_SEC * SR:]
        inputs = _feature_extractor(
            a,
            sampling_rate=SR,
            return_tensors="np",
            padding="max_length",
            max_length=WINDOW_SEC * SR,
            truncation=True,
            do_normalize=True,
        )
        feats = inputs.input_features.squeeze(0).astype(np.float32)[None, ...]
        out = sess.run(None, {_input_name: feats})
        # Despite being named "logits", this output already has sigmoid applied.
        return float(np.ravel(out[0])[0])
    except Exception as e:
        logger.warning("EOU predict failed (%s); treating as no-signal", e)
        return None


if __name__ == "__main__":
    # Smoke test: confirms the model loads and produces a number. Noise in gives
    # a meaningless score out, which is fine — only the plumbing is under test.
    logging.basicConfig(level=logging.INFO)
    dummy = (np.random.randn(SR * 3).astype(np.float32) * 0.02)
    print("EOU P(complete) on dummy audio:", predict_complete(dummy))
