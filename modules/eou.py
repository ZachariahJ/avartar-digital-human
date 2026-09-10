
import logging
import threading

import numpy as np

import config

logger = logging.getLogger(__name__)

SR = 16000
WINDOW_SEC = 8

_session = None
_feature_extractor = None
_input_name = None
_load_failed = False
_lock = threading.Lock()


def _load():
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
    sess = get_model()
    if sess is None:
        return None
    try:
        a = np.asarray(audio, dtype=np.float32)
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
        return float(np.ravel(out[0])[0])
    except Exception as e:
        logger.warning("EOU predict failed (%s); treating as no-signal", e)
        return None


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    dummy = (np.random.randn(SR * 3).astype(np.float32) * 0.02)
    print("EOU P(complete) on dummy audio:", predict_complete(dummy))
