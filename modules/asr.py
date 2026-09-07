import re
import threading
import numpy as np
from funasr import AutoModel
import config

_model = None
_model_lock = threading.Lock()


def get_model():
    """The ASR model, loaded on first use and shared afterwards.

    Double-checked locking, because the startup pre-warm and a first user
    request race here and loading twice would put two copies on the GPU.
    """
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                _model = AutoModel(
                    model=config.ASR_MODEL,
                    device=f"cuda:{config.ASR_GPU}",
                    trust_remote_code=True,
                )
    return _model


def _clean_text(text: str) -> str:
    """Strip SenseVoice's inline markup, e.g. <|en|><|EMO_UNKNOWN|><|Speech|>.

    These tags carry language and emotion labels this pipeline does not use, and
    they would otherwise reach the LLM and the on-screen transcript verbatim.
    """
    return re.sub(r"<\|[^|]*\|>", "", text).strip()


def transcribe_array(audio_array: np.ndarray, sample_rate: int = 16000) -> str:
    """Transcribe one utterance. Returns "" when the model recognised nothing."""
    model = get_model()
    result = model.generate(input=audio_array, fs=sample_rate, language="en")
    if result and len(result) > 0:
        return _clean_text(result[0]["text"])
    return ""


if __name__ == "__main__":
    # Smoke test: check the model loads at all, without a server or a GPU pool.
    print("ASR module loaded. Call transcribe_array(audio) to use.")
    model = get_model()
    print("Model loaded successfully.")
