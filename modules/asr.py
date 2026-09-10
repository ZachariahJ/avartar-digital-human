import re
import threading
import numpy as np
from funasr import AutoModel
import config

_model = None
_model_lock = threading.Lock()


def get_model():
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
    return re.sub(r"<\|[^|]*\|>", "", text).strip()


def transcribe_array(audio_array: np.ndarray, sample_rate: int = 16000) -> str:
    model = get_model()
    result = model.generate(
        input=audio_array,
        fs=sample_rate,
        language="en",
        use_itn=True,
    )
    if result and len(result) > 0:
        return _clean_text(result[0]["text"])
    return ""


if __name__ == "__main__":
    print("ASR module loaded. Call transcribe_array(audio) to use.")
    model = get_model()
    print("Model loaded successfully.")
