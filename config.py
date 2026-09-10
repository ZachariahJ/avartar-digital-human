import os
import functools
import hashlib
import threading
from dotenv import load_dotenv

from modules.sbirt import build_system_prompt, templates

load_dotenv()

SHUTTING_DOWN = threading.Event()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ASSETS_DIR = os.path.join(BASE_DIR, "assets")
TEMP_DIR = os.path.join(BASE_DIR, "tmp")
CHECKPOINTS_DIR = os.path.join(BASE_DIR, "checkpoints")
RECORDS_DIR = os.path.join(BASE_DIR, "records")
CERTS_DIR = os.path.join(BASE_DIR, "certs")

MUSETALK_DIR = os.path.join(os.path.dirname(BASE_DIR), "MuseTalk")
MUSETALK_VERSION = "v15"
_MT_MODELS = os.path.join(MUSETALK_DIR, "models")
_MT_UNET_DIR = os.path.join(_MT_MODELS, "musetalkV15" if MUSETALK_VERSION == "v15" else "musetalk")
MUSETALK_UNET = os.path.join(_MT_UNET_DIR, "unet.pth" if MUSETALK_VERSION == "v15" else "pytorch_model.bin")
MUSETALK_UNET_CONFIG = os.path.join(_MT_UNET_DIR, "musetalk.json")
MUSETALK_VAE = os.path.join(_MT_MODELS, "sd-vae")
MUSETALK_WHISPER = os.path.join(_MT_MODELS, "whisper")

MUSETALK_MATERIAL_DIR = os.path.join(CHECKPOINTS_DIR, "musetalk_avatar")

AVATAR_VIDEO = os.path.join(ASSETS_DIR, "loop.mp4")

AVATAR_IMAGE = os.path.join(ASSETS_DIR, "avatar1.png")


@functools.lru_cache(maxsize=8)
def _video_digest(path: str, size: int, mtime: float) -> str:
    h = hashlib.md5(str(size).encode())
    with open(path, "rb") as f:
        h.update(f.read(4 << 20))
    return h.hexdigest()[:16]


def avatar_fingerprint() -> str:
    try:
        st = os.stat(AVATAR_VIDEO)
        return _video_digest(AVATAR_VIDEO, st.st_size, st.st_mtime)
    except OSError:
        return "no-avatar"

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
LLM_MODEL = "google/gemini-2.5-flash"

SYSTEM_PROMPT = build_system_prompt()

TTS_VOICE = "en-US-GuyNeural"
TTS_RATE = "-10%"

ASR_MODEL = "iic/SenseVoiceSmall"

ENABLE_VIDEO_AVATAR = 1

MUSETALK_GPUS = [1]
ASR_GPU = 1

MUSETALK_FPS = 24
MUSETALK_BATCH_SIZE = 6
MUSETALK_BBOX_SHIFT = 0
MUSETALK_EXTRA_MARGIN = 10
MUSETALK_PARSING_MODE = "jaw"
MUSETALK_LEFT_CHEEK_WIDTH = 90
MUSETALK_RIGHT_CHEEK_WIDTH = 90
MUSETALK_AUDIO_PAD_LEFT = 2
MUSETALK_AUDIO_PAD_RIGHT = 2

MUSETALK_JPEG_QUALITY = 82
STREAM_PREBUFFER_FRAMES = 12

VAD_THRESHOLD = 0.65
VAD_SILENCE_DURATION = 0.35

VAD_NEAR_FIELD_MARGIN_DB = float(os.getenv("VAD_NEAR_FIELD_MARGIN_DB", "15"))
VAD_NEAR_FIELD_RELEASE_DB = float(os.getenv("VAD_NEAR_FIELD_RELEASE_DB", "6"))
VAD_MIN_LEVEL_DBFS = float(os.getenv("VAD_MIN_LEVEL_DBFS", "-55"))
VAD_NOISE_FLOOR_INIT_DBFS = float(os.getenv("VAD_NOISE_FLOOR_INIT_DBFS", "-45"))

MIC_WARMUP_DISCARD = float(os.getenv("MIC_WARMUP_DISCARD", "0.5"))

BARGE_IN_ASR = os.getenv("BARGE_IN_ASR", "1").lower() not in ("0", "false", "no")
BARGE_IN_MIN_SPEECH = 0.30
BARGE_IN_RECHECK = 0.20

BARGE_IN_VAD = os.getenv("BARGE_IN_VAD", "1").lower() not in ("0", "false", "no")
BARGE_IN_VAD_SUSTAIN = 0.4

STATE_POLL_INTERVAL = 0.1

LLM_HISTORY_MAX_MESSAGES = 20

TEMP_FILE_TTL_SEC = 180
TEMP_CLEAN_INTERVAL_SEC = 30

MEDIA_BLOB_TTL_SEC = 300
CLIP_CACHE = os.getenv("CLIP_CACHE", "0").lower() not in ("0", "false", "no")
CLIP_CACHE_MAX_MB = int(os.getenv("CLIP_CACHE_MAX_MB", "2048"))
_SHM = "/dev/shm"
RENDER_SCRATCH_DIR = os.getenv(
    "RENDER_SCRATCH_DIR",
    os.path.join(_SHM, f"digital-human-{os.getuid()}") if os.path.isdir(_SHM) else TEMP_DIR,
)

CLIP_PREWARM = os.getenv("CLIP_PREWARM", "1").lower() not in ("0", "false", "no")
CLIP_PREWARM_IDLE_SEC = float(os.getenv("CLIP_PREWARM_IDLE_SEC", "5"))
CLIP_PREWARM_POLL_SEC = 0.5
CLIP_PREWARM_MAX_ATTEMPTS = 3

LOG_PHI = os.getenv("LOG_PHI", "0").lower() in ("1", "true", "yes")

CONSENT_LOG_PATH = os.getenv(
    "CONSENT_LOG_PATH",
    os.path.join(RECORDS_DIR, "consent_log.jsonl"),
)

USE_EOU = os.getenv("USE_EOU", "1").lower() not in ("0", "false", "no")
EOU_MODEL_PATH = os.path.join(CHECKPOINTS_DIR, "smart-turn", "smart-turn-v3.2-cpu.onnx")
EOU_THRESHOLD = 0.5
EOU_CONFIRM_CONSULTS = 2
EOU_PAUSE_DURATION = 0.20
EOU_RECHECK_DURATION = 0.15
EOU_MAX_SILENCE = 2.0
EOU_ONNX_THREADS = 2

SESSION_IDLE_TTL_SEC = 600

IDLE_VIDEO_PATH = AVATAR_VIDEO

IDLE_AUDIO_PATH = os.path.join(ASSETS_DIR, "idle_silence.mp3")
IDLE_AUDIO_DURATION = 2.0


def idle_media_path() -> str:
    return IDLE_VIDEO_PATH if ENABLE_VIDEO_AVATAR else IDLE_AUDIO_PATH

GREETING_PREAMBLE = (
    "Hello, I am an AI assistant designed to help understand some important factors "
    "that may impact your health. This information will be shared with your medical "
    "provider to help your provider better understand your current health issues. "
    "Your answers will be treated as confidential and as protected health information. "
)
CONSENT_QUESTION = templates.FIXED["consent.opening"]
GREETING_TEXT = GREETING_PREAMBLE + CONSENT_QUESTION
GREETING_CLIP_KEY = "greeting"

SERVER_HOST = os.getenv("SERVER_HOST", "0.0.0.0")
SERVER_PORT = int(os.getenv("SERVER_PORT", "17861"))
WS_PORT = int(os.getenv("WS_PORT", "17862"))

SSL_CERT_FILE = os.getenv("SSL_CERT_FILE", os.path.join(CERTS_DIR, "cert.pem"))
SSL_KEY_FILE = os.getenv("SSL_KEY_FILE", os.path.join(CERTS_DIR, "key.pem"))
ENABLE_HTTPS = os.getenv("ENABLE_HTTPS", "1").lower() not in ("0", "false", "no")
