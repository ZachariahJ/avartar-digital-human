import os
from dotenv import load_dotenv

from modules.sbirt import build_system_prompt

load_dotenv()

# Paths
FLOAT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "float")
FLOAT_CKPT = os.path.join(FLOAT_DIR, "checkpoints", "float.pth")
AVATAR_IMAGE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "avatar.png")
TEMP_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tmp")

# LLM
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
LLM_MODEL = "google/gemini-2.5-flash"

# Synthesis mode (how an LLM answer becomes speech + avatar video):
#   "stream_parallel" = stream sentence-by-sentence (text appears live in <1s) AND
#              render TTS+FLOAT for each sentence concurrently across the whole GPU
#              pool, enqueuing strictly in order. First-segment latency = 1 TTS +
#              1 FLOAT; sentences 2..N render on the other GPUs while sentence 1
#              plays, so playback doesn't stall. This is the fastest+smoothest mode.
#   "hybrid" = get the FULL answer first, show the complete text at once (stable,
#              never live-updates), then render per-sentence so the avatar starts
#              speaking in ~5s. Stable display, but text appears only after the
#              whole answer is generated.
#   "batch"  = full answer, then a SINGLE TTS + FLOAT video (one continuous clip,
#              most stable, but ~20-35s wait before it starts speaking).
#   "stream" = stream sentence-by-sentence, but FLOAT renders serially on ONE GPU
#              (legacy; sentences 2..N stall, wasting the other GPUs).
SYNTHESIS_MODE = "stream_parallel"
# The SBIRT counselor's system prompt is BUILT from the structured clinical
# framework in modules/sbirt/ (instruments, brief-intervention techniques,
# referral pathways, state machine) rather than hand-written here. This
# guarantees the Q&A always carries the complete SBIRT content, and the clinical
# material stays maintainable in one place. To edit what the counselor knows,
# change the data modules under modules/sbirt/ — not this string.
SYSTEM_PROMPT = build_system_prompt()

# Sentence splitting for synthesis: also break at commas / semicolons (not only
# sentence-final punctuation) so the FIRST chunk is shorter and the avatar starts
# talking sooner. A soft (comma) break only fires once the pending chunk is at
# least MIN_CHUNK_CHARS long, to avoid tiny choppy fragments ("Well,"). Set
# SPLIT_ON_COMMA=False to revert to sentence-only; MIN_CHUNK_CHARS=1 = every comma.
SPLIT_ON_COMMA = True
MIN_CHUNK_CHARS = 5

# TTS
TTS_VOICE = "en-US-GuyNeural"

# ASR
ASR_MODEL = "iic/SenseVoiceSmall"

# GPU allocation
FLOAT_GPUS = [0, 1, 2]   # 3 GPUs for FLOAT parallel rendering
ASR_GPU = 3               # Dedicated GPU for ASR

# FLOAT
FLOAT_NFE = 10            # Number of function evaluations (lower = faster, 7-10)
FLOAT_NFE_FIRST = 6       # Lower NFE for the FIRST sentence only — buys ~30-40% off
                          # first-segment render to get the avatar talking sooner.
                          # Set == FLOAT_NFE to disable the first-sentence speedup.

# VAD
VAD_THRESHOLD = 0.5
VAD_SILENCE_DURATION = 0.35  # seconds of silence to trigger speech_end (lower = snappier)

# --- Performance / latency tuning ---
# How often the server polls the pipeline for finished video segments and pushes
# them to the browser. Lower = video is delivered more promptly after it renders.
STATE_POLL_INTERVAL = 0.1   # seconds

# Sliding window on the LLM conversation history sent to the API. Keeps long
# sessions from ballooning the prompt (which slows first-token latency and costs).
# Counts messages (user+assistant); the system prompt is always kept on top.
LLM_HISTORY_MAX_MESSAGES = 20

# Temp-file janitor: tmp/ fills with per-sentence .wav/.mp4 clips. A background
# thread deletes clips older than the TTL so long sessions don't exhaust disk.
TEMP_FILE_TTL_SEC = 180
TEMP_CLEAN_INTERVAL_SEC = 30

# Save a debug .wav of every captured utterance to tmp/debug_speech.wav. Off by
# default — the synchronous disk write was stalling the audio event loop.
DEBUG_SAVE_AUDIO = os.getenv("DEBUG_SAVE_AUDIO", "0").lower() in ("1", "true", "yes")

# --- EOU: semantic end-of-utterance / turn detection (smart-turn v3) ---
# When enabled, a VAD-detected pause is only treated as the end of the user's turn
# if the smart-turn model agrees the utterance is semantically complete. This stops
# the avatar from cutting people off when they pause to think, while still ending
# promptly on a finished thought. Tiny ONNX model on CPU (~28ms); isolated from the
# FLOAT GPUs. If the model can't load, VAD transparently falls back to pure silence.
USE_EOU = os.getenv("USE_EOU", "1").lower() not in ("0", "false", "no")
EOU_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "checkpoints", "smart-turn", "smart-turn-v3.2-cpu.onnx",
)
EOU_THRESHOLD = 0.5          # P(complete) >= this -> end the turn (↑ more patient, ↓ snappier)
EOU_CONFIRM_CONSULTS = 2     # require this many CONSECUTIVE 'complete' verdicts before
                             # ending — hysteresis against a momentary clause-boundary
                             # spike cutting the user off mid-sentence. 1 = no hysteresis.
EOU_PAUSE_DURATION = 0.20    # silence before the FIRST EOU consult (keep short)
EOU_RECHECK_DURATION = 0.15  # re-consult cadence while silence continues
EOU_MAX_SILENCE = 2.0        # hard cap: force end after this much silence regardless of EOU
EOU_ONNX_THREADS = 2         # CPU threads for the ONNX session

# Reap a per-user session this long after its last client disconnects (idle only).
SESSION_IDLE_TTL_SEC = 600

# Idle video
IDLE_VIDEO_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "idle_loop.mp4")
IDLE_VIDEO_DURATION = 10.0  # seconds; longer = fewer loop seams

# Server / public access
SERVER_HOST = os.getenv("SERVER_HOST", "0.0.0.0")  # bind all interfaces for public access
SERVER_PORT = int(os.getenv("SERVER_PORT", "17861"))
WS_PORT = int(os.getenv("WS_PORT", "17862"))

# HTTPS — required for browser microphone (getUserMedia) on non-localhost origins.
CERTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "certs")
SSL_CERT_FILE = os.getenv("SSL_CERT_FILE", os.path.join(CERTS_DIR, "cert.pem"))
SSL_KEY_FILE = os.getenv("SSL_KEY_FILE", os.path.join(CERTS_DIR, "key.pem"))
# Enabled by default; set ENABLE_HTTPS=0 to force plain HTTP (mic then only works on localhost).
ENABLE_HTTPS = os.getenv("ENABLE_HTTPS", "1").lower() not in ("0", "false", "no")
