import os
import threading
from dotenv import load_dotenv

from modules.sbirt import build_system_prompt

load_dotenv()

# Set once on server shutdown so every background worker (state poller, temp
# janitor, FLOAT render busy-wait, LLM producer) can bail out promptly. This is
# what makes Ctrl+C exit cleanly instead of hanging or throwing tracebacks.
SHUTTING_DOWN = threading.Event()

# Paths — single source of truth. Every directory root is defined ONCE here;
# everything else (this file, main.py, modules/) derives from these, never by
# reverse-engineering a root out of some file's location (e.g. dirname of a clip
# path — that silently breaks the moment the clip moves). Move a root → one edit.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))   # digital-human/
ASSETS_DIR = os.path.join(BASE_DIR, "assets")           # source imgs + all cached clips live under here
CLIPS_DIR = os.path.join(ASSETS_DIR, "clips")           # ALL pre-rendered fixed video clips
TEMP_DIR = os.path.join(BASE_DIR, "tmp")                # per-sentence dynamic renders (transient)
CHECKPOINTS_DIR = os.path.join(BASE_DIR, "checkpoints")
RECORDS_DIR = os.path.join(BASE_DIR, "records")
CERTS_DIR = os.path.join(BASE_DIR, "certs")

FLOAT_DIR = os.path.join(os.path.dirname(BASE_DIR), "float")   # sibling repo, off-limits to edit
FLOAT_CKPT = os.path.join(FLOAT_DIR, "checkpoints", "float.pth")
AVATAR_IMAGE = os.path.join(ASSETS_DIR, "avatar.png")

# LLM
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
LLM_MODEL = "google/gemini-2.5-flash"

# The SBIRT counselor's system prompt is BUILT from the structured clinical
# framework in modules/sbirt/ (instruments, brief-intervention techniques,
# referral pathways, state machine) rather than hand-written here. This
# guarantees the Q&A always carries the complete SBIRT content, and the clinical
# material stays maintainable in one place. To edit what the counselor knows,
# change the data modules under modules/sbirt/ — not this string.
SYSTEM_PROMPT = build_system_prompt()

# Sentence splitting for synthesis breaks ONLY on sentence-final punctuation, so
# each spoken clip is a whole sentence (comma-level splitting was removed — it made
# the avatar choppy, with a silence gap and lip-sync seam at every comma, and it
# multiplied FLOAT renders). The system prompt's "open with a 4–8 word
# acknowledgment as its own sentence" rule keeps the first clip short for a fast
# start without needing sub-sentence splits.

# TTS
TTS_VOICE = "en-US-GuyNeural"

# ASR
ASR_MODEL = "iic/SenseVoiceSmall"

# GPU allocation
FLOAT_GPUS = [3]    # GPUs FLOAT renders on (one worker per listed GPU id)
ASR_GPU = 3         # GPU for ASR (currently shared with FLOAT)

# FLOAT
FLOAT_NFE = 10            # Number of function evaluations (lower = faster, 7-10)
FLOAT_NFE_FIRST = 6       # Lower NFE for the FIRST sentence only — buys ~30-40% off
                          # first-segment render to get the avatar talking sooner.
                          # Set == FLOAT_NFE to disable the first-sentence speedup.

# VAD
VAD_THRESHOLD = 0.5
VAD_SILENCE_DURATION = 0.35  # seconds of silence to trigger speech_end (lower = snappier)

# --- Barge-in during avatar playback (ASR-confirmed / semantic) ---
# While the avatar is speaking the VAD onset (speech_start) is unreliable — its own
# audio can keep the VAD "in speech", and the onset heuristic misses — so barge-in
# often doesn't fire until the clip finishes. Instead, while the avatar plays,
# transcribe the incoming mic audio and interrupt the MOMENT it turns into real words
# (a detected sentence), rejecting the avatar's own echo. Set BARGE_IN_ASR=0 to
# fall back to VAD-onset-only barge-in.
BARGE_IN_ASR = os.getenv("BARGE_IN_ASR", "1").lower() not in ("0", "false", "no")
BARGE_IN_MIN_SPEECH = 0.30   # seconds of user speech before the first ASR check
BARGE_IN_RECHECK = 0.20      # re-run ASR every this many more seconds of speech until it fires

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

# Consent audit trail (append-only JSONL): decision + timestamp + exact-wording
# version per session. No transcripts, no screening data. records/ is
# gitignored; the directory is created on first write.
CONSENT_LOG_PATH = os.getenv(
    "CONSENT_LOG_PATH",
    os.path.join(RECORDS_DIR, "consent_log.jsonl"),
)

# --- EOU: semantic end-of-utterance / turn detection (smart-turn v3) ---
# When enabled, a VAD-detected pause is only treated as the end of the user's turn
# if the smart-turn model agrees the utterance is semantically complete. This stops
# the avatar from cutting people off when they pause to think, while still ending
# promptly on a finished thought. Tiny ONNX model on CPU (~28ms); isolated from the
# FLOAT GPUs. If the model can't load, VAD transparently falls back to pure silence.
USE_EOU = os.getenv("USE_EOU", "1").lower() not in ("0", "false", "no")
EOU_MODEL_PATH = os.path.join(CHECKPOINTS_DIR, "smart-turn", "smart-turn-v3.2-cpu.onnx")
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

# Idle video (ambient loop; lives at the assets root, not a spoken clip)
IDLE_VIDEO_PATH = os.path.join(ASSETS_DIR, "idle_loop.mp4")
IDLE_VIDEO_DURATION = 10.0  # seconds; longer = fewer loop seams

# Fixed opening the counselor always says first. Because it is IDENTICAL every
# session, it is rendered to a cached clip ONCE (assets/greeting.mp4) and replayed
# — no per-session LLM/TTS/FLOAT, and it appears instantly. Editing GREETING_TEXT
# regenerates the clip on next startup (a sidecar tracks the text). The spoken part
# only; the yes/no consent branch is handled by the LLM (see modules/sbirt/workflow.py).
GREETING_TEXT = (
    "Hello, I am an AI assistant designed to help understand some important factors "
    "that may impact your health. This information will be shared with your medical "
    "provider to help your provider better understand your current health issues. "
    "Your answers will be treated as confidential and as protected health information. "
    "May I ask you some questions about your health?"
)
GREETING_VIDEO_PATH = os.path.join(CLIPS_DIR, "greeting.mp4")

# Fixed reply when the user DECLINES consent at the greeting. Cached to a clip too,
# so it is verbatim + instant like the greeting (no per-session LLM/TTS/FLOAT).
DECLINE_TEXT = "Thank you, and your provider will address these during your visit."
DECLINE_VIDEO_PATH = os.path.join(CLIPS_DIR, "decline.mp4")

# Server / public access
SERVER_HOST = os.getenv("SERVER_HOST", "0.0.0.0")  # bind all interfaces for public access
SERVER_PORT = int(os.getenv("SERVER_PORT", "17861"))
WS_PORT = int(os.getenv("WS_PORT", "17862"))

# HTTPS — required for browser microphone (getUserMedia) on non-localhost origins.
SSL_CERT_FILE = os.getenv("SSL_CERT_FILE", os.path.join(CERTS_DIR, "cert.pem"))
SSL_KEY_FILE = os.getenv("SSL_KEY_FILE", os.path.join(CERTS_DIR, "key.pem"))
# Enabled by default; set ENABLE_HTTPS=0 to force plain HTTP (mic then only works on localhost).
ENABLE_HTTPS = os.getenv("ENABLE_HTTPS", "1").lower() not in ("0", "false", "no")
