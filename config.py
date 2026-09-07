import os
import functools
import hashlib
import threading
from dotenv import load_dotenv

from modules.sbirt import build_system_prompt

load_dotenv()

# Set once on server shutdown so every background worker (state poller, temp
# janitor, MuseTalk render busy-wait, LLM producer) can bail out promptly. This is
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

MUSETALK_DIR = os.path.join(os.path.dirname(BASE_DIR), "MuseTalk")   # sibling repo
MUSETALK_VERSION = "v15"                        # "v15" or "v1" (picks the UNet below)
_MT_MODELS = os.path.join(MUSETALK_DIR, "models")
_MT_UNET_DIR = os.path.join(_MT_MODELS, "musetalkV15" if MUSETALK_VERSION == "v15" else "musetalk")
MUSETALK_UNET = os.path.join(_MT_UNET_DIR, "unet.pth" if MUSETALK_VERSION == "v15" else "pytorch_model.bin")
MUSETALK_UNET_CONFIG = os.path.join(_MT_UNET_DIR, "musetalk.json")
MUSETALK_VAE = os.path.join(_MT_MODELS, "sd-vae")
MUSETALK_WHISPER = os.path.join(_MT_MODELS, "whisper")

# Where the prepared driving material (per-frame crops, VAE latents, blend masks)
# is cached. Preparing it is a one-off multi-minute pass over AVATAR_VIDEO; it is
# keyed by avatar_fingerprint() so swapping the driving video prepares a fresh
# set instead of silently reusing the old face's crops.
MUSETALK_MATERIAL_DIR = os.path.join(CHECKPOINTS_DIR, "musetalk_avatar")
# The driving video the digital human is rendered from. MuseTalk is an INPAINTING
# model: it only repaints the mouth region, every other pixel and ALL head motion
# comes from this clip, so it must be a real clip and not a still (a still gives a
# frozen head with a moving mouth). Requirements: single face, front-facing, never
# leaves frame, 25 fps (MUSETALK_FPS), and first/last frame close together — it
# doubles as the source of the idle loop.
#
# MUST be a synthetic (AI-generated) face that does not depict, and is not
# recognisably modelled on, an identifiable real person — rendering a real
# person's likeness into a clinician persona is a portrait-rights (肖像权)
# exposure. The former avatar.png / avatar2.png / avatar3.png were
# celebrity-likeness renders and have been deleted for exactly that reason — do
# not restore them from git history and point this at them.
AVATAR_VIDEO = os.path.join(ASSETS_DIR, "loop.mp4")

# Still portrait shown in the frontend's side panel (main.py's portraitUrl). It is
# NOT what gets rendered — that is AVATAR_VIDEO — but it should be a frame of the
# same person, or the panel shows one face and the video another.
AVATAR_IMAGE = os.path.join(ASSETS_DIR, "avatar1.png")


@functools.lru_cache(maxsize=8)
def _video_digest(path: str, size: int, mtime: float) -> str:
    """md5 of a video's first 4 MB, keyed (via the args) on size+mtime so an
    edited file re-digests. Hashing only the head keeps this cheap: unlike the
    old still portrait, AVATAR_VIDEO is tens of MB and clip_stamp() calls the
    fingerprint once per cached clip at every startup.
    """
    h = hashlib.md5(str(size).encode())
    with open(path, "rb") as f:
        h.update(f.read(4 << 20))
    return h.hexdigest()[:16]


def avatar_fingerprint() -> str:
    """Content digest of AVATAR_VIDEO — the cache key for everything MuseTalk
    renders.

    Every cached artifact (the fixed protocol clips, greeting/decline/crisis
    clips, the idle loop, and the prepared driving material under
    MUSETALK_MATERIAL_DIR) is a function of BOTH the spoken text AND this
    driving video. The caches therefore key on this digest as well as the text,
    so swapping the video invalidates them and they re-render on next startup.
    Keying on text alone was a face-swap trap: changing the driving clip left
    every cached clip silently replaying the OLD face.
    """
    try:
        st = os.stat(AVATAR_VIDEO)
        return _video_digest(AVATAR_VIDEO, st.st_size, st.st_mtime)
    except OSError:
        return "no-avatar"

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
# multiplied MuseTalk renders). The system prompt's "open with a 4–8 word
# acknowledgment as its own sentence" rule keeps the first clip short for a fast
# start without needing sub-sentence splits.

# TTS
TTS_VOICE = "en-US-GuyNeural"

# ASR
ASR_MODEL = "iic/SenseVoiceSmall"

# Render the talking-head video at all. Set ENABLE_VIDEO_AVATAR=0 to run the
# counselor as a VOICE-ONLY assistant: TTS still speaks every line, but MuseTalk is
# never loaded and never renders, so the whole clip pipeline collapses to "the
# TTS file IS the clip". Everything else — ASR, VAD, EOU, barge-in, echo
# suppression, the clinical protocol — is text-driven and behaves identically.
# The browser plays the audio through the same <video> elements (an audio-only
# source drives canplay/ended/duration exactly like an mp4), so the frontend's
# double-buffered playback machinery is shared by both modes.
# Consequences of turning this off: MUSETALK_GPUS goes unused (ASR_GPU is still
# needed), startup no longer pre-renders anything on the GPU, and cached clips
# live under a separate cache key (see pipeline.clip_stamp) so the two modes
# never serve each other's files.
ENABLE_VIDEO_AVATAR = 1

# GPU allocation
MUSETALK_GPUS = [0]  # GPUs MuseTalk renders on (one worker per listed GPU id)
ASR_GPU = 0          # GPU for ASR (currently shared with MuseTalk)

# MuseTalk render knobs. There is no quality/speed dial like FLOAT's NFE — MuseTalk
# is a single-step inpainting UNet, so a clip costs one pass per frame regardless.
# Playback frame rate, and the rate whisper features are sampled at. It does NOT
# have to match AVATAR_VIDEO's own frame rate, and here it deliberately does not
# (the clip is 24 fps): lip sync is preserved either way, because output frame i
# is driven by the audio at i/MUSETALK_FPS and the browser draws frame
# floor(currentTime * MUSETALK_FPS) — both sides use THIS number, not the clip's.
#
# What the mismatch does change is head motion. One driving frame is consumed
# per output frame, so at 15 fps the clip's own motion plays back at 15/24 =
# 0.63x. The idle loop is the same file played by a <video> element at its
# native 24 fps, so the head visibly slows down when the avatar starts speaking.
# Re-encoding assets/loop.mp4 to 15 fps removes that seam (and re-keys
# avatar_fingerprint(), forcing a one-off material rebuild).
#
# 15 rather than 24 because the renderer is the bottleneck: measured throughput
# on a V100 at MUSETALK_BATCH_SIZE=4 is ~17.5 frames/s, which is 0.73x a 24 fps
# clock (the video falls behind the audio for as long as the utterance runs) but
# ~1.17x a 15 fps one. Streaming playback only works while this is above 1.0x.
MUSETALK_FPS = 15
MUSETALK_BATCH_SIZE = 6      # frames per UNet batch AND the streaming flush unit.
# Every batch is blended, JPEG-encoded and pushed to the browser the moment the
# VAE decoder returns it, so this is the first-frame latency dial: 4 frames at
# MUSETALK_FPS=24 is one sixth of a second of video per flush.
MUSETALK_BBOX_SHIFT = 0      # v1 only; v15 ignores it (upstream forces 0)
MUSETALK_EXTRA_MARGIN = 10   # v15: extra pixels below the face box, chin coverage
MUSETALK_PARSING_MODE = "jaw"      # v15 blend mask mode ("jaw" or "raw")
MUSETALK_LEFT_CHEEK_WIDTH = 90     # face-parsing cheek protection, in px
MUSETALK_RIGHT_CHEEK_WIDTH = 90
MUSETALK_AUDIO_PAD_LEFT = 2        # whisper context frames before each video frame
MUSETALK_AUDIO_PAD_RIGHT = 2       # ... and after

# Frame streaming (the browser composites JPEG frames on a canvas against one
# continuous <audio> element, instead of fetching a muxed mp4 per sentence).
MUSETALK_JPEG_QUALITY = 82   # per-frame JPEG quality pushed over the WebSocket
# Frames the client buffers before it starts the audio clock. The audio, once
# started, cannot be paused without an audible gap, so playback must not begin
# until the renderer is far enough ahead to stay ahead — and at MUSETALK_FPS=15
# it does stay ahead (see the throughput note above), so a fixed lead is enough:
# the gap grows in the renderer's favour, not the clock's.
# 12 frames is 0.8s at 15 fps.
STREAM_PREBUFFER_FRAMES = 12

# VAD
VAD_THRESHOLD = 0.5
VAD_SILENCE_DURATION = 0.35  # seconds of silence to trigger speech_end (lower = snappier)

# Discard the first N seconds of every mic stream before the VAD ever sees them.
# Opening the mic emits a start-up transient (device pop + AGC ramping down from
# max gain); measured on this box it runs ~0.5s and clips full scale (chunk #3 hit
# max=32719 with the room silent). Silero scores it as speech, so the avatar
# answers an utterance the user never spoke. 0 disables the discard.
MIC_WARMUP_DISCARD = float(os.getenv("MIC_WARMUP_DISCARD", "0.5"))

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

# Instant (full-duplex) barge-in. The ASR-confirmed path above is accurate but
# costs BARGE_IN_MIN_SPEECH + a transcription before it fires, which is audible
# as the avatar talking over the user. This fires the cascade flush on SUSTAINED
# VAD voice alone, with no transcription in the loop.
#
# It is safe here only because getUserMedia is opened with
# echoCancellation:{exact:true} (static/index.html) — without AEC the avatar's
# own leaked voice is "sustained voice" and it would interrupt itself in a loop.
# Set BARGE_IN_VAD=0 to fall back to ASR-confirmed interruption only.
BARGE_IN_VAD = os.getenv("BARGE_IN_VAD", "1").lower() not in ("0", "false", "no")
# Seconds of CONTINUOUS detected voice before the flush fires. Below ~0.12s the
# trigger starts catching lip smacks and chair creaks that survive AEC; above
# ~0.25s the user hears themselves talking over the avatar.
BARGE_IN_VAD_SUSTAIN = 0.18

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

# Write patient/clinical content to the logs VERBATIM instead of the redacted
# "<phi 7w/41c>" shape summary. Off by default. For local debugging only: with
# this on, transcripts land in run.log in the clear.
LOG_PHI = os.getenv("LOG_PHI", "0").lower() in ("1", "true", "yes")

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
# MuseTalk GPUs. If the model can't load, VAD transparently falls back to pure silence.
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

# Idle video (ambient loop). It IS the driving clip: loop.mp4 is authored to
# loop seamlessly, so the frontend can play it directly between answers and
# nothing has to be generated, cached or invalidated.
IDLE_VIDEO_PATH = AVATAR_VIDEO

# Voice-only counterpart of the idle loop. The frontend's playback state machine
# pivots on an "idle" item it can loop between answers; with no video there is
# nothing to show, so it loops this silent clip instead and the swap/drain logic
# stays byte-identical across both modes. Generated once by ffmpeg at startup
# (no GPU) — see main._ensure_idle_media.
IDLE_AUDIO_PATH = os.path.join(ASSETS_DIR, "idle_silence.mp3")
IDLE_AUDIO_DURATION = 2.0


def idle_media_path() -> str:
    """The ambient clip the frontend loops when nobody is speaking."""
    return IDLE_VIDEO_PATH if ENABLE_VIDEO_AVATAR else IDLE_AUDIO_PATH

# Fixed opening the counselor always says first. Because it is IDENTICAL every
# session, it is rendered to a cached clip ONCE (assets/greeting.mp4) and replayed
# — no per-session LLM/TTS/MuseTalk, and it appears instantly. Editing GREETING_TEXT
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
# so it is verbatim + instant like the greeting (no per-session LLM/TTS/MuseTalk).
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
