import os
import functools
import hashlib
import threading
from dotenv import load_dotenv

from modules.sbirt import build_system_prompt, templates

load_dotenv()

# Cooperative shutdown flag. Background workers poll it instead of being killed
# mid-work, which is what makes Ctrl+C exit without hung threads or tracebacks.
SHUTTING_DOWN = threading.Event()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ASSETS_DIR = os.path.join(BASE_DIR, "assets")
TEMP_DIR = os.path.join(BASE_DIR, "tmp")
CHECKPOINTS_DIR = os.path.join(BASE_DIR, "checkpoints")
RECORDS_DIR = os.path.join(BASE_DIR, "records")
CERTS_DIR = os.path.join(BASE_DIR, "certs")

# MuseTalk is a sibling checkout, not a dependency: modules/avatar.py puts this
# on sys.path and imports from it. Its models/ layout is upstream's, not ours.
MUSETALK_DIR = os.path.join(os.path.dirname(BASE_DIR), "MuseTalk")
MUSETALK_VERSION = "v15"
_MT_MODELS = os.path.join(MUSETALK_DIR, "models")
_MT_UNET_DIR = os.path.join(_MT_MODELS, "musetalkV15" if MUSETALK_VERSION == "v15" else "musetalk")
MUSETALK_UNET = os.path.join(_MT_UNET_DIR, "unet.pth" if MUSETALK_VERSION == "v15" else "pytorch_model.bin")
MUSETALK_UNET_CONFIG = os.path.join(_MT_UNET_DIR, "musetalk.json")
MUSETALK_VAE = os.path.join(_MT_MODELS, "sd-vae")
MUSETALK_WHISPER = os.path.join(_MT_MODELS, "whisper")

# Face crops, VAE latents and blend masks precomputed from AVATAR_VIDEO. Costs
# minutes to build, so it is written once per avatar_fingerprint() and reused.
MUSETALK_MATERIAL_DIR = os.path.join(CHECKPOINTS_DIR, "musetalk_avatar")

# MuseTalk inpaints the mouth region of an existing clip; head motion and every
# pixel outside the mouth come from this file. A still image therefore yields a
# frozen head with a moving mouth. Needs one front-facing face that stays in
# frame, at MUSETALK_FPS, with matching first and last frames so it also serves
# as the seamless idle loop.
#
# The face must be synthetic and not resemble an identifiable real person:
# putting a real likeness in a clinician role is a portrait-rights exposure.
# Earlier celebrity-lookalike renders were deleted for this reason.
AVATAR_VIDEO = os.path.join(ASSETS_DIR, "loop.mp4")

# Still shown in the page's side panel. Purely decorative — nothing renders from
# it — but it should be a frame of AVATAR_VIDEO, or panel and video disagree.
AVATAR_IMAGE = os.path.join(ASSETS_DIR, "avatar1.png")


@functools.lru_cache(maxsize=8)
def _video_digest(path: str, size: int, mtime: float) -> str:
    """Hash a video cheaply enough to call on every cached clip at startup.

    `size` and `mtime` are unused in the body: they are parameters so that the
    lru_cache key changes when the file is edited. Only the first 4 MB is read,
    which is enough to separate two clips but bounded for a large file.
    """
    h = hashlib.md5(str(size).encode())
    with open(path, "rb") as f:
        h.update(f.read(4 << 20))
    return h.hexdigest()[:16]


def avatar_fingerprint() -> str:
    """Identity of the current driving video, used as a cache key.

    Rendered artifacts depend on the video as much as on the words, so caches
    that key on text alone keep replaying the previous face after the video is
    swapped. Returns "no-avatar" when the file is missing, which simply makes
    every lookup miss rather than raising during startup.
    """
    try:
        st = os.stat(AVATAR_VIDEO)
        return _video_digest(AVATAR_VIDEO, st.st_size, st.st_mtime)
    except OSError:
        return "no-avatar"

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
LLM_MODEL = "google/gemini-2.5-flash"

# Assembled from the clinical data in modules/sbirt/ (instruments, techniques,
# referral pathways, state machine). Change the counselor's knowledge there, not
# by editing a prompt string.
SYSTEM_PROMPT = build_system_prompt()

# TTS
TTS_VOICE = "en-US-GuyNeural"
# Speaking rate, as edge-tts wants it: a percentage offset from the voice's
# natural pace. "-20%" is 0.8x. Slower speech only lengthens the audio; lip sync
# follows it, because MuseTalk drives frames off these exact samples.
TTS_RATE = "-10%"

ASR_MODEL = "iic/SenseVoiceSmall"

# 0 runs the counselor voice-only: MuseTalk is never imported (no GPU memory, no
# checkpoints, no material preparation) and an utterance's audio becomes the
# whole clip. Everything text-driven behaves the same. The two modes use
# separate clip cache keys, so neither ever serves the other's cached output.
ENABLE_VIDEO_AVATAR = 1

MUSETALK_GPUS = [1]
ASR_GPU = 1

# Must match AVATAR_VIDEO's own frame rate. Lip sync survives any value — audio
# and browser both index by this number — but one driving frame is consumed per
# output frame, so a mismatch replays the clip's head motion at the wrong speed,
# and the idle loop (the same file, played natively by a <video> element) then
# visibly changes speed when the avatar starts talking. Changing this means
# re-encoding loop.mp4 too, which re-keys avatar_fingerprint().
MUSETALK_FPS = 24
# UNet batch size, and also the streaming granularity: a batch is blended,
# encoded and pushed as soon as the VAE decoder returns it, so this trades
# first-frame latency against throughput.
MUSETALK_BATCH_SIZE = 6
MUSETALK_BBOX_SHIFT = 0      # v1 only; v15 forces 0 upstream
MUSETALK_EXTRA_MARGIN = 10   # v15: pixels added below the face box for the chin
MUSETALK_PARSING_MODE = "jaw"      # v15 blend mask: "jaw" or "raw"
MUSETALK_LEFT_CHEEK_WIDTH = 90     # px of cheek the face parser must not repaint
MUSETALK_RIGHT_CHEEK_WIDTH = 90
MUSETALK_AUDIO_PAD_LEFT = 2        # whisper context frames each side of the
MUSETALK_AUDIO_PAD_RIGHT = 2       # frame being generated

MUSETALK_JPEG_QUALITY = 82
# Frames the client holds before starting the audio clock. Audio cannot pause
# without an audible gap, so playback must not begin until the renderer has a
# lead. Whether that lead grows or shrinks over an utterance depends on how the
# deployment's GPU compares with MUSETALK_FPS; measure before changing this.
STREAM_PREBUFFER_FRAMES = 12

VAD_THRESHOLD = 0.65
VAD_SILENCE_DURATION = 0.35  # silence before speech_end; lower is snappier

# Near-field gate. Silero answers "is this speech", which is not the question
# the turn-taker needs answered: in a library or an open office the people at
# the next table are speech, and every interrupt path downstream is built on
# Silero's verdict. Loudness is the dimension that separates them — a speaker
# 2m away arrives 15-25dB below one at the microphone — so a chunk counts as
# speech only if it is both voice-like AND near-field. Gating here rather than
# per interrupt path is deliberate: barge-in, speech_start and speech_end all
# read the same buffer, so one gate fixes all of them.
#
# Near-field is judged against the room, not against a fixed number: the
# detector tracks the ambient level continuously and requires speech to stand
# this far above it, so a quiet library and a noisy cafe both work without
# retuning. 0 disables the gate entirely.
VAD_NEAR_FIELD_MARGIN_DB = float(os.getenv("VAD_NEAR_FIELD_MARGIN_DB", "15"))
# Once someone is speaking the bar drops to this, because unvoiced consonants
# and the tail of a sentence fall well below its loudest syllable. Without the
# hysteresis the gate chops a single utterance into fragments and each gap
# counts toward speech_end.
VAD_NEAR_FIELD_RELEASE_DB = float(os.getenv("VAD_NEAR_FIELD_RELEASE_DB", "6"))
# Backstop under the adaptive bar, for a room quiet enough that the ambient
# estimate bottoms out: nothing this faint is someone addressing the microphone,
# whatever the margin allows. Absolute dBFS, so it only means anything with
# browser AGC off (see the getUserMedia constraints in static/index.html).
VAD_MIN_LEVEL_DBFS = float(os.getenv("VAD_MIN_LEVEL_DBFS", "-55"))
# Where the ambient estimate starts, before any audio has been heard. Set high
# rather than low: it converges downward within a few hundred ms, and starting
# high means the first moments of a session cannot be interrupted by the room,
# whereas starting low would let everything through until it caught up.
VAD_NOISE_FLOOR_INIT_DBFS = float(os.getenv("VAD_NOISE_FLOOR_INIT_DBFS", "-45"))

# Seconds of microphone audio dropped before the VAD sees anything. Opening a
# mic emits a start-up transient (device pop plus AGC winding down from full
# gain) that clips full scale and reads as speech, producing an utterance nobody
# spoke. Measured at ~0.5s on this hardware. 0 disables.
MIC_WARMUP_DISCARD = float(os.getenv("MIC_WARMUP_DISCARD", "0.5"))

# Barge-in path 1: transcribe incoming audio while the avatar talks and
# interrupt once it forms real words, discarding matches against what the avatar
# is currently saying. Accurate, but costs a transcription before it fires.
# VAD onset alone is unreliable here, because the avatar's own audio can hold
# the VAD in-speech and mask the user's onset.
BARGE_IN_ASR = os.getenv("BARGE_IN_ASR", "1").lower() not in ("0", "false", "no")
BARGE_IN_MIN_SPEECH = 0.30   # speech accumulated before the first ASR check
BARGE_IN_RECHECK = 0.20      # additional speech between subsequent checks

# Barge-in path 2: interrupt on sustained VAD voice with no transcription, so
# the cut is immediate. Checked before the ASR path.
#
# Only safe because getUserMedia is opened with echoCancellation:{exact:true}
# (static/index.html). Without echo cancellation the avatar's leaked voice is
# itself "sustained voice" and it interrupts itself in a loop; set this to 0 if
# a deployment shows that symptom.
BARGE_IN_VAD = os.getenv("BARGE_IN_VAD", "1").lower() not in ("0", "false", "no")
# Continuous voice required before firing. Below ~0.12s this catches lip smacks
# and chair creaks that survive echo cancellation; above ~0.25s the user hears
# themselves talking over the avatar.
BARGE_IN_VAD_SUSTAIN = 0.4

STATE_POLL_INTERVAL = 0.1   # how often finished video is pushed to the browser

# Messages of history sent to the LLM. The system prompt is always kept; this
# only bounds the conversational tail, which otherwise slows first-token latency
# as a session grows.
LLM_HISTORY_MAX_MESSAGES = 20

TEMP_FILE_TTL_SEC = 180
TEMP_CLEAN_INTERVAL_SEC = 30

# How long a dynamic sentence's audio stays fetchable after it is published. It
# only has to outlive one browser fetch. Fixed clips pin their audio instead and
# ignore this.
MEDIA_BLOB_TTL_SEC = 300
# Off while the counselor words its own replies: a cached clip can only replay
# a line decided before the turn, so caching fights per-person phrasing.
CLIP_CACHE = os.getenv("CLIP_CACHE", "0").lower() not in ("0", "false", "no")
# Ceiling on the in-RAM fixed-clip cache. The full protocol is roughly 81
# utterances at ~5MB of JPEG each, so the default holds all of them: this bounds
# a runaway rather than forcing routine eviction.
CLIP_CACHE_MAX_MB = int(os.getenv("CLIP_CACHE_MAX_MB", "2048"))
# MuseTalk's whisper feature extraction takes a filename, so each render spills
# its audio to one short-lived file. /dev/shm is RAM; the fallback TEMP_DIR sits
# on networked GPFS in this deployment and is markedly slower.
_SHM = "/dev/shm"
RENDER_SCRATCH_DIR = os.getenv(
    "RENDER_SCRATCH_DIR",
    os.path.join(_SHM, f"digital-human-{os.getuid()}") if os.path.isdir(_SHM) else TEMP_DIR,
)

# The fixed-clip cache lives in RAM, so it is rebuilt from scratch on every
# boot. The pre-warm renders it on the same GPUs that serve conversations, which
# is why it only runs while nobody is mid-turn.
CLIP_PREWARM = os.getenv("CLIP_PREWARM", "1").lower() not in ("0", "false", "no")
# Treat a conversation as still live for this long after a turn ends, so the
# pre-warm does not seize a GPU in the gap between two turns of one exchange.
CLIP_PREWARM_IDLE_SEC = float(os.getenv("CLIP_PREWARM_IDLE_SEC", "5"))
CLIP_PREWARM_POLL_SEC = 0.5
CLIP_PREWARM_MAX_ATTEMPTS = 3   # give up on a clip that keeps failing to render

# Log patient and clinical content verbatim instead of redacting it to a shape
# summary. Local debugging only: transcripts then sit in the log in the clear.
LOG_PHI = os.getenv("LOG_PHI", "0").lower() in ("1", "true", "yes")

# Append-only record of consent decisions: decision, timestamp and a hash of the
# exact wording consented to. No transcripts and no screening data. The
# directory is gitignored and created on first write.
CONSENT_LOG_PATH = os.getenv(
    "CONSENT_LOG_PATH",
    os.path.join(RECORDS_DIR, "consent_log.jsonl"),
)

# End-of-utterance detection: a small ONNX model that judges whether a pause is
# semantically the end of a turn, so the avatar does not cut in when someone
# pauses to think. Runs on CPU, isolated from the MuseTalk GPUs; if it fails to
# load the VAD falls back to plain silence timing.
USE_EOU = os.getenv("USE_EOU", "1").lower() not in ("0", "false", "no")
EOU_MODEL_PATH = os.path.join(CHECKPOINTS_DIR, "smart-turn", "smart-turn-v3.2-cpu.onnx")
EOU_THRESHOLD = 0.5          # P(complete) at or above this ends the turn
# Consecutive "complete" verdicts required. Hysteresis against a single spike at
# a clause boundary cutting someone off mid-sentence; 1 disables it.
EOU_CONFIRM_CONSULTS = 2
EOU_PAUSE_DURATION = 0.20    # silence before the first consult
EOU_RECHECK_DURATION = 0.15  # consult cadence while silence continues
EOU_MAX_SILENCE = 2.0        # end the turn regardless once silence reaches this
EOU_ONNX_THREADS = 2

SESSION_IDLE_TTL_SEC = 600   # reap a session this long after its last client left

# The driving video doubles as the ambient loop: it is authored to loop
# seamlessly, so the page can play it directly and nothing has to be generated
# or invalidated.
IDLE_VIDEO_PATH = AVATAR_VIDEO

# Voice-only equivalent. The page's playback machinery is built around having an
# idle item to loop, so silence keeps that machinery identical across both modes
# instead of forking it. Encoded once at startup by main._ensure_idle_media.
IDLE_AUDIO_PATH = os.path.join(ASSETS_DIR, "idle_silence.mp3")
IDLE_AUDIO_DURATION = 2.0


def idle_media_path() -> str:
    """The clip the page loops when nobody is speaking, for the current mode."""
    return IDLE_VIDEO_PATH if ENABLE_VIDEO_AVATAR else IDLE_AUDIO_PATH

# Study-verbatim consent wording, spoken as two utterances: this preamble, then
# the question itself, which belongs to the protocol so that it can be re-asked
# from the same place as every other question when no answer arrives.
GREETING_PREAMBLE = (
    "Hello, I am an AI assistant designed to help understand some important factors "
    "that may impact your health. This information will be shared with your medical "
    "provider to help your provider better understand your current health issues. "
    "Your answers will be treated as confidential and as protected health information. "
)
CONSENT_QUESTION = templates.FIXED["consent.opening"]
# What the person actually hears before deciding, and therefore what the consent
# audit hashes. Composed rather than written out so the two halves cannot drift
# apart; the bytes are unchanged from when it was one string, so existing
# records still match.
GREETING_TEXT = GREETING_PREAMBLE + CONSENT_QUESTION
GREETING_CLIP_KEY = "greeting"

SERVER_HOST = os.getenv("SERVER_HOST", "0.0.0.0")
SERVER_PORT = int(os.getenv("SERVER_PORT", "17861"))
WS_PORT = int(os.getenv("WS_PORT", "17862"))

# Browsers only grant microphone access in a secure context, so a non-localhost
# deployment needs working certificates here or the page cannot listen at all.
SSL_CERT_FILE = os.getenv("SSL_CERT_FILE", os.path.join(CERTS_DIR, "cert.pem"))
SSL_KEY_FILE = os.getenv("SSL_KEY_FILE", os.path.join(CERTS_DIR, "key.pem"))
ENABLE_HTTPS = os.getenv("ENABLE_HTTPS", "1").lower() not in ("0", "false", "no")
