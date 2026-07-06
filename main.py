import os
import sys
import re
import time
import json
import asyncio
import logging
import threading
from contextlib import asynccontextmanager
import numpy as np


import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

os.makedirs(config.TEMP_DIR, exist_ok=True)

from starlette.applications import Starlette
from starlette.routing import Route, WebSocketRoute, Mount
from starlette.responses import FileResponse, JSONResponse
from starlette.requests import Request
from starlette.websockets import WebSocket
from starlette.staticfiles import StaticFiles

from modules.pipeline import Pipeline
from modules.vad import VoiceActivityDetector
from modules import asr, privacy

BASE_DIR = config.BASE_DIR   # single source of truth (config.py); don't redefine the project root here


# --------------- Per-session state ---------------
# Each browser (identified by a client-generated `sid`) gets a fully isolated
# Session: its own Pipeline (chat + LLM history + video queue + state machine),
# its own VAD stream state, its own mic toggle, and its own set of state-WS
# clients. Nothing is shared between sessions, so two users never cross-talk.

class Session:
    def __init__(self, sid: str = "default"):
        self.pipeline = Pipeline(audit_key=sid)
        self.vad = VoiceActivityDetector()
        self.mic_enabled = False
        self.speech_started_notified = False
        # ASR-confirmed barge-in bookkeeping (per audio stream).
        self.barge_done = False   # already barged-in for the current utterance
        self.barge_last_n = 0     # sample count at the last barge ASR check
        self.state_clients: set[WebSocket] = set()
        # state_poller bookkeeping (per session)
        self.last_state = None
        # (role, content) snapshot of the chat last pushed to this session's
        # clients, so the poller sends only the CHANGED suffix each tick instead of
        # re-serializing the whole history.
        self.sent_chat: list = []
        self.empty_since = None  # wall-clock when state_clients last dropped to 0


sessions: dict[str, Session] = {}
_sessions_lock = threading.Lock()


def get_or_create_session(sid: str) -> Session:
    """Get the session for `sid`, creating it (incl. a fresh VAD) on first use.
    Heavy (loads Silero VAD) — call via run_in_executor from async handlers."""
    with _sessions_lock:
        s = sessions.get(sid)
        if s is None:
            s = Session(sid)
            sessions[sid] = s
            logger.info("New session %s (total sessions: %d)", sid, len(sessions))
        return s


async def session_for(scope) -> Session:
    """Resolve the Session for a WebSocket/Request from its `?sid=` query param
    (falls back to a shared 'default' session for old clients without one)."""
    sid = scope.query_params.get("sid") or "default"
    return await asyncio.get_running_loop().run_in_executor(None, get_or_create_session, sid)


# --------------- HTTP routes ---------------

async def index(request: Request):
    return FileResponse(os.path.join(BASE_DIR, "static", "index.html"))


async def serve_video(request: Request):
    """Serve video files from tmp/ (per-sentence) and assets/clips/ (fixed clips)."""
    subdir = request.path_params["subdir"]
    filename = request.path_params["filename"]

    # Map URL token -> real directory. filename is a bare basename (str converter
    # rejects slashes), so no path-traversal guard is needed here.
    roots = {
        "tmp": os.path.join(BASE_DIR, "tmp"),
        "clips": config.CLIPS_DIR,          # assets/clips
        "assets": config.ASSETS_DIR,        # assets/ root (idle_loop.mp4)
    }
    root = roots.get(subdir)
    if root is None:
        return JSONResponse({"error": "not found"}, status_code=404)

    filepath = os.path.join(root, filename)
    if not os.path.isfile(filepath):
        return JSONResponse({"error": "not found"}, status_code=404)

    return FileResponse(filepath, media_type="video/mp4")


async def api_toggle(request: Request):
    session = await session_for(request)
    session.mic_enabled = not session.mic_enabled
    status = "on" if session.mic_enabled else "off"
    if not session.mic_enabled:
        session.pipeline.cancel_response()  # Stop silences the avatar mid-response
    logger.info(f"Mic toggled: {status}")
    return JSONResponse({"mic": status})


async def api_greet(request: Request):
    """Counselor leads: deliver the fixed opening. The client calls this ONLY after
    the mic is confirmed live, so we never greet on a denied mic or on a Clear (both
    of which previously happened when greeting was tied to the mic toggle)."""
    session = await session_for(request)
    p = session.pipeline
    if session.mic_enabled and not p.chat_history:
        p.start_greeting()
    return JSONResponse({"status": "ok"})


async def api_reset(request: Request):
    session = await session_for(request)
    # reset() sleeps ~100ms to let in-flight threads observe cancellation; run it off
    # the event loop so a Clear doesn't stall video delivery for every other session.
    await asyncio.get_running_loop().run_in_executor(None, session.pipeline.reset)
    session.mic_enabled = False
    session.sent_chat = []  # next delta re-pushes from scratch
    await broadcast(session.state_clients, {"type": "reset"})
    # Return the authoritative mic state so the client syncs without a 2nd toggle.
    return JSONResponse({"status": "ok", "mic": "off"})


async def api_text(request: Request):
    """Handle text input (fallback/testing)."""
    body = await request.json()
    text = body.get("text", "").strip()
    if not text:
        return JSONResponse({"error": "empty"}, status_code=400)

    session = await session_for(request)
    # Typing interrupts the current answer (like a voice barge-in): cancel the
    # in-flight response and drop its queued clips before starting the new turn.
    session.pipeline.on_speech_start()
    threading.Thread(target=session.pipeline.on_speech_end_text, args=(text,), daemon=True).start()
    return JSONResponse({"status": "processing"})


async def api_test_asr(request: Request):
    """Diagnostic: receive raw int16 PCM audio, run ASR, return result."""
    body = await request.body()
    audio_i16 = np.frombuffer(body, dtype=np.int16)
    audio_f32 = audio_i16.astype(np.float32) / 32768.0

    duration = len(audio_f32) / 16000
    rms = float(np.sqrt(np.mean(audio_f32 ** 2)))
    max_val = float(np.max(np.abs(audio_f32)))

    logger.info(f"[TestASR] Received {len(audio_i16)} samples, "
                f"duration={duration:.2f}s, rms={rms:.4f}, max={max_val:.4f}")

    # Save for inspection
    import scipy.io.wavfile as wavfile
    debug_path = os.path.join(BASE_DIR, "tmp", "test_asr_input.wav")
    wavfile.write(debug_path, 16000, audio_i16)
    logger.info(f"[TestASR] Saved to {debug_path}")

    # Run ASR
    from modules import asr
    text = asr.transcribe_array(audio_f32, sample_rate=16000)
    logger.info("[TestASR] Result: %s", privacy.phi(text))

    return JSONResponse({
        "text": text,
        "duration": f"{duration:.2f}",
        "rms": f"{rms:.4f}",
        "max": f"{max_val:.4f}",
    })


# --------------- Audio WebSocket ---------------

_word_re = re.compile(r"[^a-z0-9 ]+")


def _looks_like_echo(text: str, pipeline) -> bool:
    """The mic during playback may pick up the avatar's OWN leaked voice. Treat a
    transcription as echo (ignore it) if its words are contained in what the avatar
    is currently saying; real user speech won't match. Empty -> echo."""
    def norm(s):
        return " ".join(_word_re.sub(" ", s.lower()).split())
    u = norm(text)
    if not u:
        return True
    resp = ""
    for m in reversed(pipeline.chat_history):
        if m.get("role") == "assistant":
            resp = m.get("content", "")
            break
    return u in norm(resp)


async def ws_audio(websocket: WebSocket):
    await websocket.accept()
    if config.SHUTTING_DOWN.is_set():
        await websocket.close()
        return
    session = await session_for(websocket)
    logger.info("Audio WebSocket client connected")
    session.speech_started_notified = False
    _chunk_count = 0

    try:
        while True:
            data = await websocket.receive_bytes()
            # Stop dispatching new VAD work once shutting down so no executor job is
            # left running to block/hang the event loop teardown on Ctrl+C.
            if config.SHUTTING_DOWN.is_set():
                break
            if not session.mic_enabled:
                continue

            audio_chunk = np.frombuffer(data, dtype=np.int16)
            _chunk_count += 1
            if _chunk_count <= 3:
                logger.info(f"[AudioDebug] chunk #{_chunk_count}: len={len(audio_chunk)}, "
                           f"max={np.max(np.abs(audio_chunk))}, "
                           f"rms={np.sqrt(np.mean(audio_chunk.astype(np.float32)**2)):.1f}")

            # Run Silero VAD in a worker thread so its inference doesn't block the
            # asyncio event loop (which also pushes video to the frontend — VAD on
            # the loop made the two contend and made video delivery stutter).
            event, audio_data = await asyncio.get_running_loop().run_in_executor(
                None, session.vad.process_chunk, audio_chunk
            )

            # ASR-confirmed (semantic) barge-in: while the avatar is speaking, the VAD
            # onset is unreliable, so transcribe the user's speech-so-far and interrupt
            # the MOMENT it becomes real words (rejecting the avatar's own echo).
            if config.BARGE_IN_ASR and session.pipeline.state in ("processing", "speaking"):
                pending = session.vad.pending_audio()
                if pending is None:
                    session.barge_done = False
                    session.barge_last_n = 0
                elif not session.barge_done:
                    n = len(pending)
                    first = int(config.BARGE_IN_MIN_SPEECH * 16000)
                    step = int(config.BARGE_IN_RECHECK * 16000)
                    due = (n >= first) if session.barge_last_n == 0 \
                        else (n >= session.barge_last_n + step)
                    if due:
                        session.barge_last_n = n
                        text = await asyncio.get_running_loop().run_in_executor(
                            None, asr.transcribe_array, pending)
                        if text.strip() and not _looks_like_echo(text, session.pipeline):
                            logger.info("[barge-in] user speech detected: %s -> interrupting",
                                        privacy.phi(text))
                            session.barge_done = True
                            session.speech_started_notified = True
                            session.pipeline.on_speech_start()
            else:
                session.barge_done = False
                session.barge_last_n = 0

            if event == "speech_start" and not session.speech_started_notified:
                session.speech_started_notified = True
                session.pipeline.on_speech_start()

            elif event == "speech_end":
                session.speech_started_notified = False
                session.barge_done = False
                session.barge_last_n = 0
                # Debug: log audio stats
                duration = len(audio_data) / 16000
                rms = np.sqrt(np.mean(audio_data**2))
                logger.info(f"[AudioDebug] speech_end: duration={duration:.2f}s, "
                           f"rms={rms:.4f}, max={np.max(np.abs(audio_data)):.4f}")
                # Optional debug wav — OFF by default. The synchronous disk write
                # stalled the audio loop on every utterance. Enable: DEBUG_SAVE_AUDIO=1
                if config.DEBUG_SAVE_AUDIO:
                    import scipy.io.wavfile as wavfile
                    debug_path = os.path.join(BASE_DIR, "tmp", "debug_speech.wav")
                    wavfile.write(debug_path, 16000, (audio_data * 32767).astype(np.int16))
                    logger.info(f"[AudioDebug] Saved debug audio to {debug_path}")
                session.pipeline.on_speech_end(audio_data)

    except Exception as e:
        logger.info(f"Audio WebSocket disconnected: {e}")


# --------------- State WebSocket ---------------

async def ws_state(websocket: WebSocket):
    await websocket.accept()
    if config.SHUTTING_DOWN.is_set():
        await websocket.close()
        return
    session = await session_for(websocket)
    session.state_clients.add(websocket)
    session.empty_since = None
    logger.info(f"State WebSocket client connected (session clients: {len(session.state_clients)})")
    # Bring this client fully up to date on connect (full chat + current state);
    # every later push is just the changed suffix (delta) computed by the poller.
    try:
        chat = session.pipeline.get_chat_history()
        await websocket.send_text(json.dumps(
            {"type": "chat", "from": 0, "msgs": chat, "total": len(chat)}))
        await websocket.send_text(json.dumps(
            {"type": "state", "state": session.pipeline.state}))
    except Exception:
        pass
    try:
        # Keep connection alive; client doesn't send data
        while True:
            await websocket.receive_text()
    except Exception:
        pass
    finally:
        session.state_clients.discard(websocket)
        if not session.state_clients:
            session.empty_since = time.time()
        logger.info(f"State WebSocket client disconnected (session clients: {len(session.state_clients)})")


async def broadcast(clients: set, msg: dict):
    """Send a JSON message to the given set of state clients."""
    if not clients:
        return
    text = json.dumps(msg)
    disconnected = set()
    # Iterate over a snapshot: `await send_text` yields control, during which a
    # client may connect/disconnect and mutate the set. Iterating the live set
    # then raises "Set changed size during iteration", which would kill the
    # state_poller task and freeze all video delivery mid-response.
    for ws in list(clients):
        try:
            await ws.send_text(text)
        except Exception:
            disconnected.add(ws)
    clients.difference_update(disconnected)


# --------------- Background poller ---------------

async def _poll_session(session: Session):
    """Push one session's ready videos + state/chat changes to its own clients."""
    pipeline = session.pipeline
    clients = session.state_clients

    # Drain ALL segments ready this tick (not just one) so finished clips reach
    # the browser promptly instead of one-per-poll-interval.
    while True:
        item = pipeline.get_next_video()
        if item is None:
            break  # queue empty for now
        if item is False:
            await broadcast(clients, {
                "type": "video_end",
                "state": pipeline.state,
            })
            break
        # New video segment
        video_path = item["video"]
        if video_path.startswith(os.path.join(BASE_DIR, "tmp")):
            video_url = "/video/tmp/"  + os.path.basename(video_path)
        else:
            video_url = "/video/clips/" + os.path.basename(video_path)

        # Latency: how long the finished clip sat in the queue before delivery.
        t_enq = item.get("_t_enqueue")
        if t_enq is not None:
            logger.info("[latency] deliver %s after %.2fs in queue",
                        os.path.basename(video_path), time.perf_counter() - t_enq)

        await broadcast(clients, {
            "type": "video",
            "video_url": video_url,
            "subtitle": item["sentence"],
            "state": "speaking",
        })

    # Broadcast state changes (chat is pushed separately as a delta below).
    current_state = pipeline.state
    if current_state != session.last_state:
        session.last_state = current_state
        await broadcast(clients, {
            "type": "state",
            "state": current_state,
        })

    # Push only the CHANGED suffix of the chat (delta) — the assistant's streamed
    # sentences land in chat_history well before the first clip renders, so the
    # reply TEXT still appears in ~1s, but a long conversation is never
    # re-serialized and re-sent in full every tick.
    await _push_chat_delta(session)

    # Consent declined: end the session — turn the mic off server-side and tell the
    # client to stop capturing (it lets the goodbye clip finish, then resets to Start).
    if pipeline.ended:
        pipeline.ended = False
        session.mic_enabled = False
        await broadcast(clients, {"type": "stop"})


def _chat_delta(prev: list, chat: list):
    """Return (from_index, new_snapshot) where `chat` first differs from the
    already-sent `prev` snapshot. Messages are only appended, updated at the tail
    (streaming), or truncated at the tail (rollback/reset), so the first differing
    index onward is the minimal delta. Returns (None, None) if unchanged."""
    cur = [(m["role"], m.get("content", "")) for m in chat]
    i = 0
    n = min(len(cur), len(prev))
    while i < n and cur[i] == prev[i]:
        i += 1
    if i == len(cur) and len(cur) == len(prev):
        return None, None
    return i, cur


async def _push_chat_delta(session: Session):
    """Send only the changed suffix of this session's chat to its clients."""
    chat = session.pipeline.get_chat_history()
    i, cur = _chat_delta(session.sent_chat, chat)
    if i is None:
        return
    session.sent_chat = cur
    await broadcast(session.state_clients, {
        "type": "chat",
        "from": i,
        "msgs": chat[i:],
        "total": len(chat),
    })


async def state_poller():
    """Background task: poll every session's pipeline and push to its own clients."""
    try:
        while True:
            await asyncio.sleep(config.STATE_POLL_INTERVAL)
            if config.SHUTTING_DOWN.is_set():
                return
            # Never let a transient error kill this task: it is the ONLY producer of
            # video/state pushes to every frontend, so if it dies all avatars freeze
            # mid-response and stay broken until restart. (asyncio.CancelledError is
            # a BaseException, so this `except Exception` correctly lets shutdown
            # cancellation through — do not widen it to BaseException.)
            try:
                for session in list(sessions.values()):
                    if not session.state_clients:
                        continue
                    await _poll_session(session)
            except Exception:
                logger.exception("state_poller iteration failed; continuing")
    except asyncio.CancelledError:
        # Clean shutdown: the lifespan handler cancelled us. Exit quietly.
        return


# --------------- App setup ---------------

def _warmup_models():
    """Load the heavy models at startup so the FIRST user interaction is fast.
    Otherwise the first request pays a one-time ~10s+ cold load of the FLOAT GPU
    pool (and the ASR model). Runs in a daemon thread so the server starts
    listening immediately; get_pool()/get_model() are lock-guarded, so a user
    request arriving mid-warmup just waits on the same load (no double load).
    """
    try:
        logger.info("Pre-warming models (FLOAT pool on GPUs %s + ASR on GPU %s)...",
                    config.FLOAT_GPUS, config.ASR_GPU)
        from modules import avatar, asr
        avatar.get_pool()
        if config.SHUTTING_DOWN.is_set():
            return
        asr.get_model()
        if getattr(config, "USE_EOU", False) and not config.SHUTTING_DOWN.is_set():
            from modules import eou
            eou.get_model()  # self-handles failure -> falls back to silence VAD
        # Pre-render the fixed greeting + decline clips to cache once, so the opening
        # and a consent-decline both play instantly (no LLM/TTS/FLOAT at request time).
        if not config.SHUTTING_DOWN.is_set():
            from modules.pipeline import prewarm_fixed_clips
            prewarm_fixed_clips()
        logger.info("Model pre-warm complete; first response will be fast.")
    except Exception as e:
        logger.warning("Model pre-warm failed (will lazy-load on demand): %s", e)


def _reap_idle_sessions():
    """Drop sessions whose clients have all been gone for a grace period, so
    distinct browsers over time don't leak Pipeline/VAD objects forever. The
    shared 'default' session is never reaped.

    Deliberately does NOT require pipeline.state == 'idle': barge-ins and
    aborted turns can strand a session in 'listening'/'speaking' forever, and
    with zero connected clients nobody is watching anyway — cancel any
    in-flight response, then reap."""
    grace = getattr(config, "SESSION_IDLE_TTL_SEC", 600)
    now = time.time()
    with _sessions_lock:
        for sid in list(sessions.keys()):
            if sid == "default":
                continue
            s = sessions[sid]
            if (not s.state_clients
                    and s.empty_since and now - s.empty_since > grace):
                s.pipeline.cancel_response()   # stop in-flight work first
                del sessions[sid]
                logger.info("Reaped idle session %s (total sessions: %d)", sid, len(sessions))


def _temp_janitor():
    """Delete stale per-sentence clips from TEMP_DIR so long sessions don't fill
    the disk. Each utterance produces a .wav + .mp4 that are only needed until the
    browser has fetched and played them; anything older than the TTL is safe to
    remove. Runs forever in a daemon thread. Also reaps idle sessions.
    """
    ttl = getattr(config, "TEMP_FILE_TTL_SEC", 180)
    interval = getattr(config, "TEMP_CLEAN_INTERVAL_SEC", 30)
    exts = (".mp4", ".wav", ".txt", ".mp3")
    # Wait on the shutdown event instead of time.sleep so Ctrl+C exits this daemon
    # thread promptly (a long time.sleep left it to be killed mid-work at teardown).
    while not config.SHUTTING_DOWN.is_set():
        try:
            now = time.time()
            removed = 0
            for name in os.listdir(config.TEMP_DIR):
                if not name.endswith(exts):
                    continue
                path = os.path.join(config.TEMP_DIR, name)
                try:
                    if os.path.isfile(path) and now - os.path.getmtime(path) > ttl:
                        os.remove(path)
                        removed += 1
                except OSError:
                    pass
            if removed:
                logger.info("[janitor] removed %d stale temp files", removed)
            _reap_idle_sessions()
        except Exception:
            logger.exception("temp janitor iteration failed; continuing")
        config.SHUTTING_DOWN.wait(interval)


@asynccontextmanager
async def lifespan(app):
    """Start background work on boot and tear it down deterministically on shutdown
    so Ctrl+C exits cleanly — cancels the state poller (else asyncio logs "Task was
    destroyed but it is pending!") and flips SHUTTING_DOWN so the daemon threads,
    the FLOAT render busy-wait, and the LLM producer stop instead of being killed
    mid-work. (Uses the lifespan API, which works across Starlette versions;
    on_startup/on_shutdown were removed in newer Starlette.)"""
    poller = asyncio.create_task(state_poller())
    threading.Thread(target=_warmup_models, name="warmup", daemon=True).start()
    threading.Thread(target=_temp_janitor, name="janitor", daemon=True).start()
    try:
        yield
    finally:
        config.SHUTTING_DOWN.set()
        poller.cancel()
        await asyncio.gather(poller, return_exceptions=True)


# Ensure static directory exists
os.makedirs(os.path.join(BASE_DIR, "static"), exist_ok=True)

app = Starlette(
    routes=[
        Route("/", index),
        Route("/video/{subdir}/{filename}", serve_video),
        Route("/api/toggle", api_toggle, methods=["POST"]),
        Route("/api/greet", api_greet, methods=["POST"]),
        Route("/api/reset", api_reset, methods=["POST"]),
        Route("/api/text", api_text, methods=["POST"]),
        Route("/api/test_asr", api_test_asr, methods=["POST"]),
        WebSocketRoute("/ws/audio", ws_audio),
        WebSocketRoute("/ws/state", ws_state),
        Mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static"),
    ],
    lifespan=lifespan,
)


if __name__ == "__main__":
    import uvicorn

    try:
        # Generate idle video if it doesn't exist
        if not os.path.exists(config.IDLE_VIDEO_PATH):
            logger.info("Generating idle loop video (first-time setup)...")
            from modules import avatar
            try:
                avatar.generate_idle_video(duration=config.IDLE_VIDEO_DURATION)
                logger.info(f"Idle video saved to {config.IDLE_VIDEO_PATH}")
            except Exception as e:
                logger.warning(f"Could not generate idle video: {e}")

        # Enable HTTPS for public access (browser mic requires a secure context
        # on any non-localhost origin). Falls back to plain HTTP if certs are missing.
        ssl_kwargs = {}
        have_certs = os.path.exists(config.SSL_CERT_FILE) and os.path.exists(config.SSL_KEY_FILE)
        if config.ENABLE_HTTPS and have_certs:
            ssl_kwargs = {
                "ssl_certfile": config.SSL_CERT_FILE,
                "ssl_keyfile": config.SSL_KEY_FILE,
            }
            scheme = "https"
        else:
            scheme = "http"
            if config.ENABLE_HTTPS and not have_certs:
                logger.warning(
                    "ENABLE_HTTPS is set but certs not found at %s / %s — serving plain HTTP. "
                    "Microphone will only work via localhost.",
                    config.SSL_CERT_FILE, config.SSL_KEY_FILE,
                )

        logger.info(
            "Serving on %s://%s:%d  (open %s://<your-host>:%d/ from the public internet)",
            scheme, config.SERVER_HOST, config.SERVER_PORT, scheme, config.SERVER_PORT,
        )
        uvicorn.run(
            app,
            host=config.SERVER_HOST,
            port=config.SERVER_PORT,
            log_level="info",
            **ssl_kwargs,
        )
    except KeyboardInterrupt:
        # Ctrl+C BEFORE uvicorn installs its signal handlers (e.g. during first-run
        # idle-video generation) would otherwise dump a raw KeyboardInterrupt
        # traceback. Exit cleanly instead.
        config.SHUTTING_DOWN.set()
        logger.info("Interrupted during startup; exiting.")
        sys.exit(0)
