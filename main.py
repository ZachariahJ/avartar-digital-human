import os
import sys
import re
import time
import subprocess
import json
import asyncio
import logging
import threading
import queue
from contextlib import asynccontextmanager
import numpy as np


import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


from starlette.applications import Starlette
from starlette.routing import Route, WebSocketRoute, Mount
from starlette.responses import FileResponse, JSONResponse, HTMLResponse, Response
from starlette.requests import Request
from starlette.websockets import WebSocket
from starlette.staticfiles import StaticFiles

from modules.pipeline import Pipeline
from modules.vad import VoiceActivityDetector
from modules import asr, clipcache, privacy

BASE_DIR = config.BASE_DIR


class Session:

    def __init__(self, sid: str = "default"):
        self.pipeline = Pipeline(audit_key=sid)
        self.vad = VoiceActivityDetector()
        self.mic_enabled = False
        self.speech_started_notified = False
        self.barge_done = False
        self.barge_last_n = 0
        self.state_clients: set[WebSocket] = set()
        self.last_state = None
        self.active_seg = None
        self.seg_id = 0
        self.seg_frames = 0
        self.sent_chat: list = []
        self.empty_since = None


sessions: dict[str, Session] = {}
_sessions_lock = threading.Lock()


def get_or_create_session(sid: str) -> Session:
    with _sessions_lock:
        s = sessions.get(sid)
        if s is None:
            s = Session(sid)
            sessions[sid] = s
            logger.info("New session %s (total sessions: %d)", sid, len(sessions))
        return s


async def session_for(scope) -> Session:
    sid = scope.query_params.get("sid") or "default"
    return await asyncio.get_running_loop().run_in_executor(None, get_or_create_session, sid)


def _app_config() -> dict:
    idle_name = os.path.basename(config.idle_media_path())
    return {
        "videoAvatar": config.ENABLE_VIDEO_AVATAR,
        "idleUrl": "/video/assets/" + idle_name,
        "portraitUrl": "/video/assets/" + os.path.basename(config.AVATAR_IMAGE),
    }


async def index(request: Request):
    with open(os.path.join(BASE_DIR, "static", "index.html"), encoding="utf-8") as f:
        html = f.read()
    html = html.replace(
        "<!--APP_CONFIG-->",
        "<script>window.APP_CONFIG = %s;</script>" % json.dumps(_app_config()),
    )
    return HTMLResponse(html, headers={"Cache-Control": "no-store"})


_MEDIA_TYPES = {".mp4": "video/mp4", ".mp3": "audio/mpeg", ".png": "image/png"}


async def serve_video(request: Request):
    if request.path_params["subdir"] != "assets":
        return JSONResponse({"error": "not found"}, status_code=404)
    filename = request.path_params["filename"]
    filepath = os.path.join(config.ASSETS_DIR, filename)
    if not os.path.isfile(filepath):
        return JSONResponse({"error": "not found"}, status_code=404)
    return FileResponse(
        filepath,
        media_type=_MEDIA_TYPES.get(os.path.splitext(filename)[1].lower(),
                                    "application/octet-stream"),
        headers={"Cache-Control": "no-cache"},
    )


async def serve_media(request: Request):
    data = clipcache.fetch(clipcache.token_from_url(request.path_params["name"]))
    if data is None:
        return JSONResponse({"error": "not found"}, status_code=404)
    return Response(data, media_type=clipcache.MEDIA_TYPE,
                    headers={"Cache-Control": "no-store"})


async def api_toggle(request: Request):
    session = await session_for(request)
    session.mic_enabled = not session.mic_enabled
    status = "on" if session.mic_enabled else "off"
    if not session.mic_enabled:
        session.pipeline.cancel_response()
    logger.info(f"Mic toggled: {status}")
    return JSONResponse({"mic": status})


async def api_greet(request: Request):
    session = await session_for(request)
    p = session.pipeline
    if session.mic_enabled and not p.chat_history:
        p.start_greeting()
    return JSONResponse({"status": "ok"})


async def api_reset(request: Request):
    session = await session_for(request)
    await asyncio.get_running_loop().run_in_executor(None, session.pipeline.reset)
    session.mic_enabled = False
    session.sent_chat = []
    await broadcast(session.state_clients, {"type": "reset"})
    return JSONResponse({"status": "ok", "mic": "off"})


async def api_text(request: Request):
    body = await request.json()
    text = body.get("text", "").strip()
    if not text:
        return JSONResponse({"error": "empty"}, status_code=400)

    session = await session_for(request)
    session.pipeline.on_speech_start()
    threading.Thread(target=session.pipeline.on_speech_end_text, args=(text,), daemon=True).start()
    return JSONResponse({"status": "processing"})


def _looks_like_echo(text: str, pipeline) -> bool:
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


async def _barge_in(session: Session, reason: str):
    logger.info("[barge-in] %s -> cascade flush", reason)
    session.pipeline.on_speech_start()
    session.active_seg = None
    await broadcast(session.state_clients, {"type": "flush", "id": session.seg_id})


async def ws_audio(websocket: WebSocket):
    await websocket.accept()
    if config.SHUTTING_DOWN.is_set():
        await websocket.close()
        return
    session = await session_for(websocket)
    logger.info("Audio WebSocket client connected")
    session.speech_started_notified = False
    _chunk_count = 0
    _warmup_needed = int(config.MIC_WARMUP_DISCARD * 16000)
    _warmup_samples = 0

    try:
        while True:
            data = await websocket.receive_bytes()
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

            if _warmup_samples < _warmup_needed:
                _warmup_samples += len(audio_chunk)
                if _warmup_samples >= _warmup_needed:
                    logger.info("[AudioDebug] mic warm-up: discarded %.2fs (%d chunks)",
                                _warmup_samples / 16000, _chunk_count)
                continue

            event, audio_data = await asyncio.get_running_loop().run_in_executor(
                None, session.vad.process_chunk, audio_chunk
            )

            speaking = session.pipeline.state in ("processing", "speaking")

            if config.BARGE_IN_VAD and speaking and not session.barge_done:
                pending = session.vad.pending_audio()
                if pending is not None and \
                        len(pending) >= int(config.BARGE_IN_VAD_SUSTAIN * 16000):
                    session.barge_done = True
                    session.speech_started_notified = True
                    await _barge_in(session, "%.0fms sustained voice"
                                    % (config.BARGE_IN_VAD_SUSTAIN * 1000))
                    continue

            if config.BARGE_IN_ASR and speaking:
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
                            session.barge_done = True
                            session.speech_started_notified = True
                            await _barge_in(session, "ASR confirmed: %s"
                                            % privacy.phi(text))
            else:
                session.barge_done = False
                session.barge_last_n = 0

            if event == "speech_start" and not session.speech_started_notified:
                session.speech_started_notified = True
                await _barge_in(session, "VAD speech_start")

            elif event == "speech_end":
                session.speech_started_notified = False
                session.barge_done = False
                session.barge_last_n = 0
                duration = len(audio_data) / 16000
                rms = np.sqrt(np.mean(audio_data**2))
                logger.info(f"[AudioDebug] speech_end: duration={duration:.2f}s, "
                           f"rms={rms:.4f}, max={np.max(np.abs(audio_data)):.4f}")
                session.pipeline.on_speech_end(audio_data)

    except Exception as e:
        logger.info(f"Audio WebSocket disconnected: {e}")


async def ws_state(websocket: WebSocket):
    await websocket.accept()
    if config.SHUTTING_DOWN.is_set():
        await websocket.close()
        return
    session = await session_for(websocket)
    session.state_clients.add(websocket)
    session.empty_since = None
    logger.info(f"State WebSocket client connected (session clients: {len(session.state_clients)})")
    try:
        chat = session.pipeline.get_chat_history()
        await websocket.send_text(json.dumps(
            {"type": "chat", "from": 0, "msgs": chat, "total": len(chat)}))
        await websocket.send_text(json.dumps(
            {"type": "state", "state": session.pipeline.state}))
    except Exception:
        pass
    try:
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
    if not clients:
        return
    text = json.dumps(msg)
    disconnected = set()
    for ws in list(clients):
        try:
            await ws.send_text(text)
        except Exception:
            disconnected.add(ws)
    clients.difference_update(disconnected)


async def broadcast_bytes(clients: set, payload: bytes):
    if not clients:
        return
    disconnected = set()
    for ws in list(clients):
        try:
            await ws.send_bytes(payload)
        except Exception:
            disconnected.add(ws)
    clients.difference_update(disconnected)


_MAX_FRAMES_PER_TICK = 24


async def _pump_segments(session: Session):
    pipeline = session.pipeline
    clients = session.state_clients

    while True:
        seg = session.active_seg
        if seg is None:
            item = pipeline.get_next_video()
            if item is None:
                return
            if item is False:
                await broadcast(clients, {"type": "video_end", "state": pipeline.state})
                return
            seg = session.active_seg = item
            session.seg_id += 1
            session.seg_frames = 0
            if seg._t_enqueue:
                logger.info("[latency] segment %d announced after %.2fs in queue",
                            session.seg_id, time.perf_counter() - seg._t_enqueue)
            await broadcast(clients, {
                "type": "segment",
                "id": session.seg_id,
                "audio_url": seg.audio_url,
                "fps": seg.fps,
                "prebuffer": config.STREAM_PREBUFFER_FRAMES,
                "subtitle": seg.sentence,
                "state": "speaking",
            })

        for _ in range(_MAX_FRAMES_PER_TICK):
            if seg.cancelled.is_set():
                await broadcast(clients, {"type": "flush", "id": session.seg_id})
                session.active_seg = None
                return
            try:
                frame = seg.frames.get_nowait()
            except queue.Empty:
                return
            if frame is None:
                await broadcast(clients, {"type": "segment_end", "id": session.seg_id,
                                          "frames": session.seg_frames})
                session.active_seg = None
                break
            await broadcast_bytes(
                clients,
                session.seg_id.to_bytes(4, "big")
                + session.seg_frames.to_bytes(4, "big")
                + frame,
            )
            session.seg_frames += 1
        else:
            return


async def _poll_session(session: Session):
    pipeline = session.pipeline
    clients = session.state_clients

    await _pump_segments(session)

    current_state = pipeline.state
    if current_state != session.last_state:
        session.last_state = current_state
        await broadcast(clients, {
            "type": "state",
            "state": current_state,
        })

    await _push_chat_delta(session)

    if pipeline.ended:
        pipeline.ended = False
        session.mic_enabled = False
        await broadcast(clients, {"type": "stop"})


def _chat_delta(prev: list, chat: list):
    cur = [(m["role"], m.get("content", "")) for m in chat]
    i = 0
    n = min(len(cur), len(prev))
    while i < n and cur[i] == prev[i]:
        i += 1
    if i == len(cur) and len(cur) == len(prev):
        return None, None
    return i, cur


async def _push_chat_delta(session: Session):
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
    try:
        while True:
            await asyncio.sleep(config.STATE_POLL_INTERVAL)
            if config.SHUTTING_DOWN.is_set():
                return
            try:
                for session in list(sessions.values()):
                    if not session.state_clients:
                        continue
                    await _poll_session(session)
            except Exception:
                logger.exception("state_poller iteration failed; continuing")
    except asyncio.CancelledError:
        return


def _warmup_models():
    try:
        if config.ENABLE_VIDEO_AVATAR:
            logger.info("Pre-warming models (MuseTalk pool on GPUs %s + ASR on GPU %s)...",
                        config.MUSETALK_GPUS, config.ASR_GPU)
            from modules import avatar
            avatar.get_pool()
        else:
            logger.info("Voice-only mode (ENABLE_VIDEO_AVATAR=0): skipping MuseTalk; "
                        "pre-warming ASR on GPU %s...", config.ASR_GPU)
        if config.SHUTTING_DOWN.is_set():
            return
        asr.get_model()
        if config.USE_EOU and not config.SHUTTING_DOWN.is_set():
            from modules import eou
            eou.get_model()
        if not config.SHUTTING_DOWN.is_set():
            from modules.pipeline import prewarm_fixed_clips
            threading.Thread(target=prewarm_fixed_clips, name="clip-prewarm",
                             daemon=True).start()
        logger.info("Model pre-warm complete; first response will be fast.")
    except Exception as e:
        logger.warning("Model pre-warm failed (will lazy-load on demand): %s", e)


def _reap_idle_sessions():
    grace = config.SESSION_IDLE_TTL_SEC
    now = time.time()
    with _sessions_lock:
        for sid in list(sessions.keys()):
            if sid == "default":
                continue
            s = sessions[sid]
            if (not s.state_clients
                    and s.empty_since and now - s.empty_since > grace):
                s.pipeline.cancel_response()
                del sessions[sid]
                logger.info("Reaped idle session %s (total sessions: %d)", sid, len(sessions))


def _ensure_idle_media():
    if config.ENABLE_VIDEO_AVATAR:
        return
    if os.path.exists(config.IDLE_AUDIO_PATH):
        return
    logger.info("Generating idle silence clip...")
    os.makedirs(os.path.dirname(config.IDLE_AUDIO_PATH), exist_ok=True)
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-f", "lavfi",
             "-i", "anullsrc=r=24000:cl=mono",
             "-t", str(config.IDLE_AUDIO_DURATION),
             "-c:a", "libmp3lame", "-b:a", "48k",
             config.IDLE_AUDIO_PATH],
            check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        logger.info("Idle silence saved to %s", config.IDLE_AUDIO_PATH)
    except (OSError, subprocess.CalledProcessError) as e:
        logger.warning("Could not generate idle silence: %s", e)


def _temp_janitor():
    ttl = config.TEMP_FILE_TTL_SEC
    interval = config.TEMP_CLEAN_INTERVAL_SEC
    exts = (".mp3",)
    while not config.SHUTTING_DOWN.is_set():
        try:
            now = time.time()
            removed = 0
            try:
                names = os.listdir(config.RENDER_SCRATCH_DIR)
            except OSError:
                names = []
            for name in names:
                if not name.endswith(exts):
                    continue
                path = os.path.join(config.RENDER_SCRATCH_DIR, name)
                try:
                    if os.path.isfile(path) and now - os.path.getmtime(path) > ttl:
                        os.remove(path)
                        removed += 1
                except OSError:
                    pass
            if removed:
                logger.info("[janitor] removed %d stale temp files", removed)
            expired = clipcache.sweep()
            if expired:
                logger.info("[janitor] expired %d media blobs; %s",
                            expired, clipcache.stats())
            _reap_idle_sessions()
        except Exception:
            logger.exception("temp janitor iteration failed; continuing")
        config.SHUTTING_DOWN.wait(interval)


@asynccontextmanager
async def lifespan(app):
    poller = asyncio.create_task(state_poller())
    threading.Thread(target=_warmup_models, name="warmup", daemon=True).start()
    threading.Thread(target=_temp_janitor, name="janitor", daemon=True).start()
    try:
        yield
    finally:
        config.SHUTTING_DOWN.set()
        poller.cancel()
        await asyncio.gather(poller, return_exceptions=True)


os.makedirs(os.path.join(BASE_DIR, "static"), exist_ok=True)

app = Starlette(
    routes=[
        Route("/", index),
        Route("/video/{subdir}/{filename}", serve_video),
        Route("/media/{name}", serve_media),
        Route("/api/toggle", api_toggle, methods=["POST"]),
        Route("/api/greet", api_greet, methods=["POST"]),
        Route("/api/reset", api_reset, methods=["POST"]),
        Route("/api/text", api_text, methods=["POST"]),
        WebSocketRoute("/ws/audio", ws_audio),
        WebSocketRoute("/ws/state", ws_state),
        Mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static"),
    ],
    lifespan=lifespan,
)


if __name__ == "__main__":
    import uvicorn

    try:
        _ensure_idle_media()

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
        config.SHUTTING_DOWN.set()
        logger.info("Interrupted during startup; exiting.")
        sys.exit(0)
