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
    """Everything belonging to one browser, keyed by a client-generated `sid`.

    Sessions share nothing, so two people using the server at once cannot see or
    interrupt each other's conversation. Fields are read by the async handlers
    and written by the state poller; both run on the event loop, so they need no
    locking among themselves.
    """

    def __init__(self, sid: str = "default"):
        self.pipeline = Pipeline(audit_key=sid)
        self.vad = VoiceActivityDetector()
        self.mic_enabled = False
        self.speech_started_notified = False
        self.barge_done = False   # already interrupted for the current utterance
        self.barge_last_n = 0     # samples seen at the last barge-in ASR check
        self.state_clients: set[WebSocket] = set()
        self.last_state = None
        # The poller forwards one segment at a time and waits for its
        # end-of-frames sentinel before taking the next, which is what keeps
        # utterances in order on the wire while several render concurrently.
        self.active_seg = None
        self.seg_id = 0
        self.seg_frames = 0     # frame index within active_seg, as sent
        # (role, content) of the chat as this session's clients last saw it, so
        # each tick sends only the changed tail instead of the whole history.
        self.sent_chat: list = []
        self.empty_since = None  # when state_clients last fell to zero


sessions: dict[str, Session] = {}
_sessions_lock = threading.Lock()


def get_or_create_session(sid: str) -> Session:
    """Return the session for `sid`, creating it on first use.

    Creating one loads Silero VAD, which is slow enough to stall the event loop,
    so async callers must reach this through run_in_executor.
    """
    with _sessions_lock:
        s = sessions.get(sid)
        if s is None:
            s = Session(sid)
            sessions[sid] = s
            logger.info("New session %s (total sessions: %d)", sid, len(sessions))
        return s


async def session_for(scope) -> Session:
    """Session for a request or WebSocket, from its `?sid=` query parameter.

    A client that sends no sid shares one "default" session with every other
    such client.
    """
    sid = scope.query_params.get("sid") or "default"
    return await asyncio.get_running_loop().run_in_executor(None, get_or_create_session, sid)


def _app_config() -> dict:
    """Settings the page needs before it renders anything.

    These are inlined into the HTML rather than sent over the state socket
    because the page chooses its ambient clip while its first script runs, long
    before any socket opens. Delivered asynchronously, the browser would request
    the other mode's idle clip and 404 first.
    """
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
    # The inlined config above changes with the server's mode, so a cached copy
    # of this page would keep requesting media the server no longer serves.
    return HTMLResponse(html, headers={"Cache-Control": "no-store"})


_MEDIA_TYPES = {".mp4": "video/mp4", ".mp3": "audio/mpeg", ".png": "image/png"}


async def serve_video(request: Request):
    """Serve one file from assets/: the idle loop, the portrait, idle silence.

    Spoken utterances never come through here. Their audio lives in RAM behind a
    /media/<token> URL and their frames go down the state socket as bytes.

    The no-cache header is load-bearing. These paths are stable URLs over
    mutable bytes, so replacing the avatar must not leave browsers replaying the
    old face because a heuristic judged the cached copy fresh. "no-cache" asks
    for revalidation, not for the file to go unstored. Starlette's FileResponse
    sends an etag but does not answer conditional requests, so every
    revalidation transfers the whole file — acceptable at one fetch per page
    load.
    """
    if request.path_params["subdir"] != "assets":
        return JSONResponse({"error": "not found"}, status_code=404)
    # Starlette's str converter rejects slashes, so this is always a bare
    # filename and cannot traverse out of assets/.
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
    """Serve one utterance's audio from the in-RAM store (modules/clipcache.py).

    Tokens are minted per publish, so a URL names exact bytes and can never be
    answered with another session's audio. An expired token is an ordinary 404;
    by then the browser has long since fetched and played the clip.

    no-store rather than no-cache: this is clinical speech and there is no reuse
    to win, since each URL is fetched once and fixed clips lose their tokens
    when the process exits.
    """
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
        session.pipeline.cancel_response()  # switching off also cuts the current answer
    logger.info(f"Mic toggled: {status}")
    return JSONResponse({"mic": status})


async def api_greet(request: Request):
    """Speak the opening line, so the counselor leads rather than waits.

    The client calls this only once the microphone is confirmed live. Greeting
    on the mic toggle instead would also greet when permission was denied, or
    again after a Clear.
    """
    session = await session_for(request)
    p = session.pipeline
    if session.mic_enabled and not p.chat_history:
        p.start_greeting()
    return JSONResponse({"status": "ok"})


async def api_reset(request: Request):
    session = await session_for(request)
    # reset() blocks ~100ms waiting for in-flight threads to notice cancellation.
    # Off the event loop, so one client's Clear does not stall video delivery to
    # every other session.
    await asyncio.get_running_loop().run_in_executor(None, session.pipeline.reset)
    session.mic_enabled = False
    session.sent_chat = []  # forces the next delta to resend the whole history
    await broadcast(session.state_clients, {"type": "reset"})
    # Reporting the mic state saves the client a second round trip to sync it.
    return JSONResponse({"status": "ok", "mic": "off"})


async def api_text(request: Request):
    """Accept typed input, as a fallback when the microphone is unusable."""
    body = await request.json()
    text = body.get("text", "").strip()
    if not text:
        return JSONResponse({"error": "empty"}, status_code=400)

    session = await session_for(request)
    # Typing interrupts exactly like speaking does: drop the in-flight answer and
    # its queued clips before the new turn starts.
    session.pipeline.on_speech_start()
    threading.Thread(target=session.pipeline.on_speech_end_text, args=(text,), daemon=True).start()
    return JSONResponse({"status": "processing"})


def _looks_like_echo(text: str, pipeline) -> bool:
    """True if `text` is probably the avatar's own voice coming back in.

    While a clip plays, the microphone can pick up what the avatar is saying. A
    transcript whose words are contained in the avatar's current line is treated
    as echo and ignored; genuine speech will not match. Empty text counts as
    echo, since there is nothing to act on.
    """
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
    """Stop the current response on the server and in the browser at once.

    pipeline.on_speech_start() cascades the server side: the LLM stream closes,
    no further sentence is sent for synthesis, every live segment is cancelled
    (which stops its MuseTalk render at the next batch and empties its frame
    queue) and the delivery queue is cleared. One synthesis already in flight
    cannot be stopped; see modules.tts.synthesize.

    The flush message is sent here rather than left to the next poll tick. The
    poller only runs every STATE_POLL_INTERVAL, and those 100ms of the avatar
    still talking are precisely what instant barge-in exists to remove.
    """
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
    # Warm-up state is per connection, not per session: the client opens a new
    # socket each time the mic starts, which is exactly when a fresh start-up
    # transient arrives and the count needs to restart.
    _warmup_needed = int(config.MIC_WARMUP_DISCARD * 16000)
    _warmup_samples = 0

    try:
        while True:
            data = await websocket.receive_bytes()
            # Dispatch no new VAD work during shutdown, or an executor job
            # outlives the loop and hangs teardown on Ctrl+C.
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

            # Discard the head of the stream before the VAD sees it. The mic
            # start-up transient clips and reads as speech, producing an
            # utterance nobody spoke; dropping it here rather than after
            # process_chunk means the VAD never enters is_speaking and there is
            # no buffer state to unwind.
            if _warmup_samples < _warmup_needed:
                _warmup_samples += len(audio_chunk)
                if _warmup_samples >= _warmup_needed:
                    logger.info("[AudioDebug] mic warm-up: discarded %.2fs (%d chunks)",
                                _warmup_samples / 16000, _chunk_count)
                continue

            # Silero inference goes to a worker thread because this same event
            # loop also pushes video; running the VAD on it makes the two
            # contend and the video visibly stutters.
            event, audio_data = await asyncio.get_running_loop().run_in_executor(
                None, session.vad.process_chunk, audio_chunk
            )

            speaking = session.pipeline.state in ("processing", "speaking")

            # Fast path: sustained voice alone cuts the response, with no
            # transcription in the loop. It is checked first; the ASR path below
            # only runs if this is disabled or has not yet reached its
            # threshold.
            #
            # This gives up the ASR path's echo rejection, which compared the
            # transcript against what the avatar was saying — there is nothing
            # to compare here. It works only because getUserMedia is opened with
            # echoCancellation:{exact:true}, making cancellation guaranteed
            # rather than hoped for. An avatar that interrupts itself means echo
            # cancellation is failing; BARGE_IN_VAD=0 falls back to the slower,
            # self-checking path.
            if config.BARGE_IN_VAD and speaking and not session.barge_done:
                pending = session.vad.pending_audio()
                if pending is not None and \
                        len(pending) >= int(config.BARGE_IN_VAD_SUSTAIN * 16000):
                    session.barge_done = True
                    session.speech_started_notified = True
                    await _barge_in(session, "%.0fms sustained voice"
                                    % (config.BARGE_IN_VAD_SUSTAIN * 1000))
                    continue

            # Slow path: the VAD's speech onset is unreliable while the avatar
            # talks, so transcribe what has been heard so far and interrupt as
            # soon as it forms words that are not the avatar's own echo.
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
    # Send the full chat and state once on connect. Everything after this is a
    # tail delta from the poller, which only works against a known baseline.
    try:
        chat = session.pipeline.get_chat_history()
        await websocket.send_text(json.dumps(
            {"type": "chat", "from": 0, "msgs": chat, "total": len(chat)}))
        await websocket.send_text(json.dumps(
            {"type": "state", "state": session.pipeline.state}))
    except Exception:
        pass
    try:
        # The client never sends on this socket; receiving is only how a
        # disconnect is noticed.
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
    """Send one JSON message to a set of state clients, dropping dead ones."""
    if not clients:
        return
    text = json.dumps(msg)
    disconnected = set()
    # Iterate a snapshot. `await send_text` yields, and a client connecting or
    # disconnecting in that window mutates the set; iterating it live raises
    # "Set changed size during iteration", which kills the poller task and
    # freezes video delivery for everyone.
    for ws in list(clients):
        try:
            await ws.send_text(text)
        except Exception:
            disconnected.add(ws)
    clients.difference_update(disconnected)


async def broadcast_bytes(clients: set, payload: bytes):
    """Send one binary frame to a set of state clients, dropping dead ones."""
    if not clients:
        return
    disconnected = set()
    for ws in list(clients):
        try:
            await ws.send_bytes(payload)
        except Exception:
            disconnected.add(ws)
    clients.difference_update(disconnected)


# Frames one session may send per poll tick. At a 0.1s interval this is several
# times the rate playback consumes, so the pump never becomes the bottleneck —
# but it still bounds how long a single session can hold the event loop when
# many are streaming at once.
_MAX_FRAMES_PER_TICK = 24


async def _pump_segments(session: Session):
    """Forward whatever audio and frames are ready, in strict utterance order.

    Both channels share the one state socket:
      - text:   {"type":"segment"} opens an utterance (audio URL and fps),
                {"type":"segment_end"} closes it, {"type":"flush"} abandons it.
      - binary: [uint32 segment id][uint32 frame index][JPEG bytes]

    Every frame carries its segment id so that a client which has just flushed
    can discard frames still arriving for the abandoned utterance rather than
    painting them over the idle loop.
    """
    pipeline = session.pipeline
    clients = session.state_clients

    while True:
        seg = session.active_seg
        if seg is None:
            item = pipeline.get_next_video()
            if item is None:
                return                       # renderer has produced nothing yet
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
                # Ending this normally would let the client play out what it has
                # already buffered. A flush tells it to throw that away.
                await broadcast(clients, {"type": "flush", "id": session.seg_id})
                session.active_seg = None
                return
            try:
                frame = seg.frames.get_nowait()
            except queue.Empty:
                return                       # still rendering; resume next tick
            if frame is None:
                await broadcast(clients, {"type": "segment_end", "id": session.seg_id,
                                          "frames": session.seg_frames})
                session.active_seg = None
                break                        # utterance complete; take the next
            await broadcast_bytes(
                clients,
                session.seg_id.to_bytes(4, "big")
                + session.seg_frames.to_bytes(4, "big")
                + frame,
            )
            session.seg_frames += 1
        else:
            return                           # per-tick budget spent


async def _poll_session(session: Session):
    """Push one session's ready video, state and chat changes to its clients."""
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

    # Sentences reach chat_history as the LLM streams them, well before their
    # clips render, so the reply text shows up quickly. Sending only the changed
    # tail keeps that cheap as the conversation grows.
    await _push_chat_delta(session)

    # The pipeline closed the session: consent refused, the user aborted, or the
    # protocol reached its end and spoke the closing line. Drop the mic here and
    # tell the client to stop capturing; it plays the goodbye out first, then
    # returns to Start.
    if pipeline.ended:
        pipeline.ended = False
        session.mic_enabled = False
        await broadcast(clients, {"type": "stop"})


def _chat_delta(prev: list, chat: list):
    """First index at which `chat` differs from `prev`, plus a new snapshot.

    Messages are only ever appended, edited at the tail while streaming, or
    truncated at the tail on reset, so everything from the first difference
    onward is the minimal delta. Returns (None, None) when nothing changed.
    """
    cur = [(m["role"], m.get("content", "")) for m in chat]
    i = 0
    n = min(len(cur), len(prev))
    while i < n and cur[i] == prev[i]:
        i += 1
    if i == len(cur) and len(cur) == len(prev):
        return None, None
    return i, cur


async def _push_chat_delta(session: Session):
    """Send the changed tail of this session's chat to its clients."""
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
    """Drive every session's pushes to its clients, for the life of the server."""
    try:
        while True:
            await asyncio.sleep(config.STATE_POLL_INTERVAL)
            if config.SHUTTING_DOWN.is_set():
                return
            # This task is the only source of video and state pushes for every
            # client, so one transient error must not end it — every avatar
            # would freeze mid-sentence until the server restarted. Catching
            # Exception rather than BaseException is deliberate: it lets
            # CancelledError through so shutdown still works.
            try:
                for session in list(sessions.values()):
                    if not session.state_clients:
                        continue
                    await _poll_session(session)
            except Exception:
                logger.exception("state_poller iteration failed; continuing")
    except asyncio.CancelledError:
        # Cancelled by the lifespan handler on shutdown; nothing to report.
        return


def _warmup_models():
    """Load the heavy models now so the first person to speak does not wait.

    Without this the first request pays for the MuseTalk pool and the ASR model
    cold, and on the first boot after the driving video changes it also pays for
    preparing the driving material, which takes minutes.

    Runs in a daemon thread so the server accepts connections immediately.
    get_pool() and get_model() hold locks, so a request arriving mid-warmup
    waits on the same load rather than starting a second one.
    """
    try:
        if config.ENABLE_VIDEO_AVATAR:
            logger.info("Pre-warming models (MuseTalk pool on GPUs %s + ASR on GPU %s)...",
                        config.MUSETALK_GPUS, config.ASR_GPU)
            from modules import avatar
            avatar.get_pool()
        else:
            # Voice-only never imports MuseTalk at all: no GPU memory, no UNet
            # or VAE checkpoints, no material preparation, and no multi-minute
            # first-boot render.
            logger.info("Voice-only mode (ENABLE_VIDEO_AVATAR=0): skipping MuseTalk; "
                        "pre-warming ASR on GPU %s...", config.ASR_GPU)
        if config.SHUTTING_DOWN.is_set():
            return
        asr.get_model()
        if config.USE_EOU and not config.SHUTTING_DOWN.is_set():
            from modules import eou
            eou.get_model()  # handles its own failure by falling back to the VAD
        # TTS is a separate service, and every spoken line depends on it. Report
        # it unreachable now rather than letting the failure first surface as a
        # silent avatar.
        from modules import tts
        if tts.healthy():
            logger.info("TTS server reachable at %s", config.TTS_SERVER_URL)
        else:
            logger.error("TTS server NOT reachable at %s — start it with "
                         "scripts/tts_server.sh. Fixed clips cannot be "
                         "pre-warmed and dynamic replies will be silent "
                         "(their text still reaches the chat).",
                         config.TTS_SERVER_URL)

        # Refill the clip cache with every fixed utterance. Deliberately not
        # awaited: it renders on the same GPUs that answer people, so it runs in
        # its own thread and yields as soon as anyone speaks. A session that
        # starts before it finishes simply renders what it needs on demand.
        if not config.SHUTTING_DOWN.is_set():
            from modules.pipeline import prewarm_fixed_clips
            threading.Thread(target=prewarm_fixed_clips, name="clip-prewarm",
                             daemon=True).start()
        logger.info("Model pre-warm complete; first response will be fast.")
    except Exception as e:
        logger.warning("Model pre-warm failed (will lazy-load on demand): %s", e)


def _reap_idle_sessions():
    """Drop sessions whose clients have all been gone for the grace period.

    Without this, every browser that ever connects leaks a Pipeline and a VAD
    for the life of the process. The shared "default" session is exempt.

    Reaping deliberately ignores pipeline.state. Barge-ins and aborted turns can
    strand a session in "listening" or "speaking" indefinitely, and with no
    clients attached nobody is watching either way, so any in-flight response is
    simply cancelled first.
    """
    grace = config.SESSION_IDLE_TTL_SEC
    now = time.time()
    with _sessions_lock:
        for sid in list(sessions.keys()):
            if sid == "default":
                continue
            s = sessions[sid]
            if (not s.state_clients
                    and s.empty_since and now - s.empty_since > grace):
                s.pipeline.cancel_response()   # release GPU work before dropping it
                del sessions[sid]
                logger.info("Reaped idle session %s (total sessions: %d)", sid, len(sessions))


def _ensure_idle_media():
    """Make sure the clip the page loops between answers exists.

    In video mode there is nothing to do: the idle loop is the driving video,
    which must already exist for rendering to work at all.

    Voice-only needs a few seconds of silence, encoded once. The page's playback
    machinery is built around always having an idle item to loop, so supplying
    silence keeps that machinery identical in both modes instead of forking it.
    """
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
    """Periodically expire played-out audio, leaked scratch files and dead sessions.

    Renders delete their own scratch file, so the only ones reaching this sweep
    are those a crash leaked before the cleanup could run. Loops in a daemon
    thread for the life of the process.
    """
    ttl = config.TEMP_FILE_TTL_SEC
    interval = config.TEMP_CLEAN_INTERVAL_SEC
    exts = (".wav",)
    # Waiting on the event rather than sleeping lets Ctrl+C end this thread at
    # once, instead of killing it partway through a sweep.
    while not config.SHUTTING_DOWN.is_set():
        try:
            now = time.time()
            removed = 0
            try:
                names = os.listdir(config.RENDER_SCRATCH_DIR)
            except OSError:
                names = []            # nothing rendered yet, so nothing to sweep
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
    """Start the background workers on boot and stop them deterministically.

    Shutdown order matters for a clean Ctrl+C: the poller is cancelled (leaving
    it pending makes asyncio log "Task was destroyed but it is pending!") and
    SHUTTING_DOWN is set so the daemon threads, the MuseTalk render loop and the
    LLM producer finish their current step and return instead of being killed
    mid-work.

    Uses the lifespan API rather than on_startup/on_shutdown, which newer
    Starlette versions removed.
    """
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

        # Without a secure context the browser refuses microphone access on any
        # non-localhost origin, so missing certificates make a public deployment
        # useless rather than merely insecure. Warned about below.
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
        # Ctrl+C before uvicorn installs its own signal handlers — during
        # first-run idle-clip generation, say — would otherwise print a raw
        # traceback.
        config.SHUTTING_DOWN.set()
        logger.info("Interrupted during startup; exiting.")
        sys.exit(0)
