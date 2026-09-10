
import logging
import secrets
import threading
import time
from collections import OrderedDict

import config

logger = logging.getLogger(__name__)

MEDIA_TYPE = "audio/mpeg"
_MEDIA_EXT = ".mp3"


class Blob:

    __slots__ = ("data", "expires_at")

    def __init__(self, data: bytes, expires_at: float):
        self.data = data
        self.expires_at = expires_at


class Clip:

    __slots__ = ("key", "stamp", "audio", "frames", "nbytes")

    def __init__(self, key: str, stamp: str, audio: bytes,
                 frames: tuple[bytes, ...], nbytes: int):
        self.key = key
        self.stamp = stamp
        self.audio = audio
        self.frames = frames
        self.nbytes = nbytes


_lock = threading.Lock()
_blobs: dict[str, Blob] = {}
_clips: "OrderedDict[str, Clip]" = OrderedDict()
_clip_bytes = 0


def publish(data: bytes, ttl: float | None = None) -> str:
    if ttl is None:
        ttl = config.MEDIA_BLOB_TTL_SEC
    token = secrets.token_urlsafe(12)
    with _lock:
        _blobs[token] = Blob(data, time.monotonic() + ttl)
    return token


def url_for(token: str) -> str:
    return f"/media/{token}{_MEDIA_EXT}"


def publish_url(data: bytes, ttl: float | None = None) -> str:
    return url_for(publish(data, ttl))


def token_from_url(url: str) -> str:
    return url.rsplit("/", 1)[-1].rsplit(".", 1)[0]


def fetch(token: str) -> bytes | None:
    with _lock:
        blob = _blobs.get(token)
        if blob is None:
            return None
        if blob.expires_at <= time.monotonic():
            del _blobs[token]
            return None
        return blob.data


def sweep() -> int:
    now = time.monotonic()
    with _lock:
        dead = [t for t, b in _blobs.items() if b.expires_at <= now]
        for t in dead:
            del _blobs[t]
    return len(dead)


def get_clip(key: str, stamp: str) -> Clip | None:
    stale = False
    with _lock:
        clip = _clips.get(key)
        if clip is not None and clip.stamp != stamp:
            _drop_locked(key)
            clip, stale = None, True
        elif clip is not None:
            _clips.move_to_end(key)
    if stale:
        logger.info("[clipcache] dropped %s: it was rendered from a different "
                    "stamp (text, avatar or fps changed)", key)
    return clip


def has_clip(key: str, stamp: str) -> bool:
    return get_clip(key, stamp) is not None


def put_clip(key: str, stamp: str, audio: bytes, frames) -> Clip:
    frames = tuple(frames or ())
    nbytes = len(audio) + sum(len(f) for f in frames)
    clip = Clip(key, stamp, audio, frames, nbytes)
    global _clip_bytes
    with _lock:
        old = _clips.get(key)
        if old is not None:
            _clip_bytes -= old.nbytes
            del _clips[key]
        _clips[key] = clip
        _clip_bytes += nbytes
        cap = config.CLIP_CACHE_MAX_MB * 1024 * 1024
        while _clip_bytes > cap and len(_clips) > 1:
            _, victim = _clips.popitem(last=False)
            _clip_bytes -= victim.nbytes
            logger.info("[clipcache] evicted %s (%.1f MB) to stay under %d MB",
                        victim.key, victim.nbytes / 1e6, config.CLIP_CACHE_MAX_MB)
        total, count = _clip_bytes, len(_clips)
    logger.info("[clipcache] cached %s (%d frames, %.1f MB); %d clips, %.1f MB held",
                key, len(frames), nbytes / 1e6, count, total / 1e6)
    return clip


def _drop_locked(key: str) -> None:
    global _clip_bytes
    clip = _clips.pop(key, None)
    if clip is not None:
        _clip_bytes -= clip.nbytes


def stats() -> dict:
    with _lock:
        return {
            "clips": len(_clips),
            "clip_mb": round(_clip_bytes / 1e6, 1),
            "blobs": len(_blobs),
        }
