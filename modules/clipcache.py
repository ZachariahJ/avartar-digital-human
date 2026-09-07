"""In-memory store for everything the browser plays.

Two things live here:

  * blobs — the bytes behind a ``/media/<token>.wav`` URL. A dynamic sentence's
    audio used to be a file in ``tmp/`` that a janitor deleted after a TTL; it is
    now bytes with the same TTL, and the URL is the only handle anyone gets.
  * clips — a fully rendered FIXED utterance: its audio blob plus every JPEG
    frame, under the utterance's stable key. This replaces the
    ``assets/clips/<key>.frames/`` directory tree.

Why RAM. ``assets/clips`` sits on GPFS, and replaying one cached fixed clip
meant ~300 sequential ``open()`` calls across a network filesystem: 0.4-0.7s of
pure I/O before the segment could even be announced, all of it in front of the
user. Frames leave the GPU as bytes and now reach the browser as bytes.

The trade is deliberate and has a real cost: nothing survives a restart, so
``pipeline.prewarm_fixed_clips`` re-renders the protocol on every boot instead
of reading it back. That is why the pre-warm runs at idle and yields its GPU to
any live conversation, and why a cache miss during a conversation streams the
utterance (first frame out fast) rather than blocking on a full render.

Thread-safety: one lock over both maps, held only for dict work — never across
a render. Every stored value is immutable ``bytes``, so a caller reads what it
was handed without holding anything.
"""

import logging
import secrets
import threading
import time
from collections import OrderedDict

import config

logger = logging.getLogger(__name__)

# Every blob here is one utterance's audio and the TTS emits exactly one format
# — so the type is a constant, not a field. The extension on the URL is cosmetic
# (the Content-Type header is what the browser obeys) but it costs four
# characters and makes every log line readable.
MEDIA_TYPE = "audio/wav"
_MEDIA_EXT = ".wav"


class Blob:
    """Bytes reachable by URL. `expires_at` None means pinned: something (a
    cached clip) owns it and only that owner may release it."""

    __slots__ = ("data", "expires_at")

    def __init__(self, data: bytes, expires_at: float | None):
        self.data = data
        self.expires_at = expires_at


class Clip:
    """One fixed utterance, rendered: pinned audio blob + every frame in order.

    `stamp` is what the clip was rendered FROM (text, avatar fingerprint, fps —
    see pipeline.clip_stamp). A get() with a different stamp is a miss, not a
    stale hit, which is what stops an edited GREETING_TEXT from replaying the
    old wording out of a still-warm cache.
    """

    __slots__ = ("key", "stamp", "audio_token", "frames", "nbytes")

    def __init__(self, key: str, stamp: str, audio_token: str,
                 frames: tuple[bytes, ...], nbytes: int):
        self.key = key
        self.stamp = stamp
        self.audio_token = audio_token
        self.frames = frames
        self.nbytes = nbytes


_lock = threading.Lock()
_blobs: dict[str, Blob] = {}
# LRU: most recently used last, so eviction pops from the front.
_clips: "OrderedDict[str, Clip]" = OrderedDict()
_clip_bytes = 0


# --------------- blobs ---------------

def publish(data: bytes, ttl: float | None = -1.0) -> str:
    """Store `data` and return its token. Default TTL is MEDIA_BLOB_TTL_SEC;
    `ttl=None` pins it (the caller is then responsible for release())."""
    if ttl == -1.0:
        ttl = config.MEDIA_BLOB_TTL_SEC
    token = secrets.token_urlsafe(12)
    expires_at = None if ttl is None else time.monotonic() + ttl
    with _lock:
        _blobs[token] = Blob(data, expires_at)
    return token


def url_for(token: str) -> str:
    """The URL main.serve_media answers on for `token`."""
    return f"/media/{token}{_MEDIA_EXT}"


def publish_url(data: bytes, ttl: float | None = -1.0) -> str:
    """publish() + url_for() — what a caller that only wants a URL uses."""
    return url_for(publish(data, ttl))


def token_from_url(url: str) -> str:
    """The token inside a /media/<token>.<ext> URL (or bare filename). The URL
    is the only handle most of the code holds, so this is how it gets back to
    the blob — parsed in ONE place rather than re-derived at each call site."""
    return url.rsplit("/", 1)[-1].rsplit(".", 1)[0]


def fetch(token: str) -> bytes | None:
    """The bytes for a live token, or None. Serving does NOT extend a
    TTL: the window is measured from publication, when we know the utterance is
    about to be played once."""
    with _lock:
        blob = _blobs.get(token)
        if blob is None:
            return None
        if blob.expires_at is not None and blob.expires_at <= time.monotonic():
            del _blobs[token]
            return None
        return blob.data


def release(token: str) -> None:
    with _lock:
        _blobs.pop(token, None)


def sweep() -> int:
    """Drop expired unpinned blobs. Called by the janitor; returns the count."""
    now = time.monotonic()
    with _lock:
        dead = [t for t, b in _blobs.items()
                if b.expires_at is not None and b.expires_at <= now]
        for t in dead:
            del _blobs[t]
    return len(dead)


# --------------- fixed clips ---------------

def get_clip(key: str, stamp: str) -> Clip | None:
    """The cached clip for `key`, or None if absent or rendered from a
    different stamp (in which case the stale one is dropped here).

    The stale clip's audio blob is released OUTSIDE the lock: it is pinned, so
    nothing else would ever reclaim it, and release() takes the same
    non-reentrant lock this function is holding.
    """
    stale = None
    with _lock:
        clip = _clips.get(key)
        if clip is not None and clip.stamp != stamp:
            stale = _drop_locked(key)
            clip = None
        elif clip is not None:
            _clips.move_to_end(key)
    if stale is not None:
        release(stale)
        logger.info("[clipcache] dropped %s: it was rendered from a different "
                    "stamp (text, avatar or fps changed)", key)
    return clip


def has_clip(key: str, stamp: str) -> bool:
    return get_clip(key, stamp) is not None


def put_clip(key: str, stamp: str, audio: bytes, frames, token: str | None = None) -> Clip:
    """Cache one rendered fixed utterance, evicting LRU clips to stay under the
    byte cap. `frames` is empty in voice-only mode — an audio-only clip is a
    complete clip there, not a half-rendered one.

    `token` names a blob already holding this audio (the segment that was just
    played from it); the clip pins that one instead of storing the audio twice.
    """
    frames = tuple(frames or ())
    nbytes = len(audio) + sum(len(f) for f in frames)
    if not (token and pin(token)):
        token = publish(audio, ttl=None)   # pinned: the clip owns it
    clip = Clip(key, stamp, token, frames, nbytes)
    global _clip_bytes
    evicted = []
    with _lock:
        old = _clips.get(key)
        if old is not None:
            _clip_bytes -= old.nbytes
            evicted.append(old.audio_token)
            del _clips[key]
        _clips[key] = clip
        _clip_bytes += nbytes
        cap = config.CLIP_CACHE_MAX_MB * 1024 * 1024
        while _clip_bytes > cap and len(_clips) > 1:
            _, victim = _clips.popitem(last=False)
            _clip_bytes -= victim.nbytes
            evicted.append(victim.audio_token)
            logger.info("[clipcache] evicted %s (%.1f MB) to stay under %d MB",
                        victim.key, victim.nbytes / 1e6, config.CLIP_CACHE_MAX_MB)
        total, count = _clip_bytes, len(_clips)
    for t in evicted:
        release(t)
    logger.info("[clipcache] cached %s (%d frames, %.1f MB); %d clips, %.1f MB held",
                key, len(frames), nbytes / 1e6, count, total / 1e6)
    return clip


def _drop_locked(key: str) -> str | None:
    """Remove one clip. Caller holds _lock; releasing the blob is the caller's
    job AFTER dropping the lock (release() takes it again)."""
    global _clip_bytes
    clip = _clips.pop(key, None)
    if clip is None:
        return None
    _clip_bytes -= clip.nbytes
    return clip.audio_token


def stats() -> dict:
    """Cache occupancy, for logs and diagnostics."""
    with _lock:
        return {
            "clips": len(_clips),
            "clip_mb": round(_clip_bytes / 1e6, 1),
            "blobs": len(_blobs),
        }
