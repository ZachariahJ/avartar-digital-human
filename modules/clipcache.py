"""Everything the browser plays, held in memory rather than on disk.

Two kinds of entry:

  * blobs — bytes addressed by a token, which is what a ``/media/<token>`` URL
    resolves to. Dynamic sentences publish here with a TTL.
  * clips — one fully rendered fixed utterance under a stable key: its audio,
    pinned, plus every JPEG frame in order.

The reason for memory is latency. The previous on-disk cache lived on networked
GPFS, where replaying one cached utterance meant roughly 300 sequential opens,
costing 0.4-0.7s before playback could even be announced — all of it visible to
the user. Frames leave the GPU as bytes and now stay bytes the whole way.

The cost is that nothing survives a restart, so the fixed clips must be
re-rendered on every boot. That is why the pre-warm yields its GPU to live
conversations, and why a cache miss during a conversation streams the utterance
out as it renders instead of waiting for the whole thing.

Thread safety: one lock covers both maps and is held only for dictionary work,
never across a render. Stored values are immutable bytes, so a caller can use
what it was handed after releasing the lock.
"""

import logging
import secrets
import threading
import time
from collections import OrderedDict

import config

logger = logging.getLogger(__name__)

# Every blob is one utterance's audio and the TTS emits a single format, so this
# is a constant rather than per-blob state. The extension is cosmetic — browsers
# follow the Content-Type header — but it makes URLs readable in logs.
MEDIA_TYPE = "audio/mpeg"
_MEDIA_EXT = ".mp3"


class Blob:
    """Bytes reachable by token.

    An `expires_at` of None means pinned: some clip owns these bytes and only
    that owner may release them, so the sweeper must leave them alone.
    """

    __slots__ = ("data", "expires_at")

    def __init__(self, data: bytes, expires_at: float | None):
        self.data = data
        self.expires_at = expires_at


class Clip:
    """One fixed utterance, fully rendered: pinned audio plus frames in order.

    `stamp` describes everything the render depended on — the text, the avatar
    and the frame rate. Looking up with a different stamp misses rather than
    hitting, which is what stops edited wording or a swapped avatar from being
    replayed from a cache that is still warm.
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
# Ordered so eviction can pop the least recently used from the front.
_clips: "OrderedDict[str, Clip]" = OrderedDict()
_clip_bytes = 0


def publish(data: bytes, ttl: float | None = -1.0) -> str:
    """Store bytes and return the token that addresses them.

    Args:
        data: the bytes to store.
        ttl: seconds until expiry. The default sentinel means
            config.MEDIA_BLOB_TTL_SEC; None pins the blob, and the caller then
            owns it and must call release().

    Tokens are random, so a URL built from one cannot be guessed or enumerated
    into somebody else's audio.
    """
    if ttl == -1.0:
        ttl = config.MEDIA_BLOB_TTL_SEC
    token = secrets.token_urlsafe(12)
    expires_at = None if ttl is None else time.monotonic() + ttl
    with _lock:
        _blobs[token] = Blob(data, expires_at)
    return token


def url_for(token: str) -> str:
    """The URL main.serve_media answers for this token."""
    return f"/media/{token}{_MEDIA_EXT}"


def publish_url(data: bytes, ttl: float | None = -1.0) -> str:
    """Store bytes and return their URL, for callers that never need the token."""
    return url_for(publish(data, ttl))


def token_from_url(url: str) -> str:
    """Recover the token from a media URL or bare filename.

    Most callers only ever hold the URL, so the parsing lives here rather than
    being repeated — and getting it wrong at one call site would silently 404.
    """
    return url.rsplit("/", 1)[-1].rsplit(".", 1)[0]


def fetch(token: str) -> bytes | None:
    """The bytes for a live token, or None if it is unknown or expired.

    Serving does not extend the TTL. The window runs from publication, which is
    the point at which the utterance is known to be about to play once.
    """
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
    """Drop every expired unpinned blob and return how many went."""
    now = time.monotonic()
    with _lock:
        dead = [t for t, b in _blobs.items()
                if b.expires_at is not None and b.expires_at <= now]
        for t in dead:
            del _blobs[t]
    return len(dead)


def get_clip(key: str, stamp: str) -> Clip | None:
    """The cached clip for `key`, or None if absent or built from another stamp.

    A stamp mismatch drops the stale entry rather than returning it. Its audio
    is released after the lock is dropped: the blob is pinned, so nothing else
    will ever reclaim it, and release() would deadlock on this same
    non-reentrant lock.
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
    """Cache one rendered fixed utterance, evicting as needed to stay under cap.

    Args:
        key: stable identifier for the utterance.
        stamp: what it was rendered from; see get_clip.
        audio: the utterance's audio.
        frames: JPEG frames in order. Empty in voice-only mode, where an
            audio-only clip is complete rather than half rendered.
        token: an existing blob already holding this audio, typically the one
            the segment just played from. Given it, the clip pins that blob
            instead of storing a second copy.

    Returns:
        The cached Clip.
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
    """Remove one clip and return its audio token, or None if it was absent.

    The caller must hold _lock, and must release the returned token only after
    dropping it — release() takes the same non-reentrant lock.
    """
    global _clip_bytes
    clip = _clips.pop(key, None)
    if clip is None:
        return None
    _clip_bytes -= clip.nbytes
    return clip.audio_token


def stats() -> dict:
    """Current occupancy: clip count, clip megabytes and live blob count."""
    with _lock:
        return {
            "clips": len(_clips),
            "clip_mb": round(_clip_bytes / 1e6, 1),
            "blobs": len(_blobs),
        }
