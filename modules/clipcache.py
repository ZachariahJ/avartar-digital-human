"""Everything the browser plays, held in memory rather than on disk.

Two structures, answering two different questions:

  * blobs — bytes addressed by a token, which is what a ``/media/<token>`` URL
    resolves to. This is delivery, not caching: the browser can only take audio
    through a URL, so every utterance about to play publishes one, and it is
    swept once its TTL has passed. Nothing is ever reused from here.
  * clips — the cache proper. One fully rendered fixed utterance under a stable
    key: its audio and every JPEG frame in order, so that a line the protocol
    speaks to every patient is rendered once per process rather than once per
    patient.

A clip holds its own bytes and publishes a fresh blob each time it is played.
That keeps the dependency one-way — clips use blobs, blobs know nothing of
clips — and it is why blobs need only one lifetime rule. An earlier design had
the clip take ownership of the blob its first render published, to save a copy
of the audio; that copy never existed (bytes are shared by reference, so a
second publish costs one dict entry), and the ownership it introduced meant
evicting a clip could delete audio a segment was still playing from.

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
    """Bytes reachable by token until `expires_at`, then swept."""

    __slots__ = ("data", "expires_at")

    def __init__(self, data: bytes, expires_at: float):
        self.data = data
        self.expires_at = expires_at


class Clip:
    """One fixed utterance, fully rendered: its audio plus frames in order.

    `stamp` describes everything the render depended on — the text, the avatar
    and the frame rate. Looking up with a different stamp misses rather than
    hitting, which is what stops edited wording or a swapped avatar from being
    replayed from a cache that is still warm.

    The audio is held here as bytes, not as a blob token: a clip outlives any
    single delivery, so it cannot depend on a blob that expires.
    """

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
# Ordered so eviction can pop the least recently used from the front.
_clips: "OrderedDict[str, Clip]" = OrderedDict()
_clip_bytes = 0


def publish(data: bytes, ttl: float | None = None) -> str:
    """Store bytes for delivery and return the token that addresses them.

    Args:
        data: the bytes to store. They are referenced, not copied, so
            publishing the same audio twice costs one dict entry.
        ttl: seconds until expiry; None means config.MEDIA_BLOB_TTL_SEC. This
            is the window in which the browser must fetch the URL, not a
            retention policy — nothing is ever served from here twice.

    Tokens are random, so a URL built from one cannot be guessed or enumerated
    into somebody else's audio.
    """
    if ttl is None:
        ttl = config.MEDIA_BLOB_TTL_SEC
    token = secrets.token_urlsafe(12)
    with _lock:
        _blobs[token] = Blob(data, time.monotonic() + ttl)
    return token


def url_for(token: str) -> str:
    """The URL main.serve_media answers for this token."""
    return f"/media/{token}{_MEDIA_EXT}"


def publish_url(data: bytes, ttl: float | None = None) -> str:
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
        if blob.expires_at <= time.monotonic():
            del _blobs[token]
            return None
        return blob.data


def sweep() -> int:
    """Drop every expired blob and return how many went."""
    now = time.monotonic()
    with _lock:
        dead = [t for t, b in _blobs.items() if b.expires_at <= now]
        for t in dead:
            del _blobs[t]
    return len(dead)


def get_clip(key: str, stamp: str) -> Clip | None:
    """The cached clip for `key`, or None if absent or built from another stamp.

    A stamp mismatch drops the stale entry rather than returning it.
    """
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
    """Cache one rendered fixed utterance, evicting as needed to stay under cap.

    Args:
        key: stable identifier for the utterance.
        stamp: what it was rendered from; see get_clip.
        audio: the utterance's audio. Held by reference; a blob published from
            the same bytes is unaffected by this clip's eviction.
        frames: JPEG frames in order. Empty in voice-only mode, where an
            audio-only clip is complete rather than half rendered.

    Returns:
        The cached Clip.
    """
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
    """Remove one clip. The caller must hold _lock."""
    global _clip_bytes
    clip = _clips.pop(key, None)
    if clip is not None:
        _clip_bytes -= clip.nbytes


def stats() -> dict:
    """Current occupancy: clip count, clip megabytes and live blob count."""
    with _lock:
        return {
            "clips": len(_clips),
            "clip_mb": round(_clip_bytes / 1e6, 1),
            "blobs": len(_blobs),
        }
