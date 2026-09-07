"""HTTP client for the local GPT-SoVITS speech synthesis server.

GPT-SoVITS is a separate process rather than an import because it needs torch
2.14/cu126 while MuseTalk pins 2.0.1/cu118, and one virtualenv cannot satisfy
both. Keeping it behind HTTP also means it can be restarted, moved to another
host or have its model replaced without disturbing the avatar.

The voice is cloned zero-shot from the reference pair in config, so changing how
the counselor sounds is a matter of replacing those two files rather than
touching this module.
"""

import logging
import threading

import requests

import config

logger = logging.getLogger(__name__)

# Reused so that keep-alive spares each sentence a TCP handshake.
_session = requests.Session()


def synthesize(text: str,
               cancel_event: threading.Event | None = None) -> bytes | None:
    """Speak `text`. Returns a complete wav, or None if cancelled or failed.

    Args:
        text: one sentence. The caller has already split on sentence-final
            punctuation, and the server is told not to split it further.
        cancel_event: checked once, before the request is sent, so a turn that
            was already interrupted does not start new work.

    Returns:
        wav bytes, or None. Never raises: a caller that gets None keeps the text
        on screen and continues the turn without audio.

    Returns bytes rather than a path because nothing the browser plays is
    written to disk; the audio is published to modules.clipcache and served from
    memory. wav rather than a compressed format because there is no bandwidth to
    win — the clip travels over the link already carrying JPEG frames — and
    uncompressed audio saves MuseTalk a decode.

    Cancellation cannot interrupt a synthesis already under way: the server
    produces the whole utterance in one call, so an interruption mid-synthesis
    still pays for that work and the next sentence waits behind it, which for a
    long sentence has been measured at up to ~3.8s.
    """
    if cancel_event and cancel_event.is_set():
        return None

    try:
        r = _session.post(
            f"{config.TTS_SERVER_URL}/tts",
            json={
                "text": text,
                "text_lang": config.TTS_LANG,
                "ref_audio_path": config.TTS_REF_AUDIO,
                "prompt_text": config.TTS_REF_TEXT,
                "prompt_lang": config.TTS_LANG,
                # cut0 means "do not split". The default methods break on
                # internal punctuation and leave an audible gap at every seam.
                "text_split_method": "cut0",
            },
            timeout=config.TTS_TIMEOUT_SEC,
        )
    except requests.RequestException as e:
        logger.warning("TTS request failed (%s). Is the GPT-SoVITS server up? "
                       "scripts/tts_server.sh", e)
        return None

    if r.status_code != 200:
        logger.warning("TTS server returned %s: %s", r.status_code, r.text[:200])
        return None
    return r.content


def healthy() -> bool:
    """True if the TTS server answers at all.

    Any HTTP status counts as healthy. A parameterless GET returns 500, and that
    500 is itself proof something is listening. The only distinction being drawn
    is "nothing on that port" versus "server running"; confirming the model
    actually synthesizes would cost a GPU second on every startup.
    """
    try:
        _session.get(f"{config.TTS_SERVER_URL}/tts", timeout=5)
        return True
    except requests.RequestException:
        return False
