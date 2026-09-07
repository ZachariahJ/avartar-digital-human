"""Text to speech via a LOCAL GPT-SoVITS server.

Why a separate process rather than an import: GPT-SoVITS needs torch 2.14/cu126
and MuseTalk pins 2.0.1/cu118, so one virtualenv cannot hold both. GPT-SoVITS
runs as its own service (scripts/tts_server.sh) on its own GPU and this module
is just its HTTP client — which also means TTS can be restarted, moved to
another host or have its model swapped without touching the avatar.

Why the response is streamed rather than fetched whole: `cancel_event` is polled
between chunks, so a barge-in STOPS synthesis instead of waiting it out. That is
the same contract the old edge-tts path had, for the same reason — the flush
cancels the LLM and MuseTalk, and TTS must not be the one stage that keeps
running to completion for audio nobody will hear.

The voice is cloned zero-shot from assets/voice/reference.wav, which was
generated with the previous edge-tts voice: the migration to a local model
therefore does NOT change how the counselor sounds. Swap that pair of files
(the wav and its verbatim transcript) to change the voice.
"""

import logging
import struct
import threading

import requests

import config

logger = logging.getLogger(__name__)

# One session: keep-alive means a synthesis does not pay TCP setup per sentence.
_session = requests.Session()


def _finalize_wav(buf: bytes) -> bytes | None:
    """Patch the RIFF/data lengths of a STREAMED wav.

    The server emits its header before it knows how long the utterance will be,
    so both size fields arrive as zero (api_v2.wave_header_chunk writes an empty
    frame). Browsers and librosa mostly cope by reading to EOF, but "mostly" is
    not a contract — a zero-length wav is a decode error waiting for the one
    client that reads the header instead of the file. We know the real length
    once the stream ends, so we write it in.
    """
    if len(buf) < 44 or buf[:4] != b"RIFF" or buf[8:12] != b"WAVE":
        return None
    data_at = buf.find(b"data", 12)
    if data_at < 0:
        return None
    out = bytearray(buf)
    struct.pack_into("<I", out, 4, len(out) - 8)                  # RIFF size
    struct.pack_into("<I", out, data_at + 4, len(out) - data_at - 8)  # data size
    return bytes(out)


def synthesize(text: str,
               cancel_event: threading.Event | None = None) -> bytes | None:
    """Synthesize `text`. Returns a complete wav, or None if cancelled/failed.

    Bytes, not a path: nothing the browser plays is written to disk. The audio
    is published to modules.clipcache and served from RAM (see
    pipeline.render_into); the only file it becomes is the short-lived scratch
    copy MuseTalk needs, because its feature extraction takes a filename.

    wav rather than a compressed format because there is nothing to gain by
    compressing it: the clip is fetched once over the same link that is already
    carrying the JPEG frame stream, and PCM saves MuseTalk's librosa a decode.

    A failure here degrades to "skip this clip" (None), never an exception — the
    caller keeps the text on screen and the turn continues.
    """
    if cancel_event and cancel_event.is_set():
        return None

    payload = {
        "text": text,
        "text_lang": config.TTS_LANG,
        "ref_audio_path": config.TTS_REF_AUDIO,
        "prompt_text": config.TTS_REF_TEXT,
        "prompt_lang": config.TTS_LANG,
        "media_type": "wav",
        # 1 = fragment-by-fragment at FULL quality. The point is not to deliver
        # audio early (the caller needs the whole utterance before MuseTalk can
        # start) but to get a cancellation point per fragment.
        "streaming_mode": 1,
        # The caller already hands us exactly one sentence (llm.chat_stream
        # splits on sentence-final punctuation), so the server must not split it
        # again — cut0 is "no split".
        "text_split_method": "cut0",
        "batch_size": config.TTS_BATCH_SIZE,
        "speed_factor": config.TTS_SPEED,
        "parallel_infer": True,
    }

    try:
        with _session.post(f"{config.TTS_SERVER_URL}/tts", json=payload,
                           stream=True, timeout=config.TTS_TIMEOUT_SEC) as r:
            if r.status_code != 200:
                logger.warning("TTS server returned %s: %s",
                               r.status_code, r.text[:200])
                return None
            buf = bytearray()
            for chunk in r.iter_content(chunk_size=None):
                if cancel_event is not None and cancel_event.is_set():
                    return None       # barge-in: drop the connection mid-stream
                buf += chunk
    except requests.RequestException as e:
        logger.warning("TTS request failed (%s). Is the GPT-SoVITS server up? "
                       "scripts/tts_server.sh", e)
        return None

    wav = _finalize_wav(bytes(buf))
    if wav is None:
        logger.warning("TTS returned %d bytes that are not a wav", len(buf))
    return wav


def healthy() -> bool:
    """True if the TTS server answers HTTP at all.

    ANY status counts — a parameterless GET /tts is a 500, and that 500 is
    itself proof the service is up. This only has to separate "nobody is
    listening on that port" from "the model is loaded and serving"; a real
    synthesis is the only thing that can tell you more, and the startup warm-up
    is not the place to spend a GPU second on one.
    """
    try:
        _session.get(f"{config.TTS_SERVER_URL}/tts", timeout=5)
        return True
    except requests.RequestException:
        return False


if __name__ == "__main__":
    import sys, time
    logging.basicConfig(level=logging.INFO)
    t = time.perf_counter()
    audio = synthesize(sys.argv[1] if len(sys.argv) > 1
                       else "Hello, how are you feeling today?")
    dt = time.perf_counter() - t
    print(f"{len(audio) if audio else 0} bytes in {dt*1000:.0f} ms")
