"""Text to speech (edge-tts), in memory.

Returns bytes, never a path: nothing the browser plays is written to disk. The
audio is published to modules.clipcache and served from RAM (see
pipeline.render_into); the only file it ever becomes is the short-lived scratch
copy MuseTalk needs, because its feature extraction takes a filename.
"""

import asyncio
import io
import threading

import edge_tts

import config


async def _synthesize(text: str, voice: str, rate: str,
                      cancel_event: threading.Event | None = None) -> bytes | None:
    """Stream the synthesis into memory, abandoning it the moment `cancel_event` fires.

    communicate.save() is deliberately NOT used. It runs the whole request to
    completion with no way in, so a barge-in could not stop TTS — the flush would
    cancel the LLM and MuseTalk and then sit waiting for audio nobody would hear.
    Iterating .stream() gives a cancellation point per chunk.

    Returns the complete mp3, or None if it was abandoned mid-stream (half an
    utterance is worse than none).
    """
    communicate = edge_tts.Communicate(text, voice, rate=rate)
    buf = io.BytesIO()
    async for chunk in communicate.stream():
        if cancel_event is not None and cancel_event.is_set():
            return None
        if chunk["type"] == "audio":
            buf.write(chunk["data"])
    return buf.getvalue()


def synthesize(text: str,
               cancel_event: threading.Event | None = None) -> bytes | None:
    """Synthesize text to speech. Returns mp3 bytes, or None if cancelled/failed.

    The format is MPEG layer III, full stop — edge-tts emits 24kHz mono at
    48kbit/s. That is why the blob is published as audio/mpeg: in voice-only
    mode these bytes go straight to the browser, which honours the Content-Type,
    and a wrong label there is a decode error rather than a silent mislabel.
    """
    if cancel_event and cancel_event.is_set():
        return None

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    # edge-tts hits a remote Microsoft endpoint per sentence; a transient failure
    # must degrade to "skip this clip" (return None), never raise and freeze the turn.
    def _run():
        return asyncio.run(_synthesize(text, config.TTS_VOICE, config.TTS_RATE,
                                       cancel_event))

    try:
        if loop and loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                return pool.submit(_run).result()
        return _run()
    except Exception:
        return None


if __name__ == "__main__":
    audio = synthesize("Hello, I am your AI assistant, nice to meet you!")
    print(f"TTS produced {len(audio) if audio else 0} bytes of mp3")
