import asyncio
import tempfile
import threading
import edge_tts
import config


async def _synthesize(text: str, voice: str, output_path: str) -> str:
    communicate = edge_tts.Communicate(text, voice)
    await communicate.save(output_path)
    return output_path


def synthesize(text: str, output_path: str | None = None,
               cancel_event: threading.Event | None = None) -> str | None:
    """Synthesize text to speech. Returns path to an mp3, or None if cancelled.

    The suffix is .mp3 because edge-tts emits MPEG layer III, full stop — the
    voice is 24kHz mono at 48kbit/s. It used to be named .wav, which was
    harmless only as long as the sole consumer was ffmpeg (it sniffs content and
    ignores the name). In voice-only mode (config.ENABLE_VIDEO_AVATAR=0) this
    file is served straight to the browser, which honours the extension and the
    Content-Type derived from it — a .wav name there means a decode error.
    """
    if cancel_event and cancel_event.is_set():
        return None

    if output_path is None:
        output_path = tempfile.mktemp(suffix=".mp3", dir=config.TEMP_DIR)
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    # edge-tts hits a remote Microsoft endpoint per sentence; a transient failure
    # must degrade to "skip this clip" (return None), never raise and freeze the turn.
    try:
        if loop and loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                pool.submit(asyncio.run, _synthesize(text, config.TTS_VOICE, output_path)).result()
        else:
            asyncio.run(_synthesize(text, config.TTS_VOICE, output_path))
    except Exception:
        return None
    return output_path


if __name__ == "__main__":
    import os
    os.makedirs(config.TEMP_DIR, exist_ok=True)
    path = synthesize("Hello, I am your AI assistant, nice to meet you!")
    print(f"TTS output saved to: {path}")
