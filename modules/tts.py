import asyncio
import os
import tempfile
import threading
import edge_tts
import config


async def _synthesize(text: str, voice: str, output_path: str,
                      cancel_event: threading.Event | None = None) -> bool:
    """Stream the synthesis to disk, abandoning it the moment `cancel_event` fires.

    communicate.save() is deliberately NOT used. It runs the whole request to
    completion with no way in, so a barge-in could not stop TTS — the flush would
    cancel the LLM and MuseTalk and then sit waiting for audio nobody would hear.
    Iterating .stream() gives a cancellation point per chunk.

    Returns True if the file is complete, False if it was abandoned mid-stream
    (the caller deletes the partial file: half an utterance is worse than none).
    """
    communicate = edge_tts.Communicate(text, voice)
    with open(output_path, "wb") as f:
        async for chunk in communicate.stream():
            if cancel_event is not None and cancel_event.is_set():
                return False
            if chunk["type"] == "audio":
                f.write(chunk["data"])
    return True


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
    def _run():
        return asyncio.run(_synthesize(text, config.TTS_VOICE, output_path, cancel_event))

    try:
        if loop and loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                done = pool.submit(_run).result()
        else:
            done = _run()
    except Exception:
        done = False
    if not done:
        # Cancelled or failed: drop the partial mp3 so nothing downstream can
        # pick it up and render half a sentence.
        try:
            os.remove(output_path)
        except OSError:
            pass
        return None
    return output_path


if __name__ == "__main__":
    os.makedirs(config.TEMP_DIR, exist_ok=True)
    path = synthesize("Hello, I am your AI assistant, nice to meet you!")
    print(f"TTS output saved to: {path}")
