
import asyncio
import io
import threading

import edge_tts

import config


async def _synthesize(text: str, voice: str, rate: str,
                      cancel_event: threading.Event | None = None) -> bytes | None:
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
    if cancel_event and cancel_event.is_set():
        return None

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

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
