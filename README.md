# Digital Human Chatbot

Real-time digital human dialogue system: browser mic capture → VAD segmentation → ASR → streaming LLM → sentence-level TTS → FLOAT talking-head synthesis → video stream playback. Supports user barge-in (interrupting the avatar mid-response).

## Architecture

```
Browser mic ─WS(17862)─▶ VAD ─speech_end─▶ Pipeline
Pipeline: ASR ─▶ LLM(stream) ─▶ split into sentences ─▶ TTS ─▶ FLOAT ─▶ video queue
WS(/ws/state) pushes the next video segment and subtitle to the frontend
Barge-in: VAD speech_start ─▶ cancel_event ─▶ clear queue
```

Server: Starlette + uvicorn (single port 17861, HTTP + two WebSocket routes).
Frontend: `static/index.html`, plain HTML/JS, no build step.

## Layout

```
digital-human/
├── main.py                  # Starlette entrypoint (HTTP + WS)
├── config.py                # All configuration (GPU allocation, models, prompt, ...)
├── modules/
│   ├── pipeline.py          # Orchestration: ASR → LLM → TTS → FLOAT, with barge-in
│   ├── vad.py               # Silero VAD real-time segmentation
│   ├── asr.py               # FunASR (SenseVoiceSmall) on GPU 3
│   ├── llm.py               # OpenRouter (gemini-2.5-flash) streaming
│   ├── tts.py               # edge-tts (en-US-GuyNeural)
│   ├── avatar.py            # FLOAT multi-GPU pool (0/1/2) + idle video
│   ├── audio_server.py      # Browser audio WS handler
│   └── sbirt/               # SBIRT clinical framework (data-driven, config-free)
│       ├── instruments.py   #   validated screening tools (AUDIT, DAST-10, CAGE-AID, CRAFFT, ...)
│       ├── intervention.py  #   MI/OARS, FRAMES, stages of change, readiness rulers
│       ├── referral.py      #   ASAM levels of care, MAT, resources, crisis protocol
│       ├── workflow.py      #   the SBIRT conversation state machine
│       └── prompt.py        #   build_system_prompt(): renders ALL of the above into the prompt
├── static/
│   ├── index.html           # Frontend UI
│   └── test_audio.html      # Mic / ASR debug page
├── assets/
│   ├── avatar.png           # Avatar portrait (FLOAT input)
│   └── idle_loop.mp4        # Idle loop video (auto-generated on first run)
└── requirements.txt
```

`tmp/` (runtime-generated audio/video) and `.env` (API key) are listed in `.gitignore` and are not tracked.

## Dependencies

- Upstream model repo: [FLOAT](https://github.com/deepbrainai-research/float). Place it in a sibling directory at `../float/`, with the checkpoint at `../float/checkpoints/float.pth` (run `download_checkpoints.sh` to fetch it).
- Python: conda env `float` (Python 3.10), PyTorch 2.6.0+cu118, transformers 4.49, numpy<2.
- Hardware: 4× A100 80GB (GPUs 0/1/2 for FLOAT parallel rendering, GPU 3 for ASR).
- ffmpeg (already installed inside the `float` conda env).

```bash
pip install -r requirements.txt
```

## Configuration

Copy `.env.example` to `.env` and fill in your OpenRouter API key:

```
OPENROUTER_API_KEY=sk-or-v1-...
```

All other settings live in `config.py`. Common knobs:

| Setting | Default | Description |
|---|---|---|
| `LLM_MODEL` | `google/gemini-2.5-flash` | OpenRouter model |
| `TTS_VOICE` | `en-US-GuyNeural` | edge-tts voice |
| `ASR_MODEL` | `iic/SenseVoiceSmall` | FunASR model |
| `FLOAT_GPUS` | `[0, 1, 2]` | GPUs used by FLOAT in parallel |
| `ASR_GPU` | `3` | Dedicated GPU for ASR |
| `FLOAT_NFE` | `10` | FLOAT inference steps (lower = faster) |
| `VAD_THRESHOLD` | `0.5` | Silero VAD trigger threshold |
| `VAD_SILENCE_DURATION` | `0.5` | Seconds of silence to mark sentence end |
| `SYSTEM_PROMPT` | built from `modules/sbirt/` | SBIRT counselor prompt — **do not edit the string**; edit the clinical data modules under `modules/sbirt/` and it rebuilds automatically |

## Running

```bash
source activate float
cd /path/to/digital-human
python main.py
```

On startup:

- The first run auto-generates `assets/idle_loop.mp4` (~1 minute).
- Each FLOAT GPU loads the model in roughly 60s.
- Open `http://<host>:17861/` in a browser, grant microphone permission, click Mic to start a conversation.

## Endpoints

| Path | Description |
|---|---|
| `GET /` | Frontend page |
| `GET /video/{tmp,assets}/{file}` | Static video serving |
| `POST /api/toggle` | Toggle the microphone on/off |
| `POST /api/reset` | Clear conversation and video queue |
| `POST /api/text` | Text-input fallback (debug without a mic) |
| `POST /api/test_asr` | Upload PCM and test ASR end-to-end |
| `WS /ws/audio` | Browser pushes 16 kHz int16 PCM here |
| `WS /ws/state` | Server pushes video segments and state changes |

## Known Pitfalls

- Port 7860/7861 are often occupied; this app defaults to 17861.
- `transformers` cannot be upgraded to 5.x — FLOAT is incompatible.
- `numpy` must be <2, `opencv-python` must be <4.11.
- `edge-tts` is async; when calling from a synchronous thread you need `asyncio.run` or a thread pool. See `modules/tts.py`.
- On first launch, `assets/idle_loop.mp4` is auto-generated if missing — this requires the FLOAT checkpoint to already be in place.
