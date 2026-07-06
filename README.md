# Digital Human SBIRT Screener

Real-time digital human for SBIRT alcohol/drug screening: browser mic capture
→ VAD segmentation (+ smart-turn EOU) → ASR → **deterministic clinical state
machine** → cached fixed clips / bounded LLM utterances → FLOAT talking-head
video. Supports user barge-in (interrupting the avatar mid-response).

## Architecture

```
Browser mic ─WS /ws/audio─▶ VAD(+EOU) ─speech_end─▶ Pipeline
Pipeline: ASR ─▶ crisis net (deterministic) ─▶ NLU coder (option/number/consent)
          ─▶ clinical state machine (modules/sbirt/runtime.py)
          ─▶ fixed utterances: pre-rendered cached clips (assets/clips/)
             LLM utterances: bounded single-sentence phrasing ─▶ TTS ─▶ FLOAT
WS(/ws/state) pushes the next video segment and subtitle to the frontend
Barge-in: VAD/ASR speech_start ─▶ cancel_event + turn bump ─▶ clear queue
```

**Deterministic clinical core** (`modules/sbirt/`): AUDIT/DAST scores, risk
zones, question order/skip rules, arm routing and zone feedback are computed
by code (`instruments.py` + `runtime.py` + `templates.py`),
never by the LLM. The LLM's jobs are narrow: coding free-text answers onto
option codes (with AMBIGUOUS → clarification, never guessing), and phrasing
single bounded utterances (reflections/summaries). A deterministic crisis
keyword net (`crisis.py`) runs before everything and cannot be down. Consent
decisions land in an append-only audit log (`records/`). See
`DECISIONS_FOR_REVIEW.md` for open clinical/compliance decision points.

Server: Starlette + uvicorn (single port 17861, HTTP + two WebSocket routes).
Frontend: `static/index.html`, plain HTML/JS, no build step.

## Layout

```
digital-human/
├── main.py                  # Starlette entrypoint (HTTP + WS), per-sid sessions
├── config.py                # All configuration (GPU allocation, models, prompt, ...)
├── DECISIONS_FOR_REVIEW.md  # Open clinical/compliance decision points
├── modules/
│   ├── pipeline.py          # Orchestration: ASR → coder → machine → clips/TTS/FLOAT
│   ├── vad.py               # Silero VAD real-time segmentation
│   ├── eou.py               # smart-turn v3 semantic end-of-utterance (CPU ONNX)
│   ├── asr.py               # FunASR (SenseVoiceSmall)
│   ├── llm.py               # OpenRouter: NLU coding + bounded utterances (+ crisis chat)
│   ├── tts.py               # edge-tts (en-US-GuyNeural)
│   ├── avatar.py            # FLOAT GPU pool + idle video
│   ├── privacy.py           # PHI-safe logging (no opt-out) + consent audit trail
│   └── sbirt/               # SBIRT clinical framework — the deterministic core
│       ├── instruments.py   #   structured tools + deterministic scoring in one file
│       ├── runtime.py       #   the executable clinical state machine
│       ├── templates.py     #   versioned verbatim script (questions/feedback/BI)
│       ├── crisis.py        #   deterministic crisis keyword net + fixed responses
│       ├── intervention.py  #   MI/OARS, FRAMES, stages of change (prompt data)
│       ├── referral.py      #   ASAM levels, MAT, resources, crisis protocol (prompt data)
│       ├── workflow.py      #   conceptual SBIRT node map (prompt data)
│       └── prompt.py        #   build_system_prompt() — now used for crisis turns
├── tests/                   # pytest: gold-standard case cards + integration
├── static/
│   ├── index.html           # Frontend UI
├── assets/
│   ├── avatar.png           # Avatar portrait (FLOAT input)
│   ├── clips/               # Pre-rendered fixed protocol clips (auto-generated)
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
| `LLM_MODEL` | `google/gemini-2.5-flash` | OpenRouter model (NLU coding + bounded utterances) |
| `TTS_VOICE` | `en-US-GuyNeural` | edge-tts voice |
| `ASR_MODEL` | `iic/SenseVoiceSmall` | FunASR model |
| `FLOAT_GPUS` | `[2]` | GPUs used by FLOAT (list; parallel across entries) |
| `ASR_GPU` | `2` | GPU for ASR (currently shared with FLOAT) |
| `FLOAT_NFE` | `10` | FLOAT inference steps (lower = faster) |
| `VAD_THRESHOLD` | `0.5` | Silero VAD trigger threshold |
| `VAD_SILENCE_DURATION` | `0.35` | Seconds of silence to mark sentence end |
| `CONSENT_LOG_PATH` | `records/consent_log.jsonl` | Consent audit trail |
| `SYSTEM_PROMPT` | built from `modules/sbirt/` | Crisis-turn counselor prompt — **do not edit the string**; edit the clinical data modules under `modules/sbirt/` and it rebuilds automatically |

Clinical content (questions, options, scores, zones, feedback wording) lives in
`modules/sbirt/instruments.py` + `templates.py`, sourced verbatim from the
study documents in `SBIRT_Reference/`. Changing any fixed text bumps the clip
cache automatically (sidecar text check) — bump `templates.SCRIPT_VERSION` too.

## Tests

```bash
# Deterministic core (no GPU/network needed):
python -m pytest tests/ -q
# Full suite incl. real-Pipeline integration (needs the float env's deps):
~/.conda/envs/float/bin/python -m pytest tests/ -q
```

Gold-standard fixtures under `tests/fixtures/` are hand-scored from the case
cards in `SBIRT_Reference/` — fix code, never fixtures.

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
