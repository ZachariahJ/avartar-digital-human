# Digital Human SBIRT Screener

Real-time digital human for SBIRT alcohol/drug screening: browser mic capture
→ VAD segmentation (+ smart-turn EOU) → ASR → **generic turn engine over a
declarative protocol program** (deterministic control) + **one LLM call per
turn** (understanding + this turn's voice) → cached fixed clips / dynamic
short renders → FLOAT talking-head video. Supports user barge-in
(interrupting the avatar mid-response).

## Architecture

The split that everything else hangs off: **the deterministic spine decides
WHAT happens next; the LLM only decides HOW this one turn sounds.**

```
Browser mic ─WS /ws/audio─▶ VAD(+EOU) ─speech_end─▶ Pipeline
Pipeline: ASR ─▶ crisis net (deterministic, runs first)
          ─▶ llm.turn() — ONE call: classify the utterance relative to the
             current ask (answer/continuation/question/tangent/crisis/
             abort/correction/unclear), code it if it's an answer, produce
             this turn's bounded reply
          ─▶ turn.validate — pure gatekeeper: an illegal answer degrades to
             'unclear'; ONLY a validated answer may advance the engine
          ─▶ generic engine (runtime.py) executing flow.PROTOCOL:
               answer        → advance; speak ack + next step's beats
                               (confirm-flagged items: read back the CODED
                               option first; commit only on the yes)
               continuation  → absorb into the previous capture; HOLD
               question      → answer THEM from state facts; re-ask; HOLD
               tangent       → acknowledge, return to the ask; HOLD
               correction    → overwrite the earlier item; re-derive skips
                               and score; audit old→new; re-pose the ask
               abort         → graceful close, partial data kept, mic off
               crisis        → pause the protocol permanently (union w/ net)
               unclear       → clarify toward the candidate options; HOLD
          ─▶ beats: Say  = verbatim study text → pre-rendered cached clip
                    Speak = the ack (dynamic, rides the low-NFE fast path)
                    LLMSay= composed ask/summary → phrase → TTS → FLOAT
WS(/ws/state) pushes the next video segment and subtitle to the frontend
Barge-in: VAD/ASR speech_start ─▶ cancel_event + turn bump ─▶ clear queue
```

**The protocol is data, not handlers** (`modules/sbirt/flow.py`): the whole
study session — consent gate, 3-question pre-screen, alcohol/drug arms,
AUDIT/DAST administration, zone feedback, brief intervention, closes — is ONE
step program (`Gate / Ask / Tell / RunItems / Route / End`) executed by a
single interpreter in `runtime.py`. There is no per-question code: adding a
question = adding an instrument `Item` or a flow step. Open questions are
never discarded — every capture lands in session state; the alcohol Q/F is a
slot-ask (drink/amount/frequency) that asks ONE missing slot at a time and
holds through multi-breath answers.

**Deterministic clinical core** (`modules/sbirt/`): AUDIT/DAST scores, risk
zones, question order/skip rules (declarative `SkipRule` data on each
instrument), arm routing and zone feedback are computed by code
(`instruments.py` + `flow.py` + `runtime.py`), never by the LLM. The LLM's
output enters the engine ONLY as a `turn.TurnOut` validated against the
current expectation (guess-free: an undetermined timeframe/quantity degrades
to a clarification, never a code). Wording is state-honest: lines that
reference earlier content (e.g. "the standard drink definition we just
discussed") have variants picked from what was ACTUALLY delivered
(`session.covered`), and the close is assembled from the real path taken
(declined sessions never get promised "a few more questions"). A
deterministic crisis keyword net (`crisis.py`) runs before everything and
cannot be down; an NLU-flagged crisis unions with it. Consent decisions land
in an append-only audit log (`records/`); the audit record carries codes/
scores/zones and capture KEYS only — never transcripts.

**Turn-level safety patches**: score-critical items (AUDIT 1–6) whose answer
was coded by semantic mapping are read back as the CODED option and held
uncommitted until the person confirms — a "no" re-collects the item; exact-
wording answers skip the read-back (T20). "Actually it's more like three
times a week" corrects the earlier item, re-derives the skip landscape from
the declarative rules, and logs old→new for audit (T21). "I'm done" anywhere
closes gracefully with partial data kept and the abort recorded (T22).
(Encrypted at-rest state persistence with crash/reconnect resume — T24 —
was built and then shelved for the research phase; recover it from git
history, commit message "Persist encrypted session state".)

Server: Starlette + uvicorn (single port 17861, HTTP + two WebSocket routes).
Frontend: `static/index.html`, plain HTML/JS, no build step.

## Layout

```
digital-human/
├── main.py                  # Starlette entrypoint (HTTP + WS), per-sid sessions
├── config.py                # All configuration (GPU allocation, models, prompt, ...)
├── modules/
│   ├── pipeline.py          # Orchestration: ASR → turn NLU → engine → clips/TTS/FLOAT
│   ├── vad.py               # Silero VAD real-time segmentation
│   ├── eou.py               # smart-turn v3 semantic end-of-utterance (CPU ONNX)
│   ├── asr.py               # FunASR (SenseVoiceSmall)
│   ├── llm.py               # OpenRouter: the per-turn NLU+voice call + bounded utterances (+ crisis chat)
│   ├── tts.py               # edge-tts (en-US-GuyNeural)
│   ├── avatar.py            # FLOAT GPU pool + idle video
│   ├── privacy.py           # PHI-safe logging (no opt-out) + consent audit trail
│   └── sbirt/               # SBIRT clinical framework — the deterministic core
│       ├── instruments.py   #   structured tools + deterministic scoring in one file
│       ├── flow.py          #   the study protocol as ONE declarative step program
│       ├── runtime.py       #   the generic turn engine executing flow.PROTOCOL
│       ├── turn.py          #   TurnOut: the validated per-turn NLU contract
│       ├── templates.py     #   verbatim script + content units (questions/feedback/BI)
│       ├── crisis.py        #   deterministic crisis keyword net + fixed responses
│       ├── intervention.py  #   MI/OARS, FRAMES, stages of change (prompt data)
│       ├── referral.py      #   ASAM levels, MAT, resources, crisis protocol (prompt data)
│       ├── workflow.py      #   conceptual SBIRT node map (prompt data)
│       └── prompt.py        #   build_system_prompt() — now used for crisis turns
├── scripts/
│   └── sim_conversation.py  # text-layer acceptance sim (real LLM, no GPU stack)
├── tests/                   # pytest suite — LOCAL ONLY, untracked (see "Tests")
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
- Hardware: CUDA GPU(s); the defaults put FLOAT and ASR together on GPU 3
  (`config.FLOAT_GPUS` / `config.ASR_GPU` — list more GPUs in `FLOAT_GPUS`
  to render sentence clips in parallel).
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
| `LLM_MODEL` | `google/gemini-2.5-flash` | OpenRouter model (the per-turn NLU+voice call, bounded utterances, crisis chat) |
| `TTS_VOICE` | `en-US-GuyNeural` | edge-tts voice |
| `ASR_MODEL` | `iic/SenseVoiceSmall` | FunASR model |
| `FLOAT_GPUS` | `[3]` | GPUs used by FLOAT (list; parallel across entries) |
| `ASR_GPU` | `3` | GPU for ASR (currently shared with FLOAT) |
| `FLOAT_NFE` | `10` | FLOAT inference steps (lower = faster) |
| `VAD_THRESHOLD` | `0.5` | Silero VAD trigger threshold |
| `VAD_SILENCE_DURATION` | `0.35` | Seconds of silence to mark sentence end |
| `CONSENT_LOG_PATH` | `records/consent_log.jsonl` | Consent audit trail |
| `SYSTEM_PROMPT` | built from `modules/sbirt/` | Crisis-turn counselor prompt — **do not edit the string**; edit the clinical data modules under `modules/sbirt/` and it rebuilds automatically |

Clinical content (questions, options, scores, zones, feedback wording) lives in
`modules/sbirt/instruments.py` + `templates.py`, sourced verbatim from the
study documents in `SBIRT_Reference/`. Changing any fixed text regenerates its
cached clip automatically (sidecar text check).

## Tests

The pytest suite is deliberately **not tracked** in this repo (`tests/` is
gitignored; the repo ships lean). It lives in git history — rematerialize it
into your working tree with:

```bash
# Restore tests/ from the parent of the commit that removed it (the grep
# survives history rewrites; the suite lands untracked, hidden by .gitignore):
git restore --source="$(git log --format=%H -1 --all --grep='removed test files')^" -- tests/
```

Then:

```bash
# Deterministic core (no GPU/network needed):
python -m pytest tests/ -q
# Full suite incl. real-Pipeline integration (needs the float env's deps):
~/.conda/envs/float/bin/python -m pytest tests/ -q
# Conversation-quality acceptance at the text layer (REAL LLM, no GPU stack;
# this one IS tracked in the repo). Three scenarios: replays of BOTH field
# transcripts that motivated fixes (the rigid-dialogue one and the
# "ten twenty times / more than one / you spoke too fast" one) plus a full
# conversational alcohol-BI walk incl. the T20 read-backs:
python scripts/sim_conversation.py
```

Gold-standard fixtures under `tests/fixtures/` are hand-scored from the case
cards in `SBIRT_Reference/` — fix code, never fixtures. Run the suite before
touching anything under `modules/sbirt/` (scores/zones/skip rules have no
other guardrail). The float-env suite also locks the generation budget: a
full session renders dynamic clips only for the acks/composed asks/BI
summaries — fixed content plays from `assets/clips/` with ZERO runtime FLOAT
renders.

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
