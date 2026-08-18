# Digital Human — Voice SBIRT Screening Counselor

A real-time, voice-driven digital human that administers an SBIRT
(Screening, Brief Intervention, Referral to Treatment) substance-use
screening over the browser: the avatar speaks, listens, and walks the person
through a validated clinical protocol — pre-screen, AUDIT / DAST-10,
zone feedback, and a brief intervention — then hands the results to their·
provider.

The design principle throughout: **the clinical protocol is deterministic
code; the LLM only understands and phrases.** Which question comes next,
every score, zone, skip rule, and branch is decided by a reviewable state
machine over declarative instrument data — never by the model.

## How a turn flows

```
mic (browser) ──WebSocket──▶ VAD (Silero) + EOU (smart-turn v3)
                                    │ speech_end
                                    ▼
                              ASR (SenseVoice)
                                    │ text
                                    ▼
              NLU: one LLM call → validated TurnOut   (modules/llm.py, sbirt/turn.py)
                                    │ answer / question / crisis / ...
                                    ▼
              Clinical engine: advance / hold          (sbirt/runtime.py + flow.py)
                                    │ what to say next
                                    ▼
              TTS (edge-tts) ─▶ FLOAT avatar render ─▶ video queue ─▶ browser
```

Fixed protocol lines (question stems, permissions, feedback) are
pre-rendered once into cached clips and replayed verbatim; only
conversational glue is generated per turn. The voice layer supports
barge-in (ASR-confirmed while the avatar speaks), semantic end-of-utterance
detection so think-pauses don't cut people off, and pause-split continuation
merge so a resumed sentence reaches ASR in one piece.

## Repository layout

```
main.py                    Starlette app: HTTP routes, audio/state WebSockets, sessions
config.py                  Single source of truth for paths, models, tuning, env vars
requirements.txt           Python dependencies (see Getting started for torch/FLOAT)
.env.example               Template for the .env file (API key)
modules/
  pipeline.py              Orchestrator: turn lifecycle, barge-in, clip cache, delivery
  vad.py                   Silero VAD wrapper (+ EOU gating of speech_end)
  eou.py                   smart-turn v3 semantic end-of-utterance (ONNX, CPU)
  asr.py                   SenseVoice speech-to-text
  llm.py                   OpenRouter client: per-turn NLU + bounded phrasing
  tts.py                   edge-tts synthesis
  avatar.py                FLOAT talking-head rendering (sibling repo, GPU)
  privacy.py               PHI-free logging helpers + consent audit trail
  sbirt/
    flow.py                THE protocol as one declarative program (steps + routes)
    runtime.py             Generic turn engine: ClinicalSession, advance, crisis/abort
    instruments.py         AUDIT, DAST-10, pre-screen: items, options, scores, skip rules
    coding.py              Deterministic code derivation (frequencies, drink quantities)
    turn.py                TurnOut: the validated NLU contract (the only LLM→engine channel)
    templates.py           Verbatim fixed script (study wording) + content units
    state_view.py          Renders the full interview state into the LLM context each turn
    crisis.py              Deterministic crisis net (fixed responses, 988/911)
    prompt.py              build_system_prompt() for the crisis-turn counselor
    workflow.py            Narrative SBIRT state machine (used by prompt.py)
    intervention.py        MI/OARS, FRAMES, readiness-ruler reference data
    referral.py            ASAM levels of care, MAT, crisis protocol reference data
assets/
  avatar*.png              Source portrait(s) the avatar is rendered from
  idle_loop.mp4            Ambient idle loop shown between turns
static/
  index.html               The single-page browser client (mic, video, chat)
```

## Getting started

### Prerequisites

- Linux with an NVIDIA GPU (FLOAT rendering and ASR run on GPU; see
  `FLOAT_GPUS` / `ASR_GPU` in `config.py`).
- Python 3.10+.
- **PyTorch** matching your CUDA version (not pinned in
  `requirements.txt` — install per https://pytorch.org). Needed by Silero
  VAD, SenseVoice, and FLOAT.
- **FLOAT** checked out as a sibling directory `../float` with its
  checkpoint at `../float/checkpoints/float.pth`
  (https://github.com/deepbrainai-research/float).
- An **OpenRouter API key** (the NLU/phrasing model is
  `google/gemini-2.5-flash`).
- Optional: the smart-turn v3 EOU model at
  `checkpoints/smart-turn/smart-turn-v3.2-cpu.onnx`. If absent, turn-taking
  transparently falls back to silence-duration VAD.

### Install

```bash
git clone <this repo> digital-human
cd digital-human
pip install -r requirements.txt
# then install torch for your CUDA version, e.g.
# pip install torch --index-url https://download.pytorch.org/whl/cu121
```

### Configure

```bash
cp .env.example .env        # then put your real key in it
```

| Variable | Default | Meaning |
|---|---|---|
| `OPENROUTER_API_KEY` | — | Required. LLM access for NLU + phrasing |
| `SERVER_HOST` / `SERVER_PORT` | `0.0.0.0` / `17861` | Bind address / port |
| `ENABLE_HTTPS` | `1` | Serve TLS from `certs/cert.pem` + `certs/key.pem` |
| `USE_EOU` | `1` | Semantic end-of-utterance gating of VAD |
| `BARGE_IN_ASR` | `1` | ASR-confirmed interruption while the avatar speaks |
| `DEBUG_SAVE_AUDIO` | `0` | Save each captured utterance to `tmp/` for debugging |

The browser microphone requires a secure context: provide certificates at
`certs/cert.pem` / `certs/key.pem` (self-signed is fine), or set
`ENABLE_HTTPS=0` and use `http://localhost:17861` only. All tuning knobs
(VAD/EOU thresholds, FLOAT quality vs. latency, history window, GPU ids)
live in `config.py` with inline documentation.

### Run

```bash
python main.py
```

Then open `https://<host>:17861/`. The first run is slow on purpose: it
downloads Silero VAD via torch.hub, generates the idle loop, and pre-renders
every fixed protocol line into a cached clip under `assets/clips/` —
after that, fixed content plays instantly with zero per-session synthesis.

Press **Start** in the UI: the avatar speaks the fixed greeting and asks for
consent; from there the protocol engine drives the whole screening.

### Restart the running service

Every module is loaded into memory at startup, so **Python changes need a
restart — frontend changes do not**: `GET /` re-reads `static/index.html` from
disk on every request, so a browser refresh (hard-refresh; no `Cache-Control`
is set) picks up UI edits immediately.

```bash
# 1. stop whoever holds the port (graceful; uvicorn drains in ~2s)
kill -TERM $(ss -lptnH "sport = :17861" | grep -oP 'pid=\K[0-9]+')

# 2. start it detached, capturing BOTH stdout and stderr
LD_LIBRARY_PATH=:/usr/local/cuda/lib64 PYTHONUNBUFFERED=1 \
  setsid nohup /home/ryu11/.conda/envs/float/bin/python main.py \
  > run.log 2>&1 < /dev/null &

# 3. wait for the models, then confirm it answers
grep -q "Model pre-warm complete" run.log && \
  curl -sk -o /dev/null -w "%{http_code}\n" https://127.0.0.1:17861/
```

`conda activate float && python main.py` is the equivalent foreground form; the
absolute interpreter path above is what the detached command needs.

Stop it by **port owner**, not by name: `pgrep -f "...main.py"` also matches the
shell running the command whenever that pattern appears in its own command line,
so `kill -TERM $(pgrep -f ...)` can take out your own session instead.

Startup takes about 20 s — FLOAT onto its GPU, SenseVoice ASR, and the
smart-turn EOU ONNX model all pre-warm before the first turn. `2>&1` is not
optional: the application log goes to **stderr**, so a plain
`python main.py > run.log` captures nothing but the banner.

Smoke-test the environment *before* stopping the old process — a missing
transitive dependency only surfaces at startup, once the running instance is
already gone:

```bash
python -c "import uvicorn, starlette, click, funasr, onnxruntime" && echo OK
```

(`click` is uvicorn's own dependency and is not listed in `requirements.txt`
by name; it has gone missing from the env before, which turns any restart into
an outage.)

One GPU caveat: stopping the server releases its ~4.6 GB, and the restart has
to take that memory back. If the box is otherwise near capacity, another job
can claim the gap — restart when you can watch it come back up, not blind.

## HTTP / WebSocket surface

| Endpoint | Purpose |
|---|---|
| `GET /` | Browser client |
| `GET /static/case_card.html` | Usability Testing-Randomized Case Selection — the tester's role-play card. Static page, no server or model involvement; **Refresh** draws one case at random, a page reload draws nothing. |
| `WS /ws/audio` | Mic audio in (16 kHz PCM), VAD/EOU/barge-in server-side |
| `WS /ws/state` | Push channel: video segments, captions, session state |
| `POST /api/greet` | Start a session (plays the greeting, arms consent) |
| `POST /api/text` | Typed input as an alternative to voice |
| `POST /api/reset` | Reset the session |
| `POST /api/toggle` | Mic on/off |
| `POST /api/test_asr` | One-shot ASR round-trip check |

## Safety & privacy

- **Deterministic crisis net**: self-harm/danger cues trigger a fixed,
  clinician-reviewed response (988 / 911) from a cached clip — no LLM on
  that path — and pause the protocol for the rest of the session.
- **Never-guess coding**: an utterance only advances the protocol after
  validation against the current expectation; ambiguous answers are
  clarified, score-critical semantic codes are read back for confirmation
  before they commit.
- **PHI-free logs**: log lines carry codes, node names, and hashes — never
  transcripts. The consent decision (yes/no + wording version, no content)
  is appended to a local audit file under `records/`.
- Results are for the person's medical provider; the app makes no referral
  decisions and speaks no diagnoses.
