# Digital Human — Voice SBIRT Screening Counselor

A real-time, voice-driven digital human that administers an SBIRT
(Screening, Brief Intervention, Referral to Treatment) substance-use
screening over the browser: the avatar speaks, listens, and walks the person
through a validated clinical protocol — pre-screen, AUDIT / DAST-10,
zone feedback, and a brief intervention — then hands the results to their
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

Fixed protocol lines (question stems, permissions, feedback) are pre-rendered
once into cached clips and replayed verbatim; only conversational glue is
generated per turn. The voice layer adds:

- **Barge-in** — ASR-confirmed interruption while the avatar is speaking.
- **Semantic EOU** — think-pauses don't cut people off.
- **Pause-split merge** — a resumed sentence reaches ASR in one piece.

## Repository layout

```
main.py              Starlette app: HTTP routes, audio/state WebSockets, sessions
config.py            Single source of truth for paths, models, tuning, env vars
restart.sh           Stop port owner → rotate log → start detached → probe
static/index.html    The single-page browser client (mic, video, chat)
assets/              avatar*.png source portrait — MUST be a synthetic face, no
                     identifiable real person (肖像权); idle_loop.mp4; clips/
modules/             Voice pipeline: pipeline.py (turn lifecycle, barge-in, clip
                     cache, delivery), vad, eou, asr, llm, tts, avatar (FLOAT),
                     privacy (PHI-free logging + consent audit trail)
modules/sbirt/       THE clinical layer:
  flow.py              the protocol as one declarative program (steps + routes)
  runtime.py           generic turn engine: ClinicalSession, advance, crisis/abort
  instruments.py       AUDIT, DAST-10, pre-screen: items, options, scores, skip rules
  coding.py            deterministic code derivation (frequencies, drink quantities)
  turn.py              TurnOut: the validated NLU contract (the only LLM→engine channel)
  templates.py         verbatim fixed script (study wording) + content units
  state_view.py        renders the full interview state into the LLM context each turn
  crisis.py            deterministic crisis net (fixed responses, 988/911)
  prompt.py            build_system_prompt() for the crisis-turn counselor
  workflow.py          narrative SBIRT state machine (used by prompt.py)
  intervention.py      MI/OARS, FRAMES, readiness-ruler reference data
  referral.py          ASAM levels of care, MAT, crisis protocol reference data
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

The browser mic needs a secure context: put certificates (self-signed is fine)
at `certs/cert.pem` / `certs/key.pem`, or set `ENABLE_HTTPS=0` and use
`http://localhost:17861` only. Every other tuning knob lives in `config.py`
with inline documentation.

### Run

```bash
python main.py
```

Then open `https://<host>:17861/`. The first run is slow on purpose: it
downloads Silero VAD via torch.hub, generates the idle loop, and pre-renders
every fixed protocol line into a cached clip under `assets/clips/` —
after that, fixed content plays instantly with zero per-session synthesis.

Clips are fingerprinted against their spoken text and the reference portrait,
so changing `GREETING_TEXT`, a protocol line, or `config.AVATAR_IMAGE`
regenerates whatever it affects. A face swap therefore costs one full
re-render (~89 clips, ~15 min on a single FLOAT GPU) and can never leave stale
clips playing the old face.

Press **Start** in the UI: the avatar speaks the fixed greeting and asks for
consent; from there the protocol engine drives the whole screening.

### Restart the running service

Python changes need a restart; frontend changes do not — `GET /` re-reads
`static/index.html` from disk on every request, so a hard-refresh picks up UI
edits immediately.

```bash
./restart.sh        # env smoke-test → stop port owner → rotate log → start → probe
```

Prints the HTTP status once the models are warm (~20 s). It refuses to stop the
old process if the environment is broken; the reasons behind each step are
commented in the script. `conda activate float && python main.py` is the
equivalent foreground form.

## HTTP / WebSocket surface

| Endpoint | Purpose |
|---|---|
| `GET /` | Browser client |
| `GET /static/case_card.html` | Tester's role-play card (see below) |
| `WS /ws/audio` | Mic audio in (16 kHz PCM), VAD/EOU/barge-in server-side |
| `WS /ws/state` | Push channel: video segments, captions, session state |
| `POST /api/greet` | Start a session (plays the greeting, arms consent) |
| `POST /api/text` | Typed input as an alternative to voice |
| `POST /api/reset` | Reset the session |
| `POST /api/toggle` | Mic on/off |
| `POST /api/test_asr` | One-shot ASR round-trip check |

`case_card.html` is *Usability Testing-Randomized Case Selection* — a static
page with no server or model involvement. **Refresh** draws one case at random;
a page reload draws nothing.

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
