import json
import logging
import re
import threading
from openai import OpenAI
import config
from modules.sbirt.turn import TurnOut, expected_item
from modules.sbirt.turn import validate as validate_turn

logger = logging.getLogger(__name__)

# Lazily built so an empty/misconfigured OPENROUTER_API_KEY doesn't crash the
# whole server at import time (OpenAI('') raises on construction).
_client_obj = None
_client_lock = threading.Lock()


def _client() -> OpenAI:
    global _client_obj
    if _client_obj is None:
        with _client_lock:
            if _client_obj is None:
                _client_obj = OpenAI(
                    base_url=config.OPENROUTER_BASE_URL,
                    api_key=config.OPENROUTER_API_KEY,
                )
    return _client_obj


def build_messages(history: list[dict], patient: dict | None = None) -> list[dict]:
    """Build the API message list: system prompt (plus any known patient facts) on
    top, then a snapshot of the caller-owned conversation history.

    The patient profile is injected fresh EVERY turn, independent of the history
    window, so key facts (age, sex, screening answers) survive the sliding-window
    trim and the model never loses them even after old turns scroll out."""
    system = config.SYSTEM_PROMPT
    if patient:
        system += ("\n\n=== KNOWN PATIENT (persists for the whole session) ===\n"
                   "These were established earlier — do NOT re-ask a field that is "
                   "already filled; use them to tailor screening and tone:\n"
                   + json.dumps(patient, ensure_ascii=False))
    return [{"role": "system", "content": system}] + list(history)


_EXTRACT_SYSTEM = (
    "You extract structured facts for an SBIRT substance-use screening from a "
    "counseling conversation. Output ONLY a JSON object with any of these keys you "
    "can determine from what the USER explicitly said; omit any you are unsure "
    "about. Keys: age (int), sex ('male'|'female'|'other'), substances (list of "
    "substances actually used), alcohol_use, tobacco_use, drug_use, rx_misuse "
    "(short strings), screening (object mapping instrument -> answers/score so far), "
    "readiness_stage, risk_level, notes (one short string). Never invent anything; "
    "if the user gave no new factual info, output {}."
)


def extract_patient_facts(history: list[dict]) -> dict:
    """Best-effort structured extraction of patient facts from recent conversation.
    Returns {} on ANY failure — this is a non-critical add-on that must never break
    a turn (it runs off the hot path and only updates the profile for next turn)."""
    convo = "\n".join(f"{m['role']}: {m.get('content', '')}" for m in history[-8:])
    if not convo.strip():
        return {}
    try:
        resp = _client().chat.completions.create(
            model=config.LLM_MODEL,
            messages=[{"role": "system", "content": _EXTRACT_SYSTEM},
                      {"role": "user", "content": convo}],
            temperature=0,
        )
        raw = (resp.choices[0].message.content or "").strip()
        start, end = raw.find("{"), raw.rfind("}")
        if start == -1 or end == -1 or end < start:
            return {}
        data = json.loads(raw[start:end + 1])
        return data if isinstance(data, dict) else {}
    except Exception as e:
        logger.info("patient extraction skipped (%s)", e)
        return {}


def chat(messages: list[dict]) -> str:
    """Non-streaming completion. `messages` is the full API message list (system +
    history + current user) built by the caller. This function does NOT mutate any
    history — the caller (Pipeline) owns and updates the conversation state."""
    response = _client().chat.completions.create(
        model=config.LLM_MODEL,
        messages=messages,
    )
    return response.choices[0].message.content or ""


def chat_stream(messages: list[dict],
                cancel_event: threading.Event | None = None):
    """Yield complete sentences as they arrive from the LLM stream.

    `messages` is the full API message list built by the caller. This function is
    pure: it does NOT touch any history — the caller accumulates the yielded
    sentences and owns the conversation state. Checks cancel_event between chunks.
    """
    stream = _client().chat.completions.create(
        model=config.LLM_MODEL,
        messages=messages,
        stream=True,
    )

    buffer = ""
    # Split ONLY on sentence-final punctuation so each spoken clip is a whole
    # sentence. The system prompt's short-acknowledgment-first rule keeps the first
    # clip short enough for a fast start without sub-sentence (comma) splitting.
    sentence_endings = {"。", "！", "？", ".", "!", "?", "\n"}

    for chunk in stream:
        if cancel_event and cancel_event.is_set():
            break

        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta.content
        if delta:
            buffer += delta

            # Extract ALL complete sentences from the buffer.
            while True:
                split_pos = -1
                for i, ch in enumerate(buffer):
                    if ch in sentence_endings:
                        split_pos = i
                        break

                if split_pos == -1:
                    break

                sentence = buffer[: split_pos + 1].strip()
                buffer = buffer[split_pos + 1 :]
                if sentence:
                    yield sentence

    # Flush remaining buffer
    if buffer.strip() and not (cancel_event and cancel_event.is_set()):
        yield buffer.strip()


# =====================================================================
# Constrained NLU (P4 -> T4): free text -> ONE validated TurnOut per turn.
# The coder NEVER guesses: quantities/timeframes the user did not state
# route to a clarification, not to a code (case-card AUDIT item 10 rule).
# Deterministic pre-matching handles the unambiguous fast path with zero
# LLM latency; everything semantic goes through ONE turn() call whose
# output is gated by turn.validate before it can move anything.
# =====================================================================

AMBIGUOUS = "AMBIGUOUS"

_YES_RE = re.compile(r"^\s*(yes|yeah|yep|yup|sure|correct|i do|i have)\b", re.I)
_NO_RE = re.compile(r"^\s*(no|nope|nah|never|not really|i don'?t|i do not|i haven'?t)\b", re.I)

_WORD_NUMBERS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
}
_DIGIT_RE = re.compile(r"\b(10|[0-9])\b")

def _prematch_option(options, text: str):
    """Deterministic pre-pass: exact label/alias match, and yes/no shortcuts
    for binary No/Yes items (short utterances only — a longer reply may carry
    a question or a caveat the LLM must see). Returns a code or None."""
    t = " ".join(text.strip().lower().split())
    if not t:
        return None
    for i, opt in enumerate(options):
        if t == opt.label.lower() or t in (a.lower() for a in opt.aliases):
            return i
    labels = [o.label for o in options]
    if labels == ["No", "Yes"] and len(t.split()) <= 3:
        if _YES_RE.match(t):
            return 1
        if _NO_RE.match(t):
            return 0
    return None


def code_number(user_text: str, low: int = 0, high: int = 10) -> dict:
    """Code a spoken 0-10 ruler answer. Deterministic digit/word scan first;
    exactly one distinct in-range number -> that value, else AMBIGUOUS.
    (No LLM: a readiness number the user didn't clearly say must be re-asked.)"""
    t = user_text.lower()
    found = {int(m) for m in _DIGIT_RE.findall(t)}
    found |= {v for w, v in _WORD_NUMBERS.items()
              if re.search(rf"\b{w}\b", t)}
    found = {n for n in found if low <= n <= high}
    if len(found) == 1:
        return {"value": found.pop()}
    return {"status": AMBIGUOUS}


# --------------- The ONE per-turn NLU + voice call (T4/T11/T12) ---------------

# Whole-utterance consent matches only: "yes but what does that mean" must
# reach the LLM (it is a question, not a bare yes).
_CONSENT_YES = {"yes", "yeah", "yep", "yup", "sure", "ok", "okay", "alright",
                "all right", "of course", "sure thing", "go ahead",
                "yes please", "fine", "sounds good"}
_CONSENT_NO = {"no", "nope", "nah", "no thanks", "no thank you", "not now",
               "not really", "i'd rather not", "id rather not"}

_TURN_SYSTEM = """You are the natural-language understanding AND the voice of one turn of a
structured SBIRT health-screening avatar (a VOICE conversation). The clinical
protocol — which question comes next, scores, risk zones, routing — is
decided by external code, NEVER by you.

The avatar just asked:
{ask}

Expected form of an answer:
{expectation}

Session facts (the ONLY source for factual claims in your reply):
{facts}

Live interview state (program-rendered from the session every turn — treat it
as session facts too). It is the full map of the screening: every question,
what is already answered, what is being asked right now, and what comes next.
Use it to answer meta questions ("how many questions are left", "what is this
for", "didn't I already tell you that") and to avoid re-asking anything it
shows as answered:
{interview_state}

Read the user's utterance and output ONLY a compact JSON object:
{{"action": "...", "code": null, "item": null, "slots": {{}}, "text": null, "value": null, "per": null, "unit": null, "beverage": null, "reply": "..."}}

action — exactly one of:
  "answer"       it answers the current ask (even partially for slots)
  "continuation" it adds to / completes their PREVIOUS answer instead
  "question"     they are asking YOU something
  "tangent"      an off-topic aside or small talk
  "crisis"       ANY sign of self-harm, overdose, danger, acute distress
  "abort"        they clearly want to stop the WHOLE conversation ("stop",
                 "i'm done", "i don't want to do this anymore") — NOT a
                 plain "no" to the current may-I question (that's "answer")
  "correction"   they are CHANGING an answer they already gave to an earlier
                 question ("actually it's more like three times a week") —
                 see answers_already_given in the session facts
  "dont_know"    they cannot answer or would rather not answer THIS question
                 ("i don't know", "no idea", "can't remember", "skip that
                 one", "rather not say") — NOT stopping the whole session,
                 and NOT an answer of "never"/"no"
  "unclear"      none of the above is safe to assume

Coding rules (guess-free — a wrong code corrupts a validated screening):
- Option items: "code" = the option number the utterance determines. Their
  wording does NOT need to match the option label — when everything they
  stated falls inside exactly ONE option, code that option. Examples:
  "ten or twenty times" -> "One or more"; "yesterday" -> "Within the last
  year". Refusing a codable answer is as wrong as guessing.
- EXTRACTION items: when the expectation below says the engine computes the
  option from extracted fields, do NOT pick the option yourself — fill the
  fields exactly as described there (value/per for frequencies, value/unit/
  beverage for drink amounts) and leave "code" null. The deterministic
  engine does the bucketing.
- But NEVER guess: if what they said fits MORE THAN ONE option ("more than
  one", "a few", "sometimes"), or omits the detail the options differ by
  (timeframe, count), use action "unclear" — do not pick a side.
- Yes/no permission asks: "code" 1 = they agree, 0 = they decline.
- Number asks: "code" = the single 0-10 number they said; two different
  numbers or none -> "unclear".
- Open asks: put the captured answer in "text"; when slots are listed in the
  expectation, put each piece they actually gave under its slot name in
  "slots" (never invent a slot they did not address). An amount or frequency
  slot needs a usable rough quantity: a bare "more than one" / "a lot", or
  an answer in the wrong dimension (a count of TIMES when asked how many
  DRINKS) -> action "unclear", never a capture.
- correction: "item" = the item number being corrected (from
  answers_already_given in the session facts) and "code" = its NEW option
  number — BOTH only when unambiguous. If you cannot tell which question
  they mean or which option is now right, use "unclear" and ask which.

reply rules — you speak WITH the person, warm and plain-spoken:
- answer: reply is ONLY a brief acknowledgment of what they said, at most 8
  words, one sentence ending with a period. Vary it every time; never echo
  an acknowledgment already used in the conversation. No questions, no
  advice, no new information.
- question: answer THEIR question in one or two short sentences using ONLY
  the session facts above — if something was not covered, say so honestly —
  then re-ask the current ask briefly in fresh words. This includes asking
  you to repeat or slow down ("you spoke too fast", "say that again"): give
  the key information again in one plain sentence from the session facts,
  then re-pose the current ask.
- tangent: one warm sentence acknowledging what they said, then gently
  return to the current ask.
- continuation: briefly acknowledge the added detail, then re-pose the
  current ask in a few words.
- correction: confirm the change in a few words (say the new answer back);
  the engine re-poses the current question after you.
- unclear: ONE short clarifying question aimed at exactly what is missing.
  If their words sit between specific choices, name those choices ("Would
  that be five or six drinks, or more like seven to nine?"); if they
  answered a different dimension, ask which they meant ("Is that how many
  drinks you have at one time, or how often you drink?"). NEVER re-read the
  question word-for-word, and never reuse a clarification wording already
  used in this conversation — each attempt must get more specific, not
  repeat itself. Never suggest which choice to pick.
- crisis / abort / dont_know: leave reply empty — a fixed response takes
  over (for dont_know the protocol itself offers a recall anchor or moves on).
- NEVER: scores, risk zones, diagnoses, clinical jargon, lecturing, stacked
  questions, or stock phrases like "I hear you."
"""


def _expectation_text(expect) -> str:
    kind = expect.kind
    if kind == "consent":
        return "A yes or no."
    if kind == "confirm":
        return ("A yes or no: you just read their previous answer back and "
                "asked if you understood it right. 1 = confirmed, 0 = they "
                "say it was wrong.")
    if kind == "option":
        item = expected_item(expect)
        lines = "\n".join(f"  {i}: {o.label}"
                          for i, o in enumerate(item.options))
        coding = getattr(item, "coding", "choice")
        if coding in ("freq_q1", "freq5"):
            return (
                "A frequency. The engine computes the option from your "
                "extraction — fill \"value\" (how many times, a number) and "
                "\"per\" (\"day\"|\"week\"|\"month\"|\"year\"); leave "
                "\"code\" null. \"every week\" -> value 1, per \"week\"; "
                "\"always\" / \"every single day\" -> value 1, per \"day\"; "
                "\"twice a month\" -> value 2, per \"month\"; \"never\" -> "
                "value 0. A rate you cannot pin down -> \"unclear\".\n"
                f"The choices, for context only:\n{lines}")
        if coding == "quantity_drinks":
            return (
                "An amount of drinking. The engine computes the option from "
                "your extraction — fill \"value\" (a number) and \"unit\": "
                "\"drinks\" when they count drinks, otherwise the container "
                "or volume they said (\"liter\", \"bottle\", \"shot\", "
                "\"glass\", \"can\", \"oz\", \"ml\", \"pint\", \"fifth\", "
                "\"handle\"), plus \"beverage\" when known from this or an "
                "earlier turn (\"whiskey\", \"beer\", \"wine\"); leave "
                "\"code\" null. \"ten or more\" -> value 10, unit "
                "\"drinks\"; \"a liter of whiskey\" -> value 1, unit "
                "\"liter\", beverage \"whiskey\". An amount with no usable "
                "number -> \"unclear\".\n"
                f"The choices, for context only:\n{lines}")
        return f"One of these choices (code = number):\n{lines}"
    if kind == "number":
        return "A single number from 0 to 10."
    if kind == "open" and expect.missing:
        return (f"Free text. Slots: {', '.join(expect.slots)}. "
                f"Still missing: {', '.join(expect.missing)} "
                f"(the ask was about {expect.missing[0]!r}).")
    if kind == "open":
        return "Free text."
    return "The session has ended; no answer is expected."


# Whole-utterance matches only, same rule as the consent sets: "i don't know
# if that counts" carries content and must reach the LLM.
_DONT_KNOW = {"i don't know", "i dont know", "don't know", "dont know",
              "dunno", "i dunno", "no idea", "i have no idea", "not sure",
              "i'm not sure", "im not sure", "i can't remember",
              "i cant remember", "can't remember", "cant remember",
              "i don't remember", "i dont remember", "no clue",
              "i'd rather not say", "id rather not say", "rather not say"}


def _prepass(user_text: str, expect) -> TurnOut | None:
    """Zero-latency deterministic paths for unambiguous short answers.
    reply stays empty — skipping the acknowledgment beats guessing one."""
    t = " ".join(user_text.strip().lower().split()).rstrip(".!,")
    if t in _DONT_KNOW and expect.kind in ("consent", "confirm", "option",
                                           "number"):
        # T25/F1: a first-class result — the pipeline probes once with a
        # recall anchor, then marks the item missing. Never an unclear loop.
        return TurnOut(action="dont_know")
    if expect.kind in ("consent", "confirm"):
        if t in _CONSENT_YES:
            return TurnOut(action="answer", code=1, exact=True)
        if t in _CONSENT_NO:
            return TurnOut(action="answer", code=0, exact=True)
        return None
    if expect.kind == "option":
        code = _prematch_option(expected_item(expect).options, user_text)
        if code is not None:
            # exact=True: the utterance WAS the option's wording, so a T20
            # confirm item commits without a read-back.
            return TurnOut(action="answer", code=code, exact=True)
        return None
    if expect.kind == "number":
        got = code_number(user_text)
        if "value" in got:
            return TurnOut(action="answer", code=got["value"], exact=True)
        return None
    return None


def turn(user_text: str, expect, *, ask_text: str, history: list[dict],
         patient: dict | None = None, facts: dict | None = None,
         interview_state: str = "") -> TurnOut:
    """ONE call per user utterance: classify what the utterance IS relative
    to the current ask, code it if it is an answer, and produce this turn's
    bounded reply. The result is ALWAYS gated by turn.validate — an illegal
    answer comes back as 'unclear' and moves nothing. Total failure returns
    unclear with an empty reply (the pipeline then re-asks deterministically),
    so the protocol can never be stranded by the model."""
    pre = _prepass(user_text, expect)
    if pre is not None:
        return validate_turn(pre, expect)

    all_facts = dict(facts or {})
    if patient:
        all_facts["patient"] = patient
    system = _TURN_SYSTEM.format(
        ask=ask_text or "(the opening consent question)",
        expectation=_expectation_text(expect),
        facts=json.dumps(all_facts, ensure_ascii=False),
        interview_state=interview_state or "(not available this turn)")
    messages = ([{"role": "system", "content": system}]
                + list(history[-6:])
                + [{"role": "user", "content": user_text}])
    for attempt in range(2):
        try:
            resp = _client().chat.completions.create(
                model=config.LLM_MODEL, messages=messages, temperature=0)
            raw = (resp.choices[0].message.content or "").strip()
            start, end = raw.find("{"), raw.rfind("}")
            if start == -1 or end <= start:
                raise ValueError("no JSON object in turn output")
            out = TurnOut.model_validate_json(raw[start:end + 1])
            # `exact` belongs to the deterministic pre-pass and assumed/
            # boundary/note to validate()'s derivation: a model claiming any
            # of them must never skip (or fake) a T20 read-back.
            out = out.model_copy(update={"exact": False, "assumed": False,
                                         "boundary": False, "note": ""})
            return validate_turn(out, expect)
        except Exception as e:
            # Log the exception TYPE only — a pydantic/JSON error message can
            # embed the raw model output, which may quote the user (no-PHI
            # logs is a non-negotiable).
            logger.info("turn() attempt %d failed (%s)",
                        attempt + 1, type(e).__name__)
            messages.append({"role": "user", "content":
                             "Your last output was invalid. Output ONLY the "
                             "JSON object described, nothing else."})
    return TurnOut(action="unclear", reply="")


# --------------- Bounded single-utterance generation (P4) ---------------

_UTTER_SYSTEM = (
    "You are the voice of a structured SBIRT screening avatar. The clinical "
    "protocol — which question comes next, scores, risk zones, referrals — is "
    "decided by external code, NEVER by you. Produce exactly the single short "
    "utterance the INSTRUCTION asks for: one or two sentences, warm, "
    "plain-spoken, conversational, no clinical jargon, no scores or zone "
    "names, no new questions unless the instruction says to ask one. "
    "Output only the utterance text."
)


def phrase_utterance(instruction: str, history: list[dict],
                     patient: dict | None = None) -> str:
    """One bounded LLM utterance at the current protocol node (summaries,
    reflections, clarifications). Falls back to '' on failure — the caller
    treats an empty utterance as skippable, the protocol continues."""
    system = _UTTER_SYSTEM
    if patient:
        system += "\nKnown patient facts: " + json.dumps(patient, ensure_ascii=False)
    messages = ([{"role": "system", "content": system}]
                + list(history[-6:])
                + [{"role": "user", "content": f"INSTRUCTION: {instruction}"}])
    try:
        resp = _client().chat.completions.create(
            model=config.LLM_MODEL, messages=messages)
        return (resp.choices[0].message.content or "").strip()
    except Exception as e:
        logger.warning("phrase_utterance failed (%s); skipping utterance", e)
        return ""


if __name__ == "__main__":
    msgs = build_messages([{"role": "user", "content": "Hello, please briefly introduce yourself"}])
    print(f"LLM reply: {chat(msgs)}")
