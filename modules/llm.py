"""Every call this project makes to a language model.

Three distinct jobs, deliberately kept apart because they fail differently:

  * turn() — the one constrained call per user utterance. It classifies what
    the person said relative to the question on the table and codes it. Its
    output is always gated by turn.validate, so a bad model response holds the
    protocol in place rather than corrupting a screening.
  * phrase_utterance() — wording for a single utterance the protocol has
    already decided to deliver.
  * extract_patient_facts() — background profile extraction, the only
    open-ended generation left. Nothing here writes a whole reply any more:
    the protocol decides what is said, and a crisis closes the session with
    a fixed line.

The clinical protocol lives in modules/sbirt/ and is decided by code. Nothing
here chooses what to ask, what a score is, or where a session goes next.
"""

import json
import logging
import re
import threading
from openai import OpenAI
import config
from modules.sbirt.turn import TurnOut, expected_item
from modules.sbirt.turn import validate as validate_turn

logger = logging.getLogger(__name__)

# Built on first use, not at import: OpenAI("") raises, so a missing or
# misconfigured API key would otherwise take the whole server down at startup
# instead of failing the first request that needs a model.
_client_obj = None
_client_lock = threading.Lock()


def _client() -> OpenAI:
    """The shared API client, constructed on first use."""
    global _client_obj
    if _client_obj is None:
        with _client_lock:
            if _client_obj is None:
                _client_obj = OpenAI(
                    base_url=config.OPENROUTER_BASE_URL,
                    api_key=config.OPENROUTER_API_KEY,
                )
    return _client_obj


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
    """Pull whatever structured facts the recent conversation supports.

    Returns {} on any failure, including a malformed response. This runs in the
    background and its result only affects the next turn's prompt, so failing
    quietly is correct — it must never be able to break a turn in progress.
    """
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


# Coding a screening answer must never involve a guess: a quantity or timeframe
# the person did not actually state has to become a clarifying question, not a
# code, or the resulting score is invalid. Unambiguous short answers are matched
# deterministically here at no latency; everything else goes through one turn()
# call whose output is validated before it can move the protocol.

AMBIGUOUS = "AMBIGUOUS"

_YES_RE = re.compile(r"^\s*(yes|yeah|yep|yup|sure|correct|i do|i have)\b", re.I)
_NO_RE = re.compile(r"^\s*(no|nope|nah|never|not really|i don'?t|i do not|i haven'?t)\b", re.I)

_WORD_NUMBERS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
}
_DIGIT_RE = re.compile(r"\b(10|[0-9])\b")

def _prematch_option(options, text: str):
    """Code an option answer without a model call, or None if it is not obvious.

    Matches an option's label or alias exactly, plus yes/no shortcuts on binary
    items. The length limit on those shortcuts matters: a longer reply that
    merely starts with "yes" may carry a question or a caveat that the model
    needs to see.
    """
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
    """Read a single number out of a spoken ruler answer.

    Returns {"value": n} only when exactly one distinct in-range number was
    said, otherwise {"status": AMBIGUOUS}. Deliberately has no model fallback:
    two numbers in one sentence ("a four, maybe a seven") is a genuine ambiguity
    that must be asked about rather than resolved by inference.
    """
    t = user_text.lower()
    found = {int(m) for m in _DIGIT_RE.findall(t)}
    found |= {v for w, v in _WORD_NUMBERS.items()
              if re.search(rf"\b{w}\b", t)}
    found = {n for n in found if low <= n <= high}
    if len(found) == 1:
        return {"value": found.pop()}
    return {"status": AMBIGUOUS}


# Matched against the whole utterance, never as a prefix: "yes but what does
# that mean" is a question, and treating it as consent would skip the answer the
# person actually needs.
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
    """Describe, for the model, what a valid answer to the current ask looks like.

    Extraction items are the subtle case: for those the model is told to report
    raw fields and leave the option code null, because the bucketing is
    deterministic and a model that picks the bucket itself can silently shift a
    score across a threshold.
    """
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


# Whole-utterance matches only, for the same reason as the consent sets: "i
# don't know if that counts" carries content the model needs to see.
_DONT_KNOW = {"i don't know", "i dont know", "don't know", "dont know",
              "dunno", "i dunno", "no idea", "i have no idea", "not sure",
              "i'm not sure", "im not sure", "i can't remember",
              "i cant remember", "can't remember", "cant remember",
              "i don't remember", "i dont remember", "no clue",
              "i'd rather not say", "id rather not say", "rather not say"}


def _prepass(user_text: str, expect) -> TurnOut | None:
    """Answer without a model call when the utterance is unmistakable.

    Returns None when anything is in doubt, leaving the decision to turn(). The
    reply is left empty: no acknowledgment at all reads better than a canned one.
    """
    t = " ".join(user_text.strip().lower().split()).rstrip(".!,")
    if t in _DONT_KNOW and expect.kind in ("consent", "confirm", "option",
                                           "number"):
        # Distinct from "unclear": the pipeline offers one recall aid and then
        # records the item as missing, so this can never become a re-ask loop.
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
            # exact means the person said the option's own wording, so there is
            # nothing for a read-back to confirm and the code commits directly.
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
    """Classify, code and reply to one user utterance, in a single call.

    Args:
        user_text: what the person said.
        expect: the current expectation from the protocol state machine.
        ask_text: the question they are responding to.
        history: recent conversation, for context.
        patient: known patient facts, if any.
        facts: what the machine knows deterministically — the only permitted
            factual source for the reply.
        interview_state: a rendering of the whole interview, so meta questions
            ("how many are left") can be answered honestly.

    Returns:
        A TurnOut that has passed turn.validate. Anything illegal comes back as
        "unclear", which holds the protocol in place. Repeated failure returns
        unclear with an empty reply and the pipeline re-asks deterministically,
        so a misbehaving model can stall a turn but never strand a session.
    """
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
            # These fields decide whether an answer is read back to the person
            # for confirmation, and they are owned by the pre-pass and by
            # validate(). Clearing them stops a model from asserting certainty
            # it has not earned and skipping that confirmation.
            out = out.model_copy(update={"exact": False, "assumed": False,
                                         "boundary": False, "note": ""})
            return validate_turn(out, expect)
        except Exception as e:
            # Type only. A pydantic or JSON error message embeds the raw model
            # output, which can quote the patient verbatim into the log.
            logger.info("turn() attempt %d failed (%s)",
                        attempt + 1, type(e).__name__)
            messages.append({"role": "user", "content":
                             "Your last output was invalid. Output ONLY the "
                             "JSON object described, nothing else."})
    return TurnOut(action="unclear", reply="")


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
    """Word one utterance the protocol has already decided to deliver.

    Used for summaries, reflections and clarifications, where the content is
    fixed but the phrasing should suit the person. Returns "" on failure; the
    caller skips that utterance and the protocol continues rather than stalling.
    """
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
