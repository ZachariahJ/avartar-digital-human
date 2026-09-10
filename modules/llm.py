
import json
import logging
import re
import threading
from openai import OpenAI
import config
from modules.sbirt.turn import TurnOut, expected_item
from modules.sbirt.turn import validate as validate_turn

logger = logging.getLogger(__name__)

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



AMBIGUOUS = "AMBIGUOUS"

_YES_RE = re.compile(r"^\s*(yes|yeah|yep|yup|sure|correct|i do|i have)\b", re.I)
_NO_RE = re.compile(r"^\s*(no|nope|nah|never|not really|i don'?t|i do not|i haven'?t)\b", re.I)

_WORD_NUMBERS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
}
_DIGIT_RE = re.compile(r"\b(10|[0-9])\b")

def _prematch_option(options, text: str):
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
    t = user_text.lower()
    found = {int(m) for m in _DIGIT_RE.findall(t)}
    found |= {v for w, v in _WORD_NUMBERS.items()
              if re.search(rf"\b{w}\b", t)}
    found = {n for n in found if low <= n <= high}
    if len(found) == 1:
        return {"value": found.pop()}
    return {"status": AMBIGUOUS}


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

Session facts — the ONLY source for anything you say about THIS person:
{facts}

Live interview state (program-rendered from the session every turn — treat it
as session facts too). It is the full map of the screening: every question,
what is already answered, what is being asked right now, and what comes next.
Use it to answer meta questions ("how many questions are left", "what is this
for", "didn't I already tell you that") and to avoid re-asking anything it
shows as answered:
{interview_state}

Read the user's utterance and output ONLY a compact JSON object:
{{"action": "...", "code": null, "item": null, "slots": {{}}, "text": null, "value": null, "per": null, "unit": null, "beverage": null, "harvest": [], "reply": "..."}}

"harvest" — facts they stated about questions OTHER than the one on the table.
People answer in paragraphs, and anything you leave out here is lost: the
engine will ask them for it again as if they had never said it. So whenever
the utterance settles a question listed under TARGET KEYS above, add an entry:
  {{"target": "<a key from TARGET KEYS>", "code": <option number>,
    "text": "<for open questions only>", "quote": "<their own words>"}}
- Use "code" for pre-screen and instrument items, "text" for open questions.
- "quote" is what they actually said, short — the engine reads it back to
  them before it records anything, so it must be recognisably theirs.
- NEVER harvest the question currently on the table; that is what the fields
  above are for. Never harvest a question the state shows as answered.
- Same certainty bar as coding: if it fits more than one option, leave it out.
- Nothing here is committed on your say-so. Every entry is read back to the
  person for a yes first, so a harvest is a proposal, not a recording.

action — exactly one of:
  "answer"       it answers the current ask (even partially for slots)
  "continuation" it adds to / completes their PREVIOUS answer instead
  "question"     they are asking YOU something
  "tangent"      an off-topic aside or small talk
  "discomfort"   they say they are unwell, exhausted, in pain, or that they
                 cannot face this right now ("i'm not feeling well", "i'm
                 sick today", "i'm too tired for this") — physical or
                 emotional, but NOT danger and NOT a request to stop
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

"reply" is YOUR WORDS TO THE PERSON, and it is the only thing you say.

THE ONE RULE ABOUT QUESTIONS: never write a screening question yourself, not
even reworded, shortened or hinted at. The engine speaks the question — as
validated, word for word — immediately after your reply, every single turn.
So write only what comes BEFORE that question, and never end your reply with
the question itself. Writing it yourself makes the person hear it twice.

What to put there, by action:
- answer: acknowledge what they said in your own fresh words, one short
  sentence. If they also asked something, answer that too. Never reuse an
  acknowledgment already used in this conversation.
- question: ANSWER THEM. This is the whole point of your presence in this
  turn — one or two short, plain sentences, then stop. A person who asks
  what a word means, why you are asking, whether something counts, or what
  happens to their answers deserves a real answer, not a request to
  rephrase themselves. Where they ask about the interview itself — how far
  along, what is left, what they already said, what has been explained —
  answer from the session facts and interview state above. Where they ask
  what an ordinary word means ("what counts as a tobacco product", "what is
  a standard drink"), just explain it plainly, the way a nurse would.
  If you genuinely do not know, say so in one sentence.
- tangent: one warm sentence acknowledging what they said, and stop.
- discomfort: one short sentence naming what they actually said — the
  specific thing, not the category — and stop. The engine offers to stop
  for today straight after you, so do not offer it yourself and do not
  reassure them that it will be quick.
- continuation: acknowledge the added detail in a few words, and stop.
- correction: say the new answer back in a few words to confirm it.
- unclear: name in ONE short sentence exactly what you still need. If their
  words sit between specific choices, name those choices ("Would that be
  five or six drinks, or more like seven to nine?"); if they answered a
  different dimension, say which one you meant ("I meant how many drinks at
  one time, rather than how often."). Never reuse a clarification already
  used in this conversation — each attempt gets more specific. Never
  suggest which choice to pick.
- crisis / abort / dont_know: leave reply empty — a fixed response takes
  over (for dont_know the protocol itself offers a recall anchor or moves on).

Facts you may state: what the session facts and interview state above show,
and ordinary general knowledge about everyday words and how screening works.
Facts you may NEVER state: anything about THIS person that is not in the
state above, any score, risk zone, diagnosis or interpretation of their
answers, any promise about what happens next, and any advice about their
substance use. Those are the engine's, not yours.
- NEVER: clinical jargon, lecturing, stacked questions, or stock phrases
  like "I hear you."
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


_DONT_KNOW = {"i don't know", "i dont know", "don't know", "dont know",
              "dunno", "i dunno", "no idea", "i have no idea", "not sure",
              "i'm not sure", "im not sure", "i can't remember",
              "i cant remember", "can't remember", "cant remember",
              "i don't remember", "i dont remember", "no clue",
              "i'd rather not say", "id rather not say", "rather not say"}


def _prepass(user_text: str, expect) -> TurnOut | None:
    t = " ".join(user_text.strip().lower().split()).rstrip(".!,")
    if t in _DONT_KNOW and expect.kind in ("consent", "confirm", "option",
                                           "number"):
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
    pre = _prepass(user_text, expect)
    if pre is not None:
        return validate_turn(pre, expect, ask_text=ask_text)

    all_facts = dict(facts or {})
    if patient:
        all_facts["patient"] = patient
    system = _TURN_SYSTEM.format(
        ask=ask_text or "(the opening consent question)",
        expectation=_expectation_text(expect),
        facts=json.dumps(all_facts, ensure_ascii=False),
        interview_state=interview_state or "(not available this turn)")
    messages = ([{"role": "system", "content": system}]
                + list(history)
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
            out = out.model_copy(update={"exact": False, "assumed": False,
                                         "boundary": False, "note": ""})
            return validate_turn(out, expect, ask_text=ask_text)
        except Exception as e:
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
    "Say nothing the conversation above has already said this turn: do not "
    "restate what the person just told you and do not repeat an "
    "acknowledgment that has already been given — the last assistant message "
    "may be the first half of the very utterance you are finishing. "
    "Output only the utterance text."
)


def phrase_utterance(instruction: str, history: list[dict],
                     patient: dict | None = None) -> str:
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
