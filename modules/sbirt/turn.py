"""The only channel through which a language model can reach the clinical engine.

Everything the model has to say about one utterance arrives as a single TurnOut:
what kind of thing was said, how it codes, and the words to speak back. There is
no other route in, and the model never sees or touches the protocol pointer.

validate() is the gate. Anything not provably a legal answer to the question
actually on the table is downgraded to "unclear", which holds position and
re-asks. So the engine advances on validated input only, and a confidently wrong
model produces a repeated question rather than a corrupted screening.

Pure: parsing and lookups, no model call and no I/O, so every rejection path can
be tested directly.
"""

from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, Field, field_validator

from . import coding
from .instruments import BY_KEY, PRE_SCREEN

# What an utterance is, relative to the question currently being asked. Only
# "answer" can move the protocol; everything else holds position.
#
#   answer       responds to the current ask, wholly or in part
#   continuation adds to their previous answer rather than this question, so
#                the addition is absorbed and the current ask stands
#   question     they are asking us something; answered from state, then the
#                ask is re-posed
#   tangent      an aside; acknowledged, then back to the ask
#   discomfort   they are unwell, worn out, or say they cannot face this now.
#                Not a crisis and not an abort: the engine offers to stop and
#                lets them choose, because pressing the question again is what
#                makes an interview feel deaf.
#   crisis       distress or danger. Ends the session: the engine speaks
#                the emergency numbers and closes, and does not counsel on.
#   abort        they want to stop entirely. Distinct from declining the
#                current permission gate, which is an ordinary answer — the
#                first ends the session, the second is a screening result.
#   correction   they are changing an answer already given to an earlier item,
#                which is overwritten with skips and scores re-derived
#   dont_know    they cannot or would rather not answer this one. Its own
#                outcome rather than "unclear", because the two need opposite
#                handling: unclear re-asks, this one offers a recall aid once
#                and then records the item missing and moves on.
#   unclear      nothing above can be safely assumed
Action = Literal["answer", "continuation", "question", "tangent", "discomfort",
                 "crisis", "abort", "correction", "dont_know", "unclear"]


class Harvest(BaseModel):
    """One fact the person stated about a question that is NOT on the table.

    Carried alongside the answer rather than instead of it, because people
    answer in paragraphs: "no cigarettes" arrives while a different question is
    pending, and "fentanyl, every day" answers three questions at once. Without
    somewhere to put those, the engine asks again for what it was just told.

    Nothing here is ever committed on the model's say-so — a candidate is read
    back to the person and only their yes writes it. `quote` is what makes that
    read-back possible: it is their own words, so they can recognise what is
    being confirmed.
    """

    model_config = {"extra": "forbid"}

    target: str            # a target key, as listed in the interview state
    code: int | None = None    # option and pre-screen targets
    text: str | None = None    # open targets
    quote: str = ""            # their words, for the read-back


class TurnOut(BaseModel):
    """Everything one model call is permitted to say about one utterance.

    Unknown fields are rejected rather than ignored, so a model that starts
    inventing keys fails to parse and is retried instead of quietly having its
    extra state accepted.
    """

    model_config = {"extra": "forbid"}

    action: Action
    # An option index for option items and permission gates (0 no, 1 yes), the
    # number itself for ruler asks, or the new option index for a correction.
    code: int | None = None
    # Corrections only: which already-answered item of the active instrument is
    # being changed.
    item: int | None = None
    # Open asks with declared slots: whichever ones this utterance filled.
    # validate() discards names that were not declared.
    slots: dict[str, str] = Field(default_factory=dict)
    # Answers to OTHER questions that this utterance happened to contain. Kept
    # through every action, including the ones that clear the payload: somebody
    # can volunteer a fact while asking a question or changing the subject, and
    # that is exactly when the old code threw it away.
    harvest: list[Harvest] = Field(default_factory=list)
    # Open asks without slots: the captured answer.
    text: str | None = None
    # Raw extraction for frequency and quantity items. The model reports what
    # was said — "every week" as 1 per week, "a liter of whiskey" as 1 liter of
    # whiskey — and coding.py decides which bucket that falls in. Splitting it
    # this way keeps a threshold decision, which can change a score, out of the
    # model's hands.
    value: float | None = None
    per: str | None = None
    unit: str | None = None
    beverage: str | None = None
    # What to say this turn, and the only thing the model speaks: the question
    # itself always comes from the protocol immediately afterwards, so this
    # never contains one.
    reply: str = ""
    # The utterance was the option's own wording, so there is nothing for a
    # read-back to confirm. Set by the deterministic pre-pass alone: llm.turn
    # clears it from model output, because a model asserting it would be
    # asserting its way past a confirmation step.
    exact: bool = False
    # Derived by validate(), and likewise cleared from model output. A unit
    # conversion was assumed, or the value sits close to a bucket boundary;
    # either earns a read-back, and `note` is how it is explained aloud.
    assumed: bool = False
    boundary: bool = False
    note: str = ""

    @field_validator("reply", "text", mode="before")
    @classmethod
    def _strip(cls, v):
        return v.strip() if isinstance(v, str) else v


def _unclear(out: TurnOut, why: str) -> TurnOut:
    """Strip everything but the reply and mark the turn unclear.

    The reply is kept deliberately: when the model could not code an answer it
    has usually already named what was missing, and that is worth saying before
    the engine puts the question back.
    """
    return out.model_copy(update={"action": "unclear", "code": None,
                                  "item": None, "slots": {}, "text": None,
                                  "value": None, "per": None, "unit": None,
                                  "beverage": None, "assumed": False,
                                  "boundary": False, "note": ""})
    # harvest is deliberately absent from that update: what they volunteered is
    # still true even when what they answered could not be coded.


def expected_item(expect):
    """The item an option expectation refers to, from either question source."""
    if expect.instrument == "prescreen":
        return PRE_SCREEN[expect.item_index].item
    return BY_KEY[expect.instrument].items[expect.item_index]


# A target key names one answerable thing, in the one spelling that the state
# view shows the model, validate() checks, and the engine writes through. Three
# shapes, because there are three places an answer can land:
#   prescreen.<key>      -> session.prescreen[key]
#   <instrument>.<index> -> session.responses[instrument][index]
#   <ask key>            -> session.answers[ask key]


def target_item(target: str):
    """The Item a target names, or None when it does not name a coded one."""
    head, _, tail = target.rpartition(".")
    if head == "prescreen":
        return next((q.item for q in PRE_SCREEN if q.key == tail), None)
    if head in BY_KEY and tail.isdigit():
        items = BY_KEY[head].items
        idx = int(tail)
        return items[idx] if idx < len(items) else None
    return None


def expect_target(expect) -> str | None:
    """The target key for the question currently on the table, if it has one."""
    if expect.kind == "option" and expect.instrument is not None:
        if expect.instrument == "prescreen":
            return f"prescreen.{PRE_SCREEN[expect.item_index].key}"
        return f"{expect.instrument}.{expect.item_index}"
    if expect.kind in ("open", "number"):
        return expect.ask_key
    return None


def validate_harvest(out: TurnOut, expect) -> list[Harvest]:
    """Keep only the volunteered facts that name something real.

    Shape only — whether a target is already answered needs the session, so
    runtime.record_harvest decides that. A bad entry is dropped on its own and
    never costs the answer it arrived with.
    """
    here = expect_target(expect)
    kept = []
    for h in out.harvest:
        if not h.target or h.target == here:
            continue          # the answer to the current ask is not a harvest
        item = target_item(h.target)
        if item is not None:
            if isinstance(h.code, int) and 0 <= h.code < len(item.options):
                kept.append(h)
            continue
        # An open target: it must carry words, and we cannot check its key
        # here — record_harvest drops keys the protocol does not know.
        if h.text:
            kept.append(h)
    return kept


_WORD = re.compile(r"[a-z0-9']+")
# A run this long shared with the pending question is a quotation, not a
# coincidence — six words of overlap happens, thirteen does not.
_ECHO_RUN = 7


def scrub_reply(reply: str, ask_text: str) -> str:
    """Drop any sentence of the reply that reads back the pending question.

    The engine speaks the question itself, so a reply that also contains it
    makes the person hear it twice. Enforced here rather than asked for in the
    prompt, because the prompt already asks and the model still does it.
    """
    if not reply or not ask_text:
        return reply
    ask = _WORD.findall(ask_text.lower())
    if len(ask) < _ECHO_RUN:
        return reply
    runs = {tuple(ask[i:i + _ECHO_RUN])
            for i in range(len(ask) - _ECHO_RUN + 1)}
    kept = []
    for sentence in re.split(r"(?<=[.!?])\s+", reply):
        words = _WORD.findall(sentence.lower())
        echo = any(tuple(words[i:i + _ECHO_RUN]) in runs
                   for i in range(len(words) - _ECHO_RUN + 1))
        if not echo:
            kept.append(sentence)
    return " ".join(kept).strip() or ""


def validate(out: TurnOut, expect, ask_text: str = "") -> TurnOut:
    """Downgrade anything that is not provably a legal answer to `expect`.

    Args:
        out: what the model produced.
        expect: the engine's current expectation.
        ask_text: the question on the table, so a reply that quotes it back can
            be stripped before anybody hears it twice.

    Returns:
        The same TurnOut when it is legal, otherwise one marked "unclear" with
        its payload cleared. Non-answer actions pass through with fields that
        are meaningless for them dropped.

    The model's claim to have answered something is never sufficient on its own;
    only what this function admits can move the protocol.
    """
    out = out.model_copy(update={
        "reply": scrub_reply(out.reply, ask_text),
        "harvest": validate_harvest(out, expect),
    })
    if out.action == "correction":
        # Only the shape can be checked here: the instrument is active and not
        # the pre-screen, the item exists, is not the one being asked right now,
        # and the new code is legal for it. Whether that item was ever actually
        # answered needs the session, so runtime.correct checks it and holds if
        # not.
        if (expect.kind == "option" and expect.instrument
                and expect.instrument != "prescreen"
                and isinstance(out.item, int) and isinstance(out.code, int)):
            items = BY_KEY[expect.instrument].items
            if (0 <= out.item < len(items)
                    and out.item != expect.item_index
                    and 0 <= out.code < len(items[out.item].options)):
                return out.model_copy(update={"slots": {}, "text": None})
        return _unclear(out, "correction needs a known earlier item + option")

    if out.action != "answer":
        # A continuation keeps its payload, which gets folded into the previous
        # capture. Nothing else has anything to say beyond its reply.
        if out.action == "continuation":
            return out
        if (out.code is not None or out.item is not None or out.slots
                or out.text or out.value is not None):
            return out.model_copy(update={"code": None, "item": None,
                                          "slots": {}, "text": None,
                                          "value": None, "per": None,
                                          "unit": None, "beverage": None})
        return out

    kind = expect.kind
    if kind in ("consent", "confirm"):
        if out.code in (0, 1):
            return out
        return _unclear(out, f"{kind} needs yes(1)/no(0)")
    if kind == "option":
        item = expected_item(expect)
        if item.coding != "choice" and not out.exact:
            # Frequency and quantity scales never accept a code the model chose:
            # it is computed from the extracted numbers, or the turn stays
            # unclear. A model-picked bucket would be legal and therefore
            # unreviewable when wrong, and these items decide the score.
            derived = coding.derive(item.coding, value=out.value,
                                    per=out.per, unit=out.unit,
                                    beverage=out.beverage)
            if derived is None:
                return _unclear(
                    out, f"{item.coding} answer not derivable from "
                         f"extraction (value={out.value!r} per={out.per!r} "
                         f"unit={out.unit!r} beverage={out.beverage!r})")
            return out.model_copy(update={
                "code": derived.code, "assumed": derived.assumed,
                "boundary": derived.boundary, "note": derived.note,
                "slots": {}, "text": None})
        if isinstance(out.code, int) and 0 <= out.code < len(item.options):
            return out
        return _unclear(out, "option code out of range")
    if kind == "number":
        if isinstance(out.code, int) and 0 <= out.code <= 10:
            return out
        return _unclear(out, "ruler needs 0-10")
    if kind == "open":
        # Reached by getattr rather than imported: runtime imports this module,
        # so importing runtime here would be circular.
        missing = tuple(getattr(expect, "missing", ()) or ())
        if missing:
            declared = set(getattr(expect, "slots", ()) or ())
            slots = {k: v for k, v in out.slots.items()
                     if k in declared and str(v).strip()}
            if slots:
                return out.model_copy(update={"slots": slots})
            # Answered as one unsplit phrase. It belongs to the slot just asked
            # for, which is the first still missing.
            if out.text:
                return out.model_copy(
                    update={"slots": {missing[0]: out.text}})
            return _unclear(out, "open slot answer captured nothing")
        if out.text:
            return out
        return _unclear(out, "open answer captured nothing")
    # The session is over, or the expectation is one this function does not
    # know: either way there is nothing an answer could advance.
    return _unclear(out, f"no answer possible at kind={kind!r}")
