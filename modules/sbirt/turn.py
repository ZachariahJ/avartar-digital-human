"""The per-turn NLU contract (T4): what one LLM call may tell the engine.

`TurnOut` is the ONLY channel from the language model into the clinical
engine. One call per user utterance produces BOTH the understanding (action +
coded answer/slots) and the words to speak for this turn (`reply`). The model
never sees or moves the protocol pointer: `validate()` is the pure gatekeeper
that downgrades anything not provably a legal answer for the CURRENT
expectation to `unclear`, so the engine only ever advances on validated input
(guess-free by construction, same contract the old per-kind coders had).

Pure module: pydantic parsing + instrument lookups. No LLM, no IO — every
downgrade branch is unit-testable.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator

from . import coding
from .instruments import BY_KEY, PRE_SCREEN

# What a turn's utterance can BE, relative to the machine's current ask:
#   answer       — answers the current ask (fully, or partly for slot asks)
#   continuation — adds to / completes the user's PREVIOUS answer (machine
#                  holds; the engine absorbs the addition, never re-asks)
#   question     — the user asks US something (reply answers it from state
#                  facts, then re-poses the ask; machine holds)
#   tangent      — off-topic aside (reply briefly acknowledges + returns;
#                  machine holds)
#   crisis       — distress/danger cue (deterministic crisis path takes over;
#                  UNION with crisis.detect — either may fire)
#   abort        — they want to STOP the whole conversation (T22): the engine
#                  closes gracefully with partial data kept; never a re-ask,
#                  never a retention attempt. Distinct from answering "no"
#                  to the current permission gate (that is an `answer`).
#   correction   — they are CHANGING an answer they already gave to an
#                  earlier item of the ACTIVE instrument (T21): `item` names
#                  the corrected item, `code` its new option. The engine
#                  overwrites, re-derives skips/score from the declarative
#                  data, and records old→new for audit.
#   dont_know    — they cannot or prefer not to answer THIS question ("i
#                  don't know", "can't remember", "skip that one") — a first-
#                  class result, NOT unclear (T25/F1). The engine probes once
#                  with a recall anchor (WHO manual p.18: estimate from the
#                  heaviest period in the past year), then marks the item
#                  missing and MOVES ON — never an endless re-ask. Distinct
#                  from abort (stopping everything) and from answering "no".
#   unclear      — none of the above is safe to assume (reply gently re-asks)
Action = Literal["answer", "continuation", "question", "tangent", "crisis",
                 "abort", "correction", "dont_know", "unclear"]


class TurnOut(BaseModel):
    """One turn's structured output. Extra fields are rejected so a drifting
    model response fails parsing (and retries) instead of smuggling state."""

    model_config = {"extra": "forbid"}

    action: Action
    # For option/yesno items and consent gates: the option INDEX (gates:
    # 0=no, 1=yes). For number asks (readiness ruler): the number itself.
    # For corrections: the corrected item's NEW option index.
    code: int | None = None
    # For corrections only: the 0-based index of the already-answered item
    # being corrected (within the active instrument).
    item: int | None = None
    # For open slot-asks: whichever declared slots the utterance filled,
    # e.g. {"drink": "whiskey", "amount": "12 oz"}. Unknown slot names are
    # dropped by validate(), never stored.
    slots: dict[str, str] = Field(default_factory=dict)
    # For open single-capture asks: the captured answer text.
    text: str | None = None
    # T26 extraction fields (the model EXTRACTS, coding.py buckets): for
    # frequency items `value` + `per` ("every week" -> 1, "week"); for
    # drink-quantity items `value` + `unit` [+ `beverage`] ("one liter of
    # whiskey" -> 1, "liter", "whiskey"; "ten or more" -> 10, "drinks").
    value: float | None = None
    per: str | None = None
    unit: str | None = None
    beverage: str | None = None
    # What the avatar says THIS turn. For `answer`: a short acknowledgment
    # only (the engine's own next utterances follow it). For question/
    # tangent/continuation/unclear: the complete bounded response (answer
    # the person / acknowledge / clarify, then re-pose the current ask).
    reply: str = ""
    # Set by the DETERMINISTIC pre-pass only (never trusted from model
    # output — llm.turn force-clears it after parsing): the utterance was
    # the option's exact wording, so a T20 confirm item commits without a
    # read-back.
    exact: bool = False
    # Set by validate()'s deterministic derivation only (also force-cleared
    # from model output): a conversion default entered the coding / the
    # value sat near a bucket edge — either one earns a read-back (F5) —
    # and the spoken conversion note for it (F6).
    assumed: bool = False
    boundary: bool = False
    note: str = ""

    @field_validator("reply", "text", mode="before")
    @classmethod
    def _strip(cls, v):
        return v.strip() if isinstance(v, str) else v


def _unclear(out: TurnOut, why: str) -> TurnOut:
    """Downgrade to unclear, KEEPING the model's reply (it is usually already
    a sensible clarification); the engine treats unclear as hold-and-re-ask."""
    return out.model_copy(update={"action": "unclear", "code": None,
                                  "item": None, "slots": {}, "text": None,
                                  "value": None, "per": None, "unit": None,
                                  "beverage": None, "assumed": False,
                                  "boundary": False, "note": ""})


def expected_item(expect):
    """The Item under an option expectation (prescreen or instrument)."""
    if expect.instrument == "prescreen":
        return PRE_SCREEN[expect.item_index].item
    return BY_KEY[expect.instrument].items[expect.item_index]


def validate(out: TurnOut, expect) -> TurnOut:
    """Pure gatekeeper: an `answer` that is not a legal answer for the CURRENT
    expectation is downgraded to `unclear` (T4 rule — the pointer only ever
    moves on validated input; the model's claim alone moves nothing).
    Non-answer actions pass through with their payload fields cleared where
    meaningless. `expect` is the engine's runtime.Expect."""
    if out.action == "correction":
        # T21: shape-gate only what turn.py CAN know without the session —
        # active non-prescreen instrument, a real item index other than the
        # one currently being asked, a legal option code for THAT item.
        # Whether the item was actually ANSWERED is the engine's check
        # (runtime.correct); an inapplicable correction holds, never moves.
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
        # Continuations carry slots/text (absorbed into the PREVIOUS capture);
        # everything else carries only a reply.
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
            # T26/F4: for frequency/quantity scales a semantic answer is
            # never taken as a model-claimed code — the code is COMPUTED
            # from the extracted fields (coding.py), or the turn stays
            # unclear. "Legal but wrong" buckets cannot pass here.
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
        # Slot metadata lives on the engine's Expect; accessed via getattr so
        # this module never imports runtime (runtime imports us).
        missing = tuple(getattr(expect, "missing", ()) or ())
        if missing:                      # slot ask: keep only declared slots
            declared = set(getattr(expect, "slots", ()) or ())
            slots = {k: v for k, v in out.slots.items()
                     if k in declared and str(v).strip()}
            if slots:
                return out.model_copy(update={"slots": slots})
            if out.text:                 # whole answer, unsplit → the slot we
                return out.model_copy(   # just asked for (declared order)
                    update={"slots": {missing[0]: out.text}})
            return _unclear(out, "open slot answer captured nothing")
        if out.text:
            return out
        return _unclear(out, "open answer captured nothing")
    # kind == "end" (session over) or unknown: nothing advances.
    return _unclear(out, f"no answer possible at kind={kind!r}")
