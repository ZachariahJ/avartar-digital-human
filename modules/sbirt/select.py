"""Where the interview stands, derived rather than remembered.

Everything here is a pure read of a ClinicalSession. There is no pointer to
advance and nothing to keep in step with the audio: `select` walks the form
table every time it is asked, so a prompt that was never voiced is simply
selected again on the next tick.

The one rule this module must never break is purity. The tick loop calls
`select` several times per iteration — to decide what to deliver, to render
the interview state for the NLU — and the old Route
functions mutated on the way past, which is precisely why a re-entered walk
could not find its way back into the arm it was already inside.
"""

from __future__ import annotations

import copy

from .form import (ARM_INSTRUMENT, ARM_ORDER, Field, FORMS, INTERRUPTS,
                   Prompt)
from .instruments import assess, BY_KEY, next_item_index
from .runtime_types import Expect


def unit_id(session, field: Field) -> str:
    """The field's identity in the voiced ledger.

    A variant extends it, so one field can owe several distinct utterances over
    its life: one per missing slot, one per risk zone. Without that the whole
    multi-slot ask would enter the ledger after its first sub-question.
    """
    if field.variant is None:
        return field.id
    return f"{field.id}:{field.variant(session)}"


def target_answered(session, target: str) -> bool:
    """Whether a slot address already holds an answer.

    Three shapes, because there are three places an answer can land:
    prescreen.<key>, <instrument>.<index>, and everything else in answers.
    """
    head, _, tail = target.rpartition(".")
    if head == "prescreen":
        return tail in session.prescreen
    if head in BY_KEY and tail.isdigit():
        return int(tail) in session.responses.get(head, {})
    return target in session.answers


def filled(session, field: Field) -> bool:
    """Whether this field has nothing left to contribute.

    A "tell" is filled by having been VOICED, not by having been composed. That
    inversion is the whole repair: the old engine marked an education line
    covered while it was still assembling the beat, so a line lost to a barge-in
    was recorded as delivered and the follow-up question then referred back to a
    definition nobody had heard.
    """
    if field.kind in ("tell", "end"):
        return unit_id(session, field) in session.spoken
    if field.kind == "confirm":
        # Cleared by whoever resolves the read-back, never by being spoken.
        return False
    if field.slots:
        got = session.slots.get(field.slot, {})
        return all(name in got for name in field.slots)
    return target_answered(session, field.slot)


def _confirm_field(session) -> Field | None:
    p = session.pending_confirm
    if p is None:
        return None
    head, _, tail = p["target"].rpartition(".")
    if head in BY_KEY and tail.isdigit():
        arm = next(a for a, k in ARM_INSTRUMENT.items() if k == head)
        return Field(f"confirm.{p['target']}", kind="confirm", prompt=Prompt(),
                     arm=arm, instrument=head, item_index=int(tail))
    return Field(f"confirm.{p['target']}", kind="confirm", prompt=Prompt())


def select(session) -> Field | None:
    """The first field, in document order, that is unfilled and in scope.

    Returns None when the interview has nothing left to say or collect.
    """
    confirm = _confirm_field(session)
    if confirm is not None:
        return confirm
    for form in INTERRUPTS + FORMS:
        if not form.when(session):
            continue
        for fld in form.fields:
            if filled(session, fld) or not fld.when(session):
                continue
            return fld
        if form.terminal:
            # The interview is over. Falling through from here is what let a
            # closed session start re-asking the questions it had just closed on.
            return None
    return None


def answerable(session) -> Field | None:
    """The field an answer would land on, looking past anything still to say.

    While a Tell is being voiced the person can already answer the question
    behind it, so the NLU has to be listening for that answer now. The old
    interpreter got this for free by running through every Tell to the next
    pause before it spoke a word.

    Walks a copy so the hypothesis that the pending Tells were heard cannot
    leak into the real ledger.
    """
    if session.pending_confirm is not None:
        return select(session)
    probe = copy.copy(session)
    probe.spoken = set(session.spoken)
    while True:
        fld = select(probe)
        if fld is None or fld.kind == "end":
            return None
        if fld.kind != "tell":
            return fld
        probe.spoken.add(unit_id(probe, fld))


def missing_slots(session, field: Field) -> tuple[str, ...]:
    if not field.slots:
        return ()
    got = session.slots.get(field.slot, {})
    return tuple(name for name in field.slots if name not in got)


def field_expect(session, field: Field | None) -> Expect:
    """The answer shape the NLU and turn.validate are handed.

    Expect is unchanged from the previous engine on purpose: it is the seam that
    let the protocol be rewritten underneath llm.turn and turn.validate without
    touching either.
    """
    if field is None:
        return Expect("end")
    if field.kind == "confirm":
        return Expect("confirm", instrument=field.instrument or None,
                      item_index=field.item_index,
                      ask_key=session.pending_confirm["target"])
    if field.kind == "consent":
        return Expect("consent", ask_key=field.slot)
    if field.kind == "option":
        return Expect("option", instrument=field.instrument,
                      item_index=field.item_index)
    if field.kind == "number":
        return Expect("number", ask_key=field.slot)
    if field.kind == "open":
        return Expect("open", ask_key=field.slot, slots=field.slots,
                      missing=missing_slots(session, field))
    return Expect("end")


def current_expect(session) -> Expect:
    return field_expect(session, answerable(session))


def current_node(session) -> str:
    """A human-readable name for where the session stands, for logs and audit."""
    if session.crisis:
        return "crisis"
    if session.aborted:
        return "aborted"
    fld = select(session)
    return fld.id if fld is not None else "closed"


def current_arm(session) -> str | None:
    """Which arm is being worked, derived from what is in scope.

    The old engine popped this off a queue as it went, so once an arm was
    entered the fact that it had been queued survived nowhere.
    """
    fld = answerable(session) or select(session)
    if fld is not None and fld.arm:
        return fld.arm
    return None


def assessment_for(session, ins_key: str):
    """The scored assessment for an instrument, or None while it is unfinished.

    Derived on every read. Storing it made the risk zone — like the program
    counter — a side effect of having passed a particular instruction.
    """
    responses = session.responses.get(ins_key, {})
    missing = session.missing.get(ins_key, {})
    if not responses and not missing:
        return None
    if next_item_index(BY_KEY[ins_key], responses, missing) is not None:
        return None
    return assess(BY_KEY[ins_key], responses, missing)


def assessments(session) -> dict:
    out = {}
    for arm in ARM_ORDER:
        ins_key = ARM_INSTRUMENT[arm]
        a = assessment_for(session, ins_key)
        if a is not None:
            out[ins_key] = a
    return out


def candidate_for(session, target: str) -> str | None:
    """A volunteered answer waiting to be read back for this slot."""
    if not target:
        return None
    if target in session.candidates and target not in session.refused_candidates:
        return target
    return None
