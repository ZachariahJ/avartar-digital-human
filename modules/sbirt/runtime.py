"""Session state, and the writes that move it.

Nothing here decides what comes next. Position is derived by `select` from the
answers collected plus the ledger of what was actually voiced, so this module
only ever records: an answer, a refusal, a read-back, a correction, a latch.

That split is the repair. The previous engine advanced a program counter before
the audio was delivered, so a question lost to a barge-in was recorded as asked
and never came back. There is no counter left to run ahead.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field as dc_field

from . import coding, select as sel, templates
from .form import Field, OPEN_TARGETS
from .instruments import BY_KEY, option_score, PRE_SCREEN
from .runtime_types import Expect, LLMSay, ProtocolError, Say, Speak
from .turn import TurnOut, target_item

logger = logging.getLogger(__name__)

__all__ = ["Say", "LLMSay", "Speak", "Expect", "ProtocolError",
           "ClinicalSession", "record", "mark_missing", "repose",
           "note_stall", "note_misread", "note_aside",
           "DONT_KNOW_LIMIT", "UNCLEAR_LIMIT", "MISREAD_LIMIT",
           "ASIDE_LIMIT", "confirm_reason", "request_confirm",
           "resolve_confirm", "record_harvest", "absorb",
           "correct", "enter_crisis", "enter_abort", "offer_pause",
           "resolve_pause", "OPEN_TARGETS"]


@dataclass
class ClinicalSession:
    """Everything the interview has learned, and nothing about where it is.

    The deleted fields are the point: pc, node, expect, last_step, arm, arms,
    covered and assessments were all forms of remembered position, and every one
    of them could get ahead of what the person had actually heard. They are
    derived now — see modules/sbirt/select.py.
    """

    # Answers. The only thing position is derived from.
    prescreen: dict[str, int] = dc_field(default_factory=dict)
    responses: dict[str, dict[int, int]] = dc_field(default_factory=dict)
    readiness: dict[str, int] = dc_field(default_factory=dict)
    answers: dict[str, str] = dc_field(default_factory=dict)
    slots: dict[str, dict[str, str]] = dc_field(default_factory=dict)
    declined: list[str] = dc_field(default_factory=list)
    missing: dict[str, dict] = dc_field(default_factory=dict)

    # The voiced ledger: units confirmed to have reached the person. Written
    # only after delivery is acknowledged, never while composing a line.
    spoken: set[str] = dc_field(default_factory=set)

    # Latches that put an interrupt form ahead of document order.
    crisis: bool = False
    aborted: bool = False
    pause_pending: bool = False

    # Side records, unchanged by the rewrite.
    corrections: list[dict] = dc_field(default_factory=list)
    inconsistencies: list[dict] = dc_field(default_factory=list)
    fired_rules: set[str] = dc_field(default_factory=set)
    candidates: dict[str, dict] = dc_field(default_factory=dict)
    refused_candidates: set = dc_field(default_factory=set)
    pending_confirm: dict | None = None
    last_ask_key: str | None = None
    stalls: int = 0
    misreads: int = 0
    asides: int = 0
    reasks: dict[str, int] = dc_field(default_factory=dict)

    @property
    def consent(self) -> str | None:
        return self.answers.get("consent")

    @property
    def node(self) -> str:
        return sel.current_node(self)

    @property
    def expect(self) -> Expect:
        return sel.current_expect(self)

    @property
    def arm(self) -> str | None:
        return sel.current_arm(self)

    @property
    def assessments(self) -> dict:
        return sel.assessments(self)


# --- recording an answer ----------------------------------------------------


def record(session: ClinicalSession, field: Field, out: TurnOut) -> None:
    """Apply one validated answer to one field.

    Raises on a payload that does not fit the field: turn.validate has already
    downgraded anything ambiguous, so a mismatch here is wiring gone wrong
    rather than a person being unclear.
    """
    session.stalls = session.misreads = 0
    session.asides = 0
    session.reasks.pop(sel.unit_id(session, field), None)

    if field.kind == "consent":
        if out.code not in (0, 1):
            raise ProtocolError(
                f"gate {field.id!r} needs yes(1)/no(0), got {out.code!r}")
        yes = out.code == 1
        session.answers[field.slot] = "yes" if yes else "no"
        if not yes:
            session.declined.append(field.slot)
            logger.info("[clinical] declined: %s", field.slot)
        return

    if field.kind == "option":
        if not isinstance(out.code, int):
            raise ProtocolError(f"item {field.id!r} needs an int code")
        if field.instrument == "prescreen":
            q = PRE_SCREEN[field.item_index]
            if not 0 <= out.code < len(q.item.options):
                raise ProtocolError(f"prescreen {q.key}: invalid code {out.code}")
            session.prescreen[q.key] = out.code
            return
        instrument = BY_KEY[field.instrument]
        option_score(instrument, field.item_index, out.code)
        session.responses.setdefault(field.instrument, {})[field.item_index] = out.code
        return

    if field.kind == "number":
        if not isinstance(out.code, int) or not 0 <= out.code <= 10:
            raise ProtocolError(f"ruler needs 0-10, got {out.code!r}")
        session.answers[field.slot] = str(out.code)
        if field.arm:
            session.readiness[field.arm] = out.code
        session.last_ask_key = field.slot
        return

    if field.kind == "open":
        if field.slots:
            if not out.slots:
                raise ProtocolError(
                    f"slot ask {field.slot!r} got no slots — unvalidated input?")
            store = session.slots.setdefault(field.slot, {})
            store.update(out.slots)
            session.last_ask_key = field.slot
            if all(name in store for name in field.slots):
                session.answers[field.slot] = "; ".join(
                    f"{name}: {store[name]}" for name in field.slots)
            return
        if not out.text:
            raise ProtocolError(f"open ask {field.slot!r} got empty capture")
        session.answers[field.slot] = out.text
        session.last_ask_key = field.slot
        return

    raise ProtocolError(f"field {field.id!r} of kind {field.kind!r} takes no answer")


def repose(session: ClinicalSession, unit: str) -> None:
    """Owe a question again.

    Dropping the ledger entry is the whole re-ask mechanism: the next selection
    finds the field unspoken and delivers it. There is nothing to rewind because
    nothing moved.

    The tally is what stops the repeat being word-for-word: wording that failed
    once fails again, and hearing it verbatim twice sounds broken.
    """
    session.reasks[unit] = session.reasks.get(unit, 0) + 1
    session.spoken.discard(unit)


# --- unanswerable -----------------------------------------------------------

DONT_KNOW_LIMIT = 2
UNCLEAR_LIMIT = 3
MISREAD_LIMIT = 3
ASIDE_LIMIT = 2


def note_aside(session: ClinicalSession) -> int:
    session.asides += 1
    return session.asides


def note_stall(session: ClinicalSession) -> int:
    session.stalls += 1
    return session.stalls


def note_misread(session: ClinicalSession) -> int:
    """Count an output the engine could not read, against the engine.

    Separate from `stalls` on purpose: a model that keeps returning an unusable
    shape must not spend the person's allowance for not knowing an answer. It
    still needs a bound of its own, or a question the model can never encode
    traps them on it forever.
    """
    session.misreads += 1
    return session.misreads


def mark_missing(session: ClinicalSession, field: Field,
                 reason: str = "no_answer") -> tuple:
    """Record that a field cannot be answered, and let selection move on.

    Every branch writes a slot, so the field simply falls out of scope — the old
    jump to a decline label is now just the write that a spoken "no" would have
    made. Returns the beats that acknowledge the skip, if any.

    This is driven by repeated dont_know/unclear ANSWERS. Silence never reaches
    here: a question nobody responds to is re-asked once and then waited on.
    """
    session.stalls = session.misreads = 0
    skip_line = (Say("item.skipped", templates.FIXED["item.skipped"]),)

    if field.kind == "confirm":
        p = session.pending_confirm
        session.pending_confirm = None
        if p is not None:
            session.candidates.pop(p["target"], None)
            head, _, tail = p["target"].rpartition(".")
            if head in BY_KEY and tail.isdigit():
                session.missing.setdefault(head, {})[int(tail)] = "unconfirmed"
            logger.info("[clinical] confirm unresolvable: %s marked missing",
                        p["target"])
        return skip_line

    if field.kind == "option":
        if field.instrument == "prescreen":
            q = PRE_SCREEN[field.item_index]
            session.prescreen[q.key] = 0
            session.missing.setdefault("prescreen", {})[field.item_index] = reason
        else:
            session.missing.setdefault(
                field.instrument, {})[field.item_index] = reason
        logger.info("[clinical] item missing (%s): %s", reason, field.id)
        return skip_line

    if field.kind == "consent":
        session.answers[field.slot] = "no"
        session.declined.append(field.slot)
        session.missing.setdefault("gates", {})[field.slot] = reason
        logger.info("[clinical] gate unanswerable (%s): %s -> decline path",
                    reason, field.slot)
        return ()

    if field.kind in ("open", "number"):
        session.missing.setdefault("asks", {})[field.slot] = reason
        if field.kind != "number":
            session.answers.setdefault(field.slot, "(not answered)")
        else:
            session.answers.setdefault(field.slot, "")
        logger.info("[clinical] ask unanswerable (%s): %s", reason, field.slot)
        return skip_line

    return ()


# --- read-backs -------------------------------------------------------------


def _conflict(session: ClinicalSession, field: Field,
              out: TurnOut) -> tuple[str, str] | None:
    """Whether this answer contradicts one already given.

    Each rule surfaces at most once per session so nobody is challenged
    repeatedly, but every detection is recorded either way so the provider sees
    the contradiction even if the person stands by both answers.
    """
    if field.instrument != "audit" or not isinstance(out.code, int):
        return None
    items = BY_KEY["audit"].items
    responses = session.responses.get("audit", {})

    if (field.item_index == 2 and 0 in responses
            and coding.Q3_MIN_PER_WEEK[out.code]
            > coding.Q1_MAX_PER_WEEK[responses[0]]):
        rule = "audit.q3_vs_q1"
        session.inconsistencies.append(
            {"rule": rule, "item": 2, "code": out.code,
             "against_item": 0, "against_code": responses[0]})
        if rule in session.fired_rules:
            return None
        session.fired_rules.add(rule)
        q1 = items[0].options[responses[0]].label.lower()
        return rule, f"you drink about {q1}"

    if field.item_index == 0:
        qf = session.slots.get("alcohol.qf", {}).get("frequency", "")
        rate = coding.parse_freq_text(qf)
        if (rate is not None
                and rate > coding.Q1_MAX_PER_WEEK[out.code]
                * coding.QF_VS_Q1_MARGIN + 0.1):
            rule = "audit.q1_vs_qf"
            session.inconsistencies.append(
                {"rule": rule, "item": 0, "code": out.code,
                 "against": "alcohol.qf.frequency"})
            if rule in session.fired_rules:
                return None
            session.fired_rules.add(rule)
            return rule, f"you usually drink {qf}"
    return None


def confirm_reason(session: ClinicalSession, field: Field,
                   out: TurnOut) -> dict | None:
    """Whether this answer should be read back before it is committed.

    Reading answers back is the exception rather than the rule: a screening that
    confirms everything is tedious enough that people stop listening to it.
    """
    if (field.kind != "option" or not field.instrument
            or field.instrument == "prescreen"
            or not isinstance(out.code, int)):
        return None
    conflict = _conflict(session, field, out)
    if conflict:
        return {"reason": "conflict", "rule": conflict[0], "prior": conflict[1]}
    if out.exact:
        return None
    item = BY_KEY[field.instrument].items[field.item_index]
    if not item.confirm:
        return None
    if out.assumed or out.boundary:
        return {"reason": "conversion" if out.assumed else "boundary",
                "note": out.note}
    if item.coding == "choice":
        return {"reason": "semantic"}
    return None


def write_target(session: ClinicalSession, target: str,
                 code: int | None, text: str | None) -> None:
    head, _, tail = target.rpartition(".")
    if head == "prescreen":
        session.prescreen[tail] = code
    elif head in BY_KEY and tail.isdigit():
        session.responses.setdefault(head, {})[int(tail)] = code
    else:
        session.answers[target] = text


def request_confirm(session: ClinicalSession, field: Field, out: TurnOut,
                    reason: dict | None = None) -> None:
    session.pending_confirm = {"target": field.slot, "code": out.code,
                               "text": out.text, "source": "asked",
                               **(reason or {})}


def request_volunteered_confirm(session: ClinicalSession, target: str) -> None:
    """Put a volunteered answer to the person before anything is recorded.

    Nothing a model harvested is ever committed on its own say-so; the quote is
    what lets the person recognise what they are being asked to confirm.
    """
    session.pending_confirm = {"target": target, "source": "volunteered",
                               **session.candidates[target]}
    logger.info("[clinical] reading back a volunteered answer for %s", target)


def resolve_confirm(session: ClinicalSession, yes: bool) -> None:
    p = session.pending_confirm
    if p is None:
        raise ProtocolError("confirm resolution without a pending answer")
    session.pending_confirm = None
    session.stalls = session.misreads = 0
    target = p["target"]
    session.candidates.pop(target, None)
    session.spoken.discard(f"confirm.{target}")
    if yes:
        write_target(session, target, p.get("code"), p.get("text"))
        logger.info("[clinical] confirm accepted: %s (%s)", target, p["source"])
    else:
        if p["source"] == "volunteered":
            session.refused_candidates.add(target)
        logger.info("[clinical] confirm DENIED: %s re-collected", target)


# --- volunteered answers, continuations, corrections ------------------------


def record_harvest(session: ClinicalSession, out: TurnOut) -> list[str]:
    """Hold on to what the person said about questions that were not on the table.

    People answer in paragraphs. Anything dropped here is asked for again as if
    they had never said it.
    """
    kept = []
    for h in out.harvest:
        target = h.target
        if (target in session.refused_candidates
                or sel.target_answered(session, target)):
            continue
        if target_item(target) is None and target not in OPEN_TARGETS:
            continue
        session.candidates[target] = {"code": h.code, "text": h.text,
                                      "quote": h.quote}
        kept.append(target)
    if kept:
        logger.info("[clinical] volunteered answers noted for %s", kept)
    return kept


def absorb(session: ClinicalSession, out: TurnOut) -> None:
    """Fold a second breath into the answer it continues."""
    key = session.last_ask_key
    if key is None:
        return
    if out.slots:
        session.slots.setdefault(key, {}).update(out.slots)
    if out.text:
        prev = session.answers.get(key, "")
        session.answers[key] = (prev + " " + out.text).strip()


def correct(session: ClinicalSession, field: Field, out: TurnOut) -> bool:
    """Overwrite an earlier answer and let skips and scores re-derive.

    Returns False when the correction names nothing that can be changed, so the
    caller holds position and asks which answer was meant.
    """
    if field.kind != "option" or not field.instrument \
            or field.instrument == "prescreen":
        return False
    responses = session.responses.get(field.instrument, {})
    if out.item not in responses or not isinstance(out.code, int):
        return False
    instrument = BY_KEY[field.instrument]
    option_score(instrument, out.item, out.code)
    old = responses[out.item]
    if old == out.code:
        return True
    session.stalls = session.misreads = 0
    responses[out.item] = out.code
    session.corrections.append({"instrument": field.instrument,
                                "item": out.item, "old": old, "new": out.code})
    logger.info("[clinical] correction: %s item %d code %d -> %d",
                field.instrument, out.item, old, out.code)
    return True


# --- latches ----------------------------------------------------------------


def enter_crisis(session: ClinicalSession) -> None:
    session.crisis = True
    logger.warning("[clinical] session closed on crisis at %s", session.node)


def enter_abort(session: ClinicalSession) -> None:
    session.aborted = True
    logger.info("[clinical] session aborted by user at %s", session.node)


def offer_pause(session: ClinicalSession) -> None:
    session.asides = 0
    session.pause_pending = True
    logger.info("[clinical] offering to pause at %s", session.node)


def resolve_pause(session: ClinicalSession, keep_going: bool) -> None:
    """Settle the offer to stop.

    Carrying on clears the offer completely — the answer and its ledger entry
    both — so a later moment of discomfort can offer again cleanly rather than
    finding the question already answered.
    """
    session.pause_pending = False
    session.answers.pop("pause.offer", None)
    session.spoken.discard("pause.offer")
    if not keep_going:
        enter_abort(session)
        return
    session.stalls = session.misreads = 0
