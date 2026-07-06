"""The generic turn engine (T6): ONE interpreter over the declarative
protocol program in flow.py — no per-question handler code anywhere.

Until now every protocol node had its own `_on_xxx` handler that hard-wired
the next node. Now the protocol IS data (flow.PROTOCOL) and this module is
the only executor: it walks steps, collects the utterances of a turn, pauses
wherever user input is needed, and consumes exactly ONE validated TurnOut per
turn. The LLM's understanding of an utterance arrives ONLY as a validated
`turn.TurnOut`; the pointer, the scores, the zones, the skip rules and every
branch remain deterministic code (instruments.py / flow.py).

Contract with the pipeline:
  • `ClinicalSession` is per-user mutable state (owned by the Pipeline).
  • `advance(session, out)` consumes ONE VALIDATED answer (turn.validate
    guarantees legality for the current expectation) and returns the next
    Step: what to say + what to expect next.
  • `absorb(session, out)` folds a `continuation` into the most recent open
    capture WITHOUT moving the machine (the two-breath answer fix).
  • `enter_crisis(session)` — the deterministic crisis net (crisis.py) or the
    NLU may call this at ANY point; the protocol then pauses permanently for
    the session and every later turn is an LLM crisis-protocol turn. There is
    deliberately no automatic resume (a human-review decision, not a
    pattern's).
  • question / tangent / unclear turns never reach this module: the pipeline
    speaks the bounded reply and the machine holds (`repeat_step` re-emits
    the pause if a re-ask is needed).

Pure Python over flow.py / instruments.py / templates.py — no LLM, no IO —
so every branch is testable against the case cards.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from . import templates
from .flow import (ARM_INSTRUMENT, ARM_ORDER, Ask, End, Gate, Label, PROTOCOL,
                   RunItems, Route, Tell, close_unit, label_index)
from .instruments import (assess, Assessment, BY_KEY, InvalidResponse,
                          next_item_index, option_score, PRE_SCREEN)
from .turn import TurnOut

logger = logging.getLogger(__name__)


# --------------- What a turn can speak ---------------

@dataclass(frozen=True)
class Say:
    """A FIXED utterance (verbatim script) — cacheable as a pre-rendered clip."""
    key: str    # stable content key, e.g. "audit.item.3" (cache identity)
    text: str


@dataclass(frozen=True)
class LLMSay:
    """One LLM-generated utterance, bounded to the current node by
    `instruction`. The LLM may phrase; it may not decide where the protocol
    goes next."""
    instruction: str


@dataclass(frozen=True)
class Speak:
    """Already-resolved dynamic text (e.g. the NLU turn's acknowledgment).
    Rendered per-turn, never cached (it depends on what the user just said)."""
    text: str


@dataclass(frozen=True)
class Expect:
    """What the next user input means (drives the NLU turn call)."""
    kind: str                        # consent | option | number | open | end
    instrument: str | None = None    # for kind="option": instrument key or "prescreen"
    item_index: int | None = None    # for kind="option"
    ask_key: str | None = None       # the Gate/Ask this pause belongs to
    slots: tuple[str, ...] = ()      # declared slots of a slot-ask
    missing: tuple[str, ...] = ()    # slots still unfilled (ask ONE at a time)


@dataclass(frozen=True)
class Step:
    node: str
    utterances: tuple
    expect: Expect


class ProtocolError(RuntimeError):
    """The pipeline fed an event that doesn't match the machine's expectation —
    always a wiring bug, never user error (user ambiguity must not advance)."""


# --------------- Per-session clinical state ---------------

@dataclass
class ClinicalSession:
    pc: int = 0                                # index of the paused step
    node: str = "consent"
    expect: Expect = field(default_factory=lambda: Expect("consent"))
    consent: str | None = None                 # "yes" | "no"
    prescreen: dict[str, int] = field(default_factory=dict)   # key -> code
    arms: list[str] = field(default_factory=list)             # pending arms
    arm: str | None = None                                    # active arm
    responses: dict[str, dict[int, int]] = field(default_factory=dict)
    assessments: dict[str, Assessment] = field(default_factory=dict)
    readiness: dict[str, int] = field(default_factory=dict)   # arm -> 0..10
    declined: list[str] = field(default_factory=list)         # declined permission keys (audit)
    corrections: list[dict] = field(default_factory=list)     # T21 audit: old→new codes
    covered: set[str] = field(default_factory=set)            # delivered unit ids (T9)
    answers: dict[str, str] = field(default_factory=dict)     # open captures (in-memory only)
    slots: dict[str, dict[str, str]] = field(default_factory=dict)  # slot captures
    last_ask_key: str | None = None            # target for continuation absorption
    # T20: an LLM-coded answer to a confirm item, held here (uncommitted)
    # until the person confirms the read-back: {instrument, item_index, code}.
    pending_confirm: dict | None = None
    crisis: bool = False
    aborted: bool = False                      # user stopped the session (T22)
    last_step: Step | None = None

    def instrument(self):
        return BY_KEY[ARM_INSTRUMENT[self.arm]]

    def to_audit_dict(self) -> dict:
        """Structured, non-free-text summary for the audit record (P7b):
        codes/scores/zones only — open captures appear as KEYS (what was
        answered), never as text (no transcripts)."""
        return {
            "node": self.node,
            "consent": self.consent,
            "prescreen": dict(self.prescreen),
            "responses": {k: dict(v) for k, v in self.responses.items()},
            "assessments": {
                k: {"score": a.score, "zone": a.zone, "complete": a.complete}
                for k, a in self.assessments.items()
            },
            "readiness": dict(self.readiness),
            "declined": list(self.declined),
            "corrections": [dict(c) for c in self.corrections],
            "covered": sorted(self.covered),
            "answered": sorted(self.answers),
            "slots_filled": {k: sorted(v) for k, v in self.slots.items()},
            "crisis": self.crisis,
            "aborted": self.aborted,
        }


# --------------- @resolvers: state-dependent units/asks ---------------
# The closed vocabulary here is locked by tests/test_flow_contract.py — a new
# @name in the program without a resolver arm below fails that test first.

def _resolve_say(session: ClinicalSession, name: str) -> Say:
    arm = session.arm
    if name == "@alcohol.screen.permission":
        # T9: only claim "the standard drink definition we just discussed"
        # if the education unit was actually delivered this session.
        key = ("alcohol.screen.permission"
               if "alcohol.edu.standard_drink" in session.covered
               else "alcohol.screen.permission.no_defn")
        return Say(key, templates.FIXED[key])
    if name == "@feedback":
        instrument_key = ARM_INSTRUMENT[arm]
        zone = session.assessments[instrument_key].zone
        return Say(f"feedback.{instrument_key}.{zone}",
                   templates.feedback_text(instrument_key, zone))
    if name == "@bi.permission":
        return Say(f"bi.permission.{arm}", templates.bi_permission(arm))
    if name == "@bi.likes":
        return Say(f"bi.likes.{arm}", templates.bi_likes(arm))
    if name == "@bi.dislikes":
        return Say(f"bi.dislikes.{arm}", templates.bi_dislikes(arm))
    if name == "@bi.recommend":
        return Say(f"bi.recommend.{arm}", templates.bi_recommend(arm))
    if name == "@bi.ruler":
        return Say(f"bi.ruler.{arm}", templates.bi_ruler(arm))
    if name == "@bi.why_not_lower":
        v = session.readiness[arm]
        return Say(f"bi.why_not_lower.{v}", templates.bi_why_not_lower(v))
    if name == "@bi.why_not_higher":
        v = session.readiness[arm]
        return Say(f"bi.why_not_higher.{v}", templates.bi_why_not_higher(v))
    if name == "@close":
        key = close_unit(session)
        return Say(key, templates.FIXED[key])
    raise ProtocolError(f"unknown resolver {name!r}")


def _points_instruction(session: ClinicalSession, unit: templates.Unit) -> str:
    """Instruction for a non-verbatim unit: the reviewable points plus the
    person's own captured words as grounding (their words, not paraphrase
    fodder from the model's imagination)."""
    ground = []
    a = session.answers
    if unit.id == "bi.summary.balance":
        ground = [f"They LIKE: {a.get('bi.likes', '(not captured)')}",
                  f"They DISLIKE: {a.get('bi.dislikes', '(not captured)')}"]
    elif unit.id == "bi.summary.rulers":
        ground = [f"Readiness {session.readiness.get(session.arm, '?')}/10.",
                  f"Why not lower: {a.get('bi.why_not_lower', '(not captured)')}",
                  f"Why not higher: {a.get('bi.why_not_higher', '(not captured)')}"]
    elif unit.id == "bi.reflect":
        ground = [f"They said: {a.get('bi.leaves_you', '(not captured)')}"]
    return " ".join(unit.points) + (("\n" + "\n".join(ground)) if ground else "")


def _tell_beats(session: ClinicalSession, unit: str) -> list:
    if unit.startswith("@"):
        say = _resolve_say(session, unit)
        session.covered.add(say.key)
        return [say]
    if unit in templates.POINTS_UNITS:
        session.covered.add(unit)
        return [LLMSay(_points_instruction(session,
                                           templates.POINTS_UNITS[unit]))]
    session.covered.add(unit)
    return [Say(unit, templates.FIXED[unit])]


def _ask_beats(session: ClinicalSession, step: Ask,
               missing: tuple[str, ...]) -> list:
    """The utterance that poses an Ask (nothing for ask='included' — a
    preceding Tell already spoke the question)."""
    if step.ask == "included":
        return []
    if step.ask == "fixed":
        if step.key.startswith("@"):
            return [_resolve_say(session, step.key)]
        return [Say(step.key, templates.FIXED[step.key])]
    # ask == "compose": the LLM phrases it. Slot-asks ask ONE missing slot
    # at a time (never a stacked question); plain composes fill {slot}s
    # from captured state.
    if step.slots:
        point = dict(step.slot_points)[missing[0]]
        return [LLMSay("Ask the person, in one short natural question, "
                       f"{point}. Ask nothing else.")]
    instruction = " ".join(step.points).format(
        drugs_kind=session.answers.get("drugs.kind", "the drugs they use"))
    return [LLMSay(instruction)]


# --------------- Pause bookkeeping ---------------

def _pause(session: ClinicalSession, node: str, beats: list,
           expect: Expect) -> Step:
    session.node = node
    session.expect = expect
    step = Step(node, tuple(beats), expect)
    session.last_step = step
    logger.info("[clinical] -> %s (expect %s)", node, expect.kind)
    return step


def repeat_step(session: ClinicalSession) -> Step:
    """Re-emit the current pause (after a hold turn the pipeline may need to
    re-pose the ask; the machine does not move)."""
    if session.last_step is None:
        return start(session)
    return session.last_step


# --------------- The interpreter ---------------

def _ask_key(step: Ask) -> str:
    """Canonical capture/node key of an Ask: the '@' marks how the ask TEXT
    resolves (per-arm Say), never how the capture is keyed."""
    return step.key.lstrip("@")


def _ask_missing(session: ClinicalSession, step: Ask) -> tuple[str, ...]:
    filled = session.slots.get(_ask_key(step), {})
    return tuple(s for s in step.slots if s not in filled)


def _run(session: ClinicalSession, beats: list) -> Step:
    """Execute steps from session.pc, collecting utterances, until a step
    needs user input (pause) or the program ends."""
    while True:
        step = PROTOCOL[session.pc]

        if isinstance(step, Label):
            session.pc += 1

        elif isinstance(step, Route):
            target = step.fn(session)
            session.pc = label_index(target)

        elif isinstance(step, Tell):
            beats.extend(_tell_beats(session, step.unit))
            session.pc += 1

        elif isinstance(step, Gate):
            key = step.key
            if key.startswith("@"):
                resolved = _resolve_say(session, key)
                if not step.ask_included:
                    beats.append(resolved)
                key = resolved.key
            elif not step.ask_included:
                beats.append(Say(key, templates.FIXED[key]))
            return _pause(session, key, beats,
                          Expect("consent", ask_key=key))

        elif isinstance(step, Ask):
            missing = _ask_missing(session, step) if step.slots else ()
            beats.extend(_ask_beats(session, step, missing))
            kind = "number" if step.kind == "number" else "open"
            key = _ask_key(step)
            return _pause(session, key, beats,
                          Expect(kind, ask_key=key,
                                 slots=step.slots, missing=missing))

        elif isinstance(step, RunItems):
            itemset = step.itemset
            if itemset == "prescreen":
                idx = next((i for i in range(len(PRE_SCREEN))
                            if PRE_SCREEN[i].key not in session.prescreen),
                           None)
                if idx is None:
                    session.pc += 1
                    continue
                q = PRE_SCREEN[idx]
                beats.append(Say(f"prescreen.{q.key}", q.item.text))
                return _pause(session, f"prescreen.{q.key}", beats,
                              Expect("option", instrument="prescreen",
                                     item_index=idx))
            instrument = BY_KEY[itemset]
            responses = session.responses.setdefault(itemset, {})
            idx = next_item_index(instrument, responses)
            if idx is None:
                assessment = assess(instrument, responses)
                session.assessments[itemset] = assessment
                logger.info("[clinical] %s complete: score=%d zone=%s",
                            itemset, assessment.score, assessment.zone)
                session.pc += 1
                continue
            preamble_key = f"{itemset}.preamble"
            if (instrument.preamble and not responses
                    and preamble_key not in session.covered):
                session.covered.add(preamble_key)
                beats.append(Say(preamble_key, instrument.preamble))
            item = instrument.items[idx]
            beats.append(Say(f"{itemset}.item.{idx}", item.text))
            return _pause(session, f"screening.{itemset}.{idx}", beats,
                          Expect("option", instrument=itemset,
                                 item_index=idx))

        elif isinstance(step, End):
            if step.close:
                beats.extend(_tell_beats(session, step.close))
            return _pause(session, step.node, beats, Expect("end"))

        else:  # pragma: no cover — program integrity tests prevent this
            raise ProtocolError(f"unknown step type at pc={session.pc}")


# --------------- Entry / crisis ---------------

def start(session: ClinicalSession) -> Step:
    """The fixed greeting (config.GREETING_TEXT, delivered by the pipeline)
    already asked for consent; the machine starts by expecting that answer."""
    session.pc = 0
    return _pause(session, "consent", [],
                  Expect("consent", ask_key="consent.opening"))


def enter_crisis(session: ClinicalSession) -> Step:
    """Deterministic crisis: pause the protocol permanently for this session.
    The pipeline speaks the fixed crisis response (crisis.py); every later
    turn is an LLM crisis-protocol turn."""
    session.crisis = True
    return _pause(session, "crisis", [], Expect("open"))


def enter_abort(session: ClinicalSession) -> Step:
    """User asked to stop the whole session (T22): close gracefully from ANY
    node with the fixed abort goodbye — no re-ask, no retention attempt.
    Everything coded so far stays in the session state (partial data is
    real data for the provider); the abort itself is recorded for audit."""
    session.aborted = True
    key = "close.aborted"
    session.covered.add(key)
    logger.info("[clinical] session aborted by user at node %s", session.node)
    return _pause(session, "aborted",
                  [Say(key, templates.FIXED[key])], Expect("end"))


_CRISIS_INSTRUCTION = (
    "Crisis protocol is active. In one or two sentences, respond with empathy "
    "and urgency to what the person just said, keep them talking, and repeat "
    "the crisis lines (call or text 988; call 911 if in immediate danger) "
    "when appropriate. Do not resume any screening."
)


def crisis_step(session: ClinicalSession) -> Step:
    """One LLM crisis-protocol turn — every turn while the session is in
    crisis (whether the deterministic net or the NLU flagged it)."""
    return _pause(session, "crisis", [LLMSay(_CRISIS_INSTRUCTION)],
                  Expect("open"))


# --------------- Consuming validated input ---------------

def _consume(session: ClinicalSession, out: TurnOut) -> None:
    """Apply ONE validated answer to the paused step and move the pointer.
    Raises ProtocolError when the payload doesn't fit the pause — that is
    always pipeline wiring gone wrong, never user ambiguity (turn.validate
    already downgraded anything ambiguous)."""
    step = PROTOCOL[session.pc]

    if isinstance(step, Gate):
        if out.code not in (0, 1):
            raise ProtocolError(
                f"gate {session.node!r} needs yes(1)/no(0), got {out.code!r}")
        key = session.expect.ask_key or step.key
        if key == "consent.opening":
            session.consent = "yes" if out.code == 1 else "no"
        if out.code == 1:
            session.pc += 1
        else:
            session.declined.append(key)
            session.pc = label_index(step.on_no)
        return

    if isinstance(step, Ask):
        key = _ask_key(step)
        if step.kind == "number":
            if not isinstance(out.code, int) or not 0 <= out.code <= 10:
                raise ProtocolError(f"ruler needs 0-10, got {out.code!r}")
            session.answers[key] = str(out.code)
            session.readiness[session.arm] = out.code
            session.last_ask_key = key
            session.pc += 1
            return
        # Open ask: EVERY capture lands in state (no discarded turns).
        if step.slots:
            if not out.slots:
                # turn.validate maps a bare text answer onto the asked slot,
                # so a slotless answer here means UNVALIDATED input — fail
                # loudly instead of holding this ask forever.
                raise ProtocolError(
                    f"slot ask {key!r} got no slots — unvalidated input?")
            store = session.slots.setdefault(key, {})
            store.update(out.slots)
            session.last_ask_key = key
            if _ask_missing(session, step):
                return                    # hold: ask the next missing slot
            session.answers[key] = "; ".join(
                f"{s}: {store[s]}" for s in step.slots)
            session.pc += 1
            return
        if not out.text:
            raise ProtocolError(f"open ask {key!r} got empty capture")
        session.answers[key] = out.text
        session.last_ask_key = key
        session.pc += 1
        return

    if isinstance(step, RunItems):
        exp = session.expect
        if exp.kind != "option" or exp.item_index is None:
            raise ProtocolError(f"item pause expected option, got {out!r}")
        code = out.code
        if not isinstance(code, int):
            raise ProtocolError(f"item {exp.item_index} needs an int code")
        if exp.instrument == "prescreen":
            q = PRE_SCREEN[exp.item_index]
            if not 0 <= code < len(q.item.options):
                raise ProtocolError(f"prescreen {q.key}: invalid code {code}")
            session.prescreen[q.key] = code
            return                        # RunItems loops to the next item
        instrument = BY_KEY[exp.instrument]
        # Validates the code against the instrument (raises InvalidResponse
        # on a coder bug rather than silently mis-scoring).
        option_score(instrument, exp.item_index, code)
        session.responses[exp.instrument][exp.item_index] = code
        return                            # RunItems loops / completes

    raise ProtocolError(
        f"no input expected at pc={session.pc} ({type(step).__name__})")


def advance(session: ClinicalSession, out: TurnOut) -> Step:
    """Consume ONE VALIDATED answer and move the protocol forward. The
    pipeline must have run turn.validate first; anything else is a wiring
    bug and raises ProtocolError."""
    if session.crisis:
        return crisis_step(session)
    if session.expect.kind == "end":
        raise ProtocolError("session already ended; nothing advances")
    if out.action != "answer":
        raise ProtocolError(
            f"advance() only takes validated answers, got {out.action!r}")
    _consume(session, out)
    return _run(session, [])


def needs_confirm(session: ClinicalSession, out: TurnOut) -> bool:
    """True when this validated answer must be read back before committing
    (T20): an instrument item marked confirm, coded by SEMANTIC mapping (the
    deterministic exact-wording pre-pass sets out.exact and skips this)."""
    exp = session.expect
    if (exp.kind != "option" or not exp.instrument
            or exp.instrument == "prescreen" or out.exact):
        return False
    item = BY_KEY[exp.instrument].items[exp.item_index]
    return item.confirm and isinstance(out.code, int)


def _confirm_pause(session: ClinicalSession) -> Step:
    """(Re-)emit the read-back pause for session.pending_confirm."""
    p = session.pending_confirm
    item = BY_KEY[p["instrument"]].items[p["item_index"]]
    label = item.options[p["code"]].label
    instruction = (
        f"You understood the person's answer to {item.text!r} as: {label!r}. "
        "In ONE short natural sentence, say that understanding back to them "
        "and ask if that's right (a yes/no). Do not re-ask the question "
        "itself and do not suggest a different answer.")
    return _pause(session, f"confirm.{p['instrument']}.{p['item_index']}",
                  [LLMSay(instruction)],
                  Expect("confirm", instrument=p["instrument"],
                         item_index=p["item_index"]))


def request_confirm(session: ClinicalSession, out: TurnOut) -> Step:
    """Hold a confirm item's coded answer UNCOMMITTED and ask the person to
    verify the read-back (T20). The read-back speaks the CODED option label,
    never a paraphrase of their raw words — the point is to surface a
    mis-coding while it can still be fixed."""
    exp = session.expect
    session.pending_confirm = {"instrument": exp.instrument,
                               "item_index": exp.item_index,
                               "code": out.code}
    return _confirm_pause(session)


def resolve_confirm(session: ClinicalSession, yes: bool) -> Step:
    """Consume the yes/no verdict on a read-back: yes commits the held code
    and the protocol moves on; no discards it and the SAME item is re-posed
    for a fresh answer. Either way the machine stays deterministic — the
    paused RunItems step recomputes what to ask next."""
    pending = session.pending_confirm
    session.pending_confirm = None
    if pending is None:
        raise ProtocolError("confirm resolution without a pending answer")
    if yes:
        session.responses.setdefault(
            pending["instrument"], {})[pending["item_index"]] = pending["code"]
        logger.info("[clinical] confirm accepted: %s item %d",
                    pending["instrument"], pending["item_index"])
    else:
        logger.info("[clinical] confirm DENIED: %s item %d re-collected",
                    pending["instrument"], pending["item_index"])
    return _run(session, [])


def correct(session: ClinicalSession, out: TurnOut) -> Step | None:
    """Apply a VALIDATED correction (T21): overwrite an already-answered item
    of the ACTIVE instrument with its new code, record old→new for audit, and
    re-emit the current pause. The downstream consequences are re-derived, not
    patched: next_item_index re-reads the declarative skip rules (a corrected
    Q2/Q3 can un-skip items 4-8 or newly skip them), and the score/zone are
    only ever computed at completion from the corrected codes. Returns None
    when the target isn't an answered item of the active instrument — the
    pipeline then holds and clarifies instead of moving anything."""
    exp = session.expect
    if (exp.kind != "option" or not exp.instrument
            or exp.instrument == "prescreen"):
        return None
    responses = session.responses.get(exp.instrument, {})
    if out.item not in responses or not isinstance(out.code, int):
        return None
    instrument = BY_KEY[exp.instrument]
    # Validates the new code (raises InvalidResponse on a coder bug rather
    # than silently mis-scoring) — turn.validate already shape-gated it.
    option_score(instrument, out.item, out.code)
    old = responses[out.item]
    if old == out.code:
        return repeat_step(session)          # no-op change: just re-pose
    responses[out.item] = out.code
    session.corrections.append({"instrument": exp.instrument,
                                "item": out.item, "old": old,
                                "new": out.code})
    logger.info("[clinical] correction: %s item %d code %d -> %d",
                exp.instrument, out.item, old, out.code)
    # Re-run from the paused RunItems step: the next unanswered, unskipped
    # item is recomputed under the corrected skip landscape.
    return _run(session, [])


def absorb(session: ClinicalSession, out: TurnOut) -> None:
    """Fold a `continuation` into the most recent open capture WITHOUT
    moving the machine — the two-breath answer lands where it belongs
    instead of being coded against the wrong expectation."""
    key = session.last_ask_key
    if key is None:
        return
    if out.slots:
        session.slots.setdefault(key, {}).update(out.slots)
    if out.text:
        prev = session.answers.get(key, "")
        session.answers[key] = (prev + " " + out.text).strip()
