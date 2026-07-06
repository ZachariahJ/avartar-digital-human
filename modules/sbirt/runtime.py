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

from . import coding, templates
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
    # until the person confirms the read-back: {instrument, item_index, code,
    # reason, note, prior} (reason/note/prior per confirm_reason).
    pending_confirm: dict | None = None
    # F1/F2: items the person could not / would not answer, keyed by itemset
    # ("prescreen" | instrument key | "asks" | "gates") -> {item: reason}.
    # They score 0 (lower bound) and mark the assessment incomplete — the
    # provider record shows exactly what went unanswered, never a false
    # picture of complete data.
    missing: dict[str, dict] = field(default_factory=dict)
    # F2: consecutive failed turns (unclear / dont_know) at the CURRENT
    # pause. Reset by every successful consume; at the limit the pipeline
    # stops re-asking and degrades (mark_missing) instead of looping.
    stalls: int = 0
    # Phase 5 audit trail: detected cross-item contradictions (codes only,
    # never free text) + which rules already fired (one read-back per rule).
    inconsistencies: list[dict] = field(default_factory=list)
    fired_rules: set[str] = field(default_factory=set)
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
                k: {"score": a.score, "zone": a.zone, "complete": a.complete,
                    "missing_items": list(a.missing)}
                for k, a in self.assessments.items()
            },
            "readiness": dict(self.readiness),
            "declined": list(self.declined),
            "corrections": [dict(c) for c in self.corrections],
            "missing": {k: dict(v) for k, v in self.missing.items()},
            "inconsistencies": [dict(c) for c in self.inconsistencies],
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
            unanswerable = session.missing.get(itemset, {})
            idx = next_item_index(instrument, responses, unanswerable)
            if idx is None:
                assessment = assess(instrument, responses, unanswerable)
                session.assessments[itemset] = assessment
                logger.info(
                    "[clinical] %s complete: score=%d zone=%s%s",
                    itemset, assessment.score, assessment.zone,
                    (f" (LOWER BOUND — items {sorted(unanswerable)} "
                     "unanswered)") if unanswerable else "")
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
    session.stalls = 0                     # progress: the stall streak ends
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


def _conflict(session: ClinicalSession, out: TurnOut) -> tuple[str, str] | None:
    """Phase 5: does this incoming answer contradict something already
    established? Returns (rule_key, spoken-prior-statement) or None. Each
    rule fires ONE read-back per session; every detection is recorded in
    session.inconsistencies (codes only — the provider sees the contradiction
    even if the person confirms both answers)."""
    exp = session.expect
    if exp.instrument != "audit" or not isinstance(out.code, int):
        return None
    items = BY_KEY["audit"].items
    responses = session.responses.get("audit", {})

    # AUDIT Q3 (6+ drinks) more often than Q1 says they drink at all.
    if (exp.item_index == 2 and 0 in responses
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

    # AUDIT Q1 far below the drinking frequency captured in the Q/F slots.
    if exp.item_index == 0:
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


def confirm_reason(session: ClinicalSession, out: TurnOut) -> dict | None:
    """Round-2 confirmation discipline (F5): the read-back is the EXCEPTION.
    A validated answer to a confirm item is read back only when
      • a conversion default entered the coding (out.assumed — the person
        must hear "1 liter of whiskey is about 23 standard drinks"),
      • the normalized value sat near a bucket edge (out.boundary),
      • it contradicts an earlier answer (Phase 5 rules), or
      • the item has no deterministic derivation (coding == "choice") and
        the code came from semantic mapping — nothing else vouches for it.
    Exact-wording answers and cleanly derived codes commit directly.
    (Rolling back the always-confirm default of the original T20 is a
    clinical decision — PENDING CLINICIAN REVIEW.)
    Returns None (commit directly) or the read-back's reason payload."""
    exp = session.expect
    if (exp.kind != "option" or not exp.instrument
            or exp.instrument == "prescreen"
            or not isinstance(out.code, int)):
        return None
    conflict = _conflict(session, out)
    if conflict:
        return {"reason": "conflict", "rule": conflict[0],
                "prior": conflict[1]}
    if out.exact:
        return None
    item = BY_KEY[exp.instrument].items[exp.item_index]
    if not item.confirm:
        return None
    if out.assumed or out.boundary:
        return {"reason": "conversion" if out.assumed else "boundary",
                "note": out.note}
    if item.coding == "choice":
        return {"reason": "semantic"}
    return None                            # cleanly derived: commit directly


def needs_confirm(session: ClinicalSession, out: TurnOut) -> bool:
    """Compatibility wrapper over confirm_reason (T20 entry point)."""
    return confirm_reason(session, out) is not None


def _confirm_pause(session: ClinicalSession) -> Step:
    """(Re-)emit the read-back pause for session.pending_confirm.

    F6: the read-back is DETERMINISTIC text that always speaks the coded
    option label (plus the conversion note / conflicting prior when that is
    why we are asking) — never an LLM paraphrase of the person's raw words,
    so the value the person confirms IS the value about to be committed."""
    p = session.pending_confirm
    item = BY_KEY[p["instrument"]].items[p["item_index"]]
    label = item.options[p["code"]].label
    if p.get("note"):
        text = (f"{p['note']} — so that would be {label}. "
                "Did I get that right?")
    elif p.get("prior"):
        text = (f"Earlier I heard that {p['prior']}, so I want to make sure "
                f"I have this right: {label} — is that right?")
    else:
        text = f"So that's {label} — did I get that right?"
    return _pause(session, f"confirm.{p['instrument']}.{p['item_index']}",
                  [Speak(text)],
                  Expect("confirm", instrument=p["instrument"],
                         item_index=p["item_index"]))


def request_confirm(session: ClinicalSession, out: TurnOut,
                    reason: dict | None = None) -> Step:
    """Hold a confirm item's coded answer UNCOMMITTED and ask the person to
    verify the read-back (T20). The read-back speaks the CODED option label,
    never a paraphrase of their raw words — the point is to surface a
    mis-coding while it can still be fixed."""
    exp = session.expect
    session.pending_confirm = {"instrument": exp.instrument,
                               "item_index": exp.item_index,
                               "code": out.code, **(reason or {})}
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
    session.stalls = 0
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
    session.stalls = 0
    responses[out.item] = out.code
    session.corrections.append({"instrument": exp.instrument,
                                "item": out.item, "old": old,
                                "new": out.code})
    logger.info("[clinical] correction: %s item %d code %d -> %d",
                exp.instrument, out.item, old, out.code)
    # Re-run from the paused RunItems step: the next unanswered, unskipped
    # item is recomputed under the corrected skip landscape.
    return _run(session, [])


# --------------- Stalls and the missing-data exit (F1/F2) ---------------

# How many consecutive failed turns the SAME pause tolerates before the
# engine stops re-asking and degrades. dont_know escapes after one anchored
# probe (2nd occurrence); unclear gets one more clarification attempt.
DONT_KNOW_LIMIT = 2
UNCLEAR_LIMIT = 3


def note_stall(session: ClinicalSession) -> int:
    """Count one failed turn (unclear / dont_know) at the current pause and
    return the streak. Every successful consume resets it."""
    session.stalls += 1
    return session.stalls


def mark_missing(session: ClinicalSession, reason: str = "no_answer") -> Step:
    """F2's exit: the person cannot (or will not) answer the current pause
    after the engine's probes — record it as MISSING and move the protocol
    on. Never a guess: missing instrument items score 0 (the total is a
    lower bound, Assessment.missing carries the flag for the provider);
    an unanswerable permission gate degrades to its decline path (the safe
    direction — nothing is screened without a clear yes); an unanswerable
    open/number ask is recorded as not answered. This is the guaranteed
    terminus that makes every clarification loop finite (F7)."""
    session.stalls = 0
    exp = session.expect
    step = PROTOCOL[session.pc]
    skip_line = Say("item.skipped", templates.FIXED["item.skipped"])

    if exp.kind == "confirm":
        # A read-back that can't get a yes/no: drop the UNVERIFIED candidate
        # code and mark the item missing — an unconfirmable score-critical
        # answer must not be committed on the model's claim alone.
        p = session.pending_confirm
        session.pending_confirm = None
        if p is not None:
            session.missing.setdefault(
                p["instrument"], {})[p["item_index"]] = "unconfirmed"
            logger.info("[clinical] confirm unresolvable: %s item %d "
                        "marked missing", p["instrument"], p["item_index"])
        return _run(session, [skip_line])

    if exp.kind == "option" and exp.instrument:
        if exp.instrument == "prescreen":
            q = PRE_SCREEN[exp.item_index]
            # Route as negative (score 0) but record the truth: the arm was
            # not screened because the question went unanswered.
            session.prescreen[q.key] = 0
            session.missing.setdefault(
                "prescreen", {})[exp.item_index] = reason
        else:
            session.missing.setdefault(
                exp.instrument, {})[exp.item_index] = reason
        logger.info("[clinical] item missing (%s): %s", reason, session.node)
        return _run(session, [skip_line])

    if isinstance(step, Gate):
        key = exp.ask_key or step.key
        if key == "consent.opening":
            session.consent = "no"
        session.declined.append(key)
        session.missing.setdefault("gates", {})[key] = reason
        logger.info("[clinical] gate unanswerable (%s): %s -> decline path",
                    reason, key)
        session.pc = label_index(step.on_no)
        return _run(session, [])           # the decline path speaks for itself

    if isinstance(step, Ask):
        key = _ask_key(step)
        session.missing.setdefault("asks", {})[key] = reason
        if step.kind != "number":
            session.answers.setdefault(key, "(not answered)")
        # A missing readiness ruler leaves session.readiness unset; the
        # protocol's _after_ruler route skips the ruler follow-ups.
        logger.info("[clinical] ask unanswerable (%s): %s", reason, key)
        session.pc += 1
        return _run(session, [skip_line])

    return repeat_step(session)            # end/unknown: nothing to skip


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
