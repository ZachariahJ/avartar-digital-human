"""Executes the protocol in flow.py. One interpreter, no per-question code.

Walks the program step by step, gathering what to say until it reaches
something that needs an answer, then stops. Exactly one validated answer moves
it on. The pointer, the scores, the zones, the skip rules and every branch are
computed here from data; a model's understanding of an utterance enters only as
a validated TurnOut, and can reach nothing else.

What the pipeline may call:

  advance(session, out)   consume one validated answer and return the next
                          step: what to say, and what to expect after it. The
                          answer must already have passed turn.validate.
  absorb(session, out)    fold a continuation into the most recent open
                          capture without moving the machine, for an answer
                          given in two breaths.
  correct(session, out)   overwrite an earlier answer and re-derive from it.
  enter_crisis(session)   close the session on a crisis the model flagged:
                          speak the emergency numbers and stop. There is
                          deliberately no counseling loop and no resume —
                          deciding somebody is safe is not a decision this
                          code, or a model, should make, so it hands off.
  enter_abort(session)    close early, keeping what was coded.
  repeat_step(session)    re-emit the current pause.

Questions, asides and unclear turns never reach this module at all — the
pipeline replies and the machine holds where it is.

Pure Python over the protocol data, with no model and no I/O, so every branch
can be tested directly against the study's case cards.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from . import coding, templates
from .flow import (ARM_INSTRUMENT, Ask, End, Gate, Label, PROTOCOL,
                   RunItems, Route, Tell, close_unit, label_index)
from .instruments import (assess, Assessment, BY_KEY, next_item_index,
                          option_score, PRE_SCREEN)
from .turn import TurnOut

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Say:
    """Verbatim script. Identical every session, so it can be a cached clip."""
    key: str    # stable content key, e.g. "audit.item.3" (cache identity)
    text: str


@dataclass(frozen=True)
class LLMSay:
    """An utterance the model words, from an instruction the protocol wrote.

    It chooses phrasing and nothing else; the instruction fixes the content and
    the protocol has already decided what comes after.
    """
    instruction: str


@dataclass(frozen=True)
class Speak:
    """Text already produced this turn, such as an acknowledgment.

    Never cached: it exists because of what the person just said.
    """
    text: str


@dataclass(frozen=True)
class Expect:
    """What kind of answer the machine is waiting for, and to what."""

    kind: str                        # consent | option | number | open | end
    instrument: str | None = None    # which instrument, or "prescreen"
    item_index: int | None = None    # which of its items
    ask_key: str | None = None       # the gate or ask this pause belongs to
    slots: tuple[str, ...] = ()      # all slots this ask declares
    missing: tuple[str, ...] = ()    # those still unfilled; one is asked at a time


@dataclass(frozen=True)
class Step:
    """One turn's output: where the protocol is, what to say, what to expect."""

    node: str
    utterances: tuple
    expect: Expect


class ProtocolError(RuntimeError):
    """An event reached the machine that does not fit what it was expecting.

    Always a wiring bug. An ambiguous or unexpected user reply cannot cause this
    — those are downgraded before they get here, and hold the protocol in place.
    """


@dataclass
class ClinicalSession:
    """All the clinical state for one conversation. Owned by the Pipeline."""

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
    corrections: list[dict] = field(default_factory=list)     # old and new codes
    covered: set[str] = field(default_factory=set)            # what has been said
    answers: dict[str, str] = field(default_factory=dict)     # open captures (in-memory only)
    slots: dict[str, dict[str, str]] = field(default_factory=dict)  # slot captures
    last_ask_key: str | None = None            # target for continuation absorption
    # A coded answer waiting to be confirmed by the person. Held rather than
    # written, so an answer that turns out to be wrong never enters the score.
    pending_confirm: dict | None = None
    # Items the person could not or would not answer, by itemset. They score
    # zero, which makes the total a lower bound, and they mark the assessment
    # incomplete — so the provider sees what is unanswered rather than a result
    # that merely looks whole.
    missing: dict[str, dict] = field(default_factory=dict)
    # Consecutive failed turns at the current question. Reset by any successful
    # answer; at the limit the question is abandoned rather than asked again.
    stalls: int = 0
    # Contradictions found between answers, recorded as codes only. The fired
    # set bounds it to one read-back per rule, so nobody is challenged twice
    # about the same inconsistency.
    inconsistencies: list[dict] = field(default_factory=list)
    fired_rules: set[str] = field(default_factory=set)
    crisis: bool = False
    aborted: bool = False                      # the person stopped the session
    last_step: Step | None = None

    def instrument(self):
        """The instrument for the arm currently running."""
        return BY_KEY[ARM_INSTRUMENT[self.arm]]

    def to_audit_dict(self) -> dict:
        """The session as a record for the provider, with no patient words in it.

        Codes, scores and zones only. Open answers appear as the keys that were
        answered, never as what was said, so the record can be kept without
        keeping a transcript.
        """
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


def _resolve_say(session: ClinicalSession, name: str) -> Say:
    """Turn an "@name" from the protocol into the fixed line it means here.

    The valid names are fixed and covered by the flow contract test, so a new
    "@name" in the protocol without an arm below fails a test rather than a
    conversation.
    """
    arm = session.arm
    if name == "@alcohol.screen.permission":
        # Only refer back to the standard-drink definition if the education was
        # actually delivered; the permission before it can decline.
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
    """Build the instruction for a unit the model has to word.

    The points come from the reviewable data; the grounding is what this person
    actually said, quoted from captured state. Without it the model has nothing
    to reflect back except its own invention.
    """
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
    """The utterances for one Tell, and record that its content was delivered."""
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
    """The utterance that poses an Ask, or nothing if it was already spoken."""
    if step.ask == "included":
        return []
    if step.ask == "fixed":
        if step.key.startswith("@"):
            return [_resolve_say(session, step.key)]
        return [Say(step.key, templates.FIXED[step.key])]
    # Composed by the model. A slot ask covers exactly one missing slot, so the
    # person is never handed several questions at once.
    if step.slots:
        point = dict(step.slot_points)[missing[0]]
        return [LLMSay("Ask the person, in one short natural question, "
                       f"{point}. Ask nothing else.")]
    instruction = " ".join(step.points).format(
        drugs_kind=session.answers.get("drugs.kind", "the drugs they use"))
    return [LLMSay(instruction)]


def _pause(session: ClinicalSession, node: str, beats: list,
           expect: Expect) -> Step:
    """Stop and wait for an answer, recording where and for what."""
    session.node = node
    session.expect = expect
    step = Step(node, tuple(beats), expect)
    session.last_step = step
    logger.info("[clinical] -> %s (expect %s)", node, expect.kind)
    return step


def repeat_step(session: ClinicalSession) -> Step:
    """Re-emit the current pause, so an ask can be re-posed without moving."""
    if session.last_step is None:
        return start(session)
    return session.last_step


def _ask_key(step: Ask) -> str:
    """The key an Ask's answer is stored under.

    The "@" prefix says how the question's wording is resolved, which has
    nothing to do with where the answer goes, so it is stripped here.
    """
    return step.key.lstrip("@")


def _ask_missing(session: ClinicalSession, step: Ask) -> tuple[str, ...]:
    """Which of an Ask's slots are still unfilled, in declared order."""
    filled = session.slots.get(_ask_key(step), {})
    return tuple(s for s in step.slots if s not in filled)


def _run(session: ClinicalSession, beats: list) -> Step:
    """Run the protocol from where it stands until it needs an answer or ends.

    Collects everything to say along the way, so one turn can span several
    steps.
    """
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


def start(session: ClinicalSession) -> Step:
    """Begin a session, waiting for the consent answer.

    The greeting already asked the question, so the machine's first act is to
    expect a reply rather than to speak.
    """
    session.pc = 0
    return _pause(session, "consent", [],
                  Expect("consent", ask_key="consent.opening"))


def enter_crisis(session: ClinicalSession) -> Step:
    """Close the session because the model flagged a crisis.

    Terminal, like enter_abort. The fixed line hands over the emergency numbers
    and says their provider will follow up; nothing further is spoken and the
    screening does not resume. What was coded so far stays, and the crisis
    itself is recorded.
    """
    session.crisis = True
    key = "close.crisis"
    session.covered.add(key)
    logger.warning("[clinical] session closed on crisis at node %s", session.node)
    return _pause(session, "crisis",
                  [Say(key, templates.FIXED[key])], Expect("end"))


def enter_abort(session: ClinicalSession) -> Step:
    """Close the session early because the person asked to stop.

    Works from any node, and makes no attempt to keep them. What was coded so
    far stays: partial data is still useful to their provider, and the stop
    itself is recorded.
    """
    session.aborted = True
    key = "close.aborted"
    session.covered.add(key)
    logger.info("[clinical] session aborted by user at node %s", session.node)
    return _pause(session, "aborted",
                  [Say(key, templates.FIXED[key])], Expect("end"))


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
        if step.slots:
            if not out.slots:
                # Validation maps a bare answer onto the slot being asked, so
                # arriving with none means this input never went through it.
                # Failing loudly beats holding this question forever.
                raise ProtocolError(
                    f"slot ask {key!r} got no slots — unvalidated input?")
            store = session.slots.setdefault(key, {})
            store.update(out.slots)
            session.last_ask_key = key
            if _ask_missing(session, step):
                return                    # more slots to fill; ask the next
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
            return                        # the step re-runs for the next item
        instrument = BY_KEY[exp.instrument]
        # Called for its validation, not its result: an out-of-range code
        # raises here rather than quietly mis-scoring the instrument.
        option_score(instrument, exp.item_index, code)
        session.responses[exp.instrument][exp.item_index] = code
        return

    raise ProtocolError(
        f"no input expected at pc={session.pc} ({type(step).__name__})")


def advance(session: ClinicalSession, out: TurnOut) -> Step:
    """Consume one validated answer and return the next step.

    The answer must already have passed turn.validate; anything else is a
    wiring bug and raises rather than advancing the protocol.
    """
    if session.expect.kind == "end":
        raise ProtocolError("session already ended; nothing advances")
    if out.action != "answer":
        raise ProtocolError(
            f"advance() only takes validated answers, got {out.action!r}")
    _consume(session, out)
    return _run(session, [])


def _conflict(session: ClinicalSession, out: TurnOut) -> tuple[str, str] | None:
    """Whether this answer contradicts one already given.

    Returns the rule that fired and the prior answer phrased for speaking, or
    None. Each rule triggers at most one read-back per session, so nobody is
    challenged repeatedly — but every detection is recorded either way, so the
    provider sees the contradiction even if the person stands by both answers.
    """
    exp = session.expect
    if exp.instrument != "audit" or not isinstance(out.code, int):
        return None
    items = BY_KEY["audit"].items
    responses = session.responses.get("audit", {})

    # More heavy-drinking occasions than drinking occasions, which cannot be.
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

    # A coded frequency well below what they described conversationally.
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
    """Whether this answer should be read back before it is committed, and why.

    Returns None to commit directly, or a payload describing the reason.

    Reading answers back is the exception rather than the rule, because a
    screening that confirms everything is tedious enough that people start
    agreeing to move it along. So it happens only where being wrong would move
    a score and nothing else vouches for the code: a unit conversion was
    assumed and the person deserves to hear it, the value sits near a bucket
    edge, the answer contradicts an earlier one, or the item has no
    deterministic derivation and the code came from semantic mapping alone.

    Exact-wording answers and cleanly computed codes commit without asking.
    Narrowing this from confirming every item is a clinical decision — pending
    clinician review.
    """
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
    return None


def _confirm_pause(session: ClinicalSession) -> Step:
    """Ask the person to verify the answer being held.

    The wording is built here rather than by the model, and always states the
    coded option label. Confirming a paraphrase of what they said would confirm
    the wrong thing: what needs checking is the code about to be committed.
    """
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
    """Hold this answer uncommitted and ask the person to verify it.

    The point is to surface a mis-coding while it can still be corrected, so
    nothing reaches the score until they agree.
    """
    exp = session.expect
    session.pending_confirm = {"instrument": exp.instrument,
                               "item_index": exp.item_index,
                               "code": out.code, **(reason or {})}
    return _confirm_pause(session)


def resolve_confirm(session: ClinicalSession, yes: bool) -> Step:
    """Act on their verdict: commit the held code, or discard it and re-ask.

    Either way the next question is recomputed rather than assumed, so a
    rejected answer simply leaves its item unanswered.
    """
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
    """Overwrite an earlier answer with a corrected one.

    Returns the re-emitted pause, or None when the target is not an answered
    item of the active instrument — the pipeline then asks which question they
    meant rather than changing anything.

    Consequences are re-derived rather than patched. Skip rules are re-read, so
    a corrected answer can newly skip later items or bring skipped ones back,
    and the score is only ever computed at completion from whatever the codes
    then are. Patching the total instead would leave it disagreeing with the
    answers it supposedly came from.
    """
    exp = session.expect
    if (exp.kind != "option" or not exp.instrument
            or exp.instrument == "prescreen"):
        return None
    responses = session.responses.get(exp.instrument, {})
    if out.item not in responses or not isinstance(out.code, int):
        return None
    instrument = BY_KEY[exp.instrument]
    # Called for its validation, not its result, as when the answer was first
    # recorded.
    option_score(instrument, out.item, out.code)
    old = responses[out.item]
    if old == out.code:
        return repeat_step(session)          # they corrected it to itself
    session.stalls = 0
    responses[out.item] = out.code
    session.corrections.append({"instrument": exp.instrument,
                                "item": out.item, "old": old,
                                "new": out.code})
    logger.info("[clinical] correction: %s item %d code %d -> %d",
                exp.instrument, out.item, old, out.code)
    return _run(session, [])


# Consecutive failed turns one question tolerates before it is abandoned rather
# than asked again. "I don't know" gets one recall aid before the exit; an
# unclear answer gets one further attempt at clarifying.
DONT_KNOW_LIMIT = 2
UNCLEAR_LIMIT = 3


def note_stall(session: ClinicalSession) -> int:
    """Record one failed turn at the current question and return the streak.

    Reset by any successful answer, so only consecutive failures count.
    """
    session.stalls += 1
    return session.stalls


def mark_missing(session: ClinicalSession, reason: str = "no_answer") -> Step:
    """Give up on the current question, record why, and move on.

    This is what makes every clarification loop finite: whatever the question,
    there is always a way out that is not another attempt at asking it.

    Nothing is ever guessed. An instrument item scores zero and is flagged, so
    the total is a lower bound the provider can see is incomplete. A permission
    gate takes its refusal path, which is the safe direction — nothing gets
    screened without a clear yes. An open or numeric question is simply
    recorded as unanswered.
    """
    session.stalls = 0
    exp = session.expect
    step = PROTOCOL[session.pc]
    skip_line = Say("item.skipped", templates.FIXED["item.skipped"])

    if exp.kind == "confirm":
        # The held code was never verified, and it was held precisely because
        # something about it was doubtful. Committing it now would commit the
        # answer nobody could confirm.
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
            # Routed as negative so the session can continue, but recorded as
            # unanswered — the arm was skipped for want of an answer, not
            # because there was nothing there.
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
        return _run(session, [])

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
