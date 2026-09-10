
from __future__ import annotations

import logging
from dataclasses import dataclass, field

from . import coding, templates
from .flow import (ARM_INSTRUMENT, Ask, End, Gate, Label, PROTOCOL,
                   RunItems, Route, Tell, close_unit, label_index)
from .instruments import (assess, Assessment, BY_KEY, next_item_index,
                          option_score, PRE_SCREEN)
from .turn import Harvest, TurnOut, expect_target, target_item, validate_harvest

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Say:
    key: str
    text: str


@dataclass(frozen=True)
class LLMSay:
    instruction: str


@dataclass(frozen=True)
class Speak:
    text: str


@dataclass(frozen=True)
class Expect:

    kind: str
    instrument: str | None = None
    item_index: int | None = None
    ask_key: str | None = None
    slots: tuple[str, ...] = ()
    missing: tuple[str, ...] = ()


@dataclass(frozen=True)
class Step:

    node: str
    utterances: tuple
    expect: Expect


class ProtocolError(RuntimeError):
    pass


@dataclass
class ClinicalSession:

    pc: int = 0
    node: str = "consent"
    expect: Expect = field(default_factory=lambda: Expect("consent"))
    consent: str | None = None
    prescreen: dict[str, int] = field(default_factory=dict)
    arms: list[str] = field(default_factory=list)
    arm: str | None = None
    responses: dict[str, dict[int, int]] = field(default_factory=dict)
    assessments: dict[str, Assessment] = field(default_factory=dict)
    readiness: dict[str, int] = field(default_factory=dict)
    declined: list[str] = field(default_factory=list)
    corrections: list[dict] = field(default_factory=list)
    covered: set[str] = field(default_factory=set)
    answers: dict[str, str] = field(default_factory=dict)
    slots: dict[str, dict[str, str]] = field(default_factory=dict)
    last_ask_key: str | None = None
    pending_confirm: dict | None = None
    candidates: dict[str, dict] = field(default_factory=dict)
    refused_candidates: set = field(default_factory=set)
    missing: dict[str, dict] = field(default_factory=dict)
    stalls: int = 0
    asides: int = 0
    inconsistencies: list[dict] = field(default_factory=list)
    fired_rules: set[str] = field(default_factory=set)
    volunteered: set = field(default_factory=set)
    crisis: bool = False
    aborted: bool = False
    last_step: Step | None = None

    def instrument(self):
        return BY_KEY[ARM_INSTRUMENT[self.arm]]

    def to_audit_dict(self) -> dict:
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
            "volunteered": sorted(self.volunteered),
            "slots_filled": {k: sorted(v) for k, v in self.slots.items()},
            "crisis": self.crisis,
            "aborted": self.aborted,
        }


def _resolve_say(session: ClinicalSession, name: str) -> Say:
    arm = session.arm
    if name == "@alcohol.screen.permission":
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
    raise ProtocolError(f"unknown resolver {name!r}")


def _points_instruction(session: ClinicalSession, unit: templates.Unit) -> str:
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
    if unit == "@close":
        unit = close_unit(session)
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
    if step.ask == "included":
        return []
    if step.ask == "fixed":
        if step.key.startswith("@"):
            return [_resolve_say(session, step.key)]
        return [Say(step.key, templates.FIXED[step.key])]
    if step.slots:
        point = dict(step.slot_points)[missing[0]]
        return [LLMSay("Ask the person, in one short natural question, "
                       f"{point}. Ask nothing else.")]
    instruction = " ".join(step.points).format(
        drugs_kind=session.answers.get("drugs.kind", "the drugs they use"))
    return [LLMSay(instruction)]


def _resume_with(session: ClinicalSession, beats: list, target: str) -> Step:
    c = session.candidates[target]
    session.pending_confirm = {"target": target, "source": "volunteered", **c}
    logger.info("[clinical] reading back a volunteered answer for %s", target)
    beats.append(Speak(_confirm_text(session.pending_confirm)))
    return _pause(session, f"confirm.{target}", beats,
                  Expect("confirm", ask_key=target))


def _pause(session: ClinicalSession, node: str, beats: list,
           expect: Expect) -> Step:
    session.node = node
    session.expect = expect
    step = Step(node, tuple(beats), expect)
    session.last_step = step
    logger.info("[clinical] -> %s (expect %s)", node, expect.kind)
    return step


def repeat_step(session: ClinicalSession) -> Step:
    if session.last_step is None:
        return start(session)
    return session.last_step


def current_ask(session: ClinicalSession):
    step = session.last_step
    if step is None or not step.utterances:
        return None
    return step.utterances[-1]


def _ask_key(step: Ask) -> str:
    return step.key.lstrip("@")


def _ask_missing(session: ClinicalSession, step: Ask) -> tuple[str, ...]:
    filled = session.slots.get(_ask_key(step), {})
    return tuple(s for s in step.slots if s not in filled)


def _run(session: ClinicalSession, beats: list) -> Step:
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
            if _ask_key(step) in session.answers:
                session.pc += 1
                continue
            target = (_candidate_for(session, _ask_key(step))
                      if step.kind == "open" and not step.slots else None)
            if target:
                return _resume_with(session, beats, target)
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
                target = _candidate_for(session, f"prescreen.{q.key}")
                if target:
                    return _resume_with(session, beats, target)
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
            target = _candidate_for(session, f"{itemset}.{idx}")
            if target:
                return _resume_with(session, beats, target)
            item = instrument.items[idx]
            beats.append(Say(f"{itemset}.item.{idx}", item.text))
            return _pause(session, f"screening.{itemset}.{idx}", beats,
                          Expect("option", instrument=itemset,
                                 item_index=idx))

        elif isinstance(step, End):
            if step.close:
                beats.extend(_tell_beats(session, step.close))
            return _pause(session, step.node, beats, Expect("end"))

        else:
            raise ProtocolError(f"unknown step type at pc={session.pc}")


def start(session: ClinicalSession) -> Step:
    session.pc = 0
    return _run(session, [])


def enter_crisis(session: ClinicalSession) -> Step:
    session.crisis = True
    key = "close.crisis"
    session.covered.add(key)
    logger.warning("[clinical] session closed on crisis at node %s", session.node)
    return _pause(session, "crisis",
                  [Say(key, templates.FIXED[key])], Expect("end"))


def enter_abort(session: ClinicalSession) -> Step:
    session.aborted = True
    key = "close.aborted"
    session.covered.add(key)
    logger.info("[clinical] session aborted by user at node %s", session.node)
    return _pause(session, "aborted",
                  [Say(key, templates.FIXED[key])], Expect("end"))


def _consume(session: ClinicalSession, out: TurnOut) -> None:
    session.stalls = 0
    session.asides = 0
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
                raise ProtocolError(
                    f"slot ask {key!r} got no slots — unvalidated input?")
            store = session.slots.setdefault(key, {})
            store.update(out.slots)
            session.last_ask_key = key
            if _ask_missing(session, step):
                return
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
            return
        instrument = BY_KEY[exp.instrument]
        option_score(instrument, exp.item_index, code)
        session.responses[exp.instrument][exp.item_index] = code
        return

    raise ProtocolError(
        f"no input expected at pc={session.pc} ({type(step).__name__})")


def advance(session: ClinicalSession, out: TurnOut) -> Step:
    if session.expect.kind == "end":
        raise ProtocolError("session already ended; nothing advances")
    if out.action != "answer":
        raise ProtocolError(
            f"advance() only takes validated answers, got {out.action!r}")
    _consume(session, out)
    return _run(session, [])


def _conflict(session: ClinicalSession, out: TurnOut) -> tuple[str, str] | None:
    exp = session.expect
    if exp.instrument != "audit" or not isinstance(out.code, int):
        return None
    items = BY_KEY["audit"].items
    responses = session.responses.get("audit", {})

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


def write_target(session: ClinicalSession, target: str,
                 code: int | None, text: str | None) -> None:
    head, _, tail = target.rpartition(".")
    if head == "prescreen":
        session.prescreen[tail] = code
    elif head in BY_KEY and tail.isdigit():
        session.responses.setdefault(head, {})[int(tail)] = code
    else:
        session.answers[target] = text


def _confirm_text(p: dict) -> str:
    item = target_item(p["target"])
    said = f"Earlier you mentioned {p['quote']}" if p.get("quote") else ""
    if item is None:
        value = p.get("text") or ""
        if said:
            return f"{said} — should I put that down as your answer?"
        return f"So that's {value} — did I get that right?"
    label = item.options[p["code"]].label
    if said:
        return f"{said} — so that would be {label}. Is that right?"
    if p.get("note"):
        return f"{p['note']} — so that would be {label}. Did I get that right?"
    if p.get("prior"):
        return (f"Earlier I heard that {p['prior']}, so I want to make sure "
                f"I have this right: {label} — is that right?")
    return f"So that's {label} — did I get that right?"


def _confirm_pause(session: ClinicalSession) -> Step:
    p = session.pending_confirm
    return _pause(session, f"confirm.{p['target']}",
                  [Speak(_confirm_text(p))],
                  Expect("confirm", ask_key=p["target"]))


def request_confirm(session: ClinicalSession, out: TurnOut,
                    reason: dict | None = None) -> Step:
    session.pending_confirm = {"target": expect_target(session.expect),
                               "code": out.code, "text": out.text,
                               "source": "asked", **(reason or {})}
    return _confirm_pause(session)


def resolve_confirm(session: ClinicalSession, yes: bool) -> Step:
    p = session.pending_confirm
    session.pending_confirm = None
    if p is None:
        raise ProtocolError("confirm resolution without a pending answer")
    session.stalls = 0
    target = p["target"]
    session.candidates.pop(target, None)
    if yes:
        write_target(session, target, p.get("code"), p.get("text"))
        if p["source"] == "volunteered":
            session.volunteered.add(target)
        logger.info("[clinical] confirm accepted: %s (%s)", target, p["source"])
    else:
        if p["source"] == "volunteered":
            session.refused_candidates.add(target)
        logger.info("[clinical] confirm DENIED: %s re-collected", target)
    return _run(session, [])


OPEN_TARGETS = frozenset(
    step.key.lstrip("@") for step in PROTOCOL
    if isinstance(step, Ask) and step.kind == "open")


def record_harvest(session: ClinicalSession, out: TurnOut) -> list[str]:
    kept = []
    for h in out.harvest:
        target = h.target
        if target in session.refused_candidates or _target_answered(session, target):
            continue
        if target_item(target) is None and target not in OPEN_TARGETS:
            continue
        session.candidates[target] = {"code": h.code, "text": h.text,
                                      "quote": h.quote}
        kept.append(target)
    if kept:
        logger.info("[clinical] volunteered answers noted for %s", kept)
    return kept


def _target_answered(session: ClinicalSession, target: str) -> bool:
    head, _, tail = target.rpartition(".")
    if head == "prescreen":
        return tail in session.prescreen
    if head in BY_KEY and tail.isdigit():
        return int(tail) in session.responses.get(head, {})
    return target in session.answers


def _candidate_for(session: ClinicalSession, target: str) -> str | None:
    if target in session.candidates and target not in session.refused_candidates:
        return target
    return None


def offer_pause(session: ClinicalSession) -> Step:
    session.asides = 0
    logger.info("[clinical] offering to pause at node %s", session.node)
    return _pause(session, "pause.offer",
                  [Say("aside.offer_pause", templates.FIXED["aside.offer_pause"])],
                  Expect("consent", ask_key="pause.offer"))


def resolve_pause(session: ClinicalSession, keep_going: bool) -> Step:
    if not keep_going:
        return enter_abort(session)
    session.stalls = 0
    return _run(session, [])


def correct(session: ClinicalSession, out: TurnOut) -> Step | None:
    exp = session.expect
    if (exp.kind != "option" or not exp.instrument
            or exp.instrument == "prescreen"):
        return None
    responses = session.responses.get(exp.instrument, {})
    if out.item not in responses or not isinstance(out.code, int):
        return None
    instrument = BY_KEY[exp.instrument]
    option_score(instrument, out.item, out.code)
    old = responses[out.item]
    if old == out.code:
        return repeat_step(session)
    session.stalls = 0
    responses[out.item] = out.code
    session.corrections.append({"instrument": exp.instrument,
                                "item": out.item, "old": old,
                                "new": out.code})
    logger.info("[clinical] correction: %s item %d code %d -> %d",
                exp.instrument, out.item, old, out.code)
    return _run(session, [])


DONT_KNOW_LIMIT = 2
UNCLEAR_LIMIT = 3
ASIDE_LIMIT = 2


def note_aside(session: ClinicalSession) -> int:
    session.asides += 1
    return session.asides


def note_stall(session: ClinicalSession) -> int:
    session.stalls += 1
    return session.stalls


def mark_missing(session: ClinicalSession, reason: str = "no_answer") -> Step:
    session.stalls = 0
    exp = session.expect
    step = PROTOCOL[session.pc]
    skip_line = Say("item.skipped", templates.FIXED["item.skipped"])

    if exp.kind == "confirm":
        p = session.pending_confirm
        session.pending_confirm = None
        if p is not None:
            session.candidates.pop(p["target"], None)
            head, _, tail = p["target"].rpartition(".")
            if head in BY_KEY and tail.isdigit():
                session.missing.setdefault(head, {})[int(tail)] = "unconfirmed"
            logger.info("[clinical] confirm unresolvable: %s marked missing",
                        p["target"])
        return _run(session, [skip_line])

    if exp.kind == "option" and exp.instrument:
        if exp.instrument == "prescreen":
            q = PRE_SCREEN[exp.item_index]
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
        logger.info("[clinical] ask unanswerable (%s): %s", reason, key)
        session.pc += 1
        return _run(session, [skip_line])

    return repeat_step(session)


def absorb(session: ClinicalSession, out: TurnOut) -> None:
    key = session.last_ask_key
    if key is None:
        return
    if out.slots:
        session.slots.setdefault(key, {}).update(out.slots)
    if out.text:
        prev = session.answers.get(key, "")
        session.answers[key] = (prev + " " + out.text).strip()
