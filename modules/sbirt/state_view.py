"""Shows the model the whole interview, not just the question in front of it.

A model that can only see the current item cannot answer "how many are left",
"what is this for" or "didn't I already tell you that" — and a model that cannot
answer those will invent an answer. This renders the entire session as one
compact block, included in every turn, so those questions are answered from
state.

The contract this must keep:

  * Pure. The same session renders the same string, with no I/O and no
    mutation.
  * Re-rendered in full every turn. Nothing here is incremental or cached, so
    the session remains the only source of truth and this cannot go stale.
  * No patient content. Answered items appear as their code and official option
    label; open captures appear only as having been captured. Nothing the
    person said in their own words is ever rendered.
  * Bounded by MAX_CHARS, since this is prepended to every single turn.
"""

from __future__ import annotations

from .flow import ARM_INSTRUMENT, ARM_ORDER
from .instruments import BY_KEY, PRE_SCREEN, _skipped_items
from .runtime import ClinicalSession, Say, Speak

# Roughly a few hundred tokens, paid on every turn. The tests assert that every
# reachable protocol state renders within it.
MAX_CHARS = 3000

_TITLE = {"audit": "AUDIT", "dast_10": "DAST-10"}

_ROLE = (
    "[YOUR ROLE THIS TURN]\n"
    "- Do: respond to what the person just said (answer their question from "
    "this state, acknowledge feelings, handle tangents), then steer the "
    "conversation back to the CURRENT GOAL above.\n"
    "- Do NOT: pick the next question, compute or reveal scores/zones, "
    "reword a verbatim question stem, or promise anything this map does not "
    "show. The protocol engine decides all routing and scoring."
)


def _prescreen_done(session: ClinicalSession) -> bool:
    """Whether every pre-screen question has been answered."""
    return all(q.key in session.prescreen for q in PRE_SCREEN)


def _prescreen_line(session: ClinicalSession) -> str:
    """The pre-screen results, including which question is being asked now."""
    exp = session.expect
    missing = session.missing.get("prescreen", {})
    parts = []
    for i, q in enumerate(PRE_SCREEN):
        if q.key in session.prescreen:
            word = "POSITIVE" if session.prescreen[q.key] > 0 else "negative"
            if i in missing:
                word += " (unanswered, routed negative)"
        elif exp.instrument == "prescreen" and exp.item_index == i:
            word = "BEING ASKED NOW"
        else:
            word = "to ask"
        parts.append(f"{q.key}={word}")
    return "Pre-screen: " + ", ".join(parts)


def _arm_lines(session: ClinicalSession) -> list[str]:
    """Each arm and where it stands: skipped, active, queued, scored or declined."""
    lines = []
    for arm in ARM_ORDER:
        ins_key = ARM_INSTRUMENT[arm]
        name = _TITLE[ins_key]
        if session.prescreen.get(arm, 0) == 0:
            status = "skipped (pre-screen negative)"
        elif arm == session.arm:
            status = "ACTIVE"
        elif arm in session.arms:
            status = "up next"
        elif ins_key in session.assessments:
            a = session.assessments[ins_key]
            status = (f"done (score {a.score}, zone {a.zone or 'unbanded'}"
                      f"{'' if a.complete else ', partial'})")
        else:
            status = "ended (permission declined)"
        lines.append(f"  {arm} → {name}: {status}")
    # Tobacco is screened for but has no arm, so say so explicitly — otherwise
    # a positive answer looks like something the interview forgot about.
    if session.prescreen.get("tobacco", 0) > 0:
        lines.append("  tobacco: POSITIVE flag noted for the provider "
                     "(this protocol has no tobacco question arm)")
    return lines


def _item_lines(session: ClinicalSession, ins_key: str) -> list[str]:
    """One line per item of the active instrument.

    Runs of items sharing a status are collapsed into a range, which is what
    keeps a ten-item instrument inside the character budget.
    """
    ins = BY_KEY[ins_key]
    responses = session.responses.get(ins_key, {})
    missing = session.missing.get(ins_key, {})
    skipped = _skipped_items(ins, responses)
    exp = session.expect
    current = (exp.item_index
               if exp.instrument == ins_key and exp.item_index is not None
               else None)

    lines: list[str] = []
    run: list[int] = []
    run_status = ""

    def flush() -> None:
        if not run:
            return
        rng = (f"Q{run[0] + 1}" if len(run) == 1
               else f"Q{run[0] + 1}–Q{run[-1] + 1}")
        lines.append(f"  {rng}: {run_status}")
        run.clear()

    for i in range(len(ins.items)):
        if i == current:
            flush()
            tag = (" (read-back awaiting their yes/no)"
                   if exp.kind == "confirm" else "")
            lines.append(f"  Q{i + 1} ← CURRENT{tag}")
        elif i in responses:
            flush()
            label = ins.items[i].options[responses[i]].label
            lines.append(f"  Q{i + 1}: answered {responses[i]} ({label})")
        elif i in missing:
            flush()
            lines.append(f"  Q{i + 1}: unanswered ({missing[i]}) — scored 0, "
                         "flagged for the provider")
        else:
            status = "skipped by rule" if i in skipped else "to ask"
            if run and run_status != status:
                flush()
            run_status = status
            run.append(i)
    flush()

    if ins_key in session.assessments:
        a = session.assessments[ins_key]
        lines.append(f"  {_TITLE[ins_key]} complete: score {a.score} → zone "
                     f"{a.zone or 'unbanded'}"
                     f"{'' if a.complete else ' (lower bound, items missing)'}")
    for rule in ins.skip_rules:
        lines.append(f"  Skip rule: {rule.note}")
    return lines


def _next_line(session: ClinicalSession) -> str:
    """What remains after the current point, so "how much is left" is answerable."""
    rest = ", ".join(f"{a} → {_TITLE[ARM_INSTRUMENT[a]]}"
                     for a in session.arms)
    tail = f" → then {rest}" if rest else ""
    if session.node.startswith("bi.") or session.node.startswith("confirm.bi"):
        return (f"Then: finish the brief intervention{tail} → closing → "
                "the provider follows up.")
    if not _prescreen_done(session):
        return ("Then: a full instrument for each positive pre-screen area "
                "(alcohol → AUDIT, drugs → DAST-10), zone feedback, brief "
                "intervention where indicated → closing → provider follow-up.")
    return (f"Then: zone feedback (permission-gated) → brief intervention if "
            f"the zone is not healthy{tail} → closing → the provider "
            "follows up.")


def _ask_text(session: ClinicalSession) -> tuple[str, str]:
    """The pending question, and a note that it has already been spoken.

    The second half matters: these lines play from cached clips, so a model that
    assumed it still had to ask would repeat the question the person just heard.
    """
    exp = session.expect
    if exp.kind == "option" and exp.instrument:
        item = (PRE_SCREEN[exp.item_index].item
                if exp.instrument == "prescreen"
                else BY_KEY[exp.instrument].items[exp.item_index])
        return item.text, ("this stem was already spoken from a cached clip "
                           "— do not read it out again unless asked")
    step = session.last_step
    if step is not None:
        for u in reversed(step.utterances):
            if isinstance(u, Say):
                return u.text, ("this line was already spoken from a cached "
                                "clip — do not read it out again unless asked")
            if isinstance(u, Speak):
                return u.text, "this read-back was already spoken this turn"
    if exp.kind == "consent" and exp.ask_key == "consent.opening":
        return ("(the opening greeting already asked for consent to a few "
                "health questions)"), "already spoken"
    return ("(the ask was composed by the LLM in its own words this turn)",
            "already spoken")


def _answer_shape(session: ClinicalSession) -> str:
    """What a valid answer to the pending question looks like."""
    exp = session.expect
    kind = exp.kind
    if kind == "consent":
        return "A yes or no."
    if kind == "confirm":
        return ("A yes or no to the read-back (yes commits the coded answer; "
                "no re-collects the item).")
    if kind == "option":
        item = (PRE_SCREEN[exp.item_index].item
                if exp.instrument == "prescreen"
                else BY_KEY[exp.instrument].items[exp.item_index])
        opts = " / ".join(f"{i}={o.label}"
                          for i, o in enumerate(item.options))
        coding = getattr(item, "coding", "choice")
        if coding in ("freq_q1", "freq5"):
            return (f"A frequency — extract value+per, the engine computes "
                    f"the code. Options: {opts}")
        if coding == "quantity_drinks":
            return (f"A drink amount — extract value+unit[+beverage], the "
                    f"engine computes the code. Options: {opts}")
        return f"One option code: {opts}"
    if kind == "number":
        return "A single number from 0 to 10."
    if kind == "open" and exp.slots:
        return (f"Free text filling slots ({', '.join(exp.slots)}); still "
                f"missing: {', '.join(exp.missing) or 'none'}.")
    if kind == "open":
        return "Free text."
    return "The session has ended; no answer is expected."


def _goal_lines(session: ClinicalSession) -> list[str]:
    """What this turn is for: the pending ask, or why there is not one."""
    if session.crisis:
        return ["A crisis was flagged and the session is closed — the "
                "emergency numbers were given and their provider will follow "
                "up. No answer is expected."]
    if session.aborted:
        return ["The person ended the session — it is closed. No answer is "
                "expected; their partial answers stay recorded for the "
                "provider."]
    if session.expect.kind == "end":
        return ["The session is complete and closed. No answer is expected; "
                "the provider will follow up with them."]
    text, delivery = _ask_text(session)
    return [f"Pending ask (node {session.node}): \"{text}\"",
            f"Expected answer: {_answer_shape(session)}",
            f"Delivery note: {delivery}."]


def _phase_lines(session: ClinicalSession) -> list[str]:
    """Where the session is overall, and who owns the decisions that follow."""
    if session.crisis:
        stage = "CRISIS — closed after a crisis was flagged"
    elif session.aborted:
        stage = "ABORTED — closed early by the person"
    elif session.expect.kind == "end":
        stage = "Closing (session complete)"
    elif session.node.startswith("bi."):
        stage = f"Brief Intervention (BI) — {session.arm} arm"
    elif not _prescreen_done(session):
        stage = "Screening (S) — pre-screen"
    elif session.arm:
        stage = (f"Screening (S) — {session.arm} arm "
                 f"({_TITLE[ARM_INSTRUMENT[session.arm]]})")
    else:
        stage = "Screening (S)"
    return [f"SBIRT stage: {stage}",
            "All results go to the provider after the session; referral "
            "decisions are the provider's, not made in this conversation."]


def render_interview_state(session: ClinicalSession) -> str:
    """The whole interview as one block, for inclusion in this turn's context.

    A pure read of the session; see the module docstring for what this is
    required to guarantee.
    """
    out = ["=== INTERVIEW STATE (program-owned; re-rendered every turn) ==="]

    out.append("[MAP]")
    out.append(_prescreen_line(session))
    if _prescreen_done(session):
        out.append("Arms (protocol order):")
        out.extend(_arm_lines(session))
    if session.arm:
        ins_key = ARM_INSTRUMENT[session.arm]
        out.append(f"{_TITLE[ins_key]} items ({session.arm} arm):")
        out.extend(_item_lines(session, ins_key))
    out.append(_next_line(session))

    out.append("[CURRENT GOAL]")
    out.extend(_goal_lines(session))

    # Placeholder for answers volunteered before their question is reached.
    out.append("[HARVESTED CANDIDATES]")
    out.append("(none yet — early-answer harvesting lands in a later task)")

    out.append("[PHASE]")
    out.extend(_phase_lines(session))

    out.append(_ROLE)
    return "\n".join(out)
