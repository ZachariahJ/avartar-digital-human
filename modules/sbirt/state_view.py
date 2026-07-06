"""Read-only rendering of the FULL interview state for the LLM.

Step 1 of the context refactor: every conversational failure traced back to
the LLM seeing only the current item — never the map. This module renders the
whole ClinicalSession as ONE compact text block injected into every turn's
context, so the model can answer "how many questions are left / what is this
for / what did I already say" from state instead of guessing.

Contract:
  • PURE: same session state -> same string; no IO, no session mutation.
  • Full re-render every turn — ClinicalSession stays the single source of
    truth; nothing here is incremental or cached per session.
  • No PHI: answered items render as option CODE + official option label
    only; open captures render as "(captured)" — never transcript text.
  • Compact: MAX_CHARS bounds the whole block (snapshot tests assert it).
"""

from __future__ import annotations

from .flow import ARM_INSTRUMENT, ARM_ORDER
from .instruments import BY_KEY, PRE_SCREEN, _skipped_items
from .runtime import ClinicalSession, Say, Speak

# Hard budget for the rendered block (~4 chars/token — a few hundred tokens).
# tests/test_state_view.py asserts every covered protocol state fits.
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
    return all(q.key in session.prescreen for q in PRE_SCREEN)


def _prescreen_line(session: ClinicalSession) -> str:
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
    if session.prescreen.get("tobacco", 0) > 0:
        lines.append("  tobacco: POSITIVE flag noted for the provider "
                     "(this protocol has no tobacco question arm)")
    return lines


def _item_lines(session: ClinicalSession, ins_key: str) -> list[str]:
    """One line per item of the active instrument, with consecutive
    same-status items ("to ask" / rule-skips) collapsed into ranges."""
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
    """The verbatim text of the pending ask + how it was delivered."""
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
    if session.crisis:
        return ["Crisis protocol is active — no protocol question is "
                "pending. Respond with empathy and the crisis lines "
                "(988; 911 if in immediate danger). No screening resumes."]
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
    if session.crisis:
        stage = "CRISIS PAUSE — the protocol is paused for the rest of the session"
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
    """The whole interview as one compact, PHI-free text block (see module
    docstring for the contract). Pure read of ClinicalSession."""
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

    out.append("[HARVESTED CANDIDATES]")
    out.append("(none yet — early-answer harvesting lands in a later task)")

    out.append("[PHASE]")
    out.extend(_phase_lines(session))

    out.append(_ROLE)
    return "\n".join(out)
