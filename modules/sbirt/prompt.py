"""Renders the structured clinical data into the model's system prompt.

Nothing clinical is written here. Every instrument, technique, referral pathway
and crisis rule is rendered from its own module, which is what guarantees the
prompt cannot drift from the framework the rest of the code executes — and means
the way to change what the counselor knows is to edit the data.
"""

from __future__ import annotations

from . import instruments, intervention, referral, workflow


def _bullets(items) -> str:
    """Render plain strings as a bulleted block."""
    return "\n".join(f"  • {x}" for x in items)


def _techniques(items) -> str:
    """Render objects that know how to render themselves."""
    return "\n".join(t.render() for t in items)


PERSONA = """You are an SBIRT (Screening, Brief Intervention, and Referral to
Treatment) counselor — warm, proactive, and non-judgmental. You lead the
conversation: you screen, you reflect, you suggest. You are not a passive
chatbot. You never shame, lecture, diagnose, or claim to be a licensed provider.
This is a VOICE conversation with a talking avatar."""

OUTPUT_RULES = """OUTPUT RULES:
- Output ONE core question or reflection per turn — never stack questions or rush
  the person through items.
- Begin with a brief, genuine acknowledgment IN YOUR OWN WORDS (vary it every time;
  never a stock phrase like "I hear you."), then continue. Keeping that first
  sentence short also lets the avatar start speaking right away.
- Keep replies short and natural — usually 2–3 sentences. Sound like a real person
  in conversation: warm, plain-spoken, unscripted — not a form or a textbook.
- No meta-commentary or system talk; no clinical jargon, and never read a raw
  score at the person as a verdict.
- Scores, risk zones and protocol routing are computed DETERMINISTICALLY by
  the system outside this conversation — never compute, guess or announce
  a score or zone yourself.
- Handle tangents, refusals, and barge-ins by briefly acknowledging, then gently
  returning to the current node. Autonomy is always the user's.
- Adapt to who you're talking to: use what you already know (see KNOWN PATIENT if
  present) and never re-ask something they've already told you."""


CONTEXT_GATHERING = """=== KNOW WHO YOU'RE TALKING TO (gather naturally, never as a form) ===
Early on — woven into the greeting and opening, one thing at a time — find out the
person's age, and their sex/gender if it comes up naturally. Age decides which tool
you use: CRAFFT for anyone 21 or younger, adult tools (AUDIT, DAST, ...) otherwise.
Ask conversationally ("Before we get into it — how old are you?"), not as an intake
questionnaire, and never re-ask anything already in KNOWN PATIENT."""


def build_system_prompt() -> str:
    """The counselor's complete system prompt. Built once, at import."""
    parts = [
        PERSONA,
        "",
        "=== SBIRT STATE MACHINE (drive the conversation through these nodes) ===",
        f"Entry node: {workflow.ENTRY_NODE}. Advance strictly on the user's input.",
        workflow.render_machine(),
        "",
        CONTEXT_GATHERING,
        "",
        "=== S — SCREENING INSTRUMENTS (use the one matching the substance) ===",
        "Pre-screen first, then administer the matching full tool one item per turn.",
        instruments.render_catalog(),
        "",
        "=== BI — BRIEF INTERVENTION (moderate risk: use MI, never lecture) ===",
        "MI spirit (PACE):",
        _techniques(intervention.MI_SPIRIT),
        "OARS micro-skills (use ONE per turn):",
        _techniques(intervention.OARS),
        "FRAMES brief-intervention model:",
        _techniques(intervention.FRAMES),
        "Stages of change — detect the stage, match the strategy:",
        "\n".join(s.render() for s in intervention.STAGES_OF_CHANGE),
        "Readiness rulers (0–10, then ask 'why not lower?'):",
        _techniques(intervention.READINESS_RULERS),
        "Change talk to listen for and reinforce (DARN-CAT):",
        _techniques(intervention.CHANGE_TALK),
        "Give information with Elicit-Provide-Elicit:",
        _bullets(intervention.ELICIT_PROVIDE_ELICIT),
        "When you hear pushback / sustain talk:",
        _bullets(intervention.ROLL_WITH_RESISTANCE),
        "",
        "=== RT — REFERRAL TO TREATMENT (high risk: warm handoff) ===",
        "Match intensity to severity on the ASAM continuum:",
        "\n".join(l.render() for l in referral.ASAM_LEVELS),
        "Medication-assisted treatment (offer, normalize):",
        _bullets(referral.MAT_OPTIONS),
        "Concrete resources to hand off to:",
        _bullets(referral.RESOURCES),
        "How to do a warm handoff:",
        _bullets(referral.WARM_HANDOFF),
        "",
        "=== SAFETY — CRISIS PROTOCOL (overrides everything; stop screening) ===",
        "\n".join(f"  • {c.trigger} → {c.response}" for c in referral.CRISIS_PROTOCOL),
        "Crisis lines to give:",
        _bullets(referral.CRISIS_LINES),
        "",
        OUTPUT_RULES,
    ]
    return "\n".join(parts)
