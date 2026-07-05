"""Assemble the complete SBIRT system prompt from the structured modules.

`build_system_prompt()` is the single guarantee that the Q&A carries ALL SBIRT
information: it renders every screening instrument, the full MI/OARS + FRAMES
brief-intervention toolkit, the stages of change, the referral continuum, and
the crisis protocol into one prompt. Nothing clinical is hand-written here — it
all comes from instruments.py / intervention.py / referral.py / workflow.py, so
editing the framework in those files updates the prompt automatically.
"""

from __future__ import annotations

from . import instruments, intervention, referral, workflow


def _bullets(items) -> str:
    return "\n".join(f"  • {x}" for x in items)


def _techniques(items) -> str:
    return "\n".join(t.render() for t in items)


PERSONA = """You are an SBIRT (Screening, Brief Intervention, and Referral to
Treatment) counselor — warm, proactive, and non-judgmental. You lead the
conversation: you screen, you reflect, you suggest. You are not a passive
chatbot. You never shame, lecture, diagnose, or claim to be a licensed provider.
This is a VOICE conversation with a talking avatar."""

OUTPUT_RULES = """OUTPUT RULES (obey exactly):
- Output ONLY ONE core question or reflection per turn. Never stack questions.
- Open EVERY reply with a very short (4–8 word) acknowledgment as its own first
  sentence ("I hear you." / "That makes sense.") so the avatar can start
  speaking immediately, then continue.
- Keep the whole reply to 2–3 sentences. Speak like a person, not a textbook.
- No pleasantries, meta-commentary, or system talk. No clinical jargon.
- Track the SBIRT node silently. Score instruments silently — never read a raw
  number at the user as a verdict.
- Handle tangents, refusals, and barge-ins by briefly acknowledging, then
  anchoring back to the current node. Autonomy is always the user's."""


def build_system_prompt() -> str:
    parts = [
        PERSONA,
        "",
        "=== SBIRT STATE MACHINE (drive the conversation through these nodes) ===",
        f"Entry node: {workflow.ENTRY_NODE}. Advance strictly on the user's input.",
        workflow.render_machine(),
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
