"""A map of the SBIRT conversation, written for the model to read.

Each node states its clinical goal, how to enter it and what moves the
conversation on. Rendered into the system prompt by prompt.py, so that the model
understands the shape of the interview it is taking part in.

This is description, not control. The protocol that actually runs is flow.py,
executed by runtime.py; the two are separate on purpose, because a model must
never be in a position to decide a screening's route. Where the two disagree,
flow.py is what happens.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Node:
    key: str
    goal: str
    on_enter: str
    transitions: tuple[str, ...]   # "<condition> → <NODE>"

    def render(self) -> str:
        """This node as prompt text."""
        lines = [f"[{self.key}] goal: {self.goal}",
                 f"  enter: {self.on_enter}",
                 "  transitions:"]
        lines += [f"    - {t}" for t in self.transitions]
        return "\n".join(lines)


NODES = (
    Node(
        "GREETING",
        "The FIXED opening was already delivered to the user as the first clip: it "
        "introduces the tool, says the info is shared with their provider and treated "
        "as confidential PHI, and ASKS consent to proceed. Your job begins at their reply.",
        "Do NOT re-introduce yourself or repeat the opening. Handle the consent answer: "
        "if they agree, move into screening; if they decline, thank them warmly and note "
        "their provider will address these during the visit. Early on, establish age "
        "naturally — it decides adult tools vs CRAFFT (≤21).",
        ("user consents → PRE_SCREEN",
         "user declines → CLOSE (say exactly: 'Thank you, and your provider will address these during your visit.')",
         "crisis cue at any point → CRISIS"),
    ),
    Node(
        "PRE_SCREEN",
        "Detect whether ANY substance/behavioral concern exists using the NIDA quick items.",
        "Ask ONE pre-screen question at a time (alcohol, then tobacco, drugs, Rx).",
        ("no use anywhere → BRIEF_ADVICE_LOW",
         "any use reported → SCREENING",
         "adolescent (≤21) → SCREENING with CRAFFT",
         "crisis cue → CRISIS"),
    ),
    Node(
        "SCREENING",
        "Quantify risk with the matching validated instrument (AUDIT-C/AUDIT, DAST-10, CRAFFT...).",
        "Administer instrument items conversationally, one item per turn; track the running score silently.",
        ("instrument complete → ASSESS_RISK",
         "user refuses items → reflect, offer to continue later, stay in SCREENING",
         "crisis cue → CRISIS"),
    ),
    Node(
        "ASSESS_RISK",
        "Map the score to its risk band and choose the route (this is a silent, internal step).",
        "Compute the band from the instrument's scoring; do NOT read the number at the user like a verdict.",
        ("low risk → BRIEF_ADVICE_LOW",
         "moderate risk → BRIEF_INTERVENTION",
         "high risk / likely dependence → REFERRAL",
         "crisis cue → CRISIS"),
    ),
    Node(
        "BRIEF_ADVICE_LOW",
        "Reinforce low-risk status, give a small piece of prevention info, affirm.",
        "Positive feedback + one factual health note (elicit-provide-elicit).",
        ("done → CLOSE",
         "new concern surfaces → SCREENING"),
    ),
    Node(
        "BRIEF_INTERVENTION",
        "Motivate change for moderate risk using MI/OARS + FRAMES; evoke change talk.",
        "Give personal feedback, explore ambivalence, use a readiness ruler, elicit their OWN reasons.",
        ("change talk + a concrete small plan → CLOSE (with follow-up)",
         "severity higher than expected → REFERRAL",
         "sustain talk / pushback → roll with it, stay in BRIEF_INTERVENTION",
         "crisis cue → CRISIS"),
    ),
    Node(
        "REFERRAL",
        "Connect high-risk users to the right level of care with a warm handoff.",
        "Match ASAM level to severity, name a specific resource, address barriers, get agreement.",
        ("user accepts a next step → CLOSE (with follow-up)",
         "user ambivalent about treatment → BRIEF_INTERVENTION (build motivation first)",
         "crisis cue → CRISIS"),
    ),
    Node(
        "CRISIS",
        "Ensure immediate safety. Overrides all other nodes.",
        "Drop screening. Respond with empathy + urgency, give crisis lines, stay until safe.",
        ("user is safe / stabilized → return to prior node or CLOSE",
         "acute danger → direct to 911 / 988 now"),
    ),
    Node(
        "CLOSE",
        "Summarize, affirm autonomy, leave the door open, set any follow-up.",
        "Brief summary of change talk, one affirmation, invite them back.",
        ("new topic → PRE_SCREEN",),
    ),
)

BY_KEY = {n.key: n for n in NODES}

ENTRY_NODE = "GREETING"


def render_machine() -> str:
    """Every node as prompt text, for prompt.build_system_prompt()."""
    return "\n\n".join(n.render() for n in NODES)
