"""The SBIRT state machine (the "engineer perspective" from CLAUDE.md).

Defines the conversation nodes, what each node's clinical goal is, how to enter
it, and the conditions that transition to the next node. The LLM drives the
machine turn-by-turn; this module is the authoritative map it follows so state
transitions are deterministic and edge cases (tangents, refusals, barge-ins) are
handled by anchoring back to the current node rather than improvising.
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
        lines = [f"[{self.key}] goal: {self.goal}",
                 f"  enter: {self.on_enter}",
                 "  transitions:"]
        lines += [f"    - {t}" for t in self.transitions]
        return "\n".join(lines)


NODES = (
    Node(
        "GREETING",
        "Open warmly, set the frame (brief, confidential check-in), get consent to talk.",
        "Introduce yourself in one line and ask an open opening question.",
        ("user engages → PRE_SCREEN",
         "user declines → CLOSE (respect autonomy)",
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
    return "\n\n".join(n.render() for n in NODES)
