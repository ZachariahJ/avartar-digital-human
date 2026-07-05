"""Brief Intervention content (the "BI" in SBIRT).

Motivational Interviewing (MI) scaffolding the counselor must use for moderate
risk: the MI spirit (PACE), micro-skills (OARS), the FRAMES brief-intervention
model, Prochaska/DiClemente stages of change with a matched strategy for each,
the 0–10 readiness rulers, and change-talk cues (DARN-CAT).

All plain data so the clinical technique set can be maintained in one place.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Technique:
    name: str
    detail: str
    example: str = ""

    def render(self) -> str:
        line = f"  • {self.name}: {self.detail}"
        if self.example:
            line += f'  e.g. "{self.example}"'
        return line


# MI spirit — the stance underneath every reflection.
MI_SPIRIT = (
    Technique("Partnership", "Collaborate; the user is the expert on their own life, not a passive patient."),
    Technique("Acceptance", "Absolute worth, accurate empathy, autonomy support, affirmation."),
    Technique("Compassion", "Actively promote the user's welfare; their interests come first."),
    Technique("Evocation", "Draw motivation OUT of the user rather than installing it from outside."),
)

# OARS — the core micro-skills. Exactly one per turn keeps it conversational.
OARS = (
    Technique("Open questions", "Invite elaboration, not yes/no.",
              "What worries you most about your drinking?"),
    Technique("Affirmations", "Name a genuine strength or effort.",
              "It took real honesty to say that out loud."),
    Technique("Reflections", "Mirror meaning/feeling; go for complex over simple reflections.",
              "So it started on weekends, and now it's most nights."),
    Technique("Summaries", "Gather change talk and hand it back to build momentum.",
              "Let me pull that together — you've noticed X, you're worried about Y..."),
)

# FRAMES — the evidence-based brief-intervention checklist.
FRAMES = (
    Technique("Feedback", "Give personal, non-judgmental feedback tied to the screen result."),
    Technique("Responsibility", "Emphasize the choice to change is theirs alone."),
    Technique("Advice", "Offer clear advice to change — only after asking permission."),
    Technique("Menu", "Provide a menu of concrete options, not a single mandate."),
    Technique("Empathy", "Use a warm, reflective, non-confrontational style throughout."),
    Technique("Self-efficacy", "Reinforce optimism and the user's own capacity to change."),
)


@dataclass(frozen=True)
class Stage:
    name: str
    marker: str        # how the user sounds in this stage
    strategy: str      # what the counselor does next

    def render(self) -> str:
        return f"  • {self.name} — sounds like: {self.marker}\n      do: {self.strategy}"


# Transtheoretical stages of change + the matched MI strategy for each.
STAGES_OF_CHANGE = (
    Stage("Pre-contemplation", "No problem; not considering change.",
          "Raise awareness gently, offer info with permission, avoid pushing. Plant a seed."),
    Stage("Contemplation", "Ambivalent; 'yes but'.",
          "Explore ambivalence, decisional balance, tip the scale by evoking change talk."),
    Stage("Preparation", "Intends to change soon; asking how.",
          "Strengthen commitment, help build a concrete, small, specific plan."),
    Stage("Action", "Actively changing.",
          "Affirm effort, problem-solve barriers, reinforce self-efficacy."),
    Stage("Maintenance", "Sustaining change.",
          "Support relapse prevention, celebrate wins, plan for high-risk situations."),
)

# The three MI rulers. Asking the follow-up ("why not lower?") pulls change talk.
READINESS_RULERS = (
    Technique("Importance ruler", "On 0–10, how important is changing this?",
              "You said 6 — why a 6 and not a 3?"),
    Technique("Confidence ruler", "On 0–10, how confident are you that you could?",
              "What would move you from a 4 to a 6?"),
    Technique("Readiness ruler", "On 0–10, how ready are you to start?",
              "What would need to be true for that number to climb?"),
)

# DARN-CAT — listen for and reinforce these; they predict actual change.
CHANGE_TALK = (
    Technique("Desire", "Wanting change.", "I wish I didn't need it to sleep."),
    Technique("Ability", "Confidence in changing.", "I quit once before, so I could again."),
    Technique("Reasons", "Specific arguments for change.", "My kid noticed."),
    Technique("Need", "Urgency.", "I have to do something."),
    Technique("Commitment", "Will do.", "I'm going to cut back."),
    Technique("Activation", "Ready/willing/about to.", "I'm ready to try."),
    Technique("Taking steps", "Already acting.", "I poured them out this morning."),
)

# The safe way to give information inside MI without lecturing.
ELICIT_PROVIDE_ELICIT = (
    "ASK permission and what they already know",
    "PROVIDE one small piece of neutral information",
    "ASK what they make of it",
)

# Respond to 'sustain talk'/pushback by rolling with it — never argue.
ROLL_WITH_RESISTANCE = (
    "Do NOT argue for change — arguing hardens the other side.",
    "Reflect the resistance (simple, amplified, or double-sided reflection).",
    "Reframe, emphasize autonomy ('it's completely your call'), then re-open.",
)
