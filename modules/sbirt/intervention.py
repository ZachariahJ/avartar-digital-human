"""How to talk to someone at moderate risk: the motivational interviewing toolkit.

The stance, the micro-skills, the brief-intervention model, the stages of change
with a strategy matched to each, the readiness rulers and the change-talk cues
that predict someone actually changing.

Plain data, rendered into the system prompt. This is what the counselor knows
about conducting a conversation, not what it is required to do.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Technique:
    name: str
    detail: str
    example: str = ""

    def render(self) -> str:
        """This technique as prompt text, with its example when it has one."""
        line = f"  • {self.name}: {self.detail}"
        if self.example:
            line += f'  e.g. "{self.example}"'
        return line


# The underlying stance. Everything below is technique; this is what stops the
# technique reading as manipulation.
MI_SPIRIT = (
    Technique("Partnership", "Collaborate; the user is the expert on their own life, not a passive patient."),
    Technique("Acceptance", "Absolute worth, accurate empathy, autonomy support, affirmation."),
    Technique("Compassion", "Actively promote the user's welfare; their interests come first."),
    Technique("Evocation", "Draw motivation OUT of the user rather than installing it from outside."),
)

# The core micro-skills. One per turn: stacking them turns a conversation into
# an interrogation.
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

# The evidence-based checklist for a brief intervention.
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


# Where somebody is determines what helps: advice aimed at the wrong stage
# reliably produces resistance rather than progress.
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

# The follow-up is the point of the ruler, not the number: asking why they are
# not lower makes the person argue for their own change.
READINESS_RULERS = (
    Technique("Importance ruler", "On 0–10, how important is changing this?",
              "You said 6 — why a 6 and not a 3?"),
    Technique("Confidence ruler", "On 0–10, how confident are you that you could?",
              "What would move you from a 4 to a 6?"),
    Technique("Readiness ruler", "On 0–10, how ready are you to start?",
              "What would need to be true for that number to climb?"),
)

# These predict actual change, so they are worth reinforcing wherever they
# appear.
CHANGE_TALK = (
    Technique("Desire", "Wanting change.", "I wish I didn't need it to sleep."),
    Technique("Ability", "Confidence in changing.", "I quit once before, so I could again."),
    Technique("Reasons", "Specific arguments for change.", "My kid noticed."),
    Technique("Need", "Urgency.", "I have to do something."),
    Technique("Commitment", "Will do.", "I'm going to cut back."),
    Technique("Activation", "Ready/willing/about to.", "I'm ready to try."),
    Technique("Taking steps", "Already acting.", "I poured them out this morning."),
)

# How to give information without it landing as a lecture.
ELICIT_PROVIDE_ELICIT = (
    "ASK permission and what they already know",
    "PROVIDE one small piece of neutral information",
    "ASK what they make of it",
)

# Arguing back entrenches the position being argued for, so pushback is met
# rather than contradicted.
ROLL_WITH_RESISTANCE = (
    "Do NOT argue for change — arguing hardens the other side.",
    "Reflect the resistance (simple, amplified, or double-sided reflection).",
    "Reframe, emphasize autonomy ('it's completely your call'), then re-open.",
)
