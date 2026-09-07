"""Detects a crisis without asking a model, and answers it with fixed words.

Whether somebody is in danger is not a judgement this system delegates. Fixed,
reviewable patterns are matched against the transcript, and a hit routes the
session into the crisis protocol with a fixed spoken response — so what a person
in crisis hears is auditable, and does not depend on a model being available,
being right, or being consistent between runs.

The model's own crisis handling stays active alongside this. Either firing is
enough: patterns catch what the model misses, and the model catches cues no
pattern can express. The asymmetry is deliberate — a false positive costs one
unnecessary safety message, a false negative can cost a life.

The patterns assume ASR output: lowercase, no reliable punctuation, no
capitalisation to lean on. Two consequences worth knowing before editing them:

  * "withdrawal" on its own cannot be a trigger, because DAST item 9 asks
    whether the person has ever experienced withdrawal symptoms — the pattern
    would fire on every positive answer to a routine screening question. That
    category therefore matches the acute presentation instead.
  * Negation is not handled, so "I'm not suicidal" fires. That is the intended
    trade, not an oversight.

The spoken responses are clinical safety copy and are fixed strings, never
generated or paraphrased. They await clinician sign-off.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class CrisisHit:
    category: str      # "suicide" | "overdose" | "withdrawal" | "acute_danger"
    pattern: str       # what fired; contains no user text, so it is loggable


# Listed in severity order, and checked in that order, so the first hit is the
# most serious one present rather than merely the first mentioned.
_PATTERNS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("suicide", (
        r"\bsuicid\w*",
        # Restricted to "myself": "this hangover is killing me" is ordinary
        # speech, and especially likely in a drinking conversation.
        r"\bkill(?:ing)?\s+myself\b",
        r"\bend(?:ing)?\s+(?:my|it)\s+(?:life|all)\b",
        r"\btak(?:e|ing)\s+my\s+(?:own\s+)?life\b",
        r"\b(?:hurt|harm|cutt?)(?:ing)?\s+myself\b",
        r"\bself[\s-]?harm\w*",
        r"\bwant(?:ed)?\s+to\s+die\b",
        r"\bbetter\s+off\s+dead\b",
        r"\b(?:don'?t|do\s+not|no\s+reason\s+to)\s+want\s+to\s+(?:live|be\s+alive)\b",
        r"\bno\s+reason\s+to\s+(?:live|keep\s+going)\b",
    )),
    ("overdose", (
        r"\boverdos\w*",
        r"\btook\s+too\s+many\s+(?:pills|of\s+them)\b",
        r"\btook\s+(?:a\s+)?whole\s+bottle\b",
        r"\bcan'?t\s+wake\s+(?:him|her|them)\s+up\b",
        r"\bnot\s+breathing\b",
    )),
    ("withdrawal", (
        # The acute presentation only; see the note in the module docstring
        # about why the word itself cannot appear here.
        r"\bseizures?\b",
        r"\bdelirium\s+tremens\b",
        r"\bthe\s+dts\b",
        r"\bhallucinat\w*",
        r"\bshak(?:ing|es)\s+(?:real\s+)?bad(?:ly)?\b",
    )),
    ("acute_danger", (
        r"\b(?:kill|hurt|shoot|stab)(?:ing)?\s+(?:him|her|them|someone|somebody|my\s+\w+)\b",
        r"\bgoing\s+to\s+hurt\s+\w+\b",
        r"\bpassed\s+out\s+and\s+won'?t\s+wake\b",
    )),
)

_COMPILED = tuple(
    (category, tuple(re.compile(p, re.IGNORECASE) for p in patterns))
    for category, patterns in _PATTERNS
)


def detect(text: str) -> CrisisHit | None:
    """The most severe crisis indication in one utterance, or None.

    A pure function with no model call and no I/O, so it cannot be slow,
    unavailable or nondeterministic at the moment it matters most.
    """
    if not text or not text.strip():
        return None
    for category, patterns in _COMPILED:
        for rx in patterns:
            if rx.search(text):
                return CrisisHit(category=category, pattern=rx.pattern)
    return None


# One fixed response per category. Phone numbers are spaced out ("9 8 8")
# because the synthesizer otherwise reads them as a single number, which is
# useless to someone trying to dial. Pre-rendered into cached clips, so a crisis
# reply plays immediately rather than waiting on generation.
RESPONSES: dict[str, str] = {
    "suicide": (
        "Thank you for telling me — I'm really glad you said that, and I want "
        "to make sure you're safe. Please reach the 988 Suicide and Crisis "
        "Lifeline right now by calling or texting 9 8 8. If you are in "
        "immediate danger, please call 9 1 1. Would you like to talk about "
        "what's going on?"
    ),
    "overdose": (
        "That sounds like it could be a medical emergency. If you or someone "
        "with you may have overdosed, please call 9 1 1 right away. If "
        "opioids may be involved and naloxone, also called Narcan, is "
        "available, use it. Please get medical help now — we can continue "
        "talking after you are safe."
    ),
    "withdrawal": (
        "What you're describing can be a sign of serious withdrawal, and "
        "stopping alcohol or sedatives suddenly can be dangerous. Please seek "
        "urgent medical care now, or call 9 1 1 if it's severe. Please don't "
        "try to get through this alone or quit cold turkey without medical "
        "support."
    ),
    "acute_danger": (
        "It sounds like someone may be in immediate danger. Please call "
        "9 1 1 right now. If you can, stay somewhere safe. You can also call "
        "or text 9 8 8 to talk to a crisis counselor at any time."
    ),
}

# Checked at import rather than at use: a missing category would otherwise
# surface as a KeyError during a crisis.
assert set(RESPONSES) == {c for c, _ in _PATTERNS}
