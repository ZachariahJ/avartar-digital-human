"""The study protocol as ONE declarative program (T5-T10).

Every question, permission gate, content unit and branch of the SBIRT session
is a STEP in `PROTOCOL` below, executed by the generic engine in runtime.py.
There is no per-question handler code anywhere: adding a question = adding a
step (or an instrument Item); changing the order = reordering data.

Step vocabulary:
  Label(name)       jump target; also the human-readable node name in logs.
  Route(fn)         deterministic branch: fn(session) -> label name. Route
                    functions may update queue state (e.g. pop the next arm)
                    but never speak and never read anything the machine
                    didn't code deterministically.
  Tell(unit)        deliver one content unit, then fall through. `unit` is a
                    templates.FIXED key (verbatim, cacheable), a
                    templates.POINTS_UNITS id (LLM phrases the points), or an
                    "@resolver" computed from state by the engine (e.g.
                    "@feedback" -> the zone feedback for the completed screen).
  Gate(key, on_no)  yes/no permission. Speaks FIXED[key] as the ask unless
                    ask_included=True (a preceding Tell already asked it).
                    "no" is recorded in session.declined and jumps to on_no.
  Ask(key, kind)    one conversational question the user answers in free
                    text (kind="open") or with a 0-10 number (kind="number").
                    ask="fixed"    speak FIXED[key] / @resolver verbatim
                    ask="included" a preceding Tell already asked it
                    ask="compose"  the LLM phrases the ask from points;
                                   with `slots`, it asks ONE missing slot at
                                   a time (never a multi-question stack) and
                                   the engine holds until all slots fill.
                    The captured answer ALWAYS lands in session state —
                    there is no ask whose answer is discarded.
  RunItems(itemset) administer a whole instrument ("prescreen", "audit",
                    "dast_10") one item per turn via next_item_index (skip
                    rules included); scoring happens at completion. Adding
                    an item to the instrument data changes NOTHING here.
  End(node, close)  terminal: speak the close unit (or nothing for the
                    consent-declined end, whose fixed goodbye the pipeline
                    owns) and stop expecting input.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from .templates import FEEDBACK_ASKS_BI

# Which full instrument each positive pre-screen arm opens (protocol order:
# alcohol before drugs, matching the study dialogue).
ARM_INSTRUMENT = {"alcohol": "audit", "drugs": "dast_10"}
ARM_ORDER = ("alcohol", "drugs")


# --------------- Step types ---------------

@dataclass(frozen=True)
class Label:
    name: str


@dataclass(frozen=True)
class Route:
    fn: Callable


@dataclass(frozen=True)
class Tell:
    unit: str


@dataclass(frozen=True)
class Gate:
    key: str
    on_no: str
    ask_included: bool = False


@dataclass(frozen=True)
class Ask:
    key: str
    kind: str                                  # "open" | "number"
    ask: str = "fixed"                         # "fixed" | "included" | "compose"
    slots: tuple[str, ...] = ()
    slot_points: tuple[tuple[str, str], ...] = ()   # slot -> what to ask for it
    points: tuple[str, ...] = ()               # compose without slots


@dataclass(frozen=True)
class RunItems:
    itemset: str                               # "prescreen" | instrument key


@dataclass(frozen=True)
class End:
    node: str                                  # "closed" | "declined"
    close: str = ""                            # unit to speak; "" = silent


# --------------- Deterministic routes ---------------

def _after_prescreen(session) -> str:
    """Queue the positive arms in protocol order; nothing positive closes the
    session with the all-negative affirmation. Pre-screen options are
    (negative, positive) scored 0/1, so code > 0 = positive."""
    session.arms = [arm for arm in ARM_ORDER
                    if session.prescreen.get(arm, 0) > 0]
    return "arm.next" if session.arms else "close.all_negative"


def _next_arm(session) -> str:
    if session.arms:
        session.arm = session.arms.pop(0)
        return session.arm                     # "alcohol" | "drugs"
    return "close.completed"


def _after_ruler(session) -> str:
    """The two ruler follow-ups and the ruler summary presuppose a readiness
    number; when the ruler went unanswered (marked missing after the F2
    probes), the BI continues at the wrap-up instead of dereferencing a
    number that was never given."""
    if session.readiness.get(session.arm) is None:
        return "bi.wrap"
    return "bi.followups"


def _after_feedback(session) -> str:
    """Healthy zone finishes the arm; every other zone routes into the brief
    intervention. Whether the BI permission still needs ASKING depends on
    whether the zone's feedback text already ends with the ask
    (templates.FEEDBACK_ASKS_BI) — the machine routes dependent explicitly
    even though the source's dependent-drug text drops the question."""
    instrument_key = ARM_INSTRUMENT[session.arm]
    zone = session.assessments[instrument_key].zone
    if zone == "healthy":
        return "arm.next"
    if (instrument_key, zone) in FEEDBACK_ASKS_BI:
        return "bi.included"
    return "bi.ask"


def close_unit(session) -> str:
    """Which close the session earned (T10): a session that ended because the
    user declined a screening/feedback/BI permission must not promise more
    questions right after 'that's your call' — the study close text is only
    for paths that actually completed. Declining the optional education alone
    does NOT count (the protocol continued normally after it).
    PENDING CLINICIAN REVIEW."""
    declined_hard = any(not k.endswith("edu.permission")
                        for k in session.declined)
    return "close.declined" if declined_hard else "close"


# --------------- The protocol program ---------------

PROTOCOL: tuple = (
    # Consent — the fixed greeting clip already asked it; the machine's job
    # starts at the user's reply. Decline ends the session via the pipeline's
    # fixed decline path.
    Gate("consent.opening", on_no="declined", ask_included=True),

    # Pre-screen: the study's 3 questions, one per turn.
    RunItems("prescreen"),
    Route(_after_prescreen),

    Label("arm.next"),
    Route(_next_arm),

    # ---------------- Alcohol arm ----------------
    Label("alcohol"),
    # Q/F exploration — the source's three-question stack is now three slots
    # asked ONE at a time (and a user who answers all three in one breath is
    # asked nothing twice). Context for the conversation; the AUDIT items
    # below are what gets scored.
    Ask("alcohol.qf", kind="open", ask="compose",
        slots=("drink", "amount", "frequency"),
        slot_points=(("drink", "what they like to drink"),
                     ("amount", "how much they usually drink"),
                     ("frequency", "how often they usually drink"))),
    Gate("alcohol.edu.permission", on_no="alcohol.screen"),
    Tell("alcohol.edu.standard_drink"),
    Tell("alcohol.edu.limits"),
    Label("alcohol.screen"),
    # Variant picked from state.covered: only claims "the standard drink
    # definition we just discussed" if the education was actually delivered.
    Tell("@alcohol.screen.permission"),
    Gate("alcohol.screen.permission", on_no="arm.declined", ask_included=True),
    RunItems("audit"),
    Gate("alcohol.feedback.permission", on_no="arm.declined"),
    Tell("@feedback"),
    Route(_after_feedback),

    # ---------------- Drug arm ----------------
    Label("drugs"),
    Ask("drugs.kind", kind="open", ask="fixed"),
    Ask("drugs.qf", kind="open", ask="compose",
        points=("Ask how much and how often the person uses the drug or "
                "drugs they just named ({drugs_kind}). One natural sentence; "
                "ask nothing else.",)),
    Gate("drugs.screen.permission", on_no="arm.declined"),
    RunItems("dast_10"),
    Gate("drugs.feedback.permission", on_no="arm.declined"),
    Tell("@feedback"),
    Route(_after_feedback),

    # ------- Brief intervention (shared; @keys resolve per session.arm) -------
    Label("bi.ask"),
    Gate("@bi.permission", on_no="arm.declined"),
    Route(lambda s: "bi.body"),
    Label("bi.included"),
    Gate("@bi.permission", on_no="arm.declined", ask_included=True),
    Label("bi.body"),
    Ask("@bi.likes", kind="open", ask="fixed"),
    Ask("@bi.dislikes", kind="open", ask="fixed"),
    Tell("bi.summary.balance"),
    Tell("@bi.recommend"),
    Tell("@bi.ruler"),
    Ask("bi.ruler", kind="number", ask="included"),
    Route(_after_ruler),
    Label("bi.followups"),
    Tell("@bi.why_not_lower"),
    Ask("bi.why_not_lower", kind="open", ask="included"),
    Tell("@bi.why_not_higher"),
    Ask("bi.why_not_higher", kind="open", ask="included"),
    Tell("bi.summary.rulers"),
    Label("bi.wrap"),
    Tell("bi.leaves_you"),
    Ask("bi.leaves_you", kind="open", ask="included"),
    Tell("bi.reflect"),
    Route(lambda s: "arm.next"),

    # ---------------- Shared endings ----------------
    Label("arm.declined"),
    Tell("permission.declined"),
    Route(lambda s: "arm.next"),

    Label("close.all_negative"),
    Tell("prescreen.all_negative"),
    End("closed", close="close"),

    Label("close.completed"),
    End("closed", close="@close"),

    Label("declined"),
    End("declined"),
)

LABELS: dict[str, int] = {
    step.name: i for i, step in enumerate(PROTOCOL) if isinstance(step, Label)
}


def label_index(name: str) -> int:
    """Index of a jump target. KeyError here is a program bug (a Route/Gate
    naming a label that doesn't exist) — the integrity test catches it."""
    return LABELS[name]
