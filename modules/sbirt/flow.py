"""The whole study protocol, written as data rather than code.

PROTOCOL below is the session: every question, permission, piece of content and
branch, in order. runtime.py is a generic interpreter for it, and there is no
per-question handler anywhere — adding a question means adding a step, and
changing the order means moving one. That is what makes the protocol reviewable
by someone who does not read Python, and diffable when the study changes.

The step vocabulary:

  Label(name)       a jump target, and the node name that appears in logs.
  Route(fn)         a branch. fn(session) returns a label name. Routes may
                    update bookkeeping such as which arm is next, but they
                    never speak and never consult anything the machine has not
                    coded deterministically.
  Tell(unit)        deliver one piece of content, then continue. The unit is
                    either verbatim text, a set of points for the model to
                    word, or an "@name" the engine resolves from session state.
  Gate(key, on_no)  a yes/no permission. A refusal is recorded and jumps to
                    on_no. ask_included means a preceding Tell already posed
                    the question, so the gate must not ask it twice.
  Ask(key, kind)    a question answered in free text or with a 0-10 number.
                    Its `ask` field says where the wording comes from: fixed
                    text, an ask already included in a preceding Tell, or
                    composed by the model from points. With slots, exactly one
                    missing slot is asked per turn — never a stacked question —
                    and the engine holds position until all of them fill.
                    Every captured answer is stored; no ask discards its answer.
  RunItems(itemset) administer a whole instrument, one item per turn, applying
                    its skip rules and scoring it on completion. Adding an item
                    to an instrument requires no change here.
  End(node, close)  stop expecting input, after speaking the close unit. The
                    consent-refused ending speaks nothing, because the pipeline
                    owns that fixed goodbye.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from .templates import FEEDBACK_ASKS_BI

# Which instrument each positive pre-screen arm opens, and the order the arms
# run in — alcohol before drugs, following the study dialogue.
ARM_INSTRUMENT = {"alcohol": "audit", "drugs": "dast_10"}
ARM_ORDER = ("alcohol", "drugs")


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
    slot_points: tuple[tuple[str, str], ...] = ()   # what to ask for each slot
    points: tuple[str, ...] = ()               # for composing without slots


@dataclass(frozen=True)
class RunItems:
    itemset: str                               # "prescreen" | instrument key


@dataclass(frozen=True)
class End:
    node: str                                  # "closed" | "declined"
    close: str = ""                            # unit to speak; "" for silence


def _after_prescreen(session) -> str:
    """Queue every arm the pre-screen came back positive on.

    Pre-screen items are scored 0 for negative and 1 for positive, so anything
    above zero opens that arm. Nothing positive means there is nothing to screen
    and the session closes with the all-negative affirmation.
    """
    session.arms = [arm for arm in ARM_ORDER
                    if session.prescreen.get(arm, 0) > 0]
    return "arm.next" if session.arms else "close.all_negative"


def _next_arm(session) -> str:
    """Move to the next queued arm, or close once they are all done."""
    if session.arms:
        session.arm = session.arms.pop(0)
        return session.arm                     # "alcohol" | "drugs"
    return "close.completed"


def _after_ruler(session) -> str:
    """Skip the ruler follow-ups when there is no readiness number.

    Both follow-ups and the summary quote the number back, so with the item
    unanswered they would have nothing to say. The intervention continues at the
    wrap-up instead.
    """
    if session.readiness.get(session.arm) is None:
        return "bi.wrap"
    return "bi.followups"


def _after_feedback(session) -> str:
    """Finish the arm on a healthy result, otherwise go to the intervention.

    Some zones' feedback text ends by asking permission for the intervention
    itself, and those must not then be asked again — hence two entry points.
    The dependent zone is routed explicitly even though the study's wording for
    it omits the question.
    """
    instrument_key = ARM_INSTRUMENT[session.arm]
    zone = session.assessments[instrument_key].zone
    if zone == "healthy":
        return "arm.next"
    if (instrument_key, zone) in FEEDBACK_ASKS_BI:
        return "bi.included"
    return "bi.ask"


def close_unit(session) -> str:
    """Which closing line the session has earned.

    The standard close promises follow-up questions, which reads badly right
    after somebody declined a permission and was told that was their call. So
    any refusal of a screening, feedback or intervention permission selects the
    other close.

    Refusing the optional education does not count: the protocol carried on
    normally afterwards, so nothing was cut short.

    Pending clinician review.
    """
    declined_hard = any(not k.endswith("edu.permission")
                        for k in session.declined)
    return "close.declined" if declined_hard else "close"


PROTOCOL: tuple = (
    # The greeting clip already asked for consent, so the machine starts at the
    # reply rather than by asking again.
    Gate("consent.opening", on_no="declined", ask_included=True),

    RunItems("prescreen"),
    Route(_after_prescreen),

    Label("arm.next"),
    Route(_next_arm),

    Label("alcohol"),
    # The study asks what, how much and how often as one stacked question.
    # Split into slots, it is asked one part at a time, and somebody who volun-
    # teers all three at once is not asked again. Conversational context only —
    # the scored answers are the instrument items below.
    Ask("alcohol.qf", kind="open", ask="compose",
        slots=("drink", "amount", "frequency"),
        slot_points=(("drink", "what they like to drink"),
                     ("amount", "how much they usually drink"),
                     ("frequency", "how often they usually drink"))),
    Gate("alcohol.edu.permission", on_no="alcohol.screen"),
    Tell("alcohol.edu.standard_drink"),
    Tell("alcohol.edu.limits"),
    Label("alcohol.screen"),
    # Resolved from state, so this only refers back to the standard-drink
    # definition when the education was actually delivered — reachable here
    # either way, since the gate above can skip it.
    Tell("@alcohol.screen.permission"),
    Gate("alcohol.screen.permission", on_no="arm.declined", ask_included=True),
    RunItems("audit"),
    Gate("alcohol.feedback.permission", on_no="arm.declined"),
    Tell("@feedback"),
    Route(_after_feedback),

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

    # Shared by both arms; the "@" units resolve against whichever arm is
    # currently running.
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
    """Where a label sits in PROTOCOL.

    A KeyError means a Route or Gate names a label that does not exist, which is
    a bug in the protocol data rather than anything a session can cause. The
    integrity test catches it before it can reach a conversation.
    """
    return LABELS[name]
