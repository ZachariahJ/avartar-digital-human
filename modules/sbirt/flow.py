
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from .templates import FEEDBACK_ASKS_BI

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
    kind: str
    ask: str = "fixed"
    slots: tuple[str, ...] = ()
    slot_points: tuple[tuple[str, str], ...] = ()
    points: tuple[str, ...] = ()


@dataclass(frozen=True)
class RunItems:
    itemset: str


@dataclass(frozen=True)
class End:
    node: str
    close: str = ""


def _after_prescreen(session) -> str:
    session.arms = [arm for arm in ARM_ORDER
                    if session.prescreen.get(arm, 0) > 0]
    return "arm.next" if session.arms else "close.all_negative"


def _next_arm(session) -> str:
    if session.arms:
        session.arm = session.arms.pop(0)
        return session.arm
    return "close.completed"


def _after_ruler(session) -> str:
    if session.readiness.get(session.arm) is None:
        return "bi.wrap"
    return "bi.followups"


def _after_feedback(session) -> str:
    instrument_key = ARM_INSTRUMENT[session.arm]
    zone = session.assessments[instrument_key].zone
    if zone == "healthy":
        return "arm.next"
    if (instrument_key, zone) in FEEDBACK_ASKS_BI:
        return "bi.included"
    return "bi.ask"


def close_unit(session) -> str:
    declined_hard = any(not k.endswith("edu.permission")
                        for k in session.declined)
    return "close.declined" if declined_hard else "close"


PROTOCOL: tuple = (
    Gate("consent.opening", on_no="declined"),

    RunItems("prescreen"),
    Route(_after_prescreen),

    Label("arm.next"),
    Route(_next_arm),

    Label("alcohol"),
    Ask("alcohol.qf", kind="open", ask="compose",
        slots=("drink", "amount", "frequency"),
        slot_points=(("drink", "what they like to drink"),
                     ("amount", "how much they usually drink"),
                     ("frequency", "how often they usually drink"))),
    Gate("alcohol.edu.permission", on_no="alcohol.screen"),
    Tell("alcohol.edu.standard_drink"),
    Tell("alcohol.edu.limits"),
    Label("alcohol.screen"),
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
    End("declined", close="close.consent_declined"),
)

LABELS: dict[str, int] = {
    step.name: i for i, step in enumerate(PROTOCOL) if isinstance(step, Label)
}


def label_index(name: str) -> int:
    return LABELS[name]
