
from __future__ import annotations

_USE_NOUN = {"alcohol": "alcohol", "drugs": "drug"}
_STOP_NOUN = {"alcohol": "alcohol", "drugs": "drugs"}


FIXED: dict[str, str] = {
    "consent.opening": (
        "May I ask you some questions about your health?"
    ),

    "alcohol.edu.permission": (
        "May I provide you some more information about drinking alcohol?"
    ),
    "alcohol.edu.standard_drink": (
        "We define a drink as: a 12-ounce beer, or a 5 oz glass of wine, or a "
        "shot--1.5 ounces of spirits, like whiskey or vodka. These all have "
        "about the same about of alcohol in them."
    ),
    "alcohol.edu.limits": (
        "It is recommended that men under that age of 65 have no more than 14 "
        "drinks per week and no more than 0-4 drinks per day. "
        "It is recommended that women of all ages have no more than 7 standard "
        "drinks per week and no more than 0-3 drinks per day. "
        "It is recommended that men 65 and over have no more than 7 standard "
        "drinks per week and no more than 0-3 drinks per day. "
        "It is recommended that people under that legal age for drinking and "
        "pregnant women have no alcohol."
    ),
    "alcohol.screen.permission": (
        "May I ask you a few more questions about your use of alcohol in the "
        "past year? In these questions a drink refers to the standard drink "
        "definition we just discussed."
    ),
    "alcohol.screen.permission.no_defn": (
        "May I ask you a few more questions about your use of alcohol in the "
        "past year?"
    ),
    "alcohol.feedback.permission": (
        "May I give you feedback on the questions you just answered about "
        "your alcohol use?"
    ),

    "drugs.kind": "What kind of drugs do you use?",
    "drugs.screen.permission": (
        "May I ask you a few more questions about your drug use in the past "
        "year?"
    ),
    "drugs.feedback.permission": (
        "May I give you some feedback about the questions you just answered "
        "about your drug use?"
    ),

    "prescreen.all_negative": (
        "Thank you for answering those questions. Based on your answers, your "
        "use is not likely to cause you any health problems."
    ),
    "permission.declined": (
        "That's completely your call, and that's fine."
    ),
    "close.aborted": (
        "Of course — we can stop here, and that's completely fine. Thank "
        "you for your time today. Anything you shared stays confidential, "
        "and your provider can pick this up with you whenever you're ready."
    ),
    "close.consent_declined": (
        "Thank you, and your provider will address these during your visit."
    ),
    "close.crisis": (
        "Thank you for telling me that. Please get help right now: if you are "
        "in immediate danger, call nine one one. You can also call or text "
        "nine eight eight, the Suicide and Crisis Lifeline, at any time. "
        "I am going to stop here, and your medical provider will follow up "
        "with you."
    ),
    "item.skipped": (
        "That's okay — we can set that one aside. I'll make a note of it "
        "for your provider."
    ),
    "bi.leaves_you": "So where does this leave you?",

    "aside.offer_pause": (
        "We can stop here for today if you'd rather — your provider can pick "
        "this up with you whenever you're ready. Would you like to keep going?"
    ),
}


FEEDBACK: dict[tuple[str, str], str] = {
    ("audit", "healthy"): (
        "Based on your answers you are using alcohol within normal "
        "recommended limits. This means that your use of alcohol is not "
        "likely to cause you any health problems. I encourage you to continue "
        "using alcohol in this manner so that you do not experience health "
        "problems due to your alcohol use."
    ),
    ("audit", "risky"): (
        "Based on your answers you are using alcohol at risky levels. This "
        "means that you may experience negative consequences from your "
        "alcohol use that may impact your health and well-being. It is "
        "recommended that you cut down on the amount and frequency of your "
        "alcohol use, but you need to be ready. May I ask you some more "
        "questions about this?"
    ),
    ("audit", "harmful"): (
        "Based on your answers you are using alcohol at harmful levels. This "
        "means that you are experiencing negative consequences from your "
        "alcohol use that negatively impact your health and well-being. It is "
        "recommended that you cut down on the amount and frequency of your "
        "alcohol use, but you need be ready. May I ask you some more "
        "questions about this?"
    ),
    ("audit", "dependent"): (
        "Based on your answers you are using alcohol at levels that indicate "
        "you have become dependent on alcohol. This means that you are "
        "experiencing negative consequences from your alcohol use that "
        "negatively impact your health and well-being. It is recommended that "
        "you cut down on the amount and frequency of your alcohol use, but "
        "you need to be ready. May I ask you some more questions about this?"
    ),
    ("dast_10", "healthy"): (
        "Based on your answers you are using drugs in a manner that is not "
        "likely to negatively impact your health and well-being. I "
        "congratulate you on taking care of your health and encourage you to "
        "not use drugs in a manner that may cause health problems for you."
    ),
    ("dast_10", "risky"): (
        "Based on your answers you are using drugs at risky levels. This "
        "means that you may experience negative consequences from your drug "
        "use that may impact your health and well-being. It is recommended "
        "that you stop using drugs or cut down on the amount and frequency of "
        "your drug use, but you need be ready. May I ask you some more "
        "questions about this?"
    ),
    ("dast_10", "harmful"): (
        "Based on your answers you are using drugs at harmful levels. This "
        "means that you are experiencing negative consequences from your drug "
        "use that negatively impact your health and well-being. It is "
        "recommended that you stop using drugs or cut down on the amount and "
        "frequency of your drug use, but you need be ready. May I ask you "
        "some more questions about this?"
    ),
    ("dast_10", "dependent"): (
        "Based on your answers you are using drugs at levels that indicate "
        "you have become dependent on drugs. This means that you are "
        "experiencing negative consequences from your drug use that "
        "negatively impact your health and well-being. It is recommended that "
        "you stop using drugs or cut down on the amount and frequency of your "
        "drug use, but you need be ready."
    ),
}

FEEDBACK_ASKS_BI: frozenset[tuple[str, str]] = frozenset(
    k for k, v in FEEDBACK.items()
    if v.rstrip().endswith("May I ask you some more questions about this?")
)


def feedback_text(instrument_key: str, zone: str) -> str:
    return FEEDBACK[(instrument_key, zone)]


def bi_permission(arm: str) -> str:
    return f"May I ask you some more questions about your {_USE_NOUN[arm]} use?"


def bi_likes(arm: str) -> str:
    return f"What do you like about {_USE_NOUN[arm]} use?"


def bi_dislikes(arm: str) -> str:
    return f"What do you dislike about {_USE_NOUN[arm]} use?"


def bi_recommend(arm: str) -> str:
    return (f"Based on your answers I recommend that you cut down or stop "
            f"using {_STOP_NOUN[arm]}, but you have to be ready.")


def bi_ruler(arm: str) -> str:
    s = _STOP_NOUN[arm]
    return (f"Based on a scale from 0 to 10, with 0 meaning you are not at all "
            f"ready to cut down or stop using {s} and 10 meaning you are "
            f"ready right now to cut down or stop using {s}, where would you "
            f"say you are?")


def bi_why_not_lower(value: int) -> str:
    return f"Why are you a {value} and not a 1 or 2?"


def bi_why_not_higher(value: int) -> str:
    return f"Why are you a {value} and not a 9 or 10?"


from dataclasses import dataclass


@dataclass(frozen=True)
class Unit:

    id: str
    verbatim: bool
    literal: str = ""
    points: tuple[str, ...] = ()


POINTS_UNITS: dict[str, Unit] = {
    u.id: u for u in (
        Unit("bi.summary.balance", verbatim=False, points=(
            "Summarize back FIRST what the person said they LIKE about their "
            "use, THEN what they DISLIKE, using their own words where possible.",
            "One or two sentences. No advice, no new clinical content.",
        )),
        Unit("bi.summary.rulers", verbatim=False, points=(
            "Summarize the person's reasons for not being a 9 or 10, then "
            "their reasons for not being a 1 or 2, using their own words.",
            "One or two sentences. No advice.",
        )),
        Unit("bi.reflect", verbatim=False, points=(
            "In one brief sentence, reflect what the person just said about "
            "where this leaves them. No new questions.",
        )),
        Unit("close", verbatim=False, points=(
            "Thank them for taking part in this process.",
            "Tell them staff will follow up with them about their "
            "experiences — this means the study's later survey, not more "
            "questions from you now.",
            "Two sentences at most. No new questions, no advice, no summary "
            "of what they said.",
        )),
        Unit("close.declined", verbatim=False, points=(
            "Thank them for their time today.",
            "Tell them their provider can pick any of this up with them "
            "during their visit, whenever they are ready.",
            "Two sentences at most. Do not revisit what they declined, and "
            "do not promise a follow-up survey — this session was cut short.",
        )),
    )
}


def all_fixed_utterances() -> dict[str, str]:
    from .instruments import BY_KEY, PRE_SCREEN

    out = dict(FIXED)
    for (ins_key, zone), text in FEEDBACK.items():
        out[f"feedback.{ins_key}.{zone}"] = text
    for q in PRE_SCREEN:
        out[f"prescreen.{q.key}"] = q.item.text
    for ins_key in ("audit", "dast_10"):
        ins = BY_KEY[ins_key]
        if ins.preamble:
            out[f"{ins_key}.preamble"] = ins.preamble
        for i, item in enumerate(ins.items):
            out[f"{ins_key}.item.{i}"] = item.text
    for arm in ("alcohol", "drugs"):
        out[f"bi.permission.{arm}"] = bi_permission(arm)
        out[f"bi.likes.{arm}"] = bi_likes(arm)
        out[f"bi.dislikes.{arm}"] = bi_dislikes(arm)
        out[f"bi.recommend.{arm}"] = bi_recommend(arm)
        out[f"bi.ruler.{arm}"] = bi_ruler(arm)
    for v in range(11):
        out[f"bi.why_not_lower.{v}"] = bi_why_not_lower(v)
        out[f"bi.why_not_higher.{v}"] = bi_why_not_higher(v)
    return out
