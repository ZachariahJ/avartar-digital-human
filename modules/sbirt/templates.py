"""Versioned fixed script for the study protocol (P5 data).

Every fixed thing the counselor says — permissions, standard-drink education,
zone feedback, brief-intervention lines, closing — transcribed VERBATIM from
the study's authoritative dialogue ("AI SBIRT app dialogue for study.docx" in
SBIRT_Reference/). The runtime speaks these strings exactly; the LLM never
generates or paraphrases them. Because they are fixed, every one of them can
be pre-rendered to a cached clip (see pipeline.ensure_fixed_clip).

Clip staleness is handled by the sidecar text check in ensure_fixed_clip;
the consent audit anchors the exact greeting wording by content hash.

Documented normalizations of source-document typos (flagged for clinician
review; nothing else was reworded):
  • "How much you usually drink?"        → "How much do you usually drink?"
  • "I recommended that you cut down"    → "I recommend that you cut down"
  • ruler: "10 meaning you ready"        → "... you are ready"
  • The Dependent-alcohol feedback in the source embeds a duplicate
    "May I give you feedback..." permission sentence (already asked one turn
    earlier); the duplicate is dropped here.
"""

from __future__ import annotations

# Substance-arm wording: the source writes "alcohol/drug use" and "using
# alcohol/drugs" meaning "pick the arm's substance"; instantiated per arm.
_USE_NOUN = {"alcohol": "alcohol", "drugs": "drug"}    # "... about {x} use?"
_STOP_NOUN = {"alcohol": "alcohol", "drugs": "drugs"}  # "... stop using {x}"


# --------------- Fixed lines, keyed for clip caching ---------------
FIXED: dict[str, str] = {
    # Alcohol arm — standard-drink education. (The old three-questions-in-one
    # "alcohol.qf" line is gone: the flow's slot-ask covers drink/amount/
    # frequency ONE question at a time, phrased by the LLM.)
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
    # Spoken INSTEAD of the line above when the user declined the standard-
    # drink education: it must not claim a definition "we just discussed" was
    # discussed. The flow picks the variant from state.covered. Wording is the
    # source line minus the false reference. PENDING CLINICIAN REVIEW.
    "alcohol.screen.permission.no_defn": (
        "May I ask you a few more questions about your use of alcohol in the "
        "past year?"
    ),
    "alcohol.feedback.permission": (
        "May I give you feedback on the questions you just answered about "
        "your alcohol use?"
    ),

    # Drug arm
    "drugs.kind": "What kind of drugs do you use?",
    "drugs.screen.permission": (
        "May I ask you a few more questions about your drug use in the past "
        "year?"
    ),
    "drugs.feedback.permission": (
        "May I give you some feedback about the questions you just answered "
        "about your drug use?"
    ),

    # Close (spoken at the end of a COMPLETED session — study text verbatim;
    # its "few more questions about your experiences" refers to the study's
    # post-session experience survey)
    "close": (
        "Thank you for participating in this process. We have a few more "
        "questions to ask you about your experiences."
    ),
    # Close spoken when the session ends because the user DECLINED a screening/
    # feedback/BI permission: it must not promise more questions right after
    # "that's your call" (the session actually ends here — the old close
    # contradicted itself on this path). Source has no dedicated text; minimal
    # autonomy-respecting goodbye. PENDING CLINICIAN REVIEW.
    "close.declined": (
        "Thank you for your time today. Your provider can pick any of this "
        "up with you during your visit, whenever you're ready."
    ),
    # All pre-screens negative: the source dialogue has no dedicated text, so a
    # minimal neutral affirmation precedes the standard close. PENDING
    # CLINICIAN REVIEW.
    "prescreen.all_negative": (
        "Thank you for answering those questions. Based on your answers, your "
        "use is not likely to cause you any health problems."
    ),
    # User declines a mid-protocol permission (education / screening items /
    # feedback / more questions). Source gives no text; minimal autonomy-
    # respecting line. PENDING CLINICIAN REVIEW.
    "permission.declined": (
        "That's completely your call, and that's fine."
    ),
    # User asks to STOP the whole conversation mid-protocol (T22 abort).
    # Source gives no text; autonomy-respecting goodbye, no retention
    # attempt, answers so far stay recorded for the provider. PENDING
    # CLINICIAN REVIEW.
    "close.aborted": (
        "Of course — we can stop here, and that's completely fine. Thank "
        "you for your time today. Anything you shared stays confidential, "
        "and your provider can pick this up with you whenever you're ready."
    ),
    # A question the person cannot / will not answer after the recall-anchor
    # probe (F1/F2 missing-data exit): acknowledge and move on — the manual's
    # guidance is to note uncertainties on the record, not to interrogate
    # (SBIRT_REF.pdf p.18). Source gives no text; PENDING CLINICIAN REVIEW.
    "item.skipped": (
        "That's okay — we can set that one aside. I'll make a note of it "
        "for your provider."
    ),
    # Brief-intervention closer (source line; the flow Tells it right before
    # listening for the open answer).
    "bi.leaves_you": "So where does this leave you?",
}


# --------------- Zone feedback (verbatim, keyed by instrument + zone) --------
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

# Zones whose feedback text ends by asking permission to continue into the
# brief intervention ("May I ask you some more questions about this?").
# NOTE: the source's Dependent-DRUG text does NOT end with that question;
# the machine still routes dependent to the BI/referral arm explicitly.
FEEDBACK_ASKS_BI: frozenset[tuple[str, str]] = frozenset(
    k for k, v in FEEDBACK.items()
    if v.rstrip().endswith("May I ask you some more questions about this?")
)


def feedback_text(instrument_key: str, zone: str) -> str:
    """The verbatim feedback for a completed screen. KeyError = a zone the
    protocol does not define — that must surface, never be improvised."""
    return FEEDBACK[(instrument_key, zone)]


# --------------- Brief intervention (decisional balance + ruler) -------------
def bi_permission(arm: str) -> str:
    """Source line 'May I ask you some more questions about your alcohol/drug
    use?' — spoken when a feedback text doesn't already end with the BI ask."""
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


BI_LEAVES_YOU = FIXED["bi.leaves_you"]   # single source: the FIXED entry


# --------------- Content units (T8): what a turn must convey ---------------
# A Unit is ONE thing the counselor delivers in a turn. verbatim=True units
# speak `literal` exactly (study fidelity; cacheable as a clip). verbatim=False
# units give the LLM `points` — the clinical content it must cover in its own
# words for THIS person (typically grounded in state the engine passes along,
# e.g. the user's own likes/dislikes wording). The LLM may rephrase points;
# it may not drop, extend, or contradict them.

from dataclasses import dataclass


@dataclass(frozen=True)
class Unit:
    id: str
    verbatim: bool
    literal: str = ""             # verbatim=True: the exact text
    points: tuple[str, ...] = ()  # verbatim=False: content the LLM must cover


# Non-verbatim units: the brief-intervention reflections/summaries the LLM
# phrases each time (these were inline instructions in runtime.py; as data
# they are reviewable and testable). {likes}/{dislikes}/{ruler}... slots are
# filled by the engine from captured state before handing to the LLM.
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
    )
}


def all_fixed_utterances() -> dict[str, str]:
    """Every fixed utterance the protocol can ever speak, keyed exactly as the
    runtime emits them (runtime.Say.key). Single source for clip pre-warming
    (pipeline.prewarm_fixed_clips) and for the everything-is-cacheable tests.
    Parameterized lines are enumerated over their full domain (ruler 0-10) so
    the runtime never has to render a 'fixed' line on the fly."""
    from .instruments import BY_KEY, PRE_SCREEN  # local: avoid import cycles

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
    return out   # bi.leaves_you arrives via FIXED
