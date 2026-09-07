"""Everything the counselor says word for word, transcribed from the study script.

Permissions, the standard-drink education, zone feedback, the brief-intervention
lines and the closings. The engine speaks these exactly; nothing here is ever
generated or paraphrased, because study fidelity depends on every participant
hearing the same words. Being fixed is also what makes them cacheable as
pre-rendered clips.

Edits to these strings are clinical changes, not copy edits. Four departures
from the source document were made deliberately and are recorded here:

  * "How much you usually drink?" reads "How much do you usually drink?"
  * "I recommended that you cut down" reads "I recommend that you cut down"
  * the ruler's "10 meaning you ready" reads "you are ready"
  * the dependent-alcohol feedback repeats a permission sentence that was
    already asked a turn earlier; the repeat is dropped.

Several entries below have no source text at all, because the study document
does not cover the path. Each is marked, and all of them are pending clinician
review.
"""

from __future__ import annotations

# The source writes "alcohol/drug use", meaning whichever arm is running. The
# two nouns differ because English does: "drug use" but "stop using drugs".
_USE_NOUN = {"alcohol": "alcohol", "drugs": "drug"}
_STOP_NOUN = {"alcohol": "alcohol", "drugs": "drugs"}


FIXED: dict[str, str] = {
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
    # Replaces the line above when the education was declined, since that one
    # refers back to a definition "we just discussed". The source line minus
    # that reference; nothing else changed. Pending clinician review.
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

    # Spoken at the end of a completed session. Its "few more questions about
    # your experiences" refers to the study's post-session survey, not to
    # anything this conversation will ask.
    "close": (
        "Thank you for participating in this process. "
        "Our staff will follow up with you about your experiences."
    ),
    # For a session that ends because a permission was declined. The close
    # above promises more questions, which contradicts having just told the
    # person it was their call. No source text exists for this path; this is a
    # minimal goodbye that respects the refusal. Pending clinician review.
    "close.declined": (
        "Thank you for your time today. Your provider can pick any of this "
        "up with you during your visit, whenever you're ready."
    ),
    # When nothing screened positive. No source text; a neutral affirmation
    # before the standard close. Pending clinician review.
    "prescreen.all_negative": (
        "Thank you for answering those questions. Based on your answers, your "
        "use is not likely to cause you any health problems."
    ),
    # For declining a permission mid-protocol, where the session continues. No
    # source text. Pending clinician review.
    "permission.declined": (
        "That's completely your call, and that's fine."
    ),
    # For stopping the whole conversation. No source text. Deliberately makes
    # no attempt to keep the person; answers already given still reach their
    # provider. Pending clinician review.
    "close.aborted": (
        "Of course — we can stop here, and that's completely fine. Thank "
        "you for your time today. Anything you shared stays confidential, "
        "and your provider can pick this up with you whenever you're ready."
    ),
    # For an item the person still cannot answer after being offered a recall
    # aid. The manual's guidance is to note uncertainty on the record rather
    # than press (SBIRT_REF.pdf p.18). No source text; pending clinician review.
    "item.skipped": (
        "That's okay — we can set that one aside. I'll make a note of it "
        "for your provider."
    ),
    "bi.leaves_you": "So where does this leave you?",
}


# Keyed by instrument and zone; every combination the protocol can reach must
# have an entry here.
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

# Zones whose feedback already ends by asking permission for the intervention.
# The flow uses this to avoid asking twice. The dependent-drug text is absent
# because the source does not end it with that question, so that route asks
# explicitly.
FEEDBACK_ASKS_BI: frozenset[tuple[str, str]] = frozenset(
    k for k, v in FEEDBACK.items()
    if v.rstrip().endswith("May I ask you some more questions about this?")
)


def feedback_text(instrument_key: str, zone: str) -> str:
    """The verbatim feedback for a completed screen.

    A KeyError means the protocol reached a zone with no authored text, which
    must surface as a bug — improvising feedback about somebody's screening
    result is exactly what this module exists to prevent.
    """
    return FEEDBACK[(instrument_key, zone)]


def bi_permission(arm: str) -> str:
    """Ask permission for the intervention, when the feedback did not already."""
    return f"May I ask you some more questions about your {_USE_NOUN[arm]} use?"


def bi_likes(arm: str) -> str:
    """Half of the decisional balance. Asked before the dislikes, deliberately:
    leading with what someone values about their use is what makes the exercise
    read as curiosity rather than as a set-up."""
    return f"What do you like about {_USE_NOUN[arm]} use?"


def bi_dislikes(arm: str) -> str:
    """The other half of the decisional balance."""
    return f"What do you dislike about {_USE_NOUN[arm]} use?"


def bi_recommend(arm: str) -> str:
    """The recommendation. Ends on readiness, which keeps the choice theirs."""
    return (f"Based on your answers I recommend that you cut down or stop "
            f"using {_STOP_NOUN[arm]}, but you have to be ready.")


def bi_ruler(arm: str) -> str:
    """The 0-10 readiness question, with both ends of the scale spelled out."""
    s = _STOP_NOUN[arm]
    return (f"Based on a scale from 0 to 10, with 0 meaning you are not at all "
            f"ready to cut down or stop using {s} and 10 meaning you are "
            f"ready right now to cut down or stop using {s}, where would you "
            f"say you are?")


def bi_why_not_lower(value: int) -> str:
    """Asks them to argue upward from their own number, which evokes change talk."""
    return f"Why are you a {value} and not a 1 or 2?"


def bi_why_not_higher(value: int) -> str:
    """The counterpart, which surfaces what is holding them back."""
    return f"Why are you a {value} and not a 9 or 10?"


from dataclasses import dataclass


@dataclass(frozen=True)
class Unit:
    """One thing the counselor delivers in a turn.

    A verbatim unit is spoken exactly as written and can be pre-rendered as a
    clip. A non-verbatim one instead lists the clinical content the turn must
    convey, and is worded for this particular person — typically around
    something they said. The model may rephrase those points; it may not drop,
    extend or contradict them.
    """

    id: str
    verbatim: bool
    literal: str = ""             # for verbatim units
    points: tuple[str, ...] = ()  # for non-verbatim ones


# The intervention's reflections and summaries, which have to be worded around
# what this person actually said. The braced slots are filled by the engine from
# captured state before the model ever sees them.
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
    """Every fixed utterance the protocol can speak, keyed as the runtime emits it.

    The one source for both clip pre-warming and the tests that assert nothing
    fixed is ever rendered mid-conversation. Parameterised lines are enumerated
    across their whole domain, so a fixed line is never a cache miss.
    """
    from .instruments import BY_KEY, PRE_SCREEN  # imported here to break a cycle

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
    # Every possible ruler value, so both follow-ups are cacheable clips rather
    # than a render at the moment they are needed.
    for v in range(11):
        out[f"bi.why_not_lower.{v}"] = bi_why_not_lower(v)
        out[f"bi.why_not_higher.{v}"] = bi_why_not_higher(v)
    return out
