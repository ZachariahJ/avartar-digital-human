"""Turning a selected field into the beats that voice it.

Pure with respect to the session: composing a line records nothing. The old
engine added a key to `covered` while it was still assembling the beat, so a
line lost to a barge-in counted as delivered — the ledger is written by whoever
confirms the audio actually landed, and never here.
"""

from __future__ import annotations

from . import templates
from .form import ARM_INSTRUMENT, Field
from .instruments import BY_KEY, PRE_SCREEN
from .runtime_types import LLMSay, ProtocolError, Say, Speak
from .select import assessment_for, missing_slots, unit_id
from .turn import target_item


def _greeting_text() -> str:
    # Imported here rather than at module scope: config imports modules.sbirt to
    # build its prompt, so anything in this package that imports config at load
    # time closes the cycle.
    import config

    return config.GREETING_PREAMBLE


def _verbatim(session, key: str) -> str:
    """The study text behind a keyed prompt.

    Item and pre-screen wording lives on the instrument rather than in FIXED, so
    those keys are recognised here instead of being duplicated into templates.
    """
    if key == "greeting":
        return _greeting_text()
    head, _, tail = key.rpartition(".")
    if head == "prescreen":
        # Only the three questions themselves live on the instrument; other
        # prescreen.* keys (the all-negative close) are ordinary fixed lines.
        for q in PRE_SCREEN:
            if q.key == tail:
                return q.item.text
    if tail == "preamble" and head in BY_KEY:
        return BY_KEY[head].preamble
    base, sep, index = key.partition(".item.")
    if sep and base in BY_KEY and index.isdigit():
        return BY_KEY[base].items[int(index)].text
    return templates.FIXED[key]


def resolve_say(session, field: Field, name: str) -> Say:
    """Expand an @-prefixed prompt into a keyed, cacheable line.

    The derived key is what reaches the clip cache and the audit trail, so it
    must match what templates.all_fixed_utterances() enumerates for pre-warm.
    """
    arm = field.arm
    if name == "@alcohol.screen.permission":
        # Chosen from the LEDGER: without the education actually having been
        # heard, the wording that refers back to "the standard drink definition
        # we just discussed" would be describing something nobody was told.
        key = ("alcohol.screen.permission"
               if "alcohol.edu.standard_drink" in session.spoken
               else "alcohol.screen.permission.no_defn")
        return Say(key, templates.FIXED[key])
    if name == "@feedback":
        ins_key = ARM_INSTRUMENT[arm]
        assessment = assessment_for(session, ins_key)
        zone = assessment.zone if assessment else ""
        return Say(f"feedback.{ins_key}.{zone}",
                   templates.feedback_text(ins_key, zone))
    if name == "@bi.permission":
        return Say(f"bi.permission.{arm}", templates.bi_permission(arm))
    if name == "@bi.likes":
        return Say(f"bi.likes.{arm}", templates.bi_likes(arm))
    if name == "@bi.dislikes":
        return Say(f"bi.dislikes.{arm}", templates.bi_dislikes(arm))
    if name == "@bi.recommend":
        return Say(f"bi.recommend.{arm}", templates.bi_recommend(arm))
    if name == "@bi.ruler":
        return Say(f"bi.ruler.{arm}", templates.bi_ruler(arm))
    if name == "@bi.why_not_lower":
        v = session.readiness[arm]
        return Say(f"bi.why_not_lower.{v}", templates.bi_why_not_lower(v))
    if name == "@bi.why_not_higher":
        v = session.readiness[arm]
        return Say(f"bi.why_not_higher.{v}", templates.bi_why_not_higher(v))
    raise ProtocolError(f"unknown resolver {name!r}")


def _unit_instruction(session, field: Field, unit: templates.Unit) -> str:
    """A briefing block, plus the answers it must be grounded in.

    Without the grounding the model is asked to summarise a conversation it can
    only see the tail of, and invents the half it cannot.
    """
    ground = []
    a = session.answers
    arm = field.arm
    if unit.id == "bi.summary.balance":
        ground = [f"They LIKE: {a.get('bi.likes', '(not captured)')}",
                  f"They DISLIKE: {a.get('bi.dislikes', '(not captured)')}"]
    elif unit.id == "bi.summary.rulers":
        ground = [f"Readiness {session.readiness.get(arm, '?')}/10.",
                  f"Why not lower: {a.get('bi.why_not_lower', '(not captured)')}",
                  f"Why not higher: {a.get('bi.why_not_higher', '(not captured)')}"]
    elif unit.id == "bi.reflect":
        ground = [f"They said: {a.get('bi.leaves_you', '(not captured)')}"]
    return " ".join(unit.points) + (("\n" + "\n".join(ground)) if ground else "")


def confirm_text(pending: dict) -> str:
    """The read-back sentence for a volunteered or uncertain answer.

    Quotes them back in their own words wherever possible: a person can only
    confirm what they recognise as something they actually said.
    """
    item = target_item(pending["target"])
    said = (f"Earlier you mentioned {pending['quote']}"
            if pending.get("quote") else "")
    if item is None:
        value = pending.get("text") or ""
        if said:
            return f"{said} — should I put that down as your answer?"
        return f"So that's {value} — did I get that right?"
    label = item.options[pending["code"]].label
    if said:
        return f"{said} — so that would be {label}. Is that right?"
    if pending.get("note"):
        return f"{pending['note']} — so that would be {label}. Did I get that right?"
    if pending.get("prior"):
        return (f"Earlier I heard that {pending['prior']}, so I want to make "
                f"sure I have this right: {label} — is that right?")
    return f"So that's {label} — did I get that right?"


def _ask_beats(session, field: Field) -> tuple:
    """The question itself, with no re-ask framing around it.

    Kept separate so the framing can quote the question without asking this
    module for the question through the framing again.
    """
    if field.kind == "confirm":
        return (Speak(confirm_text(session.pending_confirm)),)

    prompt = field.prompt
    if prompt.carried:
        # An earlier field's prompt already contained this question.
        return ()

    if field.slots:
        missing = missing_slots(session, field)
        if not missing:
            return ()
        point = dict(field.slot_points)[missing[0]]
        return (LLMSay("Ask the person, in one short natural question, "
                       f"{point}. Ask nothing else."),)

    if prompt.key:
        return (Say(prompt.key, _verbatim(session, prompt.key)),)

    if prompt.resolver:
        if prompt.resolver == "@close":
            from .form import close_unit

            unit = templates.POINTS_UNITS[close_unit(session)]
            return (LLMSay(_unit_instruction(session, field, unit)),)
        return (resolve_say(session, field, prompt.resolver),)

    if prompt.unit:
        unit = templates.POINTS_UNITS[prompt.unit]
        return (LLMSay(_unit_instruction(session, field, unit)),)

    if prompt.points:
        instruction = " ".join(prompt.points).format(
            drugs_kind=session.answers.get("drugs.kind", "the drugs they use"))
        return (LLMSay(instruction),)

    raise ProtocolError(f"field {field.id!r} has nothing to say")


def beats_for(session, field: Field) -> tuple:
    """Everything that voices one field, in order.

    A field is one delivery unit, so this is the whole of what the person hears
    for it — there is no partial credit and nothing here reaches the ledger.

    From the second attempt the study wording is preceded by a line naming what
    is still missing. The question itself is never reworded: it is validated
    text, and the engine speaks it verbatim every time.
    """
    beats = _ask_beats(session, field)
    if not beats:
        # A carried prompt, or a slot ask with nothing left to collect. Silence
        # must stay silent; framing it would invent an utterance.
        return beats
    framing = reask_instruction(session, field)
    return ((LLMSay(framing),) + beats) if framing else beats


def ask_text(session, field: Field | None) -> str:
    """The pending question as plain text, for the NLU's ask_text argument."""
    if field is None:
        return ""
    for beat in reversed(_ask_beats(session, field)):
        if isinstance(beat, (Say, Speak)):
            return beat.text
    return ""


REASKS = (
    # Attempt 2: name the gap. Attempt 3 and beyond: name it and make leaving it
    # easy, because a third identical failure is usually the question, not them.
    "They have just been asked {subject} and the answer did not come through. "
    "In ONE short sentence, say plainly which part you still need — the "
    "specific choices, or the unit you are after. Do not apologise and do not "
    "blame them.",
    "They have now been asked {subject} more than once without it landing. In "
    "ONE or two short sentences, name concretely what would count as an "
    "answer, and say that skipping it is fine if they would rather move on. Do "
    "not apologise and do not blame them.",
)

_REASK_TAIL = (
    " Stay on the topic and introduce no other topic. Do NOT ask the question "
    "or reword it; it is put to them for you straight afterwards."
)


def _subject(session, field: Field) -> str:
    """What the pending ask is about, in a form a briefing can name.

    Study wording is quoted so the model works from the exact question. A slot
    ask has no verbatim text — it is itself composed by the model — so the slot
    point stands in for it.
    """
    text = ask_text(session, field)
    if text:
        return f'this question: "{text}"'
    if field.slots:
        missing = missing_slots(session, field)
        if missing:
            return f"this: {dict(field.slot_points)[missing[0]]}"
    return ""


def reask_instruction(session, field: Field) -> str:
    """The line that precedes a repeat of the ask.

    Empty on the first attempt: the validated question stands on its own, and
    framing it before it has failed once only delays it.
    """
    if session.reasks.get(unit_id(session, field), 0) < 1:
        return ""
    subject = _subject(session, field)
    if not subject:
        return ""
    n = session.reasks[unit_id(session, field)]
    return REASKS[min(n, len(REASKS)) - 1].format(subject=subject) + _REASK_TAIL


PROBES = {
    "probe.option": (
        "The person cannot answer this question: \"{question}\". In one or "
        "two gentle sentences, help them estimate: suggest thinking about the "
        "period in the past year when it was most true of them, and mention "
        "that a rough guess is fine — or we can skip it and move on. Stay on "
        "the topic of that question and introduce no other topic. Do NOT "
        "re-ask the question; it is repeated for you straight afterwards."),
    "probe.number": (
        "The person cannot answer this question: \"{question}\". In one "
        "gentle sentence, say it doesn't have to be exact and that whatever "
        "number feels closest is fine — or offer to skip it. Stay on the topic "
        "of that question and introduce no other topic. Do NOT re-ask the "
        "question; it is repeated for you straight afterwards."),
}


def probe_instruction(session, field: Field) -> str:
    """The estimate-help briefing, anchored to the question actually pending.

    Without the question text the briefing was topic-blind, so a probe on a
    tobacco item drifted into asking about drinks.
    """
    instruction = PROBES.get(field.probe)
    if instruction is None:
        return ""
    return instruction.format(question=ask_text(session, field))
