"""The protocol as a form of guarded fields, not a list of instructions.

Every field declares WHEN it is in scope. Nothing declares where to go next, so
there is no program counter that can run ahead of the audio: position is a pure
function of the answers collected so far plus the ledger of what was actually
voiced. A prompt lost to a barge-in is simply selected again.

Two mechanics carry most of the weight:

  * `id` is not `slot`. Two fields may write the same slot, which is how a
    mutually exclusive branch becomes two guarded fields instead of a label and
    a jump: whichever one is in scope asks, and answering it fills the slot that
    takes both out of scope.
  * `variant` extends a field's ledger identity, so one field can owe several
    distinct utterances over its life — one per missing slot, one per risk zone.

Guards are pure. The old Route functions mutated the session (one wrote the arm
queue, the next popped it), which is exactly why a re-entered walk could not
find its way back into the arm it was already inside.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from .instruments import BY_KEY, PRE_SCREEN
from .templates import FEEDBACK_ASKS_BI

ARM_INSTRUMENT = {"alcohol": "audit", "drugs": "dast_10"}
ARM_ORDER = ("alcohol", "drugs")

Guard = Callable[["object"], bool]


@dataclass(frozen=True)
class Prompt:
    """How a field is voiced.

    `key` names a study-verbatim line in templates.FIXED. Only keyed lines are
    clip-cacheable and pre-warmable, so the key must match what the pre-warm
    catalogue emits or every delivery pays a live render.

    `resolver` names a line whose text depends on session state. `unit` names a
    briefing block in templates.POINTS_UNITS; `points` carries literal
    instruction text. Both are worded by the model and cost a live render.

    `carried` means an EARLIER field's prompt already contained this question,
    so this field listens without speaking.
    """

    key: str = ""
    resolver: str = ""
    unit: str = ""
    points: tuple[str, ...] = ()
    carried: bool = False


CARRIED = Prompt(carried=True)


def _always(session) -> bool:
    return True


@dataclass(frozen=True)
class Field:
    """One answerable thing, or one thing to say.

    `kind` decides both what a legal answer looks like and how the field is
    filled: "tell" and "end" are filled by having been voiced, everything else
    by a slot write. "confirm" is never filled here — a read-back is cleared by
    whoever resolves it.
    """

    id: str
    kind: str                       # consent | option | number | open | tell | end | confirm
    slot: str = ""
    when: Guard = _always
    prompt: Prompt = Prompt()
    arm: str = ""
    instrument: str = ""
    item_index: int | None = None
    slots: tuple[str, ...] = ()
    slot_points: tuple[tuple[str, str], ...] = ()
    on_decline: str = ""            # "" | "arm" | "session"
    probe: str = ""
    variant: Callable[["object"], str] | None = None


@dataclass(frozen=True)
class Form:
    """A group of fields sharing a scope test.

    `terminal` means the interview is over once this form is satisfied. Without
    it a closed session falls through to the forms below and starts re-asking
    questions it has already been told to stop asking — a crisis close would be
    followed by the next pre-screen item.
    """

    id: str
    when: Guard
    fields: tuple[Field, ...]
    terminal: bool = False


# --- guard vocabulary -------------------------------------------------------
#
# Deliberately small and composable: a guard that cannot be expressed here is a
# guard the field table cannot be read at a glance, and this table is the whole
# clinical route.


def granted(slot: str) -> Guard:
    return lambda s: s.answers.get(slot) == "yes"


def declined_slot(slot: str) -> Guard:
    return lambda s: s.answers.get(slot) == "no"


def prescreen_positive(arm: str) -> Guard:
    return lambda s: s.prescreen.get(arm, 0) > 0


def spoken_variant(field_id: str) -> Guard:
    """Whether a field has been voiced, under any of its variants.

    A variant extends the ledger key, so a field whose wording depends on state
    cannot be matched by its id alone.
    """
    prefix = field_id + ":"
    return lambda s: any(u == field_id or u.startswith(prefix) for u in s.spoken)


def all_(*guards: Guard) -> Guard:
    return lambda s: all(g(s) for g in guards)


def not_(guard: Guard) -> Guard:
    return lambda s: not guard(s)


def prescreen_done(session) -> bool:
    return all(q.key in session.prescreen for q in PRE_SCREEN)


def no_positive_arm(session) -> bool:
    return not any(session.prescreen.get(a, 0) > 0 for a in ARM_ORDER)


# --- derived clinical state -------------------------------------------------
#
# These read through to instruments rather than to a cached result. The old
# engine computed the zone inside the RunItems branch at the instant the last
# item filled and stored it, which made the zone — like the program counter — a
# side effect of having passed a particular instruction rather than a function
# of the answers.


def _responses(session, ins_key: str) -> dict:
    return session.responses.get(ins_key, {})


def _missing(session, ins_key: str) -> dict:
    return session.missing.get(ins_key, {})


def item_is_next(ins_key: str, index: int) -> Guard:
    """The whole sub-iteration, as a guard.

    next_item_index() already owns the order, the WHO skip rules and the items
    marked unanswerable. Restating any of that in the field table is exactly the
    drift a validated instrument cannot afford, so this defers to it and at most
    one item field of an instrument is ever in scope.
    """
    from .instruments import next_item_index

    def guard(session) -> bool:
        return next_item_index(BY_KEY[ins_key], _responses(session, ins_key),
                               _missing(session, ins_key)) == index

    return guard


def instrument_started(ins_key: str) -> Guard:
    return lambda s: bool(_responses(s, ins_key)) or bool(_missing(s, ins_key))


def instrument_complete(ins_key: str) -> Guard:
    from .instruments import next_item_index

    def guard(session) -> bool:
        responses = _responses(session, ins_key)
        if not responses and not _missing(session, ins_key):
            return False
        return next_item_index(BY_KEY[ins_key], responses,
                               _missing(session, ins_key)) is None

    return guard


def zone_of(arm: str):
    """The risk zone for an arm, or "" while it cannot yet be known."""
    from .instruments import assess, next_item_index

    ins_key = ARM_INSTRUMENT[arm]

    def read(session) -> str:
        responses = _responses(session, ins_key)
        missing = _missing(session, ins_key)
        if not responses and not missing:
            return ""
        if next_item_index(BY_KEY[ins_key], responses, missing) is not None:
            return ""
        return assess(BY_KEY[ins_key], responses, missing).zone or ""

    return read


def zone_needs_bi(arm: str) -> Guard:
    read = zone_of(arm)
    return lambda s: read(s) not in ("", "healthy")


def zone_asks_bi(arm: str) -> Guard:
    """Whether the feedback line already ended by asking the BI permission.

    The risky/harmful/dependent texts finish with "May I ask you some more
    questions about this?", so a separate permission ask would double-ask.
    """
    read = zone_of(arm)
    ins_key = ARM_INSTRUMENT[arm]
    return lambda s: (ins_key, read(s)) in FEEDBACK_ASKS_BI


def hard_decline_in(arm: str) -> Guard:
    """A refusal that cut this arm short.

    Declining the optional education does not count: the protocol carried on
    normally afterwards, so nothing was cut short. Same rule as close_unit().
    """
    keys = (f"{arm}.screen.permission", f"{arm}.feedback.permission",
            f"bi.permission.{arm}")
    return lambda s: any(s.answers.get(k) == "no" for k in keys)


def arm_finished(arm: str) -> Guard:
    """Whether an arm has nothing further to contribute.

    Used only by the closing form; the arm's own fields guard themselves.
    """
    ins_key = ARM_INSTRUMENT[arm]

    def guard(session) -> bool:
        if session.prescreen.get(arm, 0) == 0:
            return True
        if hard_decline_in(arm)(session):
            return True
        if not instrument_complete(ins_key)(session):
            return False
        if zone_needs_bi(arm)(session):
            return (granted(f"bi.permission.{arm}")(session)
                    and "bi.leaves_you" in session.answers)
        return True

    return guard


def close_unit(session) -> str:
    """Which closing line the session has earned.

    The standard close promises follow-up questions, which reads badly straight
    after somebody declined a permission and was told that was their call.
    """
    hard = any(hard_decline_in(a)(session) for a in ARM_ORDER)
    return "close.declined" if hard else "close"


def first_missing_slot(slot: str, names: tuple[str, ...]):
    """Variant for a multi-slot ask: the slot being collected right now.

    Without this the whole ask enters the ledger after its first sub-question
    and the remaining slots are never voiced.
    """

    def read(session) -> str:
        filled = session.slots.get(slot, {})
        for name in names:
            if name not in filled:
                return name
        return "done"

    return read


# --- the form table ---------------------------------------------------------


def _prescreen_fields() -> tuple[Field, ...]:
    return tuple(
        Field(f"prescreen.{q.key}", kind="option", slot=f"prescreen.{q.key}",
              instrument="prescreen", item_index=i,
              prompt=Prompt(key=f"prescreen.{q.key}"),
              probe="probe.option")
        for i, q in enumerate(PRE_SCREEN))


def _instrument_fields(arm: str, ins_key: str) -> tuple[Field, ...]:
    gate = granted(f"{arm}.screen.permission")
    fields = []
    if BY_KEY[ins_key].preamble:
        # Emitted inline by the old RunItems branch, so it could not be replayed
        # when it was lost. As a field it is ledgered and re-deliverable.
        fields.append(Field(
            f"{ins_key}.preamble", kind="tell", arm=arm,
            when=all_(gate, not_(instrument_started(ins_key))),
            prompt=Prompt(key=f"{ins_key}.preamble")))
    for i in range(len(BY_KEY[ins_key].items)):
        fields.append(Field(
            f"{ins_key}.{i}", kind="option", slot=f"{ins_key}.{i}",
            instrument=ins_key, item_index=i, arm=arm,
            when=all_(gate, item_is_next(ins_key, i)),
            prompt=Prompt(key=f"{ins_key}.item.{i}"),
            probe="probe.option"))
    return tuple(fields)


def _bi_fields(arm: str) -> tuple[Field, ...]:
    """The brief intervention for one arm.

    The ask slots are deliberately NOT per-arm. A person who screens positive on
    both arms is asked what they like about it once, not twice, and the second
    arm's copies fall out of scope because the shared slot is already filled —
    which is what the old engine did by skipping an Ask whose key was already in
    session.answers.
    """
    permitted = granted(f"bi.permission.{arm}")
    feedback_unit = f"{arm}.feedback"

    def ruler_recorded(session) -> bool:
        return session.readiness.get(arm) is not None

    return (
        # Two fields, one slot. The carried one listens for a permission the
        # feedback line already asked for; the asked one poses it itself. They
        # cannot both be in scope, and answering either takes both out.
        Field(f"bi.permission.{arm}.carried", kind="consent",
              slot=f"bi.permission.{arm}", arm=arm, prompt=CARRIED,
              on_decline="arm",
              when=all_(granted(f"{arm}.feedback.permission"),
                        zone_asks_bi(arm), spoken_variant(feedback_unit))),
        Field(f"bi.permission.{arm}.asked", kind="consent",
              slot=f"bi.permission.{arm}", arm=arm, on_decline="arm",
              prompt=Prompt(resolver="@bi.permission"),
              when=all_(granted(f"{arm}.feedback.permission"),
                        zone_needs_bi(arm), not_(zone_asks_bi(arm)))),

        Field(f"{arm}.bi.likes", kind="open", slot="bi.likes", arm=arm,
              when=permitted, prompt=Prompt(resolver="@bi.likes")),
        Field(f"{arm}.bi.dislikes", kind="open", slot="bi.dislikes", arm=arm,
              when=permitted, prompt=Prompt(resolver="@bi.dislikes")),
        Field(f"{arm}.bi.summary.balance", kind="tell", arm=arm,
              when=permitted, prompt=Prompt(unit="bi.summary.balance")),
        Field(f"{arm}.bi.recommend", kind="tell", arm=arm,
              when=permitted, prompt=Prompt(resolver="@bi.recommend")),

        # Tell("@bi.ruler") + Ask(ask="included") were one question split in two,
        # because the old interpreter returned from its Ask branch before it
        # could append a beat. One field, one ledger entry.
        Field(f"{arm}.bi.ruler", kind="number", slot="bi.ruler", arm=arm,
              when=permitted, prompt=Prompt(resolver="@bi.ruler"),
              probe="probe.number"),

        Field(f"{arm}.bi.why_not_lower", kind="open", slot="bi.why_not_lower",
              arm=arm, when=all_(permitted, ruler_recorded),
              prompt=Prompt(resolver="@bi.why_not_lower")),
        Field(f"{arm}.bi.why_not_higher", kind="open", slot="bi.why_not_higher",
              arm=arm, when=all_(permitted, ruler_recorded),
              prompt=Prompt(resolver="@bi.why_not_higher")),
        Field(f"{arm}.bi.summary.rulers", kind="tell", arm=arm,
              when=all_(permitted, ruler_recorded),
              prompt=Prompt(unit="bi.summary.rulers")),

        Field(f"{arm}.bi.leaves_you", kind="open", slot="bi.leaves_you",
              arm=arm, when=permitted, prompt=Prompt(key="bi.leaves_you")),
        Field(f"{arm}.bi.reflect", kind="tell", arm=arm, when=permitted,
              prompt=Prompt(unit="bi.reflect")),
    )


_QF_SLOTS = ("drink", "amount", "frequency")

# The study asks what, how much and how often as one stacked question. Split
# into slots it is asked one part at a time, and somebody who volunteers all
# three at once is never asked again.
ALCOHOL_QF = Field(
    "alcohol.qf", kind="open", slot="alcohol.qf", arm="alcohol",
    slots=_QF_SLOTS,
    slot_points=(("drink", "what they like to drink"),
                 ("amount", "how much they usually drink"),
                 ("frequency", "how often they usually drink")),
    variant=first_missing_slot("alcohol.qf", _QF_SLOTS))


def _alcohol_fields() -> tuple[Field, ...]:
    return (
        ALCOHOL_QF,
        Field("alcohol.edu.permission", kind="consent",
              slot="alcohol.edu.permission", arm="alcohol",
              prompt=Prompt(key="alcohol.edu.permission")),
        Field("alcohol.edu.standard_drink", kind="tell", arm="alcohol",
              when=granted("alcohol.edu.permission"),
              prompt=Prompt(key="alcohol.edu.standard_drink")),
        Field("alcohol.edu.limits", kind="tell", arm="alcohol",
              when=granted("alcohol.edu.permission"),
              prompt=Prompt(key="alcohol.edu.limits")),
        # The wording variant is chosen from the LEDGER, so an education line
        # lost to a barge-in can no longer be claimed as "the standard drink
        # definition we just discussed".
        Field("alcohol.screen.permission", kind="consent",
              slot="alcohol.screen.permission", arm="alcohol",
              on_decline="arm",
              prompt=Prompt(resolver="@alcohol.screen.permission")),
    ) + _instrument_fields("alcohol", "audit") + (
        Field("alcohol.feedback.permission", kind="consent",
              slot="alcohol.feedback.permission", arm="alcohol",
              on_decline="arm",
              when=all_(granted("alcohol.screen.permission"),
                        instrument_complete("audit")),
              prompt=Prompt(key="alcohol.feedback.permission")),
        Field("alcohol.feedback", kind="tell", arm="alcohol",
              when=granted("alcohol.feedback.permission"),
              prompt=Prompt(resolver="@feedback"),
              variant=zone_of("alcohol")),
    ) + _bi_fields("alcohol") + (
        Field("alcohol.declined", kind="tell", arm="alcohol",
              when=hard_decline_in("alcohol"),
              prompt=Prompt(key="permission.declined")),
    )


def _drugs_fields() -> tuple[Field, ...]:
    return (
        Field("drugs.kind", kind="open", slot="drugs.kind", arm="drugs",
              prompt=Prompt(key="drugs.kind")),
        Field("drugs.qf", kind="open", slot="drugs.qf", arm="drugs",
              prompt=Prompt(points=(
                  "Ask how much and how often the person uses the drug or "
                  "drugs they just named ({drugs_kind}). One natural sentence; "
                  "ask nothing else.",))),
        Field("drugs.screen.permission", kind="consent",
              slot="drugs.screen.permission", arm="drugs", on_decline="arm",
              prompt=Prompt(key="drugs.screen.permission")),
    ) + _instrument_fields("drugs", "dast_10") + (
        Field("drugs.feedback.permission", kind="consent",
              slot="drugs.feedback.permission", arm="drugs",
              on_decline="arm",
              when=all_(granted("drugs.screen.permission"),
                        instrument_complete("dast_10")),
              prompt=Prompt(key="drugs.feedback.permission")),
        Field("drugs.feedback", kind="tell", arm="drugs",
              when=granted("drugs.feedback.permission"),
              prompt=Prompt(resolver="@feedback"),
              variant=zone_of("drugs")),
    ) + _bi_fields("drugs") + (
        Field("drugs.declined", kind="tell", arm="drugs",
              when=hard_decline_in("drugs"),
              prompt=Prompt(key="permission.declined")),
    )


# The greeting is a real field rather than something the pipeline hand-assembles
# before the machine starts, so it is ledgered, replayable and pre-warmable like
# every other line. Splitting it from the consent question is deliberate: the
# two together run about 25 seconds, and a lost preamble must not drag the
# consent question through a second reading.
CONSENT = Form("consent", when=_always, fields=(
    Field("greeting", kind="tell", prompt=Prompt(key="greeting")),
    Field("consent.opening", kind="consent", slot="consent",
          on_decline="session", prompt=Prompt(key="consent.opening")),
))

CONSENT_DECLINED = Form("consent.declined", when=declined_slot("consent"),
                        terminal=True, fields=(
    Field("close.consent_declined", kind="end",
          prompt=Prompt(key="close.consent_declined")),
))

PRESCREEN = Form("prescreen", when=granted("consent"),
                 fields=_prescreen_fields())

ALCOHOL = Form("alcohol", when=all_(granted("consent"), prescreen_done,
                                    prescreen_positive("alcohol")),
               fields=_alcohol_fields())

DRUGS = Form("drugs", when=all_(granted("consent"), prescreen_done,
                                prescreen_positive("drugs")),
             fields=_drugs_fields())

CLOSING = Form("closing", when=all_(granted("consent"), prescreen_done,
                                    *[arm_finished(a) for a in ARM_ORDER]),
               terminal=True, fields=(
    Field("prescreen.all_negative", kind="tell",
          when=no_positive_arm, prompt=Prompt(key="prescreen.all_negative")),
    Field("close", kind="end", prompt=Prompt(resolver="@close"),
          variant=lambda s: close_unit(s)),
))

FORMS: tuple[Form, ...] = (CONSENT, CONSENT_DECLINED, PRESCREEN,
                           ALCOHOL, DRUGS, CLOSING)

# Selected ahead of document order. Unlike the old parked program counter these
# are re-selected every tick until they have actually been heard, so the crisis
# line can no longer be swallowed by the very barge-in that flagged the crisis.
INTERRUPTS: tuple[Form, ...] = (
    Form("crisis", when=lambda s: s.crisis, terminal=True, fields=(
        Field("close.crisis", kind="end",
              prompt=Prompt(key="close.crisis")),)),
    Form("aborted", when=lambda s: s.aborted, terminal=True, fields=(
        Field("close.aborted", kind="end",
              prompt=Prompt(key="close.aborted")),)),
    Form("pause", when=lambda s: s.pause_pending, fields=(
        Field("pause.offer", kind="consent", slot="pause.offer",
              prompt=Prompt(key="aside.offer_pause")),)),
)

ALL_FIELDS: tuple[Field, ...] = tuple(
    f for form in INTERRUPTS + FORMS for f in form.fields)

OPEN_TARGETS = frozenset(f.slot for f in ALL_FIELDS
                         if f.kind == "open" and f.slot)
