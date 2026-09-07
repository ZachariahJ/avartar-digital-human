"""The screening instruments themselves: questions, options, scores and zones.

Each instrument is plain data, so the clinical content can be reviewed and
changed here without touching a prompt string or any conversational code.

Two tiers, which look similar but are not:

  * Administered instruments carry structured items — the official interview
    wording, an ordered option list, per-option scores and skip rules. The
    scoring below, the state machine, the answer coding and the clip pre-render
    all read the same rows, so there is no second copy to fall out of step.
  * Reference instruments carry plain strings. They exist only to be rendered
    into the prompt as background and are never administered or scored.

Wording is taken from the study's authoritative sources and should not be
adjusted for readability — these are validated instruments, and paraphrasing an
item invalidates its score:

  * AUDIT items, options and skip rules: WHO AUDIT manual, Box 4 interview
    version.
  * DAST-10 as worded for this study: its case cards. Note the item 3
    deviation recorded at that item.
  * Pre-screen and the four feedback zones: the study's app dialogue document.

Further references: Skinner DAST-10; Ewing CAGE and Brown CAGE-AID; NIDA Quick
Screen and NM ASSIST; TAPS; Knight CRAFFT 2.1.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Collection, Mapping


@dataclass(frozen=True)
class Option:
    """One answer choice: its exact wording and the points it contributes.

    Aliases are matched as whole answers, lowercased and exactly — never
    fuzzily — so that a match can code an answer with no model involved. Which
    phrasings count as equivalent to an option is a clinical decision, so they
    ship empty and semantic matching is left to the coder that refuses to guess.
    Pending clinician review.
    """

    label: str
    score: int
    aliases: tuple[str, ...] = ()


@dataclass(frozen=True)
class Item:
    """One administered question, as officially worded, with its answer options.

    An answer is recorded as the option's index — its code — and the points it
    contributes are that option's score. The two are usually the same number,
    but not always: some items score 0, 2 and 4 for codes 0, 1 and 2. Confusing
    them silently mis-scores an instrument, so code and score are kept distinct
    everywhere they appear.

    verbatim means the question is spoken exactly as written. The engine may add
    an acknowledgment around it but never rephrase it, because these wordings
    are what the instrument was validated with. Conversational questions that
    are not instrument items belong in flow.py rather than here.

    confirm marks a score-bearing item whose answer may need reading back before
    it is committed. It is a permission for a read-back, not a demand for one:
    what actually triggers it is a conversion assumption, a near-boundary value
    or a contradiction with an earlier answer. A refusal re-collects the item.

    coding says how a semantic answer becomes a code. "choice" items take the
    coded option directly; the frequency and quantity scales instead require the
    raw numbers to be extracted, and compute the option themselves — see
    coding.py for why that distinction matters.
    """

    text: str
    options: tuple[Option, ...]
    note: str = ""  # a skip rule or scoring deviation, rendered with the item
    verbatim: bool = True
    confirm: bool = False
    coding: str = "choice"   # "choice" | "freq_q1" | "freq5" | "quantity_drinks"

    @property
    def kind(self) -> str:
        """"yesno" or "scale", for the asking layer.

        Derived from the options rather than declared, so it cannot disagree
        with them.
        """
        labels = tuple(o.label.lower() for o in self.options)
        return "yesno" if labels == ("no", "yes") else "scale"


@dataclass(frozen=True)
class SkipRule:
    """When `when` holds, the listed items are never asked and contribute zero.

    The predicate receives contributed scores, never raw option codes. On items
    where the two differ, a predicate written against codes would be quietly
    wrong, and this removes the possibility of writing one.

    Adding a skip rule is adding a row here; the engine needs no change.
    """

    skip: tuple[int, ...]
    when: Callable[[Mapping[int, int]], bool]
    note: str = ""


@dataclass(frozen=True)
class RiskBand:
    """A range of scores, what it is called, and what the protocol does about it.

    `zone` is the stable key the feedback templates and tests key on, which is
    why it is separate from the human-readable label: the label can be reworded,
    the zone cannot. Empty for reference instruments, which are never scored.
    """

    label: str
    low: int          # inclusive
    high: int         # inclusive
    action: str
    zone: str = ""

    def contains(self, score: int) -> bool:
        """Whether a raw score falls in this band."""
        return self.low <= score <= self.high


@dataclass(frozen=True)
class Instrument:
    """One screening tool.

    `items` holds Item objects for instruments this system administers, and
    plain strings for the reference-only ones — which is what distinguishes the
    two tiers described in the module docstring.
    """

    key: str
    name: str
    domain: str            # alcohol | drugs | tobacco | combined | adolescent
    when_to_use: str
    items: tuple[Item | str, ...]
    response_scale: str
    scoring: str
    bands: tuple[RiskBand, ...]
    preamble: str = ""     # spoken once before the first item
    skip_rules: tuple[SkipRule, ...] = ()

    def render(self) -> str:
        """This instrument as prompt text, items and bands included."""
        lines = [f"### {self.name}  ({self.domain}, {len(self.items)} items)",
                 f"When to use: {self.when_to_use}",
                 f"Response scale: {self.response_scale}",
                 "Items:"]
        for i, item in enumerate(self.items, 1):
            if isinstance(item, str):
                lines.append(f"  {i}. {item}")
            else:
                opts = " / ".join(f"{o.score}={o.label}" for o in item.options)
                line = f"  {i}. {item.text}  [{opts}]"
                if item.note:
                    line += f"  ({item.note})"
                lines.append(line)
        lines.append(f"Scoring: {self.scoring}")
        lines.append("Risk bands → action:")
        lines += [f"  • {b.low}–{b.high}: {b.label} → {b.action}" for b in self.bands]
        return "\n".join(lines)


def risk_band_for(instrument: Instrument, score: int) -> RiskBand | None:
    """The band a raw score falls in, or None if no band covers it."""
    for band in instrument.bands:
        if band.contains(score):
            return band
    return None


# Shared answer scales, worded exactly as the WHO AUDIT interview form has them.
_FREQ_5 = (
    Option("Never", 0),
    Option("Less than monthly", 1),
    Option("Monthly", 2),
    Option("Weekly", 3),
    Option("Daily or almost daily", 4),
)
_YES_NO_TIMEFRAMED = (
    Option("No", 0),
    Option("Yes, but not in the last year", 2),
    Option("Yes, during the last year", 4),
)
_NO_YES = (Option("No", 0), Option("Yes", 1))


# The study's own three opening questions, from its dialogue document and case
# cards. Not the NIDA Quick Screen further down, which is reference only and is
# never administered. Any score above zero opens that domain's full instrument.

@dataclass(frozen=True)
class PreScreenQuestion:
    """One opening question, and which arm a positive answer opens."""

    key: str      # "tobacco" | "alcohol" | "drugs"
    item: Item


PRE_SCREEN: tuple[PreScreenQuestion, ...] = (
    PreScreenQuestion("tobacco", Item(
        "Do you smoke cigarettes or use other tobacco products?",
        _NO_YES,
    )),
    PreScreenQuestion("alcohol", Item(
        "When was the last time you had more than 4 drinks in one day?",
        (Option("Never or more than a year ago", 0),
         Option("Within the last year", 1)),
    )),
    PreScreenQuestion("drugs", Item(
        "How many times in the past year have you used an illegal drug or "
        "used a prescription medication for nonmedical reasons?",
        (Option("None", 0), Option("One or more", 1)),
    )),
)


NIDA_QUICK_SCREEN = Instrument(
    key="nida_quick",
    name="NIDA Quick Screen (pre-screen)",
    domain="combined",
    when_to_use="Reference only — this app administers the study's own 3-question pre-screen.",
    items=(
        "In the PAST YEAR, how often have you had 5+ (men) / 4+ (women) drinks in a day?",
        "In the past year, how often have you used tobacco products?",
        "In the past year, how often have you used prescription drugs for non-medical reasons?",
        "In the past year, how often have you used an illegal drug?",
    ),
    response_scale="Never / Once or twice / Monthly / Weekly / Daily-or-almost-daily",
    scoring="Any answer above 'Never' is a positive pre-screen for that substance class.",
    bands=(
        RiskBand("No use reported", 0, 0, "Affirm, brief education, close screen"),
        RiskBand("Any use reported", 1, 1, "Open the matching full instrument below"),
    ),
)

AUDIT_C = Instrument(
    key="audit_c",
    name="AUDIT-C (alcohol, brief)",
    domain="alcohol",
    when_to_use="Positive alcohol pre-screen; fast alcohol risk triage.",
    items=(
        "How often did you have a drink containing alcohol in the past year? (0–4)",
        "How many standard drinks on a typical drinking day? (0–4)",
        "How often did you have 6+ drinks on one occasion? (0–4)",
    ),
    response_scale="Each item 0–4",
    scoring="Sum items (0–12).",
    bands=(
        RiskBand("Low risk", 0, 2, "Positive feedback, done"),
        RiskBand("Positive screen (≥3 women / ≥4 men)", 3, 7, "Give full AUDIT or brief intervention"),
        RiskBand("Strong positive", 8, 12, "Brief intervention + consider referral"),
    ),
)

# The official WHO interview version. Its skip rules are data further down: a
# "Never" to item 1, or nothing at all on items 2 and 3, jumps to items 9-10.
AUDIT = Instrument(
    key="audit",
    name="AUDIT (alcohol, full 10-item)",
    domain="alcohol",
    when_to_use="Positive alcohol pre-screen (>4 drinks in a day within the past year).",
    preamble=(
        "Now I am going to ask you some questions about your use of alcoholic "
        "beverages during this past year."
    ),
    # confirm marks the items where a mis-coded answer does the most damage:
    # the quantity and frequency triad, and the dependence-domain items. These
    # carry the most score weight and are the most exposed to a semantic answer
    # being read the wrong way. Which items qualify is a clinical decision —
    # pending clinician review.
    items=(
        Item("How often do you have a drink containing alcohol?",
             (Option("Never", 0),
              Option("Monthly or less", 1),
              Option("2 to 4 times a month", 2),
              Option("2 to 3 times a week", 3),
              Option("4 or more times a week", 4)),
             note="0 (Never) → skip to items 9–10", confirm=True,
             coding="freq_q1"),
        Item("How many drinks containing alcohol do you have on a typical day "
             "when you are drinking?",
             (Option("1 or 2", 0),
              Option("3 or 4", 1),
              Option("5 or 6", 2),
              Option("7, 8, or 9", 3),
              Option("10 or more", 4)), confirm=True,
             coding="quantity_drinks"),
        Item("How often do you have six or more drinks on one occasion?",
             _FREQ_5,
             note="if items 2+3 total 0 → skip to items 9–10", confirm=True,
             coding="freq5"),
        Item("How often during the last year have you found that you were not "
             "able to stop drinking once you had started?", _FREQ_5,
             confirm=True, coding="freq5"),
        Item("How often during the last year have you failed to do what was "
             "normally expected from you because of drinking?", _FREQ_5,
             confirm=True, coding="freq5"),
        Item("How often during the last year have you needed a first drink in "
             "the morning to get yourself going after a heavy drinking session?",
             _FREQ_5, confirm=True, coding="freq5"),
        Item("How often during the last year have you had a feeling of guilt "
             "or remorse after drinking?", _FREQ_5, coding="freq5"),
        Item("How often during the last year have you been unable to remember "
             "what happened the night before because you had been drinking?",
             _FREQ_5, coding="freq5"),
        Item("Have you or someone else been injured as a result of your "
             "drinking?", _YES_NO_TIMEFRAMED),
        Item("Has a relative or friend or a doctor or another health worker "
             "been concerned about your drinking or suggested you cut down?",
             _YES_NO_TIMEFRAMED),
    ),
    response_scale="Items 1–8: 0–4; items 9–10: 0/2/4",
    scoring="Sum all items (0–40). Zones per WHO manual.",
    # The official form's skip rules, as data. Predicates read scores, not codes.
    skip_rules=(
        SkipRule(skip=tuple(range(1, 8)),
                 when=lambda s: s.get(0) == 0,
                 note="item 1 'Never' → skip items 2–8, go to items 9–10"),
        SkipRule(skip=tuple(range(3, 8)),
                 when=lambda s: 1 in s and 2 in s and s[1] + s[2] == 0,
                 note="items 2+3 total 0 → skip items 4–8, go to items 9–10"),
    ),
    bands=(
        RiskBand("Zone I – Low risk (Healthy)", 0, 7,
                 "Alcohol education / affirmation", zone="healthy"),
        RiskBand("Zone II – Risky/hazardous", 8, 15,
                 "Brief intervention (simple advice)", zone="risky"),
        RiskBand("Zone III – Harmful", 16, 19,
                 "Brief intervention + brief counseling + monitor", zone="harmful"),
        RiskBand("Zone IV – Likely dependence", 20, 40,
                 "Refer to specialist assessment/treatment", zone="dependent"),
    ),
)

# Worded as this study's case cards have it, which differs from the standard
# instrument at item 3. Skinner asks whether the person is always able to stop
# and reverse-scores it, so "No" earns the point; this study asks whether they
# are unable to stop, so "Yes" does. The two are equivalent, but the scoring
# must follow the wording actually spoken — swapping one without the other
# inverts the item.
DAST_10 = Instrument(
    key="dast_10",
    name="DAST-10 (drug use, non-alcohol)",
    domain="drugs",
    when_to_use="Positive drug pre-screen (illicit or non-medical prescription use).",
    preamble="These questions refer to the past 12 months.",
    items=(
        Item("Have you used drugs other than those required for medical "
             "reasons?", _NO_YES),
        Item("Do you abuse more than one drug at a time?", _NO_YES),
        Item("Are you unable to stop using drugs when you want to?", _NO_YES,
             note="study wording; standard DAST-10 asks 'always able to stop' reverse-scored"),
        Item("Have you ever had blackouts or flashbacks as a result of drug "
             "use?", _NO_YES),
        Item("Do you ever feel bad or guilty about your drug use?", _NO_YES),
        Item("Does your spouse (or parents) ever complain about your "
             "involvement with drugs?", _NO_YES),
        Item("Have you neglected your family because of your use of drugs?",
             _NO_YES),
        Item("Have you engaged in illegal activities in order to obtain "
             "drugs?", _NO_YES),
        Item("Have you ever experienced withdrawal symptoms (felt sick) when "
             "you stopped taking drugs?", _NO_YES),
        Item("Have you had medical problems as a result of your drug use "
             "(e.g., memory loss, hepatitis, convulsions, bleeding)?", _NO_YES),
    ),
    response_scale="Yes/No, 1 point per problem answer",
    scoring="Count problem answers (0–10).",
    # Four zones, as the study's dialogue defines them and its feedback text is
    # keyed to. A deliberate departure from the standard banding, which splits
    # 1-5 into low and moderate rather than treating it as one zone.
    bands=(
        RiskBand("Healthy (no problems reported)", 0, 0,
                 "Affirm, education", zone="healthy"),
        RiskBand("Risky", 1, 5,
                 "Brief intervention (advice), monitor", zone="risky"),
        RiskBand("Harmful", 6, 8,
                 "Brief intervention + consider referral", zone="harmful"),
        RiskBand("Dependent", 9, 10,
                 "Refer to intensive assessment/treatment", zone="dependent"),
    ),
)

CAGE_AID = Instrument(
    key="cage_aid",
    name="CAGE-AID (alcohol + drugs, ultra-brief)",
    domain="combined",
    when_to_use="Very fast lifetime dependence flag when time is short.",
    items=(
        "Felt you ought to CUT down on drinking/drug use?",
        "Have people ANNOYED you by criticizing your drinking/drug use?",
        "Felt bad or GUILTY about your drinking/drug use?",
        "Ever had a drink/used drugs first thing (EYE-opener) to steady nerves?",
    ),
    response_scale="Yes/No",
    scoring="Count 'Yes' (0–4).",
    bands=(
        RiskBand("Negative", 0, 1, "Continue routine screening"),
        RiskBand("Clinically significant", 2, 4, "Full assessment + brief intervention/referral"),
    ),
)

TAPS = Instrument(
    key="taps",
    name="TAPS (Tobacco, Alcohol, Prescription, Substance)",
    domain="combined",
    when_to_use="One brief pass across all four substance classes in primary care.",
    items=(
        "Past-12-month use frequency: Tobacco",
        "Past-12-month use frequency: Alcohol",
        "Past-12-month use frequency: illicit/street drugs",
        "Past-12-month use frequency: prescription meds used non-medically",
    ),
    response_scale="Daily / Weekly / Monthly / Less-than-monthly / Never",
    scoring="Any 'Monthly or more' triggers substance-specific follow-up items.",
    bands=(
        RiskBand("No problem use", 0, 0, "Affirm"),
        RiskBand("Problem use / higher risk", 1, 1, "Brief intervention or referral by substance"),
    ),
)

CRAFFT = Instrument(
    key="crafft",
    name="CRAFFT 2.1 (adolescents ≤21)",
    domain="adolescent",
    when_to_use="Screen users aged 12–21. Use INSTEAD of adult tools.",
    items=(
        "Ridden in a CAR driven by someone (incl. self) who was high/using?",
        "Use substances to RELAX, feel better, or fit in?",
        "Use substances while by yourself, ALONE?",
        "FORGET things you did while using?",
        "FAMILY/friends tell you to cut down?",
        "Gotten into TROUBLE while using?",
    ),
    response_scale="Yes/No",
    scoring="Count 'Yes' (0–6).",
    bands=(
        RiskBand("Low risk", 0, 1, "Praise, encouragement"),
        RiskBand("Positive – higher risk", 2, 6, "Brief intervention + consider referral"),
    ),
)

# The order these are rendered into the prompt.
ALL_INSTRUMENTS: tuple[Instrument, ...] = (
    NIDA_QUICK_SCREEN,
    AUDIT_C,
    AUDIT,
    DAST_10,
    CAGE_AID,
    TAPS,
    CRAFFT,
)

BY_KEY = {ins.key: ins for ins in ALL_INSTRUMENTS}


def render_catalog() -> str:
    """Every instrument as prompt text, in registry order."""
    return "\n\n".join(ins.render() for ins in ALL_INSTRUMENTS)

# Scoring lives in this file with the instrument data on purpose: a reviewer can
# see the items and exactly what they turn into without following a reference.
# All pure functions — no model, no I/O.
#
# Responses throughout are {item index: option code}, both zero-based, and a
# code's contribution is that option's score. Skipped items contribute zero, as
# the official forms specify.


class InvalidResponse(ValueError):
    """An item index or option code that does not exist on the instrument."""


def _item(instrument: Instrument, item_index: int) -> Item:
    """The Item at an index, raising InvalidResponse if it is not one."""
    try:
        item = instrument.items[item_index]
    except IndexError:
        raise InvalidResponse(
            f"{instrument.key} has no item {item_index}") from None
    if not isinstance(item, Item):
        raise InvalidResponse(
            f"{instrument.key} item {item_index} is prompt-only reference "
            f"data (plain string), not administrable")
    return item


def option_score(instrument: Instrument, item_index: int, code: int) -> int:
    """Points contributed by answering an item with a given option.

    Raises InvalidResponse for an unknown item or code. Returning zero instead
    would turn a coding bug into a quietly understated screening score.
    """
    item = _item(instrument, item_index)
    if not 0 <= code < len(item.options):
        raise InvalidResponse(
            f"{instrument.key} item {item_index} has no option code {code} "
            f"(valid: 0–{len(item.options) - 1})")
    return item.options[code].score


def _skipped_items(instrument: Instrument,
                   responses: Mapping[int, int]) -> frozenset[int]:
    """Which items the skip rules remove, given the answers so far."""
    if not instrument.skip_rules:
        return frozenset()
    scores = {i: option_score(instrument, i, code)
              for i, code in responses.items()}
    skipped: set[int] = set()
    for rule in instrument.skip_rules:
        if rule.when(scores):
            skipped.update(rule.skip)
    return frozenset(skipped)


def next_item_index(instrument: Instrument, responses: Mapping[int, int],
                    missing: Collection[int] = ()) -> int | None:
    """The next item to ask, or None once the instrument is finished.

    Args:
        instrument: the instrument being administered.
        responses: answers so far.
        missing: items the person could not or would not answer.

    Items are asked in order, and skipped ones are never asked. Missing items
    are likewise not re-asked and contribute zero — but unlike skipped items
    they mark the assessment incomplete, because the score is then a floor
    rather than a result.
    """
    skipped = _skipped_items(instrument, responses)
    for i in range(len(instrument.items)):
        if i in skipped or i in responses or i in missing:
            continue
        return i
    return None


def is_complete(instrument: Instrument, responses: Mapping[int, int],
                missing: Collection[int] = ()) -> bool:
    """Whether every item has been answered, skipped or marked missing."""
    return next_item_index(instrument, responses, missing) is None


def total_score(instrument: Instrument, responses: Mapping[int, int]) -> int:
    """The instrument's raw score. Every recorded response is validated."""
    skipped = _skipped_items(instrument, responses)
    return sum(option_score(instrument, i, code)
               for i, code in responses.items() if i not in skipped)


@dataclass(frozen=True)
class Assessment:
    """The screening result for one instrument.

    Anything in `missing` was left unanswered and scored zero, which makes
    `score` a lower bound and the zone a floor rather than a finding. That
    distinction is carried on the result and into the audit record, so a
    provider sees which items to follow up instead of a result that looks
    complete. Whether the spoken feedback should mention it is pending
    clinician review.
    """

    instrument_key: str
    score: int
    complete: bool
    band: RiskBand | None     # None only when the score falls outside every band
    missing: tuple[int, ...] = ()

    @property
    def zone(self) -> str:
        """The stable zone key, or "" when no band matched."""
        return self.band.zone if self.band else ""

    @property
    def action(self) -> str:
        """What the protocol does about this zone, or "" when no band matched."""
        return self.band.action if self.band else ""


def assess(instrument: Instrument, responses: Mapping[int, int],
           missing: Collection[int] = ()) -> Assessment:
    """Score the answers so far and place them in a risk zone.

    Safe to call mid-instrument: an incomplete result carries the running
    subtotal, and says so. Missing items score zero and are listed on the
    result, so a zone derived from a partial screening can be recognised as one.
    """
    score = total_score(instrument, responses)
    return Assessment(
        instrument_key=instrument.key,
        score=score,
        complete=is_complete(instrument, responses, missing),
        band=risk_band_for(instrument, score),
        missing=tuple(sorted(missing)),
    )
