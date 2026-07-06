"""Validated SBIRT screening instruments (the "S" in SBIRT).

Single source of truth for the screening tools the counselor is allowed to use.
Each instrument is plain data — items, scoring rule, and risk bands — so the
clinical content can be reviewed and updated here WITHOUT touching prompt strings
or pipeline code. `render()` turns any instrument into prompt-ready text, and
`risk_band_for()` maps a computed score to its SBIRT action.

The instruments this app actually ADMINISTERS (per the study protocol in
SBIRT_Reference/) carry fully structured items — official interview wording plus
an ordered option list with per-option scores — so the deterministic scoring
functions below, the runtime state machine (runtime.py), constrained NLU
coding, and fixed-clip pre-generation all read from the same data. The remaining
instruments are prompt-only reference and keep plain-string items.

Authoritative sources (see SBIRT_Reference/):
  • AUDIT items/options/skip rules — WHO AUDIT manual, Box 4 interview version
    ("AUDIT.pdf" / "AUDIT Manual.pdf").
  • DAST-10 items as worded for THIS study — "Case Cards UH - SBIRT Client
    Generic Provider.pdf" (see the item-3 deviation note below).
  • Pre-screen ("SBIRT 3 Questions") and the four feedback risk zones —
    "AI SBIRT app dialogue for study.docx".
Other references: Skinner DAST-10; Ewing CAGE / Brown CAGE-AID; NIDA Quick
Screen / NM ASSIST; TAPS tool; Knight CRAFFT 2.1.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Collection, Mapping


@dataclass(frozen=True)
class Option:
    """One answer choice: exact wording + the points it contributes.

    `aliases` are EXACT spoken equivalents (matched lowercased, whole-answer)
    for deterministic pre-matching only — never fuzzy. Populating them is a
    clinical coding decision (PENDING CLINICIAN REVIEW), so they ship empty;
    semantic mapping stays with the never-guess LLM coder.
    """

    label: str
    score: int
    aliases: tuple[str, ...] = ()


@dataclass(frozen=True)
class Item:
    """One administered question: official wording + ordered options.

    A recorded response is the option's INDEX in `options` (its "code");
    the contributed score is `options[code].score`. For most items code ==
    score, but e.g. AUDIT items 9-10 score 0/2/4 for codes 0/1/2.

    `verbatim=True` (the conservative default, pending the study's ruling on
    exact-wording requirements) means the stem is SPOKEN exactly as written;
    the conversational engine may add acknowledgment around it but never
    rephrase it. Open/number questions that are protocol conversation (not
    instrument items) live in the flow layer, not here.

    `confirm=True` (T20) marks score-bearing items whose coded answer is read
    back for a yes/no confirmation BEFORE it is committed. Round 2 makes the
    read-back the exception, not the default: exact-wording answers and
    cleanly derived codes (see `coding` below) commit directly; the read-back
    fires only when a conversion assumption entered the coding, the value sat
    near a bucket boundary, or the answer contradicts an earlier one
    (runtime.confirm_reason). A "no" re-collects the item.

    `coding` names the deterministic derivation for semantic answers
    (coding.py, T26): "choice" items accept a directly coded option;
    "freq_q1"/"freq5" items require an extracted rate (value + per) and
    "quantity_drinks" items an extracted amount (value + unit [+ beverage]) —
    the option code is then COMPUTED from the scale tables, never taken from
    the model's claim.
    """

    text: str
    options: tuple[Option, ...]
    note: str = ""  # skip rule / scoring deviation, rendered with the item
    verbatim: bool = True
    confirm: bool = False
    coding: str = "choice"   # "choice" | "freq_q1" | "freq5" | "quantity_drinks"

    @property
    def kind(self) -> str:
        """Presentation hint for the NLU/ask layer: "yesno" | "scale".
        Derived from the options so it can never drift from them."""
        labels = tuple(o.label.lower() for o in self.options)
        return "yesno" if labels == ("no", "yes") else "scale"


@dataclass(frozen=True)
class SkipRule:
    """Declarative skip logic: when `when(scores)` is true, every item index
    in `skip` is never asked and contributes 0 (official-form behavior).

    `scores` maps answered item index -> CONTRIBUTED SCORE (never the raw
    option code), so instruments where code != score (AUDIT items 9-10)
    can never confuse a predicate. Adding a skip rule = adding data here;
    the engine (`next_item_index`) needs no changes.
    """

    skip: tuple[int, ...]
    when: Callable[[Mapping[int, int]], bool]
    note: str = ""


@dataclass(frozen=True)
class RiskBand:
    """A scoring zone and the SBIRT action it triggers.

    `zone` is the stable key used by the study's feedback templates and tests:
    one of "healthy" | "risky" | "harmful" | "dependent" for administered
    instruments; "" for prompt-only reference instruments.
    """

    label: str
    low: int          # inclusive lower bound of the raw score
    high: int         # inclusive upper bound
    action: str       # which SBIRT step this zone routes to
    zone: str = ""

    def contains(self, score: int) -> bool:
        return self.low <= score <= self.high


@dataclass(frozen=True)
class Instrument:
    key: str
    name: str
    domain: str            # alcohol | drugs | tobacco | combined | adolescent
    when_to_use: str
    items: tuple[Item | str, ...]
    response_scale: str
    scoring: str
    bands: tuple[RiskBand, ...]
    preamble: str = ""     # spoken once before item 1 (from the study protocol)
    skip_rules: tuple[SkipRule, ...] = ()   # declarative skip logic (see SkipRule)

    def render(self) -> str:
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
    """Map a raw score to its risk band (None if out of range)."""
    for band in instrument.bands:
        if band.contains(score):
            return band
    return None


# --- Shared option scales (exact wordings from the WHO AUDIT Box 4 form) -----
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


# --- Pre-screen: the study's "SBIRT 3 Questions" ------------------------------
# This app's pre-screen per the study protocol (app dialogue + case cards) —
# NOT the 4-item NIDA Quick Screen below, which stays as prompt-only reference.
# score > 0 on a question = positive pre-screen for that domain.

@dataclass(frozen=True)
class PreScreenQuestion:
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

# Official WHO interview version (Box 4). Skip rules are encoded in next_item_index()
# from the notes below: item 1 "Never" skips to items 9-10; items 2+3 totalling
# 0 skips to items 9-10. Skipped items score 0.
AUDIT = Instrument(
    key="audit",
    name="AUDIT (alcohol, full 10-item)",
    domain="alcohol",
    when_to_use="Positive alcohol pre-screen (>4 drinks in a day within the past year).",
    preamble=(
        "Now I am going to ask you some questions about your use of alcoholic "
        "beverages during this past year."
    ),
    # confirm=True (T20): the quantity/frequency triad (items 1-3) and the
    # dependence-domain items (4-6, per the manual's Box 2 domains) are the
    # score-critical answers most exposed to semantic mis-coding, so an
    # LLM-coded answer is read back before it commits. Which items carry the
    # flag is a clinical coding decision — PENDING CLINICIAN REVIEW.
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
    # WHO AUDIT Box 4 skip rules as data (predicates read contributed SCORES):
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

# Items as worded in THIS study's protocol (case cards). NOTE on item 3: the
# standard Skinner DAST-10 asks "Are you ALWAYS ABLE to stop using drugs when
# you want to?" and reverse-scores it ('No' = 1). The study's interview script
# flips the wording to "unable to stop", scored POSITIVELY ('Yes' = 1) — the
# two are equivalent, but coding must follow the wording actually asked.
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
    # Zones per the study's app dialogue (feedback is keyed to these four).
    # The standard Skinner banding splits 1-5 into low (1-2) / moderate (3-5);
    # this study's protocol treats 1-5 as one "Risky" zone.
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

# Ordered registry — iterate this to render the full catalog into the prompt.
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
    return "\n\n".join(ins.render() for ins in ALL_INSTRUMENTS)

# =====================================================================
# Deterministic scoring — same file as the instrument data on purpose:
# a reader sees the AUDIT/DAST items AND exactly how they turn into a
# score + risk zone in one place. Pure functions, no LLM, no IO.
#
# A response set is {item_index: option_code} (both 0-based); the code is
# the option's index in Item.options and its score is options[code].score
# (AUDIT items 9-10: codes 0/1/2 score 0/2/4). Skip logic lives on each
# instrument as declarative SkipRule data (see AUDIT.skip_rules); skipped
# items contribute 0, per the official form.
# =====================================================================


class InvalidResponse(ValueError):
    """An item index or option code that does not exist on the instrument."""


def _item(instrument: Instrument, item_index: int) -> Item:
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
    """Score contributed by answering `item_index` with option `code`.
    Raises InvalidResponse for an unknown item or code — a coding layer bug
    must surface, never silently score 0."""
    item = _item(instrument, item_index)
    if not 0 <= code < len(item.options):
        raise InvalidResponse(
            f"{instrument.key} item {item_index} has no option code {code} "
            f"(valid: 0–{len(item.options) - 1})")
    return item.options[code].score


def _skipped_items(instrument: Instrument,
                   responses: Mapping[int, int]) -> frozenset[int]:
    """Item indexes removed by the instrument's declarative skip_rules, given
    answers so far. Rules read contributed SCORES (never raw codes), so the
    generalization is safe for instruments where code != score."""
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
    """The next item to administer, honoring skip rules; None when complete.
    Items are asked in order; skipped items are never asked. `missing` items
    (persistent don't-know / declined single items, F1/F2) are never re-asked
    and contribute 0 like officially skipped items — but they mark the
    assessment incomplete (see Assessment.missing)."""
    skipped = _skipped_items(instrument, responses)
    for i in range(len(instrument.items)):
        if i in skipped or i in responses or i in missing:
            continue
        return i
    return None


def is_complete(instrument: Instrument, responses: Mapping[int, int],
                missing: Collection[int] = ()) -> bool:
    return next_item_index(instrument, responses, missing) is None


def total_score(instrument: Instrument, responses: Mapping[int, int]) -> int:
    """Sum of contributed scores. Skipped items count 0 (official form).
    Validates every recorded response."""
    skipped = _skipped_items(instrument, responses)
    return sum(option_score(instrument, i, code)
               for i, code in responses.items() if i not in skipped)


@dataclass(frozen=True)
class Assessment:
    """The authoritative screening result for one instrument.

    `missing` lists items the person could not / would not answer (F1/F2):
    they scored 0, so `score` is a LOWER BOUND and the zone a floor — the
    audit record carries the flag so the provider sees which items to follow
    up rather than a false picture of complete data. (Whether the spoken
    feedback should also mention it: PENDING CLINICIAN REVIEW.)
    """

    instrument_key: str
    score: int
    complete: bool
    band: RiskBand | None     # None only if score is out of any band's range
    missing: tuple[int, ...] = ()

    @property
    def zone(self) -> str:
        return self.band.zone if self.band else ""

    @property
    def action(self) -> str:
        return self.band.action if self.band else ""


def assess(instrument: Instrument, responses: Mapping[int, int],
           missing: Collection[int] = ()) -> Assessment:
    """Compute the deterministic score + risk zone for the responses so far.
    `complete=False` means items remain; the score is the running subtotal.
    Unanswerable items in `missing` score 0 and are carried on the result."""
    score = total_score(instrument, responses)
    return Assessment(
        instrument_key=instrument.key,
        score=score,
        complete=is_complete(instrument, responses, missing),
        band=risk_band_for(instrument, score),
        missing=tuple(sorted(missing)),
    )
