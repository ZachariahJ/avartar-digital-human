
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Collection, Mapping


@dataclass(frozen=True)
class Option:

    label: str
    score: int
    aliases: tuple[str, ...] = ()


@dataclass(frozen=True)
class Item:

    text: str
    options: tuple[Option, ...]
    note: str = ""
    verbatim: bool = True
    confirm: bool = False
    coding: str = "choice"

    @property
    def kind(self) -> str:
        labels = tuple(o.label.lower() for o in self.options)
        return "yesno" if labels == ("no", "yes") else "scale"


@dataclass(frozen=True)
class SkipRule:

    skip: tuple[int, ...]
    when: Callable[[Mapping[int, int]], bool]
    note: str = ""


@dataclass(frozen=True)
class RiskBand:

    label: str
    low: int
    high: int
    action: str
    zone: str = ""

    def contains(self, score: int) -> bool:
        return self.low <= score <= self.high


@dataclass(frozen=True)
class Instrument:

    key: str
    name: str
    domain: str
    when_to_use: str
    items: tuple[Item | str, ...]
    response_scale: str
    scoring: str
    bands: tuple[RiskBand, ...]
    preamble: str = ""
    skip_rules: tuple[SkipRule, ...] = ()


def risk_band_for(instrument: Instrument, score: int) -> RiskBand | None:
    for band in instrument.bands:
        if band.contains(score):
            return band
    return None


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



@dataclass(frozen=True)
class PreScreenQuestion:

    key: str
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


AUDIT = Instrument(
    key="audit",
    name="AUDIT (alcohol, full 10-item)",
    domain="alcohol",
    when_to_use="Positive alcohol pre-screen (>4 drinks in a day within the past year).",
    preamble=(
        "Now I am going to ask you some questions about your use of alcoholic "
        "beverages during this past year."
    ),
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


BY_KEY = {ins.key: ins for ins in (AUDIT, DAST_10)}




class InvalidResponse(ValueError):
    pass


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
    item = _item(instrument, item_index)
    if not 0 <= code < len(item.options):
        raise InvalidResponse(
            f"{instrument.key} item {item_index} has no option code {code} "
            f"(valid: 0–{len(item.options) - 1})")
    return item.options[code].score


def _skipped_items(instrument: Instrument,
                   responses: Mapping[int, int]) -> frozenset[int]:
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
    skipped = _skipped_items(instrument, responses)
    return sum(option_score(instrument, i, code)
               for i, code in responses.items() if i not in skipped)


@dataclass(frozen=True)
class Assessment:

    instrument_key: str
    score: int
    complete: bool
    band: RiskBand | None
    missing: tuple[int, ...] = ()

    @property
    def zone(self) -> str:
        return self.band.zone if self.band else ""

    @property
    def action(self) -> str:
        return self.band.action if self.band else ""


def assess(instrument: Instrument, responses: Mapping[int, int],
           missing: Collection[int] = ()) -> Assessment:
    score = total_score(instrument, responses)
    return Assessment(
        instrument_key=instrument.key,
        score=score,
        complete=is_complete(instrument, responses, missing),
        band=risk_band_for(instrument, score),
        missing=tuple(sorted(missing)),
    )
