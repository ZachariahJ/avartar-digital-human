"""Validated SBIRT screening instruments (the "S" in SBIRT).

Single source of truth for the screening tools the counselor is allowed to use.
Each instrument is plain data — items, scoring rule, and risk bands — so the
clinical content can be reviewed and updated here WITHOUT touching prompt strings
or pipeline code. `render()` turns any instrument into prompt-ready text, and
`risk_band_for()` maps a computed score to its SBIRT action (the "data
structuring" the engineer persona is asked to do).

References: WHO AUDIT manual; Skinner DAST-10; Ewing CAGE / Brown CAGE-AID;
NIDA Quick Screen / NM ASSIST; TAPS tool; Knight CRAFFT 2.1.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RiskBand:
    """A scoring zone and the SBIRT action it triggers."""

    label: str
    low: int          # inclusive lower bound of the raw score
    high: int         # inclusive upper bound
    action: str       # which SBIRT step this zone routes to

    def contains(self, score: int) -> bool:
        return self.low <= score <= self.high


@dataclass(frozen=True)
class Instrument:
    key: str
    name: str
    domain: str            # alcohol | drugs | tobacco | combined | adolescent
    when_to_use: str
    items: tuple[str, ...]
    response_scale: str
    scoring: str
    bands: tuple[RiskBand, ...]

    def render(self) -> str:
        lines = [f"### {self.name}  ({self.domain}, {len(self.items)} items)",
                 f"When to use: {self.when_to_use}",
                 f"Response scale: {self.response_scale}",
                 "Items:"]
        lines += [f"  {i}. {item}" for i, item in enumerate(self.items, 1)]
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


# --- Pre-screen: single-question filters (NIDA Quick Screen frequency item) ----
# A "yes" to any of these opens the matching full instrument. This keeps a
# low-risk user from being walked through a 10-item questionnaire.
NIDA_QUICK_SCREEN = Instrument(
    key="nida_quick",
    name="NIDA Quick Screen (pre-screen)",
    domain="combined",
    when_to_use="First substance question. Ask before any full instrument.",
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

AUDIT = Instrument(
    key="audit",
    name="AUDIT (alcohol, full 10-item)",
    domain="alcohol",
    when_to_use="Positive AUDIT-C or unclear alcohol risk; gives a treatment zone.",
    items=(
        "Frequency of drinking (0–4)",
        "Typical quantity per occasion (0–4)",
        "Frequency of 6+ drinks on one occasion (0–4)",
        "Unable to stop drinking once started, past year (0–4)",
        "Failed to do what was normally expected because of drinking (0–4)",
        "Needed a first drink in the morning (eye-opener) (0–4)",
        "Guilt or remorse after drinking (0–4)",
        "Unable to remember the night before (blackout) (0–4)",
        "You or someone else injured because of your drinking (0/2/4)",
        "Others concerned / suggested you cut down (0/2/4)",
    ),
    response_scale="Items 1–8: 0–4; items 9–10: 0/2/4",
    scoring="Sum all items (0–40). Zones per WHO manual.",
    bands=(
        RiskBand("Zone I – Low risk", 0, 7, "Alcohol education / affirmation"),
        RiskBand("Zone II – Risky/hazardous", 8, 15, "Brief intervention (simple advice)"),
        RiskBand("Zone III – Harmful", 16, 19, "Brief intervention + brief counseling + monitor"),
        RiskBand("Zone IV – Likely dependence", 20, 40, "Refer to specialist assessment/treatment"),
    ),
)

DAST_10 = Instrument(
    key="dast_10",
    name="DAST-10 (drug use, non-alcohol)",
    domain="drugs",
    when_to_use="Positive drug pre-screen (illicit or non-medical prescription use).",
    items=(
        "Used drugs other than those required for medical reasons?",
        "Abuse more than one drug at a time?",
        "Always able to stop using drugs when you want to? (reverse-scored)",
        "Had blackouts or flashbacks from drug use?",
        "Feel bad or guilty about your drug use?",
        "Does your spouse/parents ever complain about your drug use?",
        "Neglected your family because of drug use?",
        "Engaged in illegal activities to obtain drugs?",
        "Experienced withdrawal symptoms when you stopped taking drugs?",
        "Had medical problems as a result of your drug use?",
    ),
    response_scale="Yes/No (item 3 is reverse-scored: 'No' = 1)",
    scoring="Count problem answers (0–10).",
    bands=(
        RiskBand("No problems reported", 0, 0, "Affirm, education"),
        RiskBand("Low level", 1, 2, "Brief intervention (advice), monitor"),
        RiskBand("Moderate level", 3, 5, "Brief intervention + brief treatment"),
        RiskBand("Substantial level", 6, 8, "Refer for assessment"),
        RiskBand("Severe level", 9, 10, "Refer to intensive assessment/treatment"),
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
