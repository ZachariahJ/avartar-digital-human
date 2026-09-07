"""Where to send someone, and what to do when screening must stop.

Reference content for the referral end of SBIRT: levels of care, medications,
national resources and how to hand somebody over, plus the crisis situations
that take priority over any of it.

Rendered into the system prompt. The executable crisis detection lives in
crisis.py; this is the guidance the model works from once that has fired.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LevelOfCare:
    code: str
    name: str
    fits: str

    def render(self) -> str:
        """This level as prompt text."""
        return f"  • ASAM {self.code} — {self.name}: {self.fits}"


# Matching intensity to severity matters in both directions: over-referring is
# a failure as much as under-referring.
ASAM_LEVELS = (
    LevelOfCare("0.5", "Early intervention", "At-risk use, no diagnosis yet; education + brief intervention."),
    LevelOfCare("1.0", "Outpatient", "Mild severity; <9 hrs/week counseling."),
    LevelOfCare("2.1", "Intensive outpatient (IOP)", "9–19 hrs/week; needs structure but can live at home."),
    LevelOfCare("2.5", "Partial hospitalization (PHP)", "20+ hrs/week, day treatment."),
    LevelOfCare("3.x", "Residential / inpatient", "Unsafe or unsupportive home; 24-hr support needed."),
    LevelOfCare("4.0", "Medically managed intensive inpatient", "Acute withdrawal or medical instability."),
)

# First-line treatment, not a last resort, and widely believed otherwise — hence
# the final entry, which exists to pre-empt that objection.
MAT_OPTIONS = (
    "Opioid use: buprenorphine, methadone, or extended-release naltrexone.",
    "Alcohol use: naltrexone, acamprosate, or disulfiram.",
    "Tobacco: nicotine replacement, varenicline, or bupropion.",
    "Frame MAT as evidence-based medicine, not 'trading one addiction for another'.",
)

# National and always available, so a handoff never depends on knowing where the
# person is.
RESOURCES = (
    "SAMHSA National Helpline: 1-800-662-HELP (4357) — free, confidential, 24/7.",
    "FindTreatment.gov — searchable treatment locator.",
    "Mutual-help groups: AA, NA, SMART Recovery, Moderation Management.",
    "Primary care physician for coordinated medical care and MAT.",
    "988 Suicide & Crisis Lifeline (call or text 988).",
)

# Motivation is perishable: a connection made now beats a phone number to call
# later.
WARM_HANDOFF = (
    "Ask permission before referring.",
    "Offer a specific, named next step (not just 'see a counselor').",
    "Connect in the moment when possible; otherwise help schedule the first contact.",
    "Address barriers (cost, transport, stigma, childcare) out loud.",
    "Set a concrete follow-up so the loop is closed.",
)


@dataclass(frozen=True)
class CrisisFlag:
    trigger: str
    response: str


# Any of these stops the screening outright. Note that withdrawal is on the list
# because abrupt cessation can kill, which makes "just quit" actively dangerous
# advice rather than merely unhelpful.
CRISIS_PROTOCOL = (
    CrisisFlag("Suicidal ideation / self-harm / intent",
               "Empathy + urgency; give 988 and 911; do not leave it as a casual topic."),
    CrisisFlag("Overdose risk (opioids, mixing depressants)",
               "Discuss naloxone access and never-use-alone; urge immediate medical help."),
    CrisisFlag("Alcohol or benzodiazepine withdrawal (tremor, seizures, DTs)",
               "MEDICAL EMERGENCY — abrupt cessation can be fatal; direct to urgent medical care, do not advise quitting cold turkey."),
    CrisisFlag("Acute intoxication / immediate danger to self or others",
               "Direct to 911 / emergency services now."),
)

CRISIS_LINES = (
    "988 Suicide & Crisis Lifeline — call or text 988",
    "911 for immediate medical emergencies",
    "SAMHSA National Helpline — 1-800-662-4357",
)
