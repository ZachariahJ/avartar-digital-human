"""Deterministic answer coding (T26): extracted facts -> option codes.

Round-2 backbone. The NLU turn no longer picks frequency/quantity buckets
itself: for items whose `Item.coding` names a scale here, the model only
EXTRACTS what the person said — a count and a period ("every week" -> value 1,
per "week"), or an amount, unit and beverage ("one liter of whiskey" ->
value 1, unit "liter", beverage "whiskey") — and the code is COMPUTED from
the tables below. "Legal but wrong" codes (the over-coding class of field
defects) become structurally impossible at the bucketing step; what remains
fallible is extraction itself, which is why conversions and boundary values
still get a T20 read-back.

Everything here is data + arithmetic, defined once and reviewable:
  • bucket thresholds per option scale (incl. the "once a week" boundary
    ruling: 1/week falls INSIDE "2 to 4 times a month", whose label spans
    4.3 times a month — not "2 to 3 times a week");
  • a beverage -> ABV table and a container -> volume table for converting
    spoken amounts into the study's standard drink (12 oz beer / 5 oz wine /
    1.5 oz spirits — 0.6 fl oz of ethanol each, templates.FIXED
    ["alcohol.edu.standard_drink"]), per the WHO manual's instruction that
    screening must define drinks in local terms (SBIRT_REF.pdf p.15). The
    conversion no longer depends on whether the person accepted the
    standard-drink education;
  • cross-item consistency bounds (AUDIT Q1 vs Q3; Q/F frequency vs Q1).

Pure module: no LLM, no IO, no session state — every table row and boundary
is unit-testable. Threshold and ABV values are clinical data, PENDING
CLINICIAN REVIEW like the rest of the instrument content.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# --------------- The study's standard drink ---------------

# 0.6 US fl oz of pure ethanol (≈14 g): 12 oz x 5% beer = 5 oz x 12% wine =
# 1.5 oz x 40% spirits, exactly the three examples the study's education
# unit speaks.
ETHANOL_ML_PER_DRINK = 17.74

_OZ_ML = 29.574

# Default alcohol-by-volume per beverage (fractions). Data, not code:
# override/extend here, never in the extraction prompt.
ABV: dict[str, float] = {
    "beer": 0.05, "light beer": 0.042, "lager": 0.05, "ale": 0.055,
    "stout": 0.06, "ipa": 0.065, "malt liquor": 0.07,
    "hard seltzer": 0.05, "seltzer": 0.05, "cider": 0.05,
    "wine": 0.12, "red wine": 0.13, "white wine": 0.12,
    "champagne": 0.12, "prosecco": 0.11, "sangria": 0.11,
    "port": 0.20, "sherry": 0.17, "vermouth": 0.16,
    "sake": 0.15, "soju": 0.17, "mead": 0.11,
    "whiskey": 0.40, "whisky": 0.40, "bourbon": 0.40, "scotch": 0.40,
    "rye": 0.40, "vodka": 0.40, "gin": 0.40, "rum": 0.40,
    "tequila": 0.40, "brandy": 0.40, "cognac": 0.40, "mezcal": 0.42,
    "moonshine": 0.50, "baijiu": 0.52, "everclear": 0.75,
    "liquor": 0.40, "spirits": 0.40, "hard liquor": 0.40,
    "liqueur": 0.25, "schnapps": 0.25, "absinthe": 0.60,
}

# Spirits containers imply a spirits ABV even when no beverage was named.
_SPIRIT_UNITS = {"shot", "jigger", "fifth", "handle", "nip", "mini"}

# Container/unit -> milliliters. `None` means beverage-dependent (looked up
# in _SIZED_UNITS below); a unit absent from both tables cannot be converted
# and the answer stays unclear (the model then asks, never guesses).
ML_PER_UNIT: dict[str, float | None] = {
    "ml": 1.0, "milliliter": 1.0, "cl": 10.0, "liter": 1000.0,
    "litre": 1000.0, "l": 1000.0, "half liter": 500.0,
    "oz": _OZ_ML, "ounce": _OZ_ML, "fluid ounce": _OZ_ML,
    "shot": 1.5 * _OZ_ML, "jigger": 1.5 * _OZ_ML,
    "double": 3.0 * _OZ_ML, "double shot": 3.0 * _OZ_ML,
    "nip": 50.0, "mini": 50.0, "flask": 8 * _OZ_ML,
    "pint": 473.0, "half pint": 237.0, "quart": 946.0,
    "cup": 237.0, "tall boy": 473.0, "forty": 1183.0,
    "fifth": 750.0, "handle": 1750.0, "magnum": 1500.0,
    "pitcher": 1893.0, "six pack": 6 * 355.0, "six-pack": 6 * 355.0,
    "twelve pack": 12 * 355.0, "case": 24 * 355.0,
    "box": 3000.0,                      # boxed wine
    "bottle": None, "can": None, "glass": None,
}

# Beverage-dependent container sizes (ml).
_SIZED_UNITS: dict[str, dict[str, float]] = {
    "bottle": {"beer": 355.0, "wine": 750.0, "spirits": 750.0},
    "can": {"beer": 355.0, "wine": 250.0, "spirits": 355.0},
    "glass": {"beer": 355.0, "wine": 148.0, "spirits": 1.5 * _OZ_ML},
}

_DRINK_UNITS = {"", "drink", "drinks", "standard drink", "standard drinks",
                "beer", "beers", "glass of wine", "shot of liquor"}


def _family(beverage: str) -> str:
    """Coarse family for beverage-dependent container sizes."""
    abv = ABV.get(beverage, 0.0)
    if abv >= 0.20:
        return "spirits"
    if abv >= 0.10:
        return "wine"
    return "beer"


@dataclass(frozen=True)
class Derived:
    """A deterministically computed option code plus the confirm triggers.

    `assumed`  — a default (ABV, container size) entered the computation:
                 the read-back must surface the conversion.
    `boundary` — the normalized value sits close to a bucket edge: worth a
                 read-back even though the arithmetic is exact.
    `note`     — the human-readable conversion, spoken in the read-back
                 ("1 liter of whiskey is about 23 standard drinks").
    """

    code: int
    assumed: bool = False
    boundary: bool = False
    note: str = ""


# --------------- Frequency scales ---------------

_PER_WEEK = {"day": 7.0, "week": 1.0, "month": 1 / 4.345, "year": 1 / 52.18}

# (upper bound in times-per-week, code); above the last bound -> top code.
# AUDIT Q1: Never / Monthly or less / 2-4 a month / 2-3 a week / 4+ a week.
# Boundary ruling (decision point, defined ONCE here): once a week = 4.3
# times a month -> inside "2 to 4 times a month" (code 2).
_Q1_BOUNDS = ((0.25, 1), (1.5, 2), (3.5, 3))
# WHO _FREQ_5: Never / Less than monthly / Monthly / Weekly / Daily-or-almost.
# "Daily or almost daily" starts at 5 times a week.
_FREQ5_BOUNDS = ((0.23, 1), (0.9, 2), (5.0, 3))

_FREQ_SCALES = {"freq_q1": (_Q1_BOUNDS, 4), "freq5": (_FREQ5_BOUNDS, 4)}

_BOUNDARY_MARGIN = 0.15   # within 15% of a threshold -> read back


def per_week(value: float, per: str | None) -> float | None:
    """Normalize a spoken rate to times-per-week; None when not computable."""
    if value is None or value < 0:
        return None
    if value == 0:
        return 0.0
    factor = _PER_WEEK.get((per or "").strip().lower())
    if factor is None:
        return None
    return value * factor


def derive_frequency(scale: str, value: float | None,
                     per: str | None) -> Derived | None:
    """Bucket an extracted rate onto a frequency scale. None = not derivable
    (the turn stays unclear and the avatar clarifies)."""
    bounds, top = _FREQ_SCALES[scale]
    rate = per_week(value, per) if value is not None else None
    if rate is None:
        return None
    if rate == 0:
        return Derived(code=0)
    code = top
    for bound, c in bounds:
        if rate <= bound:
            code = c
            break
    boundary = any(abs(rate - b) / b < _BOUNDARY_MARGIN for b, _ in bounds)
    return Derived(code=code, boundary=boundary)


# --------------- Drink quantities ---------------

# AUDIT Q2 buckets: 1-2 / 3-4 / 5-6 / 7-9 / 10+ (edges between buckets).
_Q2_EDGES = (2.5, 4.5, 6.5, 9.5)


def _q2_code(drinks: float) -> int:
    for code, edge in enumerate(_Q2_EDGES):
        if drinks < edge:
            return code
    return 4


def standard_drinks(value: float, unit: str, beverage: str) -> float | None:
    """Spoken amount -> standard drinks; None when the unit/beverage cannot
    be resolved from the tables (never guessed)."""
    unit = " ".join(unit.strip().lower().split())
    if unit.endswith("s") and unit not in ML_PER_UNIT:
        unit = unit[:-1]
    beverage = " ".join((beverage or "").strip().lower().split())
    if unit not in ML_PER_UNIT:
        return None
    ml = ML_PER_UNIT[unit]
    if ml is None:                       # beverage-dependent container
        if not beverage or beverage not in ABV:
            return None
        ml = _SIZED_UNITS[unit][_family(beverage)]
    if beverage in ABV:
        abv = ABV[beverage]
    elif unit in _SPIRIT_UNITS:
        abv = ABV["spirits"]
    else:
        return None
    return value * ml * abv / ETHANOL_ML_PER_DRINK


def derive_quantity(value: float | None, unit: str | None,
                    beverage: str | None) -> Derived | None:
    """Bucket an extracted amount onto the AUDIT Q2 scale. Drink counts map
    directly; volumes convert through the ABV/container tables (assumed=True
    -> the read-back speaks the conversion, which doubles as the standard-
    drink explanation for people who declined the education unit)."""
    if value is None or value <= 0:
        return None
    u = " ".join((unit or "").strip().lower().split())
    if u in _DRINK_UNITS:
        drinks = float(value)
        boundary = any(abs(drinks - e) < 0.5 for e in _Q2_EDGES)
        return Derived(code=_q2_code(drinks), boundary=boundary)
    drinks = standard_drinks(float(value), u, beverage or "")
    if drinks is None:
        return None
    n = max(1, int(round(drinks)))
    amount = f"{value:g} {u}" if value != 1 else f"1 {u}"
    of = f" of {beverage}" if beverage else ""
    note = (f"{amount}{of} is about {n} standard "
            f"drink{'s' if n != 1 else ''}")
    return Derived(code=_q2_code(drinks), assumed=True, note=note)


def derive(coding: str, *, value: float | None = None, per: str | None = None,
           unit: str | None = None, beverage: str | None = None
           ) -> Derived | None:
    """Dispatch on Item.coding. None = not derivable from what was extracted."""
    if coding in _FREQ_SCALES:
        return derive_frequency(coding, value, per)
    if coding == "quantity_drinks":
        return derive_quantity(value, unit, beverage)
    raise ValueError(f"unknown coding kind {coding!r}")


# --------------- Free-text frequency (Q/F slot) parsing ---------------

_WORD_N = {"once": 1, "twice": 2, "one": 1, "two": 2, "three": 3, "four": 4,
           "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10}

_N_TIMES_RE = re.compile(
    r"\b(\d+|once|twice|one|two|three|four|five|six|seven|eight|nine|ten)"
    r"(?:\s+times?)?\s+(?:a|per|every)\s+(day|week|month|year)\b")

_DAILY_RE = re.compile(
    r"\b(every\s+(?:single\s+)?(?:day|night|evening|morning)|daily|"
    r"all the time|always)\b")

_SIMPLE = (
    (re.compile(r"\bevery other day\b"), 3.5),
    (re.compile(r"\bmost (?:days|nights|evenings)\b"), 5.5),
    (re.compile(r"\b(?:once\s+)?(?:a|per|every)\s+week\b|\bweekly\b"), 1.0),
    (re.compile(r"\b(?:once\s+)?(?:a|per|every)\s+month\b|\bmonthly\b"),
     1 / 4.345),
)


def parse_freq_text(text: str) -> float | None:
    """Conservative times-per-week from a free-text frequency phrase (the
    Q/F slot). Only unambiguous patterns parse; anything else -> None (the
    consistency rules must never fire on a guess)."""
    t = " ".join((text or "").strip().lower().split())
    if not t:
        return None
    if _DAILY_RE.search(t):
        return 7.0
    m = _N_TIMES_RE.search(t)
    if m:
        n = m.group(1)
        n = float(n) if n.isdigit() else float(_WORD_N[n])
        return per_week(n, m.group(2))
    for pattern, rate in _SIMPLE:
        if pattern.search(t):
            return rate
    return None


# --------------- Cross-item consistency bounds (Phase 5) ---------------

# Highest drinking rate (times/week) each AUDIT Q1 code can honestly mean,
# and the lowest rate of 6+-drink occasions each Q3 code implies. A Q3
# answer implying MORE heavy-drinking days than Q1 allows drinking days at
# all is a contradiction worth one gentle read-back.
Q1_MAX_PER_WEEK = {0: 0.0, 1: 0.25, 2: 1.5, 3: 3.5, 4: 14.0}
Q3_MIN_PER_WEEK = {0: 0.0, 1: 0.0, 2: 0.2, 3: 0.9, 4: 5.0}

# A Q/F frequency this many times above the Q1 code's ceiling flags the Q1
# answer (margin absorbs casual overstatement like "every day" said loosely).
QF_VS_Q1_MARGIN = 1.9
