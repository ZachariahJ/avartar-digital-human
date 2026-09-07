"""Turns what somebody said about drinking into a scored option, arithmetically.

For frequency and quantity items the model does not choose the answer bucket.
It reports what was said — "every week" as one per week, "a liter of whiskey" as
one liter of whiskey — and the bucket is computed here from the tables below.

That division exists because of a specific failure mode. A model that picks the
bucket itself can return a code that is perfectly legal and simply wrong, and
nothing downstream can tell: the answer looks like every other answer, and the
score is silently off. Moving the threshold decision into arithmetic makes that
class of error impossible. Extraction can still be wrong, which is why
conversions and near-boundary values are read back to the person.

The tables are the reviewable part, and the values in them are clinical
decisions rather than implementation details:

  * where each option's bucket begins and ends, including the one genuinely
    ambiguous case — once a week is 4.3 times a month, which the code places
    inside "2 to 4 times a month" rather than "2 to 3 times a week";
  * alcohol by volume per drink and millilitres per container, which convert a
    spoken amount into standard drinks. The WHO manual requires screening to
    define drinks in local terms (SBIRT_REF.pdf p.15), and doing the conversion
    here means it does not depend on whether the person accepted the
    standard-drink education;
  * bounds for catching answers that contradict each other across items.

Pure arithmetic and lookups: no model, no I/O, no session state, so every row
and every boundary can be tested directly. Pending clinician review, like the
rest of the instrument content.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# 0.6 US fl oz of pure ethanol, about 14g. The three examples the education unit
# speaks all come to this: 12 oz of 5% beer, 5 oz of 12% wine, 1.5 oz of 40%
# spirits.
ETHANOL_ML_PER_DRINK = 17.74

_OZ_ML = 29.574

# Alcohol by volume, as a fraction. Extend this table to teach the system a new
# drink; putting it in the extraction prompt instead would make the conversion
# depend on the model rather than on reviewable data.
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

# Naming one of these implies spirits, so "three shots" is convertible even
# though no drink was named.
_SPIRIT_UNITS = {"shot", "jigger", "fifth", "handle", "nip", "mini"}

# Millilitres per container. None means the size depends on what is in it — a
# bottle of beer and a bottle of wine differ — and is looked up below. A unit in
# neither table cannot be converted, so the answer stays unclear and the person
# is asked, rather than a size being assumed.
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

# Sizes for the containers above whose volume depends on the drink.
_SIZED_UNITS: dict[str, dict[str, float]] = {
    "bottle": {"beer": 355.0, "wine": 750.0, "spirits": 750.0},
    "can": {"beer": 355.0, "wine": 250.0, "spirits": 355.0},
    "glass": {"beer": 355.0, "wine": 148.0, "spirits": 1.5 * _OZ_ML},
}

_DRINK_UNITS = {"", "drink", "drinks", "standard drink", "standard drinks",
                "beer", "beers", "glass of wine", "shot of liquor"}


def _family(beverage: str) -> str:
    """Group a drink as beer, wine or spirits by strength, for container sizes."""
    abv = ABV.get(beverage, 0.0)
    if abv >= 0.20:
        return "spirits"
    if abv >= 0.10:
        return "wine"
    return "beer"


@dataclass(frozen=True)
class Derived:
    """A computed option code, and whether it should be read back for confirmation.

    assumed:  a table default entered the calculation — a strength or a
              container size — so the person never stated the number the code
              rests on, and the conversion has to be said aloud.
    boundary: the value sits close enough to a bucket edge that a small
              extraction error would change the code, even though the
              arithmetic itself is exact.
    note:     that conversion in plain words, e.g. "1 liter of whiskey is about
              23 standard drinks".
    """

    code: int
    assumed: bool = False
    boundary: bool = False
    note: str = ""


_PER_WEEK = {"day": 7.0, "week": 1.0, "month": 1 / 4.345, "year": 1 / 52.18}

# (upper bound in times per week, code). Anything above the last bound takes the
# top code. The scales are Never / Monthly or less / 2-4 a month / 2-3 a week /
# 4+ a week, and Never / Less than monthly / Monthly / Weekly / Daily or almost.
#
# The one judgement call is settled here and nowhere else: once a week is 4.3
# times a month, which puts it inside "2 to 4 times a month" rather than "2 to 3
# times a week". "Daily or almost daily" is taken to start at 5 times a week.
_Q1_BOUNDS = ((0.25, 1), (1.5, 2), (3.5, 3))
_FREQ5_BOUNDS = ((0.23, 1), (0.9, 2), (5.0, 3))

_FREQ_SCALES = {"freq_q1": (_Q1_BOUNDS, 4), "freq5": (_FREQ5_BOUNDS, 4)}

_BOUNDARY_MARGIN = 0.15   # this close to a threshold earns a read-back


def per_week(value: float, per: str | None) -> float | None:
    """Convert a rate to times per week, or None if the period is unrecognised."""
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
    """Place an extracted rate on a frequency scale.

    Returns None when the extraction does not determine a rate, which leaves the
    turn unclear so the person is asked rather than guessed at.
    """
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


# Edges between the 1-2 / 3-4 / 5-6 / 7-9 / 10+ buckets, placed between whole
# drinks so that a count never lands exactly on a threshold.
_Q2_EDGES = (2.5, 4.5, 6.5, 9.5)


def _q2_code(drinks: float) -> int:
    """Which quantity bucket a number of standard drinks falls in."""
    for code, edge in enumerate(_Q2_EDGES):
        if drinks < edge:
            return code
    return 4


def standard_drinks(value: float, unit: str, beverage: str) -> float | None:
    """Convert a spoken amount into standard drinks.

    Returns None whenever the unit or the drink is not in the tables. Nothing is
    approximated: an unknown container has no defensible size, and inventing one
    would move a score.
    """
    unit = " ".join(unit.strip().lower().split())
    # Depluralise only if the plural is not itself a table key, so a unit that
    # genuinely ends in "s" survives.
    if unit.endswith("s") and unit not in ML_PER_UNIT:
        unit = unit[:-1]
    beverage = " ".join((beverage or "").strip().lower().split())
    if unit not in ML_PER_UNIT:
        return None
    ml = ML_PER_UNIT[unit]
    if ml is None:
        # The container's size depends on what is in it, so without the drink
        # there is nothing to look up.
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
    """Place an extracted amount on the quantity scale.

    A count of drinks maps straight across. A volume is converted through the
    tables, and marked assumed so that the conversion is spoken back — which
    also serves as the standard-drink explanation for anyone who declined the
    education unit.
    """
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
    """Compute an option code for an item, or None if the extraction is insufficient.

    Raises ValueError for an unknown coding kind, which is a data error in the
    instrument rather than anything a conversation can produce.
    """
    if coding in _FREQ_SCALES:
        return derive_frequency(coding, value, per)
    if coding == "quantity_drinks":
        return derive_quantity(value, unit, beverage)
    raise ValueError(f"unknown coding kind {coding!r}")


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
    """Read a rate out of a free-text phrase, or None if it is not unambiguous.

    Deliberately narrow. Its output feeds the consistency checks below, which
    accuse the person of contradicting themselves — so a guess here would
    produce a confusing challenge to something they never said.
    """
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


# The most drinking days a "how often" answer can mean, against the fewest
# heavy-drinking days a "how often six or more" answer implies. Claiming more
# heavy days than drinking days is impossible, and worth one gentle read-back.
Q1_MAX_PER_WEEK = {0: 0.0, 1: 0.25, 2: 1.5, 3: 3.5, 4: 14.0}
Q3_MIN_PER_WEEK = {0: 0.0, 1: 0.0, 2: 0.2, 3: 0.9, 4: 5.0}

# How far a conversational frequency may exceed the coded one before the coded
# answer is questioned. The margin absorbs loose speech — "every day" said as a
# figure of speech is common and is not a contradiction.
QF_VS_Q1_MARGIN = 1.9
