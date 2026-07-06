"""Zone feedback templates (P5 acceptance): given a zone, the exact verbatim
study text is returned — and undefined combinations must fail loudly."""

import pytest

from modules.sbirt import templates
from modules.sbirt.instruments import AUDIT, DAST_10

ZONES = ("healthy", "risky", "harmful", "dependent")


def test_every_instrument_zone_has_verbatim_text():
    for key in ("audit", "dast_10"):
        for zone in ZONES:
            text = templates.feedback_text(key, zone)
            assert text.startswith("Based on your answers"), (key, zone)


def test_zone_keys_cover_every_band():
    for ins in (AUDIT, DAST_10):
        assert [b.zone for b in ins.bands] == list(ZONES)


def test_feedback_matches_study_bands():
    # Study doc: AUDIT risky = 8-15; DAST harmful = 6-8 etc. Spot anchors.
    assert "risky levels" in templates.feedback_text("audit", "risky")
    assert "harmful levels" in templates.feedback_text("dast_10", "harmful")
    assert "dependent on alcohol" in templates.feedback_text("audit", "dependent")
    assert "within normal recommended limits" in templates.feedback_text("audit", "healthy")


def test_undefined_zone_raises():
    with pytest.raises(KeyError):
        templates.feedback_text("audit", "made_up_zone")
    with pytest.raises(KeyError):
        templates.feedback_text("crafft", "risky")   # protocol defines none


def test_bi_ask_membership():
    # These five feedback texts end with the BI entry ask; the other three
    # (both healthies + dependent-drug) do not — the machine appends the ask
    # itself for dependent-drug and skips BI for healthy.
    asks = templates.FEEDBACK_ASKS_BI
    assert ("audit", "risky") in asks and ("audit", "harmful") in asks
    assert ("audit", "dependent") in asks
    assert ("dast_10", "risky") in asks and ("dast_10", "harmful") in asks
    assert ("audit", "healthy") not in asks
    assert ("dast_10", "healthy") not in asks
    assert ("dast_10", "dependent") not in asks


def test_parameterized_bi_lines():
    assert templates.bi_likes("alcohol") == "What do you like about alcohol use?"
    assert templates.bi_dislikes("drugs") == "What do you dislike about drug use?"
    assert "stop using drugs" in templates.bi_ruler("drugs")
    assert "stop using alcohol" in templates.bi_recommend("alcohol")
    assert templates.bi_why_not_lower(8) == "Why are you a 8 and not a 1 or 2?"
    assert templates.bi_why_not_higher(5) == "Why are you a 5 and not a 9 or 10?"


def test_all_fixed_utterances_enumeration():
    cat = templates.all_fixed_utterances()
    # 14 FIXED (12 original - the retired three-in-one alcohol.qf + T9
    # screen-permission no-defn variant + T10 declined close + bi.leaves_you
    # folded into FIXED) + 8 feedback + 3 prescreen + 2 preambles + 20 items
    # + 2*5 BI arm lines + 22 ruler variants = 79
    assert len(cat) == 79, f"enumeration changed: {len(cat)} keys"
    assert "alcohol.qf" not in cat, "the triple-question stack is retired"
    for key, text in cat.items():
        assert text and text.strip(), key
    # Every AUDIT/DAST question and all 11 ruler variants are pre-warmable.
    assert "audit.item.9" in cat and "dast_10.item.0" in cat
    assert all(f"bi.why_not_lower.{v}" in cat for v in range(11))
