"""Integrity of the declarative protocol program (flow.py) and the per-turn
NLU contract (turn.py) — pure, no LLM, no pipeline.

These lock the DATA layer the generic engine executes: every jump target
resolves, every fixed ask exists in templates, the @resolver vocabulary is
closed (the engine must implement exactly this set), and validate() never
lets an illegal answer advance anything.
"""

from types import SimpleNamespace

import pytest

from modules.sbirt import flow, templates
from modules.sbirt.flow import (Ask, End, Gate, Label, PROTOCOL, LABELS,
                                RunItems, Route, Tell)
from modules.sbirt.turn import TurnOut, validate


# --------------------------- flow program integrity ---------------------------

def test_labels_unique():
    names = [s.name for s in PROTOCOL if isinstance(s, Label)]
    assert len(names) == len(set(names))


def test_static_jump_targets_resolve():
    for step in PROTOCOL:
        if isinstance(step, Gate):
            assert step.on_no in LABELS, f"Gate {step.key}: {step.on_no}"


def test_route_functions_return_known_labels():
    """Exercise every deterministic route over its full branch space."""
    # _after_prescreen: nothing positive / alcohol / drugs / both.
    for pre, want in [({}, "close.all_negative"),
                      ({"alcohol": 1}, "arm.next"),
                      ({"drugs": 1}, "arm.next"),
                      ({"alcohol": 1, "drugs": 1, "tobacco": 1}, "arm.next")]:
        s = SimpleNamespace(prescreen=pre, arms=None)
        assert flow._after_prescreen(s) == want
        assert want in LABELS
    # tobacco alone opens NO arm (no tobacco instrument in this study).
    s = SimpleNamespace(prescreen={"tobacco": 1}, arms=None)
    assert flow._after_prescreen(s) == "close.all_negative"

    # _next_arm pops in protocol order, then closes.
    s = SimpleNamespace(arms=["alcohol", "drugs"], arm=None)
    assert flow._next_arm(s) == "alcohol" and s.arm == "alcohol"
    assert flow._next_arm(s) == "drugs" and s.arm == "drugs"
    assert flow._next_arm(s) == "close.completed"
    for name in ("alcohol", "drugs", "close.completed"):
        assert name in LABELS

    # _after_feedback: healthy / asks-BI zone / dependent-drug (no ask in text).
    mk = lambda arm, zone: SimpleNamespace(
        arm=arm, assessments={flow.ARM_INSTRUMENT[arm]:
                              SimpleNamespace(zone=zone)})
    assert flow._after_feedback(mk("alcohol", "healthy")) == "arm.next"
    assert flow._after_feedback(mk("alcohol", "risky")) == "bi.included"
    assert flow._after_feedback(mk("drugs", "dependent")) == "bi.ask"
    for name in ("bi.included", "bi.ask"):
        assert name in LABELS

    # lambda routes.
    for step in PROTOCOL:
        if isinstance(step, Route) and step.fn.__name__ == "<lambda>":
            assert step.fn(None) in LABELS


def test_fixed_asks_exist_in_templates():
    """Every non-@ fixed utterance the program can speak has study text."""
    for step in PROTOCOL:
        if isinstance(step, Gate) and not step.ask_included \
                and not step.key.startswith("@"):
            assert step.key in templates.FIXED, step.key
        if isinstance(step, Ask) and step.ask == "fixed" \
                and not step.key.startswith("@"):
            assert step.key in templates.FIXED, step.key
        if isinstance(step, Tell) and not step.unit.startswith("@"):
            assert (step.unit in templates.FIXED
                    or step.unit in templates.POINTS_UNITS), step.unit


def test_resolver_vocabulary_is_closed():
    """The engine must implement EXACTLY these @resolvers — a new @name in the
    program without an engine resolver (or vice versa) fails here first."""
    used = set()
    for step in PROTOCOL:
        for name in (getattr(step, "unit", ""), getattr(step, "key", ""),
                     getattr(step, "close", "")):
            if isinstance(name, str) and name.startswith("@"):
                used.add(name)
    assert used == {"@alcohol.screen.permission", "@feedback",
                    "@bi.permission", "@bi.likes", "@bi.dislikes",
                    "@bi.recommend", "@bi.ruler", "@bi.why_not_lower",
                    "@bi.why_not_higher", "@close"}


def test_close_variants():
    """T10: a hard permission decline earns the no-more-questions close; an
    education-only decline still earns the study close."""
    assert flow.close_unit(SimpleNamespace(declined=[])) == "close"
    assert flow.close_unit(SimpleNamespace(
        declined=["alcohol.edu.permission"])) == "close"
    assert flow.close_unit(SimpleNamespace(
        declined=["alcohol.screen.permission"])) == "close.declined"
    assert flow.close_unit(SimpleNamespace(
        declined=["alcohol.edu.permission",
                  "drugs.feedback.permission"])) == "close.declined"
    for key in ("close", "close.declined"):
        assert key in templates.FIXED


def test_conditional_screen_permission_variants():
    """T9: the no-education variant must not claim a discussed definition."""
    with_defn = templates.FIXED["alcohol.screen.permission"]
    without = templates.FIXED["alcohol.screen.permission.no_defn"]
    assert "just discussed" in with_defn
    assert "just discussed" not in without
    assert "standard drink" not in without


def test_ends_cover_both_terminals():
    ends = [s for s in PROTOCOL if isinstance(s, End)]
    assert {e.node for e in ends} == {"closed", "declined"}
    # The consent-declined end is silent (pipeline speaks the fixed goodbye).
    assert [e for e in ends if e.node == "declined"][0].close == ""


def test_qf_is_slot_ask_not_triple_stack():
    """毛病5: the source's three-question stack is now one slot-ask; the
    old three-in-one fixed line is no longer part of the program."""
    qf = [s for s in PROTOCOL
          if isinstance(s, Ask) and s.key == "alcohol.qf"][0]
    assert qf.slots == ("drink", "amount", "frequency")
    assert dict(qf.slot_points).keys() == set(qf.slots)
    for step in PROTOCOL:
        assert not (isinstance(step, Tell) and step.unit == "alcohol.qf")


# --------------------------- turn.validate gatekeeping ---------------------------

def _expect(kind, **kw):
    base = dict(instrument=None, item_index=None, ask_key=None,
                slots=(), missing=())
    base.update(kw)
    return SimpleNamespace(kind=kind, **base)


def test_validate_consent():
    ok = validate(TurnOut(action="answer", code=1, reply="ok"),
                  _expect("consent"))
    assert ok.action == "answer" and ok.code == 1
    bad = validate(TurnOut(action="answer", code=3), _expect("consent"))
    assert bad.action == "unclear" and bad.code is None
    none = validate(TurnOut(action="answer"), _expect("consent"))
    assert none.action == "unclear"


def test_validate_option_range_uses_real_instrument():
    exp = _expect("option", instrument="audit", item_index=8)  # 3 options
    assert validate(TurnOut(action="answer", code=2), exp).action == "answer"
    assert validate(TurnOut(action="answer", code=3), exp).action == "unclear"
    exp0 = _expect("option", instrument="prescreen", item_index=0)
    assert validate(TurnOut(action="answer", code=1), exp0).action == "answer"


def test_validate_number():
    exp = _expect("number")
    assert validate(TurnOut(action="answer", code=10), exp).action == "answer"
    assert validate(TurnOut(action="answer", code=11), exp).action == "unclear"
    assert validate(TurnOut(action="answer", code=-1), exp).action == "unclear"


def test_validate_open_single_and_slots():
    exp = _expect("open")
    assert validate(TurnOut(action="answer", text="whiskey"),
                    exp).action == "answer"
    assert validate(TurnOut(action="answer"), exp).action == "unclear"

    sl = _expect("open", slots=("drink", "amount", "frequency"),
                 missing=("amount", "frequency"))
    out = validate(TurnOut(action="answer",
                           slots={"amount": "12 oz", "bogus": "x"}), sl)
    assert out.action == "answer" and out.slots == {"amount": "12 oz"}
    # Unsplit whole-text answer lands in the first missing slot.
    txt = validate(TurnOut(action="answer", text="about twelve ounces"), sl)
    assert txt.action == "answer" and "amount" in txt.slots
    empty = validate(TurnOut(action="answer", slots={"bogus": "x"}), sl)
    assert empty.action == "unclear"


def test_validate_never_advances_at_end():
    out = validate(TurnOut(action="answer", code=1), _expect("end"))
    assert out.action == "unclear"


def test_non_answer_actions_pass_through_clean():
    q = validate(TurnOut(action="question", code=2, text="x",
                         reply="We haven't discussed it yet."),
                 _expect("consent"))
    assert q.action == "question" and q.code is None and q.text is None
    assert q.reply == "We haven't discussed it yet."
    cont = validate(TurnOut(action="continuation",
                            slots={"frequency": "weekly"}),
                    _expect("consent"))
    assert cont.action == "continuation"          # payload kept for absorption
    assert cont.slots == {"frequency": "weekly"}


def test_reply_downgrade_keeps_model_reply():
    """A downgraded answer keeps its reply — usually already a usable
    clarification — so the engine can hold-and-re-ask with it."""
    out = validate(TurnOut(action="answer", code=99, reply="Could you say "
                           "roughly how often — never, monthly, or weekly?"),
                   _expect("option", instrument="audit", item_index=0))
    assert out.action == "unclear" and out.reply.startswith("Could you")


def test_turnout_rejects_extra_fields():
    with pytest.raises(Exception):
        TurnOut(action="answer", zone="risky")
