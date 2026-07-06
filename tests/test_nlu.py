"""Constrained NLU (P4 -> T4): deterministic pre-pass, strict JSON handling,
never-guess semantics — now through the ONE llm.turn() call whose output is
gated by turn.validate. LLM calls are faked — no network in tests."""

import json
from types import SimpleNamespace

import pytest

pytest.importorskip("openai")

from modules import llm
from modules.sbirt.instruments import AUDIT, DAST_10
from modules.sbirt.runtime import Expect
from modules.sbirt.turn import TurnOut


def fake_client(replies):
    """A stand-in OpenAI client yielding canned completions in order."""
    replies = list(replies)
    calls = []

    def create(**kwargs):
        calls.append(kwargs)
        content = replies.pop(0)
        return SimpleNamespace(choices=[SimpleNamespace(
            message=SimpleNamespace(content=content))])

    client = SimpleNamespace(chat=SimpleNamespace(
        completions=SimpleNamespace(create=create)))
    client.calls = calls
    return client


def forbid_llm(monkeypatch):
    def boom():
        raise AssertionError("LLM must not be called on a deterministic path")
    monkeypatch.setattr(llm, "_client", boom)


def turn_kwargs(**over):
    kw = dict(ask_text="q?", history=[], patient=None, facts={})
    kw.update(over)
    return kw


# ---------------- deterministic pre-match (no LLM at all) ----------------

def test_prematch_exact_label():
    opts = AUDIT.items[0].options
    assert llm._prematch_option(opts, "2 to 3 times a week") == 3
    assert llm._prematch_option(opts, "  NEVER ") == 0
    assert llm._prematch_option(opts, "a couple times, I guess") is None


def test_prematch_binary_yes_no():
    opts = DAST_10.items[0].options       # (No, Yes)
    assert llm._prematch_option(opts, "yeah") == 1
    assert llm._prematch_option(opts, "Yes I have") == 1
    assert llm._prematch_option(opts, "nope") == 0
    assert llm._prematch_option(opts, "never") == 0
    assert llm._prematch_option(opts, "well sometimes") is None


def test_prematch_binary_shortcut_needs_short_utterance():
    # "yes but ..." may carry a question or a caveat — the LLM must see it.
    opts = DAST_10.items[0].options
    assert llm._prematch_option(opts, "yes but only weed on weekends") is None


def test_prematch_binary_shortcut_only_for_no_yes_items():
    # AUDIT item 9 options are timeframed (No / Yes-not-last-year / Yes-last-
    # year): a bare "yes" must NOT shortcut — the timeframe is undetermined.
    opts = AUDIT.items[8].options
    assert llm._prematch_option(opts, "yes") is None
    assert llm._prematch_option(opts, "no") == 0   # exact label match is fine


# ---------------- readiness ruler (deterministic only) ----------------

@pytest.mark.parametrize("text,expected", [
    ("8", {"value": 8}),
    ("I'd say a five", {"value": 5}),
    ("probably ten", {"value": 10}),
    ("zero honestly", {"value": 0}),
    ("a 7, maybe 8", {"status": "AMBIGUOUS"}),
    ("I don't know", {"status": "AMBIGUOUS"}),
    ("twenty", {"status": "AMBIGUOUS"}),
    ("about 15", {"status": "AMBIGUOUS"}),
])
def test_code_number(text, expected):
    assert llm.code_number(text) == expected


# ---------------- turn(): deterministic fast paths (zero LLM) ----------------

def test_turn_consent_prepass_no_llm(monkeypatch):
    forbid_llm(monkeypatch)
    exp = Expect("consent", ask_key="alcohol.edu.permission")
    assert llm.turn("Sure.", exp, **turn_kwargs()).code == 1
    assert llm.turn("no thank you", exp, **turn_kwargs()).code == 0
    out = llm.turn("yes", exp, **turn_kwargs())
    assert out.action == "answer" and out.reply == ""


def test_turn_option_prepass_no_llm(monkeypatch):
    forbid_llm(monkeypatch)
    exp = Expect("option", instrument="audit", item_index=0)
    out = llm.turn("monthly or less", exp, **turn_kwargs())
    assert out.action == "answer" and out.code == 1


def test_turn_number_prepass_no_llm(monkeypatch):
    forbid_llm(monkeypatch)
    exp = Expect("number", ask_key="bi.ruler")
    out = llm.turn("an 8 I guess", exp, **turn_kwargs())
    assert out.action == "answer" and out.code == 8


# ---------------- turn(): strict-JSON LLM path ----------------

def _j(**kw):
    base = {"action": "answer", "code": None, "slots": {}, "text": None,
            "reply": ""}
    base.update(kw)
    return json.dumps(base)


def test_turn_valid_option_answer(monkeypatch):
    monkeypatch.setattr(llm, "_client", lambda: fake_client(
        [_j(code=2, reply="A few times a month — thanks.")]))
    exp = Expect("option", instrument="audit", item_index=0)
    out = llm.turn("a few times a month", exp, **turn_kwargs())
    assert out.action == "answer" and out.code == 2
    assert out.reply.startswith("A few times")


def test_turn_out_of_range_code_downgrades_to_unclear(monkeypatch):
    monkeypatch.setattr(llm, "_client", lambda: fake_client(
        [_j(code=9, reply="Roughly how often?")]))
    exp = Expect("option", instrument="audit", item_index=0)
    out = llm.turn("whatever", exp, **turn_kwargs())
    assert out.action == "unclear" and out.code is None
    assert out.reply == "Roughly how often?", "model's re-ask is kept"


def test_turn_no_timeframe_stays_unclear(monkeypatch):
    # The case-card AUDIT item-10 rule: "told me once or twice ... sometimes"
    # must clarify, never guess a timeframed option.
    monkeypatch.setattr(llm, "_client", lambda: fake_client(
        [_j(action="unclear", reply="Was that within the last year?")]))
    exp = Expect("option", instrument="audit", item_index=9)
    out = llm.turn("my spouse has told me once or twice that I drink too much",
                   exp, **turn_kwargs())
    assert out.action == "unclear" and out.code is None


def test_turn_retries_invalid_json_then_succeeds(monkeypatch):
    client = fake_client(["not json at all", _j(code=1, reply="Okay.")])
    monkeypatch.setattr(llm, "_client", lambda: client)
    exp = Expect("option", instrument="audit", item_index=0)
    out = llm.turn("monthly-ish", exp, **turn_kwargs())
    assert out.action == "answer" and out.code == 1
    assert len(client.calls) == 2


def test_turn_llm_error_is_unclear_never_guess(monkeypatch):
    def create(**kwargs):
        raise RuntimeError("network down")
    client = SimpleNamespace(chat=SimpleNamespace(
        completions=SimpleNamespace(create=create)))
    monkeypatch.setattr(llm, "_client", lambda: client)
    exp = Expect("option", instrument="audit", item_index=0)
    out = llm.turn("some answer", exp, **turn_kwargs())
    assert out.action == "unclear" and out.reply == "", \
        "NLU failure must clarify, never guess"


def test_turn_question_action_passthrough(monkeypatch):
    monkeypatch.setattr(llm, "_client", lambda: fake_client(
        [_j(action="question", code=1,
            reply="We haven't gone over it yet. Shall we continue?")]))
    exp = Expect("consent", ask_key="alcohol.screen.permission")
    out = llm.turn("have we ever discussed the standard drink definition?",
                   exp, **turn_kwargs())
    assert out.action == "question"
    assert out.code is None, "a question must carry no coded answer"
    assert "haven't gone over it" in out.reply


def test_turn_slot_extraction_filters_undeclared(monkeypatch):
    monkeypatch.setattr(llm, "_client", lambda: fake_client(
        [_j(slots={"drink": "whiskey", "amount": "12 oz", "bogus": "x"},
            reply="Whiskey, twelve ounces — got it.")]))
    exp = Expect("open", ask_key="alcohol.qf",
                 slots=("drink", "amount", "frequency"),
                 missing=("drink", "amount", "frequency"))
    out = llm.turn("i like whiskey and i drink twelve oun per time",
                   exp, **turn_kwargs())
    assert out.action == "answer"
    assert out.slots == {"drink": "whiskey", "amount": "12 oz"}


def test_turn_prompt_carries_ask_and_facts(monkeypatch):
    client = fake_client([_j(code=1)])
    monkeypatch.setattr(llm, "_client", lambda: client)
    exp = Expect("consent", ask_key="alcohol.screen.permission")
    llm.turn("hmm okay I suppose so", exp, **turn_kwargs(
        ask_text="May I ask you a few more questions?",
        facts={"standard_drink_definition_discussed": False}))
    sent = client.calls[0]["messages"][0]["content"]
    assert "May I ask you a few more questions?" in sent
    assert '"standard_drink_definition_discussed": false' in sent
