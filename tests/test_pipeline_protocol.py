"""End-to-end protocol wiring through the REAL Pipeline (P4 acceptance,
migrated to the T6 engine + llm.turn NLU).

Runs only where the heavy stack imports (the float conda env); GPU renders and
network LLM calls are stubbed, everything else — deterministic pre-match,
turn validation, generic engine, scoring, histories, video queue — is the
real code path.
"""

import json
import os
import queue

import pytest

pytest.importorskip("funasr")
pytest.importorskip("torch")

import modules.pipeline as P
from modules import llm
from modules.sbirt import runtime
from modules.sbirt.turn import TurnOut, expected_item, validate

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures")


def load(name):
    with open(os.path.join(FIXTURES, name), encoding="utf-8") as f:
        return json.load(f)


def scripted_turn(user_text, expect, *, ask_text, history, patient=None,
                  facts=None):
    """Deterministic stand-in for llm.turn: REAL pre-match + REAL validate,
    canned semantics for what would need the network. Open answers surface as
    text and let validate() route them onto the asked slot — exactly the
    production behavior for a one-slot-at-a-time answer."""
    t = user_text.strip().lower()
    if expect.kind == "consent":
        if t.startswith(("yes", "sure", "ok", "fine")):
            out = TurnOut(action="answer", code=1)
        elif t.startswith("no"):
            out = TurnOut(action="answer", code=0)
        else:
            out = TurnOut(action="unclear", reply="")
    elif expect.kind == "option":
        pre = llm._prematch_option(expected_item(expect).options, user_text)
        out = (TurnOut(action="answer", code=pre) if pre is not None
               else TurnOut(action="unclear", reply=""))
    elif expect.kind == "number":
        got = llm.code_number(user_text)
        out = (TurnOut(action="answer", code=got["value"])
               if "value" in got else TurnOut(action="unclear", reply=""))
    else:
        out = TurnOut(action="answer", text=user_text)
    return validate(out, expect)


@pytest.fixture()
def pipeline(monkeypatch):
    # No GPU / no network: renders return fake paths; NLU turn scripted.
    monkeypatch.setattr(P, "ensure_fixed_clip",
                        lambda text, path: "/fake/" + os.path.basename(path))
    monkeypatch.setattr(P.tts, "synthesize", lambda text, out, ev=None: "/fake.wav")
    monkeypatch.setattr(P.avatar, "generate_video", lambda *a, **k: "/fake.mp4")
    monkeypatch.setattr(P.llm, "phrase_utterance",
                        lambda instruction, history, patient=None: "[reflection]")
    monkeypatch.setattr(P.llm, "extract_patient_facts", lambda h: {})
    monkeypatch.setattr(P.llm, "turn", scripted_turn)
    # Full-counselor path must NEVER run in a non-crisis protocol session.
    def no_full_llm(*a, **k):
        raise AssertionError("full LLM synthesis ran during a protocol turn")
    monkeypatch.setattr(P.llm, "chat_stream", no_full_llm)
    monkeypatch.setattr(P.llm, "chat", no_full_llm)
    p = P.Pipeline()
    return p


def drain(p):
    """Pop everything currently queued (up to the end-of-response marker)."""
    spoken = []
    while True:
        try:
            item = p.video_queue.get_nowait()
        except queue.Empty:
            break
        if item is None:
            break
        spoken.append(item["sentence"])
    return spoken


def say_turn(p, text):
    """Synchronous user turn via the text path; returns spoken sentences."""
    p.cancel_event.clear()
    p._turn += 1
    p._process_text(text, p._turn)
    return drain(p)


def test_full_alcohol_bi_protocol(pipeline):
    p = pipeline
    fix = load("alcohol_bi_case.json")
    p.start_greeting()
    import time
    for _ in range(50):
        if p.clinical.node == "consent" and p.chat_history:
            break
        time.sleep(0.05)
    greeting = drain(p)          # the fixed greeting clip queued by the thread
    assert greeting and greeting[0].startswith("Hello, I am an AI assistant")

    # Consent -> 3 pre-screens (exact option labels: coded WITHOUT any LLM).
    assert say_turn(p, "yes, that's fine") and p.clinical.node == "prescreen.tobacco"
    say_turn(p, "no")
    say_turn(p, "within the last year")
    spoken = say_turn(p, "none")
    # Q/F is now a composed slot-ask: one question at a time, three slots.
    assert p.clinical.node == "alcohol.qf"
    assert p.clinical.expect.missing == ("drink", "amount", "frequency")
    assert spoken[-1] == "[reflection]", "composed ask, not a fixed 3-in-1"

    # The two-breath regression from the field transcript: answers arrive in
    # pieces; the machine holds the SAME ask and only the gap is re-asked.
    say_turn(p, "wine mostly")
    assert p.clinical.node == "alcohol.qf"
    assert p.clinical.expect.missing == ("amount", "frequency")
    say_turn(p, "two or three glasses")
    spoken = say_turn(p, "most evenings")
    assert p.clinical.node == "alcohol.edu.permission"
    assert "wine mostly" in p.clinical.answers["alcohol.qf"]

    spoken = say_turn(p, "sure")                          # edu permission
    assert any("12-ounce beer" in s for s in spoken), "standard-drink education"
    spoken = say_turn(p, "yes")                           # AUDIT permission
    assert any("How often do you have a drink" in s for s in spoken)

    # The ten AUDIT answers, phrased as exact labels (deterministic coding).
    answers = ["2 to 3 times a week", "3 or 4", "less than monthly", "never",
               "less than monthly", "never", "less than monthly", "never",
               "no", "yes, but not in the last year"]
    for a in answers[:-1]:
        say_turn(p, a)
    spoken = say_turn(p, answers[-1])
    a = p.clinical.assessments["audit"]
    assert a.score == 9 and a.zone == "risky", "deterministic score, zero LLM"
    assert any("feedback" in s.lower() for s in spoken)

    spoken = say_turn(p, "yes")                           # feedback permission
    assert any("using alcohol at risky levels" in s for s in spoken)
    say_turn(p, "yes")                                    # BI entry
    say_turn(p, "helps me relax")                         # likes
    spoken = say_turn(p, "spend too much")                # dislikes -> ruler
    assert any("scale from 0 to 10" in s for s in spoken)
    spoken = say_turn(p, "an 8 I guess")                  # ruler (deterministic)
    assert p.clinical.readiness["alcohol"] == 8
    assert any("a 8 and not a 1 or 2" in s for s in spoken)
    say_turn(p, "because I know it costs too much")
    say_turn(p, "weekends are hard")
    spoken = say_turn(p, "I think I'll try cutting back")
    assert p.clinical.node == "closed"
    assert any("Thank you for participating" in s for s in spoken)

    # ONE history: the API view is a derived sliding-window suffix of the
    # chat, and the deterministic score lives in session state so window
    # trimming can never corrupt triage.
    tail = [(m["role"], m["content"]) for m in p._api_window()]
    full = [(m["role"], m["content"]) for m in p.chat_history]
    assert 0 < len(tail) <= P.config.LLM_HISTORY_MAX_MESSAGES
    assert full[-len(tail):] == tail
    assert tail[0][0] == "user", "API window must never start on an assistant turn"


def test_unclear_answer_clarifies_and_does_not_advance(pipeline, monkeypatch):
    p = pipeline
    runtime.start(p.clinical)
    say_turn(p, "yes")            # consent
    node_before = p.clinical.node
    assert node_before == "prescreen.tobacco"
    # A fuzzy answer with the NLU forced unclear (no reply -> deterministic
    # re-ask through the bounded LLMSay fallback):
    monkeypatch.setattr(P.llm, "turn",
                        lambda *a, **k: TurnOut(action="unclear", reply=""))
    spoken = say_turn(p, "well, you know how it is")
    assert p.clinical.node == node_before, "ambiguity must not move the machine"
    assert spoken == ["[reflection]"], "one bounded clarification is spoken"
    # Machine still accepts a clean answer afterwards.
    monkeypatch.setattr(P.llm, "turn", scripted_turn)
    say_turn(p, "no")
    assert p.clinical.node == "prescreen.alcohol"


def test_user_question_is_answered_and_machine_holds(pipeline, monkeypatch):
    """毛病3: the person asks US something mid-protocol — the reply answers
    THEM (from state facts) and the machine stays exactly where it was."""
    p = pipeline
    runtime.start(p.clinical)
    say_turn(p, "yes")
    node_before = p.clinical.node
    monkeypatch.setattr(P.llm, "turn", lambda *a, **k: TurnOut(
        action="question",
        reply="We haven't gone over that yet. Do you smoke or use tobacco?"))
    spoken = say_turn(p, "have we ever discussed the standard drink definition")
    assert p.clinical.node == node_before, "a question must not move the machine"
    assert spoken == ["We haven't gone over that yet. Do you smoke or use tobacco?"]
    monkeypatch.setattr(P.llm, "turn", scripted_turn)
    say_turn(p, "no")
    assert p.clinical.node == "prescreen.alcohol"


def test_continuation_is_absorbed_not_reasked(pipeline, monkeypatch):
    """毛病2: a late addition lands in the PREVIOUS capture; the pending gate
    is re-posed by the reply, never duplicated by the machine."""
    p = pipeline
    runtime.start(p.clinical)
    for text in ("yes", "no", "within the last year", "none",
                 "whiskey", "twelve ounces", "hardly ever"):
        say_turn(p, text)
    assert p.clinical.node == "alcohol.edu.permission"
    monkeypatch.setattr(P.llm, "turn", lambda *a, **k: TurnOut(
        action="continuation", slots={"frequency": "once a week"},
        reply="Once a week — noted. Would you like that information?"))
    spoken = say_turn(p, "and i drink once every week")
    assert p.clinical.node == "alcohol.edu.permission", "machine must hold"
    assert p.clinical.slots["alcohol.qf"]["frequency"] == "once a week"
    assert spoken == ["Once a week — noted. Would you like that information?"]


def test_answer_ack_is_prepended(pipeline, monkeypatch):
    """T11: when the NLU produces an acknowledgment for an answer, the person
    hears it BEFORE the next protocol content."""
    p = pipeline
    runtime.start(p.clinical)
    monkeypatch.setattr(P.llm, "turn", lambda *a, **k: validate(
        TurnOut(action="answer", code=1, reply="Great, thank you."),
        p.clinical.expect))
    spoken = say_turn(p, "yes that works")
    assert spoken[0] == "Great, thank you."
    assert "tobacco" in spoken[1], "protocol ask follows the acknowledgment"


def test_crisis_phrase_mid_screen_fires_fixed_path(pipeline):
    p = pipeline
    runtime.start(p.clinical)
    say_turn(p, "yes")
    say_turn(p, "no")
    spoken = say_turn(p, "honestly I've been thinking about ending my life")
    assert p.clinical.crisis
    from modules.sbirt import crisis as crisis_mod
    assert spoken and spoken[0] == crisis_mod.RESPONSES["suicide"]


def test_nlu_flagged_crisis_pauses_protocol(pipeline, monkeypatch):
    """T14 union: a crisis the regex net missed but the NLU caught must pause
    the protocol exactly the same way."""
    p = pipeline
    runtime.start(p.clinical)
    say_turn(p, "yes")
    monkeypatch.setattr(P.llm, "turn",
                        lambda *a, **k: TurnOut(action="crisis", reply=""))
    spoken = say_turn(p, "everything is just... very dark lately")
    assert p.clinical.crisis, "NLU crisis flag must pause the protocol"
    assert spoken == ["[reflection]"], "crisis turn speaks via the bounded LLM"


def test_consent_decline_uses_fixed_decline(pipeline, monkeypatch):
    p = pipeline
    recorded = []
    monkeypatch.setattr(P.privacy, "record_consent",
                        lambda key, decision: recorded.append((key, decision)))
    runtime.start(p.clinical)
    spoken = say_turn(p, "no thanks")
    assert p.clinical.node == "declined" and p.clinical.consent == "no"
    assert spoken and spoken[0] == P.config.DECLINE_TEXT
    assert p.ended, "decline ends the session (mic off via poller)"
    assert recorded == [("default", "no")], "decline must hit the audit trail"


def test_generation_budget_full_session(pipeline, monkeypatch):
    """P6/T18 acceptance: with a warm clip cache, a COMPLETE screening session
    performs zero FLOAT renders for fixed content — the only runtime renders
    are the dynamic face (composed slot-asks + the three BI summaries)."""
    p = pipeline
    renders = []
    monkeypatch.setattr(P.avatar, "generate_video",
                        lambda *a, **k: renders.append(1) or "/fake.mp4")
    runtime.start(p.clinical)

    say_turn(p, "yes")
    say_turn(p, "no")
    say_turn(p, "within the last year")
    say_turn(p, "none")
    say_turn(p, "wine")                                   # slot: drink
    say_turn(p, "two glasses")                            # slot: amount
    say_turn(p, "weekly")                                 # slot: frequency
    say_turn(p, "yes")                                    # edu perm
    say_turn(p, "yes")                                    # AUDIT perm
    for a in ["2 to 3 times a week", "3 or 4", "less than monthly", "never",
              "less than monthly", "never", "less than monthly", "never",
              "no", "yes, but not in the last year"]:     # 9 -> risky -> BI runs
        say_turn(p, a)
    say_turn(p, "yes")                                    # feedback perm
    say_turn(p, "yes")                                    # BI entry
    say_turn(p, "relaxing")                               # likes
    say_turn(p, "money")                                  # dislikes -> LLM summary
    say_turn(p, "8")                                      # ruler
    say_turn(p, "reasons")                                # why lower
    say_turn(p, "reasons")                                # why higher -> LLM summary
    say_turn(p, "we'll see")                              # leaves -> LLM reflect
    assert p.clinical.node == "closed"

    # Dynamic face of this walk: 3 composed Q/F slot-asks + 3 LLMSay
    # utterances (decisional-balance summary, ruler summary, closing
    # reflection). Fixed content: 0 renders.
    assert len(renders) == 6, f"FLOAT ran {len(renders)}x; budget is 6"


def test_every_protocol_clip_is_prewarmed(pipeline, monkeypatch):
    """prewarm_fixed_clips must cover exactly the machine-emittable keys."""
    from modules.sbirt import templates
    warmed = []
    monkeypatch.setattr(P, "ensure_fixed_clip",
                        lambda text, path: warmed.append(path) or path)
    P.prewarm_fixed_clips()
    warmed_names = {os.path.basename(w) for w in warmed}
    for key in templates.all_fixed_utterances():
        expected = os.path.basename(P.protocol_clip_path(key))
        assert expected in warmed_names, f"not pre-warmed: {key}"
    assert os.path.basename(P.config.GREETING_VIDEO_PATH) in warmed_names
    assert "crisis_suicide.mp4" in warmed_names
