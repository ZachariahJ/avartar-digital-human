"""Concurrency hardening (P8): superseded turns must never advance the
clinical machine; concurrent sessions never cross; stuck sessions get reaped."""

import json
import os
import threading
import time

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
    """Deterministic NLU stand-in (same shape as test_pipeline_protocol's)."""
    t = user_text.strip().lower()
    if expect.kind == "consent":
        code = (1 if t.startswith(("yes", "sure"))
                else (0 if t.startswith("no") else None))
        out = (TurnOut(action="answer", code=code) if code is not None
               else TurnOut(action="unclear", reply=""))
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


def stub_pipeline(monkeypatch, turn_fn=None):
    monkeypatch.setattr(P, "ensure_fixed_clip", lambda text, path: "/fake.mp4")
    monkeypatch.setattr(P.tts, "synthesize", lambda text, out, ev=None: "/fake.wav")
    monkeypatch.setattr(P.avatar, "generate_video", lambda *a, **k: "/fake.mp4")
    monkeypatch.setattr(P.llm, "phrase_utterance", lambda *a, **k: "[r]")
    monkeypatch.setattr(P.llm, "extract_patient_facts", lambda h: {})
    monkeypatch.setattr(P.privacy, "record_consent", lambda *a: True)
    monkeypatch.setattr(P.llm, "turn", turn_fn or scripted_turn)
    return P.Pipeline()


def test_superseded_turn_cannot_advance_machine(monkeypatch):
    """Turn A (slow NLU, would answer 'yes') is superseded by turn B ('no').
    Protocol turns are serialized: B queues on the lock while A stalls; when A
    resumes it must notice it was superseded and NEVER advance the machine —
    the final state is B's decline, not A's consent-yes. Without the lock +
    staleness re-check, A's advance() lands after B's and corrupts the flow."""
    barrier = threading.Event()

    def slow_then_fast(user_text, expect, **kw):
        if user_text == "yes slow":
            barrier.wait(timeout=5)      # A stalls inside its NLU call
            return TurnOut(action="answer", code=1)
        return TurnOut(action="answer", code=0)

    p = stub_pipeline(monkeypatch, turn_fn=slow_then_fast)
    runtime.start(p.clinical)

    p.on_speech_end_text("yes slow")     # turn A: enters the lock, stalls
    time.sleep(0.2)
    p.on_speech_end_text("no way")       # turn B: bumps the turn, queues on the lock
    time.sleep(0.2)
    assert p.clinical.node == "consent", "B must wait while A holds the lock"

    barrier.set()                        # release A: it must see it's stale
    for _ in range(50):                  # wait for both turns to settle
        if p.clinical.node == "declined":
            break
        time.sleep(0.05)
    # A ('yes') aborted without advancing; B ('no') decided the final state.
    # Had A advanced, the node would be prescreen.tobacco with consent 'yes'.
    assert p.clinical.node == "declined", "stale turn advanced the machine!"
    assert p.clinical.consent == "no"


def test_concurrent_sessions_never_cross(monkeypatch):
    """Two patients (alcohol-BI vs drug-BI cases) screened simultaneously on
    separate Pipelines: codes, scores and zones must stay per-session."""
    monkeypatch.setattr(P, "ensure_fixed_clip", lambda text, path: "/fake.mp4")
    monkeypatch.setattr(P.tts, "synthesize", lambda text, out, ev=None: "/fake.wav")
    monkeypatch.setattr(P.avatar, "generate_video", lambda *a, **k: "/fake.mp4")
    monkeypatch.setattr(P.llm, "phrase_utterance", lambda *a, **k: "[r]")
    monkeypatch.setattr(P.llm, "extract_patient_facts", lambda h: {})
    monkeypatch.setattr(P.privacy, "record_consent", lambda *a: True)
    monkeypatch.setattr(P.llm, "turn", scripted_turn)

    alcohol = ["yes", "no", "within the last year", "none",
               "wine", "two glasses", "weekly",          # Q/F slots, one each
               "yes", "yes",
               "2 to 3 times a week", "3 or 4", "less than monthly", "never",
               "less than monthly", "never", "less than monthly", "never",
               "no", "yes, but not in the last year",
               "yes", "yes", "relax", "money", "8", "a", "b", "c"]
    drugs = ["yes", "no", "never or more than a year ago", "one or more",
             "crack", "every day", "yes",
             "yes", "yes", "yes", "yes", "no", "yes", "yes", "yes", "no", "no",
             "yes", "yes", "relax", "money", "5", "a", "b", "c"]

    p1, p2 = P.Pipeline(audit_key="s1"), P.Pipeline(audit_key="s2")
    runtime.start(p1.clinical)
    runtime.start(p2.clinical)

    def drive(p, script):
        for text in script:
            p.cancel_event.clear()
            p._turn += 1
            p._process_text(text, p._turn)

    t1 = threading.Thread(target=drive, args=(p1, alcohol))
    t2 = threading.Thread(target=drive, args=(p2, drugs))
    t1.start(); t2.start(); t1.join(10); t2.join(10)

    a1, a2 = p1.clinical.assessments, p2.clinical.assessments
    assert set(a1) == {"audit"} and set(a2) == {"dast_10"}, "instruments crossed"
    assert a1["audit"].score == 9 and a1["audit"].zone == "risky"
    assert a2["dast_10"].score == 7 and a2["dast_10"].zone == "harmful"
    assert p1.clinical.readiness == {"alcohol": 8}
    assert p2.clinical.readiness == {"drugs": 5}
    # No transcript bleed between chat histories.
    h1 = " ".join(m["content"] for m in p1.chat_history)
    assert "crack" not in h1


def test_stuck_session_is_reaped(monkeypatch):
    import main
    monkeypatch.setattr(main, "sessions", {}, raising=True)
    s = main.get_or_create_session("stuck-sid")
    s.pipeline.state = "speaking"          # stranded mid-response
    s.state_clients.clear()
    s.empty_since = time.time() - 9999     # clients long gone
    main._reap_idle_sessions()
    assert "stuck-sid" not in main.sessions, \
        "session stuck in 'speaking' must still be reaped once clients are gone"
    assert s.pipeline.state == "idle", "reaper must cancel in-flight work"


def test_default_session_never_reaped(monkeypatch):
    import main
    monkeypatch.setattr(main, "sessions", {}, raising=True)
    s = main.get_or_create_session("default")
    s.empty_since = time.time() - 9999
    main._reap_idle_sessions()
    assert "default" in main.sessions
