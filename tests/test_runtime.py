"""Generic turn engine over the declarative protocol: given coded answer
sequences, the protocol must walk the correct branches deterministically
(P3 acceptance, migrated to the T6 engine — same scenarios, TurnOut driver)."""

import json
import os

import pytest

from modules.sbirt import runtime, templates
from modules.sbirt.runtime import ClinicalSession, LLMSay, Say, Speak
from modules.sbirt.turn import TurnOut, validate

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures")


def load(name):
    with open(os.path.join(FIXTURES, name), encoding="utf-8") as f:
        return json.load(f)


def new_session():
    s = ClinicalSession()
    runtime.start(s)
    return s


def say_keys(step):
    return [u.key for u in step.utterances if isinstance(u, Say)]


def adv(s, kind, value):
    """Old-driver shim: build the answer TurnOut the NLU would and push it
    through the SAME validate gate the pipeline uses (so e.g. a bare text
    answer to a slot-ask lands on the asked slot, exactly like production)."""
    if kind == "consent":
        out = TurnOut(action="answer", code=1 if value == "yes" else 0)
    elif kind in ("option", "number"):
        out = TurnOut(action="answer", code=value)
    else:
        out = TurnOut(action="answer", text=value)
    return runtime.advance(s, validate(out, s.expect))


def qf(s, drink="wine", amount="2-3 glasses", frequency="most evenings"):
    """Fill the alcohol Q/F slot-ask in one breath."""
    return runtime.advance(s, validate(TurnOut(action="answer", slots={
        "drink": drink, "amount": amount, "frequency": frequency}), s.expect))


def drive_prescreen(s, tobacco, alcohol, drugs):
    step = adv(s, "consent", "yes")
    assert say_keys(step) == ["prescreen.tobacco"]
    step = adv(s, "option", tobacco)
    assert say_keys(step) == ["prescreen.alcohol"]
    step = adv(s, "option", alcohol)
    assert say_keys(step) == ["prescreen.drugs"]
    return adv(s, "option", drugs)


def drive_screening(s, instrument_key, codes):
    """Answer instrument items in the order the machine asks them."""
    step = s.last_step
    while step.expect.kind == "option" and step.expect.instrument == instrument_key:
        idx = step.expect.item_index
        step = adv(s, "option", codes[idx])
    return step


def test_alcohol_bi_case_full_walk():
    fix = load("alcohol_bi_case.json")
    s = new_session()

    # Pre-screen: tobacco NO, alcohol positive, drugs none -> alcohol arm only.
    step = drive_prescreen(s, fix["prescreen"]["tobacco"],
                           fix["prescreen"]["alcohol"], fix["prescreen"]["drugs"])
    assert s.arms == [] and s.arm == "alcohol"
    # Q/F is a composed slot-ask now: one LLM-phrased question, three slots.
    assert step.expect.kind == "open" and step.expect.ask_key == "alcohol.qf"
    assert step.expect.missing == ("drink", "amount", "frequency")
    assert isinstance(step.utterances[-1], LLMSay)

    # All three slots in one breath -> education permission -> education.
    step = qf(s)
    assert say_keys(step) == ["alcohol.edu.permission"]
    step = adv(s, "consent", "yes")
    assert say_keys(step) == ["alcohol.edu.standard_drink", "alcohol.edu.limits",
                              "alcohol.screen.permission"]

    # AUDIT permission -> preamble + item 0; then all 10 items in order (no skips).
    step = adv(s, "consent", "yes")
    assert say_keys(step) == ["audit.preamble", "audit.item.0"]
    asked = [0]
    while step.expect.kind == "option":
        idx = step.expect.item_index
        step = adv(s, "option", fix["codes"][idx])
        if step.expect.kind == "option":
            asked.append(step.expect.item_index)
    assert asked == list(range(10)), "AUDIT must ask all 10 items in order"

    # Deterministic assessment recorded; zone drives the branch.
    a = s.assessments["audit"]
    assert a.score in fix["expected_scores"] and a.zone == "risky"

    # Feedback permission -> risky feedback (ends with the BI ask) -> BI.
    assert say_keys(step) == ["alcohol.feedback.permission"]
    step = adv(s, "consent", "yes")
    assert say_keys(step) == ["feedback.audit.risky"]
    assert step.expect.kind == "consent"          # BI entry ask is inside the text

    step = adv(s, "consent", "yes")
    assert say_keys(step) == ["bi.likes.alcohol"]
    step = adv(s, "open", fix["likes"])
    assert say_keys(step) == ["bi.dislikes.alcohol"]
    step = adv(s, "open", fix["dislikes"])
    # Decisional-balance summary is the LLM's ONE bounded job here — grounded
    # in the person's OWN captured words (no discarded answers).
    assert isinstance(step.utterances[0], LLMSay)
    assert fix["likes"] in step.utterances[0].instruction
    assert fix["dislikes"] in step.utterances[0].instruction
    assert say_keys(step) == ["bi.recommend.alcohol", "bi.ruler.alcohol"]
    assert step.expect.kind == "number"

    step = adv(s, "number", fix["readiness"])
    assert s.readiness["alcohol"] == 8
    assert say_keys(step) == [f"bi.why_not_lower.{fix['readiness']}"]
    assert f"a {fix['readiness']} and not a 1 or 2" in step.utterances[0].text
    step = adv(s, "open", "I know I should cut back")
    assert say_keys(step) == [f"bi.why_not_higher.{fix['readiness']}"]
    step = adv(s, "open", "not sure I can on weekends")
    assert isinstance(step.utterances[0], LLMSay)
    assert say_keys(step) == ["bi.leaves_you"]
    step = adv(s, "open", "I guess I'll try cutting down")

    # Single positive arm -> close (nothing hard-declined -> study close).
    assert s.node == "closed" and step.expect.kind == "end"
    assert say_keys(step)[-1] == "close"


def test_qf_slots_fill_across_turns_without_reasking():
    """毛病2/毛病5: a two-breath answer holds the SAME ask and only the
    missing slots get asked — the machine never advances mid-answer and
    never re-asks what was already given."""
    s = new_session()
    drive_prescreen(s, 0, 1, 0)
    assert s.expect.missing == ("drink", "amount", "frequency")

    step = runtime.advance(s, TurnOut(
        action="answer", slots={"drink": "whiskey", "amount": "12 oz"}))
    assert s.node == "alcohol.qf", "machine must hold on the same ask"
    assert step.expect.missing == ("frequency",), "only the gap is re-asked"
    assert isinstance(step.utterances[0], LLMSay)

    step = runtime.advance(s, TurnOut(
        action="answer", slots={"frequency": "once a week"}))
    assert say_keys(step) == ["alcohol.edu.permission"], "now it advances"
    assert "whiskey" in s.answers["alcohol.qf"]
    assert "once a week" in s.answers["alcohol.qf"]


def test_continuation_absorbs_into_previous_capture():
    """毛病2: a late addition while the machine already waits at a gate is
    folded into the previous open capture; the machine does not move."""
    s = new_session()
    drive_prescreen(s, 0, 1, 0)
    qf(s, frequency="sometimes")
    assert s.node == "alcohol.edu.permission"
    runtime.absorb(s, TurnOut(action="continuation",
                              slots={"frequency": "once a week"}))
    assert s.node == "alcohol.edu.permission", "absorb must not move the machine"
    assert s.slots["alcohol.qf"]["frequency"] == "once a week"


def test_screen_permission_variant_tracks_education(T9=True):
    """毛病4: the AUDIT permission only references 'the standard drink
    definition we just discussed' when the education was actually given."""
    # Education accepted -> the referencing study line.
    s = new_session()
    drive_prescreen(s, 0, 1, 0)
    qf(s)
    step = adv(s, "consent", "yes")
    assert "alcohol.edu.standard_drink" in s.covered
    assert say_keys(step)[-1] == "alcohol.screen.permission"

    # Education declined -> the no-reference variant; nothing 'discussed'.
    s2 = new_session()
    drive_prescreen(s2, 0, 1, 0)
    qf(s2)
    step2 = adv(s2, "consent", "no")
    assert "alcohol.edu.standard_drink" not in s2.covered
    assert say_keys(step2) == ["alcohol.screen.permission.no_defn"]
    assert "just discussed" not in step2.utterances[0].text


def test_drug_bi_case_full_walk():
    fix = load("drug_bi_case.json")
    s = new_session()
    step = drive_prescreen(s, 0, 0, fix["prescreen"]["drugs"])
    assert s.arm == "drugs"
    assert say_keys(step) == ["drugs.kind"]

    step = adv(s, "open", "crack and marijuana")
    # Parameterized quantity/frequency question is LLM-phrased, bounded,
    # and grounded in the drugs the person actually named.
    assert len(step.utterances) == 1 and isinstance(step.utterances[0], LLMSay)
    assert "crack and marijuana" in step.utterances[0].instruction
    step = adv(s, "open", "4-5 times per week")
    assert s.answers["drugs.qf"] == "4-5 times per week"
    assert say_keys(step) == ["drugs.screen.permission"]

    step = adv(s, "consent", "yes")
    assert say_keys(step) == ["dast_10.preamble", "dast_10.item.0"]
    step = drive_screening(s, "dast_10", fix["codes"])

    a = s.assessments["dast_10"]
    assert a.score == 7 and a.zone == "harmful"
    assert say_keys(step) == ["drugs.feedback.permission"]
    step = adv(s, "consent", "yes")
    assert say_keys(step) == ["feedback.dast_10.harmful"]

    step = adv(s, "consent", "yes")       # BI entry
    step = adv(s, "open", "relax")         # likes -> dislikes
    step = adv(s, "open", "money")         # -> ruler
    step = adv(s, "number", fix["readiness"])
    assert s.readiness["drugs"] == 5
    adv(s, "open", "a")
    adv(s, "open", "b")
    step = adv(s, "open", "c")
    assert s.node == "closed"
    # Every open answer of the walk landed in state (no discarded turns).
    for key in ("drugs.kind", "drugs.qf", "bi.likes", "bi.dislikes",
                "bi.why_not_lower", "bi.why_not_higher", "bi.leaves_you"):
        assert key in s.answers, f"open capture lost: {key}"


def test_dual_positive_runs_alcohol_then_drugs():
    s = new_session()
    step = drive_prescreen(s, 0, 1, 1)
    assert s.arm == "alcohol" and s.arms == ["drugs"]
    assert step.expect.ask_key == "alcohol.qf"
    # Decline the alcohol screening -> machine moves ON to the drug arm.
    qf(s)
    adv(s, "consent", "no")                # no education
    step = adv(s, "consent", "no")         # decline AUDIT
    assert s.arm == "drugs"
    assert say_keys(step) == ["permission.declined", "drugs.kind"]
    # The audit trail records the CANONICAL permission identity — the T9
    # wording variant is presentation, not a different permission.
    assert "alcohol.screen.permission" in s.declined


def test_declined_screen_earns_no_promise_close():
    """毛病6/T10: ending because permissions were declined must not promise
    'a few more questions' right after 'that's your call'."""
    s = new_session()
    drive_prescreen(s, 0, 1, 0)            # alcohol arm only
    qf(s)
    adv(s, "consent", "no")                # no education
    step = adv(s, "consent", "no")         # decline AUDIT -> nothing left
    assert s.node == "closed"
    assert say_keys(step) == ["permission.declined", "close.declined"]
    text = " ".join(u.text for u in step.utterances)
    assert "few more questions" not in text


def test_audit_q1_never_skips_to_item_9():
    s = new_session()
    drive_prescreen(s, 0, 1, 0)
    qf(s)
    adv(s, "consent", "no")
    step = adv(s, "consent", "yes")        # start AUDIT
    assert step.expect.item_index == 0
    step = adv(s, "option", 0)             # Q1 = Never
    assert step.expect.item_index == 8, "skip rule must jump to item 9"
    step = adv(s, "option", 0)
    assert step.expect.item_index == 9
    step = adv(s, "option", 0)
    a = s.assessments["audit"]
    assert a.score == 0 and a.zone == "healthy" and a.complete


def test_healthy_zone_skips_brief_intervention():
    s = new_session()
    drive_prescreen(s, 0, 1, 0)
    qf(s)
    adv(s, "consent", "no")
    adv(s, "consent", "yes")
    adv(s, "option", 0)                    # Q1 Never -> skip
    adv(s, "option", 0)                    # item 9
    step = adv(s, "option", 0)             # item 10 -> feedback perm
    step = adv(s, "consent", "yes")
    assert say_keys(step) == ["feedback.audit.healthy", "close"]
    assert s.node == "closed", "healthy zone must NOT enter BI"


def test_all_negative_prescreen_closes():
    s = new_session()
    step = drive_prescreen(s, 0, 0, 0)
    assert s.node == "closed"
    assert say_keys(step) == ["prescreen.all_negative", "close"]


def test_consent_decline_terminates():
    s = new_session()
    step = adv(s, "consent", "no")
    assert s.node == "declined" and step.expect.kind == "end"
    assert s.consent == "no"
    assert step.utterances == (), "pipeline owns the fixed decline goodbye"


def test_crisis_pauses_protocol_permanently():
    s = new_session()
    drive_prescreen(s, 0, 1, 0)
    runtime.enter_crisis(s)
    assert s.crisis and s.node == "crisis"
    step = adv(s, "open", "I feel awful")
    assert isinstance(step.utterances[0], LLMSay)
    assert step.expect.kind == "open"
    # No screening resumption, ever (deliberate: human decision, not pattern's).
    step = adv(s, "open", "ok")
    assert s.node == "crisis"


def test_wrong_payload_raises():
    s = new_session()
    with pytest.raises(runtime.ProtocolError):
        runtime.advance(s, TurnOut(action="answer", code=5))  # gate needs 0/1


def test_advance_rejects_non_answers():
    """Only VALIDATED answers may move the machine — question/tangent/unclear
    turns are the pipeline's to hold, never the engine's to consume."""
    s = new_session()
    with pytest.raises(runtime.ProtocolError):
        runtime.advance(s, TurnOut(action="question", reply="?"))
    with pytest.raises(runtime.ProtocolError):
        runtime.advance(s, TurnOut(action="unclear"))


def test_repeat_step_is_stable():
    s = new_session()
    step1 = drive_prescreen(s, 0, 1, 0)
    step2 = runtime.repeat_step(s)
    assert step1 == step2, "ambiguity must not move the machine"


def test_audit_dict_has_no_free_text():
    fix = load("drug_bi_case.json")
    s = new_session()
    drive_prescreen(s, 0, 0, 1)
    adv(s, "open", "SECRET DRUG NAME")
    adv(s, "open", "SECRET AMOUNT")
    adv(s, "consent", "yes")
    drive_screening(s, "dast_10", fix["codes"])
    d = json.dumps(s.to_audit_dict())
    assert "SECRET" not in d, "audit record must never carry transcripts"
    assert '"score": 7' in d
    # Open captures appear as KEYS only (what was answered, never the words).
    assert "drugs.kind" in json.loads(d)["answered"]


def test_every_fixed_key_the_machine_emits_exists_in_templates():
    # Walk all four case cards, collecting every Say the machine produced.
    # Each (key, text) must EXACTLY match the pre-warm enumeration — same key
    # must always mean same text, or the clip cache would serve stale audio.
    catalog = templates.all_fixed_utterances()
    seen = set()

    def collect(step):
        for u in step.utterances:
            if isinstance(u, Say):
                assert u.key in catalog, f"Say key not pre-warmable: {u.key}"
                assert catalog[u.key] == u.text, f"key/text drift: {u.key}"
                seen.add(u.key)

    for fixture in ("alcohol_bi_case.json", "drug_bi_case.json",
                    "alcohol_complete_case_3.json", "drug_complete_case_3.json"):
        fix = load(fixture)
        s = new_session()
        collect(drive_prescreen(s, fix["prescreen"]["tobacco"],
                                fix["prescreen"]["alcohol"],
                                fix["prescreen"]["drugs"]))
        while s.node not in ("closed", "declined"):
            kind = s.expect.kind
            if kind == "consent":
                step = adv(s, "consent", "yes")
            elif kind == "option":
                step = adv(s, "option", fix["codes"][s.expect.item_index])
            elif kind == "number":
                step = adv(s, "number", fix["readiness"])
            else:
                step = adv(s, "open", "x")
            collect(step)
    assert "close" in seen
