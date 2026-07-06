"""T13 acceptance: drive the REAL protocol engine + REAL llm.turn over a
scripted user at the TEXT layer (no ASR/TTS/FLOAT — those are byte-identical
delivery, not conversation logic).

Three scenarios:
  1. REPLAY of the field transcript that motivated the refactor (whiskey /
     twelve ounces in two breaths, education declined, "have we ever
     discussed the standard drink definition?", declines more questions) —
     asserts each of the six documented defects is gone.
  2. A full conversational alcohol-BI walk with non-label phrasings, so the
     LLM (not the pre-match) does real coding — asserts the deterministic
     score/zone and the one-question-per-turn rule.
  3. REPLAY of the second field transcript (2026-07, "ten twenty times" /
     "more than one" / "you spoke too fast") — asserts codable answers are
     coded first try, vague or wrong-dimension answers are clarified (never
     captured), nothing is re-asked word-for-word, and a repeat request is
     answered instead of ignored.

Needs OPENROUTER_API_KEY (network). Run:  python scripts/sim_conversation.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config                                             # noqa: E402
from modules import llm                                   # noqa: E402
from modules.sbirt import crisis, runtime, templates      # noqa: E402
from modules.sbirt.instruments import BY_KEY              # noqa: E402
from modules.sbirt.runtime import LLMSay, Say, Speak      # noqa: E402


class TextSession:
    """The pipeline's protocol turn, text-only (mirrors Pipeline._protocol_turn
    + _deliver_step semantics 1:1 — anything that diverges here is a bug in
    ONE of the two, so keep them structurally parallel)."""

    def __init__(self):
        self.s = runtime.ClinicalSession()
        runtime.start(self.s)
        self.history = [{"role": "assistant", "content": config.GREETING_TEXT}]
        self.transcript = [("avatar", config.GREETING_TEXT, "greeting")]

    def facts(self):
        c = self.s
        facts = {
            "current_phase": c.node,
            "standard_drink_definition_discussed":
                "alcohol.edu.standard_drink" in c.covered,
            "drinking_limits_discussed": "alcohol.edu.limits" in c.covered,
            "permissions_declined_so_far": list(c.declined),
            "active_topic": c.arm,
        }
        for unit_key, fact_key in (
                ("alcohol.edu.standard_drink", "standard_drink_definition"),
                ("alcohol.edu.limits", "recommended_drinking_limits")):
            if unit_key in c.covered:
                facts[fact_key] = templates.FIXED[unit_key]
        exp = c.expect
        if (exp.kind == "option" and exp.instrument
                and exp.instrument != "prescreen"):
            items = BY_KEY[exp.instrument].items
            facts["answers_already_given"] = [
                {"item": i, "question": items[i].text,
                 "answer": items[i].options[code].label}
                for i, code in sorted(
                    c.responses.get(exp.instrument, {}).items())]
        return facts

    def last_ask(self):
        step = self.s.last_step
        if step:
            for u in reversed(step.utterances):
                if isinstance(u, Say):
                    return u.text
        for m in reversed(self.history):
            if m["role"] == "assistant" and m["content"].strip():
                return m["content"]
        return config.GREETING_TEXT

    def speak(self, beats, ack=""):
        parts = [ack] if ack else []
        for u in beats:
            if isinstance(u, Say):
                parts.append(u.text)
            elif isinstance(u, Speak):
                parts.append(u.text)
            elif isinstance(u, LLMSay):
                t = llm.phrase_utterance(u.instruction, self.history[-6:])
                if t.strip():
                    parts.append(t)
        text = " ".join(p for p in parts if p.strip())
        if text:
            self.history.append({"role": "assistant", "content": text})
        return text

    def user(self, text):
        """One user turn. Returns (avatar_reply_text, action)."""
        self.transcript.append(("you", text, ""))
        self.history.append({"role": "user", "content": text})

        hit = crisis.detect(text)
        if hit:
            reply = crisis.RESPONSES[hit.category]
            runtime.enter_crisis(self.s)
            self.transcript.append(("avatar", reply, "CRISIS(net)"))
            self.history.append({"role": "assistant", "content": reply})
            return reply, "crisis-net"

        out = llm.turn(text, self.s.expect, ask_text=self.last_ask(),
                       history=self.history[-8:], patient=None,
                       facts=self.facts())

        if out.action == "crisis":
            runtime.enter_crisis(self.s)
            step = runtime.crisis_step(self.s)
            reply = self.speak(step.utterances)
            self.transcript.append(("avatar", reply, "CRISIS(nlu)"))
            return reply, "crisis"

        if out.action == "abort":
            step = runtime.enter_abort(self.s)
            reply = self.speak(step.utterances)
            self.transcript.append(("avatar", reply, "ABORT"))
            return reply, "abort"

        if out.action == "correction":
            step = runtime.correct(self.s, out)
            if step is not None:
                reply = self.speak(step.utterances, ack=out.reply)
                self.transcript.append(
                    ("avatar", reply, f"correction→{self.s.node}"))
                return reply, "correction"
            # fall through to the hold path (clarify which item they mean)

        if out.action == "answer":
            step = runtime.advance(self.s, out)
            if self.s.node == "declined":
                reply = config.DECLINE_TEXT
                self.history.append({"role": "assistant", "content": reply})
            else:
                reply = self.speak(step.utterances, ack=out.reply)
            self.transcript.append(("avatar", reply, f"answer→{self.s.node}"))
            return reply, "answer"

        if out.action == "continuation":
            runtime.absorb(self.s, out)
        reply = out.reply
        if not reply:                     # deterministic fallback re-ask
            reply = llm.phrase_utterance(
                f"Gently re-ask, in one short sentence: {self.last_ask()!r}",
                self.history[-6:])
        self.history.append({"role": "assistant", "content": reply})
        self.transcript.append(("avatar", reply, f"{out.action}·hold@{self.s.node}"))
        return reply, out.action

    def dump(self):
        for who, text, note in self.transcript:
            tag = f"  [{note}]" if note else ""
            print(f"{'AVATAR' if who == 'avatar' else 'YOU   '}: {text}{tag}")
        print()


def check(ok, label, problems):
    print(("  PASS  " if ok else "  FAIL  ") + label)
    if not ok:
        problems.append(label)


def scenario_replay(problems):
    print("=" * 72)
    print("SCENARIO 1 — replay of the field transcript (the six defects)")
    print("=" * 72)
    t = TextSession()
    t.user("yes")
    t.user("no")                                  # tobacco
    t.user("within the last year")                # alcohol pre-screen +
    t.user("never")                               # drugs pre-screen -
    # 毛病5: Q/F arrives ONE question at a time now.
    r1, _ = t.user("i like drink whiskey and i drink twelve oun per time")
    node_after_first_breath = t.s.node
    # 毛病2: the second breath must land in the SAME ask, not hit a gate.
    r2, a2 = t.user("and i drink once every week")
    qf_done = t.s.answers.get("alcohol.qf", "")
    t.user("no thank you")                        # decline education
    # 毛病4 setup: permission wording must not claim a discussed definition.
    perm_spoken = t.transcript[-1][1]
    # 毛病3: the user asks US something — expect an actual answer + hold.
    node_before_q = t.s.node
    r3, a3 = t.user("have we ever discussed the standard drink definition")
    node_after_q = t.s.node
    r4, _ = t.user("no thank you")                # decline the AUDIT
    t.dump()

    check(node_after_first_breath == "alcohol.qf",
          "#2/#5 first breath holds the qf slot-ask (no gate desync)", problems)
    check(a2 in ("answer", "continuation") and "week" in qf_done.lower()
          or "week" in str(t.s.slots.get("alcohol.qf", {})).lower(),
          "#1/#2 second breath captured into qf state (nothing discarded)",
          problems)
    check("just discussed" not in perm_spoken,
          "#4 permission wording never claims an undiscussed definition",
          problems)
    check(a3 == "question" and node_after_q == node_before_q and r3.strip(),
          "#3 user's question got answered; machine held position", problems)
    check(t.s.node == "closed" and "few more questions" not in r4,
          "#6 declined path closes coherently (no more-questions promise)",
          problems)
    qm = [(txt.count("?"), txt) for who, txt, n in t.transcript
          if who == "avatar" and n != "greeting"]
    worst = max(qm, key=lambda x: x[0]) if qm else (0, "")
    check(worst[0] <= 1,
          f"#5 every avatar turn asks at most ONE question (max={worst[0]})",
          problems)


def scenario_field_2(problems):
    print("=" * 72)
    print("SCENARIO 3 — replay of the second field transcript "
          "(ten-twenty times / more than one / spoke too fast)")
    print("=" * 72)
    t = TextSession()
    t.user("yes")
    t.user("yes i do")                             # tobacco +
    t.user("yesterday")                            # alcohol + (last year)
    # Defect: "about ten twenty times" was clarified TWICE instead of being
    # coded as "One or more" (it fits exactly one option).
    r, a = t.user("about ten twenty times")
    check(a == "answer" and t.s.prescreen.get("drugs") == 1,
          "codable range answer coded FIRST try (ten-twenty -> one-or-more)",
          problems)
    if t.s.prescreen.get("drugs") != 1:            # recover to keep replaying
        t.user("one or more")
    t.user("i like whiskey a lot")                 # qf drink slot
    # Defect: "maybe ten twenty times" (TIMES, asked for DRINKS) was met with
    # the question re-read word-for-word; then "more than one" was captured.
    ask_before = t.transcript[-1][1]
    r1, a1 = t.user("maybe ten twenty times")
    amount_after_times = t.s.slots.get("alcohol.qf", {}).get("amount")
    check(amount_after_times is None,
          "wrong-dimension answer (times for drinks) NOT captured as amount",
          problems)
    check(r1.strip() and r1.strip() != ask_before.strip(),
          "clarification is not the question re-read word-for-word", problems)
    r2, a2 = t.user("more than one")
    check(t.s.slots.get("alcohol.qf", {}).get("amount") is None,
          "vague 'more than one' NOT captured as a usable amount", problems)
    t.user("maybe ten drinks")                     # usable amount
    check(bool(t.s.slots.get("alcohol.qf", {}).get("amount")),
          "usable amount captured after clarification", problems)
    t.user("every day i'm a big addict")           # frequency
    t.user("yeah sure")                            # education permission
    node_before = t.s.node
    # Defect: "you spoke too fast" was ignored — the fixed permission line
    # was simply re-posed verbatim.
    r3, a3 = t.user("oh my gosh you spoke too fast i just cannot remember it")
    perm_line = templates.FIXED["alcohol.screen.permission"]
    check(t.s.node == node_before, "repeat request holds the machine", problems)
    check(r3.strip() and r3.strip() != perm_line
          and (a3 in ("question", "tangent") or "drink" in r3.lower()),
          "repeat request gets a real response, not the verbatim re-pose",
          problems)
    t.user("yes")                                  # AUDIT permission
    t.dump()
    check(t.s.node.startswith("screening.audit"),
          "protocol reaches AUDIT item 1 after the detour", problems)
    seen = [txt for who, txt, n in t.transcript if who == "avatar"]
    dup = any(x.strip() and x.strip() == y.strip()
              for x, y in zip(seen, seen[1:]))
    check(not dup, "no avatar turn repeats the previous one verbatim",
          problems)


def scenario_full_bi(problems):
    print("=" * 72)
    print("SCENARIO 2 — full alcohol BI walk, conversational answers")
    print("=" * 72)
    t = TextSession()
    for u in ("sure, go ahead",
              "no i don't smoke",
              "yeah, within this last year",
              "no never",
              "mostly red wine",
              "usually two or three glasses",
              "maybe four nights a week",
              "ok sure",                                   # education
              "yes that's fine",                           # AUDIT permission
              "two or three times a week",                 # item 1 -> 2/3
              "three or four",                             # item 2
              "less than monthly",                         # item 3
              "never",                                     # 4
              "less than monthly",                         # 5
              "never",                                     # 6
              "less than monthly",                         # 7
              "never",                                     # 8
              "no",                                        # 9
              "yes but not in the last year",              # 10 -> score 9
              "yes",                                       # feedback perm
              "okay",                                      # BI entry
              "it helps me unwind with friends",           # likes
              "the money and the mornings after",          # dislikes
              "i'd say an eight",                          # ruler
              "because i already know i should cut back",  # why not lower
              "weekends with friends make it hard",        # why not higher
              "i think i'll try cutting down a bit"):      # leaves you
        t.user(u)
    t.dump()

    a = t.s.assessments.get("audit")
    check(a is not None and a.complete, "AUDIT completed", problems)
    if a:
        check(a.score == 9 and a.zone == "risky",
              f"deterministic score/zone from conversational answers "
              f"(score={a.score}, zone={a.zone})", problems)
    check(t.s.readiness.get("alcohol") == 8, "ruler captured as 8", problems)
    check(t.s.node == "closed", "session closed", problems)
    for key in ("alcohol.qf", "bi.likes", "bi.dislikes", "bi.leaves_you"):
        check(bool(t.s.answers.get(key)), f"open capture present: {key}",
              problems)
    qm = [(txt.count("?"), txt) for who, txt, n in t.transcript
          if who == "avatar" and n != "greeting"]
    worst = max(qm, key=lambda x: x[0]) if qm else (0, "")
    check(worst[0] <= 1,
          f"one question per turn (max={worst[0]})", problems)


if __name__ == "__main__":
    problems = []
    scenario_replay(problems)
    print()
    scenario_full_bi(problems)
    print()
    scenario_field_2(problems)
    print("\n" + "=" * 72)
    if problems:
        print(f"RESULT: {len(problems)} CHECK(S) FAILED")
        for p in problems:
            print("  -", p)
        sys.exit(1)
    print("RESULT: ALL CHECKS PASSED")
