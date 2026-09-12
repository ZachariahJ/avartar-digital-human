"""The tick loop: one thread that owns the session and never blocks.

The invariant it maintains is the whole repair:

    whenever the runner holds the floor and is idle, either there is nothing
    left to select, or the selected field's unit is in the voiced ledger.

Any tick that finds otherwise delivers again. A prompt that never reached the
transcript at all — cancelled before it was composed — needs no recovery path of
its own, because nothing wrote it to the ledger. One cut off part-way counts as
said: it is on screen, and re-reading it is worse than losing the audio.

Whose turn it is is a state here, not something inferred. It used to be the
conjunction of four liveness flags owned by three modules — no delivery thread,
no NLU thread, no cancel standing — and those flip at different moments as an
utterance moves through VAD, ASR and the model. Every gap between them was a
window in which the runner believed the floor was free: one of them, between
the cancel being cleared at speech_end and the NLU flag being raised when the
transcript landed, is how a re-asked question jumped out sixteen milliseconds
before the answer to it arrived.

Threading. This thread is the only writer of ClinicalSession, which is what
retires the old protocol lock: that lock was held across the model call, the TTS
network round-trip and a multi-second GPU render, so a second utterance queued
behind a render instead of superseding it. Work that blocks runs elsewhere and
posts its result back as an event.
"""

from __future__ import annotations

import enum
import logging
import queue
import threading
import time
from dataclasses import dataclass

import config
from modules import llm, privacy
from modules.sbirt import runtime, select as sel, voice
from modules.sbirt.form import Field
from modules.sbirt.instruments import InvalidResponse
from modules.sbirt.runtime_types import LLMSay, ProtocolError, Speak

logger = logging.getLogger(__name__)


class Floor(enum.Enum):
    """Who holds the conversational turn.

    SYSTEM is the only state in which the runner may speak. USER starts at the
    first voiced frame and PENDING covers the hand-back — the words exist but
    are not understood yet — which is precisely the stretch the old flag
    conjunction left uncovered.
    """

    SYSTEM = "system"
    USER = "user"
    PENDING = "pending"


@dataclass(frozen=True)
class Delivery:
    """An acknowledgment and a protocol prompt, as one indivisible unit.

    Either the whole thing is voiced and `unit` enters the ledger, or none of it
    counts. A half-delivered unit is exactly what let the consent question and
    the first pre-screen question disappear: beat one played, a phantom barge-in
    bumped the turn, and the second beat was dropped while the interpreter had
    already moved past the question.

    `unit` is None for something that is said but not owed — a reply that
    answers a question, a line acknowledging a skip. Those are not replayed.
    """

    unit: str | None
    beats: tuple
    ack: str = ""


@dataclass
class _Turn:
    """One user utterance being understood, off the tick thread."""

    text: str
    field: Field | None
    expect: object
    ask_text: str
    epoch: int = 0


class DialogRunner:
    """Drives one conversation. Owned by a Pipeline, one per session."""

    def __init__(self, pipeline, session: runtime.ClinicalSession):
        self.pipe = pipeline
        self.session = session
        self._q: queue.Queue = queue.Queue()
        self._inflight: str | None = None
        self._pending_ack: str = ""
        self._responding = False
        self._floor = Floor.SYSTEM
        self._floor_since = time.monotonic()
        self._epoch = 0
        self._reply_seq = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, name="dialog",
                                        daemon=True)
        self._thread.start()

    # --- inbound events -----------------------------------------------------

    def post(self, kind: str, *payload) -> None:
        self._q.put((kind, payload))

    def user_said(self, text: str) -> None:
        self.post("input", text)

    def interrupted(self) -> None:
        """A barge-in landed. The audio is already stopped; drop what it cut."""
        self.post("barge_in")

    def floor_taken(self) -> None:
        """The person has started speaking."""
        self.post("floor_user")

    def floor_pending(self) -> None:
        """They have stopped; their words are on their way through ASR."""
        self.post("floor_pending")

    def floor_released(self) -> None:
        """Nothing came of it — no transcript, or understanding failed."""
        self.post("floor_release")

    def restart(self, session: runtime.ClinicalSession) -> None:
        self.post("reset", session)

    def shutdown(self) -> None:
        self._stop.set()

    # --- the loop -----------------------------------------------------------

    def _loop(self) -> None:
        while not self._stop.is_set() and not config.SHUTTING_DOWN.is_set():
            try:
                kind, payload = self._q.get(timeout=config.DIALOG_TICK_SEC)
            except queue.Empty:
                kind, payload = None, ()
            try:
                if kind is not None:
                    self._handle(kind, payload)
                self._reclaim_floor()
                self._advance()
            except Exception:
                logger.exception("[dialog] tick failed; continuing")

    def _handle(self, kind: str, payload: tuple) -> None:
        if kind == "input":
            self._begin_turn(payload[0])
        elif kind == "turn":
            self._dispatch(*payload)
        elif kind == "delivered":
            self._delivered(*payload)
        elif kind == "barge_in":
            self._barge_in()
        elif kind == "floor_user":
            self._set_floor(Floor.USER)
        elif kind == "floor_pending":
            self._set_floor(Floor.PENDING)
        elif kind == "floor_release":
            self._release_floor()
        elif kind == "reset":
            self._reset(payload[0])

    # --- the floor ----------------------------------------------------------

    def _set_floor(self, floor: Floor) -> None:
        if floor is self._floor:
            return
        logger.info("[dialog] floor %s -> %s", self._floor.value, floor.value)
        self._floor = floor
        self._floor_since = time.monotonic()
        if floor is Floor.SYSTEM:
            self.pipe.floor_returned()

    def _release_floor(self) -> None:
        """Hand the floor back only from PENDING.

        Someone who has started speaking again keeps it: a late transcript or a
        failed model call must not talk over them.
        """
        if self._floor is Floor.PENDING:
            self._set_floor(Floor.SYSTEM)

    def _reclaim_floor(self) -> None:
        """Take the floor back when a hand-off never completes.

        Neither hand-off is guaranteed to finish. A speech_start the near-field
        gate then rejects never produces a speech_end, and an ASR or NLU thread
        that dies never produces a turn; without this the runner waits for a
        hand-back that is not coming and falls silent for the rest of the
        session.
        """
        if self._floor is Floor.SYSTEM:
            return
        limit = (config.FLOOR_USER_MAX_SEC if self._floor is Floor.USER
                 else config.FLOOR_PENDING_MAX_SEC)
        if time.monotonic() - self._floor_since < limit:
            return
        logger.info("[dialog] floor held in %s past %.0fs; reclaiming",
                    self._floor.value, limit)
        self._set_floor(Floor.SYSTEM)

    # --- understanding one utterance ---------------------------------------

    def _begin_turn(self, text: str) -> None:
        """Snapshot what the utterance is answering, then understand it off-thread."""
        self._set_floor(Floor.PENDING)
        self.pipe.history_user(text)
        fld = sel.answerable(self.session)
        turn = _Turn(text=text, field=fld,
                     expect=sel.field_expect(self.session, fld),
                     ask_text=voice.ask_text(self.session, fld),
                     epoch=self._epoch)
        threading.Thread(target=self._understand, args=(turn,),
                         name="nlu", daemon=True).start()

    def _understand(self, turn: _Turn) -> None:
        try:
            out = llm.turn(
                turn.text, turn.expect, ask_text=turn.ask_text,
                history=self.pipe.api_window(), patient=self.pipe.patient_facts(),
                facts=self.pipe.turn_facts(self.session),
                interview_state=self.pipe.interview_state(self.session))
        except Exception:
            logger.exception("[dialog] NLU failed; holding the question")
            out = None
        self.post("turn", turn, out)

    def _dispatch(self, turn: _Turn, out) -> None:
        self._release_floor()
        if turn.epoch != self._epoch:
            # A reset or a barge-in superseded the utterance this answers.
            return
        if out is None:
            return
        session = self.session
        fld = turn.field

        runtime.record_harvest(session, out)

        if out.action == "crisis":
            runtime.enter_crisis(session)
            return
        if out.action == "abort":
            runtime.enter_abort(session)
            return

        if fld is None:
            # The session is closed but they are still talking.
            self._say(LLMSay(
                "The screening session is already complete. In one warm "
                "sentence, acknowledge what the person said and remind them "
                "their provider will follow up with them."))
            return

        if out.action == "correction":
            try:
                changed = runtime.correct(session, fld, out)
            except InvalidResponse:
                logger.exception("[dialog] correction failed; holding")
                changed = False
            self._pending_ack = out.reply
            if not changed:
                self._repose(fld)
            return

        if out.action == "answer":
            self._answer(fld, out)
            return

        if out.action == "continuation":
            runtime.absorb(session, out)
            self._pending_ack = out.reply
            self._repose(fld)
            return

        if out.action == "dont_know":
            if runtime.note_stall(session) >= runtime.DONT_KNOW_LIMIT:
                self._skip(fld, "dont_know")
                return
            self._probe(fld, out.reply)
            return

        if out.action == "unclear":
            if out.unusable:
                # The engine could not read the model's output. Charging that to
                # the person skips a question they may well have answered — as
                # "2 drinks per day" was, twice, on a schema mismatch.
                if runtime.note_misread(session) >= runtime.MISREAD_LIMIT:
                    self._skip(fld, "no_answer")
                    return
                self._repose(fld)
                return
            if runtime.note_stall(session) >= runtime.UNCLEAR_LIMIT:
                self._skip(fld, "no_answer")
                return
            if out.reply:
                # The reply is itself a clarifying question; re-reading the whole
                # item after it would be the same question twice in one breath.
                self._say(Speak(out.reply))
            else:
                self._repose(fld)
            return

        if out.action == "discomfort":
            runtime.offer_pause(session)
            self._pending_ack = out.reply
            return

        if out.action == "tangent":
            if runtime.note_aside(session) >= runtime.ASIDE_LIMIT:
                runtime.offer_pause(session)
            else:
                self._repose(fld)
            self._pending_ack = out.reply
            return

        # A question about the interview: answered from state, then the ask
        # comes back as authored.
        self._pending_ack = out.reply
        self._repose(fld)

    def _answer(self, fld: Field, out) -> None:
        session = self.session
        self._pending_ack = out.reply

        if fld.id == "consent.opening":
            privacy.record_consent(self.pipe.audit_key,
                                   "yes" if out.code == 1 else "no")
        if fld.id == "pause.offer":
            keep = out.code == 1
            runtime.resolve_pause(session, keep_going=keep)
            if keep:
                # Several turns have passed, so the question has to be put
                # again or both sides sit waiting on each other.
                resumed = sel.select(session)
                if resumed is not None:
                    self._repose(resumed)
            return
        if fld.kind == "confirm":
            runtime.resolve_confirm(session, yes=(out.code == 1))
            return

        reason = runtime.confirm_reason(session, fld, out)
        if reason is not None:
            runtime.request_confirm(session, fld, out, reason)
            return
        try:
            runtime.record(session, fld, out)
        except (ProtocolError, InvalidResponse):
            logger.exception("[dialog] record failed; re-asking")
            self._pending_ack = ""
            self._repose(fld)

    # --- responses that are said but not owed -------------------------------

    def _say(self, *beats) -> None:
        ack, self._pending_ack = self._pending_ack, ""
        self._enqueue(Delivery(unit=None, beats=tuple(beats), ack=ack))

    def _repose(self, fld: Field) -> None:
        runtime.repose(self.session, sel.unit_id(self.session, fld))

    def _skip(self, fld: Field, reason: str) -> None:
        beats = runtime.mark_missing(self.session, fld, reason)
        if beats:
            self._say(*beats)

    def _probe(self, fld: Field, reply: str) -> None:
        """Help them estimate an answer they say they cannot give.

        One attempt only, following the WHO manual: anchor them to their
        heaviest period in the past year and always offer to skip, so declining
        a second time is easy.
        """
        instruction = voice.probe_instruction(self.session, fld)
        if not instruction:
            self._pending_ack = reply
            self._repose(fld)
            return
        self._pending_ack = reply
        self._say(LLMSay(instruction))
        self._repose(fld)

    # --- delivery -----------------------------------------------------------

    def _advance(self) -> None:
        """Deliver whatever the state says is owed. The repair path."""
        if self._floor is not Floor.SYSTEM or self._inflight is not None:
            return
        if not self.pipe.ready_to_speak():
            # The cancel from a barge-in is cleared when the floor comes back,
            # but the two arrive on different threads; one tick of skew is not
            # worth a delivery that would only bail and be owed again.
            return
        session = self.session
        fld = sel.select(session)
        if fld is None:
            self._finish()
            return
        unit = sel.unit_id(session, fld)
        if unit in session.spoken:
            self._await_answer(fld)
            if self._pending_ack:
                # Nothing is owed, but the person is still owed an answer.
                self._say()
            return

        # A volunteered answer is read back instead of asked for again.
        if fld.kind != "confirm" and fld.slot \
                and sel.candidate_for(session, fld.slot):
            runtime.request_volunteered_confirm(session, fld.slot)
            return

        ack, self._pending_ack = self._pending_ack, ""
        self._enqueue(Delivery(unit=unit, beats=voice.beats_for(session, fld),
                               ack=ack))

    def _enqueue(self, delivery: Delivery) -> None:
        if not delivery.beats and not delivery.ack:
            if delivery.unit is not None:
                # A carried prompt: the question was already asked by an earlier
                # field, so it owes no audio but still counts as voiced. The next
                # tick finds it spoken and starts waiting for the answer.
                self.session.spoken.add(delivery.unit)
            return
        self._inflight = delivery.unit or f"reply#{self._reply_seq}"
        self._reply_seq += 1
        self._responding = True
        threading.Thread(target=self._run_delivery, args=(delivery,),
                         name="delivery", daemon=True).start()

    def _run_delivery(self, delivery: Delivery) -> None:
        ok = False
        try:
            ok = self.pipe.speak(delivery)
        except Exception:
            logger.exception("[dialog] delivery raised; unit stays unspoken")
        self.post("delivered", delivery, ok)

    def _delivered(self, delivery: Delivery, ok: bool) -> None:
        self._inflight = None
        if ok and delivery.unit is not None:
            self.session.spoken.add(delivery.unit)
            logger.info("[dialog] voiced %s", delivery.unit)
        elif delivery.unit is not None:
            logger.info("[dialog] %s was cut short; it stays owed",
                        delivery.unit)

    def _finish(self) -> None:
        if self._responding:
            self._responding = False
            self.pipe.end_response()
            self.pipe.session_ended()

    def _await_answer(self, fld: Field | None) -> None:
        """Wait indefinitely once asked: silence is not a prompt, an answer
        or a reason to skip."""
        if fld is None or fld.kind in ("tell", "end"):
            return
        if self._responding:
            self._responding = False
            self.pipe.end_response()

    # --- interruption and reset ---------------------------------------------

    def _barge_in(self) -> None:
        self._epoch += 1
        self._pending_ack = ""

    def _reset(self, session: runtime.ClinicalSession) -> None:
        self._epoch += 1
        self.session = session
        self._inflight = None
        self._pending_ack = ""
        self._responding = False
        self._set_floor(Floor.SYSTEM)
