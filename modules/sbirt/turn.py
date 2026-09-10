
from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, Field, field_validator

from . import coding
from .instruments import BY_KEY, PRE_SCREEN

Action = Literal["answer", "continuation", "question", "tangent", "discomfort",
                 "crisis", "abort", "correction", "dont_know", "unclear"]


class Harvest(BaseModel):

    model_config = {"extra": "forbid"}

    target: str
    code: int | None = None
    text: str | None = None
    quote: str = ""


class TurnOut(BaseModel):

    model_config = {"extra": "forbid"}

    action: Action
    code: int | None = None
    item: int | None = None
    slots: dict[str, str] = Field(default_factory=dict)
    harvest: list[Harvest] = Field(default_factory=list)
    text: str | None = None
    value: float | None = None
    per: str | None = None
    unit: str | None = None
    beverage: str | None = None
    reply: str = ""
    exact: bool = False
    assumed: bool = False
    boundary: bool = False
    note: str = ""

    @field_validator("reply", "text", mode="before")
    @classmethod
    def _strip(cls, v):
        return v.strip() if isinstance(v, str) else v


def _unclear(out: TurnOut, why: str) -> TurnOut:
    return out.model_copy(update={"action": "unclear", "code": None,
                                  "item": None, "slots": {}, "text": None,
                                  "value": None, "per": None, "unit": None,
                                  "beverage": None, "assumed": False,
                                  "boundary": False, "note": ""})


def expected_item(expect):
    if expect.instrument == "prescreen":
        return PRE_SCREEN[expect.item_index].item
    return BY_KEY[expect.instrument].items[expect.item_index]




def target_item(target: str):
    head, _, tail = target.rpartition(".")
    if head == "prescreen":
        return next((q.item for q in PRE_SCREEN if q.key == tail), None)
    if head in BY_KEY and tail.isdigit():
        items = BY_KEY[head].items
        idx = int(tail)
        return items[idx] if idx < len(items) else None
    return None


def expect_target(expect) -> str | None:
    if expect.kind == "option" and expect.instrument is not None:
        if expect.instrument == "prescreen":
            return f"prescreen.{PRE_SCREEN[expect.item_index].key}"
        return f"{expect.instrument}.{expect.item_index}"
    if expect.kind in ("open", "number"):
        return expect.ask_key
    return None


def validate_harvest(out: TurnOut, expect) -> list[Harvest]:
    here = expect_target(expect)
    kept = []
    for h in out.harvest:
        if not h.target or h.target == here:
            continue
        item = target_item(h.target)
        if item is not None:
            if isinstance(h.code, int) and 0 <= h.code < len(item.options):
                kept.append(h)
            continue
        if h.text:
            kept.append(h)
    return kept


_WORD = re.compile(r"[a-z0-9']+")
_ECHO_RUN = 7


def scrub_reply(reply: str, ask_text: str) -> str:
    if not reply or not ask_text:
        return reply
    ask = _WORD.findall(ask_text.lower())
    if len(ask) < _ECHO_RUN:
        return reply
    runs = {tuple(ask[i:i + _ECHO_RUN])
            for i in range(len(ask) - _ECHO_RUN + 1)}
    kept = []
    for sentence in re.split(r"(?<=[.!?])\s+", reply):
        words = _WORD.findall(sentence.lower())
        echo = any(tuple(words[i:i + _ECHO_RUN]) in runs
                   for i in range(len(words) - _ECHO_RUN + 1))
        if not echo:
            kept.append(sentence)
    return " ".join(kept).strip() or ""


def validate(out: TurnOut, expect, ask_text: str = "") -> TurnOut:
    out = out.model_copy(update={
        "reply": scrub_reply(out.reply, ask_text),
        "harvest": validate_harvest(out, expect),
    })
    if out.action == "correction":
        if (expect.kind == "option" and expect.instrument
                and expect.instrument != "prescreen"
                and isinstance(out.item, int) and isinstance(out.code, int)):
            items = BY_KEY[expect.instrument].items
            if (0 <= out.item < len(items)
                    and out.item != expect.item_index
                    and 0 <= out.code < len(items[out.item].options)):
                return out.model_copy(update={"slots": {}, "text": None})
        return _unclear(out, "correction needs a known earlier item + option")

    if out.action != "answer":
        if out.action == "continuation":
            return out
        if (out.code is not None or out.item is not None or out.slots
                or out.text or out.value is not None):
            return out.model_copy(update={"code": None, "item": None,
                                          "slots": {}, "text": None,
                                          "value": None, "per": None,
                                          "unit": None, "beverage": None})
        return out

    kind = expect.kind
    if kind in ("consent", "confirm"):
        if out.code in (0, 1):
            return out
        return _unclear(out, f"{kind} needs yes(1)/no(0)")
    if kind == "option":
        item = expected_item(expect)
        if item.coding != "choice" and not out.exact:
            derived = coding.derive(item.coding, value=out.value,
                                    per=out.per, unit=out.unit,
                                    beverage=out.beverage)
            if derived is None:
                return _unclear(
                    out, f"{item.coding} answer not derivable from "
                         f"extraction (value={out.value!r} per={out.per!r} "
                         f"unit={out.unit!r} beverage={out.beverage!r})")
            return out.model_copy(update={
                "code": derived.code, "assumed": derived.assumed,
                "boundary": derived.boundary, "note": derived.note,
                "slots": {}, "text": None})
        if isinstance(out.code, int) and 0 <= out.code < len(item.options):
            return out
        return _unclear(out, "option code out of range")
    if kind == "number":
        if isinstance(out.code, int) and 0 <= out.code <= 10:
            return out
        return _unclear(out, "ruler needs 0-10")
    if kind == "open":
        missing = tuple(getattr(expect, "missing", ()) or ())
        if missing:
            declared = set(getattr(expect, "slots", ()) or ())
            slots = {k: v for k, v in out.slots.items()
                     if k in declared and str(v).strip()}
            if slots:
                return out.model_copy(update={"slots": slots})
            if out.text:
                return out.model_copy(
                    update={"slots": {missing[0]: out.text}})
            return _unclear(out, "open slot answer captured nothing")
        if out.text:
            return out
        return _unclear(out, "open answer captured nothing")
    return _unclear(out, f"no answer possible at kind={kind!r}")
