"""The small vocabulary shared by the form table, the voice and the pipeline.

Split out of runtime so that select and voice can name an Expect without
importing the engine that consumes them.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Say:
    """A study-verbatim line, keyed so it can be cached and pre-warmed."""

    key: str
    text: str


@dataclass(frozen=True)
class LLMSay:
    """An instruction the model words itself. Costs a live render."""

    instruction: str


@dataclass(frozen=True)
class Speak:
    """Text composed this turn — a read-back, or the model's own reply."""

    text: str


@dataclass(frozen=True)
class Expect:
    """What a legal answer looks like right now.

    Unchanged across the engine rewrite on purpose: this is the seam that lets
    llm.turn and turn.validate keep working while everything behind it moves.
    """

    kind: str
    instrument: str | None = None
    item_index: int | None = None
    ask_key: str | None = None
    slots: tuple[str, ...] = ()
    missing: tuple[str, ...] = ()


class ProtocolError(RuntimeError):
    pass
