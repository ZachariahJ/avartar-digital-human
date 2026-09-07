"""The clinical content and machinery of the SBIRT screening, in one place.

Screening, Brief Intervention and Referral to Treatment, encoded as plain Python
data so that the clinical material can be reviewed and edited without touching
any conversational code.

Two layers, easily confused:

  * The executable protocol — flow.py, runtime.py, instruments.py, coding.py,
    templates.py, turn.py, crisis.py — decides what is asked, what an answer
    scores, and where the session goes. It is deterministic, and the model
    cannot influence it.
  * The described protocol — workflow.py, intervention.py, referral.py,
    rendered by prompt.py — is background the model is given so it can converse
    competently. Nothing here decides anything.

Editing the data updates the prompt automatically; there is no prompt to
hand-maintain.
"""

from . import (crisis, instruments, intervention, referral, runtime,
               templates, workflow)
from .instruments import ALL_INSTRUMENTS, Instrument, risk_band_for
from .prompt import build_system_prompt
from .workflow import ENTRY_NODE, NODES

__all__ = [
    "build_system_prompt",
    "crisis",
    "instruments",
    "intervention",
    "referral",
    "runtime",
    "templates",
    "workflow",
    "ALL_INSTRUMENTS",
    "Instrument",
    "risk_band_for",
    "NODES",
    "ENTRY_NODE",
]
