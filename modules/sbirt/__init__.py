"""SBIRT clinical framework — the single source of truth for the counselor.

Structured, config-free, pure-Python building blocks that fully encode the
Screening, Brief Intervention, and Referral to Treatment model:

  instruments.py  — validated screening tools (AUDIT, DAST-10, CAGE-AID, ...)
  intervention.py — MI/OARS, FRAMES, stages of change, readiness rulers
  referral.py     — ASAM levels of care, MAT, resources, crisis protocol
  workflow.py     — the SBIRT conversation state machine
  prompt.py       — build_system_prompt(): renders ALL of the above into the LLM
                    system prompt (the guarantee that the Q&A carries every part
                    of SBIRT)

Edit the clinical content in the data modules; the prompt updates automatically.
"""

from . import instruments, intervention, referral, workflow
from .instruments import ALL_INSTRUMENTS, Instrument, risk_band_for
from .prompt import build_system_prompt
from .workflow import ENTRY_NODE, NODES

__all__ = [
    "build_system_prompt",
    "instruments",
    "intervention",
    "referral",
    "workflow",
    "ALL_INSTRUMENTS",
    "Instrument",
    "risk_band_for",
    "NODES",
    "ENTRY_NODE",
]
