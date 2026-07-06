"""Patient-data protection: keep PHI out of logs, and record consent.

Two small, related jobs live together here because they answer the same
question — "what do we do to protect the person's data?":

  • phi() / phi_keys() — everything a patient says (ASR transcripts),
    everything the counselor says back (LLM sentences), and the extracted
    patient profile are PHI. Wrap any such value in a log line with phi();
    it renders a content-free shape summary ("<phi 7w/41c>") so logs stay
    debuggable without ever leaking content. There is NO opt-out: PHI never
    reaches the logs, not even in local debugging.

  • record_consent() — append-only JSONL audit trail: one line per consent
    decision (what, when, and the exact greeting wording by content hash).
    No transcripts, no screening codes, no names — the session key is the
    browser-generated pseudonymous UUID. A write failure logs a warning and
    returns False; it must never break a conversation turn.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
from datetime import datetime, timezone

import config

logger = logging.getLogger(__name__)


# --------------- PHI-safe logging ---------------

def phi(value) -> str:
    """Render user/clinical content for a log line without leaking it."""
    s = str(value)
    return f"<phi {len(s.split())}w/{len(s)}c>"


def phi_keys(mapping) -> str:
    """Render a PHI-bearing dict for logs as its key list only."""
    try:
        return "keys=" + repr(sorted(mapping.keys()))
    except Exception:
        return "<phi mapping>"


# --------------- Consent audit trail ---------------

_lock = threading.Lock()


def _greeting_version() -> str:
    """Short content hash of the exact consent wording the user answered to."""
    return hashlib.sha256(config.GREETING_TEXT.encode("utf-8")).hexdigest()[:12]


def record_consent(session_key: str, decision: str) -> bool:
    """Append one consent decision. Returns True if written, False on failure
    (logged, never raised — a turn must not break on an audit write)."""
    entry = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "session": session_key,
        "decision": decision,                    # "yes" | "no"
        "greeting_sha": _greeting_version(),
    }
    try:
        path = config.CONSENT_LOG_PATH
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with _lock, open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
        return True
    except Exception as e:
        logger.warning("consent audit write failed (%s) — decision=%s", e, decision)
        return False
