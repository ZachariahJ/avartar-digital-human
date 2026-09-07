"""Two obligations around patient data: keep it out of logs, and record consent.

Everything a patient says, everything the counselor says back, and the profile
extracted from either is protected health information. Passing such a value
through phi() before logging it yields a shape summary like "<phi 7w/41c>",
which keeps logs useful for debugging without their contents ever becoming a
disclosure. config.LOG_PHI disables the redaction for local work and must stay
off wherever real patients are seen.

record_consent() writes the audit trail: one line per decision, carrying the
decision, the time and a hash of the wording consented to. No transcripts, no
screening results and no names — sessions are identified only by the browser's
pseudonymous id.
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


def phi(value) -> str:
    """Summarise a value's shape for a log line without revealing its content.

    Returns the value verbatim when config.LOG_PHI is set.
    """
    s = str(value)
    if config.LOG_PHI:
        return s
    return f"<phi {len(s.split())}w/{len(s)}c>"


def phi_keys(mapping) -> str:
    """Log which facts a mapping holds without logging any of their values.

    Returns the mapping verbatim when config.LOG_PHI is set. Falls back to a
    placeholder for anything that is not key-addressable.
    """
    if config.LOG_PHI:
        return repr(mapping)
    try:
        return "keys=" + repr(sorted(mapping.keys()))
    except Exception:
        return "<phi mapping>"


_lock = threading.Lock()


def _greeting_version() -> str:
    """Hash of the wording being consented to.

    Recorded with each decision so that a later edit to the greeting is visible
    as a version change instead of silently reinterpreting old records.
    """
    return hashlib.sha256(config.GREETING_TEXT.encode("utf-8")).hexdigest()[:12]


def record_consent(session_key: str, decision: str) -> bool:
    """Append one consent decision to the audit trail.

    Args:
        session_key: the browser's pseudonymous session id.
        decision: "yes" or "no".

    Returns:
        True if the line was written. A failure is logged and returns False
        rather than raising, so an unwritable audit file cannot break a
        conversation in progress.
    """
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
