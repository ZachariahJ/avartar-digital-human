
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
    s = str(value)
    if config.LOG_PHI:
        return s
    return f"<phi {len(s.split())}w/{len(s)}c>"


def phi_keys(mapping) -> str:
    if config.LOG_PHI:
        return repr(mapping)
    try:
        return "keys=" + repr(sorted(mapping.keys()))
    except Exception:
        return "<phi mapping>"


_lock = threading.Lock()


def _greeting_version() -> str:
    return hashlib.sha256(config.GREETING_TEXT.encode("utf-8")).hexdigest()[:12]


def record_consent(session_key: str, decision: str) -> bool:
    entry = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "session": session_key,
        "decision": decision,
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
