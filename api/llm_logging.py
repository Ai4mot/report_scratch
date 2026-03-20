from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOGS_DIR = PROJECT_ROOT / "logs"
_log_lock = Lock()


def sanitize_topic(topic: str, max_len: int = 80) -> str:
    safe = re.sub(r"[^\w\u4e00-\u9fff-]+", "_", (topic or "").strip())
    safe = safe.strip("_")
    return (safe or "untitled")[:max_len]


def create_log_path(topic: str) -> Path:
    LOGS_DIR.mkdir(exist_ok=True)
    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%S%fZ")
    return LOGS_DIR / f"{sanitize_topic(topic)}+{timestamp}.log"


def mask_sensitive(data: Any, max_len: int = 500) -> Any:
    if isinstance(data, dict):
        res = {}
        for k, v in data.items():
            k_lower = k.lower()
            if "key" in k_lower or "token" in k_lower or "auth" in k_lower:
                res[k] = "***MASKED***"
            else:
                res[k] = mask_sensitive(v, max_len)
        return res
    if isinstance(data, list):
        return [mask_sensitive(x, max_len) for x in data]
    if isinstance(data, str):
        if len(data) > max_len:
            half = max_len // 2
            return (
                data[:half]
                + f"\n...[truncated {len(data) - max_len} chars]...\n"
                + data[-half:]
            )
        return data
    return data


def append_log(log_path: str | Path | None, event: str, payload: Any) -> None:
    if not log_path:
        return

    path = Path(log_path)
    path.parent.mkdir(exist_ok=True)

    masked_payload = mask_sensitive(payload, max_len=1000)

    entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "event": event,
        "payload": masked_payload,
    }

    with _log_lock:
        with path.open("a", encoding="utf-8") as fp:
            fp.write(json.dumps(entry, ensure_ascii=False, default=str))
            fp.write("\n")
