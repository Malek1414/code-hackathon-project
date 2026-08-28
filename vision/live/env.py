"""Tiny .env loader (python-dotenv is not in the venv). Secrets stay in the
environment: never print, log or persist the values read here."""

from __future__ import annotations

import os
from pathlib import Path


def load_dotenv(path: str | Path = ".env") -> int:
    """Set KEY=VALUE lines from `path` into os.environ (existing vars win).
    Returns how many keys were loaded."""
    p = Path(path)
    if not p.exists():
        return 0
    n = 0
    for raw in p.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[7:]
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        if key and key not in os.environ:
            os.environ[key] = value
            n += 1
    return n


def rtmp_url() -> str | None:
    url = os.environ.get("FOLLOWCAM_RTMP_URL", "").strip()
    return url or None


def redact(url: str) -> str:
    """rtmp://host/app/<key> -> rtmp://host/app/*** for logs."""
    head, _, _key = url.rpartition("/")
    return f"{head}/***" if head else "***"
