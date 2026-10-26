from __future__ import annotations

import re
from urllib.parse import urlsplit


_URL_RE = re.compile(r"\b(?:https?|postgresql|mysql|redis)://[^\s\"'<>]+", re.IGNORECASE)
_BEARER_RE = re.compile(r"Bearer\s+[A-Za-z0-9._~+/=-]+", re.IGNORECASE)
_API_KEY_RE = re.compile(r"(?i)(api[_-]?key|token|secret|password|passwd)=([^&\s]+)")


def mask_config_value(value: str) -> str:
    """Return a non-identifying marker for configured endpoint values."""

    return "已配置（已脱敏）" if value else ""


def redact_sensitive_text(text: object) -> str:
    """Redact URLs, credentials and bearer tokens before returning diagnostics."""

    if text is None:
        return ""
    redacted = str(text)
    redacted = _BEARER_RE.sub("Bearer <redacted>", redacted)
    redacted = _API_KEY_RE.sub(lambda match: f"{match.group(1)}=<redacted>", redacted)
    redacted = _URL_RE.sub(_redact_url_match, redacted)
    return redacted


def _redact_url_match(match: re.Match[str]) -> str:
    value = match.group(0)
    try:
        parsed = urlsplit(value)
    except ValueError:
        return "<redacted-url>"
    if parsed.scheme == "postgresql":
        return "postgresql://<redacted>"
    return f"{parsed.scheme}://<redacted>"
