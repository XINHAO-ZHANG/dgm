"""Strip secret values before text is logged or saved.

``OPENAI_BASE_URL`` is not a secret and is left unchanged. Docker still
passes container environment as a dict; this module only cleans strings
that might already contain ``KEY=value`` or provider tokens.
"""

from __future__ import annotations

import logging
import re

_REDACTED = "[REDACTED]"

# Names containing API_KEY or SECRET. OPENAI_BASE_URL does not match.
# The value stops before whitespace or a closing quote/bracket so exception
# text keeps its surrounding syntax.
_ASSIGNMENT = re.compile(
    r"(?P<key>\b[A-Za-z_][A-Za-z0-9_]*(?:API_KEY|SECRET)[A-Za-z0-9_]*)"
    r"(?P<sep>=['\"]?|['\"]?\s*:\s*['\"])"
    r"(?P<val>[^\s'\"`,;<>\[\]\)\}\\]+)"
)

# Bare OpenRouter (sk-or-) and OpenAI-style (sk-) tokens.
_TOKEN = re.compile(r"sk-(?:or-)?[A-Za-z0-9_\-]{8,}")


def redact_secrets(text: object) -> str:
    """Return ``text`` with secret values replaced by ``[REDACTED]``."""
    if text is None:
        return ""
    raw = text if isinstance(text, str) else str(text)
    if not raw:
        return raw

    def _replace_assignment(match: re.Match[str]) -> str:
        return f"{match.group('key')}{match.group('sep')}{_REDACTED}"

    redacted = _ASSIGNMENT.sub(_replace_assignment, raw)
    return _TOKEN.sub(_REDACTED, redacted)


class SecretRedactionFilter(logging.Filter):
    """Redact secrets on the log record before a handler formats it."""

    def filter(self, record: logging.LogRecord) -> bool:
        if isinstance(record.msg, str):
            record.msg = redact_secrets(record.msg)
        elif isinstance(record.msg, BaseException):
            record.msg = redact_secrets(record.msg)
        record.args = _redact_log_args(record.args)
        if isinstance(record.exc_text, str):
            record.exc_text = redact_secrets(record.exc_text)
        return True


class SecretRedactingFormatter(logging.Formatter):
    """Formatter that redacts the line actually written to the log file."""

    def formatException(self, ei) -> str:
        return redact_secrets(super().formatException(ei))

    def format(self, record: logging.LogRecord) -> str:
        return redact_secrets(super().format(record))


def install_secret_redaction(logger: logging.Logger) -> None:
    """Attach one redaction filter. Safe to call more than once."""
    logger.filters = [
        existing
        for existing in logger.filters
        if not isinstance(existing, SecretRedactionFilter)
    ]
    logger.addFilter(SecretRedactionFilter())


def _redact_log_value(value: object) -> object:
    if isinstance(value, str):
        return redact_secrets(value)
    if isinstance(value, BaseException):
        return redact_secrets(value)
    return value


def _redact_log_args(args: object) -> object:
    if not args:
        return args
    if isinstance(args, dict):
        return {key: _redact_log_value(value) for key, value in args.items()}
    if isinstance(args, tuple):
        return tuple(_redact_log_value(value) for value in args)
    return args
