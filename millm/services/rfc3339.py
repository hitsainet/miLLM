"""RFC 3339 `date-time` checking for the vendored manifest schemas (addendum D-29).

jsonschema checks `format: date-time` only when `rfc3339-validator` is installed, and it is not, so
without this the format is silently skipped and any string passes. `datetime.fromisoformat` alone is
too lenient on Python 3.11+: it accepts ISO 8601 forms RFC 3339 does not (basic format
`20261013T120000Z`, minutes without seconds, a space separator, no offset). The rule is therefore
the RFC 3339 grammar first, then `fromisoformat` for the calendar (month 13, February 30).

`fullmatch`, never `match` with `$`: `$` also matches before a trailing newline, so
`"2026-10-13T12:00:00Z\\n"` would pass a `match`.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone

from jsonschema import FormatChecker

_RFC3339 = re.compile(
    r"\d{4}-\d{2}-\d{2}[Tt]\d{2}:\d{2}:\d{2}(\.\d+)?([Zz]|[+-]\d{2}:\d{2})"
)


def is_rfc3339(value: object) -> bool:
    """True when `value` is an RFC 3339 date-time string naming a real instant."""
    if not isinstance(value, str) or not _RFC3339.fullmatch(value):
        return False
    try:
        parsed = datetime.fromisoformat(value.upper().replace("Z", "+00:00"))
    except ValueError:
        return False
    return parsed.tzinfo is not None


def utc_now_rfc3339() -> str:
    """The current instant as RFC 3339 in UTC with a `Z` suffix, whole seconds."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def date_time_format_checker() -> FormatChecker:
    """A `FormatChecker` whose `date-time` is `is_rfc3339`, whatever optional packages exist."""
    checker = FormatChecker()
    checker.checks("date-time")(is_rfc3339)
    return checker
