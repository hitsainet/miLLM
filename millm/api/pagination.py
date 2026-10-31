"""Keyset pagination cursors (039 FR-39.15, TD39-29, FTID §5.2).

A cursor is opaque to the caller: base64url of the JSON list of the last item's sort values, plus a
tag naming the list it belongs to, so a cursor from one list handed to another is refused rather
than misread. A cursor this server cannot decode is `INVALID_CURSOR` (400), never a 500.

⚠ KEYSET, NOT OFFSET. An offset page skips or repeats a row whenever the list changes between two
reads (an import, a prune); a keyset page continues strictly after the last row it returned, in an
order that includes the primary key as its final tie-break, so every item appears exactly once.
"""

from __future__ import annotations

import base64
import binascii
import json
from typing import Any, Optional, Sequence

from millm.core.errors import InvalidCursorError


def encode_cursor(tag: str, values: Sequence[Any]) -> str:
    raw = json.dumps([tag, list(values)], separators=(",", ":"), default=str)
    return base64.urlsafe_b64encode(raw.encode("utf-8")).decode("ascii").rstrip("=")


def decode_cursor(tag: str, cursor: Optional[str], *, arity: int) -> Optional[list[Any]]:
    """The sort values a cursor carries, or None when there is no cursor. Refuses anything else."""
    if cursor is None or cursor == "":
        return None
    try:
        padded = cursor + "=" * (-len(cursor) % 4)
        decoded = json.loads(base64.urlsafe_b64decode(padded.encode("ascii")).decode("utf-8"))
    except (ValueError, binascii.Error, UnicodeError):
        raise InvalidCursorError("This cursor was not issued by this list", details={"list": tag})
    if (not isinstance(decoded, list) or len(decoded) != 2 or decoded[0] != tag
            or not isinstance(decoded[1], list) or len(decoded[1]) != arity):
        raise InvalidCursorError("This cursor was not issued by this list", details={"list": tag})
    return decoded[1]


def page_meta(next_cursor: Optional[str], limit: int) -> dict[str, Any]:
    """The `meta` block every paginated list returns."""
    return {"next": next_cursor, "limit": limit}
