"""RFC 8941 string items and the bounded `X-miLLM-Probe-Verdicts` list (039 FR-39.12, TD39-5, TD39-30).

⚠ ONE STRING SERIALIZER. Every string member or parameter the header carries — probe names,
behaviour keys, probe ids, windows, reason codes — goes through `sf_string`. Before 039 the builder
only stripped `"` from a name, so a probe named with a backslash or a non-ASCII character (names
are unconstrained at import) produced a header no structured-field parser accepts (FTID
Discrepancy 6). RFC 8941 §3.3.3: a String is printable ASCII (0x20–0x7E) in double quotes, with
`\\` and `"` escaped by a backslash.

⚠ THE TRUNCATION MEMBER IS ASCII. The FPRD wrote `"…"` (U+2026); a header containing it does not
parse (FTDD §6.5 design correction), so it is `"..."`, with `truncated` and `total=<n>` parameters.
"""

from __future__ import annotations

from typing import Iterable, Sequence

#: The final member when whole behaviour groups had to be left out (`total` is added).
TRUNCATION_TOKEN = "..."


def sf_string(value: object) -> str:
    """`value` as an RFC 8941 String: printable ASCII, `\\` and `"` escaped, anything else `?`."""
    out = []
    for char in str(value):
        code = ord(char)
        if char in ('\\', '"'):
            out.append("\\" + char)
        elif 0x20 <= code <= 0x7E:
            out.append(char)
        else:
            out.append("?")
    return '"' + "".join(out) + '"'


#: RFC 8941 §3.3.2: a Decimal has at most 12 integer digits and at most 3 fractional digits.
_DECIMAL_MAX = 999_999_999_999.999


def sf_decimal(value: float) -> str:
    """`value` as an RFC 8941 Decimal: rounded to 3 fractional digits, trailing zeros dropped
    (one kept), clamped to the 12 integer digits the grammar allows.

    ⚠ PRE-039 DEFECT, FIXED HERE (039 FR-39.12 requires the header to parse): scores were written
    with `:.6g`, which produces `-0.0105608` (7 fractional digits) or `1.2e-05` (an exponent) —
    neither is RFC 8941, and an independent parser (`http-sfv`) rejects the whole header. The
    header is the summary; the full-precision score is in the body field, the chunk and the events.
    """
    x = max(-_DECIMAL_MAX, min(_DECIMAL_MAX, float(value)))
    text = f"{x:.3f}".rstrip("0")
    if text.endswith("."):
        text += "0"
    return "0.0" if text in ("-0.0", "-0") else text


def truncation_member(total: int) -> str:
    return f'{sf_string(TRUNCATION_TOKEN)};truncated;total={int(total)}'


def bounded_members(groups: Sequence[Sequence[str]], max_bytes: int, total: int) -> str:
    """Whole groups, in order, while the list (plus the truncation member, should one be needed)
    stays within `max_bytes`; then the truncation member. Never a cut group, never a cut member.

    `groups` are one behaviour's members each, already serialized; `total` is the number of
    results the full list carries (every member of every group).
    """
    every = [m for group in groups for m in group]
    full = ", ".join(every)
    if len(full.encode("ascii")) <= max_bytes:
        return full
    tail = truncation_member(total)
    kept: list[str] = []
    for group in groups:
        candidate = kept + list(group)
        if len(", ".join(candidate + [tail]).encode("ascii")) > max_bytes:
            break
        kept = candidate
    return ", ".join(kept + [tail])


def bounded_members_keeping_leaders(leaders: Sequence[Sequence[str]], groups: Sequence[Sequence[str]],
                                    max_bytes: int, total: int) -> str:
    """ENH-006 G19: every group's LEADER (its behaviour verdict) is kept before any group's members:
    the leaders first claim the space (in order, while they and the truncation member fit), then whole
    member groups in order while they fit. Each group's kept leader stays ahead of its members, so the
    order is the results' own. Never a cut member."""
    every = [m for lead, group in zip(leaders, groups) for m in [*lead, *group]]
    full = ", ".join(every)
    if len(full.encode("ascii")) <= max_bytes:
        return full
    tail = truncation_member(total)

    def size(kept_leads: list[list[str]], kept_groups: list[list[str]]) -> int:
        flat = [m for lead, group in zip(kept_leads, kept_groups) for m in [*lead, *group]]
        return len(", ".join(flat + [tail]).encode("ascii"))

    kept_leads: list[list[str]] = [[] for _ in leaders]
    kept_groups: list[list[str]] = [[] for _ in groups]
    for i, lead in enumerate(leaders):
        trial = [list(x) for x in kept_leads]
        trial[i] = list(lead)
        if size(trial, kept_groups) > max_bytes:
            break
        kept_leads = trial
    for i, group in enumerate(groups):
        if not group:
            continue
        trial = [list(x) for x in kept_groups]
        trial[i] = list(group)
        if size(kept_leads, trial) > max_bytes:
            break
        kept_groups = trial
    flat = [m for lead, group in zip(kept_leads, kept_groups) for m in [*lead, *group]]
    return ", ".join(flat + [tail])


def join(members: Iterable[str]) -> str:
    return ", ".join(members)
