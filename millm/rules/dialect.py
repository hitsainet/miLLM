"""The rule-signal pattern dialect (FR-016.11; FTDD 016 sections 5.1, 5.2; FTID section 3.2).

miLLM 039 S2 (TD39-24): a port of miDataworks' `backend/src/rules/dialect.py` (origin/main 2082e01),
kept line-for-line so the vendored vectors (`docs/schemas/rule-signal-cases-v1.json`) pin both apps
alike; miLLM imports no sibling code.

**Syntax:** RE2's, on the subset this scanner admits, with RE2's default options: case-sensitive
unless a signal declares ``ignore_case`` (RE2's Unicode simple case folding: ``k`` matches U+212A
KELVIN SIGN); ``.`` does not match ``\\n``; ``^`` and ``$`` match only at the start and end of the
text (``$`` does NOT match before a final ``\\n``); ``\\d``, ``\\w``, ``\\s``, ``\\b`` are ASCII and
``\\s`` is ``[\\t\\n\\f\\r ]`` (no ``\\v``). Spans are Unicode code points.

**Refused, each with the code-point position of the construct's first character:**
``backreference`` (``\\1``-``\\9``, ``(?P=name)``, ``\\k<name>``), ``lookaround``, ``conditional``,
``atomic``, ``possessive``, ``inline_flag`` (``(?i)``, ``(?s:...)``, ...: options are declared
fields only), ``unicode_property`` (``\\p``, ``\\P``), ``any_byte`` (``\\C``), ``repeat_too_large``
(a count over 1,000, or nested counted repeats whose product is over 1,000, as RE2 counts them),
``too_long`` (over 2,000 code points; position 2,000), and ``syntax`` - which also covers what one
of the engines in the suite reads differently and the dialect therefore leaves out: an escape
outside ``\\d \\D \\s \\S \\w \\W \\b \\B \\A \\z \\a \\f \\t \\n \\r \\v \\0 \\xHH`` and escaped
ASCII punctuation; a quantifier on ``^``, ``$``, ``\\b``, ``\\B``, ``\\A`` or ``\\z``; ``{,n}``;
``(?<name>``, a repeated or malformed group name; ``[`` inside a character class; a class escape
or reversed range as a range endpoint; unbalanced brackets or parentheses; a quantifier with
nothing to repeat or after another quantifier (a lazy ``?`` is allowed).

Only a pattern the scanner passed reaches ``re2.compile``, with ``log_errors=False`` so pattern text
never reaches standard error. An RE2 error the scanner did not foresee is refused as ``syntax`` with
no position; the conformance run lists each one as a scanner gap that must be fixed.

This module is the ONLY importer of ``re2`` and imports it unconditionally (FTDD 016 section 8.4):
an image without the wheel fails at start, never silently falls back to another engine.
"""

from __future__ import annotations

import functools
import re
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import re2  # type: ignore[import-untyped]

MAX_PATTERN_CHARS = 2000
#: The repeat-count limit ``regex_match@1`` already applies (``text_ops.MAX_REPEAT``) and RE2's own.
MAX_REPEAT = 1000

REFUSAL_RULES = (
    "backreference",
    "lookaround",
    "conditional",
    "atomic",
    "possessive",
    "inline_flag",
    "unicode_property",
    "any_byte",
    "repeat_too_large",
    "too_long",
    "syntax",
)

#: The dialect's behaviour rules, each pinned by a pattern case tagged ``dialect.<rule>`` in
#: ``rule-signal-cases-v1.json`` (FR-016.7: the coverage test reads this tuple).
DIALECT_RULES = (
    "end_anchor_only_at_end",
    "dot_excludes_newline",
    "ascii_digit_class",
    "ascii_word_class",
    "space_class_excludes_vertical_tab",
    "unicode_simple_case_folding",
    "empty_match_iteration",
)

_ESCAPE_ATOMS = set("dDsSwWaftnrv0")
_ESCAPE_ASSERTIONS = set("bBAz")
_CLASS_ESCAPES = set("dDsSwW")
_CLASS_LITERAL_ESCAPES = set("aftnrv0")
_HEX = set("0123456789abcdefABCDEF")
_QUANT = re.compile(r"\{(\d*)(?:(,)(\d*))?\}")
_GROUP_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


@dataclass(frozen=True)
class PatternRefused(ValueError):
    """A pattern outside the dialect: the rule and the code-point position (None: RE2 refused
    what the scanner passed - a scanner gap)."""

    rule: str
    position: int | None
    message: str

    def __str__(self) -> str:
        at = "" if self.position is None else f" at {self.position}"
        return f"{self.rule}{at}: {self.message}"


def _punct(ch: str) -> bool:
    return ch.isascii() and not ch.isalnum() and ch.isprintable()


@dataclass
class _Frame:
    open_at: int
    inner_max: int = 1


def scan(pattern: str) -> PatternRefused | None:
    """The first construct outside the dialect, or None. Positions are code points."""
    if len(pattern) > MAX_PATTERN_CHARS:
        return PatternRefused(
            "too_long", MAX_PATTERN_CHARS, f"a pattern is at most {MAX_PATTERN_CHARS} characters"
        )
    n = len(pattern)
    stack: list[_Frame] = [_Frame(-1)]
    names: set[str] = set()
    # What precedes the current position: "none" (nothing to repeat), "atom", "assertion",
    # "quant" (a quantifier), "lazy" (a quantifier made lazy).
    prev = "none"
    atom_product = 1
    quant_at = -1
    i = 0

    def refuse(rule: str, at: int, message: str) -> PatternRefused:
        return PatternRefused(rule, at, message)

    while i < n:
        c = pattern[i]
        if c == "\\":
            if i + 1 >= n:
                return refuse("syntax", i, "a trailing backslash escapes nothing")
            e = pattern[i + 1]
            if e in "123456789":
                return refuse("backreference", i, "backreferences are refused")
            if e == "k":
                return refuse("backreference", i, "backreferences are refused")
            if e in "pP":
                return refuse("unicode_property", i, "Unicode property classes are refused")
            if e == "C":
                return refuse("any_byte", i, "\\C matches one byte and is refused")
            if e == "x":
                if not (i + 3 < n and set(pattern[i + 2 : i + 4]) <= _HEX):
                    return refuse("syntax", i, "write a hex escape as \\xHH")
                prev, atom_product = "atom", 1
                i += 4
                continue
            if e in _ESCAPE_ATOMS:
                prev, atom_product = "atom", 1
            elif e in _ESCAPE_ASSERTIONS:
                prev = "assertion"
            elif _punct(e):
                prev, atom_product = "atom", 1
            else:
                return refuse("syntax", i, f"the escape \\{e} is not in the dialect")
            i += 2
            continue
        if c == "[":
            end, refused = _scan_class(pattern, i)
            if refused is not None:
                return refused
            prev, atom_product = "atom", 1
            i = end
            continue
        if c == "(":
            opened = i
            if pattern.startswith("(?", i):
                nxt = pattern[i + 2 : i + 3]
                if nxt == ":":
                    i += 3
                elif nxt == "P" and pattern[i + 3 : i + 4] == "<":
                    m = _GROUP_NAME.match(pattern, i + 4)
                    if m is None or pattern[m.end() : m.end() + 1] != ">":
                        return refuse("syntax", i, "a group name is letters, digits and _")
                    if m.group() in names:
                        return refuse("syntax", i, f"the group name {m.group()!r} is repeated")
                    names.add(m.group())
                    i = m.end() + 1
                elif nxt == "P" and pattern[i + 3 : i + 4] == "=":
                    return refuse("backreference", i, "backreferences are refused")
                elif nxt in ("=", "!") or pattern[i + 2 : i + 4] in ("<=", "<!"):
                    return refuse("lookaround", i, "lookahead and lookbehind are refused")
                elif nxt == "(":
                    return refuse("conditional", i, "conditionals are refused")
                elif nxt == ">":
                    return refuse("atomic", i, "atomic groups are refused")
                elif nxt and (nxt.isalpha() or nxt == "-"):
                    return refuse("inline_flag", i, "inline flags are refused; declare options")
                else:
                    return refuse("syntax", i, "this group form is not in the dialect")
            else:
                i += 1
            stack.append(_Frame(opened))
            prev = "none"
            continue
        if c == ")":
            if len(stack) == 1:
                return refuse("syntax", i, "unbalanced parenthesis")
            frame = stack.pop()
            stack[-1].inner_max = max(stack[-1].inner_max, frame.inner_max)
            prev, atom_product = "atom", frame.inner_max
            i += 1
            continue
        if c == "|":
            prev = "none"
            i += 1
            continue
        if c in "*+?":
            if prev == "quant" and c == "?":
                prev = "lazy"
                i += 1
                continue
            if prev in ("quant", "lazy") and c == "+":
                return refuse("possessive", quant_at, "possessive quantifiers are refused")
            if prev != "atom":
                return refuse("syntax", i, _nothing(prev))
            prev, quant_at = "quant", i
            stack[-1].inner_max = max(stack[-1].inner_max, atom_product)
            i += 1
            continue
        if c == "{":
            m = _QUANT.match(pattern, i)
            if m is not None and (m.group(1) or m.group(2)):
                if not m.group(1):
                    return refuse("syntax", i, "{,n} is not in the dialect; write {0,n}")
                if prev != "atom":
                    return refuse("syntax", i, _nothing(prev))
                low = int(m.group(1))
                high = None if m.group(2) and not m.group(3) else int(m.group(3) or m.group(1))
                if low > MAX_REPEAT or (high is not None and high > MAX_REPEAT):
                    return refuse("repeat_too_large", i, f"a repeat count is at most {MAX_REPEAT}")
                if high is not None and high < low:
                    return refuse("syntax", i, "the repeat range is reversed")
                product = (high if high is not None else low) * atom_product
                if product > MAX_REPEAT:
                    return refuse(
                        "repeat_too_large",
                        i,
                        f"nested repeat counts multiply to {product}; at most {MAX_REPEAT}",
                    )
                stack[-1].inner_max = max(stack[-1].inner_max, product)
                prev, quant_at = "quant", i
                i = m.end()
                continue
            prev, atom_product = "atom", 1
            i += 1
            continue
        if c in "^$":
            prev = "assertion"
            i += 1
            continue
        prev, atom_product = "atom", 1
        i += 1
    if len(stack) > 1:
        return PatternRefused("syntax", stack[-1].open_at, "unclosed parenthesis")
    return None


def _nothing(prev: str) -> str:
    if prev == "assertion":
        return "an anchor or boundary cannot be repeated"
    if prev in ("quant", "lazy"):
        return "a quantifier cannot follow another quantifier"
    return "nothing to repeat"


def _scan_class(pattern: str, start: int) -> tuple[int, PatternRefused | None]:
    """Scan ``[...]`` from ``start``; return the index after ``]`` or a refusal."""
    n = len(pattern)
    j = start + 1
    if j < n and pattern[j] == "^":
        j += 1
    first = True
    last_literal: tuple[int, str] | None = None  # (position, char) of a range's possible low end
    pending_range = False
    while j < n:
        c = pattern[j]
        if c == "]" and not first:
            return j + 1, None
        first = False
        if c == "[":
            return j, PatternRefused("syntax", j, "'[' inside a character class is refused")
        if c == "\\":
            if j + 1 >= n:
                return j, PatternRefused("syntax", j, "a trailing backslash escapes nothing")
            e = pattern[j + 1]
            if e in "pP":
                return j, PatternRefused(
                    "unicode_property", j, "Unicode property classes are refused"
                )
            if e == "C":
                return j, PatternRefused("any_byte", j, "\\C matches one byte and is refused")
            if e == "x":
                if not (j + 3 < n and set(pattern[j + 2 : j + 4]) <= _HEX):
                    return j, PatternRefused("syntax", j, "write a hex escape as \\xHH")
                char = chr(int(pattern[j + 2 : j + 4], 16))
                width = 4
            elif e in _CLASS_ESCAPES:
                if pending_range:
                    return j, PatternRefused("syntax", j, "a class escape cannot end a range")
                if pattern[j + 2 : j + 3] == "-" and pattern[j + 3 : j + 4] not in ("]", ""):
                    return j, PatternRefused("syntax", j, "a class escape cannot start a range")
                last_literal = None
                j += 2
                continue
            elif e in _CLASS_LITERAL_ESCAPES:
                char = {
                    "a": "\a",
                    "f": "\f",
                    "t": "\t",
                    "n": "\n",
                    "r": "\r",
                    "v": "\v",
                    "0": "\0",
                }[e]
                width = 2
            elif _punct(e):
                char, width = e, 2
            else:
                return j, PatternRefused("syntax", j, f"the escape \\{e} is not in a class")
        else:
            char, width = c, 1
        if c == "-" and last_literal is not None and not pending_range and j + 1 < n:
            if pattern[j + 1] != "]":
                pending_range = True
                j += 1
                continue
        if pending_range and last_literal is not None:
            if ord(char) < ord(last_literal[1]):
                return j, PatternRefused("syntax", last_literal[0], "the range is reversed")
            pending_range = False
            last_literal = None
        else:
            last_literal = (j, char)
        j += width
    return start, PatternRefused("syntax", start, "unterminated character class")


def check(pattern: str) -> None:
    """Raise :class:`PatternRefused` when the scanner refuses ``pattern``."""
    refused = scan(pattern)
    if refused is not None:
        raise refused


@functools.lru_cache(maxsize=4096)
def compile_pattern(pattern: str, ignore_case: bool) -> Any:
    """Compile a scanner-passed pattern with RE2 (cached per ``(pattern, ignore_case)``)."""
    check(pattern)
    options = re2.Options()
    options.log_errors = False
    options.case_sensitive = not ignore_case
    try:
        return re2.compile(pattern, options)
    except re2.error as exc:
        raise PatternRefused("syntax", None, f"RE2 refused it: {exc}") from None


def finditer(compiled: Any, text: str) -> Iterator[Any]:
    """Every non-overlapping match, left to right, with Python's rule for empty matches: after a
    match ending at ``e`` the search resumes at ``e``, one code point further when the match was
    empty. The ``re2`` binding's own ``finditer`` repeats empty matches after multi-byte text
    (measured 2026-10-23: ``\\ba?\\b`` on ``日7x`` gives (1, 1) twice), so it is never used."""
    pos, n = 0, len(text)
    while pos <= n:
        m = compiled.search(text, pos)
        if m is None:
            return
        yield m
        pos = m.end() + 1 if m.end() == m.start() else m.end()


#: U+FF01 - U+FF5E -> U+0021 - U+007E and U+3000 -> U+0020: one code point to one, so spans
#: need no mapping (FTDD 016 section 5.2).
FULLWIDTH_TABLE: dict[int, int] = {
    **{cp: cp - 0xFEE0 for cp in range(0xFF01, 0xFF5F)},
    0x3000: 0x20,
}


def fold_fullwidth_ascii(text: str) -> str:
    """Full-width ASCII forms and the ideographic space folded to ASCII, one-to-one."""
    return text.translate(FULLWIDTH_TABLE)


def escape_literal(text: str) -> str:
    """``text`` as a dialect pattern that matches it literally (punctuation escaped)."""
    return "".join("\\" + c if _punct(c) else c for c in text)
