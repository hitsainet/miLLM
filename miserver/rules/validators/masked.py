"""``masked@1``: does a candidate read as a MASKED value of a format? (the producer's FR-016.P1)

A port of miDataworks' ``backend/src/rules/validators/masked.py`` (origin/main 2082e01); the vendored
``masked@1`` vectors pin it. ``valid`` here means "reads masked"; it bears no identifier
(``value_bearing=False``). The rule, in order:

1. one mask character (``X x * # • ● █``) per value, else ``mixed_mask_characters``;
2. only ASCII digits, the mask character and the format's delimiters (the country-account format
   also keeps upper-case letters), else ``illegal_character``;
3. mask and kept characters fill one of the format's lengths, else ``wrong_length``;
4. no delimiter, or one at a popular group layout, else ``irregular_layout``;
5. kept characters form a leading and/or trailing run within the format's maximum
   (``kept_characters_misplaced`` / ``too_many_kept``);
6. a neighbouring letter or digit reads ``alphanumeric_boundary``.

Its own scan: a candidate starts at a mask character or digit whose left neighbour is not a letter
or digit, and runs over those characters and the format's delimiters.

⚠ Serving never applies a veto (OQ-39.2 (a), D9): this validator serves the rule set's own cases
and any signal an operator writes with it; it never removes a probe's or a signal's result.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .algorithms import DIGITS, UPPER
from .base import Candidate, MaskShape, ValidatorDecl, Verdict
from .formats import COUNTRY_ACCOUNT, COUNTRY_ACCOUNT_REGISTRY, MASK_SHAPES, digit_layout_popular

#: The mask characters (the producer's FTDD 016 §15.1).
MASK_CHARACTERS = "Xx*#•●█"

_DIGITS = frozenset(DIGITS)
_UPPER = frozenset(UPPER)

MASKED_REASONS = (
    "masked",
    "no_characters",
    "illegal_character",
    "no_mask_character",
    "mixed_mask_characters",
    "wrong_length",
    "irregular_layout",
    "kept_characters_misplaced",
    "too_many_kept",
    "alphanumeric_boundary",
)


def _verdict(masked: bool, reason: str, normalised: str, layout: str, fmt: str) -> Verdict:
    return Verdict(masked, reason, "none", layout, normalised, fmt)


def _country_trailing_letters_kept(kinds: str, positions: list[str], window: int) -> str:
    """A country-account value masked with ``X`` reads its last ``window`` characters as kept
    from the first character that is not ``X`` onward (the account part may end in the letter X)."""
    n = len(kinds)
    tail = positions[n - window:]
    first = next((i for i, c in enumerate(tail) if c != "X"), None)
    if first is None:
        return kinds
    return kinds[: n - window + first] + "k" * (window - first)


def _quads(n: int) -> tuple[int, ...]:
    return tuple([4] * (n // 4) + ([n % 4] if n % 4 else []))


class Masked:
    decl = ValidatorDecl(
        id="masked",
        version=1,
        description="Reads whether a candidate is a MASKED value of a format (valid = masked): "
        "one mask character (X x * # • ● █) with kept digits only at the start "
        "and end, filling the format's length in a popular layout.",
        params_schema={
            "type": "object",
            "properties": {"format": {"type": "string", "enum": sorted(MASK_SHAPES)}},
            "required": ["format"],
            "additionalProperties": False,
        },
        reasons=MASKED_REASONS,
        outputs=(
            ("valid", "bool"),
            ("reason", "string"),
            ("layout", "string"),
            ("normalised", "string"),
            ("detail", "string"),
        ),
        algorithms=("mask_shape",),
        value_bearing=False,
    )

    def check(self, value: str, before: str, after: str, params: Mapping[str, Any]) -> Verdict:
        fmt = str(params["format"])
        shape = MASK_SHAPES[fmt]
        country = fmt == COUNTRY_ACCOUNT
        if not value:
            return _verdict(False, "no_characters", "", "none", fmt)
        allowed = (
            _DIGITS | set(MASK_CHARACTERS) | set(shape.delimiters) | (_UPPER if country else set())
        )
        if any(c not in allowed for c in value):
            return _verdict(False, "illegal_character", "", "none", fmt)
        positions = [c for c in value if c not in shape.delimiters]
        normalised = "".join(positions)
        masks = {c for c in positions if c in MASK_CHARACTERS}
        if not masks:
            return _verdict(False, "no_mask_character", normalised, "none", fmt)
        if country and len(masks) == 2 and "X" in masks:
            masks.discard("X")  # masked with another character, its X is a letter
        if len(masks) > 1:
            return _verdict(False, "mixed_mask_characters", normalised, "none", fmt)
        mask = next(iter(masks))
        n = len(positions)
        if n not in self._lengths(shape, positions, mask, country):
            return _verdict(False, "wrong_length", normalised, "none", fmt)
        core = "".join("0" if c not in shape.delimiters else c for c in value)
        layouts = (_quads(n),) if country else shape.layouts
        if not digit_layout_popular(core, layouts, shape.delimiters):
            return _verdict(False, "irregular_layout", normalised, "irregular", fmt)
        kinds = "".join("m" if c == mask else "k" for c in positions)
        if country and mask == "X":
            kinds = _country_trailing_letters_kept(kinds, positions, shape.keep_trailing)
        lead = len(kinds) - len(kinds.lstrip("k"))
        trail = len(kinds) - len(kinds.rstrip("k"))
        if "k" in kinds[lead: n - trail] or lead == n:
            return _verdict(False, "kept_characters_misplaced", normalised, "popular", fmt)
        if lead > shape.keep_leading or trail > shape.keep_trailing:
            return _verdict(False, "too_many_kept", normalised, "popular", fmt)
        if country and not all(
            (c in _UPPER) if i < 2 else (c in _DIGITS) for i, c in enumerate(positions[:lead])
        ):
            return _verdict(False, "kept_characters_misplaced", normalised, "popular", fmt)
        if not country and any(c not in _DIGITS for c in positions if c != mask):
            return _verdict(False, "kept_characters_misplaced", normalised, "popular", fmt)
        if before[-1:].isalnum() or after[:1].isalnum():
            return _verdict(False, "alphanumeric_boundary", normalised, "popular", fmt)
        return _verdict(True, "masked", normalised, "popular", fmt)

    @staticmethod
    def _lengths(shape: MaskShape, positions: list[str], mask: str,
                 country: bool) -> tuple[int, ...]:
        if country and len(positions) >= 2 and mask not in positions[:2]:
            code = "".join(positions[:2])
            if code in COUNTRY_ACCOUNT_REGISTRY:
                return (COUNTRY_ACCOUNT_REGISTRY[code][0],)
        return shape.lengths

    def scan(self, text: str, params: Mapping[str, Any]) -> list[Candidate]:
        fmt = str(params["format"])
        shape = MASK_SHAPES[fmt]
        country = fmt == COUNTRY_ACCOUNT
        body = _DIGITS | set(MASK_CHARACTERS) | (_UPPER if country else set())
        delims = set(shape.delimiters)
        out: list[Candidate] = []
        i, n = 0, len(text)
        while i < n:
            if not self._starts(text, i, body, country):
                i += 1
                continue
            j = i
            while j < n and text[j] in body:
                j += 1
            while j < n and text[j] in delims:
                k = j
                while k < n and text[k] in delims:
                    k += 1
                if k < n and text[k] in body:
                    j = k
                    while j < n and text[j] in body:
                        j += 1
                else:
                    break
            out.append(Candidate(i, j, self.check(text[i:j], text[:i], text[j:], params)))
            i = j
        return out

    @staticmethod
    def _starts(text: str, i: int, body: Any, country: bool) -> bool:
        if text[i] not in body or (i > 0 and text[i - 1].isalnum()):
            return False
        if not country:
            return True
        head = text[i: i + 4]
        masks = set(MASK_CHARACTERS)
        return (
            len(head) == 4
            and all(c in _UPPER or c in masks for c in head[:2])
            and all(c in _DIGITS or c in masks for c in head[2:])
        )


MASKED = Masked()
