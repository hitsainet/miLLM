"""Generic check-digit algorithms and the five generic validators (039 FR-39.19; the producer's
Q21 (a)): ``luhn``, ``mod97_10``, ``verhoeff``, ``weighted_mod`` and ``length_charset``.

Each validator checks a candidate's normalised value: ASCII spaces and hyphens are removed and
nothing else is changed; what remains must be ASCII digits (``mod97_10`` and ``length_charset``
with ``upper_alnum``: digits and upper-case letters), else ``illegal_character``. Its scan finds
maximal runs of its characters joined by single spaces or hyphens, in text order. ``tier`` and
``layout`` are always ``none``.

Every parameter comes from the rule set (weights, modulus, lengths): nothing here is specific to
one kind of value.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

from .base import Candidate, ValidatorDecl, Verdict

DIGITS = "0123456789"
UPPER = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
ALNUM_UPPER = UPPER + DIGITS

# ── algorithms ───────────────────────────────────────────────────────────────────


def luhn_mod10_ok(digits: str) -> bool:
    """Luhn (ISO/IEC 7812-1 Annex B): every second digit from the right doubled (9 subtracted
    above 9), the sum a multiple of 10."""
    total = 0
    for i, ch in enumerate(reversed(digits)):
        x = ord(ch) - 48
        if i % 2:
            x *= 2
            if x > 9:
                x -= 9
        total += x
    return total % 10 == 0


def letters_as_numbers(text: str) -> str:
    """ISO 7064 / ISO 13616 numeric form: digits kept, A=10 … Z=35."""
    return "".join(c if c in DIGITS else str(ord(c) - 55) for c in text)


def iso7064_mod97_10(text: str) -> int:
    """The number ``letters_as_numbers(text)`` modulo 97 (valid when 1)."""
    return int(letters_as_numbers(text)) % 97


def weighted_modulus_ok(digits: str, weights: list[int], modulus: int) -> bool:
    return sum((ord(c) - 48) * w for c, w in zip(digits, weights, strict=True)) % modulus == 0


_VERHOEFF_D = (
    (0, 1, 2, 3, 4, 5, 6, 7, 8, 9),
    (1, 2, 3, 4, 0, 6, 7, 8, 9, 5),
    (2, 3, 4, 0, 1, 7, 8, 9, 5, 6),
    (3, 4, 0, 1, 2, 8, 9, 5, 6, 7),
    (4, 0, 1, 2, 3, 9, 5, 6, 7, 8),
    (5, 9, 8, 7, 6, 0, 4, 3, 2, 1),
    (6, 5, 9, 8, 7, 1, 0, 4, 3, 2),
    (7, 6, 5, 9, 8, 2, 1, 0, 4, 3),
    (8, 7, 6, 5, 9, 3, 2, 1, 0, 4),
    (9, 8, 7, 6, 5, 4, 3, 2, 1, 0),
)
_VERHOEFF_P = (
    (0, 1, 2, 3, 4, 5, 6, 7, 8, 9),
    (1, 5, 7, 6, 2, 8, 3, 0, 9, 4),
    (5, 8, 0, 3, 7, 9, 6, 1, 4, 2),
    (8, 9, 1, 6, 0, 4, 3, 5, 2, 7),
    (9, 4, 5, 3, 1, 2, 6, 8, 7, 0),
    (4, 2, 8, 6, 5, 7, 3, 9, 0, 1),
    (2, 7, 9, 3, 8, 0, 6, 4, 1, 5),
    (7, 0, 4, 6, 9, 1, 3, 2, 5, 8),
)


def verhoeff_ok(digits: str) -> bool:
    """Verhoeff (dihedral group D5) check digit."""
    c = 0
    for i, ch in enumerate(reversed(digits)):
        c = _VERHOEFF_D[c][_VERHOEFF_P[i % 8][ord(ch) - 48]]
    return c == 0


def mod11_weights_10_to_2_digit(body9: str) -> str | None:
    """Mod 11 over weights 10..2: 11 -> "0"; 10 means no valid value has this body (None)."""
    total = sum((ord(c) - 48) * w for c, w in zip(body9, range(10, 1, -1), strict=True))
    check = 11 - total % 11
    if check == 11:
        return "0"
    if check == 10:
        return None
    return str(check)


# ── the generic validators ───────────────────────────────────────────────────────

GENERIC_OUTPUTS: tuple[tuple[str, str], ...] = (
    ("valid", "bool"),
    ("reason", "string"),
    ("normalised", "string"),
)

_DIGIT_RUN = re.compile(r"[0-9](?:[ -]?[0-9])*")
_ALNUM_RUN = re.compile(r"[0-9A-Z](?:[ -]?[0-9A-Z])*")
_UPPER_ALNUM = frozenset(ALNUM_UPPER)
_DIGITS = frozenset(DIGITS)


def _normalise(value: str) -> str:
    return value.replace(" ", "").replace("-", "")


def _verdict(valid: bool, reason: str, normalised: str) -> Verdict:
    return Verdict(valid, reason, "none", "none", normalised, None)


class _Generic:
    decl: ValidatorDecl
    _run: re.Pattern[str] = _DIGIT_RUN

    def check(self, value: str, before: str, after: str, params: Mapping[str, Any]) -> Verdict:
        raise NotImplementedError

    def _candidate_pattern(self, params: Mapping[str, Any]) -> re.Pattern[str]:
        return self._run

    def scan(self, text: str, params: Mapping[str, Any]) -> list[Candidate]:
        return [
            Candidate(m.start(), m.end(),
                      self.check(m.group(), text[: m.start()], text[m.end():], params))
            for m in self._candidate_pattern(params).finditer(text)
        ]


def _digits_or_refusal(value: str, minimum: int) -> tuple[str, Verdict | None]:
    d = _normalise(value)
    if not d:
        return d, _verdict(False, "no_digits", d)
    if any(c not in _DIGITS for c in d):
        return d, _verdict(False, "illegal_character", d)
    if len(d) < minimum:
        return d, _verdict(False, "too_short", d)
    return d, None


_NO_PARAMS: dict[str, Any] = {"type": "object", "properties": {}, "additionalProperties": False}


class Luhn(_Generic):
    decl = ValidatorDecl(
        id="luhn", version=1,
        description="Luhn (ISO/IEC 7812-1 Annex B) check digit over ASCII digits; spaces and "
                    "hyphens are ignored; at least two digits.",
        params_schema=_NO_PARAMS,
        reasons=("ok", "no_digits", "illegal_character", "too_short", "bad_check_digit"),
        outputs=GENERIC_OUTPUTS, algorithms=("luhn_mod10",),
    )

    def check(self, value: str, before: str, after: str, params: Mapping[str, Any]) -> Verdict:
        d, refused = _digits_or_refusal(value, 2)
        if refused is not None:
            return refused
        if not luhn_mod10_ok(d):
            return _verdict(False, "bad_check_digit", d)
        return _verdict(True, "ok", d)


class Verhoeff(_Generic):
    decl = ValidatorDecl(
        id="verhoeff", version=1,
        description="Verhoeff check digit (dihedral group D5) over ASCII digits; spaces and "
                    "hyphens are ignored; at least two digits.",
        params_schema=_NO_PARAMS,
        reasons=("ok", "no_digits", "illegal_character", "too_short", "bad_check_digit"),
        outputs=GENERIC_OUTPUTS, algorithms=("verhoeff",),
    )

    def check(self, value: str, before: str, after: str, params: Mapping[str, Any]) -> Verdict:
        d, refused = _digits_or_refusal(value, 2)
        if refused is not None:
            return refused
        if not verhoeff_ok(d):
            return _verdict(False, "bad_check_digit", d)
        return _verdict(True, "ok", d)


class Mod97_10(_Generic):
    decl = ValidatorDecl(
        id="mod97_10", version=1,
        description="ISO 7064 MOD 97-10 over ASCII digits and upper-case letters (A=10 ... Z=35); "
                    "valid when the number mod 97 is 1. The rearranging mode first moves the first "
                    "four characters to the end (the ISO 13616 rearrangement); rearrange=none "
                    "checks the value as written.",
        params_schema={
            "type": "object",
            "properties": {"rearrange": {"type": "string", "enum": ["none", "iban"]}},
            "required": ["rearrange"], "additionalProperties": False,
        },
        reasons=("ok", "no_characters", "illegal_character", "too_short", "bad_check_digit"),
        outputs=GENERIC_OUTPUTS, algorithms=("iso7064_mod97_10",),
    )
    _run = _ALNUM_RUN

    def check(self, value: str, before: str, after: str, params: Mapping[str, Any]) -> Verdict:
        t = _normalise(value)
        if not t:
            return _verdict(False, "no_characters", t)
        if any(c not in _UPPER_ALNUM for c in t):
            return _verdict(False, "illegal_character", t)
        rearranged = params.get("rearrange") != "none"
        if len(t) < (5 if rearranged else 2):
            return _verdict(False, "too_short", t)
        arranged = t[4:] + t[:4] if rearranged else t
        if iso7064_mod97_10(arranged) != 1:
            return _verdict(False, "bad_check_digit", t)
        return _verdict(True, "ok", t)


class WeightedMod(_Generic):
    decl = ValidatorDecl(
        id="weighted_mod", version=1,
        description="Weighted sum check: exactly one digit per weight and sum(digit_i * weight_i) "
                    "mod modulus is 0, the check digit last (check_position=last). Weights and "
                    "modulus are the rule set's.",
        params_schema={
            "type": "object",
            "properties": {
                "weights": {"type": "array",
                            "items": {"type": "integer", "minimum": 0, "maximum": 99},
                            "minItems": 2, "maxItems": 64},
                "modulus": {"type": "integer", "minimum": 2, "maximum": 97},
                "check_position": {"type": "string", "enum": ["last"]},
            },
            "required": ["weights", "modulus", "check_position"], "additionalProperties": False,
        },
        reasons=("ok", "no_digits", "illegal_character", "wrong_length", "bad_check_digit"),
        outputs=GENERIC_OUTPUTS, algorithms=("weighted_modulus",),
    )

    def check(self, value: str, before: str, after: str, params: Mapping[str, Any]) -> Verdict:
        d, refused = _digits_or_refusal(value, 1)
        if refused is not None:
            return refused
        weights = [int(w) for w in params["weights"]]
        if len(d) != len(weights):
            return _verdict(False, "wrong_length", d)
        if not weighted_modulus_ok(d, weights, int(params["modulus"])):
            return _verdict(False, "bad_check_digit", d)
        return _verdict(True, "ok", d)


class LengthCharset(_Generic):
    decl = ValidatorDecl(
        id="length_charset", version=1,
        description="Length and character set only (no check digit): the normalised value's length "
                    "is one of lengths and every character is in the charset (digits: 0-9; "
                    "upper_alnum: 0-9 and A-Z).",
        params_schema={
            "type": "object",
            "properties": {
                "lengths": {"type": "array",
                            "items": {"type": "integer", "minimum": 1, "maximum": 64},
                            "minItems": 1, "maxItems": 16, "uniqueItems": True},
                "charset": {"type": "string", "enum": ["digits", "upper_alnum"]},
            },
            "required": ["lengths", "charset"], "additionalProperties": False,
        },
        reasons=("ok", "no_characters", "illegal_character", "wrong_length"),
        outputs=GENERIC_OUTPUTS, algorithms=("length_and_charset",),
    )

    def _candidate_pattern(self, params: Mapping[str, Any]) -> re.Pattern[str]:
        return _ALNUM_RUN if params.get("charset") == "upper_alnum" else _DIGIT_RUN

    def check(self, value: str, before: str, after: str, params: Mapping[str, Any]) -> Verdict:
        t = _normalise(value)
        if not t:
            return _verdict(False, "no_characters", t)
        allowed = _UPPER_ALNUM if params["charset"] == "upper_alnum" else _DIGITS
        if any(c not in allowed for c in t):
            return _verdict(False, "illegal_character", t)
        if len(t) not in {int(n) for n in params["lengths"]}:
            return _verdict(False, "wrong_length", t)
        return _verdict(True, "ok", t)


GENERIC_VALIDATORS: tuple[_Generic, ...] = (Luhn(), Mod97_10(), Verhoeff(), WeightedMod(),
                                            LengthCharset())
