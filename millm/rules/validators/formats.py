"""Version 1 of the six format validators the producer registers (039 TD39-27; the producer's
FR-016.1, FR-016.3), and version 2 of the country-account format (the producer's D38-1).

Each is the composition of generic rules — an issuer-range-and-length table with a Luhn check, an
area/group/serial range table, a 3-7-1 weighted sum with a prefix table, a country length and
layout table with ISO 7064 MOD 97-10, a 10-digit mod 11, a 12-digit Verhoeff with a leading-digit
rule — whose tables are data in this module. The registered ID of each is the producer's and is
the only place a format is named (as data, read by the registry: `FORMAT_IDS`).

``scan`` finds candidates the way the producer's ``checksum_validate@1`` does: a digit run joined by
the format's delimiters (Python ``re``, ASCII by construction), or for the country-account format a
two-letter, two-digit head and what follows; each candidate is checked with the text on either side.
Spans are code points. Every reason literal is declared in ``FORMAT_REASONS`` and pinned by the
vendored vectors.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any, Callable

from .algorithms import (
    ALNUM_UPPER,
    DIGITS,
    UPPER,
    iso7064_mod97_10,
    luhn_mod10_ok,
    mod11_weights_10_to_2_digit,
    verhoeff_ok,
    weighted_modulus_ok,
)
from .base import Candidate, MaskShape, ValidatorDecl, Verdict

#: The registered ids, as the producer registers them (data; the generality guard exempts exactly
#: the ids it reads from the live registry, `millm.rules.validators.VALIDATORS`).
ISSUER_CARD, AREA_SERIAL, BANK_TRANSIT, COUNTRY_ACCOUNT, HEALTH_MOD11, RESIDENT_VERHOEFF = (
    "card_pan", "ssn_us", "aba_routing", "iban", "nhs", "aadhaar")

FORMAT_OUTPUTS: tuple[tuple[str, str], ...] = (
    ("valid", "bool"), ("reason", "string"), ("tier", "string"), ("layout", "string"),
    ("normalised", "string"), ("detail", "string"),
)


def _none(reason: str, normalised: str = "", detail: str | None = None) -> Verdict:
    return Verdict(False, reason, "none", "none", normalised, detail)


def core_parts(s: str) -> tuple[str, str, str, str] | None:
    """``(core, char before, char after, digits)``: the core runs from the first to the last ASCII
    digit of ``s`` (which carries one boundary character on each side when known)."""
    idx = [i for i, c in enumerate(s) if c in DIGITS]
    if not idx:
        return None
    a, b = idx[0], idx[-1]
    return s[a: b + 1], s[:a][-1:], s[b + 1:][:1], "".join(s[i] for i in idx)


def _only(core: str, delimiters: str) -> bool:
    return all(c in DIGITS or c in delimiters for c in core)


def digit_layout_popular(core: str, groups_options: Sequence[Sequence[int]], delims: str) -> bool:
    """No delimiter, or ONE repeated delimiter at exactly one of the popular group layouts."""
    if all(c in DIGITS for c in core):
        return True
    seps = {c for c in core if c not in DIGITS}
    if len(seps) != 1:
        return False
    sep = next(iter(seps))
    if sep not in delims:
        return False
    parts = core.split(sep)
    return any([len(p) for p in parts] == list(g) for g in groups_options)


# ── issuer range and length, with Luhn ───────────────────────────────────────────


def issuer_brand(d: str) -> str | None:
    """The brand whose issuer range AND length this number matches, else None."""
    n = len(d)
    if n < 2:
        return None
    p2 = int(d[:2])
    p3 = int(d[:3]) if n >= 3 else -1
    p4 = int(d[:4]) if n >= 4 else -1
    if d[0] == "4" and n in (13, 16, 19):
        return "visa"
    if (51 <= p2 <= 55 or 2221 <= p4 <= 2720) and n == 16:
        return "mastercard"
    if p2 in (34, 37) and n == 15:
        return "amex"
    if (p2 in (36, 38) or 300 <= p3 <= 305) and n == 14:
        return "diners"
    if (p4 == 6011 or 644 <= p3 <= 649 or p2 == 65) and 16 <= n <= 19:
        return "discover"
    if 3528 <= p4 <= 3589 and 16 <= n <= 19:
        return "jcb"
    return None


def issuer_prefix_known(d: str) -> bool:
    """Some brand's issuer range starts this number (length aside)."""
    if len(d) < 4:
        return False
    p2, p3, p4 = int(d[:2]), int(d[:3]), int(d[:4])
    return (
        d[0] == "4" or 51 <= p2 <= 55 or 2221 <= p4 <= 2720 or p2 in (34, 37, 36, 38, 65)
        or 300 <= p3 <= 305 or p4 == 6011 or 644 <= p3 <= 649 or 3528 <= p4 <= 3589
    )


def check_issuer_luhn(value: str, before: str, after: str, delimiters: str) -> Verdict:
    parts = core_parts(before[-1:] + value + after[:1])
    if parts is None:
        return _none("no_digits")
    core, pre, post, d = parts
    brand = issuer_brand(d)
    if not _only(core, delimiters):
        return _none("illegal_character", d)
    if not 12 <= len(d) <= 19:
        valid, reason = False, "wrong_length"
    elif not luhn_mod10_ok(d):
        valid, reason = False, "bad_check_digit"
    elif brand is None:
        valid, reason = False, ("wrong_length" if issuer_prefix_known(d) else "bad_prefix")
    else:
        valid, reason = True, "ok"
    popular = digit_layout_popular(core, ((4, 4, 4, 4), (4, 6, 5), (4, 6, 4)), delimiters)
    layout = "popular" if popular else "irregular"
    if pre.isalnum() or post.isalnum():
        return Verdict(valid, "alphanumeric_boundary", "none", layout, d, brand)
    if not (12 <= len(d) <= 19 and luhn_mod10_ok(d)):
        return Verdict(valid, reason, "none", layout, d, brand)
    if popular and brand is not None:
        return Verdict(valid, "ok", "medium", layout, d, brand)
    return Verdict(valid, reason if not valid else "irregular_layout", "low", layout, d, brand)


# ── area, group and serial ranges ────────────────────────────────────────────────


def area_group_serial_ok(d: str) -> bool:
    if len(d) != 9:
        return False
    area, group, serial = int(d[:3]), int(d[3:5]), int(d[5:])
    return area not in (0, 666) and area < 900 and group != 0 and serial != 0


def check_area_group_serial(value: str, before: str, after: str, delimiters: str) -> Verdict:
    parts = core_parts(before[-1:] + value + after[:1])
    if parts is None:
        return _none("no_digits")
    core, pre, post, d = parts
    if not _only(core, delimiters):
        return _none("illegal_character", d)
    if len(d) != 9:
        return _none("wrong_length", d)
    if not area_group_serial_ok(d):
        return _none("bad_range", d)
    popular = core.isdigit() or (
        len(core) == 11 and core[3] in delimiters and core[6] == core[3]
        and core.replace(core[3], "").isdigit()
    )
    layout = "popular" if popular else "irregular"
    if popular and not pre.isalpha() and not post.isalpha():
        return Verdict(True, "ok", "medium", layout, d)
    return Verdict(True, "irregular_layout" if not popular else "letter_boundary", "low", layout, d)


# ── 3-7-1 weighted sum with a prefix table ───────────────────────────────────────

WEIGHTS_371 = [3, 7, 1] * 3


def transit_prefix_ok(d: str) -> bool:
    p = int(d[:2])
    return p <= 12 or 21 <= p <= 32 or 61 <= p <= 72 or p == 80


def check_weighted_371_prefix(value: str, before: str, after: str, delimiters: str) -> Verdict:
    parts = core_parts(before[-1:] + value + after[:1])
    if parts is None:
        return _none("no_digits")
    core, pre, post, d = parts
    if not _only(core, delimiters):
        return _none("illegal_character", d)
    if len(d) != 9:
        return _none("wrong_length", d)
    if not weighted_modulus_ok(d, WEIGHTS_371, 10):
        return _none("bad_check_digit", d)
    if not transit_prefix_ok(d):
        return _none("bad_prefix", d)
    popular = core.isdigit()
    layout = "popular" if popular else "irregular"
    if popular and not pre.isalpha() and not post.isalpha():
        return Verdict(True, "ok", "medium", layout, d)
    return Verdict(True, "irregular_layout" if not popular else "letter_boundary", "low", layout, d)


# ── country, length and layout with MOD 97-10 ───────────────────────────────────

#: country -> (length, account layout, name). The layout is `<count><kind>` runs: `n` a digit,
#: `a` an upper-case letter, `c` either.
COUNTRY_ACCOUNT_REGISTRY: dict[str, tuple[int, str, str]] = {
    "AD": (24, "4n4n12c", "Andorra"),
    "AT": (20, "5n11n", "Austria"),
    "BA": (20, "3n3n8n2n", "Bosnia and Herzegovina"),
    "BE": (16, "3n7n2n", "Belgium"),
    "BG": (22, "4a4n2n8c", "Bulgaria"),
    "CH": (21, "5n12c", "Switzerland"),
    "CY": (28, "3n5n16c", "Cyprus"),
    "CZ": (24, "4n6n10n", "Czechia"),
    "DE": (22, "8n10n", "Germany"),
    "DK": (18, "4n9n1n", "Denmark"),
    "EE": (20, "2n2n11n1n", "Estonia"),
    "ES": (24, "4n4n1n1n10n", "Spain"),
    "FI": (18, "3n11n", "Finland"),
    "FO": (18, "4n9n1n", "Faroe Islands"),
    "FR": (27, "5n5n11c2n", "France"),
    "GB": (22, "4a6n8n", "United Kingdom"),
    "GI": (23, "4a15c", "Gibraltar"),
    "GL": (18, "4n9n1n", "Greenland"),
    "GR": (27, "3n4n16c", "Greece"),
    "HR": (21, "7n10n", "Croatia"),
    "HU": (28, "3n4n1n15n1n", "Hungary"),
    "IE": (22, "4a6n8n", "Ireland"),
    "IL": (23, "3n3n13n", "Israel"),
    "IS": (26, "4n2n6n10n", "Iceland"),
    "IT": (27, "1a5n5n12c", "Italy"),
    "LI": (21, "5n12c", "Liechtenstein"),
    "LT": (20, "5n11n", "Lithuania"),
    "LU": (20, "3n13c", "Luxembourg"),
    "LV": (21, "4a13c", "Latvia"),
    "MC": (27, "5n5n11c2n", "Monaco"),
    "ME": (22, "3n13n2n", "Montenegro"),
    "MK": (19, "3n10c2n", "North Macedonia"),
    "MT": (31, "4a5n18c", "Malta"),
    "NL": (18, "4a10n", "Netherlands"),
    "NO": (15, "4n6n1n", "Norway"),
    "PL": (28, "8n16n", "Poland"),
    "PT": (25, "4n4n11n2n", "Portugal"),
    "RO": (24, "4a16c", "Romania"),
    "RS": (22, "3n13n2n", "Serbia"),
    "SE": (24, "3n16n1n", "Sweden"),
    "SI": (19, "5n8n2n", "Slovenia"),
    "SK": (24, "4n6n10n", "Slovakia"),
    "SM": (27, "1a5n5n12c", "San Marino"),
    "TN": (24, "2n3n13n2n", "Tunisia"),
    "TR": (26, "5n1n16c", "Turkey"),
}


def account_layout_kinds(fmt: str) -> str:
    """``4a6n8n`` -> ``aaaannnnnnnnnnnnnn`` (one kind per character)."""
    out, num = [], ""
    for ch in fmt:
        if ch.isdigit():
            num += ch
        else:
            out.append(ch * int(num))
            num = ""
    return "".join(out)


def account_layout_matches(account: str, fmt: str) -> bool:
    kinds = account_layout_kinds(fmt)
    if len(account) != len(kinds):
        return False
    for ch, kind in zip(account, kinds, strict=True):
        if kind == "n" and ch not in DIGITS:
            return False
        if kind == "a" and ch not in UPPER:
            return False
        if kind == "c" and ch not in ALNUM_UPPER:
            return False
    return True


def _quads(t: str) -> list[str]:
    return [t[i: i + 4] for i in range(0, len(t), 4)]


def check_country_mod97(value: str, before: str, after: str, *, countries: Any = None,
                        check_layout: bool = True) -> Verdict:
    t = value.strip().replace(" ", "")
    if not t:
        return _none("no_characters")
    country = t[:2]
    layout = ("popular" if (" " not in value.strip() or value.strip().split(" ") == _quads(t))
              else "irregular")
    if not all(c in ALNUM_UPPER for c in t):
        reason = ("not_uppercase" if all(c in ALNUM_UPPER for c in t.upper()) and t.isascii()
                  else "illegal_character")
        return Verdict(False, reason, "none", layout, t, None)
    listed = sorted(countries) if countries else sorted(COUNTRY_ACCOUNT_REGISTRY)
    if country not in listed:
        return Verdict(False, "bad_prefix", "none", layout, t, None)
    length, fmt, _ = COUNTRY_ACCOUNT_REGISTRY[country]
    if len(t) != length:
        return Verdict(False, "wrong_length", "none", layout, t, country)
    if not t[2:4].isdigit() or iso7064_mod97_10(t[4:] + t[:4]) != 1:
        return Verdict(False, "bad_check_digit", "none", layout, t, country)
    if check_layout and not account_layout_matches(t[4:], fmt):
        return Verdict(False, "bad_bban_format", "none", layout, t, country)
    if before[-1:].isalnum() or after[:1].isalnum():
        return Verdict(True, "alphanumeric_boundary", "none", layout, t, country)
    return Verdict(True, "ok", "medium", layout, t, country)


# ── 10 digits, mod 11 ────────────────────────────────────────────────────────────


def check_mod11_ten(value: str, before: str, after: str, delimiters: str) -> Verdict:
    parts = core_parts(before[-1:] + value + after[:1])
    if parts is None:
        return _none("no_digits")
    core, pre, post, d = parts
    if not _only(core, delimiters):
        return _none("illegal_character", d)
    if len(d) != 10:
        return _none("wrong_length", d)
    if mod11_weights_10_to_2_digit(d[:9]) != d[9]:
        return _none("bad_check_digit", d)
    popular = digit_layout_popular(core, ((3, 3, 4),), delimiters)
    layout = "popular" if popular else "irregular"
    if pre.isalnum() or post.isalnum():
        return Verdict(True, "alphanumeric_boundary", "none", layout, d)
    if popular:
        return Verdict(True, "ok", "medium", layout, d)
    return Verdict(True, "irregular_layout", "low", layout, d)


# ── 12 digits, Verhoeff, leading digit 2-9 ──────────────────────────────────────


def check_verhoeff_twelve(value: str, before: str, after: str, delimiters: str) -> Verdict:
    parts = core_parts(before[-1:] + value + after[:1])
    if parts is None:
        return _none("no_digits")
    core, pre, post, d = parts
    if not _only(core, delimiters):
        return _none("illegal_character", d)
    if len(d) != 12:
        return _none("wrong_length", d)
    if not verhoeff_ok(d):
        return _none("bad_check_digit", d)
    if d[0] in "01":
        return _none("bad_prefix", d)
    popular = digit_layout_popular(core, ((4, 4, 4),), delimiters)
    layout = "popular" if popular else "irregular"
    if pre.isalnum() or post.isalnum():
        return Verdict(True, "alphanumeric_boundary", "none", layout, d)
    if popular:
        return Verdict(True, "ok", "medium", layout, d)
    return Verdict(True, "irregular_layout", "low", layout, d)


# ── the declarations ─────────────────────────────────────────────────────────────

_NO_PARAMS: dict[str, Any] = {"type": "object", "properties": {}, "additionalProperties": False}
_COUNTRY_PARAMS: dict[str, Any] = {
    "type": "object",
    "properties": {
        "countries": {"type": "array",
                      "items": {"type": "string", "enum": sorted(COUNTRY_ACCOUNT_REGISTRY)},
                      "uniqueItems": True,
                      "description": "The listed countries (default: every country listed)."},
        "check_bban": {"type": "boolean",
                       "description": "The country's account layout must match too (default "
                                      "true)."},
    },
    "additionalProperties": False,
}

_ACCOUNT_LENGTHS = tuple(sorted({length for length, _, _ in COUNTRY_ACCOUNT_REGISTRY.values()}))


class _Format:
    """One format: its delimiters, its check, its declared reasons, its scan."""

    def __init__(self, ident: str, *, delimiters: str, description: str, reasons: tuple[str, ...],
                 algorithms: tuple[str, ...], mask: MaskShape,
                 check: Callable[..., Verdict], params_schema: dict[str, Any] = _NO_PARAMS,
                 version: int = 1) -> None:
        self.delimiters = delimiters
        self._check = check
        self.decl = ValidatorDecl(
            id=ident, version=version, description=description, params_schema=params_schema,
            reasons=reasons, outputs=FORMAT_OUTPUTS, algorithms=algorithms, mask_shape=mask,
        )

    def check(self, value: str, before: str, after: str, params: Mapping[str, Any]) -> Verdict:
        return self._check(value, before, after, self.delimiters)

    def _spans(self, text: str) -> list[tuple[int, int]]:
        delims = re.escape(self.delimiters)
        return [(m.start(), m.end()) for m in re.finditer(rf"[0-9](?:[{delims}]*[0-9])*", text)]

    def scan(self, text: str, params: Mapping[str, Any]) -> list[Candidate]:
        return [Candidate(s, e, self.check(text[s:e], text[:s], text[e:], params))
                for s, e in self._spans(text)]


class _CountryFormat(_Format):
    def check(self, value: str, before: str, after: str, params: Mapping[str, Any]) -> Verdict:
        return check_country_mod97(
            value, before, after, countries=params.get("countries") or None,
            check_layout=bool(params["check_bban"]) if params.get("check_bban") is not None
            else True,
        )

    def _spans(self, text: str) -> list[tuple[int, int]]:
        """Version 1: every match of ``[A-Za-z]{2}[0-9]{2}(?: ?[A-Za-z0-9])+`` (so a spaced value
        followed by a word runs on through the word — published so, the producer's D38-1)."""
        return [(m.start(), m.end())
                for m in re.finditer(r"[A-Za-z]{2}[0-9]{2}(?: ?[A-Za-z0-9])+", text)]


_HEAD = re.compile(r"[A-Za-z]{2}[0-9]{2}")
_ALNUM = re.compile(r"[A-Za-z0-9]+")


def _continues_layout(group: str, at: int, kinds: str) -> bool:
    if at + len(group) > len(kinds):
        return False
    for ch, kind in zip(group, kinds[at:], strict=False):
        if kind == "n" and not ch.isdigit():
            return False
        if kind == "a" and not ch.isalpha():
            return False
    return True


def country_v2_spans(text: str) -> list[tuple[int, int]]:
    """Version 2's candidates (the producer's D38-1): the head and the run it starts, then each
    following group (one space, then letters and digits) only while it keeps the compact length
    within the country's registered length and can continue the country's layout; an unlisted
    country continues no group."""
    spans: list[tuple[int, int]] = []
    pos = 0
    while (head := _HEAD.search(text, pos)) is not None:
        start = head.start()
        run = _ALNUM.match(text, start)
        assert run is not None
        end = run.end()
        compact = len(text[start:end])
        entry = COUNTRY_ACCOUNT_REGISTRY.get(text[start: start + 2].upper())
        if entry is None:
            spans.append((start, end))
            pos = end
            continue
        limit = entry[0]
        kinds = account_layout_kinds(entry[1])
        while end + 1 < len(text) and text[end] == " ":
            group = _ALNUM.match(text, end + 1)
            if group is None:
                break
            chars = group.group()
            if compact + len(chars) > limit:
                break
            if not _continues_layout(chars, compact - 4, kinds):
                break
            compact += len(chars)
            end = group.end()
        spans.append((start, end))
        pos = end
    return spans


class _CountryFormatV2(_CountryFormat):
    def _spans(self, text: str) -> list[tuple[int, int]]:
        return country_v2_spans(text)


_COUNTRY_REASONS = ("ok", "no_characters", "not_uppercase", "illegal_character", "bad_prefix",
                    "wrong_length", "bad_check_digit", "bad_bban_format", "alphanumeric_boundary")
_COUNTRY_ALGORITHMS = ("iso7064_mod97_10", "country_length", "account_layout")

MASK_SHAPES: dict[str, MaskShape] = {
    ISSUER_CARD: MaskShape(tuple(range(12, 20)), ((4, 4, 4, 4), (4, 6, 5), (4, 6, 4)),
                           "-. /:", 8, 4),
    AREA_SERIAL: MaskShape((9,), ((3, 2, 4),), "-. ", 0, 4),
    BANK_TRANSIT: MaskShape((9,), ((3, 4, 2),), "-. ", 0, 4),
    COUNTRY_ACCOUNT: MaskShape(_ACCOUNT_LENGTHS, (), " ", 4, 4),
    HEALTH_MOD11: MaskShape((10,), ((3, 3, 4),), " -", 0, 4),
    RESIDENT_VERHOEFF: MaskShape((12,), ((4, 4, 4),), " -", 0, 4),
}

FORMAT_VALIDATORS: tuple[_Format, ...] = (
    _Format(ISSUER_CARD, delimiters="-. /:",
            description="12-19 digits: an issuer range and its length (ISO/IEC 7812), Luhn",
            reasons=("ok", "no_digits", "illegal_character", "wrong_length", "bad_check_digit",
                     "bad_prefix", "alphanumeric_boundary", "irregular_layout"),
            algorithms=("luhn_mod10", "issuer_range_and_length"),
            mask=MASK_SHAPES[ISSUER_CARD], check=check_issuer_luhn),
    _Format(AREA_SERIAL, delimiters="-. ",
            description="9 digits: area, group and serial range rules (no check digit)",
            reasons=("ok", "no_digits", "illegal_character", "wrong_length", "bad_range",
                     "irregular_layout", "letter_boundary"),
            algorithms=("area_group_serial_ranges",),
            mask=MASK_SHAPES[AREA_SERIAL], check=check_area_group_serial),
    _Format(BANK_TRANSIT, delimiters="-. ",
            description="9 digits: a 3-7-1 weighted check digit and a reserve-district prefix",
            reasons=("ok", "no_digits", "illegal_character", "wrong_length", "bad_check_digit",
                     "bad_prefix", "irregular_layout", "letter_boundary"),
            algorithms=("weighted_modulus_371", "reserve_district_prefix"),
            mask=MASK_SHAPES[BANK_TRANSIT], check=check_weighted_371_prefix),
    _CountryFormat(COUNTRY_ACCOUNT, delimiters=" ",
                   description="ISO 13616 account number: a listed country, its length and "
                               "account layout, ISO 7064 MOD 97-10",
                   reasons=_COUNTRY_REASONS, algorithms=_COUNTRY_ALGORITHMS,
                   mask=MASK_SHAPES[COUNTRY_ACCOUNT], check=check_country_mod97,
                   params_schema=_COUNTRY_PARAMS),
    _Format(HEALTH_MOD11, delimiters=" -",
            description="10 digits: a mod 11 check digit over weights 10 to 2",
            reasons=("ok", "no_digits", "illegal_character", "wrong_length", "bad_check_digit",
                     "alphanumeric_boundary", "irregular_layout"),
            algorithms=("mod11_weights_10_to_2",),
            mask=MASK_SHAPES[HEALTH_MOD11], check=check_mod11_ten),
    _Format(RESIDENT_VERHOEFF, delimiters=" -",
            description="12 digits: first digit 2-9, Verhoeff check digit",
            reasons=("ok", "no_digits", "illegal_character", "wrong_length", "bad_check_digit",
                     "bad_prefix", "alphanumeric_boundary", "irregular_layout"),
            algorithms=("verhoeff", "leading_digit_2_to_9"),
            mask=MASK_SHAPES[RESIDENT_VERHOEFF], check=check_verhoeff_twelve),
    _CountryFormatV2(COUNTRY_ACCOUNT, version=2, delimiters=" ",
                     description="ISO 13616 account number (version 2: the scan stops at a group "
                                 "past the country's length or layout)",
                     reasons=_COUNTRY_REASONS, algorithms=_COUNTRY_ALGORITHMS,
                     mask=MASK_SHAPES[COUNTRY_ACCOUNT], check=check_country_mod97,
                     params_schema=_COUNTRY_PARAMS),
)
