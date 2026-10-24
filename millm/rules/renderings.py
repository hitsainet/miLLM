"""Rendering semantics the contract pins (FTDD 016 section 4.10; executed by feature 017).

miLLM 039 S2 (TD39-24): a port of miDataworks' `backend/src/rules/renderings.py` (origin/main 2082e01),
kept line-for-line so the vendored vectors (`docs/schemas/rule-signal-cases-v1.json`) pin both apps
alike; miLLM imports no sibling code.

016 declares renderings and checks their shape; 017's build applies them. The builtins and the
mapping are defined here once so the ``rule_kinds.renderings`` cases (``normalise(render(v)) == v``)
can run in every app:

- ``spaced``: the characters joined by one space; ``split_lines``: joined by ``\\n``;
  ``spelled_out_digits``: each ASCII digit as its English word (``zero`` ... ``nine``), every other
  character kept, all joined by one space.
- ``mapping``: left to right, the longest key that matches at a position is replaced by its value;
  a character no key matches is kept. ``mapping_inverse`` applies the inverted mapping the same way
  (values must be distinct).
- ``digits_only``: keeps ASCII digits; ``strip_delimiters``: removes spaces, ``\\n``, ``-`` and ``.``;
  ``words_to_digits`` (D38-3) inverts ``spelled_out_digits``: the text is tokens separated by
  exactly one space; a digit word becomes its digit, any other token is one kept character.

A ``validator`` render or normaliser is 017's (a validator-library synthesiser); this module refuses
it rather than guessing.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

DIGIT_WORDS = ("zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine")
STRIPPED = frozenset(" \n-.")


class RenderingNotRunnable(ValueError):
    pass


def _apply_mapping(value: str, mapping: Mapping[str, str]) -> str:
    keys = sorted(mapping, key=len, reverse=True)
    out, i = [], 0
    while i < len(value):
        for key in keys:
            if key and value.startswith(key, i):
                out.append(mapping[key])
                i += len(key)
                break
        else:
            out.append(value[i])
            i += 1
    return "".join(out)


def render(rendering: Mapping[str, Any], value: str) -> str:
    spec = rendering["render"]
    if "builtin" in spec:
        kind = spec["builtin"]
        if kind == "spaced":
            return " ".join(value)
        if kind == "split_lines":
            return "\n".join(value)
        return " ".join(DIGIT_WORDS[ord(c) - 48] if c in "0123456789" else c for c in value)
    if "mapping" in spec:
        return _apply_mapping(value, spec["mapping"])
    raise RenderingNotRunnable("a validator rendering is executed by feature 017")


def words_to_digits(rendered: str) -> str:
    """The inverse of ``spelled_out_digits``: token, one space, token, ... where a token is a digit
    word or ONE kept character (a kept space is a token too: ``"1 2"`` renders ``"one   two"``)."""
    out: list[str] = []
    i = 0
    while i < len(rendered):
        for digit, word in enumerate(DIGIT_WORDS):
            end = i + len(word)
            if rendered.startswith(word, i) and (end == len(rendered) or rendered[end] == " "):
                out.append(str(digit))
                i = end
                break
        else:
            out.append(rendered[i])
            i += 1
        if i < len(rendered):
            if rendered[i] != " ":
                raise RenderingNotRunnable("words_to_digits: tokens are separated by one space")
            i += 1
    return "".join(out)


def normalise(rendering: Mapping[str, Any], rendered: str) -> str:
    spec = rendering["normaliser"]
    if "builtin" in spec:
        if spec["builtin"] == "digits_only":
            return "".join(c for c in rendered if c in "0123456789")
        if spec["builtin"] == "words_to_digits":
            return words_to_digits(rendered)
        return "".join(c for c in rendered if c not in STRIPPED)
    if "mapping_inverse" in spec:
        mapping = rendering["render"].get("mapping")
        if not mapping:
            raise RenderingNotRunnable("mapping_inverse needs a mapping render")
        inverse = {v: k for k, v in mapping.items()}
        if len(inverse) != len(mapping):
            raise RenderingNotRunnable("mapping values must be distinct to invert")
        return _apply_mapping(rendered, inverse)
    raise RenderingNotRunnable("a validator normaliser is executed by feature 017")
