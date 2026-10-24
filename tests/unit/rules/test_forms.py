"""The forms are the vendored schema's, every one accounted for (039 FR-39.18; FTASKS 11.4).

The signal kinds are READ from the vendored producer schema. Every kind is implemented inline or
deferred (operator code, D41); a kind the producer adds and this evaluator lacks is refused at
import `RULE_FORM_UNAVAILABLE`, never ignored. Every form has a shared vector exercising it.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from millm.rules import RuleSetInvalid, forms, prepare
from millm.rules.document import COMPOSITION_FORMS, signal_kinds
from tests.support import rule_set_factory as rf

VECTORS = json.loads((Path(__file__).resolve().parents[3] / "docs" / "schemas"
                      / "rule-signal-cases-v1.json").read_text())


def test_every_schema_kind_is_implemented_or_deferred():
    kinds = set(signal_kinds())
    assert kinds >= {"pattern", "proximity", "validator", "lexical_weights", "code_rule"}
    assert kinds <= forms.IMPLEMENTED | forms.DEFERRED, kinds - forms.IMPLEMENTED - forms.DEFERRED
    assert forms.IMPLEMENTED <= kinds, "an implemented form the schema does not declare"
    assert forms.DEFERRED == {"code_rule"}, "D41: only operator code is deferred"


def test_every_inline_form_has_a_shared_vector():
    covered = set(VECTORS["rule_kinds"])
    for form in forms.IMPLEMENTED:
        assert form in covered, form
    for composition in COMPOSITION_FORMS:
        assert f"composition.{composition}" in covered, composition


def test_a_kind_the_evaluator_lacks_is_refused_not_ignored():
    doc = rf.document("b-0f0f0f0f")
    doc["signals"][0]["kind"] = "telepathy"
    with pytest.raises(RuleSetInvalid) as caught:
        prepare(doc)
    assert caught.value.code == "form_unavailable"
    assert caught.value.details["signal"] == "tok"


def test_a_new_schema_kind_without_an_implementation_is_red(monkeypatch):
    """What the coverage test does when the producer adds a kind: simulated by widening the
    schema's kind list — the assertion above must then fail."""
    import sys

    # The name THIS module bound at import (patching the source module would change nothing).
    monkeypatch.setattr(sys.modules[__name__], "signal_kinds", lambda: ("pattern", "new_kind"))
    with pytest.raises(AssertionError):
        test_every_schema_kind_is_implemented_or_deferred()
