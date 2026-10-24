"""A probe row says what it was fitted on, not only where it reads.

⚠ **THE TILE IDENTIFIED THE READ POINT AND NOTHING ELSE.** A probe row read
`meta-llama/Llama-3.1-8B-Instruct · L11 · mean · dense residual · scope all`. Two probes on the
live installation differed only in the run that produced them, so the list could not tell them
apart, and nothing anywhere in miLLM said what either one detects. The operator asked for it
directly on 2026-10-01.

⚠ **AGAINST THE REAL PUBLISHED DOCUMENTS.** These fixtures are the files miStudio's 033
acceptance produced, copied byte for byte off the node. A hand-written definition would be
written by whoever also wrote the reader, which is this estate's most common reason for a green
suite over a broken feature.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from millm.api.routes.management.probes import _probe_summary
from millm.core.probe_labels import concept_of, label_mapping_of

FIXTURES = Path(__file__).resolve().parents[2] / "fixtures"
REAL_DEFINITIONS = sorted(FIXTURES.glob("lfm2_probe_definition*.json"))


def _definition(path: Path) -> dict:
    return json.loads(path.read_text())


def _row(definition: dict):
    row = MagicMock()
    row.id = "pr_1"
    row.name = definition["name"]
    row.hf_id = definition["model"]["hf_id"]
    row.layer = definition["read"]["layer"]
    row.rule = definition["aggregation"]["rule"]
    row.scope = definition["scope"]
    row.basis = definition["basis"]
    row.streamable = definition["aggregation"]["streamable"]
    row.threshold = (definition.get("decision") or {}).get("threshold")
    row.target_fpr = (definition.get("decision") or {}).get("target_fpr")
    row.rung = definition["evidence"]["rung"]
    row.armed = False
    row.paused_reason = None
    row.parity = None
    row.created_at = None
    row.definition = definition
    return row


def test_there_are_real_definitions_to_read():
    """Guard the guard: a glob that matched nothing would make every case below vacuous."""
    assert len(REAL_DEFINITIONS) >= 3


@pytest.mark.parametrize("path", REAL_DEFINITIONS, ids=lambda p: p.name)
def test_the_summary_carries_the_mapping_verbatim(path: Path):
    """Both sides of the boundary, as the corpus spells them.

    ⚠ The mapping goes out RAW. Formatting it here would put a presentation decision on the
    wire where nothing could check it against the corpus, and would make the two repos'
    captions impossible to compare.
    """
    definition = _definition(path)
    summary = _probe_summary(_row(definition))

    assert summary["label_mapping"] == {"low-stakes": "negative", "high-stakes": "positive"}
    assert summary["concept"] == "positive = high-stakes"


def test_the_mapping_is_read_from_the_definition_not_the_projection():
    """`Probe.provenance` is a projection; the document is the authority.

    The model's own docstring says so, and `probe_service` writes the projection with an extra
    `origin` key — so the two can differ, and a reader that trusted the projection would be
    reading a copy.
    """
    definition = _definition(REAL_DEFINITIONS[0])
    row = _row(definition)
    # A projection that has drifted. The summary must not notice it.
    row.provenance = {"label_mapping": {"drifted": "positive"}, "origin": "file"}

    assert _probe_summary(row)["label_mapping"] == {
        "low-stakes": "negative",
        "high-stakes": "positive",
    }


@pytest.mark.parametrize(
    "definition",
    [
        None,
        "not a document",
        {},
        {"provenance": None},
        {"provenance": "not a dict"},
        {"provenance": {}},
        {"provenance": {"label_mapping": None}},
        {"provenance": {"label_mapping": ["high-stakes"]}},
    ],
)
def test_a_malformed_document_leaves_the_caption_blank_rather_than_raising(definition):
    """A definition arrives from a file or the Hub. A list route must not 500 over one.

    Blank is the honest outcome: the alternative that matters is not a prettier message, it is
    the whole probe list failing to render because one import was odd.
    """
    assert label_mapping_of(definition) == {}


@pytest.mark.parametrize(
    "definition", [None, "x", {}, {"concept": None}, {"concept": ""}, {"concept": "   "}, {"concept": 3}]
)
def test_an_absent_concept_is_none_not_an_empty_string(definition):
    """`""` renders as a blank line in the tile; `None` renders as nothing."""
    assert concept_of(definition) is None


def test_values_are_stringified_without_being_rewritten():
    """No translation, no title-casing. The tile must match the corpus token for token."""
    mapping = label_mapping_of({"provenance": {"label_mapping": {"High-Stakes": "POSITIVE"}}})
    assert mapping == {"High-Stakes": "POSITIVE"}
