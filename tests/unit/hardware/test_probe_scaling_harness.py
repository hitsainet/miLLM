"""The Feature 37 hardware harness's pure parts, on the CPU (the run itself needs a GPU window).

The load-bearing claim is the synthetic probe's: a zero head with bias 1.0 passes miLLM's REAL
parity engine whatever the residual is, so the harness can arm N probes on any model without
miStudio — while each probe still does a real probe's arithmetic.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "tests" / "hardware"))
import probe_scaling_acceptance as h  # noqa: E402

TEMPLATE = json.loads((REPO / "tests" / "fixtures" / "lfm2_probe_definition.json").read_text())


def test_cards_are_read_by_uuid_never_by_index():
    out = ("GPU 0: NVIDIA GeForce RTX 3090 (UUID: GPU-247aa582-0d1b-e161-8156-983ed1fefc57)\n"
           "GPU 1: NVIDIA GeForce RTX 3080 Ti (UUID: GPU-f47ba814-49a2-603f-3595-275284140251)\n")
    assert h.parse_uuids(out) == ["GPU-247aa582-0d1b-e161-8156-983ed1fefc57",
                                  "GPU-f47ba814-49a2-603f-3595-275284140251"]


def test_the_grid_covers_every_card_n_kind_and_spread():
    grid = h.case_grid("m", ["GPU-a", "all"], [1, 8], ["dense", "mixed"], ["one", "many"])
    assert len(grid) == 2 * 2 * 2 * 2
    assert len({c.slug for c in grid}) == len(grid)
    assert any(c.card == "all" and "split" in c.slug for c in grid)


def test_layers_and_kinds():
    assert h.layers_for(3, "one", 16, 11) == [11, 11, 11]
    assert set(h.layers_for(16, "many", 16, 11)) == set(range(4, 12))
    assert h.kinds_for(4, "mixed") == ["dense", "sparse", "dense", "sparse"]


async def test_a_synthetic_probe_passes_the_real_parity_engine(test_session):
    from millm.db.repositories.probe_repository import ProbeRepository
    from millm.services.probe_arming import armed_probe_from_row
    from millm.services.probe_parity import ProbeParityEngine
    from millm.services.probe_service import ProbeService

    doc = h.synthetic(TEMPLATE, "hwscale-dense-0", layer=5)
    row = await ProbeService(ProbeRepository(test_session)).import_definition(
        doc, raw_bytes=1, on_conflict="fail", origin="file")
    armed = armed_probe_from_row(row, encoder=None, windows=[])
    d = int(doc["model"]["d_model"])
    torch.manual_seed(0)

    def forward(input_ids, context):  # any residual at all
        context.observe(5, torch.randn(1, input_ids.shape[1], d) * 7)

    report = ProbeParityEngine(forward).run(armed, row.definition, tolerance=1e-6)
    assert report.passed, report.as_details()
    assert row.layer == 5 and all(w == 0.0 for w in row.definition["head"]["weights"])


def test_observed_ratio_findings_and_regression():
    assert h.observed_per_pass([12.0, 10.0, 11.0], [5.0, 6.0, 4.0], 61) == pytest.approx(6 / 61)
    assert h.ratio(0.3, 0.2) == pytest.approx(1.5) and h.ratio(0.3, 0.0) is None
    recs = [{"case": "a", "ratio_decode": 1.3, "observed_decode_ms": 0.1},
            {"case": "b", "ratio_decode": 1.1, "observed_decode_ms": 0.1}]
    assert [r["case"] for r in h.findings(recs)] == ["a"]
    worse = [{"case": "a", "observed_decode_ms": 0.2}, {"case": "b", "observed_decode_ms": 0.11}]
    assert h.regression(recs, worse) == ["a"]


async def test_an_adapted_synthetic_probe_imports_for_another_model(test_session):
    """The second model of the sweep (d_model 4096, 32 layers): identity block re-pointed."""
    from millm.db.repositories.probe_repository import ProbeRepository
    from millm.services.probe_service import ProbeService

    doc = h.synthetic(h.adapt(TEMPLATE, hf_id="meta-llama/Llama-3.1-8B-Instruct", d_model=4096,
                              n_layers=32), "hwscale-dense-l", layer=20)
    row = await ProbeService(ProbeRepository(test_session)).import_definition(
        doc, raw_bytes=1, on_conflict="fail", origin="file")
    assert (row.hf_id, row.d_model, row.n_layers, row.layer) == (
        "meta-llama/Llama-3.1-8B-Instruct", 4096, 32, 20)
    assert len(row.definition["head"]["weights"]) == 4096
    assert "chat_template_sha256" not in row.definition["model"]
