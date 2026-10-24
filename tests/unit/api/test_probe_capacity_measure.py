"""The capacity measurement path (operator 2026-10-21), on the CPU with a real tiny Llama.

The run on the GPU needs an operator-approved window; this proves the path end to end without one:
the real route, the real `InferenceService.run_model_work` slot, the real model forward with a KV
cache, the real `probe_cost.measure`, one appended measurement row. Timers are injected.

Controls (recorded): dropping the "nothing armed" refusal → `test_it_refuses_while_probes_are_armed`
red; dropping the row append → `test_measuring_appends_a_row_the_capacity_then_reads` red.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest
import torch
from sqlalchemy import select

from millm.db.models.probe_capacity import ProbeCapacityMeasurement
from millm.services import probe_capacity_measure as measure_mod
from millm.services.probe_runtime import ProbeRuntimeState
from tests.unit.api.test_model_probe_limits import _model
from tests.unit.f25_fixtures import clear_loaded, make_client, make_service, word_model, word_tokenizer
from tests.unit.f28_fixtures import db, no_cuda  # noqa: F401


def test_measure_resident_times_decode_steps_and_a_prefill(monkeypatch):
    calls = []

    def clock(fn, devices):
        fn()
        calls.append([d.type for d in devices])
        return 2.0

    monkeypatch.setattr(measure_mod, "_clock", clock)
    reading = measure_mod.measure_resident(word_model(), prefill_tokens=64, context_tokens=8,
                                           steps=5, repeats=3, k=2)
    assert len(calls) == (3 + 1) + (5 + 1)  # one warm-up each
    assert reading.decode_ms_per_token == 2.0
    assert reading.prefill_ms_per_1k == pytest.approx(2.0 * 1000 / 64)
    assert reading.device == "cpu"
    assert reading.dense["layer"] == 1 and "decode_own_ms" in reading.dense
    assert "nothing armed" in reading.method


@pytest.fixture
def served(db, monkeypatch):  # noqa: F811
    from millm.api.dependencies import get_db

    monkeypatch.setattr(measure_mod, "_clock", lambda fn, devices: (fn(), 3.0)[1])
    ProbeRuntimeState.reset_for_tests()
    svc = make_service(word_model(), word_tokenizer())  # resident model_id = 1
    client, model_service = make_client(svc, None)

    async def _db():
        async with db() as session:
            yield session

    client.app.dependency_overrides[get_db] = _db
    yield client, model_service
    ProbeRuntimeState.reset_for_tests()
    clear_loaded()


async def test_measuring_appends_a_row_the_capacity_then_reads(served, db):  # noqa: F811
    client, model_service = served
    row = await _model(db, repo="tiny/tiny-llama", seed=False)
    assert row.id == 1
    model_service.get_model = AsyncMock(return_value=row)
    with no_cuda():
        r = client.post(f"/api/models/{row.id}/probe-capacity/measure")
    assert r.status_code == 200, r.text
    measured = r.json()["data"]["measured"]
    assert (measured["decode_ms_per_token"], measured["device"]) == (3.0, "cpu")
    async with db() as session:
        rows = (await session.execute(select(ProbeCapacityMeasurement))).scalars().all()
    assert [(m.repo_id, m.quantization) for m in rows] == [("tiny/tiny-llama", "FP16")]
    # 039 (TD39-3): the per-rule lines are stored, and `dense` stays the `mean` line.
    assert sorted(rows[0].lines) == ["attention", "max", "mean"]
    assert rows[0].dense == rows[0].lines["mean"]
    assert measured["lines"] == rows[0].lines


async def test_it_refuses_a_model_that_is_not_resident(served, db):  # noqa: F811
    client, model_service = served
    await _model(db, repo="tiny/first", seed=False)
    other = await _model(db, repo="tiny/other", seed=False)
    model_service.get_model = AsyncMock(return_value=other)
    r = client.post(f"/api/models/{other.id}/probe-capacity/measure")
    assert r.status_code == 409 and r.json()["error"]["details"]["reason"] == "not_resident"


async def test_it_refuses_while_probes_are_armed(served, db):  # noqa: F811
    from millm.ml.probe_head import ProbeHead
    from millm.services.probe_runtime import ArmedProbe

    client, model_service = served
    row = await _model(db, repo="tiny/tiny-llama", seed=False)
    model_service.get_model = AsyncMock(return_value=row)
    ProbeRuntimeState()._armed["pr_x"] = ArmedProbe(
        probe_id="pr_x", name="x", head=ProbeHead(weight=torch.ones(16), bias=0.0, layer=0),
        rule="mean", scope="all", layer=0, rung=2, rung_language="x")
    r = client.post(f"/api/models/{row.id}/probe-capacity/measure")
    assert r.status_code == 409 and r.json()["error"]["details"]["reason"] == "probes_armed"


def test_the_synthetic_probe_is_real_shaped(monkeypatch):
    """Live finding 2026-10-21: a synthetic head without `norm_mean`/`norm_std` skips the
    standardisation every real probe pays, so the dense line came out ~3× too cheap."""
    from millm.services import probe_cost

    seen = []
    real = probe_cost.measure

    def spy(armed, **kw):
        seen.append(armed)
        return real(armed, **kw)

    monkeypatch.setattr(probe_cost, "measure", spy)
    monkeypatch.setattr(measure_mod, "_clock", lambda fn, devices: (fn(), 1.0)[1])
    measure_mod.measure_resident(word_model(), prefill_tokens=16, context_tokens=4, steps=2,
                                 repeats=1, k=1)
    # 039 (TD39-3): one real-shaped synthetic probe per rule line — mean, max, attention.
    assert [p.rule for p in seen] == ["mean", "max", "attention"]
    for probe in seen:
        head = probe.head
        assert head.mean is not None and head.std is not None
        assert head.mean.shape == head.std.shape == head.weight.shape
    assert seen[2].head.attention_query is not None, "the attention line needs a query"
    assert seen[0].head.attention_query is None
