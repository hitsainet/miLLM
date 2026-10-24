"""Migration 031 (per-model probe limits and measured capacity) goes up with its two seeded
measurements (the 2026-10-21 hardware record) and comes down, on PostgreSQL."""

from __future__ import annotations

import asyncio

from tests.schema.test_probe_sessions_migration import _column, fresh_db  # noqa: F401
from tests.schema.test_weights_origin_migration import _alembic, _query


def test_031_creates_seeds_and_drops(fresh_db):  # noqa: F811
    _alembic(fresh_db, "upgrade", "031")
    rows = asyncio.run(_query(fresh_db, (
        "SELECT repo_id, quantization, device, decode_ms_per_token, prefill_ms_per_1k, "
        "dense->>'decode_own_ms', sparse IS NULL FROM probe_capacity_measurements ORDER BY repo_id"
    )))
    assert rows == [
        ("LiquidAI/LFM2.5-1.2B-Instruct", "FP16", "cuda:GPU-247aa582-0d1b-e161-8156-983ed1fefc57",
         6.69, None, "0.0253", False),
        ("meta-llama/Llama-3.1-8B-Instruct", "FP16",
         "cuda:GPU-247aa582-0d1b-e161-8156-983ed1fefc57", 22.68, None, "0.0236", True),
    ]
    assert asyncio.run(_query(fresh_db, _column("model_probe_limits", "max_armed"))) != []
    _alembic(fresh_db, "downgrade", "030")
    assert asyncio.run(_query(fresh_db, _column("model_probe_limits", "max_armed"))) == []
    assert asyncio.run(_query(fresh_db, _column("probe_capacity_measurements", "id"))) == []
    _alembic(fresh_db, "upgrade", "head")
