"""The pre-039 goldens (039 FTASKS 0.7), captured from `main` at `779efa11` before any 039 code
(`tests/support/capture_pre039_goldens.py`), compared with the same scenario on this tree.

* Nothing armed: every surface byte-identical (FR-39.11).
* Two probes of two behaviours armed: every pre-039 key present with its value, except the changes
  039 documents (the `ALLOWED` list and the header normalisation of `test_probe_result_surfaces`:
  a behaviour's `set` block, the header's `behaviour` / `probe` parameters and RFC 8941 decimals).
* The score response an additive superset (FR-39.13): no pre-039 key changes at all.
"""

from __future__ import annotations

import json

import pytest

from tests.support import capture_pre038_goldens as cap
from tests.support import capture_pre039_goldens as pre039
from tests.unit.api.test_probe_result_surfaces import _allowed, _compare, _pre039_header

pytestmark = pytest.mark.asyncio


#: 038's per-run values (minted per request / per finish), absent from the pre-038 harness's
#: normalisation because the pre-038 surfaces did not carry them.
VOLATILE_038 = {"result_id", "event_time", "completed_at", "evaluation_ms", "source_event_id"}


def _strip(value):
    if isinstance(value, dict):
        return {k: _strip(v) for k, v in value.items() if k not in VOLATILE_038}
    if isinstance(value, list):
        return [_strip(v) for v in value]
    return value


async def _now(name):
    return await cap.capture(next(s for s in cap.SCENARIOS if s.name == name))


async def test_nothing_armed_is_byte_identical():
    golden = json.loads(pre039.golden_path("unarmed").read_text())
    assert await _now("unarmed") == golden


async def test_the_armed_surfaces_keep_every_pre039_key():
    golden = _strip(json.loads(pre039.golden_path("armed").read_text()))
    now = _strip(await _now("armed"))
    diffs: list = []
    for surface, value in golden.items():
        current = _pre039_header(now.get(surface)) if surface == "header" else now.get(surface)
        _compare(value, current, surface, diffs)
    assert [d for d in diffs if not _allowed("armed", d[0])] == []


async def test_the_score_response_is_an_additive_superset():
    golden = _strip(json.loads(pre039.golden_path("armed").read_text())["score"])
    now = _strip((await _now("armed"))["score"])
    diffs: list = []
    _compare(golden, now, "score", diffs)
    assert diffs == [], "a pre-039 key of the score response changed"
