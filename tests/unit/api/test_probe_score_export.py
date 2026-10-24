"""Stateless export (FR-38.17, AT07; FTASKS 8.10) through the real `POST /api/probes/score`.

The gate has two halves — the OPERATOR's `TELEMETRY_STATELESS_EXPORT_PERMITTED` and the REQUEST's
`export_results` — and a result leaves only when both are true (and logs are on). The response's
`export` summary appears only when the request SENT the field (`model_fields_set`), so an older
client sees the old body.

Control M17: remove either half of the condition in `ProbeScoringService.score` (rows 2 and 3).
"""

from __future__ import annotations

import pytest

from tests.support import capture_pre038_goldens as cap
from tests.unit.telemetry.conftest import (  # noqa: F401 - fixtures
    _telemetry_off,
    flush,
    stub,
    telemetry_on,
)

ROWS = [
    # (permitted, requested, are the results exported)
    (False, False, False),
    (True, False, False),
    (False, True, False),
    (True, True, True),
]


def _score(h, rows, **extra):
    response = h.client.post("/api/probes/score", json={
        "probe_ids": [r.id for r in rows],
        "inputs": [{"token_ids": [0, 2, 6, 7, 8], "prompt_tokens": 3}],
        **extra,
    })
    assert response.status_code == 200, response.text
    return response.json()["data"]


@pytest.mark.parametrize("permitted, requested, exported", ROWS)
async def test_at07_both_halves_or_nothing(stub, telemetry_on, permitted, requested, exported):
    telemetry_on(TELEMETRY_STATELESS_EXPORT_PERMITTED=permitted)
    async with cap.Harness() as h:
        rows = await h.arm([cap.ALPHA, cap.BETA])
        data = _score(h, rows, export_results=requested)
        flush()
    ids = {v["result_id"] for entry in data["results"] for v in entry["verdicts"]}
    assert len(ids) == 6, "two probes, three default windows"
    expected = len(ids) if exported else 0
    assert data["export"] == {"requested": requested, "permitted": permitted,
                              "records_queued": expected}
    results = [r for r in stub.logs if r["event_name"] == "mi.probe.result"]
    assert len(results) == expected
    for rec in results:
        attrs = rec["attributes"]
        # The record joins the score request's own span (`millm.probe_score`).
        assert rec["trace_id"] and attrs["mi.probe.trace_id"] == rec["trace_id"]
        assert attrs["mi.probe.origin"] == "stateless"
        assert attrs["mi.probe.persistence"] == "not_persisted"
        assert attrs["mi.probe.input_index"] == 0
        assert attrs["mi.probe.result_id"] in ids


async def test_the_summary_is_absent_unless_the_field_was_sent(stub, telemetry_on):
    telemetry_on(TELEMETRY_STATELESS_EXPORT_PERMITTED=True)
    async with cap.Harness() as h:
        rows = await h.arm([cap.ALPHA])
        data = _score(h, rows)
        flush()
    assert "export" not in data
    assert [r for r in stub.logs if r["event_name"] == "mi.probe.result"] == []


async def test_with_telemetry_off_nothing_is_queued_and_the_summary_says_so():
    async with cap.Harness() as h:
        rows = await h.arm([cap.ALPHA])
        data = _score(h, rows, export_results=True)
    assert data["export"] == {"requested": True, "permitted": False, "records_queued": 0}


async def test_the_openapi_names_the_field():
    from millm.main import create_app

    schema = create_app().openapi()["components"]["schemas"]["ProbeScoreRequest"]
    assert schema["properties"]["export_results"]["default"] is False
