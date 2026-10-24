"""R31: every gated `/api/weights` call writes one audit line, including a 500 and a cancellation
(FR-31.15; 033's routes already did, 031's audited only `MiLLMError` refusals).

Negative control: narrowing `_audited` back to `except MiLLMError` turns both tests red.
"""

from __future__ import annotations

import asyncio

import pytest
from structlog.testing import capture_logs

from millm.api.routes.management.weights import _audited
from tests.unit.api.test_weights_routes import _audits, _auth, env  # noqa: F401 - fixture
from tests.unit.services.test_weight_attachment import HANDLE, TRAINER


async def test_an_unexpected_export_error_is_a_500_with_one_audit_line(env):
    env.fakes.export_raises = RuntimeError(f"expandable segment; handle {HANDLE}")
    with capture_logs() as logs:
        async with env.client() as client:
            response = await client.post("/api/weights/attach", headers=_auth())
    assert response.status_code == 500
    audits = _audits(logs)
    assert len(audits) == 1
    line = audits[0]
    assert (line["operation"], line["outcome"], line["code"], line["reason"]) == (
        "weights_attach", "refused", "INTERNAL_ERROR", "RuntimeError"
    )
    assert line["label"] == "miforge" and line["role"] == "trainer"
    assert HANDLE not in repr(audits)
    assert env.service.is_attached() is False


async def test_a_cancelled_call_is_audited_and_still_cancelled():
    with capture_logs() as logs:
        with pytest.raises(asyncio.CancelledError):
            with _audited("weights_quiesce", TRAINER, "abcd1234"):
                raise asyncio.CancelledError()
    audits = _audits(logs)
    assert len(audits) == 1
    assert (audits[0]["outcome"], audits[0]["code"], audits[0]["attachment_ref"]) == (
        "refused", "CANCELLED", "abcd1234"
    )
