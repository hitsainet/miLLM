"""`X-miLLM-Circuit-Rung` only on a response the active circuit actually steered (014 4.2 E2E).

Found on hardware 2026-10-16 (Qwen2.5-7B-Instruct, a rung-0 circuit serving `full` on L20 + L25,
deployed image `sha256:da26140f…`): with the circuit active, a request dialled to `"off"`, an
explicitly-unsteered inline request (`steering: {"features": []}`) and an inline set on one SAE
all came back with `X-miLLM-Circuit-Rung: 0; language="associated"` — beside
`X-miLLM-Steering: none` or `inline;…`. The rung header is an evidence claim; on a rung-2 circuit
it would have read "causally validated (edge)" on a response the circuit never touched.

The rule lives in `millm.api.provenance` (the one place every route and batch line builds its
headers): no rung for inline steering or a λ that resolves to 0 (decided before generation, so a
stream's committed headers obey it too), and after generation no rung unless the published
steering report names a circuit.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from millm.api import provenance
from millm.core.steering_state import SteeringItem
from millm.services.inference_service import set_steering_report
from millm.services.steering_report import NONE_REPORT, SteeringReport
from tests.unit.f28_fixtures import chat

RUNG = (2, "causally validated (edge)")
ECHO = '2; language="causally validated (edge)"'


class _Inference:
    backend_name = "serial"

    def __init__(self, resolved=None):
        self.active_circuit_rung = AsyncMock(return_value=RUNG)
        self.resolve_request_intensity = AsyncMock(return_value=resolved)

    def seed_scope_for(self, request):  # pragma: no cover - only with a seed
        return "request"


CIRCUIT_REPORT = SteeringReport.from_items(
    [SteeringItem(kind="circuit", circuit_id="circ_1", intensity=1.0)]
)


@pytest.fixture(autouse=True)
def _report():
    set_steering_report(None)
    yield
    set_steering_report(None)


def _post(request, inference, pre):
    return provenance.post_generation(
        request, inference, endpoint="chat", pre=pre, ignored_header=None
    )


class TestBeforeGeneration:
    async def test_an_undialled_request_on_a_steering_circuit_gets_the_rung(self):
        pre = await provenance.pre_generation(chat(), _Inference(), chat=True)
        assert pre.echo_circuit_rung == ECHO

    @pytest.mark.parametrize("dial", ["off", 0, 0.0])
    async def test_a_dial_resolved_to_zero_gets_no_rung(self, dial):
        pre = await provenance.pre_generation(
            chat(steering_intensity=dial), _Inference(resolved=0.0), chat=True
        )
        assert pre.echo_intensity == "0"
        assert pre.echo_circuit_rung is None

    async def test_a_nonzero_dial_keeps_the_rung(self):
        pre = await provenance.pre_generation(
            chat(steering_intensity="min"), _Inference(resolved=0.5), chat=True
        )
        assert pre.echo_circuit_rung == ECHO

    @pytest.mark.parametrize("steering", [
        {"features": []},
        {"sae_id": "sae-20", "features": [{"index": 3, "strength": 5.0}]},
    ])
    async def test_inline_steering_gets_no_rung(self, steering):
        inference = _Inference()
        pre = await provenance.pre_generation(chat(steering=steering), inference, chat=True)
        assert pre.echo_circuit_rung is None
        inference.active_circuit_rung.assert_not_called()


class TestAfterGeneration:
    def test_the_rung_stands_when_the_report_names_the_circuit(self):
        set_steering_report(CIRCUIT_REPORT)
        headers = _post(chat(), _Inference(), provenance.PreGeneration(echo_circuit_rung=ECHO))
        assert headers["X-miLLM-Circuit-Rung"] == ECHO

    @pytest.mark.parametrize("report", [
        NONE_REPORT,
        SteeringReport.unknown("read_failed"),
        SteeringReport.from_items([SteeringItem(kind="manual", sae="s", layer=20, features=1,
                                                hash="sha256:" + "0" * 64)]),
    ])
    def test_no_rung_when_what_ran_was_not_the_circuit(self, report):
        set_steering_report(report)
        headers = _post(chat(), _Inference(), provenance.PreGeneration(echo_circuit_rung=ECHO))
        assert "X-miLLM-Circuit-Rung" not in headers

    def test_no_report_keeps_the_pre_generation_decision(self):
        headers = _post(chat(), _Inference(), provenance.PreGeneration(echo_circuit_rung=ECHO))
        assert headers["X-miLLM-Circuit-Rung"] == ECHO

    def test_a_scoring_request_never_carries_a_rung(self):
        set_steering_report(CIRCUIT_REPORT)
        request = chat(logprobs=True, top_logprobs=2, max_tokens=1)
        assert request.wants_scores()
        headers = _post(request, _Inference(), provenance.PreGeneration(echo_circuit_rung=ECHO))
        assert "X-miLLM-Circuit-Rung" not in headers
