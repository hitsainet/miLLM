"""Speculative decoding and the trace reference on real requests (FTASKS 3.2, 3.3; FR-38.6, FR-38.14).

* A request served with a draft model attached reports `execution_mode: speculative` and
  `skipped / speculative_decoding` (D38.34, U7), on a tiny real Llama with a tiny real draft.
  The spike (OI-5): whether the probe hook was reached while the draft ran, recorded in the
  controls record.
* The trace reference captured when the probe context opens reaches the result (S1: the incoming
  reference only; the middleware that sets it is S2's).
"""

from __future__ import annotations

from unittest.mock import patch

import pytest
import torch

from millm.api.schemas.openai import ChatCompletionRequest, ChatMessage
from millm.ml.probe_head import ProbeHead
from millm.services.probe_result import get_probe_results
from millm.services.probe_runtime import ArmedProbe, ProbeRuntimeState
from millm.telemetry.context import _REQUEST_TRACE, TraceRef
from tests.unit.f25_fixtures import clear_loaded, make_service, word_model, word_tokenizer

pytestmark = pytest.mark.asyncio

#: The spike's finding, written by the test that measures it (read by the controls record).
SPIKE: dict = {}


@pytest.fixture
def armed(monkeypatch):
    import millm.db.base as db_base

    ProbeRuntimeState.reset_for_tests()

    def no_db():
        raise RuntimeError("no database here")

    monkeypatch.setattr(db_base, "async_session_factory", no_db)
    model, tokenizer = word_model(), word_tokenizer()
    svc = make_service(model, tokenizer)
    ProbeRuntimeState().arm(ArmedProbe(
        probe_id="pr_0123456789ab", name="probe", head=ProbeHead(weight=torch.ones(16), layer=0),
        rule="mean", scope="all", layer=0, rung=2, rung_language="detects on unseen tasks",
        threshold=0.0, windows=("all",),
    ), model)
    yield svc
    ProbeRuntimeState.reset_for_tests()
    clear_loaded()


def _chat() -> ChatCompletionRequest:
    return ChatCompletionRequest(model="tiny", messages=[ChatMessage(role="user", content="w1 w2")],
                                 max_tokens=4, temperature=0.0)


async def test_a_speculative_request_says_so_and_scores_nothing(armed, monkeypatch):
    svc = armed
    draft = word_model(seed=1)
    monkeypatch.setattr(svc, "_get_draft_model", lambda: draft)
    svc._speculative_num_tokens = 2
    observed = {"n": 0}
    real = ProbeRuntimeState._on_activations

    def counting(self, layer, hidden):
        observed["n"] += 1
        return real(self, layer, hidden)

    monkeypatch.setattr(ProbeRuntimeState, "_on_activations", counting)
    with patch("millm.services.inference_service.torch.cuda.is_available", return_value=False):
        await svc.create_chat_completion(_chat())
    [result] = get_probe_results()
    assert result.execution_mode == "speculative"
    assert result.outcome == "skipped" and result.reason == "speculative_decoding"
    assert result.score is None
    SPIKE["observe_reached"] = observed["n"]


async def test_without_a_draft_the_same_request_is_serial_and_scored(armed):
    with patch("millm.services.inference_service.torch.cuda.is_available", return_value=False):
        await armed.create_chat_completion(_chat())
    [result] = get_probe_results()
    assert result.execution_mode == "serial" and result.outcome == "completed"


async def test_the_incoming_trace_reaches_the_result(armed):
    ref = TraceRef("0af7651916cd43dd8448eb211c80319c", "b7ad6b7169203331")
    token = _REQUEST_TRACE.set(ref)
    try:
        with patch("millm.services.inference_service.torch.cuda.is_available", return_value=False):
            await armed.create_chat_completion(_chat())
    finally:
        _REQUEST_TRACE.reset(token)
    [result] = get_probe_results()
    assert (result.trace_id, result.span_id) == (ref.trace_id, ref.span_id)


async def test_no_trace_is_null_never_invented(armed):
    with patch("millm.services.inference_service.torch.cuda.is_available", return_value=False):
        await armed.create_chat_completion(_chat())
    [result] = get_probe_results()
    assert result.trace_id is None and result.span_id is None
