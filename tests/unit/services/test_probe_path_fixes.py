"""FR-27.8: every generation path opens a probe context or records why it cannot.

The discovery guard (`test_probe_paths_discovered.py`) proves each path OPENS a context before it
generates. These tests pin what each fix must SAY — the reason, on the header and on the event —
and the two latent defects the FTDD pulled into scope: concurrent continuous-batching requests
colliding on the runtime's single context (FR-27.8h), and `/v1/completions` never sending
`X-miLLM-Probe-Verdicts` (FR-27.8g). Plus the hung-thread guard's new duty: closing an open
per-request activation capture.
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

import millm.services.inference_service as inference_module
from millm.ml.model_loader import ENGINE_LLAMACPP, LoadedModel, LoadedModelState
from millm.ml.sae_config import SAEConfig
from millm.ml.sae_wrapper import LoadedSAE
from millm.services.inference_service import get_probe_verdicts
from millm.services.probe_runtime import ProbeRequestContext, ProbeRuntimeState
from tests.unit.f25_fixtures import (
    clear_loaded,
    make_client,
    make_service,
    model_row,
    word_model,
    word_tokenizer,
)
from tests.unit.services.test_probe_paths_discovered import (
    EventLog,
    FakeCBM,
    FakeLlama,
    _chat,
    _probe,
    _text,
)


@pytest.fixture(autouse=True)
def clean(monkeypatch):
    ProbeRuntimeState.reset_for_tests()

    @asynccontextmanager
    async def no_db():
        raise RuntimeError("no database here")
        yield  # pragma: no cover

    monkeypatch.setattr("millm.db.base.async_session_factory", no_db)
    yield
    ProbeRuntimeState.reset_for_tests()
    clear_loaded()


def _service(monkeypatch, *, armed=True, cbm=False, llama=False, log=None):
    log = log or EventLog()
    model = word_model()
    svc = make_service(model, word_tokenizer())
    if armed:
        ProbeRuntimeState().arm(_probe(), model)
    if cbm:
        from millm.core.config import settings

        monkeypatch.setattr(settings, "PROBE_FORCE_SERIAL", False)
        svc._cbm_backend = FakeCBM(log)
    if llama:
        LoadedModelState().set(LoadedModel(
            model_id=1, model_name="tiny", model=FakeLlama(log), tokenizer=None,
            loaded_at=datetime(2026, 10, 6), engine=ENGINE_LLAMACPP,
        ))
    records: list = []
    real = svc._probe_record

    async def record(context, verdicts=None, full_ids=None, **kwargs):
        if context is not None:
            verdicts = verdicts if verdicts is not None else context.finish()
            records.append((context, list(verdicts), kwargs))
        return await real(context, verdicts, full_ids=full_ids, **kwargs)

    svc._probe_record = record
    return svc, records


def _reasons(records):
    return [[v.not_scored_reason for v in verdicts] for _, verdicts, _ in records]


async def _drain(agen):
    return [chunk async for chunk in agen]


# ── 2.1 the detached seams ───────────────────────────────────────────────────────


class TestDetachedContexts:
    def test_a_detached_context_is_never_registered(self, monkeypatch):
        svc, _ = _service(monkeypatch)
        context = svc._probe_begin_detached("cmpl-x", "continuous_batching")
        assert isinstance(context, ProbeRequestContext)
        assert context.not_scored_reason == "continuous_batching"
        assert ProbeRuntimeState().current_request() is None

    def test_nothing_armed_means_no_context(self, monkeypatch):
        svc, _ = _service(monkeypatch, armed=False)
        assert svc._probe_begin_detached("cmpl-x", "batched_request") is None

    async def test_recording_a_detached_context_leaves_another_context_alone(self, monkeypatch):
        """FR-27.8h: `_probe_record` used to call `end_request()` unconditionally, closing
        whichever context was open — another request's, under continuous batching."""
        svc, _ = _service(monkeypatch)
        other = ProbeRuntimeState().begin_request("chatcmpl-other")
        detached = svc._probe_begin_detached("chatcmpl-mine", "continuous_batching")
        await svc._probe_record(detached, detached.finish(), full_ids=None, detached=True)
        assert ProbeRuntimeState().current_request() is other

    async def test_a_registered_record_still_closes_its_context(self, monkeypatch):
        """The control: without `detached`, the record closes the slot as before."""
        svc, _ = _service(monkeypatch)
        context = ProbeRuntimeState().begin_request("chatcmpl-a")
        await svc._probe_record(context, context.finish(), full_ids=None)
        assert ProbeRuntimeState().current_request() is None


# ── 2.2 – 2.5 each fixed path says why ───────────────────────────────────────────


class TestEachPathStatesItsReason:
    async def test_batched_chat_says_batched_request(self, monkeypatch):
        from millm.api.schemas.openai import ChatMessage

        svc, records = _service(monkeypatch)
        result = await svc.create_chat_completion(
            _chat(extra_messages=[[ChatMessage(role="user", content="w3")]])
        )
        assert len(result.choices) == 2
        assert _reasons(records) == [["batched_request"]]
        assert records[0][2] == {"detached": True}
        assert [v.not_scored_reason for v in get_probe_verdicts()] == ["batched_request"]
        assert ProbeRuntimeState().current_request() is None

    async def test_cbm_chat_says_continuous_batching(self, monkeypatch):
        svc, records = _service(monkeypatch, cbm=True)
        await svc.create_chat_completion(_chat())
        assert _reasons(records) == [["continuous_batching"]]
        assert [v.not_scored_reason for v in get_probe_verdicts()] == ["continuous_batching"]

    async def test_cbm_text_says_continuous_batching(self, monkeypatch):
        svc, records = _service(monkeypatch, cbm=True)
        await svc.create_text_completion(_text())
        assert _reasons(records) == [["continuous_batching"]]
        assert [v.not_scored_reason for v in get_probe_verdicts()] == ["continuous_batching"]

    @pytest.mark.parametrize("call", ["chat", "stream", "text"])
    async def test_llamacpp_says_engine_unsupported(self, call, monkeypatch):
        svc, records = _service(monkeypatch, llama=True)
        if call == "chat":
            await svc.create_chat_completion(_chat())
        elif call == "stream":
            chunks = await _drain(svc.stream_chat_completion(_chat(stream=True)))
            assert any("millm_probe_verdicts" in c and "engine_unsupported" in c for c in chunks), (
                "the streaming path must carry the reason in its terminal chunk"
            )
        else:
            await svc.create_text_completion(_text())
        assert _reasons(records) == [["engine_unsupported"]]


# ── 2.4 the latent collision ─────────────────────────────────────────────────────


class GatedCBM(FakeCBM):
    """Holds every generation until both requests are in flight, so they truly overlap."""

    def __init__(self, log, expected: int) -> None:
        super().__init__(log)
        self.expected = expected
        self.inflight = 0
        self.both = asyncio.Event()

    async def _gate(self):
        self.inflight += 1
        if self.inflight >= self.expected:
            self.both.set()
        await asyncio.wait_for(self.both.wait(), 5)

    async def generate(self, input_ids, max_new_tokens, request_id):
        await self._gate()
        return [7], "stop"

    async def generate_stream(self, input_ids, max_new_tokens, request_id):
        await self._gate()
        yield [7]


class TestConcurrentContinuousBatching:
    async def test_two_concurrent_cbm_streams_BOTH_record(self, monkeypatch):
        """FR-27.8h. With `_probe_begin` the second request found the runtime's slot occupied,
        got None, and recorded nothing — a probe silently quiet on half the traffic."""
        svc, records = _service(monkeypatch, cbm=True)
        svc._cbm_backend = GatedCBM(EventLog(), expected=2)
        await asyncio.gather(
            _drain(svc.stream_chat_completion(_chat(stream=True))),
            _drain(svc.stream_chat_completion(_chat(stream=True))),
        )
        assert _reasons(records) == [["continuous_batching"], ["continuous_batching"]]
        assert len({ctx.request_id for ctx, _, _ in records}) == 2
        assert ProbeRuntimeState().current_request() is None

    async def test_two_concurrent_cbm_chats_BOTH_record(self, monkeypatch):
        svc, records = _service(monkeypatch, cbm=True)
        svc._cbm_backend = GatedCBM(EventLog(), expected=2)
        await asyncio.gather(
            svc.create_chat_completion(_chat()), svc.create_chat_completion(_chat())
        )
        assert _reasons(records) == [["continuous_batching"], ["continuous_batching"]]

    async def test_a_cbm_request_does_not_close_a_serial_requests_context(self, monkeypatch):
        """The other half of FR-27.8h: a CBM request finishing while another context is open
        must leave that context open."""
        svc, _ = _service(monkeypatch, cbm=True)
        other = ProbeRuntimeState().begin_request("chatcmpl-serial")
        await svc.create_chat_completion(_chat())
        assert ProbeRuntimeState().current_request() is other


# ── 2.6 a failed generation still records ────────────────────────────────────────


class Boom(RuntimeError):
    pass


class TestFailuresStillRecord:
    async def test_batched(self, monkeypatch):
        from millm.api.schemas.openai import ChatMessage

        svc, records = _service(monkeypatch)

        def explode(*a, **k):
            raise Boom("generation failed")

        svc._generate_sync = explode
        with pytest.raises(Boom):
            await svc.create_chat_completion(
                _chat(extra_messages=[[ChatMessage(role="user", content="w3")]])
            )
        assert _reasons(records) == [["batched_request"]]

    @pytest.mark.parametrize("call", ["chat", "text", "stream"])
    async def test_cbm(self, call, monkeypatch):
        svc, records = _service(monkeypatch, cbm=True)

        async def explode(*a, **k):
            raise Boom("cbm failed")

        async def explode_stream(*a, **k):
            raise Boom("cbm failed")
            yield  # pragma: no cover

        svc._cbm_backend.generate = explode
        svc._cbm_backend.generate_stream = explode_stream
        with pytest.raises(Boom):
            if call == "chat":
                await svc.create_chat_completion(_chat())
            elif call == "text":
                await svc.create_text_completion(_text())
            else:
                await _drain(svc.stream_chat_completion(_chat(stream=True)))
        assert _reasons(records) == [["continuous_batching"]]

    @pytest.mark.parametrize("call", ["chat", "text", "stream"])
    async def test_llamacpp(self, call, monkeypatch):
        svc, records = _service(monkeypatch, llama=True)
        llama = svc._model

        def explode(*a, **k):
            raise Boom("llama failed")

        llama.create_completion = explode
        llama.create_chat_completion = explode
        if call == "stream":
            # The streaming path turns a failure into an SSE error event (headers are sent).
            chunks = await _drain(svc.stream_chat_completion(_chat(stream=True)))
            assert any('"error"' in c for c in chunks)
        else:
            # Translated by `_translate_llamacpp_error`, so the type is not Boom.
            with pytest.raises(Exception):  # noqa: B017
                if call == "chat":
                    await svc.create_chat_completion(_chat())
                else:
                    await svc.create_text_completion(_text())
        assert _reasons(records) == [["engine_unsupported"]]


# ── 2.7 /v1/completions sends the verdict header ─────────────────────────────────


class TestCompletionsVerdictHeader:
    def _client(self, armed: bool):
        model = word_model()
        svc = make_service(model, word_tokenizer())
        if armed:
            ProbeRuntimeState().arm(_probe(), model)
        client, _ = make_client(svc, model_row())
        return client

    def test_present_with_a_probe_armed(self):
        client = self._client(armed=True)
        with patch("millm.services.inference_service.torch.cuda.is_available", return_value=False):
            response = client.post(
                "/v1/completions", json={"model": "tiny", "prompt": "w1 w2", "max_tokens": 2,
                                         "temperature": 0},
            )
        assert response.status_code == 200, response.text
        header = response.headers.get("X-miLLM-Probe-Verdicts")
        assert header, "FR-27.8g: /v1/completions carried no verdict header with a probe armed"
        assert header.startswith('"guard";score=') and "window=all" in header

    def test_absent_with_nothing_armed(self):
        client = self._client(armed=False)
        with patch("millm.services.inference_service.torch.cuda.is_available", return_value=False):
            response = client.post(
                "/v1/completions", json={"model": "tiny", "prompt": "w1 w2", "max_tokens": 2,
                                         "temperature": 0},
            )
        assert response.status_code == 200, response.text
        assert "X-miLLM-Probe-Verdicts" not in response.headers

    def test_batched_chat_header_says_batched_request(self):
        """BRD-04 acceptance 17 / US-7, through the live route."""
        client = self._client(armed=True)
        with patch("millm.services.inference_service.torch.cuda.is_available", return_value=False):
            response = client.post("/v1/chat/completions", json={
                "model": "tiny", "max_tokens": 2, "temperature": 0,
                "messages": [{"role": "user", "content": "w1"}],
                "extra_messages": [[{"role": "user", "content": "w2"}]],
            })
        assert response.status_code == 200, response.text
        header = response.headers["X-miLLM-Probe-Verdicts"]
        assert 'not-scored;reason="batched_request"' in header


# ── 2.8 the hung-thread guard closes an open activation capture ──────────────────


class _HungThread:
    """A generation thread that runs its target and then claims to still be alive."""

    def __init__(self, target, args=(), kwargs=None):
        self._target, self._args, self._kwargs = target, args, kwargs or {}

    def start(self):
        self._target(*self._args, **self._kwargs)

    def join(self, timeout=None):
        return None

    def is_alive(self):
        return True


class TestHungThreadClosesCapture:
    async def test_an_open_capture_is_closed_when_the_thread_hangs(self, monkeypatch):
        svc, _ = _service(monkeypatch, armed=False)
        sae = LoadedSAE(W_enc=torch.randn(16, 8), b_enc=torch.zeros(8), W_dec=torch.randn(8, 16),
                        b_dec=torch.zeros(16), config=SAEConfig(d_in=16, d_sae=8, model_name="t",
                                                                hook_name="t", hook_layer=0))
        sae.begin_request_capture(object())
        monkeypatch.setattr(
            "millm.services.sae_service.AttachedSAEState.entries",
            lambda self: [SimpleNamespace(sae=sae, sae_id="s", layer=0)],
        )
        monkeypatch.setattr(inference_module, "Thread", _HungThread)
        with patch("millm.services.inference_service.torch.cuda.is_available", return_value=False):
            await _drain(svc.stream_chat_completion(_chat(stream=True)))
        assert sae.request_capture is None, "a hung thread could feed the next request's capture"

    def test_a_second_open_capture_is_refused(self):
        sae = LoadedSAE(W_enc=torch.randn(16, 8), b_enc=torch.zeros(8), W_dec=torch.randn(8, 16),
                        b_dec=torch.zeros(16), config=SAEConfig(d_in=16, d_sae=8, model_name="t",
                                                                hook_name="t", hook_layer=0))
        first = object()
        sae.begin_request_capture(first)
        with pytest.raises(RuntimeError, match="already open"):
            sae.begin_request_capture(object())
        assert sae.end_request_capture() is first
        assert sae.request_capture is None
