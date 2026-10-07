"""Per-request SAE activations (Feature 27, FR-27.1 – FR-27.3): the capture, its positions, the read
point, isolation, and every seam that wires it into a serving path.

A TINY REAL Llama with a REAL `LoadedSAE` hooked on layer 0 through the real `SAEHooker`; the SAE
registry lookup is the one thing patched (attaching through `SAEService` needs a database row).

Mutations this catches (FTID §8): M18 (capture never cleared — shared across requests) and M19 (read
point ignored, always pre).
"""

from __future__ import annotations

import asyncio
import threading
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from millm.api.schemas.millm_extension import ReturnSaeActivations
from millm.api.schemas.openai import ChatCompletionRequest, ChatMessage, TextCompletionRequest
from millm.ml.sae_config import SAEConfig
from millm.ml.sae_hooker import SAEHooker
from millm.ml.sae_wrapper import CAPTURE_OWNER, LoadedSAE
from millm.services.probe_runtime import ProbeRuntimeState
from millm.services.request_activations import RequestActivationCapture, resolve_positions
from tests.unit.f25_fixtures import clear_loaded, make_service, word_model, word_tokenizer

D_SAE = 8


@pytest.fixture(autouse=True)
def clean():
    ProbeRuntimeState.reset_for_tests()
    yield
    ProbeRuntimeState.reset_for_tests()
    clear_loaded()
    CAPTURE_OWNER.set(None)


def make_sae(seed: int = 3, d_sae: int = D_SAE) -> LoadedSAE:
    torch.manual_seed(seed)
    return LoadedSAE(W_enc=torch.randn(16, d_sae), b_enc=torch.zeros(d_sae),
                     W_dec=torch.randn(d_sae, 16), b_dec=torch.zeros(16),
                     config=SAEConfig(d_in=16, d_sae=d_sae, model_name="t", hook_name="t",
                                      hook_layer=0))


@pytest.fixture
def served(monkeypatch):
    """A service over a tiny model with one SAE attached (hooked) on layer 0."""
    model, tokenizer = word_model(), word_tokenizer()
    svc = make_service(model, tokenizer)
    sae = make_sae()
    handle = SAEHooker().install(model, 0, sae)
    monkeypatch.setattr(
        "millm.services.sae_service.AttachedSAEState.entries",
        lambda self: [SimpleNamespace(sae=sae, sae_id="sae_a", layer=0)],
    )
    yield svc, model, tokenizer, sae
    handle.remove()


def spec(**over) -> ReturnSaeActivations:
    base = {"top_k": 3, "positions": "all"}
    base.update(over)
    return ReturnSaeActivations(**base)


def chat(**over) -> ChatCompletionRequest:
    base = dict(model="tiny", messages=[ChatMessage(role="user", content="w1 w2 w3")],
                max_tokens=3, temperature=0.0)
    base.update(over)
    return ChatCompletionRequest(**base)


def text(**over) -> TextCompletionRequest:
    base = dict(model="tiny", prompt="w1 w2 w3", max_tokens=3, temperature=0.0)
    base.update(over)
    return TextCompletionRequest(**base)


def no_cuda():
    return patch("millm.services.inference_service.torch.cuda.is_available", return_value=False)


# ── 6.3 the capture itself ───────────────────────────────────────────────────────


class TestPositions:
    def _capture(self, positions, n_prompt=3, sae=None, **over):
        sae = sae or make_sae()
        return RequestActivationCapture(spec=spec(positions=positions, **over), n_prompt=n_prompt,
                                        sae=sae, sae_id="s", layer=0, read_point="post_steering")

    def _feed(self, cap, widths):
        for w in widths:
            cap.observe(torch.randn(1, w, 16), "post")

    @pytest.mark.parametrize(("positions", "expected"), [
        ("all", [0, 1, 2, 3, 4]),
        ("prompt", [0, 1, 2]),
        ("completion", [3, 4]),
        ("last", [4]),
        ({"start": 2, "end": 4}, [2, 3]),
    ])
    def test_each_kind_across_a_prefill_and_two_decode_steps(self, positions, expected):
        cap = self._capture(positions)
        self._feed(cap, [3, 1, 1])
        assert [k.position for k in cap.kept] == expected

    def test_the_offset_advances_by_the_full_pass_width_whatever_was_kept(self):
        cap = self._capture({"start": 4, "end": 5})
        self._feed(cap, [3, 1, 1])  # nothing kept from the first two passes
        assert [k.position for k in cap.kept] == [4]
        assert cap.offset == 5 and cap.passes == 3

    def test_features_then_top_k(self):
        sae = make_sae()
        cap = self._capture("all", sae=sae, features=[1, 5, 6], top_k=2)
        hidden = torch.randn(1, 2, 16)
        cap.observe(hidden, "post")
        acts = sae.encode(hidden[0])[:, [1, 5, 6]]
        for row, kept in zip(acts, cap.kept, strict=True):
            v, i = torch.topk(row, 2)
            assert kept.indices == [[1, 5, 6][j] for j in i.tolist()]
            assert kept.values == pytest.approx(v.tolist())

    def test_the_other_phase_is_ignored(self):
        cap = self._capture("all")
        cap.observe(torch.randn(1, 3, 16), "pre")
        assert cap.kept == [] and cap.offset == 0

    def test_worst_case_counts(self):
        assert resolve_positions(spec(positions="last"), 10, 5) == 1
        assert resolve_positions(spec(positions="prompt"), 10, 5) == 10
        assert resolve_positions(spec(positions="completion"), 10, 5) == 5
        assert resolve_positions(spec(positions="all"), 10, 5) == 15
        assert resolve_positions(spec(positions={"start": 12, "end": 100}), 10, 5) == 3


class TestChunkedEncode:
    def test_encode_runs_in_chunks(self):
        """6.5: a long prefill against a wide SAE is encoded in chunks, never all at once."""
        sae = make_sae(d_sae=64)
        calls = []
        real = sae.encode

        def counting(x):
            calls.append(x.shape[0])
            return real(x)

        sae.encode = counting
        cap = RequestActivationCapture(spec=spec(positions="all"), n_prompt=10, sae=sae,
                                       sae_id="s", layer=0, read_point="post_steering",
                                       encode_chunk=4)
        cap.observe(torch.randn(1, 10, 16), "post")
        assert calls == [4, 4, 2]
        assert len(cap.kept) == 10


# ── 6.2 the read point ───────────────────────────────────────────────────────────


class TestReadPoint:
    async def test_pre_and_post_differ_when_this_layer_steers(self, served):
        svc, _model, _tok, sae = served
        sae.set_steering(2, 30.0)
        sae.enable_steering(True)
        with no_cuda():
            post = await svc.create_chat_completion(chat(return_sae_activations=spec()))
            pre = await svc.create_chat_completion(
                chat(return_sae_activations=spec(read_point="pre_steering")))
        a, b = post.millm.sae_activations, pre.millm.sae_activations
        assert a.read_point == "post_steering" and b.read_point == "pre_steering"
        assert a.positions[0].features != b.positions[0].features

    async def test_scoring_mode_records_under_suppression_and_says_unsteered(self, served):
        svc, _model, _tok, sae = served
        sae.set_steering(2, 30.0)
        sae.enable_steering(True)
        req = text(max_tokens=1, logprobs=1, return_sae_activations=spec(positions="last"))
        with no_cuda():
            out = await svc.create_text_completion(req)
        block = out.millm.sae_activations
        assert block.read_point == "unsteered"
        n_prompt = len(word_tokenizer()("w1 w2 w3")["input_ids"])
        # FR-27.3c: `last` in scoring mode is the last PROMPT position.
        assert [p.position for p in block.positions] == [n_prompt - 1]

    def test_the_note_says_it_is_not_a_counterfactual(self):
        cap = RequestActivationCapture(spec=spec(), n_prompt=1, sae=make_sae(), sae_id="s",
                                       layer=0, read_point="post_steering")
        assert "not an unsteered counterfactual" in cap.build([1])["note"]


# ── 6.6 every seam, payload and count ────────────────────────────────────────────


class TestEverySeamIsWired:
    def _spy(self, svc):
        calls = {"begin": 0, "finish": 0}
        real_b, real_f = svc._activations_begin, svc._activations_finish

        def begin(*a, **k):
            calls["begin"] += 1
            return real_b(*a, **k)

        def finish(*a, **k):
            calls["finish"] += 1
            return real_f(*a, **k)

        svc._activations_begin, svc._activations_finish = begin, finish
        return calls

    async def test_serial_chat(self, served):
        svc = served[0]
        calls = self._spy(svc)
        with no_cuda():
            out = await svc.create_chat_completion(chat(return_sae_activations=spec()))
        assert calls == {"begin": 1, "finish": 1}
        block = out.millm.sae_activations
        n_prompt = out.usage.prompt_tokens
        processed = n_prompt + out.usage.completion_tokens - 1  # the last token is never fed
        assert [p.position for p in block.positions] == list(range(processed))
        assert block.sae_id == "sae_a" and block.layer == 0
        assert all(len(p.features) == 3 for p in block.positions)

    async def test_serial_text(self, served):
        svc = served[0]
        calls = self._spy(svc)
        with no_cuda():
            out = await svc.create_text_completion(text(return_sae_activations=spec(
                positions="prompt")))
        assert calls == {"begin": 1, "finish": 1}
        assert [p.position for p in out.millm.sae_activations.positions] == list(
            range(out.usage.prompt_tokens))

    async def test_streaming_chat_puts_one_chunk_after_the_probe_chunk(self, served):
        """6.7 / T-76: … final chunk, probe chunk, `millm` chunk, [Feature 28's steering chunk],
        [DONE]."""
        import json

        from tests.unit.services.test_probe_paths_discovered import _probe

        svc, model, _tok, _sae = served
        ProbeRuntimeState().arm(_probe(), model)
        calls = self._spy(svc)
        with no_cuda():
            chunks = [c async for c in svc.stream_chat_completion(
                chat(stream=True, return_sae_activations=spec(positions="last")))]
        assert calls == {"begin": 1, "finish": 1}
        assert chunks[-1] == "data: [DONE]\n\n"
        steering = json.loads(chunks[-2][len("data: "):])
        millm = json.loads(chunks[-3][len("data: "):])
        probe = json.loads(chunks[-4][len("data: "):])
        final = json.loads(chunks[-5][len("data: "):])
        assert steering["choices"] == [] and "millm_steering" in steering
        assert millm["choices"] == [] and "sae_activations" in millm["millm"]
        assert "millm_probe_verdicts" in probe
        assert final["choices"][0]["finish_reason"] is not None
        assert len(millm["millm"]["sae_activations"]["positions"]) == 1

    async def test_scoring_chat(self, served):
        svc = served[0]
        calls = self._spy(svc)
        with no_cuda():
            out = await svc.create_chat_completion(chat(
                max_tokens=1, logprobs=True, return_sae_activations=spec(positions="last")))
        assert calls == {"begin": 1, "finish": 1}
        assert out.millm.sae_activations.read_point == "unsteered"

    async def test_activation_requests_route_serial(self, served):
        class _CBM:
            is_running = True

            def sampling_params_match(self, *a):
                return True

        svc = served[0]
        svc._cbm_backend = _CBM()
        assert svc._use_cbm_for_request(**svc._cbm_route_kwargs(chat())) is True
        assert svc._use_cbm_for_request(
            **svc._cbm_route_kwargs(chat(return_sae_activations=spec()))) is False

    async def test_no_field_means_no_millm_and_an_unchanged_body(self, served):
        svc = served[0]
        with no_cuda():
            out = await svc.create_chat_completion(chat())
        assert out.millm is None
        assert "millm" not in out.model_dump_json()


# ── 6.8 isolation ────────────────────────────────────────────────────────────────


class TestIsolation:
    async def test_two_interleaved_requests_each_get_only_their_own(self, served):
        """BRD-04 acceptance 10. Two requests in flight together; each response's positions and
        token ids are its own prompt's."""
        svc, _model, tokenizer, _sae = served
        with no_cuda():
            a, b = await asyncio.gather(
                svc.create_chat_completion(chat(
                    messages=[ChatMessage(role="user", content="w1")],
                    return_sae_activations=spec(positions="prompt"))),
                svc.create_chat_completion(chat(
                    messages=[ChatMessage(role="user", content="w4 w5 w6 w7 w8")],
                    return_sae_activations=spec(positions="prompt"))),
            )
        for out, content in ((a, "w1"), (b, "w4 w5 w6 w7 w8")):
            rendered = tokenizer.apply_chat_template([{"role": "user", "content": content}],
                                                     tokenize=False, add_generation_prompt=True)
            ids = tokenizer(rendered)["input_ids"]
            block = out.millm.sae_activations
            assert [p.position for p in block.positions] == list(range(len(ids)))
            assert [p.token_id for p in block.positions] == ids

    def test_a_forward_without_the_owner_does_not_feed_the_capture(self, served):
        """A concurrent continuous-batching generation (or a hung thread) runs the same hook with
        no owner in its context; it must not write into this request's activations."""
        _svc, model, _tok, sae = served
        cap = RequestActivationCapture(spec=spec(), n_prompt=3, sae=sae, sae_id="sae_a", layer=0,
                                       read_point="post_steering")
        sae.begin_request_capture(cap)
        try:
            worker = threading.Thread(target=lambda: model(input_ids=torch.tensor([[2, 6, 7]])))
            worker.start()
            worker.join(10)
            assert cap.kept == [] and cap.passes == 0
            token = CAPTURE_OWNER.set(cap.owner)
            try:
                with torch.no_grad():
                    model(input_ids=torch.tensor([[2, 6, 7]]))
            finally:
                CAPTURE_OWNER.reset(token)
            assert cap.passes == 1 and len(cap.kept) == 3
        finally:
            sae.end_request_capture()

    async def test_a_failed_generation_still_closes_the_capture(self, served):
        svc, _model, _tok, sae = served

        def explode(*a, **k):
            raise RuntimeError("boom")

        svc._generate_sync = explode
        with no_cuda(), pytest.raises(RuntimeError):
            await svc.create_chat_completion(chat(return_sae_activations=spec()))
        assert sae.request_capture is None
