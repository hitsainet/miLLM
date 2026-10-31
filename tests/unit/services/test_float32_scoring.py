"""Float32 scoring on `/v1/completions` (Feature 25, FR-25.15; suite decision D61).

Consumer: miForge F11 R-02.2 (`0xcc/tdds/011_FTDD|Optional_Hot_Update.md` §6.10). miForge compares
the next-token log-probabilities miLLM serves with its own; in bfloat16 the kernel noise between
the two processes is larger than a 1 % reconstruction error, so it asks for `scoring_precision:
"float32"` and requires the echo. These tests run a TINY REAL Llama in bfloat16 on the CPU — never
a mocked model — and compare with an independently built float32 reference (`deepcopy(...).float()`),
not with anything the code under test computed.

What each class pins (negative controls are recorded in 025_FTASKS §12):
* TestNumerics — float32 answers equal the float32 reference within 1e-6; the native answer is
  the bfloat16 forward, and its gap to float32 is larger than a 1 % block error (the blind spot);
  in float32 the 1 % error stands out from a clean comparison by orders of magnitude.
* TestTheServedModelIsUntouched / TestTheCopyIsReleased — no cast in place, nothing left resident.
* TestTheEcho — the body and header say what RAN; absent when not asked.
* TestRefusals — 409 SCORING_PRECISION_UNAVAILABLE by reason, before any copy is built.
* TestPolicy / TestHttp — the field is refused where it cannot be honoured, never ignored.
"""

from __future__ import annotations

import copy
import gc
import weakref
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import torch
from fastapi.testclient import TestClient
from pydantic import ValidationError
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

from millm.api.schemas.openai import TextCompletionRequest
from millm.core.errors import InvalidScoringRequestError, ScoringPrecisionUnavailableError
from millm.ml.model_loader import LoadedModel, LoadedModelState
from millm.services import float32_scoring as f32
from millm.services.inference_service import InferenceService

WORDS = ["<s>", "[UNK]"] + [f"w{i}" for i in range(30)]
PROMPT = "w3 w7 w1 w9 w2"
ALLOWED = [5, 9, 11, 20, 3]


def _tokenizer() -> PreTrainedTokenizerFast:
    tok = Tokenizer(models.WordLevel({w: i for i, w in enumerate(WORDS)}, unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    fast = PreTrainedTokenizerFast(tokenizer_object=tok, bos_token="<s>", unk_token="[UNK]")
    fast.pad_token = "[UNK]"
    return fast


def _float32_model() -> LlamaForCausalLM:
    # initializer_range 0.2: logits spread enough that bfloat16 rounding is visible (measured
    # 0.034 against a 1 % block error of 0.015 — the shape of miForge's node numbers).
    torch.manual_seed(0)
    config = LlamaConfig(vocab_size=len(WORDS), hidden_size=64, intermediate_size=128,
                         num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=4,
                         max_position_embeddings=128, initializer_range=0.2)
    return LlamaForCausalLM(config).eval()


@pytest.fixture(autouse=True)
def _clean_state():
    state = LoadedModelState()
    state._loaded = None
    yield
    state._loaded = None
    assert not f32._COPY_LOCK.locked(), "a float32 request left the copy lock held"


@pytest.fixture
def bf16() -> LlamaForCausalLM:
    return _float32_model().to(torch.bfloat16)


def _service(model, *, dtype: str = "bfloat16", quantization: str = "none") -> InferenceService:
    LoadedModelState().set(LoadedModel(
        model_id=1, model_name="tiny", model=model, tokenizer=_tokenizer(),
        loaded_at=datetime(2026, 10, 18), memory_used_mb=1, num_parameters=1, device="cpu",
        dtype=dtype, quantization_method=quantization,
    ))
    with patch("millm.services.inference_service.torch.cuda.is_available", return_value=False):
        svc = InferenceService(model_service=None)
    svc._device = "cpu"
    return svc


def _request(**over) -> TextCompletionRequest:
    body = {"model": "tiny", "prompt": PROMPT, "max_tokens": 1, "temperature": 1.0,
            "logprobs": 5, "allowed_token_ids": ALLOWED, "add_special_tokens": False,
            "return_tokens_as_token_ids": True, "scoring_precision": "float32"}
    body.update(over)
    return TextCompletionRequest(**body)


def _reference(model, prompt: str = PROMPT) -> dict[str, float]:
    """Log-probabilities over ALLOWED from an independent forward of `model`."""
    ids = _tokenizer()(prompt, return_tensors="pt", add_special_tokens=False)
    with torch.no_grad():
        logits = model(**ids).logits[0, -1].float()
    lp = torch.log_softmax(logits[ALLOWED], 0)
    return {f"token_id:{t}": float(v) for t, v in zip(ALLOWED, lp, strict=True)}


def _scores(response) -> dict[str, float]:
    return dict(response.choices[0].logprobs.top_logprobs[0])


def _max_diff(a: dict[str, float], b: dict[str, float]) -> float:
    assert set(a) == set(b)
    return max(abs(a[k] - b[k]) for k in a)


async def _score(svc, request):
    """Score through the service with the request-scoped outcome reset, as the route does."""
    from millm.services.inference_service import reset_steering_memo

    reset_steering_memo()
    return await svc.create_text_completion(request)


class _BuildSpy:
    """Counts float32 copies and keeps WEAK references to each copy's module."""

    def __init__(self):
        self.real = f32.Float32Copy.build
        self.refs: list[weakref.ref] = []

    def __call__(self, *args, **kwargs):
        built = self.real(*args, **kwargs)
        self.refs.append(weakref.ref(built.module))
        return built


@pytest.fixture
def build_spy(monkeypatch):
    spy = _BuildSpy()
    monkeypatch.setattr(f32.Float32Copy, "build", spy)
    return spy


class TestNumerics:
    async def test_float32_equals_an_independent_float32_reference(self, bf16, build_spy):
        reference = _reference(copy.deepcopy(bf16).float())
        response = await _score(_service(bf16), _request())
        assert _max_diff(_scores(response), reference) <= 1e-6
        assert len(build_spy.refs) == 1

    async def test_native_is_the_bfloat16_forward_and_differs_by_the_rounding_gap(self, bf16):
        float32_ref = _reference(copy.deepcopy(bf16).float())
        bf16_ref = _reference(bf16)
        svc = _service(bf16)
        native = await _score(svc, _request(scoring_precision=None))
        float32 = await _score(svc, _request())
        assert _max_diff(_scores(native), bf16_ref) <= 1e-6
        gap = _max_diff(_scores(native), _scores(float32))
        # The bfloat16 rounding gap, measured from the two references, is what separates them.
        assert gap == pytest.approx(_max_diff(bf16_ref, float32_ref), abs=1e-6)
        assert gap > 1e-3

    async def test_a_one_percent_block_error_is_visible_in_float32_and_hidden_in_bf16(self, bf16):
        """The in-suite analogue of the live check (miForge FTDD §6.10's synthetic numbers).

        miForge's reference is float32 over the CLEAN weights; the served set carries a 1 % error
        in one block. In float32 the clean comparison is ~0 and the error stands out; in bfloat16
        the clean comparison alone is already larger than the error.
        """
        clean_reference = _reference(copy.deepcopy(bf16).float())
        broken = copy.deepcopy(bf16)
        with torch.no_grad():
            broken.model.layers[1].mlp.down_proj.weight.mul_(1.01)
        clean = await _score(_service(bf16), _request())
        clean_spread = _max_diff(_scores(clean), clean_reference)
        error = _max_diff(_scores(await _score(_service(broken), _request())), clean_reference)
        native_clean = _max_diff(
            _scores(await _score(_service(bf16), _request(scoring_precision=None))),
            clean_reference,
        )
        assert clean_spread <= 1e-6
        assert error > 1e-3 and error > 1000 * max(clean_spread, 1e-9)
        assert native_clean > error  # the blind spot float32 removes

    async def test_temperature_and_the_full_vocabulary_use_the_same_arithmetic(self, bf16):
        ref_model = copy.deepcopy(bf16).float()
        ids = _tokenizer()(PROMPT, return_tensors="pt", add_special_tokens=False)
        with torch.no_grad():
            logits = ref_model(**ids).logits[0, -1].float()
        full = torch.log_softmax(logits / 2.0, 0)
        values, top = torch.topk(full, 3)
        response = await _score(_service(bf16), _request(allowed_token_ids=None, logprobs=3,
                                                          temperature=2.0))
        got = _scores(response)
        assert list(got) == [f"token_id:{int(i)}" for i in top]
        assert list(got.values()) == pytest.approx([float(v) for v in values], abs=1e-6)

    async def test_several_prompts_share_one_copy(self, bf16, build_spy):
        prompts = [PROMPT, "w1 w2 w3"]
        response = await _score(_service(bf16), _request(prompt=prompts))
        assert len(build_spy.refs) == 1
        ref_model = copy.deepcopy(bf16).float()
        for choice, prompt in zip(response.choices, prompts, strict=True):
            assert _max_diff(dict(choice.logprobs.top_logprobs[0]),
                             _reference(ref_model, prompt)) <= 1e-6

    async def test_a_float32_model_needs_no_copy(self, build_spy):
        model = _float32_model()
        response = await _score(_service(model, dtype="float32"), _request())
        assert build_spy.refs == []
        assert response.scoring_precision == "float32"
        assert _max_diff(_scores(response), _reference(model)) <= 1e-6


class TestFullFloat32Matmuls:
    async def test_the_forward_runs_at_highest_matmul_precision_and_restores_it(
        self, bf16, monkeypatch
    ):
        seen: list[str] = []
        real = f32.Float32Copy.build

        def build(*a, **k):
            built = real(*a, **k)
            built.module.model.layers[0].register_forward_hook(
                lambda *_: seen.append(torch.get_float32_matmul_precision()))
            return built

        monkeypatch.setattr(f32.Float32Copy, "build", build)
        torch.set_float32_matmul_precision("medium")
        try:
            await _score(_service(bf16), _request())
            assert seen == ["highest"]
            assert torch.get_float32_matmul_precision() == "medium"
        finally:
            torch.set_float32_matmul_precision("highest")


class TestTheServedModelIsUntouched:
    async def test_no_served_tensor_is_cast_moved_or_written(self, bf16):
        before = {n: (t.dtype, t.device, t.data_ptr(), t.detach().clone())
                  for n, t in f32._named_tensors(bf16)}
        await _score(_service(bf16), _request())
        after = dict(f32._named_tensors(bf16))
        assert set(after) == set(before)
        for name, (dtype, device, ptr, value) in before.items():
            tensor = after[name]
            assert (tensor.dtype, tensor.device, tensor.data_ptr()) == (dtype, device, ptr), name
            assert torch.equal(tensor, value), name

    async def test_tied_weights_stay_tied_in_the_copy(self):
        torch.manual_seed(1)
        config = LlamaConfig(vocab_size=len(WORDS), hidden_size=32, intermediate_size=64,
                             num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2,
                             max_position_embeddings=64, tie_word_embeddings=True,
                             initializer_range=0.2)
        model = LlamaForCausalLM(config).eval().to(torch.bfloat16)
        built = f32.Float32Copy.build(model, "cpu", 1)
        try:
            assert built.module.lm_head.weight is built.module.model.embed_tokens.weight
            assert built.module.lm_head.weight.dtype == torch.float32
        finally:
            built.close()


class TestTheCopyIsReleased:
    async def test_nothing_references_the_copy_after_the_request(self, bf16, build_spy):
        await _score(_service(bf16), _request())
        gc.collect()
        assert len(build_spy.refs) == 1
        assert build_spy.refs[0]() is None, "the float32 copy is still referenced"

    async def test_released_after_a_failed_prompt_too(self, bf16, build_spy):
        with pytest.raises(InvalidScoringRequestError):
            await _score(_service(bf16), _request(allowed_token_ids=[5, 999]))
        gc.collect()
        assert len(build_spy.refs) == 1 and build_spy.refs[0]() is None
        assert not f32._COPY_LOCK.locked()

    async def test_the_copy_is_closed_exactly_once_on_success_and_on_failure(
        self, bf16, monkeypatch
    ):
        """`close()` is what returns a CARD copy's memory (empty_cache) and removes the adapter
        hooks; dropping the last reference alone frees a CPU copy, so the weakref test cannot see
        a missing close (mutation control M5). Assert the call and its count."""
        closed: list[str] = []
        real = f32.Float32Copy.close

        def close(self):
            closed.append(self.device)
            real(self)

        monkeypatch.setattr(f32.Float32Copy, "close", close)
        svc = _service(bf16)
        await _score(svc, _request())
        assert closed == ["cpu"]
        with pytest.raises(InvalidScoringRequestError):
            await _score(svc, _request(allowed_token_ids=[5, 999]))
        assert closed == ["cpu", "cpu"]

    async def test_close_empties_the_card_cache_for_a_card_copy(self):
        with patch.object(f32.torch.cuda, "is_available", return_value=True), \
                patch.object(f32.torch.cuda, "empty_cache") as empty:
            f32.Float32Copy(module=torch.nn.Linear(2, 2), device="cuda:0", needed_mb=1).close()
            f32.Float32Copy(module=torch.nn.Linear(2, 2), device="cpu", needed_mb=1).close()
        assert empty.call_count == 1


class TestTheEcho:
    async def test_body_and_outcome_say_float32(self, bf16):
        from millm.services.inference_service import get_request_outcome

        response = await _score(_service(bf16), _request())
        assert response.scoring_precision == "float32"
        assert response.model_dump(mode="json")["scoring_precision"] == "float32"
        assert get_request_outcome()["scoring_precision"] == "float32"

    async def test_native_when_asked_absent_when_not(self, bf16):
        svc = _service(bf16)
        asked = await _score(svc, _request(scoring_precision="native"))
        assert asked.scoring_precision == "native"
        silent = await _score(svc, _request(scoring_precision=None))
        assert "scoring_precision" not in silent.model_dump(mode="json")

    async def test_the_echo_is_what_ran_not_the_request_copied(self, bf16, monkeypatch):
        """If the float32 path is bypassed the echo must not appear (it is recorded by the path)."""
        svc = _service(bf16)

        async def native_only(texts, **kw):
            return await svc._score_prompts(texts, pack_size=1, **kw)

        monkeypatch.setattr(svc, "_score_prompts_float32", native_only)
        response = await _score(svc, _request())
        assert response.scoring_precision is None


class TestRefusals:
    async def test_quantized_is_refused_by_name_before_any_copy(self, bf16, build_spy):
        with pytest.raises(ScoringPrecisionUnavailableError) as info:
            await _score(_service(bf16, quantization="bitsandbytes"), _request())
        assert info.value.reason == "model_not_float32_castable"
        assert info.value.details["reason"] == "model_not_float32_castable"
        assert build_spy.refs == []

    async def test_gguf_is_refused_by_name(self, bf16):
        svc = _service(bf16)
        with patch.object(svc, "_engine_is_llamacpp", return_value=True), \
                pytest.raises(ScoringPrecisionUnavailableError) as info:
            await _score(svc, _request())
        assert info.value.reason == "model_not_float32_castable"

    async def test_a_copy_that_does_not_fit_is_refused_memory(self, bf16, build_spy, monkeypatch):
        from millm.core.config import settings

        monkeypatch.setattr(settings, "SCORING_FLOAT32_HEADROOM_MB", 10**9)
        with pytest.raises(ScoringPrecisionUnavailableError) as info:
            await _score(_service(bf16), _request())
        assert info.value.reason == "memory"
        assert info.value.details["needed_mb"] >= 1
        assert build_spy.refs == []

    async def test_a_pinned_card_without_one_is_refused_device(self, bf16, monkeypatch):
        from millm.core.config import settings

        monkeypatch.setattr(settings, "SCORING_FLOAT32_DEVICE", "cuda")
        with pytest.raises(ScoringPrecisionUnavailableError) as info:
            await _score(_service(bf16), _request())
        assert info.value.reason == "device"

    async def test_a_live_copy_makes_a_second_request_busy(self, bf16):
        svc = _service(bf16)
        assert f32.try_acquire()
        try:
            with pytest.raises(ScoringPrecisionUnavailableError) as info:
                await _score(svc, _request())
        finally:
            f32.release_lock()
        assert info.value.reason == "busy"
        assert (await _score(svc, _request())).scoring_precision == "float32"

    async def test_an_allocation_failure_is_memory_and_releases_the_lock(self, bf16, monkeypatch):
        def oom(*a, **k):
            raise RuntimeError("DefaultCPUAllocator: can't allocate memory")

        monkeypatch.setattr(f32, "_copy_tensor", oom)
        with pytest.raises(ScoringPrecisionUnavailableError) as info:
            await _score(_service(bf16), _request())
        assert info.value.reason == "memory"
        assert not f32._COPY_LOCK.locked()

    async def test_a_structure_the_copy_cannot_reproduce_is_refused(self, bf16):
        bf16.register_buffer("extra_state", torch.zeros(2), persistent=False)
        with pytest.raises(ScoringPrecisionUnavailableError) as info:
            f32.Float32Copy.build(bf16, "cpu", 1)
        assert info.value.reason == "model_not_float32_castable"
        assert "extra_state" in info.value.message

    def test_a_flash_attention_model_is_copied_with_sdpa(self, bf16):
        """flash-attention has no float32 kernel; the copy's skeleton must not inherit it."""
        bf16.config._attn_implementation = "flash_attention_2"
        skeleton = f32._clone_skeleton(bf16)
        assert skeleton.config._attn_implementation == "sdpa"
        assert bf16.config._attn_implementation == "flash_attention_2"  # served config untouched

    def test_unknown_reason_is_a_programming_error(self):
        with pytest.raises(ValueError):
            ScoringPrecisionUnavailableError("nope", "x")


class TestChooseDevice:
    def test_auto_prefers_the_card_then_the_cpu(self):
        assert f32.choose_device("auto", "cuda:0", 100, 10, cuda_free=lambda _d: 200,
                                 host_free=lambda: 0).device == "cuda:0"
        assert f32.choose_device("auto", "cuda:0", 100, 10, cuda_free=lambda _d: 50,
                                 host_free=lambda: 500).device == "cpu"
        assert f32.choose_device("auto", "cpu", 100, 10, cuda_free=lambda _d: 10**6,
                                 host_free=lambda: 500).device == "cpu"

    def test_a_pinned_device_is_never_swapped(self):
        with pytest.raises(ScoringPrecisionUnavailableError) as info:
            f32.choose_device("cpu", "cuda:0", 100, 10, cuda_free=lambda _d: 10**6,
                              host_free=lambda: 50)
        assert info.value.reason == "memory"
        with pytest.raises(ScoringPrecisionUnavailableError) as info:
            f32.choose_device("cuda", "cuda:0", 100, 10, cuda_free=lambda _d: 50,
                              host_free=lambda: 10**6)
        assert info.value.reason == "memory"

    def test_headroom_is_part_of_the_fit(self):
        with pytest.raises(ScoringPrecisionUnavailableError):
            f32.choose_device("auto", "cpu", 100, 10, host_free=lambda: 109)
        assert f32.choose_device("auto", "cpu", 100, 10, host_free=lambda: 110).device == "cpu"

    def test_unknown_host_memory_refuses(self):
        with pytest.raises(ScoringPrecisionUnavailableError):
            f32.choose_device("auto", "cpu", 1, 0, host_free=lambda: 0)

    def test_the_cgroup_limit_bounds_host_memory(self, monkeypatch):
        monkeypatch.setattr("millm.ml.memory_utils.get_available_cpu_memory_mb", lambda: 10_000)
        monkeypatch.setattr(f32, "_cgroup_available_mb", lambda: 1_000)
        assert f32.host_available_mb() == 1_000

    def test_the_estimate_counts_four_bytes_per_parameter_and_tied_once(self, bf16):
        expected = sum(t.numel() * 4 for t in {id(t): t for _, t in f32._named_tensors(bf16)
                                               if t.is_floating_point()}.values())
        assert f32.float32_copy_bytes(bf16) == expected


class TestAdapterIsScored:
    async def test_a_served_adapter_is_part_of_the_float32_forward(self, bf16, monkeypatch):
        from millm.ml.adapter_hooks import AdapterTarget, LoraHookSet

        torch.manual_seed(3)
        path = "model.layers.0.self_attn.q_proj"
        target = AdapterTarget(module_path=path, a_key="a", b_key="b", rank=2,
                               in_features=64, out_features=64)
        tensors = {"a": torch.randn(2, 64) * 0.5, "b": torch.randn(64, 2) * 0.5}
        prepared = LoraHookSet.build(bf16, [target], tensors, 2.0)
        handles = LoraHookSet.install(bf16, prepared)
        reference = copy.deepcopy(bf16).float()  # deepcopy carries the hook (float32 A, B)
        applied = MagicMock(prepared=prepared)
        monkeypatch.setattr("millm.services.adapter_service.get_adapter_state",
                            lambda: MagicMock(current=applied))
        try:
            response = await _score(_service(bf16), _request())
        finally:
            LoraHookSet.remove(handles)
        with_adapter = _reference(reference)
        assert _max_diff(_scores(response), with_adapter) <= 1e-6
        assert _max_diff(_scores(response), _reference(copy.deepcopy(bf16).float())) > 1e-3


class TestPolicy:
    def _evaluate(self, body: dict, endpoint, engine=None):
        from millm.api.request_policy import Engine, evaluate
        from millm.api.schemas.openai import ChatCompletionRequest, EmbeddingRequest

        cls = {"chat": ChatCompletionRequest, "completions": TextCompletionRequest,
               "embeddings": EmbeddingRequest}[endpoint.value]
        return evaluate(cls(**body), endpoint, engine or Engine.TRANSFORMERS, strict=False)

    def test_completions_scoring_honours_it(self):
        from millm.api.request_policy import Endpoint

        result = self._evaluate(_request().model_dump(exclude_none=True), Endpoint.COMPLETIONS)
        assert "scoring_precision" not in result.unused

    def test_completions_generation_refuses_float32(self):
        from millm.api.request_policy import Endpoint
        from millm.core.errors import FieldNotHonouredError

        with pytest.raises(FieldNotHonouredError) as info:
            self._evaluate({"model": "tiny", "prompt": "w1", "scoring_precision": "float32"},
                           Endpoint.COMPLETIONS)
        assert info.value.details["param"] == "scoring_precision"
        assert "scoring mode" in info.value.message

    def test_chat_and_embeddings_refuse_float32_and_accept_native(self):
        from millm.api.request_policy import Endpoint
        from millm.core.errors import FieldNotHonouredError

        chat = {"model": "tiny", "messages": [{"role": "user", "content": "hi"}],
                "logprobs": True, "max_tokens": 1}
        emb = {"model": "tiny", "input": "hi"}
        for body, endpoint in ((chat, Endpoint.CHAT), (emb, Endpoint.EMBEDDINGS)):
            with pytest.raises(FieldNotHonouredError) as info:
                self._evaluate({**body, "scoring_precision": "float32"}, endpoint)
            assert info.value.details["param"] == "scoring_precision"
            # The table's own reason, not the generic undeclared-field fallback (control M12a).
            assert "served on /v1/completions scoring only" in info.value.message
            result = self._evaluate({**body, "scoring_precision": "native"}, endpoint)
            assert "scoring_precision" not in result.unused

    def test_gguf_scoring_is_refused_before_any_load(self):
        from millm.api.request_policy import Endpoint, Engine
        from millm.core.errors import FieldNotHonouredError

        with pytest.raises(FieldNotHonouredError) as info:
            self._evaluate(_request().model_dump(exclude_none=True), Endpoint.COMPLETIONS,
                           Engine.LLAMACPP)
        assert "scoring_precision" in info.value.details["fields"]

    def test_schema_refuses_other_values_and_activations(self):
        with pytest.raises(ValidationError):
            _request(scoring_precision="float16")
        activations = {"top_k": 3, "positions": "last"}
        # The same activations block is valid on a native scoring request (control) ...
        assert _request(scoring_precision=None, return_sae_activations=activations)
        # ... and refused with float32, for that reason (mutation control M18).
        with pytest.raises(ValidationError, match="cannot be combined with return_sae_activations"):
            _request(return_sae_activations=activations)

    def test_a_float32_batch_row_is_never_packed(self):
        from millm.services.batch.packing import packable

        assert packable("/v1/completions", _request(scoring_precision=None))
        assert not packable("/v1/completions", _request())
        assert not packable("/v1/completions", _request(scoring_precision="native"))


class TestHttp:
    def _client(self, model, **service_kw):
        from millm.api.dependencies import get_inference_service, get_model_service
        from millm.main import create_app

        inference = _service(model, **service_kw)
        row = MagicMock()
        row.id, row.name, row.architecture, row.gguf_files = 1, "tiny", "text-generation", None
        svc = MagicMock()
        svc.find_model_by_name = AsyncMock(return_value=row)
        app = create_app()
        app.dependency_overrides[get_model_service] = lambda: svc
        app.dependency_overrides[get_inference_service] = lambda: inference
        return TestClient(app)

    def test_the_route_scores_in_float32_and_echoes_it(self, bf16):
        response = self._client(bf16).post(
            "/v1/completions", json=_request().model_dump(exclude_none=True))
        assert response.status_code == 200, response.text
        assert response.json()["scoring_precision"] == "float32"
        assert response.headers["X-miLLM-Scoring-Precision"] == "float32"
        top = response.json()["choices"][0]["logprobs"]["top_logprobs"][0]
        assert _max_diff(top, _reference(copy.deepcopy(bf16).float())) <= 1e-6

    def test_the_route_sends_no_echo_when_not_asked(self, bf16):
        body = _request(scoring_precision=None).model_dump(exclude_none=True)
        response = self._client(bf16).post("/v1/completions", json=body)
        assert response.status_code == 200, response.text
        assert "scoring_precision" not in response.json()
        assert "X-miLLM-Scoring-Precision" not in response.headers

    def test_a_refusal_is_a_409_naming_its_reason(self, bf16):
        response = self._client(bf16, quantization="bitsandbytes").post(
            "/v1/completions", json=_request().model_dump(exclude_none=True))
        assert response.status_code == 409, response.text
        error = response.json()["error"]
        assert error["code"] == "scoring_precision_unavailable"
        assert error["type"] == "invalid_request_error"
        assert error["param"] == "scoring_precision"
        assert "model_not_float32_castable" in error["message"]
        assert response.headers["X-miLLM-Scoring-Precision-Reason"] == "model_not_float32_castable"
        assert "X-miLLM-Scoring-Precision" not in response.headers

    def test_float32_on_chat_is_a_400_never_ignored(self, bf16):
        response = self._client(bf16).post("/v1/chat/completions", json={
            "model": "tiny", "messages": [{"role": "user", "content": "w1"}], "max_tokens": 1,
            "logprobs": True, "scoring_precision": "float32"})
        assert response.status_code == 400, response.text
        assert response.json()["error"]["param"] == "scoring_precision"


def test_the_error_map_row():
    from millm.api.routes.openai.errors import ERROR_STATUS_MAP

    assert ERROR_STATUS_MAP["SCORING_PRECISION_UNAVAILABLE"] == (409, "invalid_request_error")


def test_the_device_setting_is_validated():
    from millm.core.config import Settings

    assert Settings(SCORING_FLOAT32_DEVICE="CUDA:1").SCORING_FLOAT32_DEVICE == "cuda:1"
    with pytest.raises(ValidationError):
        Settings(SCORING_FLOAT32_DEVICE="gpu")
