"""Scoring-mode text completions: the next token's log-probabilities (2026-10-04).

A typed-decision judge (autotrust/JEV-9B, jevify) reads its answer from the probabilities of a few
answer tokens at the next position; nothing is generated. These tests run a TINY REAL Llama and a
real tokenizer, never a mocked model: a mock returns whatever the test told it to, which is the
"fixture agrees with the code by construction" trap this estate keeps recording.
"""

from __future__ import annotations

from datetime import datetime
from unittest.mock import patch

import pytest
import torch
from pydantic import ValidationError
from tokenizers import Tokenizer, models, pre_tokenizers, processors
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

from millm.api.schemas.openai import TextCompletionRequest
from millm.core.errors import EngineUnsupportedError, InvalidScoringRequestError
from millm.ml.model_loader import LoadedModel, LoadedModelState
from millm.services.inference_service import InferenceService
from millm.services.next_token_scores import next_token_scores

WORDS = ["<s>", "[UNK]"] + [f"w{i}" for i in range(30)]
PROMPT = "w3 w7 w1 w9 w2"


def _tokenizer(bos: bool) -> PreTrainedTokenizerFast:
    tok = Tokenizer(models.WordLevel({w: i for i, w in enumerate(WORDS)}, unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    if bos:
        tok.post_processor = processors.TemplateProcessing(single="<s> $A", special_tokens=[("<s>", 0)])
    fast = PreTrainedTokenizerFast(tokenizer_object=tok, bos_token="<s>", unk_token="[UNK]")
    fast.pad_token = "[UNK]"
    return fast


@pytest.fixture(autouse=True)
def _clean_state():
    state = LoadedModelState()
    state._loaded = None
    yield
    state._loaded = None


@pytest.fixture
def model() -> LlamaForCausalLM:
    torch.manual_seed(0)
    config = LlamaConfig(vocab_size=len(WORDS), hidden_size=16, intermediate_size=32,
                         num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=2,
                         max_position_embeddings=128)
    return LlamaForCausalLM(config).eval()


def _service(model, tokenizer) -> InferenceService:
    LoadedModelState().set(LoadedModel(
        model_id=1, model_name="tiny", model=model, tokenizer=tokenizer,
        loaded_at=datetime(2026, 10, 4), memory_used_mb=1, num_parameters=1, device="cpu",
        dtype="float32",
    ))
    with patch("millm.services.inference_service.torch.cuda.is_available", return_value=False):
        svc = InferenceService(model_service=None)
    svc._device = "cpu"
    return svc


def _request(**over) -> TextCompletionRequest:
    body = {"model": "tiny", "prompt": PROMPT, "max_tokens": 1, "temperature": 1.0,
            "logprobs": 2, "allowed_token_ids": [5, 9], "add_special_tokens": False,
            "return_tokens_as_token_ids": True}
    body.update(over)
    return TextCompletionRequest(**body)


def _logits(model, tokenizer, prompt: str, add_special_tokens: bool) -> torch.Tensor:
    ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=add_special_tokens)
    with torch.no_grad():
        return model(**ids).logits[0, -1].float()


class TestTheScoresAreTheModelsOwn:
    async def test_restricted_logprobs_equal_log_softmax_over_the_allowed_ids(self, model):
        tok = _tokenizer(bos=False)
        response = await _service(model, tok).create_text_completion(_request())
        expected = torch.log_softmax(_logits(model, tok, PROMPT, False)[[5, 9]], 0)
        top = response.choices[0].logprobs.top_logprobs[0]
        assert set(top) == {"token_id:5", "token_id:9"}
        assert top["token_id:5"] == pytest.approx(float(expected[0]), abs=1e-5)
        assert top["token_id:9"] == pytest.approx(float(expected[1]), abs=1e-5)
        chosen = "token_id:5" if expected[0] >= expected[1] else "token_id:9"
        assert response.choices[0].logprobs.tokens == [chosen]
        assert response.usage.completion_tokens == 1

    async def test_temperature_scales_the_logits(self, model):
        tok = _tokenizer(bos=False)
        response = await _service(model, tok).create_text_completion(_request(temperature=2.0))
        expected = torch.log_softmax(_logits(model, tok, PROMPT, False)[[5, 9]] / 2.0, 0)
        assert response.choices[0].logprobs.top_logprobs[0]["token_id:5"] == pytest.approx(
            float(expected[0]), abs=1e-5)

    async def test_without_a_restriction_it_is_the_full_vocabulary_top_k(self, model):
        tok = _tokenizer(bos=False)
        response = await _service(model, tok).create_text_completion(
            _request(allowed_token_ids=None, logprobs=3))
        full = torch.log_softmax(_logits(model, tok, PROMPT, False), 0)
        values, ids = torch.topk(full, 3)
        top = response.choices[0].logprobs.top_logprobs[0]
        assert list(top) == [f"token_id:{int(i)}" for i in ids]
        assert list(top.values()) == pytest.approx([float(v) for v in values], abs=1e-5)

    async def test_add_special_tokens_is_honoured(self, model):
        """A template-exact judge prompt must not gain a BOS the template does not have."""
        tok = _tokenizer(bos=True)
        svc = _service(model, tok)
        without = await svc.create_text_completion(_request(add_special_tokens=False))
        with_bos = await svc.create_text_completion(_request(add_special_tokens=True))
        assert with_bos.usage.prompt_tokens == without.usage.prompt_tokens + 1
        expected = torch.log_softmax(_logits(model, tok, PROMPT, True)[[5, 9]], 0)
        assert with_bos.choices[0].logprobs.top_logprobs[0]["token_id:5"] == pytest.approx(
            float(expected[0]), abs=1e-5)

    async def test_tokens_are_keyed_by_text_unless_ids_are_asked_for(self, model):
        tok = _tokenizer(bos=False)
        response = await _service(model, tok).create_text_completion(
            _request(return_tokens_as_token_ids=False))
        assert set(response.choices[0].logprobs.top_logprobs[0]) == {"w3", "w7"}  # ids 5 and 9


class TestItIsRoutedBeforeEveryPathThatCannotScore:
    async def test_it_never_generates(self, model):
        svc = _service(model, _tokenizer(bos=False))

        def no_generate(*a, **k):
            raise AssertionError("a scoring request reached generation")

        svc._generate_sync = no_generate
        response = await svc.create_text_completion(_request())
        assert response.choices[0].logprobs is not None

    async def test_continuous_batching_does_not_take_it(self, model):
        svc = _service(model, _tokenizer(bos=False))
        svc._use_cbm_for_request = lambda **k: True

        async def cbm(*a, **k):
            raise AssertionError("a scoring request went to continuous batching")

        svc._cbm_text_completion = cbm
        assert (await svc.create_text_completion(_request())).choices[0].logprobs is not None

    async def test_llamacpp_refuses_rather_than_dropping_the_scores(self, model):
        svc = _service(model, _tokenizer(bos=False))
        svc._engine_is_llamacpp = lambda: True
        with pytest.raises(EngineUnsupportedError):
            await svc.create_text_completion(_request())

    async def test_a_plain_completion_still_generates(self, model):
        """Scoring mode is opt-in: a request without its fields takes the old path."""
        svc = _service(model, _tokenizer(bos=False))
        called = []
        svc._score_text_completion = lambda r: called.append(r)
        svc._generate_sync = lambda kwargs: torch.cat(
            [kwargs["input_ids"], torch.tensor([[7]])], dim=-1)
        response = await svc.create_text_completion(
            TextCompletionRequest(model="tiny", prompt=PROMPT, max_tokens=1))
        assert called == [] and response.choices[0].logprobs is None


class TestRefusals:
    async def test_an_id_outside_the_vocabulary_is_a_400_not_a_500(self, model):
        svc = _service(model, _tokenizer(bos=False))
        with pytest.raises(InvalidScoringRequestError) as exc:
            await svc.create_text_completion(_request(allowed_token_ids=[5, 999]))
        assert exc.value.status_code == 400

    @pytest.mark.parametrize("over, message", [
        ({"max_tokens": None}, "max_tokens=1"),
        ({"max_tokens": 5}, "max_tokens=1"),
        ({"n": 2}, "n=1"),
        ({"allowed_token_ids": [-1]}, "non-negative"),
        ({"logprobs": 21}, "less than or equal to 20"),
    ])
    def test_the_request_is_refused_before_anything_runs(self, over, message):
        with pytest.raises(ValidationError, match=message):
            _request(**over)

    def test_the_vendor_clients_payload_parses(self):
        """autotrust/JEV-9B's documented client, field for field."""
        body = {"model": "jev-decision", "prompt": "[kind] noul\n[decision]:", "max_tokens": 1,
                "temperature": 1.0, "logprobs": 2, "allowed_token_ids": [3721, 1802],
                "add_special_tokens": False, "return_tokens_as_token_ids": True}
        req = TextCompletionRequest(**body)
        assert req.wants_scores() and req.allowed_token_ids == [3721, 1802]
        assert req.add_special_tokens is False and req.return_tokens_as_token_ids is True


class TestNextTokenScores:
    def test_the_set_is_renormalised_and_deduplicated(self):
        logits = torch.tensor([0.0, 1.0, 2.0, 3.0])
        scores = next_token_scores(logits, allowed=[1, 3, 1], temperature=1.0, top_k=5)
        assert [t for t, _ in scores.top] == [3, 1]
        assert sum(torch.tensor([lp for _, lp in scores.top]).exp().tolist()) == pytest.approx(1.0)
        assert scores.chosen_id == 3

    def test_temperature_zero_does_not_scale(self):
        logits = torch.tensor([0.0, 2.0])
        greedy = next_token_scores(logits, allowed=None, temperature=0.0, top_k=2)
        one = next_token_scores(logits, allowed=None, temperature=1.0, top_k=2)
        assert greedy.top == one.top

    def test_top_k_zero_still_returns_the_chosen_token(self):
        scores = next_token_scores(torch.tensor([0.0, 5.0, 1.0]), allowed=None, temperature=1.0, top_k=0)
        assert scores.top == [(1, scores.chosen_logprob)]

    @pytest.mark.parametrize("allowed", [[4], [-1]])
    def test_ids_outside_the_vocabulary_are_refused(self, allowed):
        with pytest.raises(ValueError, match="outside the vocabulary"):
            next_token_scores(torch.zeros(4), allowed=allowed, temperature=1.0, top_k=1)

    def test_one_position_only(self):
        with pytest.raises(ValueError, match="one position"):
            next_token_scores(torch.zeros(2, 4), allowed=None, temperature=1.0, top_k=1)


class TestTheHttpRouteCarriesItEndToEnd:
    """The fields survive the real route: the request model ignores unknown fields (so a missing
    field would silently turn a scoring request into a 512-token generation), and FastAPI filters
    the response through `response_model` (so a missing response field would drop the scores)."""

    def test_the_vendor_body_returns_restricted_logprobs(self, model):
        from unittest.mock import AsyncMock, MagicMock

        from fastapi.testclient import TestClient

        from millm.api.dependencies import get_inference_service, get_model_service
        from millm.main import create_app

        tok = _tokenizer(bos=False)
        inference = _service(model, tok)
        row = MagicMock()
        row.id, row.name, row.architecture, row.gguf_files = 1, "tiny", "text-generation", None
        svc = MagicMock()
        svc.find_model_by_name = AsyncMock(return_value=row)
        svc.load_model_and_wait = AsyncMock()
        app = create_app()
        app.dependency_overrides[get_model_service] = lambda: svc
        app.dependency_overrides[get_inference_service] = lambda: inference

        body = {"model": "tiny", "prompt": PROMPT, "max_tokens": 1, "temperature": 1.0,
                "logprobs": 2, "allowed_token_ids": [5, 9], "add_special_tokens": False,
                "return_tokens_as_token_ids": True}
        response = TestClient(app).post("/v1/completions", json=body)
        assert response.status_code == 200, response.text
        top = response.json()["choices"][0]["logprobs"]["top_logprobs"][0]
        expected = torch.log_softmax(_logits(model, tok, PROMPT, False)[[5, 9]], 0)
        assert top["token_id:5"] == pytest.approx(float(expected[0]), abs=1e-5)
        svc.load_model_and_wait.assert_not_called()

    def test_an_out_of_vocabulary_id_is_a_400_over_http(self, model):
        from unittest.mock import AsyncMock, MagicMock

        from fastapi.testclient import TestClient

        from millm.api.dependencies import get_inference_service, get_model_service
        from millm.main import create_app

        inference = _service(model, _tokenizer(bos=False))
        row = MagicMock()
        row.id, row.name, row.architecture, row.gguf_files = 1, "tiny", "text-generation", None
        svc = MagicMock()
        svc.find_model_by_name = AsyncMock(return_value=row)
        app = create_app()
        app.dependency_overrides[get_model_service] = lambda: svc
        app.dependency_overrides[get_inference_service] = lambda: inference
        response = TestClient(app).post("/v1/completions", json={
            "model": "tiny", "prompt": PROMPT, "max_tokens": 1, "allowed_token_ids": [999]})
        assert response.status_code == 400, response.text
        assert "INVALID_SCORING_REQUEST" in response.text or "vocabulary" in response.text


class _Spy(torch.nn.Module):
    """Wraps the real model: records each forward's kwargs, optionally fails or corrupts it."""

    def __init__(self, inner, *, fail=None, nan=False, on_forward=None):
        super().__init__()
        self.inner, self.fail, self.nan, self.on_forward, self.calls = inner, fail, nan, on_forward, []
        self.config = inner.config

    def forward(self, **kwargs):
        self.calls.append(kwargs)
        if self.on_forward is not None:
            self.on_forward()
        if self.fail is not None:
            raise self.fail
        out = self.inner(**kwargs)
        if self.nan:
            out.logits[..., 5] = float("nan")
        return out

    def get_input_embeddings(self):
        return self.inner.get_input_embeddings()


class TestReviewRoundOne:
    async def test_only_the_last_positions_logits_are_computed(self, model):
        """H1: a plain forward builds logits for every position (~2 GB at 4k tokens, 248k vocab)."""
        spy = _Spy(model)
        response = await _service(spy, _tokenizer(bos=False)).create_text_completion(_request())
        assert response.choices[0].logprobs is not None
        assert len(spy.calls) == 1 and spy.calls[0]["logits_to_keep"] == 1

    async def test_out_of_memory_is_the_typed_refusal(self, model):
        from millm.core.errors import GenerationOutOfMemoryError

        spy = _Spy(model, fail=torch.cuda.OutOfMemoryError("CUDA out of memory. GPU 0 has a total"))
        with pytest.raises(GenerationOutOfMemoryError):
            await _service(spy, _tokenizer(bos=False)).create_text_completion(_request())

    async def test_non_finite_logits_are_refused_not_serialised(self, model):
        from millm.core.errors import ScoringNumericalError

        spy = _Spy(model, nan=True)
        with pytest.raises(ScoringNumericalError):
            await _service(spy, _tokenizer(bos=False)).create_text_completion(_request())

    async def test_every_attached_sae_is_suppressed_during_the_pass(self, model):
        """M1: a judge's verdict must not be steered. Two SAEs, as a circuit attaches."""
        from contextlib import contextmanager
        from types import SimpleNamespace
        from unittest.mock import MagicMock

        active, seen = set(), []

        def entry(name):
            sae = MagicMock()

            @contextmanager
            def suppressed():
                active.add(name)
                try:
                    yield
                finally:
                    active.discard(name)

            sae.suppressed = suppressed
            return SimpleNamespace(sae=sae)

        spy = _Spy(model, on_forward=lambda: seen.append(set(active)))
        with patch("millm.services.sae_service.AttachedSAEState.entries",
                   return_value=[entry("a"), entry("b")]):
            await _service(spy, _tokenizer(bos=False)).create_text_completion(_request())
        assert seen == [{"a", "b"}]
        assert active == set(), "suppression outlived the pass"

    async def test_no_monitor_is_ever_begun(self, model):
        """Probes, sensing and circuit sensing record a request only once it is begun."""
        svc = _service(model, _tokenizer(bos=False))

        def refuse(*a, **k):
            raise AssertionError("a monitor was begun for a scoring request")

        svc._probe_begin = svc._sensing_begin = svc._circuit_sensing_begin = refuse
        assert (await svc.create_text_completion(_request())).choices[0].logprobs is not None

    async def test_an_empty_prompt_is_a_400(self, model):
        with pytest.raises(InvalidScoringRequestError, match="tokenises to nothing"):
            await _service(model, _tokenizer(bos=False)).create_text_completion(_request(prompt=""))

    def test_an_empty_prompt_list_is_refused(self):
        with pytest.raises(ValidationError, match="at least one prompt"):
            _request(prompt=[])

    async def test_allowed_ids_without_logprobs_return_no_logprobs_object(self, model):
        """vLLM returns `logprobs: null` when only the token was constrained."""
        response = await _service(model, _tokenizer(bos=False)).create_text_completion(
            _request(logprobs=None))
        choice = response.choices[0]
        assert choice.logprobs is None and choice.text in {"w3", "w7"}

    def test_a_gguf_model_is_refused_before_it_is_loaded(self, model):
        """M2: loading would evict the resident model and its SAEs to reach the same 400."""
        from unittest.mock import AsyncMock, MagicMock

        from fastapi.testclient import TestClient

        from millm.api.dependencies import get_inference_service, get_model_service
        from millm.main import create_app

        row = MagicMock()
        row.id, row.name, row.architecture, row.gguf_files = 2, "q-gguf", "text-generation", ["m.gguf"]
        svc = MagicMock()
        svc.find_model_by_name = AsyncMock(return_value=row)
        svc.load_model_and_wait = AsyncMock()
        inference = MagicMock()
        inference.get_loaded_model_info = lambda: None
        app = create_app()
        app.dependency_overrides[get_model_service] = lambda: svc
        app.dependency_overrides[get_inference_service] = lambda: inference
        response = TestClient(app).post("/v1/completions", json={
            "model": "q-gguf", "prompt": "x", "max_tokens": 1, "logprobs": 2})
        assert response.status_code == 400 and "GGUF" in response.text
        svc.load_model_and_wait.assert_not_called()


class TestReviewRoundTwo:
    def test_suppression_reaches_only_the_thread_that_asked(self):
        """M-A: a scoring pass in a worker thread must not unsteer a generation in another thread.
        A REAL LoadedSAE with steering on: suppressed in thread A, still steering in thread B."""
        import threading

        from millm.ml.sae_config import SAEConfig
        from millm.ml.sae_wrapper import LoadedSAE

        torch.manual_seed(1)
        sae = LoadedSAE(W_enc=torch.randn(8, 16), b_enc=torch.zeros(16), W_dec=torch.randn(16, 8),
                        b_dec=torch.zeros(8), config=SAEConfig(d_in=8, d_sae=16, model_name="t",
                                                               hook_name="t", hook_layer=0),
                        device="cpu")
        sae.set_steering(3, 5.0)
        sae.enable_steering(True)
        hidden = torch.zeros(1, 2, 8)
        steered = sae.apply_steering(hidden.clone())
        assert not torch.equal(steered, hidden), "precondition: steering changes the hidden states"

        inside, release, seen = threading.Event(), threading.Event(), {}

        def scoring_thread():
            with sae.suppressed():
                seen["a"] = sae.apply_steering(hidden.clone())
                inside.set()
                release.wait(5)

        worker = threading.Thread(target=scoring_thread)
        worker.start()
        assert inside.wait(5)
        seen["b"] = sae.apply_steering(hidden.clone())   # this thread, while A is suppressed
        release.set()
        worker.join(5)
        assert torch.equal(seen["a"], hidden), "the suppressed thread was steered"
        assert torch.equal(seen["b"], steered), "another thread's generation was unsteered"
        assert torch.equal(sae.apply_steering(hidden.clone()), steered)

    async def test_suppression_is_entered_in_the_worker_thread(self, model):
        """The context must wrap the forward IN the thread that runs it, or per-thread suppression
        suppresses nothing."""
        import threading
        from contextlib import contextmanager
        from types import SimpleNamespace
        from unittest.mock import MagicMock

        entered_in, forward_in = [], []
        sae = MagicMock()

        @contextmanager
        def suppressed():
            entered_in.append(threading.get_ident())
            yield

        sae.suppressed = suppressed
        spy = _Spy(model, on_forward=lambda: forward_in.append(threading.get_ident()))
        with patch("millm.services.sae_service.AttachedSAEState.entries",
                   return_value=[SimpleNamespace(sae=sae)]):
            await _service(spy, _tokenizer(bos=False)).create_text_completion(_request())
        assert entered_in == forward_in and len(forward_in) == 1

    async def test_a_model_without_logits_to_keep_falls_back(self, model):
        class Strict(torch.nn.Module):
            def __init__(self, inner):
                super().__init__()
                self.inner, self.config, self.calls = inner, inner.config, 0

            def forward(self, input_ids=None, attention_mask=None, use_cache=None):
                self.calls += 1
                return self.inner(input_ids=input_ids, attention_mask=attention_mask, use_cache=use_cache)

            def get_input_embeddings(self):
                return self.inner.get_input_embeddings()

        tok = _tokenizer(bos=False)
        strict = Strict(model)
        response = await _service(strict, tok).create_text_completion(_request())
        expected = torch.log_softmax(_logits(model, tok, PROMPT, False)[[5, 9]], 0)
        assert response.choices[0].logprobs.top_logprobs[0]["token_id:5"] == pytest.approx(
            float(expected[0]), abs=1e-5)
        # The first attempt fails at the call boundary (unexpected keyword), before forward's body
        # runs, so exactly one call reached the model: the retry without logits_to_keep.
        assert strict.calls == 1

    async def test_an_unrelated_type_error_is_not_swallowed(self, model):
        spy = _Spy(model, fail=TypeError("bad thing"))
        with pytest.raises(TypeError, match="bad thing"):
            await _service(spy, _tokenizer(bos=False)).create_text_completion(_request())
        assert len(spy.calls) == 1

    async def test_out_of_memory_releases_memory_and_says_scoring(self, model):
        from contextlib import contextmanager
        from types import SimpleNamespace
        from unittest.mock import MagicMock

        from millm.core.errors import GenerationOutOfMemoryError

        state = {"active": False}
        sae = MagicMock()

        @contextmanager
        def suppressed():
            state["active"] = True
            try:
                yield
            finally:
                state["active"] = False

        sae.suppressed = suppressed
        released, during = [], []
        spy = _Spy(model, fail=torch.cuda.OutOfMemoryError("CUDA out of memory. GPU 0 has a total"),
                   on_forward=lambda: during.append(state["active"]))
        with patch("millm.services.sae_service.AttachedSAEState.entries",
                   return_value=[SimpleNamespace(sae=sae)]), \
             patch("millm.services.inference_service._release_generation_memory",
                   lambda: released.append(True)):
            with pytest.raises(GenerationOutOfMemoryError) as exc:
                await _service(spy, _tokenizer(bos=False)).create_text_completion(_request())
        assert released == [True]
        assert "Scoring ran out of memory" in str(exc.value) and "max_tokens" not in str(exc.value)
        assert exc.value.details["prompt_tokens"] == 5
        assert during == [True], "the failing pass was not suppressed"
        assert state["active"] is False, "suppression outlived the failed pass"

    async def test_minus_inf_outside_the_allowed_set_is_fine(self, model):
        """L2: a head that masks padded vocabulary with -inf must not refuse a valid answer."""
        class MaskPadding(_Spy):
            def forward(self, **kwargs):
                out = self.inner(**kwargs)
                out.logits[..., 20] = float("-inf")
                return out

        response = await _service(MaskPadding(model), _tokenizer(bos=False)).create_text_completion(_request())
        assert response.choices[0].logprobs is not None

    async def test_minus_inf_on_a_requested_token_is_refused(self, model):
        from millm.core.errors import ScoringNumericalError

        class MaskAnswer(_Spy):
            def forward(self, **kwargs):
                out = self.inner(**kwargs)
                out.logits[..., 9] = float("-inf")
                return out

        with pytest.raises(ScoringNumericalError):
            await _service(MaskAnswer(model), _tokenizer(bos=False)).create_text_completion(_request())


class TestReviewRoundThree:
    @staticmethod
    def _poison(model, index, value):
        class Poison(_Spy):
            def forward(self, **kwargs):
                out = self.inner(**kwargs)
                out.logits[..., index] = value
                return out
        return Poison(model)

    @pytest.mark.parametrize("value", [float("nan"), float("inf")])
    async def test_poison_outside_the_allowed_set_does_not_refuse_a_restricted_answer(self, model, value):
        response = await _service(self._poison(model, 20, value), _tokenizer(bos=False)).create_text_completion(
            _request())
        assert response.choices[0].logprobs is not None

    @pytest.mark.parametrize("value", [float("nan"), float("inf")])
    async def test_poison_anywhere_refuses_an_unrestricted_answer(self, model, value):
        from millm.core.errors import ScoringNumericalError

        with pytest.raises(ScoringNumericalError, match="NaN or infinite"):
            await _service(self._poison(model, 20, value), _tokenizer(bos=False)).create_text_completion(
                _request(allowed_token_ids=None, logprobs=3))

    @pytest.mark.parametrize("value", [float("nan"), float("inf")])
    async def test_poison_on_an_allowed_token_is_refused_up_front(self, model, value):
        from millm.core.errors import ScoringNumericalError

        with pytest.raises(ScoringNumericalError, match="NaN or infinite"):
            await _service(self._poison(model, 9, value), _tokenizer(bos=False)).create_text_completion(
                _request())

    def test_a_vanishing_temperature_is_refused(self):
        with pytest.raises(ValidationError, match="temperature 0 or at least"):
            _request(temperature=1e-38)
        assert _request(temperature=0.0).temperature == 0.0
