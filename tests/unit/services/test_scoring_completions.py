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
