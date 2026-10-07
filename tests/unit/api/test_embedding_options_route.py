"""`/v1/embeddings` refusals through the REAL app, before any auto-load (Feature 30).

Every refusal here is decidable from the request and the model row, so none may cost a load: a
load would evict the resident model and its SAEs (FR-30.1.2, FR-30.2.7, FR-30.3.6, FR-30.3.8).
`load_model_and_wait` is the "before auto-load" spy; `inference.create_embeddings` is the
"reached the engine" spy, asserted for payload and call count.

MUTATION CONTROLS (0xcc/reviews/030_implementation_controls_2026-10-07.md):
  M5  both `dimensions` cells flipped to HONOURED  -> the dimensions refusals
  M6  the policy call moved after the auto-load    -> "not awaited" assertions
  M7  NEUTRAL["pooling"] always true               -> the GGUF `last` refusal
  M9  `param` dropped in the live handler          -> every `param` assertion
  M12 the cap check removed from the validator     -> the over-cap test
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from millm.api.schemas.openai import EmbeddingData, EmbeddingResponse, Usage
from tests.unit.f25_fixtures import make_client, model_row, unloaded_inference

NATIVE_WIDTH = 16


def _resident(name: str) -> MagicMock:
    """An inference stand-in with `name` resident and an embedding engine spy."""
    inference = MagicMock()
    inference.backend_name = "serial"
    info = MagicMock()
    info.name = name
    inference.get_loaded_model_info = lambda: info
    inference.create_embeddings = AsyncMock(return_value=EmbeddingResponse(
        data=[EmbeddingData(index=0, embedding=[0.0] * NATIVE_WIDTH)],
        model=name, usage=Usage(prompt_tokens=1),
    ))
    return inference


def _error(response) -> dict:
    assert response.status_code == 400, response.text
    return response.json()["error"]


class TestDimensionsIsRefusedBeforeLoad:
    """FR-30.1.2, FR-30.1.6, US-3, T-91: no model declares truncated-embedding support."""

    @pytest.mark.parametrize("gguf", [None, ["m.gguf"]], ids=["transformers", "llamacpp"])
    @pytest.mark.parametrize("dimensions", [64, NATIVE_WIDTH, 1],
                             ids=["smaller", "equal-to-width", "one"])
    def test_refused_at_every_value_with_no_load(self, dimensions, gguf):
        client, svc = make_client(unloaded_inference(), model_row(name="embedder", gguf_files=gguf))
        error = _error(client.post(
            "/v1/embeddings", json={"model": "embedder", "input": "hi", "dimensions": dimensions}
        ))
        assert error["code"] == "field_not_honoured"
        assert error["param"] == "dimensions"
        assert "'embedder'" in error["message"], "the refusal names the model (FR-30.1.2)"
        assert "truncated-embedding support" in error["message"]
        assert svc.load_model_and_wait.await_count == 0

    def test_the_resident_model_is_not_evicted_nor_run(self):
        inference = _resident("other-model")
        client, svc = make_client(inference, model_row(name="embedder"))
        _error(client.post(
            "/v1/embeddings", json={"model": "embedder", "input": "hi", "dimensions": 8}
        ))
        assert svc.load_model_and_wait.await_count == 0
        assert inference.create_embeddings.await_count == 0

    def test_dimensions_null_is_not_sent(self):
        inference = _resident("embedder")
        client, _svc = make_client(inference, model_row(name="embedder"))
        response = client.post(
            "/v1/embeddings", json={"model": "embedder", "input": "hi", "dimensions": None}
        )
        assert response.status_code == 200, response.text


class TestPoolingOnAGgufRow:
    """FR-30.2.7, US-5: llama.cpp fixed MEAN pooling at load."""

    @pytest.mark.parametrize("mode", ["last", "cls"])
    def test_non_mean_is_refused_before_load_without_eviction(self, mode):
        inference = _resident("other-model")
        client, svc = make_client(inference, model_row(name="gguf-embedder", gguf_files=["m.gguf"]))
        error = _error(client.post(
            "/v1/embeddings", json={"model": "gguf-embedder", "input": "hi", "pooling": mode}
        ))
        assert error["code"] == "field_not_honoured" and error["param"] == "pooling"
        assert "mean pooling" in error["message"]
        assert svc.load_model_and_wait.await_count == 0
        assert inference.create_embeddings.await_count == 0

    def test_mean_reaches_the_engine_once_with_the_request(self):
        inference = _resident("gguf-embedder")
        client, svc = make_client(inference, model_row(name="gguf-embedder", gguf_files=["m.gguf"]))
        response = client.post(
            "/v1/embeddings",
            json={"model": "gguf-embedder", "input": "hi", "pooling": "mean", "normalize": True},
        )
        assert response.status_code == 200, response.text
        assert "X-miLLM-Ignored-Fields" not in response.headers, "both fields are consumed"
        assert inference.create_embeddings.await_count == 1
        sent = inference.create_embeddings.await_args.args[0]
        assert (sent.pooling, sent.normalize, sent.input) == ("mean", True, "hi")

    @pytest.mark.parametrize("mode", ["mean", "last", "cls"])
    def test_every_mode_reaches_a_transformers_engine(self, mode):
        inference = _resident("embedder")
        client, _svc = make_client(inference, model_row(name="embedder"))
        response = client.post(
            "/v1/embeddings", json={"model": "embedder", "input": ["a", "b"], "pooling": mode}
        )
        assert response.status_code == 200, response.text
        assert inference.create_embeddings.await_count == 1
        assert inference.create_embeddings.await_args.args[0].pooling == mode


class TestSchemaRefusals:
    """FR-30.2.1, FR-30.3.6, FR-30.3.8, T-94: refused by validation, before any load."""

    def _post(self, body: dict):
        client, svc = make_client(unloaded_inference(), model_row(name="embedder"))
        return client.post("/v1/embeddings", json={"model": "embedder", **body}), svc

    def test_unknown_pooling_names_pooling(self):
        response, svc = self._post({"input": "hi", "pooling": "max"})
        error = _error(response)
        assert error["param"] == "pooling" and error["code"] == "invalid_parameter"
        assert svc.load_model_and_wait.await_count == 0

    @pytest.mark.parametrize("value, needle", [
        ("", "input must not be empty"),
        ([], "at least one string"),
        (["a", ""], "input[1] must not be empty"),
        (["a", "", "b", ""], "input[1] must not be empty (and 1 more empty)"),
    ], ids=["empty-string", "empty-list", "empty-element", "two-empty-elements"])
    def test_empty_input_is_refused(self, value, needle):
        response, svc = self._post({"input": value})
        error = _error(response)
        assert error["param"] == "input" and error["code"] == "invalid_parameter"
        assert needle in error["message"]
        assert svc.load_model_and_wait.await_count == 0

    def test_over_the_cap_names_the_count_and_the_cap(self):
        from millm.core.config import settings

        with patch.object(settings, "EMBEDDINGS_MAX_INPUTS", 3):
            response, svc = self._post({"input": ["a", "b", "c", "d"]})
            at_cap, at_cap_svc = self._post({"input": ["a", "b", "c"]})
        error = _error(response)
        assert error["param"] == "input"
        assert "input has 4 items; the limit is 3 (EMBEDDINGS_MAX_INPUTS)" in error["message"]
        assert svc.load_model_and_wait.await_count == 0
        # At the cap the request passes validation and goes on to the auto-load (nothing is
        # resident in this stand-in, so it then answers 503 model_not_loaded).
        assert at_cap_svc.load_model_and_wait.await_count == 1, at_cap.text

    def test_the_default_cap_is_256(self):
        from millm.core.config import Settings

        assert Settings.model_fields["EMBEDDINGS_MAX_INPUTS"].default == 256
        response, _ = self._post({"input": ["x"] * 257})
        assert "input has 257 items; the limit is 256" in _error(response)["message"]


class TestAnOverLimitInputIsA400NamingItsIndex:
    """FR-30.3.4 through HTTP: the live handler forwards `param` (M9)."""

    def test_service_refusal_reaches_the_client_with_param(self):
        from millm.core.errors import EmbeddingInputTooLongError

        inference = _resident("embedder")
        inference.create_embeddings = AsyncMock(side_effect=EmbeddingInputTooLongError(
            "Input 2 has 99 tokens; this model's limit is 8 tokens.",
            details={"param": "input[2]", "max_context_tokens": 8,
                     "over_limit": [{"index": 2, "tokens": 99}], "omitted": 0},
        ))
        client, _svc = make_client(inference, model_row(name="embedder"))
        error = _error(client.post(
            "/v1/embeddings", json={"model": "embedder", "input": ["a", "b", "c"]}
        ))
        assert error["code"] == "context_length_exceeded"
        assert error["type"] == "invalid_request_error"
        assert error["param"] == "input[2]"
        assert "Input 2 has 99 tokens" in error["message"]

    def test_a_vector_failure_is_a_500_naming_its_index(self):
        from millm.core.errors import EmbeddingVectorInvalidError

        inference = _resident("embedder")
        inference.create_embeddings = AsyncMock(side_effect=EmbeddingVectorInvalidError(
            "Input 0: pooled vector has zero or non-finite norm", details={"param": "input"},
        ))
        client, _svc = make_client(inference, model_row(name="embedder"))
        response = client.post("/v1/embeddings", json={"model": "embedder", "input": "x"})
        assert response.status_code == 500
        error = response.json()["error"]
        assert (error["code"], error["type"], error["param"]) == (
            "embedding_vector_invalid", "server_error", "input"
        )

    def test_an_error_without_a_param_still_answers_null(self):
        from millm.core.errors import EngineUnsupportedError

        inference = _resident("embedder")
        inference.create_embeddings = AsyncMock(side_effect=EngineUnsupportedError("no"))
        client, _svc = make_client(inference, model_row(name="embedder"))
        error = _error(client.post("/v1/embeddings", json={"model": "embedder", "input": "x"}))
        assert error["param"] is None


def test_the_route_is_served_by_the_live_app():
    """Reachability: the path is in the live app's OpenAPI document and its body schema
    declares the two new fields."""
    from millm.main import create_app

    doc = create_app().openapi()
    assert "post" in doc["paths"]["/v1/embeddings"]
    schema = doc["components"]["schemas"]["EmbeddingRequest"]["properties"]
    assert schema["pooling"]["enum"] == ["mean", "last", "cls"]
    assert schema["normalize"]["default"] is False
