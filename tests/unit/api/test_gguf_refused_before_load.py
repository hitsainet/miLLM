"""/v1/completions and /v1/embeddings must refuse a GGUF model BEFORE loading it.

The inference service refuses text completion on the llama.cpp engine, but a
refusal raised there arrives too late: the route auto-loads whatever model the
request names first, so naming a GGUF model would unload the resident
transformers model — and any SAEs attached to it — spend minutes and tens of
gigabytes bringing up llama.cpp, and then return a 400 that could never have
succeeded.

`gguf_files` on the row is set at DOWNLOAD time, so the answer is knowable with
nothing resident. chat.py already refuses streaming on exactly this signal.

Lives under tests/unit/ deliberately. `.github/workflows/backend-tests.yml`
runs `tests/unit/` ONLY — it skips tests/integration/ on the stated grounds
that those need "a real database and full stack", which is not true of the
route tests (their conftest mocks the session factory). A negative control in a
directory CI never collects is not a gate: the guard could be deleted and CI
would stay green. Nothing here needs a database — the model and inference
services are dependency_overrides, and TestClient runs no lifespan outside a
`with` block.

MUTATION CONTROLS:
  * delete the `gguf_files` guard in completions.py -> `load_model_and_wait`
    is called and the completions tests fail
  * delete the `gguf_files` guard in embeddings.py  -> same, for embeddings
  * move the `request.stream` check back below the auto-load in completions.py
    -> the stream test sees load_model_and_wait called
"""

from unittest.mock import AsyncMock, MagicMock

from fastapi.testclient import TestClient

from millm.main import create_app


def _client(model):
    from millm.api.dependencies import get_inference_service, get_model_service

    svc = MagicMock()
    svc.find_model_by_name = AsyncMock(return_value=model)
    svc.get_locked_model = AsyncMock(return_value=None)
    svc.load_model_and_wait = AsyncMock()
    inference = MagicMock()
    inference.backend_name = "serial"
    inference.get_loaded_model_info = lambda: None
    inference.create_text_completion = AsyncMock()
    inference.create_embeddings = AsyncMock()

    app = create_app()
    app.dependency_overrides[get_model_service] = lambda: svc
    app.dependency_overrides[get_inference_service] = lambda: inference
    return TestClient(app), svc, inference


def _model(gguf_files):
    m = MagicMock()
    m.id, m.name, m.architecture = 1, "qwen2.5-7b-gguf", "text-generation"
    m.gguf_files = gguf_files
    return m


BODY = {"model": "qwen2.5-7b-gguf", "prompt": "hello"}


class TestAGGUFModelIsRefusedWithoutBeingLoaded:
    def test_refused_with_a_client_error(self):
        client, _svc, _inf = _client(
            _model(["qwen2.5-7b-instruct-q4_k_m.gguf"])
        )
        response = client.post("/v1/completions", json=BODY)

        assert response.status_code == 400
        body = response.json()["error"]
        assert body["type"] == "invalid_request_error", (
            "server_error tells the caller to retry something that can never "
            "succeed on this model"
        )
        assert body["code"] == "engine_unsupported"

    def test_nothing_is_loaded_and_nothing_generates(self):
        """The point of the guard: no VRAM spent, no resident model evicted."""
        client, svc, inference = _client(
            _model(["qwen2.5-7b-instruct-q4_k_m.gguf"])
        )
        client.post("/v1/completions", json=BODY)

        svc.load_model_and_wait.assert_not_called()
        inference.create_text_completion.assert_not_called()

    def test_a_transformers_model_still_completes(self):
        """The guard must bite on GGUF rows ONLY."""
        client, svc, inference = _client(_model(None))
        client.post("/v1/completions", json=BODY)

        svc.load_model_and_wait.assert_called_once()


class TestEmbeddingsRefuseAGGUFModelWithoutLoadingIt:
    """`create_embeddings` already refuses the llama.cpp engine — but only
    after the route has auto-loaded it. llama.cpp pools internally and exposes
    no hidden states to mean-pool, so the answer was never reachable, and
    `gguf_files` on the row says so at DOWNLOAD time. Refusing post-load
    evicts a working transformers model and any SAEs attached to it, and spends
    minutes and tens of gigabytes, to reach a guaranteed 400.
    """

    BODY = {"model": "qwen2.5-7b-gguf", "input": "hello"}

    def test_refused_with_a_client_error(self):
        client, _svc, _inf = _client(_model(["qwen2.5-7b-instruct-q4_k_m.gguf"]))
        response = client.post("/v1/embeddings", json=self.BODY)

        assert response.status_code == 400
        body = response.json()["error"]
        assert body["type"] == "invalid_request_error"
        assert body["code"] == "engine_unsupported"

    def test_nothing_is_loaded_and_nothing_embeds(self):
        client, svc, inference = _client(_model(["qwen2.5-7b-instruct-q4_k_m.gguf"]))
        client.post("/v1/embeddings", json=self.BODY)

        svc.load_model_and_wait.assert_not_called()
        inference.create_embeddings.assert_not_called()

    def test_a_transformers_model_still_embeds(self):
        """The guard must bite on GGUF rows ONLY."""
        client, svc, _inf = _client(_model(None))
        client.post("/v1/embeddings", json=self.BODY)

        svc.load_model_and_wait.assert_called_once()


class TestStreamingCompletionsAreRefusedWithoutLoading:
    """Streaming has never been implemented on /v1/completions on ANY engine,
    so the answer depends on `request.stream` alone. Checked after the
    auto-load, it still cost a full model swap — evicting whatever was resident
    — to return a 400 the request body had already decided.
    """

    def test_nothing_is_loaded_for_a_streaming_request(self):
        client, svc, inference = _client(_model(None))
        response = client.post("/v1/completions", json={**BODY, "stream": True})

        assert response.status_code == 400
        svc.load_model_and_wait.assert_not_called()
        inference.create_text_completion.assert_not_called()
