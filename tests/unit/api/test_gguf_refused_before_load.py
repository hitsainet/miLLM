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


class TestAGGUFTextCompletionIsServed:
    """/v1/completions no longer turns a GGUF model away at the door.

    It used to refuse pre-load on `gguf_files`. That guard was correct while the
    service had no implementation — spending minutes and tens of GB to reach a
    guaranteed 400 is worse than refusing early — but the implementation exists
    now, so the guard was the only thing standing between a GGUF model and a
    capability Ollama has always had.

    MUTATION CONTROL: reinstate the `gguf_files` refusal in completions.py ->
    both tests fail.
    """

    @staticmethod
    def _real_response():
        """A REAL response object: FastAPI validates against response_model,
        and an AsyncMock's attributes fail that with three type errors."""
        from millm.api.schemas.openai import (
            TextCompletionChoice,
            TextCompletionResponse,
            Usage,
        )

        return TextCompletionResponse(
            id="cmpl-" + "0" * 24,
            created=0,
            model="qwen2.5-7b-gguf",
            choices=[TextCompletionChoice(index=0, text=" hi", finish_reason="stop")],
            usage=Usage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
        )

    def test_a_text_completion_request_reaches_the_engine(self):
        client, svc, inference = _client(_model(["m-q4_k_m.gguf"]))
        loaded = MagicMock()
        loaded.name = "qwen2.5-7b-gguf"
        inference.get_loaded_model_info = lambda: loaded
        inference.create_text_completion = AsyncMock(return_value=self._real_response())

        response = client.post("/v1/completions", json=BODY)

        assert response.status_code == 200, response.text
        assert inference.create_text_completion.called

    def test_the_model_is_loaded_for_it(self):
        client, svc, inference = _client(_model(["m-q4_k_m.gguf"]))
        other, wanted = MagicMock(), MagicMock()
        other.name, wanted.name = "some-other-model", "qwen2.5-7b-gguf"
        reports = iter([other, wanted, wanted, wanted])
        inference.get_loaded_model_info = lambda: next(reports, wanted)
        inference.create_text_completion = AsyncMock(return_value=self._real_response())

        client.post("/v1/completions", json=BODY)

        assert svc.load_model_and_wait.called


class TestGGUFEmbeddingsAreServed:
    """A GGUF model can embed, so /v1/embeddings no longer turns it away.

    The original refusal reasoned that llama.cpp needs embedding=True at
    CONSTRUCTION and pools internally, so parity would mean quietly returning
    differently-computed vectors. Measuring on the RTX 3090 refuted both halves:
    one instance loaded with embedding=True and MEAN pooling served BOTH
    embeddings and chat, and MEAN pooling is the same strategy the transformers
    path uses (hidden_states[-1].mean(dim=1)).

    MUTATION CONTROL: reinstate the `gguf_files` refusal in embeddings.py ->
    both tests fail.
    """

    @staticmethod
    def _real_response():
        from millm.api.schemas.openai import EmbeddingData, EmbeddingResponse, Usage

        return EmbeddingResponse(
            data=[EmbeddingData(index=0, embedding=[0.1, 0.2, 0.3])],
            model="qwen2.5-7b-gguf",
            usage=Usage(prompt_tokens=3, completion_tokens=0, total_tokens=3),
        )

    def test_an_embedding_request_reaches_the_engine(self):
        client, _svc, inference = _client(_model(["m-q4_k_m.gguf"]))
        loaded = MagicMock()
        loaded.name = "qwen2.5-7b-gguf"
        inference.get_loaded_model_info = lambda: loaded
        inference.create_embeddings = AsyncMock(return_value=self._real_response())

        response = client.post(
            "/v1/embeddings", json={"model": "qwen2.5-7b-gguf", "input": "hello"}
        )

        assert response.status_code == 200, response.text
        assert inference.create_embeddings.called

    def test_the_model_is_loaded_for_it(self):
        client, svc, inference = _client(_model(["m-q4_k_m.gguf"]))
        other, wanted = MagicMock(), MagicMock()
        other.name, wanted.name = "some-other-model", "qwen2.5-7b-gguf"
        reports = iter([other, wanted, wanted, wanted])
        inference.get_loaded_model_info = lambda: next(reports, wanted)
        inference.create_embeddings = AsyncMock(return_value=self._real_response())

        client.post("/v1/embeddings", json={"model": "qwen2.5-7b-gguf", "input": "hi"})

        assert svc.load_model_and_wait.called


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


class TestStreamingAGGUFModelIsNOTRefused:
    """The counterpart: chat streaming on GGUF must now be ALLOWED through.

    chat.py used to refuse `stream: true` on the `gguf_files` signal. That guard
    was never covered by a test — grep for `engine_unsupported` in tests/ finds
    only completions and embeddings — so deleting it left the suite green. That
    cuts both ways: nothing would have caught a mistake in its replacement
    either, which is why this positive test exists rather than an inverted one.

    MUTATION CONTROL: reinstate the `request.stream and model.gguf_files`
    refusal in chat.py -> both tests here fail.
    """

    def _chat_client(self):
        from millm.api.dependencies import get_inference_service, get_model_service

        model = _model(["zora-v1.13-q5_k_m.gguf"])
        model.name = "zora-v1.13-gguf"

        svc = MagicMock()
        svc.find_model_by_name = AsyncMock(return_value=model)
        svc.get_locked_model = AsyncMock(return_value=None)
        svc.load_model_and_wait = AsyncMock()

        async def _fake_stream(_request):
            yield 'data: {"choices":[{"delta":{"role":"assistant"}}]}\n\n'
            yield "data: [DONE]\n\n"

        inference = MagicMock()
        inference.backend_name = "llamacpp"
        # The route checks a model is resident after the auto-load; None here
        # short-circuits into "no model is currently loaded" and the test would
        # assert nothing about streaming.
        # An OBJECT with `.name`, not a dict — the route compares
        # `model_info.name` against the requested model. Returning None would
        # short-circuit into "no model is currently loaded" and assert nothing
        # about streaming; a dict raises AttributeError inside the route.
        loaded_info = MagicMock()
        loaded_info.name = "zora-v1.13-gguf"
        inference.get_loaded_model_info = lambda: loaded_info
        inference.ensure_profile_exists = AsyncMock(return_value=True)
        # Real numbers: the route compares pending against capacity, and two
        # MagicMocks raise TypeError on `>=`. The queue has to be modelled, not
        # merely present.
        inference.request_queue = MagicMock(pending_count=0, max_pending=10)
        inference.stream_chat_completion = _fake_stream

        app = create_app()
        app.dependency_overrides[get_model_service] = lambda: svc
        app.dependency_overrides[get_inference_service] = lambda: inference
        return TestClient(app), svc

    def test_a_streaming_chat_request_reaches_the_engine(self):
        client, svc = self._chat_client()

        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "zora-v1.13-gguf",
                "messages": [{"role": "user", "content": "hi"}],
                "stream": True,
            },
        )

        assert response.status_code == 200, response.text
        assert "text/event-stream" in response.headers["content-type"]
        assert response.text.rstrip().endswith("data: [DONE]")

    def test_the_model_is_actually_loaded_for_it(self):
        """The refusal used to short-circuit BEFORE the auto-load.

        Modelled with a DIFFERENT model resident, because the route correctly
        skips the load when the requested one is already there — asserting the
        call with a matching name would have failed for the right reason and
        told us nothing about the refusal.

        `get_loaded_model_info` returns the other model first and the requested
        one afterwards, which is what a real successful load looks like: the
        route re-reads it to "confirm the switch actually happened rather than
        assuming it did".
        """
        client, svc = self._chat_client()
        from millm.api.dependencies import get_inference_service

        inference = client.app.dependency_overrides[get_inference_service]()
        other, wanted = MagicMock(), MagicMock()
        other.name, wanted.name = "some-other-model", "zora-v1.13-gguf"
        reports = iter([other, wanted, wanted, wanted])
        inference.get_loaded_model_info = lambda: next(reports, wanted)

        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "zora-v1.13-gguf",
                "messages": [{"role": "user", "content": "hi"}],
                "stream": True,
            },
        )

        assert response.status_code == 200, response.text
        assert svc.load_model_and_wait.called, (
            "a GGUF streaming request must now load the model rather than "
            "being turned away at the door"
        )
