"""Small refusals and aliases (Feature 25, FR-25.3.3a/b, FR-25.3.5, FR-25.3.6, FR-25.4; SC-3).

* `n > 1` on /v1/completions is refused, on both engines, before any auto-load (T-56) — the
  service never read `n`, so n=3 returned one choice per prompt.
* Streaming chat with `n > 1` or `extra_messages` is refused — the streaming path reads neither,
  so a streamed n=3 returned one choice. Refused by the schema AND by the service, because
  Feature 26's batch runner calls the service directly (M16).
* `max_completion_tokens` is honoured as `max_tokens`; a disagreeing pair is refused (T-58).
* `user` is reported as unused, and strict mode refuses it (T-57).
* A request carrying `n > 1` (or a seed, a constraint, a penalty) never reaches the continuous
  batching manager, which would return one choice and drop the field (FR-25.3.6).
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from millm.api.schemas.openai import ChatCompletionRequest, TextCompletionRequest
from millm.core.errors import FieldNotHonouredError
from tests.unit.f25_fixtures import (
    clear_loaded,
    make_client,
    make_service,
    model_row,
    unloaded_inference,
    word_model,
    word_tokenizer,
)


@pytest.fixture(autouse=True)
def _clean():
    clear_loaded()
    yield
    clear_loaded()


def chat_body(**over):
    body = {"model": "tiny", "messages": [{"role": "user", "content": "w1"}]}
    body.update(over)
    return body


class TestNOnCompletions:
    @pytest.mark.parametrize("gguf", [None, ["m.gguf"]], ids=["transformers", "gguf"])
    def test_n_2_is_a_400_naming_n_before_any_load(self, gguf):
        client, svc = make_client(unloaded_inference(), model_row(gguf_files=gguf))
        r = client.post("/v1/completions", json={"model": "tiny", "prompt": "w1", "n": 2})
        assert r.status_code == 400, r.text
        assert r.json()["error"]["param"] == "n"
        assert svc.load_model_and_wait.call_count == 0

    def test_the_schema_refuses_it_for_direct_callers_too(self):
        """Over HTTP the table refuses `n` as well, so the route test alone cannot see the schema
        rule; a direct caller (Feature 26's batch runner builds request objects) only has the
        schema. Found by control S-comp-n surviving."""
        with pytest.raises(ValidationError) as exc:
            TextCompletionRequest(model="m", prompt="x", n=2)
        assert exc.value.errors()[0]["loc"] == ("n",)
        assert "T-56" in str(exc.value)

    def test_n_1_is_fine(self):
        assert TextCompletionRequest(model="m", prompt="x", n=1).n == 1


class TestStreamingChat:
    @pytest.mark.parametrize("over, param", [
        ({"n": 2}, "n"),
        ({"extra_messages": [[{"role": "user", "content": "w2"}]]}, "extra_messages"),
    ])
    def test_refused_over_http_naming_the_field(self, over, param):
        client, svc = make_client(unloaded_inference(), model_row())
        r = client.post("/v1/chat/completions", json=chat_body(stream=True, **over))
        assert r.status_code == 400, r.text
        assert r.json()["error"]["param"] == param
        assert svc.load_model_and_wait.call_count == 0

    def test_non_streaming_n_2_still_parses(self):
        assert ChatCompletionRequest(**chat_body(n=2)).n == 2

    @pytest.mark.parametrize("update", [
        {"n": 2},
        {"extra_messages": [[{"role": "user", "content": "w2"}]]},
    ])
    async def test_a_direct_service_call_is_refused(self, update):
        """M16: the batch runner (Feature 26) bypasses the schema."""
        svc = make_service(word_model(), word_tokenizer())
        request = ChatCompletionRequest(**chat_body(stream=True)).model_copy(update=update)

        def no_generation(*a, **k):
            raise AssertionError("a refused streaming request reached generation")

        svc._generate_in_thread = no_generation
        with pytest.raises(FieldNotHonouredError) as exc:
            async for _ in svc.stream_chat_completion(request):
                pass
        assert exc.value.details["param"] == next(iter(update))


class TestMaxCompletionTokens:
    async def test_it_limits_generation_like_max_tokens(self):
        svc = make_service(word_model(), word_tokenizer())
        seen = []
        real = svc._generate_sync

        def spy(kwargs, *a, **k):
            seen.append(kwargs["max_new_tokens"])
            return real(kwargs, *a, **k)

        svc._generate_sync = spy
        req = ChatCompletionRequest(**chat_body(max_completion_tokens=5, temperature=0))
        response = await svc.create_chat_completion(req)
        assert seen == [5]
        assert response.usage.completion_tokens <= 5

    def test_a_disagreeing_pair_is_refused_naming_both(self):
        with pytest.raises(ValidationError) as exc:
            ChatCompletionRequest(**chat_body(max_tokens=3, max_completion_tokens=5))
        assert "max_completion_tokens" in str(exc.value) and "max_tokens" in str(exc.value)

    def test_an_agreeing_pair_is_fine(self):
        req = ChatCompletionRequest(**chat_body(max_tokens=5, max_completion_tokens=5))
        assert req.max_tokens == 5

    def test_it_satisfies_scorings_max_tokens_1(self):
        req = TextCompletionRequest(model="m", prompt="x", max_completion_tokens=1, logprobs=1)
        assert req.wants_scores() and req.max_tokens == 1

    def test_over_http_the_disagreement_is_a_400(self):
        client, _ = make_client(unloaded_inference(), model_row())
        r = client.post("/v1/completions",
                        json={"model": "tiny", "prompt": "x", "max_tokens": 2,
                              "max_completion_tokens": 3})
        assert r.status_code == 400
        assert "max_completion_tokens" in r.json()["error"]["message"]

    def test_on_embeddings_it_is_refused(self):
        client, svc = make_client(unloaded_inference(), model_row())
        r = client.post("/v1/embeddings",
                        json={"model": "tiny", "input": "x", "max_completion_tokens": 3})
        assert r.status_code == 400
        assert r.json()["error"]["param"] == "max_completion_tokens"
        assert svc.load_model_and_wait.call_count == 0


class TestUser:
    @pytest.mark.parametrize("path, body", [
        ("/v1/chat/completions", chat_body(max_tokens=1)),
        ("/v1/completions", {"model": "tiny", "prompt": "w1", "max_tokens": 1}),
        ("/v1/embeddings", {"model": "tiny", "input": "w1"}),
    ])
    def test_user_is_reported_and_strict_refuses_it(self, path, body):
        client, _ = make_client(make_service(word_model(), word_tokenizer()), model_row())
        r = client.post(path, json={**body, "user": "alice"})
        assert r.status_code == 200, r.text
        assert r.headers["X-miLLM-Ignored-Fields"] == '"user"'
        r = client.post(path, json={**body, "user": "alice"},
                        headers={"X-miLLM-Strict": "true"})
        assert r.status_code == 400 and r.json()["error"]["param"] == "user"

    def test_no_request_schema_declares_user(self):
        from millm.api.schemas.openai import EmbeddingRequest

        for cls in (ChatCompletionRequest, TextCompletionRequest, EmbeddingRequest):
            assert "user" not in cls.model_fields, cls


class TestContinuousBatchingNeverDropsAListedField:
    @pytest.mark.parametrize("over", [
        {"n": 2}, {"frequency_penalty": 0.5}, {"presence_penalty": -0.5},
    ])
    async def test_served_serially_never_by_the_manager(self, over):
        svc = make_service(word_model(), word_tokenizer())

        class _Cbm:
            is_running = True

            def sampling_params_match(self, temperature, top_p):
                return True

        svc._cbm_backend = _Cbm()
        assert svc._use_cbm(), "precondition: the manager is running"

        async def cbm_path(*a, **k):
            raise AssertionError("served by the continuous batching manager")

        svc._cbm_chat_completion = cbm_path
        req = ChatCompletionRequest(**chat_body(max_tokens=2, temperature=1.0, **over))
        response = await svc.create_chat_completion(req)
        assert len(response.choices) == req.n

    async def test_a_plain_request_still_uses_the_manager(self):
        """The gate must not route everything serial (control for the test above)."""
        svc = make_service(word_model(), word_tokenizer())

        class _Cbm:
            is_running = True

            def sampling_params_match(self, temperature, top_p):
                return True

        svc._cbm_backend = _Cbm()
        calls = []

        async def cbm_path(request):
            calls.append(request)
            raise RuntimeError("stop here")

        svc._cbm_chat_completion = cbm_path
        with pytest.raises(RuntimeError, match="stop here"):
            await svc.create_chat_completion(ChatCompletionRequest(**chat_body(max_tokens=2)))
        assert len(calls) == 1
