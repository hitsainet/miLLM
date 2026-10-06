"""Structured output end to end (Feature 25, FR-25.10 – FR-25.12; SC-6 unit form).

A TINY REAL Llama with RANDOM weights and a character-level tokenizer: unconstrained it emits
noise, so a parsing, validating document is evidence the constraint ran — a fixture cannot agree
with the code by construction here.

MUTATION CONTROLS: M10 (processor not appended), M11 (truncation read from matcher state), plus
the route's subset check, CBM refusal and header, the compile call and the speculative drop.
"""

from __future__ import annotations

import json

import pytest
from jsonschema import Draft202012Validator

from millm.api.schemas.openai import ChatCompletionRequest
from millm.core.errors import ResponseFormatUnsupportedError
from tests.unit.f25_fixtures import (
    char_model,
    char_tokenizer,
    clear_loaded,
    make_client,
    make_service,
    model_row,
    unloaded_inference,
)

SCHEMA = {"type": "object",
          "properties": {"label": {"type": "string", "enum": ["humor", "not_humor"]},
                         "n": {"type": "integer", "minimum": 0, "maximum": 99}},
          "required": ["label", "n"], "additionalProperties": False}
MESSAGES = [{"role": "user", "content": "a"}]


@pytest.fixture(autouse=True)
def _clean():
    clear_loaded()
    yield
    clear_loaded()


@pytest.fixture
def svc():
    return make_service(char_model(), char_tokenizer())


def schema_format(schema=SCHEMA, strict=None):
    spec = {"name": "judge_v1", "schema": schema}
    if strict is not None:
        spec["strict"] = strict
    return {"type": "json_schema", "json_schema": spec}


def chat(**over) -> ChatCompletionRequest:
    body = {"model": "tiny", "messages": MESSAGES, "max_tokens": 120, "temperature": 1.0,
            "response_format": schema_format()}
    body.update(over)
    return ChatCompletionRequest(**body)


def _valid(text: str) -> None:
    Draft202012Validator(SCHEMA).validate(json.loads(text))


class TestTheOutputConforms:
    @pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
    async def test_json_schema_output_validates(self, svc, seed):
        resp = await svc.create_chat_completion(chat(seed=seed))
        choice = resp.choices[0]
        assert choice.finish_reason == "stop"
        _valid(choice.message.content)

    async def test_json_object_output_parses_as_an_object(self, svc):
        resp = await svc.create_chat_completion(chat(response_format={"type": "json_object"},
                                                     seed=5))
        assert resp.choices[0].finish_reason == "stop"
        assert isinstance(json.loads(resp.choices[0].message.content), dict)

    async def test_strict_false_is_still_enforced(self, svc):
        resp = await svc.create_chat_completion(chat(response_format=schema_format(strict=False),
                                                     seed=6))
        _valid(resp.choices[0].message.content)

    async def test_n_2_both_validate(self, svc):
        resp = await svc.create_chat_completion(chat(n=2, seed=7))
        assert len(resp.choices) == 2
        for c in resp.choices:
            assert c.finish_reason == "stop"
            _valid(c.message.content)

    async def test_extra_messages_rows_both_validate(self, svc):
        resp = await svc.create_chat_completion(
            chat(seed=8, extra_messages=[[{"role": "user", "content": "b"}]]))
        assert len(resp.choices) == 2
        for c in resp.choices:
            assert c.finish_reason == "stop"
            _valid(c.message.content)

    async def test_text_response_format_is_no_constraint(self, svc):
        from millm.services.inference_service import get_request_outcome, reset_request_outcome

        reset_request_outcome()
        await svc.create_chat_completion(chat(response_format={"type": "text"}, max_tokens=3))
        assert "constrained" not in get_request_outcome()


class TestTruncation:
    async def test_a_budget_ended_generation_is_length_never_stop(self, svc):
        """FR-25.12 / M11: completeness comes from the last generated token, not matcher state."""
        resp = await svc.create_chat_completion(chat(max_tokens=5, seed=1))
        choice = resp.choices[0]
        assert choice.finish_reason == "length"
        with pytest.raises(ValueError):
            json.loads(choice.message.content)

    async def test_a_partial_text_that_happens_to_parse_is_still_length(self, svc):
        """`{"n": 5}` can parse before the budget ends while a stop token never came."""
        resp = await svc.create_chat_completion(chat(response_format={"type": "json_object"},
                                                     max_tokens=2, seed=1))
        assert resp.choices[0].finish_reason == "length"


class TestOverHttp:
    @pytest.fixture
    def client(self, svc):
        c, _ = make_client(svc, model_row())
        return c

    def test_the_header_names_the_format_applied(self, client):
        body = chat(seed=0).model_dump(by_alias=True, exclude_none=True, exclude_unset=True)
        r = client.post("/v1/chat/completions", json=body)
        assert r.status_code == 200, r.text
        assert r.headers["X-miLLM-Constrained"] == 'json_schema;name="judge_v1"'
        _valid(r.json()["choices"][0]["message"]["content"])

    def test_the_header_is_present_on_a_truncated_generation(self, client):
        body = chat(seed=0, max_tokens=5).model_dump(by_alias=True, exclude_none=True,
                                                     exclude_unset=True)
        r = client.post("/v1/chat/completions", json=body)
        assert r.json()["choices"][0]["finish_reason"] == "length"
        assert r.headers["X-miLLM-Constrained"] == 'json_schema;name="judge_v1"'

    def test_json_object_header(self, client):
        r = client.post("/v1/chat/completions",
                        json={"model": "tiny", "messages": MESSAGES, "max_tokens": 60,
                              "seed": 5, "response_format": {"type": "json_object"}})
        assert r.headers["X-miLLM-Constrained"] == "json_object"

    def test_no_header_without_a_constraint(self, client):
        r = client.post("/v1/chat/completions",
                        json={"model": "tiny", "messages": MESSAGES, "max_tokens": 3})
        assert "X-miLLM-Constrained" not in r.headers

    def test_an_invalid_complete_output_is_a_500_never_a_200(self, client, monkeypatch):
        """FR-25.10.6: validation failure is an error, logged without the content."""
        import structlog

        from millm.core.errors import ConstrainedOutputInvalidError

        def fail(text, response_format):
            raise ConstrainedOutputInvalidError("bad", details={"length": len(text)})

        monkeypatch.setattr("millm.services.inference_service.validate_output", fail)
        body = chat(seed=0).model_dump(by_alias=True, exclude_none=True, exclude_unset=True)
        with structlog.testing.capture_logs() as logs:
            r = client.post("/v1/chat/completions", json=body)
        assert r.status_code == 500
        assert r.json()["error"]["code"] == "constrained_output_invalid"
        events = [e for e in logs if e["event"] == "constrained_output_invalid"]
        assert len(events) == 1 and "label" not in repr(events)


class TestRefusedBeforeLoad:
    @pytest.mark.parametrize("over, fragment", [
        ({"response_format": schema_format({"type": "integer", "multipleOf": 2})},
         "/multipleOf"),
        ({"response_format": schema_format({"type": "object",
                                            "description": "x" * (64 * 1024 + 1)})}, "bytes"),
        ({"stop": ["}"]}, "stop"),
        ({"stream": True}, "stream"),
        ({"logprobs": True, "max_tokens": 1}, "scoring"),
    ])
    def test_refused_naming_response_format(self, over, fragment):
        client, svc = make_client(unloaded_inference(), model_row())
        body = {"model": "tiny", "messages": MESSAGES, "response_format": schema_format()}
        body.update(over)
        r = client.post("/v1/chat/completions", json=body)
        assert r.status_code == 400, r.text
        assert r.json()["error"]["param"] == "response_format"
        assert fragment in r.json()["error"]["message"]
        assert svc.load_model_and_wait.call_count == 0

    def test_a_gguf_row(self):
        client, svc = make_client(unloaded_inference(), model_row(gguf_files=["m.gguf"]))
        r = client.post("/v1/chat/completions", json={"model": "tiny", "messages": MESSAGES,
                                                      "response_format": schema_format()})
        assert r.status_code == 400 and r.json()["error"]["param"] == "response_format"
        assert "GGUF" in r.json()["error"]["message"]
        assert svc.load_model_and_wait.call_count == 0

    @pytest.mark.parametrize("path, body", [
        ("/v1/completions", {"model": "tiny", "prompt": "a"}),
        ("/v1/embeddings", {"model": "tiny", "input": "a"}),
    ])
    def test_other_endpoints(self, path, body):
        client, svc = make_client(unloaded_inference(), model_row())
        r = client.post(path, json={**body, "response_format": {"type": "json_object"}})
        assert r.status_code == 400 and r.json()["error"]["param"] == "response_format"
        assert svc.load_model_and_wait.call_count == 0

    def test_while_continuous_batching_is_enabled(self):
        inference = unloaded_inference()
        inference.cbm_enabled = lambda: True
        client, svc = make_client(inference, model_row())
        r = client.post("/v1/chat/completions", json={"model": "tiny", "messages": MESSAGES,
                                                      "response_format": schema_format()})
        assert r.status_code == 400 and r.json()["error"]["param"] == "response_format"
        assert "continuous batching" in r.json()["error"]["message"]
        assert svc.load_model_and_wait.call_count == 0


class TestServiceGuards:
    async def test_llamacpp_refuses_in_the_service_too(self, svc):
        svc._engine_is_llamacpp = lambda: True
        with pytest.raises(ResponseFormatUnsupportedError):
            await svc.create_chat_completion(chat())

    async def test_a_direct_streaming_call_is_refused(self, svc):
        req = chat().model_copy(update={"stream": True})
        with pytest.raises(ResponseFormatUnsupportedError):
            async for _ in svc.stream_chat_completion(req):
                pass

    async def test_an_uncompiled_constraint_never_generates_unconstrained(self, svc):
        from millm.ml.generation_config import GenerationConfig

        gen = GenerationConfig.from_request(chat())
        inputs = svc._tokenizer("a", return_tensors="pt")
        with pytest.raises(ResponseFormatUnsupportedError):
            svc._build_generate_kwargs(gen, inputs)

    async def test_the_speculative_draft_is_dropped_for_a_constrained_request(self, svc):
        import structlog

        sentinel = object()
        svc._get_draft_model = lambda: sentinel
        seen = []
        real = svc._generate_sync

        def spy(kwargs, *a, **k):
            seen.append(dict(kwargs))
            return real(kwargs, *a, **k)

        svc._generate_sync = spy
        with structlog.testing.capture_logs() as logs:
            resp = await svc.create_chat_completion(chat(seed=0))
        assert seen and all("assistant_model" not in kw for kw in seen)
        assert any(e["event"] == "speculative_disabled_for_constraint" for e in logs)
        _valid(resp.choices[0].message.content)

    async def test_the_grammar_is_compiled_off_the_slot_and_cached(self, svc):
        await svc.create_chat_completion(chat(seed=0, max_tokens=3))
        cache = svc._grammar_cache()
        assert cache.misses == 1
        await svc.create_chat_completion(chat(seed=1, max_tokens=3))
        assert (cache.hits, cache.misses) == (1, 1)

    async def test_the_cache_is_dropped_on_unload(self, svc):
        await svc.create_chat_completion(chat(seed=0, max_tokens=3))
        before = svc._grammar_cache()
        svc.on_model_unloading()
        assert svc._grammar_cache_entry is None
        assert svc._grammar_cache() is not before
