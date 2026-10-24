"""Unused fields over HTTP (Feature 25, FR-25.1, FR-25.2; SC-1).

Real app, real tiny model. A field the request path does not use is reported in
`X-miLLM-Ignored-Fields` (or refused under `X-miLLM-Strict: true`), never dropped silently, and
reporting never changes the body.

MUTATION CONTROLS: M3 (message walk removed), M4 (values logged), M5 (policy call removed from a
route), plus the header-setting lines in each route.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from millm.api.schemas.openai import (
    ChatCompletionChoice,
    ChatCompletionResponse,
    ChatMessage,
    Usage,
)
from tests.unit.f25_fixtures import (
    clear_loaded,
    make_client,
    make_service,
    model_row,
    unloaded_inference,
    word_model,
    word_tokenizer,
)

H = "X-miLLM-Ignored-Fields"


@pytest.fixture(autouse=True)
def _clean():
    clear_loaded()
    yield
    clear_loaded()


@pytest.fixture
def client():
    inference = make_service(word_model(), word_tokenizer())
    c, svc = make_client(inference, model_row())
    return c


def chat_body(**over):
    body = {"model": "tiny", "messages": [{"role": "user", "content": "w1 w2"}],
            "max_tokens": 3, "temperature": 0}
    body.update(over)
    return body


class TestReported:
    def test_an_unknown_field_is_named_in_the_header(self, client):
        r = client.post("/v1/chat/completions", json=chat_body(foo=1))
        assert r.status_code == 200, r.text
        assert r.headers[H] == '"foo"'

    def test_a_message_field_is_reported_with_its_location(self, client):
        body = chat_body(messages=[{"role": "user", "content": "w1", "name": "alice"}])
        r = client.post("/v1/chat/completions", json=body)
        assert r.status_code == 200, r.text
        assert r.headers[H] == '"messages[0].name"'

    def test_no_header_when_nothing_was_ignored(self, client):
        r = client.post("/v1/chat/completions", json=chat_body())
        assert r.status_code == 200, r.text
        assert H not in r.headers

    def test_the_body_is_unchanged_by_an_unused_field(self, client):
        plain = client.post("/v1/chat/completions", json=chat_body()).json()
        extra = client.post("/v1/chat/completions", json=chat_body(foo=1, bar={"x": 2})).json()
        for body in (plain, extra):
            body.pop("id"), body.pop("created")
        assert plain == extra

    def test_streaming_carries_the_header(self, client):
        with client.stream("POST", "/v1/chat/completions",
                           json=chat_body(stream=True, foo=1)) as r:
            assert r.status_code == 200
            assert r.headers[H] == '"foo"'
            text = "".join(r.iter_text())
        assert "[DONE]" in text

    def test_completions_reports_too(self, client):
        r = client.post("/v1/completions",
                        json={"model": "tiny", "prompt": "w1", "max_tokens": 2, "temperature": 0,
                              "echo": True})
        assert r.status_code == 200, r.text
        assert r.headers[H] == '"echo"'

    def test_embeddings_reports_too(self, client):
        r = client.post("/v1/embeddings", json={"model": "tiny", "input": "w1", "foo": 1})
        assert r.status_code == 200, r.text
        assert r.headers[H] == '"foo"'


class TestStrict:
    def test_strict_refuses_naming_the_field(self, client):
        r = client.post("/v1/chat/completions", json=chat_body(foo=1),
                        headers={"X-miLLM-Strict": "true"})
        assert r.status_code == 400
        err = r.json()["error"]
        assert err["code"] == "unused_fields_refused" and err["param"] == "foo"
        assert err["type"] == "invalid_request_error"

    def test_strict_names_a_message_field(self, client):
        body = chat_body(messages=[{"role": "user", "content": "w1", "name": "alice"}])
        r = client.post("/v1/chat/completions", json=body, headers={"X-miLLM-Strict": "1"})
        assert r.status_code == 400
        assert "messages[0].name" in r.json()["error"]["message"]

    def test_an_unknown_strict_value_is_refused(self, client):
        r = client.post("/v1/chat/completions", json=chat_body(),
                        headers={"X-miLLM-Strict": "yes"})
        assert r.status_code == 400
        assert r.json()["error"]["param"] == "X-miLLM-Strict"

    @pytest.mark.parametrize("path, body", [
        ("/v1/completions", {"model": "tiny", "prompt": "w1", "max_tokens": 1, "foo": 1}),
        ("/v1/embeddings", {"model": "tiny", "input": "w1", "foo": 1}),
    ])
    def test_strict_on_every_endpoint(self, client, path, body):
        r = client.post(path, json=body, headers={"X-miLLM-Strict": "true"})
        assert r.status_code == 400 and r.json()["error"]["param"] == "foo"


class TestEnginePathCaseB:
    def test_chat_template_kwargs_on_a_gguf_row_is_refused_under_strict_before_load(self):
        client, svc = make_client(unloaded_inference(), model_row(gguf_files=["m.gguf"]))
        r = client.post("/v1/chat/completions",
                        json=chat_body(chat_template_kwargs={"enable_thinking": False}),
                        headers={"X-miLLM-Strict": "true"})
        assert r.status_code == 400
        assert r.json()["error"]["param"] == "chat_template_kwargs"
        assert svc.load_model_and_wait.call_count == 0

    def test_chat_template_kwargs_on_a_gguf_row_is_reported(self):
        inference = MagicMock()
        inference.backend_name = "llamacpp"
        loaded = MagicMock()
        loaded.name = "tiny"
        inference.get_loaded_model_info = lambda: loaded
        inference.active_circuit_rung = AsyncMock(return_value=None)
        inference.create_chat_completion = AsyncMock(return_value=ChatCompletionResponse(
            id="chatcmpl-" + "0" * 24, created=0, model="tiny",
            choices=[ChatCompletionChoice(index=0, message=ChatMessage(role="assistant",
                                                                       content="x"),
                                          finish_reason="stop")],
            usage=Usage(prompt_tokens=1, completion_tokens=1)))
        client, _ = make_client(inference, model_row(gguf_files=["m.gguf"]))
        r = client.post("/v1/chat/completions",
                        json=chat_body(chat_template_kwargs={"enable_thinking": False}))
        assert r.status_code == 200, r.text
        assert r.headers[H] == '"chat_template_kwargs"'


def test_a_message_extra_never_reaches_the_chat_template():
    """2.14: `extra="allow"` keeps extras for REPORTING only — the template sees role/content."""
    tok = word_tokenizer()
    seen: list = []
    real = tok.apply_chat_template

    def spy(messages, *a, **k):
        seen.append([dict(m) for m in messages])
        return real(messages, *a, **k)

    tok.apply_chat_template = spy
    client, _ = make_client(make_service(word_model(), tok), model_row())
    body = chat_body(messages=[{"role": "user", "content": "w1", "name": "alice", "x": 1}])
    assert client.post("/v1/chat/completions", json=body).status_code == 200
    assert seen, "the template was never rendered"
    for messages in seen:
        for m in messages:
            assert set(m) == {"role", "content"}, m


def test_the_unused_field_log_carries_locations_never_values(client):
    """2.15 / M4, through the route: a value may be prompt text."""
    import structlog

    sentinel = "SENTINEL-PROMPT-7c1e"
    body = chat_body(foo=sentinel,
                     messages=[{"role": "user", "content": "w1", "name": sentinel}])
    with structlog.testing.capture_logs() as logs:
        r = client.post("/v1/chat/completions", json=body)
    assert r.status_code == 200
    events = [e for e in logs if e["event"] == "request_fields_unused"]
    assert len(events) == 1 and events[0]["fields"] == ["foo", "messages[0].name"]
    assert sentinel not in repr(events)
