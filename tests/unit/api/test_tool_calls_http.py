"""Function calling through the REAL app (operator bug 2026-10-28): the route honours `tools`.

REACHABILITY. The capability is wired by one table row (`request_policy.OUTPUT_CHANGING["tools"]`
for chat/transformers) and one service call (`InferenceService._chat_prompt`). Put either back to
what it was and `test_open_webui_request_gets_tool_calls` goes red (controls R1/R2 in
`0xcc/reviews/function_calling_2026-10-28.md`).
"""

from __future__ import annotations

import json

import pytest

from tests.unit.f25_fixtures import (
    clear_loaded,
    make_client,
    make_service,
    model_row,
    unloaded_inference,
    word_model,
    word_tokenizer,
)
from tests.unit.tool_call_fixtures import CALL_REPLY, WEATHER_TOOL, family_tokenizer, scripted_model

URL = "/v1/chat/completions"
MESSAGES = [{"role": "user", "content": "Weather in Paris?"}]


@pytest.fixture(autouse=True)
def _clean():
    clear_loaded()
    yield
    clear_loaded()


def lfm_client(reply: str = CALL_REPLY["lfm2"]):
    tok = family_tokenizer("lfm2")
    client, svc = make_client(
        make_service(scripted_model(tok, reply), tok, name="LFM2.5-1.2B-Instruct"),
        model_row(name="LFM2.5-1.2B-Instruct"),
    )
    return client, svc


def body(**over):
    out = {"model": "LFM2.5-1.2B-Instruct", "messages": MESSAGES, "tools": [WEATHER_TOOL],
           "max_tokens": 200}
    out.update(over)
    return out


class TestHonoured:
    def test_open_webui_request_gets_tool_calls(self):
        """The operator's request: LFM2.5 on transformers, `tools` sent, `tool_choice` absent."""
        client, _ = lfm_client()
        r = client.post(URL, json=body())
        assert r.status_code == 200, r.text
        choice = r.json()["choices"][0]
        assert choice["finish_reason"] == "tool_calls"
        [call] = choice["message"]["tool_calls"]
        assert call["type"] == "function" and call["function"]["name"] == "get_weather"
        assert json.loads(call["function"]["arguments"]) == {"city": "Paris"}
        # Provenance headers are on a tool-call response as on any other.
        assert r.headers["X-miLLM-Backend"] == "serial"
        assert "X-miLLM-Model" in r.headers and "X-miLLM-Steering" in r.headers

    def test_streamed(self):
        client, _ = lfm_client()
        r = client.post(URL, json=body(stream=True))
        assert r.status_code == 200
        events = [json.loads(line[6:]) for line in r.text.splitlines()
                  if line.startswith("data: ") and line != "data: [DONE]"]
        deltas = [e["choices"][0] for e in events if e.get("choices")]
        assert any("tool_calls" in c["delta"] for c in deltas)
        assert [c["finish_reason"] for c in deltas if c.get("finish_reason")] == ["tool_calls"]

    def test_a_response_without_calls_has_no_tool_calls_key(self):
        client, _ = lfm_client("It is sunny.")
        message = client.post(URL, json=body()).json()["choices"][0]["message"]
        assert "tool_calls" not in message and message["content"] == "It is sunny."

    def test_tool_history_is_not_reported_unused(self):
        client, _ = lfm_client()
        history = MESSAGES + [
            {"role": "assistant", "content": None, "tool_calls": [{
                "id": "c1", "type": "function",
                "function": {"name": "get_weather", "arguments": "{\"city\": \"Paris\"}"}}]},
            {"role": "tool", "tool_call_id": "c1", "content": "18C"},
        ]
        r = client.post(URL, json=body(messages=history), headers={"X-miLLM-Strict": "true"})
        assert r.status_code == 200, r.text


class TestRefused:
    def test_a_template_that_takes_no_tools(self):
        client, _ = make_client(make_service(word_model(), word_tokenizer()), model_row())
        for stream in (False, True):
            r = client.post(URL, json=body(model="tiny", stream=stream))
            assert r.status_code == 400, r.text
            error = r.json()["error"]
            assert error["code"] == "field_not_honoured" and error["param"] == "tools"
            assert "takes no tools" in error["message"]

    def test_a_gguf_model_before_any_load(self):
        client, svc = make_client(unloaded_inference(), model_row(gguf_files=["m.gguf"]))
        r = client.post(URL, json=body(model="tiny"))
        assert r.status_code == 400 and r.json()["error"]["param"] == "tools"
        assert "transformers engine only" in r.json()["error"]["message"]
        assert svc.load_model_and_wait.call_count == 0

    def test_gguf_tool_history_is_reported_not_dropped(self):
        client, svc = make_client(unloaded_inference(), model_row(gguf_files=["m.gguf"]))
        history = MESSAGES + [{"role": "assistant", "content": None, "tool_calls": [{
            "id": "c1", "type": "function", "function": {"name": "f", "arguments": "{}"}}]}]
        r = client.post(URL, json={"model": "tiny", "messages": history},
                        headers={"X-miLLM-Strict": "true"})
        assert r.status_code == 400 and r.json()["error"]["code"] == "unused_fields_refused"
        assert "messages[1].tool_calls" in r.json()["error"]["message"]

    @pytest.mark.parametrize("over, param", [
        ({"logprobs": True, "max_tokens": 1}, "tools"),
        ({"response_format": {"type": "json_object"}}, "tools"),
        ({"tool_choice": {"type": "function", "function": {"name": "other"}}}, "tool_choice"),
    ])
    def test_combinations(self, over, param):
        client, svc = make_client(unloaded_inference(), model_row())
        r = client.post(URL, json=body(model="tiny", **over))
        assert r.status_code == 400, r.text
        assert r.json()["error"]["param"] == param
        assert svc.load_model_and_wait.call_count == 0

    @pytest.mark.parametrize("path, extra", [
        ("/v1/completions", {"prompt": "a"}),
        ("/v1/embeddings", {"input": "a"}),
    ])
    def test_other_endpoints(self, path, extra):
        client, _ = make_client(unloaded_inference(), model_row())
        r = client.post(path, json={"model": "tiny", "tools": [WEATHER_TOOL], **extra})
        assert r.status_code == 400 and r.json()["error"]["param"] == "tools"
