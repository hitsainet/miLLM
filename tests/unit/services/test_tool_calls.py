"""Function calling on the transformers engine (operator bug 2026-10-28).

Open WebUI in native function-calling mode got `'tools' is not honoured on /v1/chat/completions
for model 'LFM2.5-1.2B-Instruct' with the transformers engine: function calling is not
implemented`. These tests cover `millm/services/tool_calls.py` and its wiring in the inference
service, on the REAL chat templates of the three families production serves (LFM2.5, Qwen2.5,
Llama 3.1; `tests/unit/tool_call_fixtures.py` says where each came from).

MUTATION CONTROLS (recorded in `0xcc/reviews/function_calling_2026-10-28.md`):
  * `_chat_prompt` stops passing `tools` to the template      -> TestRender / TestServed red
  * `_chat_reply` stops parsing                               -> every served-call test red
  * the stream filter not fed                                 -> TestStreamed red
  * `template_messages` keeps the raw JSON-string arguments   -> TestHistory red
  * `_probe_note_last_user_span` ignores the tool plan        -> TestProbeSpan red
  * the name check in `parse_tool_calls` dropped              -> the unknown-name test red
  * the CBM gate ignores `has_tools`                          -> TestContinuousBatching red
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import pytest

from millm.api.schemas.openai import ChatCompletionRequest, ChatMessage
from millm.core.errors import FieldNotHonouredError
from millm.services.tool_calls import (
    HERMES,
    LFM2,
    LLAMA3_JSON,
    ToolCallStreamFilter,
    ToolPlan,
    detect_format,
    parse_tool_calls,
    plan_tools,
    template_messages,
)
from tests.unit.f25_fixtures import clear_loaded, make_service, word_model, word_tokenizer
from tests.unit.tool_call_fixtures import (
    CALL_REPLY,
    FAMILIES,
    TIME_TOOL,
    WEATHER_TOOL,
    family_tokenizer,
    scripted_model,
    template_text,
)

USER = [{"role": "user", "content": "Weather in Paris?"}]
NAMES = frozenset({"get_weather", "get_time"})


def plan(fmt: str, prefill: str = "", parallel: bool = True) -> ToolPlan:
    return ToolPlan(tools=[WEATHER_TOOL, TIME_TOOL], fmt=fmt, prefill=prefill, names=NAMES,
                    parallel=parallel)


def request(**over: Any) -> ChatCompletionRequest:
    body: dict[str, Any] = {"model": "tiny", "messages": USER, "tools": [WEATHER_TOOL, TIME_TOOL],
                            "max_tokens": 200, "temperature": 0}
    body.update(over)
    return ChatCompletionRequest(**body)


def args_of(call: dict) -> dict:
    assert call["type"] == "function" and call["id"].startswith("call_")
    assert isinstance(call["function"]["arguments"], str)
    return json.loads(call["function"]["arguments"])


@pytest.fixture(autouse=True)
def _clean():
    clear_loaded()
    yield
    clear_loaded()


def served(family: str, reply: str):
    tok = family_tokenizer(family)
    return make_service(scripted_model(tok, reply), tok)


# ---------------------------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------------------------


class TestParse:
    @pytest.mark.parametrize("family", list(FAMILIES))
    def test_each_family_reply_parses(self, family):
        parsed = parse_tool_calls(CALL_REPLY[family], plan(family))
        assert parsed is not None and parsed.content is None
        [call] = parsed.tool_calls
        assert call["function"]["name"] == "get_weather" and args_of(call) == {"city": "Paris"}

    def test_lfm2_several_calls_and_literals(self):
        text = ("<|tool_call_start|>[get_weather(city=\"Paris\", metric=true, n=2, x=null), "
                "get_time()]<|tool_call_end|>")
        parsed = parse_tool_calls(text, plan(LFM2))
        assert [c["function"]["name"] for c in parsed.tool_calls] == ["get_weather", "get_time"]
        assert args_of(parsed.tool_calls[0]) == {"city": "Paris", "metric": True, "n": 2, "x": None}
        assert args_of(parsed.tool_calls[1]) == {}

    def test_parallel_false_keeps_one(self):
        text = "<|tool_call_start|>[get_weather(city='a'), get_time()]<|tool_call_end|>"
        assert len(parse_tool_calls(text, plan(LFM2, parallel=False)).tool_calls) == 1

    def test_lfm2_content_around_the_call_is_kept(self):
        text = "Let me check.<|tool_call_start|>[get_time()]<|tool_call_end|>"
        parsed = parse_tool_calls(text, plan(LFM2))
        assert parsed.content == "Let me check." and len(parsed.tool_calls) == 1

    def test_lfm2_json_body_and_unterminated_block(self):
        text = '<|tool_call_start|>[{"name": "get_weather", "arguments": {"city": "Oslo"}}]'
        parsed = parse_tool_calls(text, plan(LFM2))
        assert args_of(parsed.tool_calls[0]) == {"city": "Oslo"}

    def test_hermes_two_blocks_and_string_arguments(self):
        text = ('<tool_call>\n{"name": "get_weather", "arguments": "{\\"city\\": \\"Rome\\"}"}\n'
                '</tool_call>\n<tool_call>\n{"name": "get_time", "arguments": {}}\n')
        parsed = parse_tool_calls(text, plan(HERMES))
        assert [c["function"]["name"] for c in parsed.tool_calls] == ["get_weather", "get_time"]
        assert args_of(parsed.tool_calls[0]) == {"city": "Rome"}

    def test_llama_python_tag_and_several_objects(self):
        text = ('<|python_tag|>{"name": "get_weather", "parameters": {"city": "Lima"}}; '
                '{"name": "get_time", "parameters": {}}')
        parsed = parse_tool_calls(text, plan(LLAMA3_JSON))
        assert [c["function"]["name"] for c in parsed.tool_calls] == ["get_weather", "get_time"]

    @pytest.mark.parametrize("fmt, text", [
        (LFM2, "<|tool_call_start|>[get_weather(city=)]<|tool_call_end|>"),
        (LFM2, "<|tool_call_start|>[get_weather('Paris')]<|tool_call_end|>"),
        (LFM2, "<|tool_call_start|>[__import__('os').system('x')]<|tool_call_end|>"),
        (LFM2, "no call here"),
        (HERMES, "<tool_call>\n{not json}\n</tool_call>"),
        (LLAMA3_JSON, "The weather is nice."),
        (LLAMA3_JSON, '{"name": "get_weather", "parameters": {"city": "x"}} and then prose'),
        (LLAMA3_JSON, '{"answer": 42}'),
    ])
    def test_unparseable_output_is_content_never_an_error(self, fmt, text):
        assert parse_tool_calls(text, plan(fmt)) is None

    @pytest.mark.parametrize("family", list(FAMILIES))
    def test_a_function_the_request_did_not_offer_is_not_a_call(self, family):
        text = CALL_REPLY[family].replace("get_weather", "delete_everything")
        assert parse_tool_calls(text, plan(family)) is None


# ---------------------------------------------------------------------------------------------
# Plan and render
# ---------------------------------------------------------------------------------------------


class TestPlan:
    @pytest.mark.parametrize("family", list(FAMILIES))
    def test_the_real_templates_are_recognised(self, family):
        assert detect_format(template_text(family)) == family

    def test_a_template_without_tools_is_refused(self):
        tok = word_tokenizer()
        with pytest.raises(FieldNotHonouredError) as exc:
            plan_tools(request(), tok)
        assert exc.value.details["param"] == "tools"
        assert "takes no tools" in exc.value.message

    def test_an_unrecognised_call_format_is_refused(self):
        """Gemma 4 writes `<|tool_call>call:name{...}` with special-token markers the decode
        removes: refused with the reason, never served with the call flattened into prose."""
        tok = SimpleNamespace(chat_template=(
            "{% for t in tools %}<|tool>{{ t }}<tool|>{% endfor %}<|tool_call>call:"))
        with pytest.raises(FieldNotHonouredError) as exc:
            plan_tools(request(), tok)
        assert "does not recognise this model's tool-call output format" in exc.value.message

    def test_none_renders_no_tools(self):
        assert plan_tools(request(tool_choice="none"), family_tokenizer("lfm2")) is None

    @pytest.mark.parametrize("family, prefill", [
        ("lfm2", "<|tool_call_start|>["),
        ("hermes", "<tool_call>\n"),
        ("llama3_json", '{"name": "'),
    ])
    def test_required_opens_the_call(self, family, prefill):
        assert plan_tools(request(tool_choice="required"), family_tokenizer(family)).prefill == (
            prefill)
        assert plan_tools(request(), family_tokenizer(family)).prefill == ""

    @pytest.mark.parametrize("family, prefill", [
        ("lfm2", "<|tool_call_start|>[get_time("),
        ("hermes", '<tool_call>\n{"name": "get_time", "arguments": '),
        ("llama3_json", '{"name": "get_time", "parameters": '),
    ])
    def test_a_named_function_is_forced(self, family, prefill):
        choice = {"type": "function", "function": {"name": "get_time"}}
        assert plan_tools(request(tool_choice=choice), family_tokenizer(family)).prefill == prefill

    def test_schema_refusals(self):
        with pytest.raises(ValueError, match="not in tools"):
            request(tool_choice={"type": "function", "function": {"name": "nope"}})
        with pytest.raises(ValueError, match="needs at least one"):
            request(tools=None, tool_choice="required")
        with pytest.raises(ValueError, match="response_format"):
            request(response_format={"type": "json_object"})


class TestRender:
    @pytest.mark.parametrize("family, shown", [
        ("lfm2", "List of tools: ["),
        ("hermes", "<tools>"),
        ("llama3_json", "Given the following functions"),
    ])
    def test_the_tools_reach_the_template(self, family, shown):
        svc = served(family, "x")
        prompt, tool_plan = svc._chat_prompt(request())
        assert shown in prompt and '"get_weather"' in prompt
        assert tool_plan is not None and tool_plan.fmt == family
        none_prompt, none_plan = svc._chat_prompt(request(tool_choice="none"))
        assert none_plan is None and '"get_weather"' not in none_prompt

    def test_a_forced_call_follows_the_generation_prompt(self):
        svc = served("lfm2", "x")
        prompt, _ = svc._chat_prompt(request(tool_choice="required"))
        assert prompt.endswith("<|im_start|>assistant\n<|tool_call_start|>[")


class TestHistory:
    """An OpenAI client sends a call's `arguments` as a JSON STRING. LFM2.5's template raises on
    one and Qwen2.5/Llama 3.1 would encode it twice; `template_messages` hands them objects."""

    HISTORY = [
        {"role": "user", "content": "Weather in Paris?"},
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": "call_1", "type": "function",
             "function": {"name": "get_weather", "arguments": "{\"city\": \"Paris\"}"}}]},
        {"role": "tool", "tool_call_id": "call_1", "content": "18C"},
    ]

    @pytest.mark.parametrize("family, rendered", [
        ("lfm2", "<|tool_call_start|>[get_weather(city='Paris')]<|tool_call_end|>"),
        ("hermes", '{"name": "get_weather", "arguments": {"city": "Paris"}}'),
        ("llama3_json", '{"name": "get_weather", "parameters": {"city": "Paris"}}'),
    ])
    def test_history_renders_in_the_models_own_format(self, family, rendered):
        svc = served(family, "x")
        prompt, _ = svc._chat_prompt(request(messages=self.HISTORY))
        assert rendered in prompt and "18C" in prompt

    def test_messages_without_calls_carry_no_tool_calls_key(self):
        """Llama 3.1's template tests `'tool_calls' in message`: a null key would be a call."""
        out = template_messages([ChatMessage(role="user", content="a")])
        assert out == [{"role": "user", "content": "a"}]

    def test_no_template_refuses_tool_history(self):
        tok = word_tokenizer(template=False)
        svc = make_service(word_model(), tok)
        with pytest.raises(FieldNotHonouredError):
            svc._chat_prompt(request(tools=None, messages=self.HISTORY))


# ---------------------------------------------------------------------------------------------
# Served (the real generate loop with a scripted reply)
# ---------------------------------------------------------------------------------------------


class TestServed:
    @pytest.mark.parametrize("family", list(FAMILIES))
    async def test_a_call_is_returned_as_tool_calls(self, family):
        resp = await served(family, CALL_REPLY[family]).create_chat_completion(request())
        choice = resp.choices[0]
        assert choice.finish_reason == "tool_calls"
        assert choice.message.content is None
        [call] = choice.message.tool_calls
        assert call["function"]["name"] == "get_weather" and args_of(call) == {"city": "Paris"}

    @pytest.mark.parametrize("family", list(FAMILIES))
    async def test_a_plain_answer_stays_content(self, family):
        resp = await served(family, "It is sunny.").create_chat_completion(request())
        choice = resp.choices[0]
        assert choice.message.content == "It is sunny." and choice.message.tool_calls is None
        assert choice.finish_reason == "stop"

    @pytest.mark.parametrize("family, rest", [
        ("lfm2", "get_weather(city='Paris')]<|tool_call_end|>"),
        ("hermes", '{"name": "get_weather", "arguments": {"city": "Paris"}}\n</tool_call>'),
        ("llama3_json", 'get_weather", "parameters": {"city": "Paris"}}'),
    ])
    async def test_required_parses_prefill_plus_reply(self, family, rest):
        svc = served(family, rest)
        resp = await svc.create_chat_completion(request(tool_choice="required"))
        assert resp.choices[0].finish_reason == "tool_calls"
        assert args_of(resp.choices[0].message.tool_calls[0]) == {"city": "Paris"}

    async def test_named_function(self):
        svc = served("lfm2", "city='Paris')]<|tool_call_end|>")
        choice = {"type": "function", "function": {"name": "get_weather"}}
        resp = await svc.create_chat_completion(request(tool_choice=choice))
        assert args_of(resp.choices[0].message.tool_calls[0]) == {"city": "Paris"}

    async def test_unparseable_forced_reply_is_content_with_its_prefix(self):
        svc = served("lfm2", "oops")
        resp = await svc.create_chat_completion(request(tool_choice="required"))
        assert resp.choices[0].message.content == "<|tool_call_start|>[oops"
        assert resp.choices[0].finish_reason == "stop"

    async def test_none_returns_the_text_as_content(self):
        svc = served("lfm2", CALL_REPLY["lfm2"])
        resp = await svc.create_chat_completion(request(tool_choice="none"))
        assert resp.choices[0].message.tool_calls is None
        assert resp.choices[0].finish_reason == "stop"

    async def test_n_2_and_batched_rows_each_parse(self):
        svc = served("lfm2", CALL_REPLY["lfm2"])
        resp = await svc.create_chat_completion(request(n=2))
        assert [c.finish_reason for c in resp.choices] == ["tool_calls", "tool_calls"]
        resp = await svc.create_chat_completion(request(extra_messages=[USER]))
        assert [c.finish_reason for c in resp.choices] == ["tool_calls", "tool_calls"]

    @pytest.mark.parametrize("family, bos_count", [("lfm2", 1), ("llama3_json", 1), ("hermes", 0)])
    async def test_one_bos_on_a_tools_render(self, family, bos_count):
        """The double-BOS rule (`prompt_encoding`) still decides: the tools render begins with
        the template's BOS text exactly as before, so the model runs exactly one BOS."""
        tok = family_tokenizer(family)
        model = scripted_model(tok, CALL_REPLY[family])
        svc = make_service(model, tok)
        await svc.create_chat_completion(request(tool_choice="required"))
        ids = model.generate_calls[0]
        bos = tok.bos_token_id
        leading = 0
        while bos is not None and leading < len(ids) and ids[leading] == bos:
            leading += 1
        assert leading == bos_count


async def _sse(svc, req) -> list[dict]:
    events = []
    async for raw in svc.stream_chat_completion(req):
        body = raw.removeprefix("data: ").strip()
        if body and body != "[DONE]":
            events.append(json.loads(body))
    return events


class TestStreamed:
    @pytest.mark.parametrize("family", list(FAMILIES))
    async def test_a_streamed_call_is_one_tool_calls_delta(self, family):
        events = await _sse(served(family, CALL_REPLY[family]), request(stream=True))
        deltas = [e["choices"][0]["delta"] for e in events if e.get("choices")]
        content = "".join(d.get("content") or "" for d in deltas)
        assert content == "", "no part of the call may stream as content"
        calls = [d["tool_calls"] for d in deltas if "tool_calls" in d]
        assert len(calls) == 1 and calls[0][0]["index"] == 0
        assert calls[0][0]["function"]["name"] == "get_weather"
        assert args_of(calls[0][0]) == {"city": "Paris"}
        finishes = [e["choices"][0]["finish_reason"] for e in events
                    if e.get("choices") and e["choices"][0].get("finish_reason")]
        assert finishes == ["tool_calls"]

    async def test_content_streams_before_the_call(self):
        reply = "Checking now. " + CALL_REPLY["lfm2"]
        events = await _sse(served("lfm2", reply), request(stream=True))
        deltas = [e["choices"][0]["delta"] for e in events if e.get("choices")]
        assert "".join(d.get("content") or "" for d in deltas) == "Checking now. "
        assert any("tool_calls" in d for d in deltas)

    async def test_a_streamed_plain_answer_is_untouched(self):
        events = await _sse(served("llama3_json", "It is sunny."), request(stream=True))
        deltas = [e["choices"][0]["delta"] for e in events if e.get("choices")]
        assert "".join(d.get("content") or "" for d in deltas) == "It is sunny."
        assert not any("tool_calls" in d for d in deltas)

    async def test_a_streamed_forced_call(self):
        svc = served("hermes", '{"name": "get_weather", "arguments": {"city": "Paris"}}')
        events = await _sse(svc, request(stream=True, tool_choice="required"))
        deltas = [e["choices"][0]["delta"] for e in events if e.get("choices")]
        assert any("tool_calls" in d for d in deltas)

    def test_the_filter_holds_a_marker_split_across_pieces(self):
        f = ToolCallStreamFilter(plan(LFM2))
        out = [f.feed(p) for p in ["Hi <|tool", "_call_start|>[get", "_time()]<|tool_call_end|>"]]
        assert "".join(out) == "Hi "
        tail, calls = f.finish()
        assert tail is None and calls[0]["function"]["name"] == "get_time"

    def test_the_filter_returns_unparseable_held_text(self):
        f = ToolCallStreamFilter(plan(HERMES))
        assert f.feed("<tool_call> nonsense") == ""
        assert f.finish() == ("<tool_call> nonsense", [])

    def test_a_partial_marker_that_never_completes_is_released(self):
        f = ToolCallStreamFilter(plan(HERMES))
        assert f.feed("a <tool") == "a "
        assert f.feed("s>") == "<tools>"


# ---------------------------------------------------------------------------------------------
# Probes and continuous batching
# ---------------------------------------------------------------------------------------------


class _Context:
    def __init__(self) -> None:
        self.probes = [SimpleNamespace(windows=("last_user",))]
        self.span: Any = None
        self.reason: Any = "unset"

    def set_last_user_span(self, span, reason):
        self.span, self.reason = span, reason


class TestProbeSpan:
    """The `last_user` window is placed in the render that was SERVED — tools and tool-call
    history included — and a forced call's opening, which follows the render, is not in it."""

    @pytest.mark.parametrize("family", ["lfm2", "hermes"])
    @pytest.mark.parametrize("choice", ["auto", "required"])
    def test_the_span_resolves_on_a_tools_render(self, family, choice):
        from millm.services.prompt_encoding import encode_rendered_chat

        svc = served(family, "x")
        req = request(tool_choice=choice)
        prompt, tool_plan = svc._chat_prompt(req)
        ids = encode_rendered_chat(svc._tokenizer, prompt, return_tensors="pt")["input_ids"]
        ctx = _Context()
        svc._probe_note_last_user_span(ctx, req.messages, ids, None, tool_plan=tool_plan)
        assert ctx.reason is None, ctx.reason
        start, end = ctx.span
        text = svc._tokenizer.decode(ids[0][start:end].tolist())
        assert "Weather in Paris?" in text and "get_weather" not in text


class TestContinuousBatching:
    def test_a_tools_request_never_routes_to_the_manager(self):
        svc = served("lfm2", "x")
        svc._use_cbm = lambda: True
        svc._cbm_backend = SimpleNamespace(sampling_params_match=lambda *_a: True)
        kwargs = svc._cbm_route_kwargs(request())
        assert kwargs["has_tools"] is True
        assert svc._use_cbm_for_request(**kwargs) is False
        assert svc._cbm_route_kwargs(request(tool_choice="none"))["has_tools"] is False
        assert svc._use_cbm_for_request(**svc._cbm_route_kwargs(request(tools=None))) is True
