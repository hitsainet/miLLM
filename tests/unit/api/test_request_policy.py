"""The request policy module itself (Feature 25, FR-25.1 – FR-25.3): table, neutral values,
presence, locations, strict parsing, header encoding.

The HTTP behaviour lives in test_unused_fields_http.py and test_request_policy_coverage.py;
this file pins the pure rules.

MUTATION CONTROLS: M1 (strict branch skipped), M3 (message walk removed) — see
0xcc/reviews/025_implementation_controls_2026-10-06.md.
"""

from __future__ import annotations

import pytest

from millm.api import request_policy as rp
from millm.api.request_policy import (
    ENGINE_UNUSED,
    NEUTRAL,
    OUTPUT_CHANGING,
    Endpoint,
    Engine,
    Honoured,
    Refused,
    encode_field_list,
    evaluate,
    parse_strict,
)
from millm.api.schemas.openai import (
    ChatCompletionRequest,
    EmbeddingRequest,
    TextCompletionRequest,
)
from millm.core.errors import (
    FieldNotHonouredError,
    InvalidParameterError,
    UnusedFieldsRefusedError,
)

SCHEMAS = {
    Endpoint.CHAT: ChatCompletionRequest,
    Endpoint.COMPLETIONS: TextCompletionRequest,
    Endpoint.EMBEDDINGS: EmbeddingRequest,
}


def chat(**over) -> ChatCompletionRequest:
    body = {"model": "m", "messages": [{"role": "user", "content": "hi"}]}
    body.update(over)
    return ChatCompletionRequest(**body)


class TestTheTableIsComplete:
    def test_every_field_has_every_cell(self):
        """No default outcome exists, so a missing cell would be a KeyError at request time."""
        expected = {(e, g) for e in Endpoint for g in Engine}
        for name, cells in OUTPUT_CHANGING.items():
            assert set(cells) == expected, name
            for outcome in cells.values():
                assert isinstance(outcome, (Honoured, Refused)), name
                if isinstance(outcome, Refused):
                    assert outcome.reason.strip(), name

    def test_the_fprd_minimum_list_is_present(self):
        """FR-25.3: at least these — and max_completion_tokens (FR-25.3.3a)."""
        required = {"response_format", "seed", "logprobs", "top_logprobs", "allowed_token_ids",
                    "n", "dimensions", "tools", "tool_choice", "logit_bias", "steering",
                    "max_completion_tokens"}
        assert required <= set(OUTPUT_CHANGING)

    def test_an_honoured_cell_is_declared_by_that_endpoints_schema(self):
        """Honouring needs a schema field to read: an honoured but undeclared field would be
        dropped while the table says honoured."""
        for name, cells in OUTPUT_CHANGING.items():
            for (endpoint, _engine), outcome in cells.items():
                if isinstance(outcome, Honoured):
                    assert name in SCHEMAS[endpoint].model_fields, (name, endpoint)

    @pytest.mark.parametrize("field, endpoint, engine, honoured", [
        ("tools", Endpoint.CHAT, Engine.TRANSFORMERS, False),
        ("logit_bias", Endpoint.CHAT, Engine.TRANSFORMERS, False),
        ("dimensions", Endpoint.EMBEDDINGS, Engine.TRANSFORMERS, False),
        ("steering", Endpoint.CHAT, Engine.TRANSFORMERS, True),
        ("steering", Endpoint.COMPLETIONS, Engine.TRANSFORMERS, True),
        ("steering", Endpoint.CHAT, Engine.LLAMACPP, False),
        ("steering", Endpoint.COMPLETIONS, Engine.LLAMACPP, False),
        ("steering", Endpoint.EMBEDDINGS, Engine.TRANSFORMERS, False),
        ("profile", Endpoint.COMPLETIONS, Engine.TRANSFORMERS, True),
        ("steering_intensity", Endpoint.COMPLETIONS, Engine.TRANSFORMERS, True),
        ("n", Endpoint.COMPLETIONS, Engine.TRANSFORMERS, False),
        ("n", Endpoint.CHAT, Engine.LLAMACPP, False),
        ("n", Endpoint.CHAT, Engine.TRANSFORMERS, True),
        ("logprobs", Endpoint.COMPLETIONS, Engine.LLAMACPP, False),
        ("top_logprobs", Endpoint.COMPLETIONS, Engine.TRANSFORMERS, False),
        ("max_completion_tokens", Endpoint.EMBEDDINGS, Engine.TRANSFORMERS, False),
        ("tool_choice", Endpoint.COMPLETIONS, Engine.LLAMACPP, False),
        ("max_completion_tokens", Endpoint.CHAT, Engine.LLAMACPP, True),
        ("max_completion_tokens", Endpoint.COMPLETIONS, Engine.TRANSFORMERS, True),
    ])
    def test_fprd_table_cells(self, field, endpoint, engine, honoured):
        assert isinstance(OUTPUT_CHANGING[field][(endpoint, engine)], Honoured) is honoured

    @pytest.mark.parametrize("field", ["steering", "profile", "steering_intensity"])
    @pytest.mark.parametrize("endpoint", [Endpoint.CHAT, Endpoint.COMPLETIONS])
    def test_steering_fields_are_honoured_except_on_scoring(self, field, endpoint):
        """Feature 28 flipped FR-25.3.7 (FR-28.4.5): honoured on transformers chat and
        completions, still refused on a scoring request (X-09, FR-25.7.2)."""
        outcome = OUTPUT_CHANGING[field][(endpoint, Engine.TRANSFORMERS)]
        assert isinstance(outcome, Honoured) and outcome.refuse_if is not None

        class _Scoring:
            def wants_scores(self):
                return True

        class _Generating:
            def wants_scores(self):
                return False

        assert "X-09" in outcome.refuse_if(_Scoring())
        assert outcome.refuse_if(_Generating()) is None

    @pytest.mark.parametrize("engine", list(Engine))
    def test_dimensions_is_refused_for_want_of_a_declaration(self, engine):
        """Feature 30 replaced "refused until Feature 30" (FR-30.1.7). Under T-91 no model
        declares truncated-embedding support, so the cell is refused on both engines and says
        why — never honoured-and-ignored."""
        outcome = OUTPUT_CHANGING["dimensions"][(Endpoint.EMBEDDINGS, engine)]
        assert isinstance(outcome, Refused)
        assert "truncated-embedding support" in outcome.reason and "T-91" in outcome.reason
        assert "dimensions" not in NEUTRAL, "dimensions has no neutral value (FR-30.1.6)"


class TestNeutralValues:
    @pytest.mark.parametrize("field, value, neutral", [
        ("n", 1, True), ("n", 2, False), ("n", True, False),
        ("logprobs", False, True), ("logprobs", True, False),
        ("response_format", {"type": "text"}, True),
        ("response_format", {"type": "json_object"}, False),
        ("tools", [], True), ("tools", [{"type": "function"}], False),
        ("logit_bias", {}, True), ("logit_bias", {"5": 1}, False),
    ])
    def test_values(self, field, value, neutral):
        assert NEUTRAL[field](value) is neutral

    def test_a_neutral_value_of_a_refused_field_passes(self):
        result = evaluate(chat(tools=[], logit_bias={}), Endpoint.CHAT, Engine.TRANSFORMERS,
                          strict=True)
        assert result.unused == []

    def test_a_zero_value_with_no_neutral_entry_is_refused(self):
        """FR-25.3.4: a refusal fires even at the field's zero value unless it is listed neutral."""
        with pytest.raises(FieldNotHonouredError) as exc:
            evaluate(chat(tool_choice="none"), Endpoint.CHAT, Engine.TRANSFORMERS, strict=False)
        assert exc.value.details["param"] == "tool_choice"

    def test_an_explicit_null_is_not_a_presence(self):
        evaluate(chat(tools=None, logit_bias=None), Endpoint.CHAT, Engine.TRANSFORMERS,
                 strict=True)


class TestPresenceMeansSent:
    def test_a_default_n_is_not_a_presence(self):
        """n defaults to 1 on chat; on llama.cpp `n` is refused for n>1 — the default must never
        count as the client having sent it."""
        req = chat()
        assert "n" not in req.model_fields_set
        evaluate(req, Endpoint.CHAT, Engine.LLAMACPP, strict=True)

    def test_n_2_on_llamacpp_is_refused(self):
        with pytest.raises(FieldNotHonouredError) as exc:
            evaluate(chat(n=2), Endpoint.CHAT, Engine.LLAMACPP, strict=False)
        assert exc.value.details["param"] == "n"

    def test_every_refused_field_is_named(self):
        with pytest.raises(FieldNotHonouredError) as exc:
            evaluate(chat(tools=[{"x": 1}], logit_bias={"1": 2}), Endpoint.CHAT,
                     Engine.TRANSFORMERS, strict=False)
        assert exc.value.details["fields"] == ["tools", "logit_bias"]
        assert "tools" in exc.value.message and "logit_bias" in exc.value.message


class TestLocations:
    def test_top_level_message_and_extra_message_locations_in_order(self):
        req = chat(
            foo=1,
            messages=[{"role": "user", "content": "a"},
                      {"role": "assistant", "content": "b", "name": "x"}],
            extra_messages=[[{"role": "user", "content": "c", "weight": 2}]],
        )
        result = evaluate(req, Endpoint.CHAT, Engine.TRANSFORMERS, strict=False)
        assert result.unused == ["foo", "messages[1].name", "extra_messages[0][0].weight"]

    def test_no_duplicates(self):
        req = chat(foo=1)
        assert evaluate(req, Endpoint.CHAT, Engine.TRANSFORMERS, strict=False).unused == ["foo"]

    def test_chat_template_kwargs_is_unused_on_llamacpp_only(self):
        """FR-25.1.2 case (b): declared, but the llama.cpp path cannot consume it."""
        assert ENGINE_UNUSED[(Endpoint.CHAT, Engine.LLAMACPP)] == {"chat_template_kwargs"}
        req = chat(chat_template_kwargs={"enable_thinking": False})
        assert evaluate(req, Endpoint.CHAT, Engine.LLAMACPP, strict=False).unused == [
            "chat_template_kwargs"]
        assert evaluate(req, Endpoint.CHAT, Engine.TRANSFORMERS, strict=False).unused == []

    def test_an_undeclared_list_field_on_an_honoured_cell_fails_closed(self, monkeypatch):
        """A listed field arriving as an extra where the table says honoured would be dropped
        while the table claims it is served — refused instead (a negative control found this
        branch untested: control P-fail-closed)."""
        cells = dict(OUTPUT_CHANGING["tools"])
        cells[(Endpoint.CHAT, Engine.TRANSFORMERS)] = rp.HONOURED
        monkeypatch.setitem(OUTPUT_CHANGING, "tools", cells)
        with pytest.raises(FieldNotHonouredError) as exc:
            evaluate(chat(tools=[{"type": "function"}]), Endpoint.CHAT, Engine.TRANSFORMERS,
                     strict=False)
        assert exc.value.details["param"] == "tools"
        assert "not implemented on this endpoint" in exc.value.message


class TestStrict:
    @pytest.mark.parametrize("value, on", [
        (None, False), ("true", True), ("TRUE", True), ("1", True), (" True ", True),
        ("false", False), ("0", False), ("False", False),
    ])
    def test_values(self, value, on):
        assert parse_strict(value) is on

    @pytest.mark.parametrize("value", ["yes", "on", "", "2", "tru"])
    def test_any_other_value_is_refused_naming_the_header(self, value):
        with pytest.raises(InvalidParameterError) as exc:
            parse_strict(value)
        assert exc.value.details["param"] == "X-miLLM-Strict"
        assert exc.value.status_code == 400

    def test_strict_refuses_naming_every_location(self):
        req = chat(foo=1, bar=2, messages=[{"role": "user", "content": "a", "name": "n"}])
        with pytest.raises(UnusedFieldsRefusedError) as exc:
            evaluate(req, Endpoint.CHAT, Engine.TRANSFORMERS, strict=True)
        assert exc.value.details["fields"] == ["foo", "bar", "messages[0].name"]
        assert exc.value.details["param"] == "foo"
        for loc in ("foo", "bar", "messages[0].name"):
            assert loc in exc.value.message

    def test_without_strict_the_same_request_is_reported(self):
        req = chat(foo=1)
        assert evaluate(req, Endpoint.CHAT, Engine.TRANSFORMERS, strict=False).unused == ["foo"]


class TestUserIsUnused:
    def test_user_is_reported_and_strict_refuses_it(self):
        """T-57 / FR-25.3.3b: `user` is never read, so it is unused."""
        req = chat(user="alice")
        assert evaluate(req, Endpoint.CHAT, Engine.TRANSFORMERS, strict=False).unused == ["user"]
        with pytest.raises(UnusedFieldsRefusedError):
            evaluate(req, Endpoint.CHAT, Engine.TRANSFORMERS, strict=True)


class TestHeaderEncoding:
    def test_sf_string_list(self):
        assert encode_field_list(["foo", "messages[2].name"], 1024) == '"foo", "messages[2].name"'

    def test_quotes_and_backslashes_are_escaped(self):
        assert encode_field_list(['a"b\\c'], 1024) == '"a\\"b\\\\c"'

    def test_crlf_is_percent_encoded_so_it_cannot_split_the_header(self):
        value = encode_field_list(["evil\r\nSet-Cookie: x=1"], 1024)
        assert "\r" not in value and "\n" not in value
        assert value == '"evil%0D%0ASet-Cookie: x=1"'

    def test_non_ascii_and_percent_are_percent_encoded(self):
        assert encode_field_list(["é%"], 1024) == '"%C3%A9%25"'
        assert all(0x20 <= ord(c) <= 0x7E for c in encode_field_list(["日本"], 1024))

    def test_bounded_with_a_count_of_what_was_dropped(self):
        locations = [f"field_{i:03d}" for i in range(200)]
        value = encode_field_list(locations, 1024)
        assert len(value) <= 1024
        assert value.endswith(" more\"")
        kept = value.count('"field_')
        assert f'"+{200 - kept} more"' in value

    def test_nothing_dropped_when_it_fits(self):
        assert "more" not in encode_field_list(["a", "b"], 1024)


class TestTheLogCarriesLocationsNeverValues:
    def test_privacy_sentinel(self):
        """FR-25.1.8 (M4): a field's value may be prompt text."""
        import structlog

        sentinel = "SENTINEL-PROMPT-TEXT-91f3"
        req = chat(foo=sentinel, messages=[{"role": "user", "content": "a", "name": sentinel}])
        with structlog.testing.capture_logs() as logs:
            rp.apply_request_policy(req, Endpoint.CHAT, rp_row(), {})
        events = [e for e in logs if e["event"] == "request_fields_unused"]
        assert len(events) == 1
        assert events[0]["fields"] == ["foo", "messages[0].name"]
        assert events[0]["endpoint"] == "chat" and events[0]["log_level"] == "warning"
        assert events[0]["request_id"]
        assert sentinel not in repr(logs)


def rp_row(gguf=None):
    from types import SimpleNamespace

    return SimpleNamespace(gguf_files=gguf)


def test_engine_comes_from_the_row():
    assert rp.engine_of(rp_row()) is Engine.TRANSFORMERS
    assert rp.engine_of(rp_row(["m.gguf"])) is Engine.LLAMACPP


def test_policy_cost_on_a_50_message_request_is_small():
    """2.10 / FPRD §8: no forward pass, no database read — a walk over model_extra. Measured and
    bounded loosely here; the measured figure is recorded in the controls record."""
    import time

    req = chat(
        foo=1,
        messages=[{"role": "user", "content": "x" * 200, "name": f"n{i}"} for i in range(50)],
    )
    t0 = time.perf_counter()
    for _ in range(200):
        evaluate(req, Endpoint.CHAT, Engine.TRANSFORMERS, strict=False)
    per_call = (time.perf_counter() - t0) / 200
    assert per_call < 0.01, per_call


def test_the_table_refuses_response_format_on_scoring_for_a_direct_caller():
    """Over HTTP the chat schema refuses this combination first, so only a caller that builds or
    copies a request object (Feature 26's batch lines call `evaluate`) reaches the table cell.
    Found by control C7-table-scoring surviving."""
    scoring = chat(max_tokens=1, logprobs=True)
    fmt = chat(response_format={"type": "json_object"}).response_format
    req = scoring.model_copy(update={"response_format": fmt})
    req.model_fields_set.add("response_format")
    with pytest.raises(FieldNotHonouredError) as exc:
        evaluate(req, Endpoint.CHAT, Engine.TRANSFORMERS, strict=False)
    assert exc.value.details["param"] == "response_format"
    plain = chat(response_format={"type": "json_object"})
    evaluate(plain, Endpoint.CHAT, Engine.TRANSFORMERS, strict=True)
