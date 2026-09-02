"""chat_template_kwargs must accept the shapes clients actually send.

Reported from Open WebUI on 2026-09-02: its "Add Custom Parameter" editor emits
a JSON ARRAY wrapping the object -- `[{"enable_thinking": false}]` -- and a
strict `dict[str, Any]` answered "Input should be a valid dictionary". The user
had no way to express a bare object through that field, so a correctly
configured client could not disable reasoning at all.

Every accepted form collapses to one dict before it reaches
apply_chat_template, so downstream code sees no new shapes.
"""

import pytest
from pydantic import ValidationError

from millm.api.schemas.openai import ChatCompletionRequest

MSGS = [{"role": "user", "content": "hi"}]


def _kwargs(value):
    return ChatCompletionRequest(
        model="m", messages=MSGS, chat_template_kwargs=value
    ).chat_template_kwargs


class TestAcceptedShapes:
    def test_plain_object(self):
        assert _kwargs({"enable_thinking": False}) == {"enable_thinking": False}

    def test_open_webui_array_wrapper(self):
        """The reported bug."""
        assert _kwargs([{"enable_thinking": False}]) == {"enable_thinking": False}

    def test_json_string(self):
        assert _kwargs('{"enable_thinking": false}') == {"enable_thinking": False}

    def test_json_string_containing_an_array(self):
        assert _kwargs('[{"enable_thinking": false}]') == {"enable_thinking": False}

    def test_multi_entry_array_merges_later_wins(self):
        assert _kwargs([{"a": 1}, {"b": 2}, {"a": 3}]) == {"a": 3, "b": 2}

    def test_none_and_blank_mean_nothing_was_asked_for(self):
        assert _kwargs(None) is None
        assert _kwargs("") is None
        assert _kwargs([]) is None


class TestRejectedShapes:
    """Liberal is not the same as silent. Real mistakes must still fail."""

    @pytest.mark.parametrize("bad", [5, True, ["not-an-object"], [1, 2], "not json"])
    def test_nonsense_is_refused(self, bad):
        with pytest.raises(ValidationError):
            _kwargs(bad)

    def test_the_error_says_what_it_got(self):
        with pytest.raises(ValidationError) as e:
            _kwargs(5)
        # A message naming the expected shape is the whole point -- the original
        # "Input should be a valid dictionary" told the reporter nothing.
        assert "enable_thinking" in str(e.value)
