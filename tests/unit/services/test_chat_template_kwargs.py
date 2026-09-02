"""Chat-template kwargs pass-through, and its refusal to fail quietly.

`enable_thinking` lives in the model's Jinja template, not in the generation
config, so the only route to it is `apply_chat_template(**kwargs)`. Measured on
granite-4.2-8b (2026-09-02): the template defaults `enable_thinking` to True and
its generation prompt then ends with an OPEN `<think>` tag, so the model resumes
inside a reasoning block. The opening tag is in the PROMPT, which is why a
client stripping `<think>...</think>` from the RESPONSE removes nothing and the
deliberation arrives as untagged prose inside the parsed answer.

The load-bearing behaviour here is the failure mode, not the happy path. Before
this change `_format_chat_messages` swallowed every template exception and fell
back to a generic Gemma-style format. Reaching that fallback after an explicit
kwargs request is wrong twice over — the model is formatted for the wrong family
AND the request is discarded — and the caller still receives a 200 with
reasoning switched on. These tests pin that such a request raises instead.
"""

from unittest.mock import MagicMock

import pytest

from millm.api.schemas.openai import ChatMessage
from millm.services.inference_service import InferenceService


def _svc(tokenizer):
    """`_tokenizer` is a read-only property over the loaded-model state, so the
    fake is installed where that property reads from rather than over it."""
    svc = InferenceService.__new__(InferenceService)
    state = MagicMock()
    state.is_loaded = True
    state.current.tokenizer = tokenizer
    svc._model_state = state
    assert svc._tokenizer is tokenizer      # the fake is actually in place
    return svc


def _messages():
    return [ChatMessage(role="user", content="hi")]


def test_kwargs_are_forwarded_verbatim_to_the_template():
    tok = MagicMock()
    tok.chat_template = "a-template"
    tok.apply_chat_template.return_value = "RENDERED"

    out = _svc(tok)._format_chat_messages(
        _messages(), {"enable_thinking": False, "reasoning_effort": "low"}
    )

    assert out == "RENDERED"
    kwargs = tok.apply_chat_template.call_args.kwargs
    # Assert the PAYLOAD, not merely that the call happened: a pass-through
    # that drops or renames the flag still "was called".
    assert kwargs["enable_thinking"] is False
    assert kwargs["reasoning_effort"] == "low"
    assert kwargs["add_generation_prompt"] is True
    assert kwargs["tokenize"] is False


def test_no_kwargs_leaves_existing_behaviour_untouched():
    tok = MagicMock()
    tok.chat_template = "a-template"
    tok.apply_chat_template.return_value = "RENDERED"

    assert _svc(tok)._format_chat_messages(_messages()) == "RENDERED"
    assert "enable_thinking" not in tok.apply_chat_template.call_args.kwargs


def test_template_error_with_kwargs_raises_instead_of_silent_fallback():
    """The whole point. A silent fallback here means thinking stays ON."""
    tok = MagicMock()
    tok.chat_template = "a-template"
    tok.apply_chat_template.side_effect = RuntimeError("bad template")

    # Match the message of THIS guard, not just the kwarg name. The
    # no-chat-template guard below also raises ValueError and also interpolates
    # sorted(template_kwargs), so a `match="enable_thinking"` passes when this
    # guard is deleted and execution falls through to that one — the two mask
    # each other and the control survives.
    with pytest.raises(ValueError, match="chat template rejected"):
        _svc(tok)._format_chat_messages(_messages(), {"enable_thinking": False})


def test_template_error_without_kwargs_still_falls_back():
    """Pre-existing resilience must survive: no request, no new failure."""
    tok = MagicMock()
    tok.chat_template = "a-template"
    tok.apply_chat_template.side_effect = RuntimeError("bad template")

    out = _svc(tok)._format_chat_messages(_messages())
    assert "hi" in out            # the Gemma-style fallback rendered something
    assert out != "RENDERED"


def test_model_without_a_chat_template_refuses_kwargs():
    """Otherwise the fallback format silently discards the request."""
    tok = MagicMock()
    tok.chat_template = None

    with pytest.raises(ValueError, match="no chat template"):
        _svc(tok)._format_chat_messages(_messages(), {"enable_thinking": False})

    # ...but is still perfectly usable when nothing was asked for.
    assert "hi" in _svc(tok)._format_chat_messages(_messages())
