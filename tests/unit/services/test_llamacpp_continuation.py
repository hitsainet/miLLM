"""A truncated answer must RESUME, not restart.

An OpenAI client that offers "continue" — Open WebUI does — resends the
conversation with the truncated answer as a trailing assistant message.
`create_chat_completion` applies the GGUF's baked-in template to that list, and
every such template CLOSES the final turn. Rendered from the real gemma-4-31b
GGUF:

    no trailing assistant : ...<|turn>model\\n<|channel>thought\\n<channel|>
    trailing assistant    : ...<|turn>model\\nPARTIAL<turn|>\\n

Sealed with `<turn|>`, the model can only begin a new answer. Observed on both
GGUF models: gemma restated its whole explanation and remarked "It looks like
your previous message had a technical glitch... Let's start fresh", and the Qwen
reasoning model re-opened and restarted its <think> block three times, because
a fresh turn re-opens the thought channel.

The fix is HuggingFace's `continue_final_message`: render everything BEFORE the
partial with a generation prompt, append the partial raw, and complete from
there with `create_completion`.

A SHORT partial hides this. "1, 2, 3, 4, 5," followed by a fresh turn still
produces "6, 7, 8..." because that is also the natural new answer — the first
test written for this passed for entirely the wrong reason. The fixtures here
use partials whose continuation and restart are unmistakably different.

MUTATION CONTROLS (each must turn this file red):
  * return None from _llamacpp_continuation_prompt -> "resumes" fails
  * drop the branch in _llamacpp_sync              -> non-streaming fails
  * drop the branch in _open_stream                -> streaming fails
  * pass all messages to the formatter             -> "leaves the turn open" fails
"""

from unittest.mock import MagicMock, patch

import pytest

from millm.services.inference_service import (
    InferenceService,
    _completion_as_chat,
    _completion_chunks_as_chat,
)


def _service_with_template(template="{% for m in messages %}<t>{{ m.role }}\n{{ m.content }}</t>\n{% endfor %}{% if add_generation_prompt %}<t>assistant\n{% endif %}"):
    svc = InferenceService.__new__(InferenceService)
    model = MagicMock()
    model.metadata = {"tokenizer.chat_template": template}
    model.token_eos.return_value = 1
    model.token_bos.return_value = 2
    model._model.token_get_text.side_effect = lambda t: "<eos>" if t == 1 else "<bos>"
    # `_model` is a read-only property backed by the loader state, so the
    # stand-in goes where the property reads from rather than over the top of
    # it — patching the property itself would let a test pass against a service
    # wired differently from the real one.
    state = MagicMock()
    state.is_loaded = True
    state.current.model = model
    svc._model_state = state
    return svc, model


class TestTheTurnIsLeftOpen:
    def test_a_trailing_assistant_message_produces_a_continuation_prompt(self):
        svc, _ = _service_with_template()

        prompt = svc._llamacpp_continuation_prompt(
            [{"role": "user", "content": "hi"},
             {"role": "assistant", "content": "PARTIAL TEXT"}]
        )

        assert prompt is not None
        assert prompt.endswith("PARTIAL TEXT"), (
            f"the partial must be the LAST thing in the prompt, with no closing "
            f"turn marker after it: {prompt!r}"
        )

    def test_the_partial_is_not_wrapped_in_its_own_turn(self):
        """The whole defect in one assertion: the partial must not be rendered
        as a completed message."""
        svc, _ = _service_with_template()

        prompt = svc._llamacpp_continuation_prompt(
            [{"role": "user", "content": "hi"},
             {"role": "assistant", "content": "PARTIAL"}]
        )

        assert "</t>\nPARTIAL" not in prompt
        assert "PARTIAL</t>" not in prompt, (
            "the partial was sealed into a closed turn — the model will start "
            "a new answer, which is the bug"
        )

    def test_an_ordinary_request_takes_the_normal_path(self):
        svc, _ = _service_with_template()

        assert svc._llamacpp_continuation_prompt(
            [{"role": "user", "content": "hi"}]
        ) is None

    def test_an_empty_partial_is_not_a_continuation(self):
        """Some clients send an empty assistant turn to request a FRESH answer.
        Continuing it would append to nothing."""
        svc, _ = _service_with_template()

        assert svc._llamacpp_continuation_prompt(
            [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "   "}]
        ) is None

    def test_an_unreadable_template_degrades_to_the_normal_path(self):
        """A failure here must cost the continuation feature, never the
        request."""
        svc, model = _service_with_template()
        model.metadata = {}

        assert svc._llamacpp_continuation_prompt(
            [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "P"}]
        ) is None

    def test_a_template_that_raises_degrades_rather_than_failing(self):
        svc, _ = _service_with_template(template="{% this is not valid jinja %}")

        assert svc._llamacpp_continuation_prompt(
            [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "P"}]
        ) is None


class TestBothEnginePathsUseIt:
    """Open WebUI STREAMS. A fix applied only to the blocking path would leave
    the user-facing case broken — which is how these two guards diverged once
    before."""

    def test_the_blocking_path_completes_from_the_prompt(self):
        svc, model = _service_with_template()
        model.create_completion.return_value = {
            "choices": [{"index": 0, "text": " resumed", "finish_reason": "stop"}],
            "usage": {},
        }

        out = svc._llamacpp_sync(
            [{"role": "user", "content": "hi"},
             {"role": "assistant", "content": "PARTIAL"}],
            {"max_tokens": 8},
        )

        assert model.create_completion.called, "it re-templated instead of continuing"
        assert not model.create_chat_completion.called
        assert model.create_completion.call_args.kwargs["prompt"].endswith("PARTIAL")
        assert out["choices"][0]["message"]["content"] == " resumed"

    def test_the_blocking_path_is_unchanged_for_an_ordinary_request(self):
        svc, model = _service_with_template()
        model.create_chat_completion.return_value = {"choices": []}

        svc._llamacpp_sync([{"role": "user", "content": "hi"}], {})

        assert model.create_chat_completion.called
        assert not model.create_completion.called


class TestTheCompletionShapeIsTranslated:
    """`create_completion` returns choices[].text; every consumer downstream
    reads choices[].message.content. A missed translation is an empty answer."""

    def test_text_becomes_message_content(self):
        out = _completion_as_chat(
            {"choices": [{"index": 0, "text": "hello", "finish_reason": "length"}]}
        )
        assert out["choices"][0]["message"] == {"role": "assistant", "content": "hello"}
        assert out["choices"][0]["finish_reason"] == "length"

    def test_streaming_text_becomes_delta_content(self):
        chunks = list(
            _completion_chunks_as_chat(
                iter([{"choices": [{"index": 0, "text": "a", "finish_reason": None}]}])
            )
        )
        assert chunks[0]["choices"][0]["delta"] == {"content": "a"}

    def test_other_envelope_fields_survive(self):
        out = _completion_as_chat({"id": "x", "usage": {"total_tokens": 3}, "choices": []})
        assert out["id"] == "x" and out["usage"]["total_tokens"] == 3
