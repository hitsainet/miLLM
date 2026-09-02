"""Reasoning/answer separation, including the case that actually bit us.

granite-4.2-8b puts the opening `<think>` in the GENERATION PROMPT, so its
completion carries reasoning, then `</think>`, then the answer -- with no
opening tag. Every naive `<think>(.*?)</think>` implementation returns nothing
for that shape and leaves the trace in `content`.

The inverse error is worse and is tested harder: a non-reasoning model must
never have its answer moved into `reasoning_content`, which would return empty
content and look like a broken model.
"""

import pytest

from millm.services.reasoning_split import (
    StreamingReasoningSplitter,
    split_reasoning,
)


class TestNonStreaming:
    def test_granite_shape_closing_tag_only(self):
        r, c = split_reasoning("deliberating here</think>THE ANSWER",
                               prompt_opened_think=True)
        assert r == "deliberating here"
        assert c == "THE ANSWER"

    def test_closing_tag_only_is_IGNORED_without_evidence(self):
        """No prompt flag, no opening tag -> not our business to reinterpret."""
        r, c = split_reasoning("a</think>b", prompt_opened_think=False)
        assert r is None
        assert c == "a</think>b"

    def test_ordinary_tagged_block(self):
        r, c = split_reasoning("<think>why</think>answer")
        assert (r, c) == ("why", "answer")

    def test_plain_answer_is_never_reclassified(self):
        r, c = split_reasoning("just an answer")
        assert r is None and c == "just an answer"

    def test_plain_answer_survives_even_when_prompt_opened_think(self):
        """Truncated mid-reasoning: no answer exists, and content must not be
        invented -- but the trace must not be silently dropped either."""
        r, c = split_reasoning("still thinking", prompt_opened_think=True)
        assert r == "still thinking"
        assert c == ""

    def test_unclosed_block_keeps_preceding_content(self):
        r, c = split_reasoning("prefix<think>cut off")
        assert r == "cut off"
        assert c == "prefix"

    def test_empty_and_none(self):
        assert split_reasoning("") == (None, "")
        assert split_reasoning(None) == (None, None)


class TestStreaming:
    def _run(self, tokens, prompt_opened_think=False):
        s = StreamingReasoningSplitter(prompt_opened_think)
        reasoning, content = "", ""
        for t in tokens:
            r, c = s.feed(t)
            reasoning += r or ""
            content += c or ""
        r, c = s.flush()
        return reasoning + (r or ""), content + (c or "")

    def test_closing_tag_split_across_tokens(self):
        """The reason a fixed tail is withheld: tokenisers break the tag up."""
        r, c = self._run(["think", "ing", "</th", "ink>", "AN", "SWER"],
                         prompt_opened_think=True)
        assert r == "thinking"
        assert c == "ANSWER"

    def test_tagged_stream(self):
        r, c = self._run(["<th", "ink>", "why", "</think>", "ans"])
        assert r == "why"
        assert c == "ans"

    def test_non_reasoning_stream_is_all_content(self):
        r, c = self._run(["Hello", " ", "world"])
        assert r == ""
        assert c == "Hello world"

    def test_no_content_is_lost_for_any_split(self):
        """Property: reasoning+content reconstructs the payload, tag aside."""
        full = "abc</think>xyz"
        for i in range(1, len(full)):
            r, c = self._run([full[:i], full[i:]], prompt_opened_think=True)
            assert r + c == "abcxyz", f"lost data splitting at {i}: {r!r},{c!r}"


class TestServiceWiring:
    """The module can be perfect and the server still ship raw traces.

    These exercise InferenceService's own helpers, so deleting the wiring turns
    them red rather than leaving the module's own tests green.
    """

    def _svc(self):
        from millm.services.inference_service import InferenceService
        return InferenceService.__new__(InferenceService)

    def test_prompt_opened_think_detects_granite_generation_prompt(self):
        svc = self._svc()
        assert svc._prompt_opened_think("<|im_start|>assistant\n<think>\n")
        assert svc._prompt_opened_think("<|im_start|>assistant\n<think>")
        # Thinking disabled renders a CLOSED pair -- must read as not-open, or
        # every non-thinking reply gets classified as reasoning.
        assert not svc._prompt_opened_think(
            "<|im_start|>assistant\n<think></think>")
        assert not svc._prompt_opened_think("plain prompt")
        assert not svc._prompt_opened_think(None)

    def test_assistant_message_splits_the_granite_shape(self):
        svc = self._svc()
        msg = svc._assistant_message(
            "weighing it up</think>FINAL",
            prompt="<|im_start|>assistant\n<think>\n",
        )
        assert msg.content == "FINAL"
        assert msg.reasoning_content == "weighing it up"
        assert msg.role == "assistant"

    def test_assistant_message_leaves_a_normal_reply_alone(self):
        svc = self._svc()
        msg = svc._assistant_message("just the answer", prompt="plain")
        assert msg.content == "just the answer"
        assert msg.reasoning_content is None

    def test_reasoning_content_is_omitted_from_the_wire_when_absent(self):
        """exclude_none keeps the payload byte-identical for old clients."""
        svc = self._svc()
        msg = svc._assistant_message("hi", prompt="plain")
        assert "reasoning_content" not in msg.model_dump_json(exclude_none=True)
