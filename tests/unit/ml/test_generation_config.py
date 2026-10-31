"""Unit tests for GenerationConfig."""

import pytest

from millm.api.schemas.openai import ChatCompletionRequest, ChatMessage, TextCompletionRequest
from millm.ml.generation_config import GenerationConfig


# =============================================================================
# Fixtures
# =============================================================================


def _chat_request(**kwargs) -> ChatCompletionRequest:
    defaults = dict(
        model="test",
        messages=[ChatMessage(role="user", content="hi")],
        temperature=1.0,
        top_p=1.0,
        max_tokens=None,
        stop=None,
        frequency_penalty=0.0,
        presence_penalty=0.0,
    )
    defaults.update(kwargs)
    return ChatCompletionRequest(**defaults)


def _text_request(**kwargs) -> TextCompletionRequest:
    defaults = dict(
        model="test",
        prompt="hello",
        temperature=1.0,
        top_p=1.0,
        max_tokens=None,
        stop=None,
        frequency_penalty=0.0,
        presence_penalty=0.0,
    )
    defaults.update(kwargs)
    return TextCompletionRequest(**defaults)


# =============================================================================
# Tests: defaults
# =============================================================================


class TestGenerationConfigDefaults:
    def test_default_max_new_tokens(self):
        assert GenerationConfig().max_new_tokens == 512

    def test_default_temperature(self):
        assert GenerationConfig().temperature == 1.0

    def test_default_do_sample(self):
        assert GenerationConfig().do_sample is True

    def test_default_no_stop_sequences(self):
        assert GenerationConfig().stop_sequences is None

    def test_default_cache_implementation_is_none(self):
        assert GenerationConfig().cache_implementation is None


# =============================================================================
# Tests: from_request (ChatCompletionRequest)
# =============================================================================


class TestFromChatCompletionRequest:
    def test_minimal_request_uses_defaults(self):
        cfg = GenerationConfig.from_request(_chat_request())
        assert cfg.max_new_tokens == 512  # max_tokens=None → 512
        assert cfg.do_sample is True       # temperature > 0

    def test_max_tokens_mapped(self):
        cfg = GenerationConfig.from_request(_chat_request(max_tokens=100))
        assert cfg.max_new_tokens == 100

    def test_temperature_zero_disables_sampling(self):
        cfg = GenerationConfig.from_request(_chat_request(temperature=0.0))
        assert cfg.do_sample is False
        assert cfg.temperature == 0.0

    def test_temperature_nonzero_enables_sampling(self):
        cfg = GenerationConfig.from_request(_chat_request(temperature=0.5))
        assert cfg.do_sample is True

    def test_top_p_mapped(self):
        cfg = GenerationConfig.from_request(_chat_request(top_p=0.9))
        assert cfg.top_p == 0.9

    def test_stop_string_becomes_single_item_list(self):
        cfg = GenerationConfig.from_request(_chat_request(stop="END"))
        assert cfg.stop_sequences == ["END"]

    def test_stop_list_preserved(self):
        cfg = GenerationConfig.from_request(_chat_request(stop=["END", "STOP"]))
        assert cfg.stop_sequences == ["END", "STOP"]

    def test_stop_none_stays_none(self):
        cfg = GenerationConfig.from_request(_chat_request(stop=None))
        assert cfg.stop_sequences is None

    def test_frequency_penalty_mapped(self):
        cfg = GenerationConfig.from_request(_chat_request(frequency_penalty=1.0))
        assert cfg.frequency_penalty == 1.0

    def test_presence_penalty_mapped(self):
        cfg = GenerationConfig.from_request(_chat_request(presence_penalty=0.5))
        assert cfg.presence_penalty == 0.5


# =============================================================================
# Tests: from_request (TextCompletionRequest)
# =============================================================================


class TestFromTextCompletionRequest:
    def test_text_request_defaults(self):
        cfg = GenerationConfig.from_request(_text_request())
        assert cfg.max_new_tokens == 512
        assert cfg.do_sample is True

    def test_text_request_max_tokens(self):
        cfg = GenerationConfig.from_request(_text_request(max_tokens=64))
        assert cfg.max_new_tokens == 64

    def test_text_request_temperature_zero(self):
        cfg = GenerationConfig.from_request(_text_request(temperature=0.0))
        assert cfg.do_sample is False


# =============================================================================
# Tests: to_generate_kwargs
# =============================================================================


class TestToGenerateKwargs:
    def test_always_includes_max_new_tokens(self):
        kwargs = GenerationConfig(max_new_tokens=200).to_generate_kwargs()
        assert kwargs["max_new_tokens"] == 200

    def test_sampling_includes_temperature_and_top_p(self):
        kwargs = GenerationConfig(temperature=0.7, top_p=0.9, do_sample=True).to_generate_kwargs()
        assert kwargs["temperature"] == 0.7
        assert kwargs["top_p"] == 0.9
        assert kwargs["do_sample"] is True

    def test_greedy_excludes_temperature_and_top_p(self):
        kwargs = GenerationConfig(do_sample=False).to_generate_kwargs()
        assert "temperature" not in kwargs
        assert "top_p" not in kwargs
        assert kwargs["do_sample"] is False

    def test_positive_frequency_penalty_raises_repetition_penalty(self):
        kwargs = GenerationConfig(frequency_penalty=2.0, do_sample=False).to_generate_kwargs()
        assert "repetition_penalty" in kwargs
        assert kwargs["repetition_penalty"] > 1.0

    def test_negative_frequency_penalty_lowers_repetition_penalty(self):
        kwargs = GenerationConfig(frequency_penalty=-1.0, do_sample=False).to_generate_kwargs()
        rp = kwargs.get("repetition_penalty", 1.0)
        assert rp < 1.0

    def test_zero_frequency_penalty_omits_repetition_penalty(self):
        kwargs = GenerationConfig(frequency_penalty=0.0, presence_penalty=0.0, do_sample=False).to_generate_kwargs()
        assert "repetition_penalty" not in kwargs

    def test_presence_penalty_adds_repetition_penalty_when_no_frequency(self):
        kwargs = GenerationConfig(presence_penalty=1.0, frequency_penalty=0.0, do_sample=False).to_generate_kwargs()
        assert "repetition_penalty" in kwargs
        assert kwargs["repetition_penalty"] > 1.0

    def test_both_penalties_combine_additively(self):
        # Both penalties are combined: combined = freq + 0.5 * presence
        # frequency=1.0, presence=1.0 → combined=1.5 → 1.0 + 1.5*0.25 = 1.375
        kwargs = GenerationConfig(
            frequency_penalty=1.0, presence_penalty=1.0, do_sample=False
        ).to_generate_kwargs()
        assert abs(kwargs["repetition_penalty"] - 1.375) < 0.01

    def test_static_cache_included_when_set(self):
        kwargs = GenerationConfig(cache_implementation="static").to_generate_kwargs()
        assert kwargs["cache_implementation"] == "static"

    def test_dynamic_cache_omits_cache_implementation(self):
        kwargs = GenerationConfig(cache_implementation=None).to_generate_kwargs()
        assert "cache_implementation" not in kwargs


# =============================================================================
# Tests: with_max_tokens / with_stop_sequences
# =============================================================================


class TestWithHelpers:
    def test_with_max_tokens_returns_new_instance(self):
        original = GenerationConfig(max_new_tokens=512)
        updated = original.with_max_tokens(100)
        assert updated.max_new_tokens == 100
        assert original.max_new_tokens == 512  # unchanged

    def test_with_max_tokens_preserves_other_fields(self):
        original = GenerationConfig(temperature=0.5, top_p=0.8)
        updated = original.with_max_tokens(100)
        assert updated.temperature == 0.5
        assert updated.top_p == 0.8

    def test_with_stop_sequences_returns_new_instance(self):
        original = GenerationConfig(stop_sequences=["OLD"])
        updated = original.with_stop_sequences(["NEW"])
        assert updated.stop_sequences == ["NEW"]
        assert original.stop_sequences == ["OLD"]

    def test_with_stop_sequences_none_clears(self):
        original = GenerationConfig(stop_sequences=["a", "b"])
        updated = original.with_stop_sequences(None)
        assert updated.stop_sequences is None
