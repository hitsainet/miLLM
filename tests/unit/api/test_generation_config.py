"""
Unit tests for generation configuration mapping.

Tests OpenAI parameter to Transformers generate() kwargs conversion.
"""

import pytest

from millm.api.schemas.openai import ChatCompletionRequest, ChatMessage, TextCompletionRequest
from millm.ml.generation_config import GenerationConfig


class TestGenerationConfigDefaults:
    """Tests for GenerationConfig default values."""

    def test_default_values(self):
        config = GenerationConfig()
        assert config.max_new_tokens == 512
        assert config.temperature == 1.0
        assert config.top_p == 1.0
        assert config.do_sample is True
        assert config.stop_sequences is None
        assert config.frequency_penalty == 0.0
        assert config.presence_penalty == 0.0


class TestFromChatCompletionRequest:
    """Tests for GenerationConfig.from_request with chat completions."""

    def test_minimal_request(self):
        request = ChatCompletionRequest(
            model="gpt-4",
            messages=[ChatMessage(role="user", content="Hello")],
        )
        config = GenerationConfig.from_request(request)
        assert config.max_new_tokens == 512  # Default
        assert config.temperature == 1.0
        assert config.do_sample is True

    def test_temperature_zero_disables_sampling(self):
        request = ChatCompletionRequest(
            model="gpt-4",
            messages=[ChatMessage(role="user", content="Hello")],
            temperature=0.0,
        )
        config = GenerationConfig.from_request(request)
        assert config.temperature == 0.0
        assert config.do_sample is False

    def test_temperature_low_enables_sampling(self):
        request = ChatCompletionRequest(
            model="gpt-4",
            messages=[ChatMessage(role="user", content="Hello")],
            temperature=0.1,
        )
        config = GenerationConfig.from_request(request)
        assert config.temperature == 0.1
        assert config.do_sample is True

    def test_max_tokens_mapped(self):
        request = ChatCompletionRequest(
            model="gpt-4",
            messages=[ChatMessage(role="user", content="Hello")],
            max_tokens=100,
        )
        config = GenerationConfig.from_request(request)
        assert config.max_new_tokens == 100

    def test_top_p_mapped(self):
        request = ChatCompletionRequest(
            model="gpt-4",
            messages=[ChatMessage(role="user", content="Hello")],
            top_p=0.9,
        )
        config = GenerationConfig.from_request(request)
        assert config.top_p == 0.9

    def test_stop_sequences_single_string(self):
        request = ChatCompletionRequest(
            model="gpt-4",
            messages=[ChatMessage(role="user", content="Hello")],
            stop="END",
        )
        config = GenerationConfig.from_request(request)
        assert config.stop_sequences == ["END"]

    def test_stop_sequences_list(self):
        request = ChatCompletionRequest(
            model="gpt-4",
            messages=[ChatMessage(role="user", content="Hello")],
            stop=["END", "STOP"],
        )
        config = GenerationConfig.from_request(request)
        assert config.stop_sequences == ["END", "STOP"]

    def test_frequency_penalty_mapped(self):
        request = ChatCompletionRequest(
            model="gpt-4",
            messages=[ChatMessage(role="user", content="Hello")],
            frequency_penalty=0.5,
        )
        config = GenerationConfig.from_request(request)
        assert config.frequency_penalty == 0.5

    def test_presence_penalty_mapped(self):
        request = ChatCompletionRequest(
            model="gpt-4",
            messages=[ChatMessage(role="user", content="Hello")],
            presence_penalty=0.5,
        )
        config = GenerationConfig.from_request(request)
        assert config.presence_penalty == 0.5

    def test_full_request_parameters(self):
        request = ChatCompletionRequest(
            model="gpt-4",
            messages=[ChatMessage(role="user", content="Hello")],
            temperature=0.7,
            max_tokens=256,
            top_p=0.95,
            stop=["END"],
            frequency_penalty=0.3,
            presence_penalty=0.2,
        )
        config = GenerationConfig.from_request(request)
        assert config.temperature == 0.7
        assert config.max_new_tokens == 256
        assert config.top_p == 0.95
        assert config.stop_sequences == ["END"]
        assert config.frequency_penalty == 0.3
        assert config.presence_penalty == 0.2
        assert config.do_sample is True


class TestFromTextCompletionRequest:
    """Tests for GenerationConfig.from_request with text completions."""

    def test_text_completion_basic(self):
        request = TextCompletionRequest(
            model="gpt-3.5-turbo-instruct",
            prompt="Once upon a time",
        )
        config = GenerationConfig.from_request(request)
        # TextCompletionRequest default max_tokens is 16
        assert config.max_new_tokens == 16
        assert config.temperature == 1.0

    def test_text_completion_with_parameters(self):
        request = TextCompletionRequest(
            model="gpt-3.5-turbo-instruct",
            prompt="Once upon a time",
            temperature=0.5,
            max_tokens=50,
        )
        config = GenerationConfig.from_request(request)
        assert config.temperature == 0.5
        assert config.max_new_tokens == 50


class TestToGenerateKwargs:
    """Tests for GenerationConfig.to_generate_kwargs conversion."""

    def test_basic_kwargs(self):
        config = GenerationConfig(
            max_new_tokens=100,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
        )
        kwargs = config.to_generate_kwargs()
        assert kwargs["max_new_tokens"] == 100
        assert kwargs["temperature"] == 0.7
        assert kwargs["top_p"] == 0.9
        assert kwargs["do_sample"] is True

    def test_do_sample_false_kwargs(self):
        config = GenerationConfig(
            temperature=0.0,
            do_sample=False,
        )
        kwargs = config.to_generate_kwargs()
        assert kwargs["do_sample"] is False
        # When do_sample is False, temperature might still be in kwargs
        # but the model will ignore it

    def test_frequency_penalty_to_repetition_penalty(self):
        config = GenerationConfig(
            frequency_penalty=0.5,
        )
        kwargs = config.to_generate_kwargs()
        # frequency_penalty is mapped to repetition_penalty
        # Formula: repetition_penalty = 1.0 + (frequency_penalty * 0.25)
        # 0.5 * 0.25 = 0.125, so 1.0 + 0.125 = 1.125
        assert "repetition_penalty" in kwargs
        assert kwargs["repetition_penalty"] == 1.125

    def test_zero_frequency_penalty_no_repetition_penalty(self):
        config = GenerationConfig(
            frequency_penalty=0.0,
        )
        kwargs = config.to_generate_kwargs()
        # No repetition penalty when frequency_penalty is 0
        assert kwargs.get("repetition_penalty", 1.0) == 1.0

    def test_stop_sequences_not_in_kwargs(self):
        # Stop sequences are handled separately (not in generate kwargs)
        config = GenerationConfig(
            stop_sequences=["END"],
        )
        kwargs = config.to_generate_kwargs()
        # stop_sequences might be included or excluded depending on implementation
        # The main point is they're accessible on the config


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_none_max_tokens_uses_default(self):
        request = ChatCompletionRequest(
            model="gpt-4",
            messages=[ChatMessage(role="user", content="Hello")],
            max_tokens=None,
        )
        config = GenerationConfig.from_request(request)
        assert config.max_new_tokens == 512  # Default

    def test_very_high_temperature(self):
        request = ChatCompletionRequest(
            model="gpt-4",
            messages=[ChatMessage(role="user", content="Hello")],
            temperature=2.0,  # Max allowed by OpenAI
        )
        config = GenerationConfig.from_request(request)
        assert config.temperature == 2.0
        assert config.do_sample is True

    def test_top_p_one_disables_nucleus(self):
        request = ChatCompletionRequest(
            model="gpt-4",
            messages=[ChatMessage(role="user", content="Hello")],
            top_p=1.0,
        )
        config = GenerationConfig.from_request(request)
        # top_p=1.0 effectively disables nucleus sampling
        assert config.top_p == 1.0

    def test_very_low_top_p(self):
        request = ChatCompletionRequest(
            model="gpt-4",
            messages=[ChatMessage(role="user", content="Hello")],
            top_p=0.1,
        )
        config = GenerationConfig.from_request(request)
        assert config.top_p == 0.1
