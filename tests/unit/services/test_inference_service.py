"""Unit tests for InferenceService."""

import base64
import json
import struct
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import torch

from millm.api.schemas.openai import (
    ChatCompletionRequest,
    ChatMessage,
    EmbeddingRequest,
    TextCompletionRequest,
)
from millm.ml.generation_config import GenerationConfig
from millm.ml.model_loader import LoadedModel, LoadedModelState
from millm.services.inference_service import InferenceService, LoadedModelInfo


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture(autouse=True)
def reset_loaded_model_state():
    """Reset the LoadedModelState singleton before each test."""
    state = LoadedModelState()
    state._loaded = None
    yield
    state._loaded = None


@pytest.fixture
def mock_model():
    """Create a mock transformer model."""
    model = MagicMock()
    model.config = MagicMock()
    model.config.max_position_embeddings = 2048
    model.device = "cpu"

    # Default: generate returns a tensor with 5 prompt tokens + 3 generated tokens
    # Prompt tokens: [1, 2, 3, 4, 5], Generated tokens: [10, 11, 12]
    model.generate = MagicMock(
        return_value=torch.tensor([[1, 2, 3, 4, 5, 10, 11, 12]])
    )
    return model


@pytest.fixture
def mock_tokenizer():
    """Create a mock tokenizer."""
    tokenizer = MagicMock()
    tokenizer.pad_token_id = 1
    tokenizer.eos_token_id = 2
    tokenizer.chat_template = None

    # Tokenizer call returns mock input tensors
    mock_input_ids = torch.tensor([[1, 2, 3, 4, 5]])
    mock_attention_mask = torch.tensor([[1, 1, 1, 1, 1]])
    encoded = MagicMock()
    encoded.input_ids = mock_input_ids
    encoded.__getitem__ = lambda self, key: {
        "input_ids": mock_input_ids,
        "attention_mask": mock_attention_mask,
    }[key]
    encoded.items = MagicMock(
        return_value=[
            ("input_ids", mock_input_ids),
            ("attention_mask", mock_attention_mask),
        ]
    )
    encoded.to = MagicMock(return_value=encoded)

    tokenizer.return_value = encoded
    tokenizer.decode = MagicMock(return_value="Hello, world!")
    return tokenizer


@pytest.fixture
def loaded_model_state(mock_model, mock_tokenizer):
    """Set up LoadedModelState with mock model and tokenizer."""
    state = LoadedModelState()
    loaded = LoadedModel(
        model_id=1,
        model_name="test-model",
        model=mock_model,
        tokenizer=mock_tokenizer,
        loaded_at=datetime(2026, 1, 1, 12, 0, 0),
        memory_used_mb=1024,
        num_parameters=2_000_000_000,
        device="cpu",
        dtype="float16",
    )
    state.set(loaded)
    return state


@pytest.fixture
def service(loaded_model_state):
    """Create an InferenceService with a loaded model."""
    with patch("millm.services.inference_service.torch") as mock_torch:
        mock_torch.cuda.is_available.return_value = False
        mock_torch.no_grad.return_value = MagicMock(
            __enter__=MagicMock(), __exit__=MagicMock()
        )
        svc = InferenceService(model_service=None, steering_service=None)
    svc._device = "cpu"
    return svc


@pytest.fixture
def service_no_model():
    """Create an InferenceService without a loaded model."""
    with patch("millm.services.inference_service.torch") as mock_torch:
        mock_torch.cuda.is_available.return_value = False
        svc = InferenceService(model_service=None, steering_service=None)
    svc._device = "cpu"
    return svc


@pytest.fixture
def chat_request():
    """Create a basic chat completion request."""
    return ChatCompletionRequest(
        model="test-model",
        messages=[
            ChatMessage(role="user", content="Hello"),
        ],
    )


@pytest.fixture
def text_request():
    """Create a basic text completion request."""
    return TextCompletionRequest(
        model="test-model",
        prompt="Once upon a time",
    )


@pytest.fixture
def embedding_request():
    """Create a basic embedding request."""
    return EmbeddingRequest(
        model="test-model",
        input="Hello, world!",
    )


# =============================================================================
# Tests: _determine_finish_reason
# =============================================================================


class TestDetermineFinishReason:
    """Tests for _determine_finish_reason method."""

    def test_returns_stop_when_tokens_less_than_max(self, service):
        """Returns 'stop' when generated tokens < max_new_tokens."""
        result = service._determine_finish_reason(
            generated_token_count=5, max_new_tokens=10
        )
        assert result == "stop"

    def test_returns_length_when_tokens_equal_max(self, service):
        """Returns 'length' when generated tokens == max_new_tokens."""
        result = service._determine_finish_reason(
            generated_token_count=10, max_new_tokens=10
        )
        assert result == "length"

    def test_returns_length_when_tokens_exceed_max(self, service):
        """Returns 'length' when generated tokens > max_new_tokens."""
        result = service._determine_finish_reason(
            generated_token_count=15, max_new_tokens=10
        )
        assert result == "length"

    def test_returns_stop_for_zero_tokens(self, service):
        """Returns 'stop' when no tokens were generated."""
        result = service._determine_finish_reason(
            generated_token_count=0, max_new_tokens=10
        )
        assert result == "stop"


# =============================================================================
# Tests: _apply_stop_sequences
# =============================================================================


class TestApplyStopSequences:
    """Tests for _apply_stop_sequences method."""

    def test_returns_text_unchanged_when_no_sequences(self, service):
        """Returns original text when stop_sequences is None."""
        text, found = service._apply_stop_sequences("Hello world", None)
        assert text == "Hello world"
        assert found is False

    def test_returns_text_unchanged_when_empty_list(self, service):
        """Returns original text when stop_sequences is empty."""
        text, found = service._apply_stop_sequences("Hello world", [])
        assert text == "Hello world"
        assert found is False

    def test_truncates_at_stop_sequence(self, service):
        """Truncates text at the first occurrence of a stop sequence."""
        text, found = service._apply_stop_sequences(
            "Hello world\nSecond line", ["\n"]
        )
        assert text == "Hello world"
        assert found is True

    def test_truncates_at_earliest_stop_sequence(self, service):
        """When multiple stop sequences match, truncates at the earliest one."""
        text, found = service._apply_stop_sequences(
            "Hello<stop>world<end>done", ["<end>", "<stop>"]
        )
        assert text == "Hello"
        assert found is True

    def test_no_match_returns_full_text(self, service):
        """Returns full text when no stop sequences are found."""
        text, found = service._apply_stop_sequences(
            "Hello world", ["<stop>", "<end>"]
        )
        assert text == "Hello world"
        assert found is False

    def test_stop_at_beginning_returns_empty(self, service):
        """Returns empty string when stop sequence is at position 0."""
        text, found = service._apply_stop_sequences(
            "<stop>Hello world", ["<stop>"]
        )
        assert text == ""
        assert found is True


# =============================================================================
# Tests: _check_context_length
# =============================================================================


class TestCheckContextLength:
    """Tests for _check_context_length method."""

    def test_passes_when_within_limit(self, service):
        """No error when prompt + max_tokens fits within context."""
        # max_position_embeddings = 2048
        service._check_context_length(prompt_tokens=100, max_new_tokens=200)
        # Should not raise

    def test_raises_when_exceeds_limit(self, service, mock_model):
        """Raises ValueError when prompt + max_tokens exceeds context."""
        with pytest.raises(ValueError, match="Context length exceeded"):
            service._check_context_length(
                prompt_tokens=1500, max_new_tokens=1000
            )

    def test_passes_when_exactly_at_limit(self, service):
        """No error when prompt + max_tokens exactly equals context limit."""
        service._check_context_length(
            prompt_tokens=1024, max_new_tokens=1024
        )
        # Should not raise

    def test_passes_when_no_config(self, service, mock_model):
        """No error when model has no max_position_embeddings."""
        mock_model.config.max_position_embeddings = None
        service._check_context_length(
            prompt_tokens=10000, max_new_tokens=10000
        )
        # Should not raise


# =============================================================================
# Tests: _build_generate_kwargs
# =============================================================================


class TestBuildGenerateKwargs:
    """Tests for _build_generate_kwargs method."""

    def test_includes_pad_and_eos_token_ids(self, service, mock_tokenizer):
        """Result includes pad_token_id and eos_token_id from tokenizer."""
        gen_config = GenerationConfig(max_new_tokens=100)
        inputs = {"input_ids": torch.tensor([[1, 2, 3]])}
        result = service._build_generate_kwargs(gen_config, inputs)

        # pad_token_id=1 (truthy), so it's used directly
        assert result["pad_token_id"] == 1
        assert result["eos_token_id"] == 2

    def test_falls_back_to_eos_when_pad_is_zero(self, service, mock_tokenizer):
        """Uses eos_token_id when pad_token_id is 0 (falsy in Python or-expression)."""
        mock_tokenizer.pad_token_id = 0
        gen_config = GenerationConfig(max_new_tokens=100)
        inputs = {"input_ids": torch.tensor([[1, 2, 3]])}
        result = service._build_generate_kwargs(gen_config, inputs)

        # 0 is falsy, so `0 or 2` evaluates to 2
        assert result["pad_token_id"] == 2

    def test_falls_back_to_eos_when_no_pad(self, service, mock_tokenizer):
        """Uses eos_token_id as pad_token_id when pad_token_id is None."""
        mock_tokenizer.pad_token_id = None
        gen_config = GenerationConfig(max_new_tokens=100)
        inputs = {"input_ids": torch.tensor([[1, 2, 3]])}
        result = service._build_generate_kwargs(gen_config, inputs)

        assert result["pad_token_id"] == 2  # falls back to eos_token_id

    def test_includes_generation_config_params(self, service):
        """Result includes parameters from GenerationConfig.to_generate_kwargs()."""
        gen_config = GenerationConfig(
            max_new_tokens=256,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
        )
        inputs = {"input_ids": torch.tensor([[1, 2, 3]])}
        result = service._build_generate_kwargs(gen_config, inputs)

        assert result["max_new_tokens"] == 256
        assert result["temperature"] == 0.7
        assert result["top_p"] == 0.9
        assert result["do_sample"] is True

    def test_moves_inputs_to_device(self, service):
        """Input tensors are moved to the service device."""
        gen_config = GenerationConfig(max_new_tokens=100)
        mock_tensor = MagicMock()
        mock_tensor.to = MagicMock(return_value=mock_tensor)
        inputs = {"input_ids": mock_tensor}

        service._build_generate_kwargs(gen_config, inputs)

        mock_tensor.to.assert_called_once_with("cpu")


# =============================================================================
# Tests: _format_chat_messages
# =============================================================================


class TestFormatChatMessages:
    """Tests for _format_chat_messages method."""

    def test_user_message_gemma_fallback(self, service):
        """Formats user message with Gemma-style turn markers."""
        messages = [ChatMessage(role="user", content="Hello")]
        result = service._format_chat_messages(messages)

        assert "<start_of_turn>user\nHello<end_of_turn>" in result
        assert result.endswith("<start_of_turn>model")

    def test_system_prepended_to_user(self, service):
        """System message is prepended to the next user message."""
        messages = [
            ChatMessage(role="system", content="You are helpful."),
            ChatMessage(role="user", content="Hello"),
        ]
        result = service._format_chat_messages(messages)

        assert (
            "<start_of_turn>user\nYou are helpful.\n\nHello<end_of_turn>"
            in result
        )

    def test_assistant_message_uses_model_turn(self, service):
        """Assistant messages use 'model' as role in turn markers."""
        messages = [
            ChatMessage(role="user", content="Hello"),
            ChatMessage(role="assistant", content="Hi there!"),
        ]
        result = service._format_chat_messages(messages)

        assert "<start_of_turn>model\nHi there!<end_of_turn>" in result

    def test_dangling_system_message(self, service):
        """System message with no following user turn is wrapped as user turn."""
        messages = [ChatMessage(role="system", content="Be concise.")]
        result = service._format_chat_messages(messages)

        assert "<start_of_turn>user\nBe concise.<end_of_turn>" in result

    def test_uses_chat_template_when_available(self, service, mock_tokenizer):
        """Uses tokenizer's chat template when available."""
        mock_tokenizer.chat_template = "{{ messages }}"
        mock_tokenizer.apply_chat_template = MagicMock(
            return_value="formatted by template"
        )

        messages = [ChatMessage(role="user", content="Hello")]
        result = service._format_chat_messages(messages)

        assert result == "formatted by template"
        mock_tokenizer.apply_chat_template.assert_called_once()

    def test_falls_back_when_template_fails(self, service, mock_tokenizer):
        """Falls back to Gemma format when chat template raises an error."""
        mock_tokenizer.chat_template = "{{ messages }}"
        mock_tokenizer.apply_chat_template = MagicMock(
            side_effect=Exception("Template error")
        )

        messages = [ChatMessage(role="user", content="Hello")]
        result = service._format_chat_messages(messages)

        # Should use fallback format
        assert "<start_of_turn>user\nHello<end_of_turn>" in result

    def test_multi_turn_conversation(self, service):
        """Formats a multi-turn conversation correctly."""
        messages = [
            ChatMessage(role="system", content="You are helpful."),
            ChatMessage(role="user", content="What is 2+2?"),
            ChatMessage(role="assistant", content="4"),
            ChatMessage(role="user", content="And 3+3?"),
        ]
        result = service._format_chat_messages(messages)

        assert "You are helpful.\n\nWhat is 2+2?" in result
        assert "<start_of_turn>model\n4<end_of_turn>" in result
        assert "<start_of_turn>user\nAnd 3+3?<end_of_turn>" in result
        assert result.endswith("<start_of_turn>model")


# =============================================================================
# Tests: create_chat_completion
# =============================================================================


class TestCreateChatCompletion:
    """Tests for create_chat_completion method."""

    @pytest.mark.asyncio
    async def test_basic_response_structure(self, service, chat_request):
        """Returns properly structured ChatCompletionResponse."""
        response = await service.create_chat_completion(chat_request)

        assert response.id.startswith("chatcmpl-")
        assert response.object == "chat.completion"
        assert response.model == "test-model"
        assert len(response.choices) == 1
        assert response.choices[0].index == 0
        assert response.choices[0].message.role == "assistant"
        assert response.choices[0].message.content == "Hello, world!"
        assert response.usage.prompt_tokens > 0
        assert response.usage.completion_tokens > 0
        assert response.usage.total_tokens == (
            response.usage.prompt_tokens + response.usage.completion_tokens
        )

    @pytest.mark.asyncio
    async def test_finish_reason_stop(self, service, chat_request, mock_model):
        """Returns finish_reason 'stop' when generation ends before max tokens."""
        # Default config: max_new_tokens=512, generated_ids has 3 tokens
        response = await service.create_chat_completion(chat_request)

        assert response.choices[0].finish_reason == "stop"

    @pytest.mark.asyncio
    async def test_finish_reason_length(
        self, service, mock_model, mock_tokenizer
    ):
        """Returns finish_reason 'length' when max_tokens is reached."""
        # Generate exactly max_tokens tokens (3 generated, max_tokens=3)
        request = ChatCompletionRequest(
            model="test-model",
            messages=[ChatMessage(role="user", content="Hello")],
            max_tokens=3,
        )
        response = await service.create_chat_completion(request)

        assert response.choices[0].finish_reason == "length"

    @pytest.mark.asyncio
    async def test_stop_sequences_applied(self, service, mock_tokenizer):
        """Stop sequences truncate the output and set finish_reason to 'stop'."""
        mock_tokenizer.decode.return_value = "Hello<stop>world"
        request = ChatCompletionRequest(
            model="test-model",
            messages=[ChatMessage(role="user", content="Hello")],
            stop=["<stop>"],
        )
        response = await service.create_chat_completion(request)

        assert response.choices[0].message.content == "Hello"
        assert response.choices[0].finish_reason == "stop"

    @pytest.mark.asyncio
    async def test_n_parameter_generates_multiple_choices(
        self, service, mock_model
    ):
        """The n parameter generates multiple completion choices."""
        request = ChatCompletionRequest(
            model="test-model",
            messages=[ChatMessage(role="user", content="Hello")],
            n=3,
        )
        response = await service.create_chat_completion(request)

        assert len(response.choices) == 3
        for i, choice in enumerate(response.choices):
            assert choice.index == i
            assert choice.message.role == "assistant"

        # Usage should reflect n completions
        assert response.usage.prompt_tokens == 5 * 3  # 5 prompt tokens * 3


# =============================================================================
# Tests: stream_chat_completion
# =============================================================================


class TestStreamChatCompletion:
    """Tests for stream_chat_completion method."""

    @pytest.fixture
    def mock_streamer(self):
        """Create a mock TextIteratorStreamer that yields tokens."""
        streamer = MagicMock()
        streamer.__iter__ = MagicMock(
            return_value=iter(["Hello", ", ", "world", "!"])
        )
        return streamer

    @pytest.mark.asyncio
    async def test_yields_sse_format(self, service, chat_request, mock_streamer):
        """Each yielded chunk starts with 'data: ' and ends with double newline."""
        with patch(
            "transformers.TextIteratorStreamer",
            return_value=mock_streamer,
        ):
            chunks = []
            async for chunk in service.stream_chat_completion(chat_request):
                chunks.append(chunk)

        assert len(chunks) > 0
        for chunk in chunks:
            assert chunk.startswith("data: ")
            assert chunk.endswith("\n\n")

    @pytest.mark.asyncio
    async def test_ends_with_done(self, service, chat_request, mock_streamer):
        """The stream always ends with 'data: [DONE]'."""
        with patch(
            "transformers.TextIteratorStreamer",
            return_value=mock_streamer,
        ):
            chunks = []
            async for chunk in service.stream_chat_completion(chat_request):
                chunks.append(chunk)

        assert chunks[-1] == "data: [DONE]\n\n"

    @pytest.mark.asyncio
    async def test_first_chunk_has_role(self, service, chat_request, mock_streamer):
        """The first data chunk contains role='assistant' in the delta."""
        with patch(
            "transformers.TextIteratorStreamer",
            return_value=mock_streamer,
        ):
            chunks = []
            async for chunk in service.stream_chat_completion(chat_request):
                chunks.append(chunk)

        # First chunk is the role chunk
        first_data = json.loads(chunks[0].removeprefix("data: ").strip())
        assert first_data["choices"][0]["delta"]["role"] == "assistant"

    @pytest.mark.asyncio
    async def test_final_chunk_has_finish_reason(
        self, service, chat_request, mock_streamer
    ):
        """The penultimate chunk (before [DONE]) has a finish_reason."""
        with patch(
            "transformers.TextIteratorStreamer",
            return_value=mock_streamer,
        ):
            chunks = []
            async for chunk in service.stream_chat_completion(chat_request):
                chunks.append(chunk)

        # Penultimate chunk should have finish_reason
        final_data_str = chunks[-2].removeprefix("data: ").strip()
        final_data = json.loads(final_data_str)
        assert final_data["choices"][0]["finish_reason"] in [
            "stop",
            "length",
        ]

    @pytest.mark.asyncio
    async def test_stop_sequence_in_stream(
        self, service, mock_tokenizer
    ):
        """Stop sequences stop the stream and set finish_reason to 'stop'."""
        streamer = MagicMock()
        streamer.__iter__ = MagicMock(
            return_value=iter(["Hello", "<stop>", "ignored"])
        )

        request = ChatCompletionRequest(
            model="test-model",
            messages=[ChatMessage(role="user", content="Hi")],
            stop=["<stop>"],
            stream=True,
        )

        with patch(
            "transformers.TextIteratorStreamer",
            return_value=streamer,
        ):
            chunks = []
            async for chunk in service.stream_chat_completion(request):
                chunks.append(chunk)

        assert chunks[-1] == "data: [DONE]\n\n"
        # The chunk before [DONE] should have finish_reason "stop"
        final_data = json.loads(chunks[-2].removeprefix("data: ").strip())
        assert final_data["choices"][0]["finish_reason"] == "stop"


# =============================================================================
# Tests: create_text_completion
# =============================================================================


class TestCreateTextCompletion:
    """Tests for create_text_completion method."""

    @pytest.mark.asyncio
    async def test_basic_response_structure(self, service, text_request):
        """Returns properly structured TextCompletionResponse."""
        response = await service.create_text_completion(text_request)

        assert response.id.startswith("cmpl-")
        assert response.object == "text_completion"
        assert response.model == "test-model"
        assert len(response.choices) == 1
        assert response.choices[0].index == 0
        assert response.choices[0].text == "Hello, world!"
        assert response.usage.prompt_tokens > 0

    @pytest.mark.asyncio
    async def test_batch_prompts(self, service, mock_model):
        """Handles list of prompts and generates one choice per prompt."""
        request = TextCompletionRequest(
            model="test-model",
            prompt=["First prompt", "Second prompt"],
        )
        response = await service.create_text_completion(request)

        assert len(response.choices) == 2
        assert response.choices[0].index == 0
        assert response.choices[1].index == 1

    @pytest.mark.asyncio
    async def test_stop_sequences_applied(self, service, mock_tokenizer):
        """Stop sequences truncate text completion output."""
        mock_tokenizer.decode.return_value = "Once upon a time\n\nThe end"
        request = TextCompletionRequest(
            model="test-model",
            prompt="Once upon a time",
            stop=["\n\n"],
        )
        response = await service.create_text_completion(request)

        assert response.choices[0].text == "Once upon a time"
        assert response.choices[0].finish_reason == "stop"

    @pytest.mark.asyncio
    async def test_finish_reason_length(self, service, mock_model):
        """Returns 'length' finish_reason when max_tokens is reached."""
        # Generated output has 3 tokens, set max_tokens=3
        request = TextCompletionRequest(
            model="test-model",
            prompt="Test",
            max_tokens=3,
        )
        response = await service.create_text_completion(request)

        assert response.choices[0].finish_reason == "length"

    @pytest.mark.asyncio
    async def test_model_name_in_response(self, service, text_request):
        """Response includes the loaded model name."""
        response = await service.create_text_completion(text_request)

        assert response.model == "test-model"

    @pytest.mark.asyncio
    async def test_usage_tokens_match(self, service, text_request):
        """Usage total_tokens equals prompt_tokens + completion_tokens."""
        response = await service.create_text_completion(text_request)

        assert response.usage.total_tokens == (
            response.usage.prompt_tokens + response.usage.completion_tokens
        )


# =============================================================================
# Tests: create_embeddings
# =============================================================================


class TestCreateEmbeddings:
    """Tests for create_embeddings method."""

    @pytest.fixture
    def mock_model_for_embeddings(self, mock_model):
        """Set up model to return hidden states for embeddings."""
        mock_output = MagicMock()
        mock_hidden = torch.randn(1, 5, 64)  # batch=1, seq_len=5, hidden=64
        mock_output.hidden_states = [mock_hidden]  # Only last layer matters
        mock_model.return_value = mock_output
        return mock_model

    @pytest.mark.asyncio
    async def test_basic_response_structure(
        self, service, embedding_request, mock_model_for_embeddings
    ):
        """Returns properly structured EmbeddingResponse."""
        response = await service.create_embeddings(embedding_request)

        assert response.object == "list"
        assert response.model == "test-model"
        assert len(response.data) == 1
        assert response.data[0].object == "embedding"
        assert response.data[0].index == 0
        assert isinstance(response.data[0].embedding, list)
        assert len(response.data[0].embedding) == 64  # hidden dim
        assert response.usage.prompt_tokens > 0
        assert response.usage.completion_tokens == 0

    @pytest.mark.asyncio
    async def test_multiple_inputs(
        self, service, mock_model_for_embeddings
    ):
        """Handles multiple input strings."""
        request = EmbeddingRequest(
            model="test-model",
            input=["Hello", "World"],
        )
        response = await service.create_embeddings(request)

        assert len(response.data) == 2
        assert response.data[0].index == 0
        assert response.data[1].index == 1

    @pytest.mark.asyncio
    async def test_base64_encoding_format(
        self, service, mock_model_for_embeddings
    ):
        """Returns base64-encoded embeddings when encoding_format='base64'."""
        request = EmbeddingRequest(
            model="test-model",
            input="Hello",
            encoding_format="base64",
        )
        response = await service.create_embeddings(request)

        embedding = response.data[0].embedding
        assert isinstance(embedding, str)

        # Verify it's valid base64 that decodes to correct number of floats
        decoded_bytes = base64.b64decode(embedding)
        num_floats = len(decoded_bytes) // 4  # 4 bytes per float32
        assert num_floats == 64  # hidden dim
        floats = struct.unpack(f"<{num_floats}f", decoded_bytes)
        assert len(floats) == 64


# =============================================================================
# Tests: State methods
# =============================================================================


class TestIsModelLoaded:
    """Tests for is_model_loaded method."""

    def test_returns_true_when_loaded(self, service):
        """Returns True when a model is loaded."""
        assert service.is_model_loaded() is True

    def test_returns_false_when_not_loaded(self, service_no_model):
        """Returns False when no model is loaded."""
        assert service_no_model.is_model_loaded() is False


class TestGetLoadedModelInfo:
    """Tests for get_loaded_model_info method."""

    def test_returns_info_when_loaded(self, service):
        """Returns LoadedModelInfo with correct attributes when model is loaded."""
        info = service.get_loaded_model_info()

        assert info is not None
        assert isinstance(info, LoadedModelInfo)
        assert info.name == "test-model"
        assert info.model_id == 1
        assert info.loaded_at == datetime(2026, 1, 1, 12, 0, 0)

    def test_returns_none_when_not_loaded(self, service_no_model):
        """Returns None when no model is loaded."""
        info = service_no_model.get_loaded_model_info()

        assert info is None


# =============================================================================
# Tests: _model and _tokenizer properties
# =============================================================================


class TestModelAndTokenizerProperties:
    """Tests for _model and _tokenizer property access."""

    def test_model_property_raises_when_not_loaded(self, service_no_model):
        """Accessing _model raises RuntimeError when no model is loaded."""
        with pytest.raises(RuntimeError, match="No model is loaded"):
            _ = service_no_model._model

    def test_tokenizer_property_raises_when_not_loaded(self, service_no_model):
        """Accessing _tokenizer raises RuntimeError when no model is loaded."""
        with pytest.raises(RuntimeError, match="No model is loaded"):
            _ = service_no_model._tokenizer

    def test_model_property_returns_model(self, service, mock_model):
        """Accessing _model returns the loaded model."""
        assert service._model is mock_model

    def test_tokenizer_property_returns_tokenizer(
        self, service, mock_tokenizer
    ):
        """Accessing _tokenizer returns the loaded tokenizer."""
        assert service._tokenizer is mock_tokenizer


# =============================================================================
# Tests: request_queue property
# =============================================================================


class TestRequestQueue:
    """Tests for request_queue property."""

    def test_request_queue_accessible(self, service):
        """The request_queue property returns a RequestQueue instance."""
        from millm.services.request_queue import RequestQueue

        assert isinstance(service.request_queue, RequestQueue)

    def test_request_queue_has_correct_defaults(self, service):
        """Request queue is initialized with the configured limits."""
        queue = service.request_queue
        assert queue.max_concurrent == 1
        assert queue.max_pending == 5
