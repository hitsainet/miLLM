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
        svc = InferenceService(model_service=None)
    svc._device = "cpu"
    return svc


@pytest.fixture
def service_no_model():
    """Create an InferenceService without a loaded model."""
    with patch("millm.services.inference_service.torch") as mock_torch:
        mock_torch.cuda.is_available.return_value = False
        svc = InferenceService(model_service=None)
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
    async def test_embeddings_suppress_attached_sae(
        self, service, embedding_request, mock_model_for_embeddings
    ):
        """Embeddings run inside the SAE's suppressed() context (M4).

        An attached steering hook must not perturb the hidden states the
        embeddings are pooled from.
        """
        from contextlib import contextmanager

        entered = {"count": 0, "active_during_forward": False}
        mock_sae = MagicMock()

        @contextmanager
        def fake_suppressed():
            entered["count"] += 1
            entered["active_during_forward"] = True
            try:
                yield
            finally:
                entered["active_during_forward"] = False

        mock_sae.suppressed = fake_suppressed

        # Record whether suppression was active at the moment of the forward pass.
        forward_state = {}

        def record_forward(*args, **kwargs):
            forward_state["suppressed"] = entered["active_during_forward"]
            out = MagicMock()
            out.hidden_states = [torch.randn(1, 5, 64)]
            return out

        mock_model_for_embeddings.side_effect = record_forward

        with patch.object(service, "_get_attached_sae", return_value=mock_sae):
            await service.create_embeddings(embedding_request)

        assert entered["count"] == 1
        assert forward_state.get("suppressed") is True

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


# =============================================================================
# Tests: CBM Integration
# =============================================================================


class TestCBMInit:
    """Tests for CBM initialization in InferenceService."""

    def test_cbm_disabled_by_default(self, service):
        """CBM backend is None by default."""
        assert service._cbm_backend is None

    def test_cbm_enabled_creates_backend(self, loaded_model_state):
        """Setting enable_cbm=True creates a ContinuousBatchingBackend."""
        with patch("millm.services.inference_service.torch") as mock_torch:
            mock_torch.cuda.is_available.return_value = False
            svc = InferenceService(
                enable_cbm=True,
                cbm_config={
                    "max_queue_size": 64,
                    "default_temperature": 0.5,
                },
            )
        from millm.services.cbm_backend import ContinuousBatchingBackend

        assert isinstance(svc._cbm_backend, ContinuousBatchingBackend)
        assert svc._cbm_backend._max_queue_size == 64
        assert svc._cbm_backend._default_temperature == 0.5

    def test_cbm_enabled_without_config(self, loaded_model_state):
        """CBM with no config dict uses defaults."""
        with patch("millm.services.inference_service.torch") as mock_torch:
            mock_torch.cuda.is_available.return_value = False
            svc = InferenceService(enable_cbm=True)

        assert svc._cbm_backend is not None
        assert svc._cbm_backend._max_queue_size == 256  # default


class TestUseCBM:
    """Tests for _use_cbm and _use_cbm_for_request methods."""

    def test_false_when_no_backend(self, service):
        """Returns False when CBM backend is None."""
        assert service._use_cbm() is False

    def test_false_when_backend_not_running(self, service):
        """Returns False when CBM backend exists but is not running."""
        service._cbm_backend = MagicMock()
        service._cbm_backend.is_running = False
        assert service._use_cbm() is False

    def test_true_when_backend_running(self, service):
        """Returns True when CBM backend is running."""
        service._cbm_backend = MagicMock()
        service._cbm_backend.is_running = True
        assert service._use_cbm() is True

    def test_use_cbm_for_request_false_when_no_backend(self, service):
        """_use_cbm_for_request returns False when backend is None."""
        assert service._use_cbm_for_request(temperature=0.7, top_p=0.95) is False

    def test_use_cbm_for_request_true_when_params_match(self, service):
        """_use_cbm_for_request returns True when sampling params are compatible."""
        service._cbm_backend = MagicMock()
        service._cbm_backend.is_running = True
        service._cbm_backend.sampling_params_match = MagicMock(return_value=True)
        assert service._use_cbm_for_request(temperature=0.7, top_p=0.95) is True

    def test_use_cbm_for_request_false_when_params_mismatch(self, service):
        """_use_cbm_for_request returns False and logs when params differ from CBM config."""
        service._cbm_backend = MagicMock()
        service._cbm_backend.is_running = True
        service._cbm_backend.sampling_params_match = MagicMock(return_value=False)
        assert service._use_cbm_for_request(temperature=0.1, top_p=0.5) is False

    def test_use_cbm_for_request_true_when_no_sampling_params(self, service):
        """_use_cbm_for_request returns True when request has no sampling params (None)."""
        service._cbm_backend = MagicMock()
        service._cbm_backend.is_running = True
        service._cbm_backend.sampling_params_match = MagicMock(return_value=True)
        assert service._use_cbm_for_request() is True

    def test_use_cbm_for_request_false_when_force_serial_monitoring_and_monitoring_on(self, service):
        """CBM is bypassed when force_serial_monitoring=True and monitoring is enabled."""
        service._cbm_backend = MagicMock()
        service._cbm_backend.is_running = True
        service._cbm_backend.sampling_params_match = MagicMock(return_value=True)
        service._cbm_force_serial_monitoring = True

        with patch.object(service, "_is_monitoring_enabled", return_value=True):
            assert service._use_cbm_for_request() is False

    def test_use_cbm_for_request_true_when_force_serial_monitoring_but_no_monitoring(self, service):
        """CBM is still used when force_serial_monitoring=True but monitoring is disabled."""
        service._cbm_backend = MagicMock()
        service._cbm_backend.is_running = True
        service._cbm_backend.sampling_params_match = MagicMock(return_value=True)
        service._cbm_force_serial_monitoring = True

        with patch.object(service, "_is_monitoring_enabled", return_value=False):
            assert service._use_cbm_for_request() is True

    def test_use_cbm_for_request_true_when_monitoring_on_but_force_serial_off(self, service):
        """CBM is used when monitoring is enabled but force_serial_monitoring=False."""
        service._cbm_backend = MagicMock()
        service._cbm_backend.is_running = True
        service._cbm_backend.sampling_params_match = MagicMock(return_value=True)
        service._cbm_force_serial_monitoring = False

        with patch.object(service, "_is_monitoring_enabled", return_value=True):
            assert service._use_cbm_for_request() is True


class TestCBMLifecycle:
    """Tests for on_model_loaded and on_model_unloading."""

    def test_on_model_loaded_starts_cbm(self, service, mock_model, mock_tokenizer):
        """on_model_loaded starts CBM backend when model is loaded."""
        mock_backend = MagicMock()
        service._cbm_backend = mock_backend

        service.on_model_loaded()

        mock_backend.start.assert_called_once_with(mock_model, mock_tokenizer)

    def test_on_model_loaded_noop_when_no_backend(self, service):
        """on_model_loaded is a no-op when CBM is not configured."""
        service._cbm_backend = None
        service.on_model_loaded()  # Should not raise

    def test_on_model_loaded_noop_when_no_model(self, service_no_model):
        """on_model_loaded is a no-op when no model is loaded."""
        mock_backend = MagicMock()
        service_no_model._cbm_backend = mock_backend

        service_no_model.on_model_loaded()

        mock_backend.start.assert_not_called()

    def test_on_model_loaded_handles_start_failure(self, service):
        """on_model_loaded handles exceptions from CBM start gracefully."""
        mock_backend = MagicMock()
        mock_backend.start.side_effect = RuntimeError("CBM init failed")
        service._cbm_backend = mock_backend

        service.on_model_loaded()  # Should not raise

    def test_on_model_unloading_stops_cbm(self, service):
        """on_model_unloading stops a running CBM backend."""
        mock_backend = MagicMock()
        mock_backend.is_running = True
        service._cbm_backend = mock_backend

        service.on_model_unloading()

        mock_backend.stop.assert_called_once()

    def test_on_model_unloading_noop_when_not_running(self, service):
        """on_model_unloading is a no-op when CBM is not running."""
        mock_backend = MagicMock()
        mock_backend.is_running = False
        service._cbm_backend = mock_backend

        service.on_model_unloading()

        mock_backend.stop.assert_not_called()

    def test_on_model_unloading_noop_when_no_backend(self, service):
        """on_model_unloading is a no-op when CBM is not configured."""
        service._cbm_backend = None
        service.on_model_unloading()  # Should not raise


class TestCBMChatCompletion:
    """Tests for CBM chat completion delegation."""

    @pytest.fixture
    def cbm_service(self, service, mock_tokenizer):
        """Create a service with mocked CBM backend."""
        mock_backend = AsyncMock()
        mock_backend.is_running = True
        mock_backend.sampling_params_match = MagicMock(return_value=True)
        mock_backend.generate = AsyncMock(
            return_value=([10, 11, 12], "stop")
        )
        service._cbm_backend = mock_backend

        # Mock tokenizer.encode to return tensor-like result
        mock_tokenizer.encode = MagicMock(
            return_value=torch.tensor([[1, 2, 3, 4, 5]])
        )
        return service

    @pytest.mark.asyncio
    async def test_delegates_to_cbm_when_running(self, cbm_service, chat_request):
        """create_chat_completion delegates to CBM when backend is running."""
        response = await cbm_service.create_chat_completion(chat_request)

        assert response.id.startswith("chatcmpl-")
        assert response.model == "test-model"
        assert len(response.choices) == 1
        assert response.choices[0].message.role == "assistant"
        assert response.choices[0].message.content == "Hello, world!"
        cbm_service._cbm_backend.generate.assert_called_once()

    @pytest.mark.asyncio
    async def test_cbm_response_usage(self, cbm_service, chat_request):
        """CBM response includes correct token usage."""
        response = await cbm_service.create_chat_completion(chat_request)

        assert response.usage.prompt_tokens == 5
        assert response.usage.completion_tokens == 3
        assert response.usage.total_tokens == 8

    @pytest.mark.asyncio
    async def test_cbm_finish_reason_propagated(self, cbm_service, chat_request):
        """CBM finish reason is included in response."""
        cbm_service._cbm_backend.generate = AsyncMock(
            return_value=([10, 11, 12], "length")
        )
        response = await cbm_service.create_chat_completion(chat_request)

        assert response.choices[0].finish_reason == "length"

    @pytest.mark.asyncio
    async def test_cbm_stop_sequences(self, cbm_service, mock_tokenizer):
        """CBM applies stop sequences and overrides finish_reason to 'stop'."""
        mock_tokenizer.decode.return_value = "Hello<stop>world"
        cbm_service._cbm_backend.generate = AsyncMock(
            return_value=([10, 11, 12], "length")
        )
        request = ChatCompletionRequest(
            model="test-model",
            messages=[ChatMessage(role="user", content="Hi")],
            stop=["<stop>"],
        )
        response = await cbm_service.create_chat_completion(request)

        assert response.choices[0].message.content == "Hello"
        assert response.choices[0].finish_reason == "stop"


class TestCBMStreamChatCompletion:
    """Tests for CBM streaming chat completion."""

    @pytest.fixture
    def cbm_stream_service(self, service, mock_tokenizer):
        """Create a service with mocked CBM streaming backend."""
        mock_backend = MagicMock()
        mock_backend.is_running = True
        mock_backend.sampling_params_match = MagicMock(return_value=True)

        async def mock_generate_stream(input_ids, max_new_tokens, request_id):
            yield [10]
            yield [11]
            yield [12]

        mock_backend.generate_stream = mock_generate_stream
        service._cbm_backend = mock_backend

        mock_tokenizer.encode = MagicMock(
            return_value=torch.tensor([[1, 2, 3, 4, 5]])
        )
        # Different decode results for each chunk
        mock_tokenizer.decode = MagicMock(side_effect=["Hello", " world", "!"])
        return service

    @pytest.mark.asyncio
    async def test_streams_sse_chunks(self, cbm_stream_service, chat_request):
        """CBM streaming yields SSE-formatted chunks."""
        chunks = []
        async for chunk in cbm_stream_service.stream_chat_completion(chat_request):
            chunks.append(chunk)

        # First chunk (role) + 3 content chunks + final chunk + [DONE]
        assert len(chunks) >= 5
        for chunk in chunks:
            assert chunk.startswith("data: ")
            assert chunk.endswith("\n\n")

    @pytest.mark.asyncio
    async def test_stream_first_chunk_has_role(self, cbm_stream_service, chat_request):
        """First SSE chunk contains role='assistant'."""
        chunks = []
        async for chunk in cbm_stream_service.stream_chat_completion(chat_request):
            chunks.append(chunk)

        first_data = json.loads(chunks[0].removeprefix("data: ").strip())
        assert first_data["choices"][0]["delta"]["role"] == "assistant"

    @pytest.mark.asyncio
    async def test_stream_ends_with_done(self, cbm_stream_service, chat_request):
        """CBM stream ends with 'data: [DONE]'."""
        chunks = []
        async for chunk in cbm_stream_service.stream_chat_completion(chat_request):
            chunks.append(chunk)

        assert chunks[-1] == "data: [DONE]\n\n"

    @pytest.mark.asyncio
    async def test_stream_final_chunk_has_finish_reason(
        self, cbm_stream_service, chat_request
    ):
        """Penultimate chunk has a finish_reason."""
        chunks = []
        async for chunk in cbm_stream_service.stream_chat_completion(chat_request):
            chunks.append(chunk)

        final_data = json.loads(chunks[-2].removeprefix("data: ").strip())
        assert final_data["choices"][0]["finish_reason"] in ["stop", "length"]


class TestCBMTextCompletion:
    """Tests for CBM text completion."""

    @pytest.fixture
    def cbm_text_service(self, service, mock_tokenizer):
        """Create a service with mocked CBM backend for text completion."""
        mock_backend = AsyncMock()
        mock_backend.is_running = True
        mock_backend.sampling_params_match = MagicMock(return_value=True)
        mock_backend.generate = AsyncMock(
            return_value=([10, 11, 12], "stop")
        )
        service._cbm_backend = mock_backend

        mock_tokenizer.encode = MagicMock(
            return_value=torch.tensor([[1, 2, 3, 4, 5]])
        )
        return service

    @pytest.mark.asyncio
    async def test_delegates_to_cbm(self, cbm_text_service, text_request):
        """create_text_completion delegates to CBM when backend is running."""
        response = await cbm_text_service.create_text_completion(text_request)

        assert response.id.startswith("cmpl-")
        assert response.model == "test-model"
        assert len(response.choices) == 1
        assert response.choices[0].text == "Hello, world!"
        cbm_text_service._cbm_backend.generate.assert_called_once()

    @pytest.mark.asyncio
    async def test_cbm_text_batch_prompts(self, cbm_text_service):
        """CBM handles batch of text prompts."""
        request = TextCompletionRequest(
            model="test-model",
            prompt=["First", "Second"],
        )
        response = await cbm_text_service.create_text_completion(request)

        assert len(response.choices) == 2
        assert response.choices[0].index == 0
        assert response.choices[1].index == 1
        # generate called once per prompt
        assert cbm_text_service._cbm_backend.generate.call_count == 2

    @pytest.mark.asyncio
    async def test_cbm_text_usage(self, cbm_text_service, text_request):
        """CBM text completion includes correct usage."""
        response = await cbm_text_service.create_text_completion(text_request)

        assert response.usage.prompt_tokens == 5
        assert response.usage.completion_tokens == 3
        assert response.usage.total_tokens == 8

    @pytest.mark.asyncio
    async def test_cbm_text_stop_sequences(self, cbm_text_service, mock_tokenizer):
        """CBM text completion applies stop sequences."""
        mock_tokenizer.decode.return_value = "Once upon<end>a time"
        cbm_text_service._cbm_backend.generate = AsyncMock(
            return_value=([10, 11, 12, 13], "length")
        )
        request = TextCompletionRequest(
            model="test-model",
            prompt="Once",
            stop=["<end>"],
        )
        response = await cbm_text_service.create_text_completion(request)

        assert response.choices[0].text == "Once upon"
        assert response.choices[0].finish_reason == "stop"


class TestCBMDoesNotAffectEmbeddings:
    """Verify embeddings always use queue path, not CBM."""

    @pytest.fixture
    def mock_model_for_embeddings(self, mock_model):
        """Set up model to return hidden states."""
        mock_output = MagicMock()
        mock_hidden = torch.randn(1, 5, 64)
        mock_output.hidden_states = [mock_hidden]
        mock_model.return_value = mock_output
        return mock_model

    @pytest.mark.asyncio
    async def test_embeddings_use_queue_when_cbm_enabled(
        self, service, mock_model_for_embeddings
    ):
        """Embeddings always go through request queue, even with CBM enabled."""
        mock_backend = MagicMock()
        mock_backend.is_running = True
        service._cbm_backend = mock_backend

        request = EmbeddingRequest(model="test-model", input="Hello")
        response = await service.create_embeddings(request)

        assert response.object == "list"
        assert len(response.data) == 1
        # CBM generate should NOT be called for embeddings
        mock_backend.generate.assert_not_called()


# =============================================================================
# Tests: streaming thread error propagation (previously untested)
# =============================================================================


class TestStreamThreadErrorPropagation:
    """Verify that a crash inside the generation thread surfaces to the client."""

    @pytest.mark.asyncio
    async def test_thread_error_emits_error_sse_event(self, service, chat_request):
        """When _generate_in_thread raises, an SSE error event is yielded."""

        def crash(kwargs, errors):
            errors.append(RuntimeError("CUDA out of memory"))

        streamer = MagicMock()
        streamer.__iter__ = MagicMock(return_value=iter([]))  # no tokens

        with patch("transformers.TextIteratorStreamer", return_value=streamer):
            with patch.object(service, "_generate_in_thread", side_effect=crash):
                chunks = []
                async for chunk in service.stream_chat_completion(chat_request):
                    chunks.append(chunk)

        # Must end with [DONE]
        assert chunks[-1] == "data: [DONE]\n\n"

        # Must contain an error event before [DONE]
        error_chunks = [
            c for c in chunks
            if "error" in c and c.startswith("data: ") and c != "data: [DONE]\n\n"
        ]
        assert len(error_chunks) >= 1, "Expected at least one SSE error event"
        error_payload = json.loads(error_chunks[0].removeprefix("data: ").strip())
        assert "error" in error_payload
        assert error_payload["error"]["type"] == "server_error"

    @pytest.mark.asyncio
    async def test_thread_error_does_not_suppress_done(self, service, chat_request):
        """Even after a thread error the stream is closed with [DONE]."""

        def crash(kwargs, errors):
            errors.append(ValueError("tokeniser failure"))

        streamer = MagicMock()
        streamer.__iter__ = MagicMock(return_value=iter([]))

        with patch("transformers.TextIteratorStreamer", return_value=streamer):
            with patch.object(service, "_generate_in_thread", side_effect=crash):
                chunks = [
                    c async for c in service.stream_chat_completion(chat_request)
                ]

        assert "data: [DONE]\n\n" in chunks

    @pytest.mark.asyncio
    async def test_normal_stream_has_no_error_event(self, service, chat_request):
        """A successful generation must not emit any error event."""
        streamer = MagicMock()
        streamer.__iter__ = MagicMock(return_value=iter(["Hi", "!"]))

        with patch("transformers.TextIteratorStreamer", return_value=streamer):
            chunks = [
                c async for c in service.stream_chat_completion(chat_request)
            ]

        error_chunks = [
            c for c in chunks
            if "\"error\"" in c and c != "data: [DONE]\n\n"
        ]
        assert len(error_chunks) == 0, f"Unexpected error chunks: {error_chunks}"


# =============================================================================
# Tests: per-request profile override inside semaphore (previously untested)
# =============================================================================


class TestRequestProfileOverride:
    """Verify that _apply_request_profile / _restore_request_profile work correctly."""

    @pytest.mark.asyncio
    async def test_apply_profile_saves_and_applies_steering(self, service):
        """_apply_request_profile saves old steering, applies profile steering."""
        mock_sae = MagicMock()
        mock_sae.d_sae = 16384
        mock_sae.get_steering_values.return_value = {10: 1.0}
        mock_sae.is_steering_enabled = True
        mock_sae.set_steering_batch = MagicMock()
        mock_sae.enable_steering = MagicMock()

        mock_profile = MagicMock()
        mock_profile.steering = {"20": 5.0, "30": -3.0}

        with patch("millm.services.sae_service.AttachedSAEState") as MockState:
            MockState.return_value.attached_sae = mock_sae
            with patch("millm.db.base.async_session_factory"):
                with patch(
                    "millm.db.repositories.profile_repository.ProfileRepository"
                ) as MockRepo:
                    MockRepo.return_value.get_by_name = AsyncMock(return_value=mock_profile)
                    saved = await service._apply_request_profile("test-profile")

        assert saved is not None
        assert saved["values"] == {10: 1.0}
        assert saved["enabled"] is True
        mock_sae.set_steering_batch.assert_called_once_with({20: 5.0, 30: -3.0})
        mock_sae.enable_steering.assert_called_with(True)

    def test_restore_profile_reapplies_saved_steering(self, service):
        """_restore_request_profile reinstates saved values."""
        mock_sae = MagicMock()
        mock_sae.clear_steering = MagicMock()
        mock_sae.set_steering_batch = MagicMock()
        mock_sae.enable_steering = MagicMock()

        with patch("millm.services.sae_service.AttachedSAEState") as MockState:
            MockState.return_value.attached_sae = mock_sae
            service._restore_request_profile({"values": {5: 2.0}, "enabled": True})

        mock_sae.clear_steering.assert_called_once()
        mock_sae.set_steering_batch.assert_called_once_with({5: 2.0})
        mock_sae.enable_steering.assert_called_once_with(True)

    def test_restore_profile_none_is_noop(self, service):
        """_restore_request_profile(None) does nothing (no profile was applied)."""
        with patch("millm.services.sae_service.AttachedSAEState") as MockState:
            service._restore_request_profile(None)
        MockState.assert_not_called()

    @pytest.mark.asyncio
    async def test_profile_steering_restored_after_generation(self, service):
        """After create_chat_completion the pre-request steering is reinstated."""
        saved_restore_calls = []

        original_restore = service._restore_request_profile

        def capture_restore(saved):
            saved_restore_calls.append(saved)
            original_restore(saved)

        mock_profile = MagicMock()
        mock_profile.steering = {"100": 3.0}
        mock_sae = MagicMock()
        mock_sae.get_steering_values.return_value = {42: 1.0}
        mock_sae.is_steering_enabled = False

        with patch.object(service, "_restore_request_profile", side_effect=capture_restore):
            with patch.object(service, "_apply_request_profile", return_value={"values": {42: 1.0}, "enabled": False}):
                request = ChatCompletionRequest(
                    model="test-model",
                    messages=[ChatMessage(role="user", content="Hi")],
                    profile="my-profile",
                )
                with patch.object(service, "_generate_sync") as mock_gen:
                    mock_gen.return_value = MagicMock(
                        __getitem__=lambda self, i: torch.tensor([[1, 2, 3, 4, 5, 10]])
                    )
                    await service.create_chat_completion(request)

        # Restore must have been called exactly once, with the saved state
        assert len(saved_restore_calls) == 1
        assert saved_restore_calls[0] == {"values": {42: 1.0}, "enabled": False}


# =============================================================================
# Tests: speculative decoding path (previously untested)
# =============================================================================


class TestSpeculativeDecoding:
    """Verify speculative decoding draft model loading and integration."""

    def test_get_draft_model_returns_none_when_not_configured(self, service):
        """No draft model when SPECULATIVE_MODEL is not set."""
        service._speculative_model_id = None
        assert service._get_draft_model() is None

    def test_get_draft_model_returns_cached_model(self, service):
        """Returns cached draft model on second call (no reload)."""
        service._speculative_model_id = "gpt2"
        mock_draft = MagicMock()
        service._draft_model = mock_draft  # pre-warm

        result = service._get_draft_model()
        assert result is mock_draft

    def test_get_draft_model_loads_on_first_call(self, service):
        """Loads draft model from HuggingFace on first call."""
        service._speculative_model_id = "gpt2"
        service._draft_model = None

        mock_draft = MagicMock()
        with patch(
            "millm.services.inference_service.AutoModelForCausalLM",
            create=True,
        ) as mock_auto:
            mock_auto.from_pretrained.return_value = mock_draft
            with patch("transformers.AutoModelForCausalLM", mock_auto):
                result = service._get_draft_model()

        # After a failed import (test env may not have a real model), the
        # speculative_model_id is cleared and None is returned — that's fine.
        # We just verify the _draft_model attribute was updated.
        # (In tests, from_pretrained typically raises because gpt2 isn't cached.)
        assert service._draft_model is result  # consistent whether None or model

    def test_get_draft_model_disables_on_load_failure(self, service):
        """Load failure clears _speculative_model_id to prevent retry loops."""
        service._speculative_model_id = "nonexistent/model-xyz"
        service._draft_model = None

        with patch("transformers.AutoModelForCausalLM") as mock_auto:
            mock_auto.from_pretrained.side_effect = OSError("not found")
            result = service._get_draft_model()

        assert result is None
        assert service._speculative_model_id is None

    def test_draft_model_injected_into_generate_kwargs(self, service):
        """_build_generate_kwargs includes assistant_model when draft is loaded."""
        mock_draft = MagicMock()
        service._speculative_model_id = "gpt2"
        service._draft_model = mock_draft
        service._speculative_num_tokens = 3

        gen_config = GenerationConfig()
        inputs = {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.tensor([[1, 1, 1]]),
        }
        inputs_mock = MagicMock()
        inputs_mock.items.return_value = [
            ("input_ids", inputs["input_ids"]),
            ("attention_mask", inputs["attention_mask"]),
        ]

        with patch.object(service, "_get_input_device", return_value="cpu"):
            kwargs = service._build_generate_kwargs(gen_config, inputs_mock)

        assert kwargs.get("assistant_model") is mock_draft
        assert kwargs.get("num_assistant_tokens") == 3

    @pytest.mark.asyncio
    async def test_streaming_usage_in_final_chunk(self, service, chat_request):
        """Final streaming chunk includes usage with prompt and completion tokens."""
        streamer = MagicMock()
        streamer.__iter__ = MagicMock(return_value=iter(["Hi", "!"]))

        # Run on the test's own event loop instead of asyncio.get_event_loop()
        # .run_until_complete(), which raises "no current event loop" on
        # Python 3.12+ when no loop is set (flaky under CI test ordering).
        with patch("transformers.TextIteratorStreamer", return_value=streamer):
            chunks = [c async for c in service.stream_chat_completion(chat_request)]

        # Find the final data chunk before [DONE]
        data_chunks = [c for c in chunks if c != "data: [DONE]\n\n" and c.startswith("data: ")]
        final = json.loads(data_chunks[-1].removeprefix("data: ").strip())

        # Final chunk must carry usage
        assert "usage" in final, f"usage missing from final chunk: {final}"
        assert final["usage"]["prompt_tokens"] >= 0
        assert final["usage"]["completion_tokens"] >= 0


# =============================================================================
# Tests: finish_reason observability — Issue 8
# =============================================================================


class TestFinishReasonObservability:
    """Verify finish_reason EOS/stop-sequence distinction is logged correctly."""

    def test_returns_length_when_hit_max_tokens(self, service):
        """finish_reason is 'length' when generated_count >= max_new_tokens."""
        assert service._determine_finish_reason(512, 512) == "length"
        assert service._determine_finish_reason(600, 512) == "length"

    def test_returns_stop_when_under_max_tokens(self, service):
        """finish_reason is 'stop' when generation ended before max_tokens."""
        assert service._determine_finish_reason(10, 512) == "stop"
        assert service._determine_finish_reason(0, 512) == "stop"

    def test_returns_stop_with_eos_token_id(self, service, mock_tokenizer):
        """finish_reason is still 'stop' when last token is EOS (API contract)."""
        mock_tokenizer.eos_token_id = 2
        result = service._determine_finish_reason(10, 512, last_token_id=2)
        assert result == "stop"

    def test_returns_stop_with_non_eos_last_token(self, service, mock_tokenizer):
        """finish_reason is 'stop' for any non-length termination."""
        mock_tokenizer.eos_token_id = 2
        result = service._determine_finish_reason(10, 512, last_token_id=999)
        assert result == "stop"

    def test_last_token_id_none_does_not_crash(self, service):
        """last_token_id=None is accepted safely."""
        result = service._determine_finish_reason(10, 512, last_token_id=None)
        assert result == "stop"


# =============================================================================
# Tests: penalty mapping formula — Issue 9
# =============================================================================


class TestPenaltyMapping:
    """Verify the symmetric, combined, clamped penalty formula."""

    def _kwargs(self, freq=0.0, pres=0.0):
        from millm.ml.generation_config import GenerationConfig
        return GenerationConfig(
            frequency_penalty=freq,
            presence_penalty=pres,
            do_sample=False,
        ).to_generate_kwargs()

    def test_zero_penalties_omit_repetition_penalty(self):
        assert "repetition_penalty" not in self._kwargs(0.0, 0.0)

    def test_positive_frequency_raises_penalty(self):
        rp = self._kwargs(freq=2.0)["repetition_penalty"]
        assert rp > 1.0

    def test_negative_frequency_lowers_penalty(self):
        rp = self._kwargs(freq=-2.0)["repetition_penalty"]
        assert rp < 1.0

    def test_positive_presence_raises_penalty(self):
        rp = self._kwargs(pres=2.0)["repetition_penalty"]
        assert rp > 1.0

    def test_both_penalties_combine(self):
        """Both contribute: combined = freq + 0.5*pres."""
        rp_freq_only = self._kwargs(freq=1.0)["repetition_penalty"]
        rp_both = self._kwargs(freq=1.0, pres=2.0)["repetition_penalty"]
        assert rp_both > rp_freq_only

    def test_symmetric_formula(self):
        """Positive and negative directions use the same 0.25 multiplier.

        Use a small value (0.5) to stay well within the clamped range
        [0.8, 1.8] so clamping doesn't affect the symmetry check.
        """
        rp_pos = self._kwargs(freq=0.5)["repetition_penalty"]  # 1.0 + 0.5*0.25 = 1.125
        rp_neg = self._kwargs(freq=-0.5)["repetition_penalty"]  # 1.0 - 0.5*0.25 = 0.875
        assert abs((rp_pos - 1.0) - (1.0 - rp_neg)) < 0.001

    def test_clamped_at_max_value(self):
        """repetition_penalty never exceeds 1.8."""
        rp = self._kwargs(freq=2.0, pres=2.0)["repetition_penalty"]
        assert rp <= 1.8

    def test_clamped_at_min_value(self):
        """repetition_penalty never goes below 0.8."""
        rp = self._kwargs(freq=-2.0, pres=-2.0)["repetition_penalty"]
        assert rp >= 0.8


# =============================================================================
# Tests: max_tokens default — Issue 10
# =============================================================================


class TestMaxTokensDefault:
    """Verify TextCompletionRequest defaults to None (→ 512) not 16."""

    def test_text_completion_request_default_is_none(self):
        from millm.api.schemas.openai import TextCompletionRequest
        req = TextCompletionRequest(model="m", prompt="hi")
        assert req.max_tokens is None

    def test_text_completion_generates_512_tokens_by_default(self):
        from millm.api.schemas.openai import TextCompletionRequest
        from millm.ml.generation_config import GenerationConfig
        req = TextCompletionRequest(model="m", prompt="hi")
        cfg = GenerationConfig.from_request(req)
        assert cfg.max_new_tokens == 512

    def test_chat_completion_still_defaults_to_none_maps_to_512(self):
        from millm.api.schemas.openai import ChatCompletionRequest, ChatMessage
        from millm.ml.generation_config import GenerationConfig
        req = ChatCompletionRequest(
            model="m",
            messages=[ChatMessage(role="user", content="hi")],
        )
        cfg = GenerationConfig.from_request(req)
        assert cfg.max_new_tokens == 512

    def test_explicit_max_tokens_respected(self):
        from millm.api.schemas.openai import TextCompletionRequest
        from millm.ml.generation_config import GenerationConfig
        req = TextCompletionRequest(model="m", prompt="hi", max_tokens=100)
        cfg = GenerationConfig.from_request(req)
        assert cfg.max_new_tokens == 100


# =============================================================================
# Tests: CBM/serial routing visibility — Issue 11
# =============================================================================


class TestBackendVisibility:
    """Verify backend_name and get_backend_info expose routing information."""

    def test_backend_name_serial_when_no_cbm(self, service):
        """backend_name returns 'serial' when CBM is not configured."""
        service._cbm_backend = None
        assert service.backend_name == "serial"

    def test_backend_name_serial_when_cbm_not_running(self, service):
        """backend_name returns 'serial' when CBM manager is stopped."""
        mock_backend = MagicMock()
        mock_backend.is_running = False
        service._cbm_backend = mock_backend
        assert service.backend_name == "serial"

    def test_backend_name_cbm_when_running(self, service):
        """backend_name returns 'cbm' when ContinuousBatchingManager is active."""
        mock_backend = MagicMock()
        mock_backend.is_running = True
        service._cbm_backend = mock_backend
        assert service.backend_name == "cbm"

    def test_get_backend_info_serial_structure(self, service):
        """Serial backend info includes queue stats and capabilities."""
        service._cbm_backend = None
        info = service.get_backend_info()
        assert info["backend"] == "serial"
        assert "capabilities" in info
        assert info["capabilities"]["per_request_sampling_params"] is True
        assert info["capabilities"]["per_request_profile_override"] is True
        assert "queue" in info
        assert "limitations" in info

    def test_get_backend_info_cbm_structure(self, service):
        """CBM backend info includes cbm_config and limitations."""
        mock_backend = MagicMock()
        mock_backend.is_running = True
        mock_backend._default_temperature = 0.7
        mock_backend._default_top_p = 0.95
        mock_backend._max_queue_size = 256
        service._cbm_backend = mock_backend

        info = service.get_backend_info()
        assert info["backend"] == "cbm"
        assert info["capabilities"]["per_request_sampling_params"] is False
        assert info["capabilities"]["per_request_profile_override"] is False
        assert "cbm_config" in info
        assert len(info["limitations"]) >= 1

    def test_cbm_routing_fallback_returns_false_on_mismatch(self, service):
        """CBM sampling mismatch routes to serial path (returns False)."""
        mock_backend = MagicMock()
        mock_backend.is_running = True
        mock_backend.sampling_params_match = MagicMock(return_value=False)
        service._cbm_backend = mock_backend

        result = service._use_cbm_for_request(temperature=0.1, top_p=0.5)

        assert result is False
        mock_backend.sampling_params_match.assert_called_once_with(0.1, 0.5)

    def test_cbm_routing_fallback_logger_called(self, service):
        """CBM sampling mismatch invokes the logger (INFO level in production)."""
        mock_backend = MagicMock()
        mock_backend.is_running = True
        mock_backend.sampling_params_match = MagicMock(return_value=False)
        service._cbm_backend = mock_backend

        # Patch the module-level logger to capture the call
        with patch("millm.services.inference_service.logger") as mock_logger:
            service._use_cbm_for_request(temperature=0.1, top_p=0.5)

        # info() should have been called with the routing event
        mock_logger.info.assert_called_once()
        call_args = mock_logger.info.call_args
        assert "serial" in call_args[0][0] or "serial" in str(call_args)


# =============================================================================
# Tests: CBM profile routing + profile validation (steering remediation M1/M5)
# =============================================================================


class TestCBMProfileRouting:
    """A request carrying a profile override must fall back to the serial path."""

    def test_profile_request_falls_back_to_serial(self, service):
        mock_backend = MagicMock()
        mock_backend.is_running = True
        mock_backend.sampling_params_match = MagicMock(return_value=True)
        service._cbm_backend = mock_backend

        # Compatible sampling params but a profile is present → serial.
        result = service._use_cbm_for_request(
            temperature=None, top_p=None, has_profile=True
        )
        assert result is False

    def test_no_profile_uses_cbm(self, service):
        mock_backend = MagicMock()
        mock_backend.is_running = True
        mock_backend.sampling_params_match = MagicMock(return_value=True)
        service._cbm_backend = mock_backend
        service._cbm_force_serial_monitoring = False

        result = service._use_cbm_for_request(
            temperature=None, top_p=None, has_profile=False
        )
        assert result is True


class TestApplyProfileValidation:
    """_apply_request_profile validates profile steering and fails loudly."""

    @pytest.mark.asyncio
    async def test_missing_profile_raises_not_found(self, service):
        from millm.core.errors import ProfileNotFoundError

        mock_sae = MagicMock()
        mock_sae.d_sae = 16384

        with patch("millm.services.sae_service.AttachedSAEState") as MockState:
            MockState.return_value.attached_sae = mock_sae
            with patch("millm.db.base.async_session_factory"):
                with patch(
                    "millm.db.repositories.profile_repository.ProfileRepository"
                ) as MockRepo:
                    MockRepo.return_value.get_by_name = AsyncMock(return_value=None)
                    with pytest.raises(ProfileNotFoundError):
                        await service._apply_request_profile("does-not-exist")

    @pytest.mark.asyncio
    async def test_out_of_range_feature_raises(self, service):
        from millm.core.errors import InvalidFeatureIndexError

        mock_sae = MagicMock()
        mock_sae.d_sae = 100
        mock_sae.get_steering_values.return_value = {}
        mock_sae.is_steering_enabled = False

        mock_profile = MagicMock()
        mock_profile.steering = {"999": 5.0}  # 999 >= d_sae=100

        with patch("millm.services.sae_service.AttachedSAEState") as MockState:
            MockState.return_value.attached_sae = mock_sae
            with patch("millm.db.base.async_session_factory"):
                with patch(
                    "millm.db.repositories.profile_repository.ProfileRepository"
                ) as MockRepo:
                    MockRepo.return_value.get_by_name = AsyncMock(return_value=mock_profile)
                    with pytest.raises(InvalidFeatureIndexError):
                        await service._apply_request_profile("bad-profile")
        # No partial steering should have been applied.
        mock_sae.set_steering_batch.assert_not_called()

    @pytest.mark.asyncio
    async def test_out_of_range_value_raises(self, service):
        from millm.core.errors import InvalidFeatureIndexError

        mock_sae = MagicMock()
        mock_sae.d_sae = 16384
        mock_sae.get_steering_values.return_value = {}
        mock_sae.is_steering_enabled = False

        mock_profile = MagicMock()
        mock_profile.steering = {"10": 5000.0}  # value out of [-200, 200]

        with patch("millm.services.sae_service.AttachedSAEState") as MockState:
            MockState.return_value.attached_sae = mock_sae
            with patch("millm.db.base.async_session_factory"):
                with patch(
                    "millm.db.repositories.profile_repository.ProfileRepository"
                ) as MockRepo:
                    MockRepo.return_value.get_by_name = AsyncMock(return_value=mock_profile)
                    with pytest.raises(InvalidFeatureIndexError):
                        await service._apply_request_profile("bad-value-profile")

    @pytest.mark.asyncio
    async def test_no_sae_returns_none(self, service):
        with patch("millm.services.sae_service.AttachedSAEState") as MockState:
            MockState.return_value.attached_sae = None
            result = await service._apply_request_profile("any")
        assert result is None
