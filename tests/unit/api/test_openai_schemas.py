"""
Unit tests for OpenAI-compatible API schemas.

Tests validation, serialization, and edge cases for all OpenAI schemas.
"""

import pytest
from pydantic import ValidationError

from millm.api.schemas.openai import (
    ChatCompletionChoice,
    ChatCompletionChunk,
    ChatCompletionChunkChoice,
    ChatCompletionChunkDelta,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
    EmbeddingData,
    EmbeddingRequest,
    EmbeddingResponse,
    ModelListResponse,
    ModelObject,
    OpenAIError,
    OpenAIErrorResponse,
    TextCompletionChoice,
    TextCompletionRequest,
    TextCompletionResponse,
    Usage,
)


class TestChatMessage:
    """Tests for ChatMessage schema."""

    def test_valid_user_message(self):
        msg = ChatMessage(role="user", content="Hello")
        assert msg.role == "user"
        assert msg.content == "Hello"

    def test_valid_assistant_message(self):
        msg = ChatMessage(role="assistant", content="Hi there!")
        assert msg.role == "assistant"
        assert msg.content == "Hi there!"

    def test_valid_system_message(self):
        msg = ChatMessage(role="system", content="You are a helpful assistant.")
        assert msg.role == "system"
        assert msg.content == "You are a helpful assistant."

    def test_invalid_role_raises_error(self):
        with pytest.raises(ValidationError):
            ChatMessage(role="invalid", content="Test")

    def test_extra_fields_ignored(self):
        msg = ChatMessage(role="user", content="Hello", name="John", extra_field="ignored")
        assert msg.role == "user"
        assert msg.content == "Hello"
        assert not hasattr(msg, "extra_field")


class TestChatCompletionRequest:
    """Tests for ChatCompletionRequest schema."""

    def test_minimal_valid_request(self):
        request = ChatCompletionRequest(
            model="gpt-4",
            messages=[ChatMessage(role="user", content="Hello")],
        )
        assert request.model == "gpt-4"
        assert len(request.messages) == 1
        assert request.stream is False
        assert request.temperature == 1.0
        assert request.max_tokens is None

    def test_full_request_with_all_parameters(self):
        request = ChatCompletionRequest(
            model="gpt-4",
            messages=[
                ChatMessage(role="system", content="You are helpful."),
                ChatMessage(role="user", content="Hello"),
            ],
            stream=True,
            temperature=0.7,
            top_p=0.9,
            max_tokens=100,
            stop=["END"],
            frequency_penalty=0.5,
            presence_penalty=0.5,
        )
        assert request.stream is True
        assert request.temperature == 0.7
        assert request.max_tokens == 100
        assert request.stop == ["END"]

    def test_temperature_validation_min(self):
        with pytest.raises(ValidationError):
            ChatCompletionRequest(
                model="gpt-4",
                messages=[ChatMessage(role="user", content="Hello")],
                temperature=-0.1,
            )

    def test_temperature_validation_max(self):
        with pytest.raises(ValidationError):
            ChatCompletionRequest(
                model="gpt-4",
                messages=[ChatMessage(role="user", content="Hello")],
                temperature=2.1,
            )

    def test_temperature_zero_valid(self):
        request = ChatCompletionRequest(
            model="gpt-4",
            messages=[ChatMessage(role="user", content="Hello")],
            temperature=0.0,
        )
        assert request.temperature == 0.0

    def test_stop_string_accepted(self):
        # Schema accepts string for stop, conversion to list happens in GenerationConfig
        request = ChatCompletionRequest(
            model="gpt-4",
            messages=[ChatMessage(role="user", content="Hello")],
            stop="STOP",
        )
        assert request.stop == "STOP"

    def test_stop_list_preserved(self):
        request = ChatCompletionRequest(
            model="gpt-4",
            messages=[ChatMessage(role="user", content="Hello")],
            stop=["END", "STOP"],
        )
        assert request.stop == ["END", "STOP"]

    def test_stop_max_four_sequences(self):
        with pytest.raises(ValidationError):
            ChatCompletionRequest(
                model="gpt-4",
                messages=[ChatMessage(role="user", content="Hello")],
                stop=["1", "2", "3", "4", "5"],
            )

    def test_extra_fields_ignored(self):
        request = ChatCompletionRequest(
            model="gpt-4",
            messages=[ChatMessage(role="user", content="Hello")],
            logprobs=True,  # Unsupported field
            n=5,  # Unsupported field
        )
        assert request.model == "gpt-4"


class TestChatCompletionResponse:
    """Tests for ChatCompletionResponse schema."""

    def test_valid_response(self):
        response = ChatCompletionResponse(
            id="chatcmpl-123abc",
            created=1700000000,
            model="gpt-4",
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message=ChatMessage(role="assistant", content="Hello!"),
                    finish_reason="stop",
                )
            ],
            usage=Usage(prompt_tokens=10, completion_tokens=5),
        )
        assert response.object == "chat.completion"
        assert response.id == "chatcmpl-123abc"
        assert len(response.choices) == 1
        assert response.usage.total_tokens == 15

    def test_response_serialization_format(self):
        response = ChatCompletionResponse(
            id="chatcmpl-123",
            created=1700000000,
            model="test-model",
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message=ChatMessage(role="assistant", content="Test"),
                    finish_reason="stop",
                )
            ],
            usage=Usage(prompt_tokens=5, completion_tokens=3),
        )
        data = response.model_dump()
        assert data["object"] == "chat.completion"
        assert "choices" in data
        assert data["choices"][0]["message"]["role"] == "assistant"


class TestChatCompletionChunk:
    """Tests for streaming chunk schemas."""

    def test_chunk_with_role_only(self):
        chunk = ChatCompletionChunk(
            id="chatcmpl-123",
            created=1700000000,
            model="gpt-4",
            choices=[
                ChatCompletionChunkChoice(
                    index=0,
                    delta=ChatCompletionChunkDelta(role="assistant"),
                    finish_reason=None,
                )
            ],
        )
        assert chunk.object == "chat.completion.chunk"
        assert chunk.choices[0].delta.role == "assistant"
        assert chunk.choices[0].delta.content is None

    def test_chunk_with_content(self):
        chunk = ChatCompletionChunk(
            id="chatcmpl-123",
            created=1700000000,
            model="gpt-4",
            choices=[
                ChatCompletionChunkChoice(
                    index=0,
                    delta=ChatCompletionChunkDelta(content="Hello"),
                    finish_reason=None,
                )
            ],
        )
        assert chunk.choices[0].delta.content == "Hello"

    def test_final_chunk_with_finish_reason(self):
        chunk = ChatCompletionChunk(
            id="chatcmpl-123",
            created=1700000000,
            model="gpt-4",
            choices=[
                ChatCompletionChunkChoice(
                    index=0,
                    delta=ChatCompletionChunkDelta(),
                    finish_reason="stop",
                )
            ],
        )
        assert chunk.choices[0].finish_reason == "stop"

    def test_chunk_serialization(self):
        chunk = ChatCompletionChunk(
            id="chatcmpl-123",
            created=1700000000,
            model="gpt-4",
            choices=[
                ChatCompletionChunkChoice(
                    index=0,
                    delta=ChatCompletionChunkDelta(content="test"),
                    finish_reason=None,
                )
            ],
        )
        json_str = chunk.model_dump_json()
        assert '"object":"chat.completion.chunk"' in json_str or '"object": "chat.completion.chunk"' in json_str


class TestTextCompletionRequest:
    """Tests for TextCompletionRequest schema."""

    def test_string_prompt(self):
        request = TextCompletionRequest(
            model="gpt-3.5-turbo-instruct",
            prompt="Once upon a time",
        )
        assert request.prompt == "Once upon a time"

    def test_list_prompt(self):
        request = TextCompletionRequest(
            model="gpt-3.5-turbo-instruct",
            prompt=["Prompt 1", "Prompt 2"],
        )
        assert request.prompt == ["Prompt 1", "Prompt 2"]

    def test_stop_string_accepted(self):
        # Schema accepts string for stop, conversion to list happens in GenerationConfig
        request = TextCompletionRequest(
            model="gpt-3.5-turbo-instruct",
            prompt="Test",
            stop="END",
        )
        assert request.stop == "END"


class TestTextCompletionResponse:
    """Tests for TextCompletionResponse schema."""

    def test_valid_response(self):
        response = TextCompletionResponse(
            id="cmpl-123abc",
            created=1700000000,
            model="gpt-3.5-turbo-instruct",
            choices=[
                TextCompletionChoice(
                    index=0,
                    text="there was a dragon",
                    finish_reason="stop",
                )
            ],
            usage=Usage(prompt_tokens=4, completion_tokens=5),
        )
        assert response.object == "text_completion"
        assert response.choices[0].text == "there was a dragon"


class TestEmbeddingRequest:
    """Tests for EmbeddingRequest schema."""

    def test_string_input(self):
        request = EmbeddingRequest(
            model="text-embedding-ada-002",
            input="Hello world",
        )
        assert request.input == "Hello world"

    def test_list_input(self):
        request = EmbeddingRequest(
            model="text-embedding-ada-002",
            input=["Hello", "World"],
        )
        assert request.input == ["Hello", "World"]


class TestEmbeddingResponse:
    """Tests for EmbeddingResponse schema."""

    def test_single_embedding(self):
        response = EmbeddingResponse(
            data=[
                EmbeddingData(index=0, embedding=[0.1, 0.2, 0.3])
            ],
            model="text-embedding-ada-002",
            usage=Usage(prompt_tokens=2, completion_tokens=0),
        )
        assert response.object == "list"
        assert len(response.data) == 1
        assert response.data[0].object == "embedding"
        assert len(response.data[0].embedding) == 3

    def test_multiple_embeddings(self):
        response = EmbeddingResponse(
            data=[
                EmbeddingData(index=0, embedding=[0.1, 0.2]),
                EmbeddingData(index=1, embedding=[0.3, 0.4]),
            ],
            model="text-embedding-ada-002",
            usage=Usage(prompt_tokens=4, completion_tokens=0),
        )
        assert len(response.data) == 2


class TestModelSchemas:
    """Tests for model listing schemas."""

    def test_model_object(self):
        model = ModelObject(
            id="gpt-4",
            created=1700000000,
            owned_by="openai",
        )
        assert model.object == "model"
        assert model.id == "gpt-4"

    def test_model_list_response(self):
        response = ModelListResponse(
            data=[
                ModelObject(id="gpt-4", created=1700000000, owned_by="openai"),
                ModelObject(id="gpt-3.5-turbo", created=1699000000, owned_by="openai"),
            ]
        )
        assert response.object == "list"
        assert len(response.data) == 2

    def test_empty_model_list(self):
        response = ModelListResponse(data=[])
        assert response.object == "list"
        assert len(response.data) == 0


class TestUsage:
    """Tests for Usage schema."""

    def test_auto_compute_total_tokens(self):
        usage = Usage(prompt_tokens=10, completion_tokens=5)
        assert usage.total_tokens == 15

    def test_explicit_total_tokens(self):
        usage = Usage(prompt_tokens=10, completion_tokens=5, total_tokens=20)
        assert usage.total_tokens == 20  # Explicit value preserved

    def test_zero_tokens(self):
        usage = Usage(prompt_tokens=0, completion_tokens=0)
        assert usage.total_tokens == 0


class TestErrorSchemas:
    """Tests for error schemas."""

    def test_openai_error(self):
        error = OpenAIError(
            message="Invalid request",
            type="invalid_request_error",
            param="temperature",
            code="invalid_parameter",
        )
        assert error.message == "Invalid request"
        assert error.param == "temperature"

    def test_openai_error_response(self):
        response = OpenAIErrorResponse(
            error=OpenAIError(
                message="Model not found",
                type="invalid_request_error",
                param="model",
                code="model_not_found",
            )
        )
        data = response.model_dump()
        assert "error" in data
        assert data["error"]["message"] == "Model not found"
