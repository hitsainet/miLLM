"""
OpenAI API compatible schemas.

Provides Pydantic models for OpenAI-compatible API endpoints.
All schemas match the OpenAI API specification for client compatibility.

Key implementation notes:
1. Use Literal types for fixed string values
2. model_dump() replaces deprecated .dict()
3. model_dump_json() for SSE chunk serialization
4. Field() with ge/le for range validation
5. extra="ignore" allows unknown fields for forward compatibility
"""

from typing import Literal, Optional, Union

from pydantic import BaseModel, Field, model_validator


# =============================================================================
# Message Types
# =============================================================================


class ChatMessage(BaseModel):
    """Chat message with role and content."""

    role: Literal["system", "user", "assistant", "tool", "function"]
    content: Optional[str] = None

    # Allow extra fields (OpenAI clients may send name, function_call, etc.)
    model_config = {"extra": "ignore"}


# =============================================================================
# Request Schemas
# =============================================================================


class ChatCompletionRequest(BaseModel):
    """
    Chat completion request - OpenAI format.

    Supports all standard OpenAI chat completion parameters.
    Unsupported fields are ignored with extra="ignore".
    """

    model: str
    messages: list[ChatMessage]
    stream: bool = False
    temperature: float = Field(default=1.0, ge=0.0, le=2.0)
    top_p: float = Field(default=1.0, ge=0.0, le=1.0)
    max_tokens: Optional[int] = Field(default=None, gt=0)
    stop: Optional[Union[str, list[str]]] = None
    frequency_penalty: float = Field(default=0.0, ge=-2.0, le=2.0)
    presence_penalty: float = Field(default=0.0, ge=-2.0, le=2.0)
    user: Optional[str] = None

    # miLLM extension - steering profile override
    profile: Optional[str] = None

    model_config = {"extra": "ignore"}

    @model_validator(mode="after")
    def validate_stop_sequences(self) -> "ChatCompletionRequest":
        """Limit stop sequences to 4 (OpenAI limit)."""
        if isinstance(self.stop, list) and len(self.stop) > 4:
            raise ValueError("Maximum 4 stop sequences allowed")
        return self


class TextCompletionRequest(BaseModel):
    """Text completion request - OpenAI format (legacy completions endpoint)."""

    model: str
    prompt: Union[str, list[str]]
    stream: bool = False
    max_tokens: Optional[int] = Field(default=16, gt=0)
    temperature: float = Field(default=1.0, ge=0.0, le=2.0)
    top_p: float = Field(default=1.0, ge=0.0, le=1.0)
    stop: Optional[Union[str, list[str]]] = None
    frequency_penalty: float = Field(default=0.0, ge=-2.0, le=2.0)
    presence_penalty: float = Field(default=0.0, ge=-2.0, le=2.0)
    user: Optional[str] = None

    model_config = {"extra": "ignore"}

    @model_validator(mode="after")
    def validate_stop_sequences(self) -> "TextCompletionRequest":
        """Limit stop sequences to 4 (OpenAI limit)."""
        if isinstance(self.stop, list) and len(self.stop) > 4:
            raise ValueError("Maximum 4 stop sequences allowed")
        return self


class EmbeddingRequest(BaseModel):
    """Embedding request - OpenAI format."""

    model: str
    input: Union[str, list[str]]

    model_config = {"extra": "ignore"}


# =============================================================================
# Response Schemas - Token Usage
# =============================================================================


class Usage(BaseModel):
    """Token usage statistics."""

    prompt_tokens: int
    completion_tokens: int = 0
    total_tokens: int = 0

    @model_validator(mode="after")
    def compute_total(self) -> "Usage":
        """Auto-compute total if not provided."""
        if self.total_tokens == 0:
            object.__setattr__(
                self, "total_tokens", self.prompt_tokens + self.completion_tokens
            )
        return self


# =============================================================================
# Response Schemas - Chat Completions
# =============================================================================


class ChatCompletionChoice(BaseModel):
    """Single completion choice in non-streaming response."""

    index: int
    message: ChatMessage
    finish_reason: Literal["stop", "length", "timeout"]


class ChatCompletionResponse(BaseModel):
    """
    Non-streaming chat completion response.

    The `id` format is "chatcmpl-{24 hex chars}".
    """

    id: str
    object: Literal["chat.completion"] = "chat.completion"
    created: int  # Unix timestamp
    model: str
    choices: list[ChatCompletionChoice]
    usage: Usage


# =============================================================================
# Response Schemas - Streaming Chat Completions
# =============================================================================


class ChatCompletionChunkDelta(BaseModel):
    """
    Delta for streaming chunks.

    Streaming pattern:
    - First chunk: role="assistant", content=None
    - Middle chunks: role=None, content="token"
    - Final chunk: role=None, content=None (finish_reason set on choice)
    """

    role: Optional[Literal["assistant"]] = None
    content: Optional[str] = None


class ChatCompletionChunkChoice(BaseModel):
    """Single choice in streaming chunk."""

    index: int
    delta: ChatCompletionChunkDelta
    finish_reason: Optional[Literal["stop", "length", "timeout"]] = None


class ChatCompletionChunk(BaseModel):
    """
    Streaming chunk response.

    Usage: chunk.model_dump_json() for SSE data field.
    """

    id: str
    object: Literal["chat.completion.chunk"] = "chat.completion.chunk"
    created: int
    model: str
    choices: list[ChatCompletionChunkChoice]


# =============================================================================
# Response Schemas - Text Completions
# =============================================================================


class TextCompletionChoice(BaseModel):
    """Single completion choice in text completion response."""

    index: int
    text: str
    finish_reason: Literal["stop", "length", "timeout"]


class TextCompletionResponse(BaseModel):
    """
    Non-streaming text completion response.

    The `id` format is "cmpl-{24 hex chars}".
    """

    id: str
    object: Literal["text_completion"] = "text_completion"
    created: int  # Unix timestamp
    model: str
    choices: list[TextCompletionChoice]
    usage: Usage


# =============================================================================
# Response Schemas - Embeddings
# =============================================================================


class EmbeddingData(BaseModel):
    """Single embedding result."""

    object: Literal["embedding"] = "embedding"
    index: int
    embedding: list[float]


class EmbeddingResponse(BaseModel):
    """Embedding response containing one or more embeddings."""

    object: Literal["list"] = "list"
    data: list[EmbeddingData]
    model: str
    usage: Usage


# =============================================================================
# Response Schemas - Models
# =============================================================================


class ModelObject(BaseModel):
    """Model metadata for /v1/models endpoint."""

    id: str
    object: Literal["model"] = "model"
    created: int  # Unix timestamp
    owned_by: str = "miLLM"


class ModelListResponse(BaseModel):
    """List of available models."""

    object: Literal["list"] = "list"
    data: list[ModelObject]


# =============================================================================
# Error Schemas
# =============================================================================


class OpenAIError(BaseModel):
    """OpenAI-format error detail."""

    message: str
    type: Literal[
        "invalid_request_error",
        "authentication_error",
        "rate_limit_error",
        "server_error",
    ]
    param: Optional[str] = None
    code: Optional[str] = None


class OpenAIErrorResponse(BaseModel):
    """OpenAI-format error response wrapper."""

    error: OpenAIError
