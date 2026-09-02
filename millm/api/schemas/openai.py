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

from typing import Any, Literal, Optional, Union

from pydantic import BaseModel, Field, model_validator, field_validator


# =============================================================================
# Message Types
# =============================================================================


class ChatMessage(BaseModel):
    """Chat message with role and content."""

    role: Literal["system", "user", "assistant", "tool", "function"]
    content: Optional[str] = None

    # Reasoning trace, separated from the answer. NOT an OpenAI field -- a
    # de-facto convention from DeepSeek, adopted by vLLM/SGLang, and what
    # Open WebUI renders as a collapsible "Thinking" section. Serialised only
    # when present (responses use exclude_none), so non-reasoning models and
    # older clients see exactly the payload they saw before.
    reasoning_content: Optional[str] = None

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
    n: int = Field(default=1, ge=1)
    max_tokens: Optional[int] = Field(default=None, gt=0)
    stop: Optional[Union[str, list[str]]] = None
    frequency_penalty: float = Field(default=0.0, ge=-2.0, le=2.0)
    presence_penalty: float = Field(default=0.0, ge=-2.0, le=2.0)
    user: Optional[str] = None

    # miLLM extension - BATCHED generation.
    #
    # A list of additional conversations to generate alongside `messages`, in
    # ONE batched forward pass. The weights are read once and amortised across
    # the batch: measured 5.59x aggregate throughput at batch 8 on
    # gemma-4-12B-it (4.63 -> 25.90 tok/s), with GPU utilisation moving from
    # 29.7% to 51.5% mean. N independent concurrent requests do NOT achieve
    # this — each re-reads the full weights — which is why this is a batch
    # field and not a higher concurrency limit.
    #
    # A batch is still ONE request holding ONE queue slot, so
    # MAX_CONCURRENT_REQUESTS stays at 1 and the steering isolation it provides
    # is untouched. Steering applies uniformly to every row, which is correct
    # for a single request; sensing is refused for a batch (it cannot attribute
    # positions to a row) and says so in the log.
    #
    # Response carries one choice per conversation, `index` in input order:
    # index 0 is `messages`, index i is `extra_messages[i-1]`. Clients must
    # demultiplex on `index`, never on wire order.
    #
    # IMPORTANT for clients: this schema is extra="ignore", so a server that
    # predates this field ACCEPTS it and silently returns a single choice.
    # Detect support via the X-miLLM-Batch response header before relying on it.
    extra_messages: Optional[list[list[ChatMessage]]] = None

    # miLLM extension - chat-template variables (the vLLM/SGLang convention).
    #
    # Forwarded verbatim as keyword arguments to
    # `tokenizer.apply_chat_template(...)`, which exposes them as Jinja
    # variables to the model's own template. This is the ONLY way to reach a
    # control that lives in the template rather than in the generation config.
    #
    # The motivating case is reasoning. granite-4.2-8b's template sets
    # `enable_thinking` to True when undefined, and its generation prompt then
    # ends with an OPEN `<think>` tag, so the model resumes inside a reasoning
    # block and emits paragraphs of deliberation before any answer. Because the
    # opening tag is in the PROMPT and not the completion, a client stripping
    # `<think>...</think>` from the response finds no opening tag and strips
    # nothing — the reasoning arrives as ordinary untagged prose and lands in
    # the caller's parsed output. Send {"enable_thinking": false} and the
    # template emits `<think></think>` instead, so the model answers directly.
    #
    # Also honoured by that template: {"reasoning_effort": "low"} (equivalently
    # {"low_effort": true}) for abbreviated reasoning.
    #
    # Unknown keys are harmless — Jinja ignores variables a template does not
    # reference — but that cuts both ways: a model whose template has no
    # `enable_thinking` will accept the flag and keep thinking. This field
    # cannot promise an effect, only delivery. What it does guarantee is that a
    # template which RAISES on these kwargs fails loudly instead of silently
    # falling back to a generic format with the request ignored.
    chat_template_kwargs: Optional[dict[str, Any]] = None

    # miLLM extension - steering profile override
    profile: Optional[str] = None

    # miLLM extension (Feature 10) - per-request cluster intensity dial.
    # Numeric lambda in [0, 2], or a symbolic position resolved server-side
    # against the active cluster's declared intensity_range: "off" -> 0,
    # "min"/"max" -> the range bounds. Applied and restored inside the
    # request boundary - concurrent requests never see each other's dial.
    steering_intensity: Optional[Union[float, Literal["off", "min", "max"]]] = None

    @field_validator("steering_intensity", mode="before")
    @classmethod
    def _reject_bool_steering_intensity(cls, v):
        # bool is an int subclass — without this, true silently becomes 1.0.
        if isinstance(v, bool):
            raise ValueError("steering_intensity must be a number or off|min|max")
        return v

    @field_validator("steering_intensity")
    @classmethod
    def _validate_steering_intensity(cls, v):
        # Runs after Union coercion, so numeric strings ("1.5") are floats here.
        if isinstance(v, float) and not 0.0 <= v <= 2.0:
            raise ValueError("steering_intensity must be within [0, 2]")
        return v

    model_config = {"extra": "ignore"}

    @model_validator(mode="after")
    def validate_stop_sequences(self) -> "ChatCompletionRequest":
        """Limit stop sequences to 4 (OpenAI limit)."""
        if isinstance(self.stop, list) and len(self.stop) > 4:
            raise ValueError("Maximum 4 stop sequences allowed")
        return self


class TextCompletionRequest(BaseModel):
    """Text completion request - OpenAI format (legacy completions endpoint).

    Known deviation from the OpenAI API: max_tokens defaults to None (→ 512
    in GenerationConfig) rather than the OpenAI spec value of 16.  The OpenAI
    default is counterproductive for a self-hosted LLM where users always want
    more than 16 tokens.  Clients that need the original 16-token behaviour
    must set max_tokens=16 explicitly.
    """

    model: str
    prompt: Union[str, list[str]]
    stream: bool = False
    n: int = Field(default=1, ge=1)
    max_tokens: Optional[int] = Field(default=None, gt=0)
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
    encoding_format: Literal["float", "base64"] = "float"
    dimensions: Optional[int] = Field(default=None, gt=0)
    user: Optional[str] = None

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
    reasoning_content: Optional[str] = None


class ChatCompletionChunkChoice(BaseModel):
    """Single choice in streaming chunk."""

    index: int
    delta: ChatCompletionChunkDelta
    finish_reason: Optional[Literal["stop", "length", "timeout"]] = None


class ChatCompletionChunk(BaseModel):
    """
    Streaming chunk response.

    Usage: chunk.model_dump_json(exclude_none=True) for SSE data field.

    The final chunk (the one that carries finish_reason) also carries a
    `usage` field so clients can track token consumption from streamed
    responses.  Intermediate chunks leave `usage=None` which is excluded
    by exclude_none=True.
    """

    id: str
    object: Literal["chat.completion.chunk"] = "chat.completion.chunk"
    created: int
    model: str
    choices: list[ChatCompletionChunkChoice]
    usage: Optional["Usage"] = None


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
    """Single embedding result. Embedding is float list or base64 string."""

    object: Literal["embedding"] = "embedding"
    index: int
    embedding: Union[list[float], str]


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
