"""
OpenAI API compatible schemas.

Provides Pydantic models for OpenAI-compatible API endpoints.
All schemas match the OpenAI API specification for client compatibility.

Key implementation notes:
1. Use Literal types for fixed string values
2. model_dump() replaces deprecated .dict()
3. model_dump_json() for SSE chunk serialization
4. Field() with ge/le for range validation
5. extra="allow" (Feature 25): unknown fields are ACCEPTED and kept in `model_extra`, so
   `millm/api/request_policy.py` can report them (X-miLLM-Ignored-Fields) or, under
   X-miLLM-Strict, refuse them. They were `extra="ignore"`, which dropped them with no trace.
   Nothing in millm dumps a request object, so an extra never reaches generation
   (test_unused_fields_http pins that a message extra never reaches apply_chat_template).
"""

from typing import Annotated, Any, Literal, Optional, Union

from pydantic import BaseModel, Field, ValidationInfo, field_validator, model_validator

from millm.api.schemas.millm_extension import MillmExtension, ReturnSaeActivations


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

    # Extra fields (OpenAI clients send name, function_call, ...) are kept so the request
    # policy can REPORT them as messages[i].<key>; they are never passed to the template.
    model_config = {"extra": "allow"}


# =============================================================================
# Structured output (Feature 25, FR-25.10)
# =============================================================================


class ResponseFormatText(BaseModel):
    """No constraint — the neutral value (FR-25.3.4)."""

    type: Literal["text"]
    model_config = {"extra": "forbid"}


class ResponseFormatJsonObject(BaseModel):
    """The output parses as one JSON object."""

    type: Literal["json_object"]
    model_config = {"extra": "forbid"}


class JsonSchemaSpec(BaseModel):
    """OpenAI's `json_schema` block. `strict: false` is accepted and the schema is STILL
    enforced (FR-25.10.4): enforcing more than asked is not a silent drop.

    `schema` is held as `schema_` because `schema` shadows a BaseModel attribute; requests are
    never dumped, so the alias only matters on input. Unknown keys are refused (`forbid`), so a
    misspelt field inside the block cannot be dropped silently.
    """

    name: str = Field(pattern=r"^[A-Za-z0-9_-]{1,64}$")
    schema_: dict[str, Any] = Field(alias="schema")
    description: Optional[str] = None
    strict: Optional[bool] = None
    model_config = {"extra": "forbid", "populate_by_name": True}


class ResponseFormatJsonSchema(BaseModel):
    type: Literal["json_schema"]
    json_schema: JsonSchemaSpec
    model_config = {"extra": "forbid"}


ResponseFormat = Annotated[
    Union[ResponseFormatText, ResponseFormatJsonObject, ResponseFormatJsonSchema],
    Field(discriminator="type"),
]


# =============================================================================
# Request Schemas
# =============================================================================

#: Below this, dividing the logits by the temperature can overflow them; 0 means "no scaling".
MIN_SCORING_TEMPERATURE = 1e-3

# `user` is deliberately NOT declared on any request (T-57, FR-25.3.3b): nothing in millm reads
# it, so it lands in `model_extra` and is reported as unused — and refused under X-miLLM-Strict.


#: The seed range (FR-25.13.1, 025_FTDD U7): 0 to 2**32-1 — fits torch and llama.cpp. Outside it
#: the request is refused, never wrapped or clamped.
SEED_MAX = 2**32 - 1


def _reject_bool_seed(v: Any) -> Any:
    # bool is an int subclass: without this, `"seed": true` silently becomes seed 1.
    if isinstance(v, bool):
        raise ValueError("seed must be an integer")
    return v


def _fold_max_completion_tokens(data: Any) -> Any:
    """`max_completion_tokens` is honoured AS `max_tokens` (T-58, FR-25.3.3a).

    Runs before field validation so every downstream reader — including the scoring check that
    requires max_tokens=1 — sees one number. Sent together with a DIFFERENT `max_tokens`, the
    request is refused naming both: picking one would silently drop the other.
    """
    if not isinstance(data, dict):
        return data
    mct = data.get("max_completion_tokens")
    if mct is None:
        return data
    mt = data.get("max_tokens")
    if mt is not None and mt != mct:
        raise ValueError(
            f"max_completion_tokens ({mct}) and max_tokens ({mt}) disagree; send one, or the "
            "same value in both"
        )
    if mt is None:
        data = dict(data)
        data["max_tokens"] = mct
    return data



class ChatCompletionRequest(BaseModel):
    """
    Chat completion request - OpenAI format.

    Supports all standard OpenAI chat completion parameters. Unknown fields are kept
    (extra="allow") and reported or refused by `millm/api/request_policy.py`.
    """

    model: str
    messages: list[ChatMessage]
    stream: bool = False
    temperature: float = Field(default=1.0, ge=0.0, le=2.0)
    top_p: float = Field(default=1.0, ge=0.0, le=1.0)
    n: int = Field(default=1, ge=1)
    max_tokens: Optional[int] = Field(default=None, gt=0)
    #: OpenAI's newer name for `max_tokens`; folded into it (T-58).
    max_completion_tokens: Optional[int] = Field(default=None, gt=0)
    stop: Optional[Union[str, list[str]]] = None
    frequency_penalty: float = Field(default=0.0, ge=-2.0, le=2.0)
    presence_penalty: float = Field(default=0.0, ge=-2.0, le=2.0)

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
    # For clients: a server that predates this field ACCEPTS it and silently returns a
    # single choice (those servers were extra="ignore"). Since Feature 25 an unknown field is
    # reported in X-miLLM-Ignored-Fields (or refused under X-miLLM-Strict), and X-miLLM-Batch
    # remains the capability probe for this one.
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

    # SCORING MODE on chat (Feature 25, FR-25.5). OpenAI's chat names and shapes: `logprobs` is a
    # BOOLEAN here (it is a count on legacy completions) and `top_logprobs` the number of
    # alternatives. Scoring is on when `logprobs` is true or `allowed_token_ids` is sent; the
    # chat template is rendered and the SAME scorer /v1/completions uses is called, so chat and
    # completion scores of one rendered prompt agree by construction.
    logprobs: Optional[bool] = None
    top_logprobs: Optional[int] = Field(default=None, ge=0, le=20)
    #: As on completions: restrict the next token to these ids, renormalised over the set.
    allowed_token_ids: Optional[list[int]] = Field(default=None, min_length=1, max_length=1024)
    #: Key tokens as "token_id:<id>" (vLLM's switch). `bytes` stays the decoded text's UTF-8.
    return_tokens_as_token_ids: bool = False

    #: Reproducible sampling (FR-25.13): applied inside the admission slot under a forked RNG,
    #: echoed in X-miLLM-Seed with the scope of the promise. Accepted and echoed under greedy
    #: decoding and scoring, where it changes nothing.
    seed: Optional[int] = Field(default=None, ge=0, le=SEED_MAX)
    #: Structured output (FR-25.10): `json_object` or `json_schema`, constrained during decoding
    #: on the transformers engine; refused everywhere else, with a reason.
    response_format: Optional[ResponseFormat] = None

    # miLLM extension - steering profile override
    profile: Optional[str] = None

    # miLLM extension (Feature 10) - per-request cluster intensity dial.
    # Numeric lambda in [0, 2], or a symbolic position resolved server-side
    # against the active cluster's declared intensity_range: "off" -> 0,
    # "min"/"max" -> the range bounds. Applied and restored inside the
    # request boundary - concurrent requests never see each other's dial.
    steering_intensity: Optional[Union[float, Literal["off", "min", "max"]]] = None

    @field_validator("chat_template_kwargs", mode="before")
    @classmethod
    def _coerce_chat_template_kwargs(cls, v):
        """Accept the shapes real clients send, not just the tidy one.

        Open WebUI's "Add Custom Parameter" editor emits a JSON ARRAY wrapping
        the object -- `[{"enable_thinking": false}]` -- and a strict
        `dict[str, Any]` rejects it with "Input should be a valid dictionary",
        which is useless to someone typing into a UI field that gave them no
        choice about the shape. Some clients send it as a JSON string too.

        Being liberal here costs nothing: every accepted form collapses to the
        same dict before it reaches apply_chat_template. Genuinely wrong input
        still fails, and says what it received.
        """
        import json as _json

        if v is None or isinstance(v, dict):
            return v

        if isinstance(v, str):
            text = v.strip()
            if not text:
                return None
            try:
                v = _json.loads(text)
            except ValueError as exc:
                raise ValueError(
                    "chat_template_kwargs must be a JSON object such as "
                    f'{{"enable_thinking": false}}; could not parse {v!r} ({exc})'
                ) from exc
            return cls._coerce_chat_template_kwargs(v)

        if isinstance(v, (list, tuple)):
            merged: dict = {}
            for item in v:
                if isinstance(item, str):
                    item = cls._coerce_chat_template_kwargs(item)
                if not isinstance(item, dict):
                    raise ValueError(
                        "chat_template_kwargs list entries must be JSON "
                        f"objects; got {type(item).__name__}: {item!r}"
                    )
                # Later entries win, matching dict.update semantics.
                merged.update(item)
            return merged or None

        raise ValueError(
            "chat_template_kwargs must be a JSON object such as "
            f'{{"enable_thinking": false}}; got {type(v).__name__}: {v!r}'
        )

    @model_validator(mode="before")
    @classmethod
    def _max_completion_tokens(cls, data: Any) -> Any:
        return _fold_max_completion_tokens(data)

    @field_validator("seed", mode="before")
    @classmethod
    def _seed_not_bool(cls, v: Any) -> Any:
        return _reject_bool_seed(v)

    @field_validator("response_format")
    @classmethod
    def _response_format_combinations(cls, v: Any, info: ValidationInfo) -> Any:
        """Refused combinations, naming `response_format` (FR-25.11.5, T-59, FR-25.6.4):
        * `stop` — a stop string can cut a document and report finish_reason "stop", which is
          truncated JSON reported as complete;
        * `stream` — refused until proven (T-59);
        * scoring — scoring returns one token's log-probabilities, not a document.
        Validated after `stream`, `stop`, `logprobs` and `allowed_token_ids` (declared earlier).
        """
        if v is None or getattr(v, "type", "text") == "text":
            return v
        data = info.data
        if data.get("stop"):
            raise ValueError("response_format cannot be combined with stop: a stop string can "
                             "cut a document and report it as complete")
        if data.get("stream"):
            raise ValueError("response_format with stream=true is not supported in this release")
        if data.get("logprobs") is True or data.get("allowed_token_ids") is not None:
            raise ValueError("response_format cannot be combined with scoring (logprobs / "
                             "allowed_token_ids): scoring returns one token, not a document")
        return v

    @field_validator("n")
    @classmethod
    def _no_n_when_streaming(cls, v: Any, info: ValidationInfo) -> Any:
        # FR-25.3.5: the streaming path reads neither `n` nor `extra_messages`; a streamed n=3
        # returned ONE choice. Refused, naming the field (validated after `stream`).
        if v > 1 and info.data.get("stream"):
            raise ValueError("n > 1 is not supported with stream=true; send n=1 or stream=false")
        return v

    @field_validator("extra_messages")
    @classmethod
    def _no_extra_messages_when_streaming(cls, v: Any, info: ValidationInfo) -> Any:
        if v and info.data.get("stream"):
            raise ValueError(
                "extra_messages (batched conversations) is not supported with stream=true"
            )
        return v

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

    #: Feature 27 (FR-27.1): this request's own SAE activations, in the response's `millm` object.
    #: Declared, so Feature 25's policy knows it (never reported as an ignored field).
    return_sae_activations: Optional[ReturnSaeActivations] = None

    model_config = {"extra": "allow"}

    @model_validator(mode="after")
    def validate_stop_sequences(self) -> "ChatCompletionRequest":
        """Limit stop sequences to 4 (OpenAI limit)."""
        if isinstance(self.stop, list) and len(self.stop) > 4:
            raise ValueError("Maximum 4 stop sequences allowed")
        return self

    @field_validator("top_logprobs")
    @classmethod
    def _top_logprobs_needs_logprobs(cls, v: Any, info: ValidationInfo) -> Any:
        # OpenAI's rule (FR-25.5.3): alternatives are only returned when logprobs is true.
        # `logprobs` is declared before `top_logprobs`, so it is already in info.data.
        if v is not None and info.data.get("logprobs") is not True:
            raise ValueError("top_logprobs requires logprobs: true")
        return v

    @model_validator(mode="after")
    def validate_scoring_mode(self) -> "ChatCompletionRequest":
        """Chat scoring carries completion scoring's limits, plus no streaming (FR-25.6.1)."""
        if self.wants_scores():
            if self.max_tokens != 1:
                raise ValueError("logprobs and allowed_token_ids require max_tokens=1")
            if self.n != 1:
                raise ValueError("logprobs and allowed_token_ids require n=1")
            if self.stream:
                raise ValueError("logprobs and allowed_token_ids require stream=false")
            if self.allowed_token_ids is not None and any(i < 0 for i in self.allowed_token_ids):
                raise ValueError("allowed_token_ids must be non-negative token ids")
            if 0.0 < self.temperature < MIN_SCORING_TEMPERATURE:
                raise ValueError(
                    f"scoring mode needs temperature 0 or at least {MIN_SCORING_TEMPERATURE}"
                )
        return self

    def wants_scores(self) -> bool:
        """Scoring mode (FR-25.5.2): `logprobs: false` alone is ordinary generation."""
        return self.logprobs is True or self.allowed_token_ids is not None


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
    #: OpenAI's newer name for `max_tokens`; folded into it (T-58).
    max_completion_tokens: Optional[int] = Field(default=None, gt=0)
    temperature: float = Field(default=1.0, ge=0.0, le=2.0)
    top_p: float = Field(default=1.0, ge=0.0, le=1.0)
    stop: Optional[Union[str, list[str]]] = None
    frequency_penalty: float = Field(default=0.0, ge=-2.0, le=2.0)
    presence_penalty: float = Field(default=0.0, ge=-2.0, le=2.0)
    #: SCORING MODE (2026-10-04). Return the log-probabilities of the next token instead of only
    #: generating text — what a typed-decision judge (Jev-style classifiers, jevify) reads its answer
    #: from. Named as OpenAI's legacy completions and vLLM name them, so their clients work unchanged.
    #: `logprobs` is how many alternatives to return per position (0–20, OpenAI's limit).
    logprobs: Optional[int] = Field(default=None, ge=0, le=20)
    #: Restrict the next token to these ids. The returned log-probabilities are then normalised over
    #: this set only (vLLM's `processed_logprobs` semantics), so a judge's answer tokens are always
    #: present even when the model would not rank them in its unrestricted top 20.
    allowed_token_ids: Optional[list[int]] = Field(default=None, min_length=1, max_length=1024)
    #: Whether the tokenizer adds its special tokens (a BOS) to the prompt. A template-exact judge
    #: prompt needs False. Defaults to the tokenizer's behaviour, as before.
    add_special_tokens: bool = True
    #: Key `top_logprobs` by "token_id:<id>" instead of the decoded token, which is ambiguous
    #: (vLLM's name for the same switch).
    return_tokens_as_token_ids: bool = False
    #: Reproducible sampling (FR-25.13). Applied per prompt — each prompt is its own generate(),
    #: so prompt i's output does not depend on prompt i-1's length (FTID I7).
    seed: Optional[int] = Field(default=None, ge=0, le=SEED_MAX)
    #: Feature 27 (FR-27.1): this request's own SAE activations, in the response's `millm` object.
    #: Declared, so Feature 25's policy knows it (never reported as an ignored field).
    return_sae_activations: Optional[ReturnSaeActivations] = None

    model_config = {"extra": "allow"}

    @field_validator("seed", mode="before")
    @classmethod
    def _seed_not_bool(cls, v: Any) -> Any:
        return _reject_bool_seed(v)

    @model_validator(mode="before")
    @classmethod
    def _max_completion_tokens(cls, data: Any) -> Any:
        return _fold_max_completion_tokens(data)

    @field_validator("n")
    @classmethod
    def _n_greater_than_one_is_refused(cls, v: int) -> int:
        # T-56 / FR-25.4.3: create_text_completion never read `n`, so n=3 returned one choice per
        # prompt. Implementing it is deferred (FR-25.4.2); until then it is refused, on every
        # engine, before any auto-load (schema validation runs first).
        if v > 1:
            raise ValueError(
                "n > 1 is not implemented on /v1/completions in this release (T-56); send n=1 "
                "and repeat the request for more completions"
            )
        return v

    @model_validator(mode="after")
    def validate_stop_sequences(self) -> "TextCompletionRequest":
        """Limit stop sequences to 4 (OpenAI limit)."""
        if isinstance(self.stop, list) and len(self.stop) > 4:
            raise ValueError("Maximum 4 stop sequences allowed")
        return self

    @model_validator(mode="after")
    def validate_scoring_mode(self) -> "TextCompletionRequest":
        """Scoring mode scores ONE next token. Refused, not approximated, outside that.

        Returning log-probabilities for a multi-token generation needs every step's distribution;
        silently scoring only the first token while generating more would hand a client numbers
        that describe a different output than the text it received.
        """
        if self.wants_scores():
            if self.max_tokens != 1:
                raise ValueError("logprobs and allowed_token_ids require max_tokens=1")
            if self.n != 1:
                raise ValueError("logprobs and allowed_token_ids require n=1")
            if self.allowed_token_ids is not None and any(i < 0 for i in self.allowed_token_ids):
                raise ValueError("allowed_token_ids must be non-negative token ids")
            if isinstance(self.prompt, list) and not self.prompt:
                raise ValueError("scoring mode needs at least one prompt")
            if 0.0 < self.temperature < MIN_SCORING_TEMPERATURE:
                # Dividing by a vanishing temperature overflows finite logits to +inf, and the
                # result would be refused as a model fault it is not (review round 3).
                raise ValueError(
                    f"scoring mode needs temperature 0 or at least {MIN_SCORING_TEMPERATURE}"
                )
        return self

    def wants_scores(self) -> bool:
        return self.logprobs is not None or self.allowed_token_ids is not None


class EmbeddingRequest(BaseModel):
    """Embedding request - OpenAI format."""

    model: str
    input: Union[str, list[str]]
    encoding_format: Literal["float", "base64"] = "float"
    # Refused on every model by the request policy (Feature 30, T-91): no model declares
    # truncated-embedding support. Declared so a non-positive value is a schema error.
    dimensions: Optional[int] = Field(default=None, gt=0)
    # Feature 30 (FR-30.2): how the last hidden layer is pooled over REAL tokens, and whether
    # the vector is L2-normalised. The defaults return the vectors served before Feature 30.
    pooling: Literal["mean", "last", "cls"] = "mean"
    normalize: bool = False

    model_config = {"extra": "allow"}

    @field_validator("input")
    @classmethod
    def _input_shape(cls, value: str | list[str]) -> str | list[str]:
        """Empty input and lists over the cap are refused here, before any auto-load
        (FR-30.3.6, FR-30.3.8, T-94).

        A FIELD validator, not a model validator: the /v1 validation handler builds `param`
        from the error's `loc`, which for a field validator is ("body", "input"). The message
        carries the index. `[]` used to auto-load the model and return a 200 with no data.
        """
        from millm.core.config import settings

        items = [value] if isinstance(value, str) else value
        if not items:
            raise ValueError("input must contain at least one string")
        empty = [i for i, text in enumerate(items) if text == ""]
        if empty:
            where = "input" if isinstance(value, str) else f"input[{empty[0]}]"
            more = f" (and {len(empty) - 1} more empty)" if len(empty) > 1 else ""
            raise ValueError(f"{where} must not be empty{more}")
        cap = settings.EMBEDDINGS_MAX_INPUTS
        if len(items) > cap:
            raise ValueError(
                f"input has {len(items)} items; the limit is {cap} (EMBEDDINGS_MAX_INPUTS)"
            )
        return value


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


class ChatLogprobAlternative(BaseModel):
    """One entry of `top_logprobs` in OpenAI's chat logprobs shape."""

    token: str
    logprob: float
    #: UTF-8 of the DECODED token text, always — `return_tokens_as_token_ids` changes `token` only.
    bytes: list[int]


class ChatLogprobToken(ChatLogprobAlternative):
    """The scored next token (FR-25.8): `content` holds exactly one of these."""

    top_logprobs: list[ChatLogprobAlternative]


class ChatLogprobs(BaseModel):
    content: list[ChatLogprobToken]


class ChatCompletionChoice(BaseModel):
    """Single completion choice in non-streaming response."""

    index: int
    message: ChatMessage
    finish_reason: Literal["stop", "length", "timeout"]
    #: Present only for chat scoring with `logprobs: true` (FR-25.8). Omitted otherwise, so a
    #: client that sends no new field receives the body it received before Feature 25.
    logprobs: Optional[ChatLogprobs] = Field(default=None, exclude_if=lambda v: v is None)


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
    #: model, revision, precision and engine (FR-25.13.8-13.10); set by the route.
    system_fingerprint: Optional[str] = Field(default=None, exclude_if=lambda v: v is None)
    #: Feature 27: the `millm` extension object. OMITTED when absent, so a client that asked for
    #: nothing new receives exactly the body it did before (FR-27.1e).
    millm: Optional[MillmExtension] = Field(default=None, exclude_if=lambda v: v is None)


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


class CompletionLogprobs(BaseModel):
    """OpenAI legacy-completions `logprobs` object, one entry per generated token."""

    tokens: list[str]
    token_logprobs: list[float]
    top_logprobs: list[dict[str, float]]
    text_offset: list[int]


class TextCompletionChoice(BaseModel):
    """Single completion choice in text completion response."""

    index: int
    text: str
    finish_reason: Literal["stop", "length", "timeout"]
    #: Present only in scoring mode (`logprobs` / `allowed_token_ids` on the request).
    logprobs: Optional[CompletionLogprobs] = None


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
    #: model, revision, precision and engine (FR-25.13.8-13.10); set by the route.
    system_fingerprint: Optional[str] = Field(default=None, exclude_if=lambda v: v is None)
    #: Feature 27: the `millm` extension object. OMITTED when absent, so a client that asked for
    #: nothing new receives exactly the body it did before (FR-27.1e).
    millm: Optional[MillmExtension] = Field(default=None, exclude_if=lambda v: v is None)


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
