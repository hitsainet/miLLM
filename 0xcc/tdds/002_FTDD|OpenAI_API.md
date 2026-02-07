# Technical Design Document: OpenAI API Compatibility

## miLLM Feature 2

**Document Version:** 1.0
**Created:** January 30, 2026
**Status:** Draft
**References:**
- Feature PRD: `002_FPRD|OpenAI_API.md`
- ADR: `000_PADR|miLLM.md`

---

## 1. Executive Summary

OpenAI API Compatibility provides drop-in replacement endpoints for the OpenAI v1 API. The design prioritizes exact format matching and client compatibility over feature completeness. Following miLLM's "Ollama-simple" philosophy, implementation focuses on the core endpoints needed for chat and embeddings with streaming support.

### Design Principles
1. **Format Fidelity:** Response formats match OpenAI exactly
2. **Transparent Steering:** SAE steering applied without changing API contract
3. **Streaming First:** SSE streaming is the primary response mode
4. **Graceful Degradation:** Unsupported parameters ignored, not errored

### Key Technical Decisions

| Area | Decision | Rationale |
|------|----------|-----------|
| Streaming | SSE via sse-starlette | Standard, well-supported |
| Generation | Transformers generate() | Native integration |
| Token Counting | tiktoken or model tokenizer | Accurate usage stats |
| Concurrency | asyncio + request queue | Simple, sufficient for single-user |
| Error Format | OpenAI-compatible | Client compatibility |

---

## 2. System Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    OpenAI-Compatible Clients                     │
│     (Open WebUI, LibreChat, Python openai, Continue.dev)        │
└─────────────────────────────────────────────────────────────────┘
                              │
                              │ HTTP / SSE
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      FastAPI Application                         │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                  OpenAI API Routes                        │   │
│  │  /v1/chat/completions  /v1/completions  /v1/embeddings   │   │
│  │  /v1/models            /v1/models/{id}                    │   │
│  └────────────────────────────┬────────────────────────────┘   │
│                               │                                  │
│                               ▼                                  │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                  InferenceService                         │   │
│  │  - generate_chat()      - generate_completion()          │   │
│  │  - generate_embeddings() - stream_tokens()               │   │
│  └────────────────────────────┬────────────────────────────┘   │
│                               │                                  │
│                               ▼                                  │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                  Request Queue                            │   │
│  │  - Manages concurrent requests                            │   │
│  │  - Prevents GPU memory conflicts                          │   │
│  └────────────────────────────┬────────────────────────────┘   │
│                               │                                  │
│                               ▼                                  │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │              Model + SAE (from ModelService)              │   │
│  │  - Loaded model in GPU memory                             │   │
│  │  - Optional SAE hook for steering                         │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

### Component Relationships

1. **Routes → InferenceService:** Routes validate requests, delegate to service
2. **InferenceService → ModelService:** Gets loaded model reference
3. **InferenceService → SteeringService:** Gets active steering configuration
4. **InferenceService → Transformers:** Performs actual generation

### Data Flow: Chat Completion (Streaming)

```
Client sends POST /v1/chat/completions
        │
        ▼
OpenAI Routes ────────► Validate request body
        │                - Check model matches loaded
        │                - Validate message format
        │                - Parse generation parameters
        │
        ▼
InferenceService ─────► Prepare generation
        │                - Format messages into prompt
        │                - Set up generation config
        │                - Initialize token streamer
        │
        ▼
Request Queue ────────► Acquire slot
        │                - Wait if queue full
        │                - Reserve GPU resources
        │
        ▼
Model.generate() ─────► Generate with streaming callback
        │                - SAE hook applied if attached
        │                - Steering values applied
        │                - Tokens yielded as generated
        │
        ▼
SSE Response ─────────► Stream chunks to client
        │                - Format as OpenAI chunk
        │                - Send via SSE
        │                - Send [DONE] at end
        │
        ▼
Release queue slot
```

---

## 3. Technical Stack

### Dependencies

```python
# Core
fastapi>=0.109.0
uvicorn[standard]>=0.27.0
pydantic>=2.0

# Streaming
sse-starlette>=1.6.0

# ML/Generation
torch>=2.0
transformers>=4.36.0

# Token counting (optional, for accurate usage)
tiktoken>=0.5.0  # Or use model tokenizer

# Async
asyncio (stdlib)
```

### Technology Justification

| Technology | Purpose | Why |
|------------|---------|-----|
| sse-starlette | SSE streaming | Clean FastAPI integration |
| TextIteratorStreamer | Token streaming | Transformers native |
| asyncio.Queue | Request queue | Simple, no external deps |
| Pydantic v2 | Request validation | Fast, OpenAPI generation |

---

## 4. Data Design

### Request/Response Schemas

```python
# millm/api/schemas/openai.py

from pydantic import BaseModel, Field
from typing import List, Optional, Union, Literal
from datetime import datetime


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant", "tool", "function"]
    content: Optional[str] = None


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    stream: bool = False
    temperature: float = Field(default=1.0, ge=0.0, le=2.0)
    top_p: float = Field(default=1.0, ge=0.0, le=1.0)
    max_tokens: Optional[int] = Field(default=None, gt=0)
    stop: Optional[Union[str, List[str]]] = None
    frequency_penalty: float = Field(default=0.0, ge=-2.0, le=2.0)
    presence_penalty: float = Field(default=0.0, ge=-2.0, le=2.0)
    n: int = Field(default=1, ge=1)
    user: Optional[str] = None
    # miLLM extension
    profile: Optional[str] = None  # Override active steering profile


class ChatCompletionChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: Literal["stop", "length", "timeout"]


class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionResponse(BaseModel):
    id: str
    object: Literal["chat.completion"] = "chat.completion"
    created: int
    model: str
    choices: List[ChatCompletionChoice]
    usage: Usage


class ChatCompletionChunkDelta(BaseModel):
    role: Optional[Literal["assistant"]] = None
    content: Optional[str] = None


class ChatCompletionChunkChoice(BaseModel):
    index: int
    delta: ChatCompletionChunkDelta
    finish_reason: Optional[Literal["stop", "length", "timeout"]] = None


class ChatCompletionChunk(BaseModel):
    id: str
    object: Literal["chat.completion.chunk"] = "chat.completion.chunk"
    created: int
    model: str
    choices: List[ChatCompletionChunkChoice]


class EmbeddingRequest(BaseModel):
    model: str
    input: Union[str, List[str]]
    encoding_format: Literal["float", "base64"] = "float"
    dimensions: Optional[int] = None
    user: Optional[str] = None


class EmbeddingData(BaseModel):
    object: Literal["embedding"] = "embedding"
    index: int
    embedding: List[float]


class EmbeddingResponse(BaseModel):
    object: Literal["list"] = "list"
    data: List[EmbeddingData]
    model: str
    usage: Usage


class ModelObject(BaseModel):
    id: str
    object: Literal["model"] = "model"
    created: int
    owned_by: str = "miLLM"


class ModelListResponse(BaseModel):
    object: Literal["list"] = "list"
    data: List[ModelObject]


class OpenAIError(BaseModel):
    message: str
    type: Literal["invalid_request_error", "authentication_error", "rate_limit_error", "server_error"]
    param: Optional[str] = None
    code: Optional[str] = None


class OpenAIErrorResponse(BaseModel):
    error: OpenAIError


class TextCompletionRequest(BaseModel):
    model: str
    prompt: Union[str, List[str]]
    stream: bool = False
    temperature: float = Field(default=1.0, ge=0.0, le=2.0)
    top_p: float = Field(default=1.0, ge=0.0, le=1.0)
    max_tokens: Optional[int] = Field(default=None, gt=0)
    stop: Optional[Union[str, List[str]]] = None
    frequency_penalty: float = Field(default=0.0, ge=-2.0, le=2.0)
    presence_penalty: float = Field(default=0.0, ge=-2.0, le=2.0)
    n: int = Field(default=1, ge=1)
    user: Optional[str] = None
    profile: Optional[str] = None


class TextCompletionChoice(BaseModel):
    index: int
    text: str
    finish_reason: Literal["stop", "length", "timeout"]


class TextCompletionResponse(BaseModel):
    id: str
    object: Literal["text_completion"] = "text_completion"
    created: int
    model: str
    choices: List[TextCompletionChoice]
    usage: Usage
```

### Generation Configuration

```python
# millm/ml/generation_config.py

from dataclasses import dataclass
from typing import Optional, List


@dataclass
class GenerationConfig:
    """Configuration for text generation."""
    max_new_tokens: int = 512
    temperature: float = 1.0
    top_p: float = 1.0
    do_sample: bool = True
    stop_sequences: Optional[List[str]] = None
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0

    @classmethod
    def from_request(cls, request) -> "GenerationConfig":
        """Create from OpenAI-style request."""
        return cls(
            max_new_tokens=request.max_tokens or 512,
            temperature=request.temperature,
            top_p=request.top_p,
            do_sample=request.temperature > 0,
            stop_sequences=request.stop if isinstance(request.stop, list) else [request.stop] if request.stop else None,
            frequency_penalty=request.frequency_penalty,
            presence_penalty=request.presence_penalty,
        )

    def to_generate_kwargs(self) -> dict:
        """
        Convert to kwargs for model.generate().
        Maps OpenAI parameters to Transformers equivalents:
        - frequency_penalty → repetition_penalty (1.0 + frequency_penalty)
        """
        kwargs = {
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature if self.do_sample else 1.0,
            "top_p": self.top_p,
            "do_sample": self.do_sample,
        }
        if self.frequency_penalty != 0.0:
            kwargs["repetition_penalty"] = 1.0 + self.frequency_penalty
        return kwargs
```

---

## 5. API Design

### Route Structure

```python
# millm/api/routes/openai/__init__.py

from fastapi import APIRouter

router = APIRouter(prefix="/v1", tags=["OpenAI"])

# Import and include sub-routers
from .chat import router as chat_router
from .completions import router as completions_router
from .models import router as models_router
from .embeddings import router as embeddings_router

router.include_router(chat_router)
router.include_router(completions_router)
router.include_router(models_router)
router.include_router(embeddings_router)
```

### Chat Completions Route

```python
# millm/api/routes/openai/chat.py

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from sse_starlette.sse import EventSourceResponse

from millm.api.schemas.openai import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionChunk,
)
from millm.services.inference_service import InferenceService
from millm.api.dependencies import get_inference_service

router = APIRouter()


@router.post("/chat/completions")
async def create_chat_completion(
    request: ChatCompletionRequest,
    inference: InferenceService = Depends(get_inference_service),
):
    """
    Create a chat completion. OpenAI-compatible endpoint.
    Supports streaming via SSE when stream=true.
    """
    # Validate model is loaded
    if not inference.is_model_loaded():
        raise HTTPException(
            status_code=503,
            detail={
                "error": {
                    "message": "No model loaded. Load a model via admin UI.",
                    "type": "server_error",
                    "code": "model_not_loaded"
                }
            }
        )

    if request.stream:
        return EventSourceResponse(
            inference.stream_chat_completion(request),
            media_type="text/event-stream"
        )
    else:
        return await inference.create_chat_completion(request)
```

### Streaming Implementation

```python
# millm/services/inference_service.py (streaming portion)

import asyncio
import json
import uuid
from datetime import datetime
from typing import AsyncGenerator

from transformers import TextIteratorStreamer
from threading import Thread


class InferenceService:
    async def stream_chat_completion(
        self,
        request: ChatCompletionRequest
    ) -> AsyncGenerator[str, None]:
        """
        Stream chat completion tokens via SSE.
        Yields formatted SSE events.
        """
        completion_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
        created = int(datetime.now().timestamp())
        model_name = self._loaded_model.model_id

        # Prepare inputs
        prompt = self._format_chat_messages(request.messages)
        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._device)

        # Set up streamer
        streamer = TextIteratorStreamer(
            self._tokenizer,
            skip_prompt=True,
            skip_special_tokens=True
        )

        # Generation config
        generation_kwargs = {
            **inputs,
            "streamer": streamer,
            "max_new_tokens": request.max_tokens or 512,
            "temperature": request.temperature,
            "top_p": request.top_p,
            "do_sample": request.temperature > 0,
        }

        # Start generation in thread
        thread = Thread(target=self._generate_in_thread, args=(generation_kwargs,))
        thread.start()

        # Send first chunk with role
        first_chunk = ChatCompletionChunk(
            id=completion_id,
            created=created,
            model=model_name,
            choices=[{
                "index": 0,
                "delta": {"role": "assistant"},
                "finish_reason": None
            }]
        )
        yield f"data: {first_chunk.model_dump_json()}\n\n"

        # Stream tokens
        for token in streamer:
            if token:
                chunk = ChatCompletionChunk(
                    id=completion_id,
                    created=created,
                    model=model_name,
                    choices=[{
                        "index": 0,
                        "delta": {"content": token},
                        "finish_reason": None
                    }]
                )
                yield f"data: {chunk.model_dump_json()}\n\n"

        # Send final chunk
        final_chunk = ChatCompletionChunk(
            id=completion_id,
            created=created,
            model=model_name,
            choices=[{
                "index": 0,
                "delta": {},
                "finish_reason": "stop"
            }]
        )
        yield f"data: {final_chunk.model_dump_json()}\n\n"
        yield "data: [DONE]\n\n"

        thread.join()

    def _generate_in_thread(self, generation_kwargs):
        """Run generation in separate thread for streaming."""
        with torch.no_grad():
            self._model.generate(**generation_kwargs)
```

### Streaming Implementation Notes

- **finish_reason:** Returns `"length"` when `generated_token_count >= max_new_tokens`, otherwise `"stop"`.
- **Stop sequences:** Checked per-token during streaming via text accumulation against the stop list.
- **Thread error handling:** Errors from the generation thread are captured and sent as SSE error events to the client.
- **Serialization:** Chunks use `exclude_none=True` for JSON serialization to omit null fields from the SSE output.

### Error Handling

```python
# millm/api/routes/openai/errors.py

from fastapi import Request
from fastapi.responses import JSONResponse

from millm.core.errors import MiLLMError, ModelNotLoadedError


def create_openai_error(
    message: str,
    error_type: str = "server_error",
    code: str = None,
    param: str = None,
    status_code: int = 500
) -> JSONResponse:
    """Create OpenAI-format error response."""
    return JSONResponse(
        status_code=status_code,
        content={
            "error": {
                "message": message,
                "type": error_type,
                "code": code,
                "param": param
            }
        }
    )


async def openai_exception_handler(request: Request, exc: MiLLMError):
    """Convert MiLLM errors to OpenAI format."""
    error_mapping = {
        "MODEL_NOT_LOADED": ("server_error", 503),
        "MODEL_NOT_FOUND": ("invalid_request_error", 404),
        "VALIDATION_ERROR": ("invalid_request_error", 400),
        "INSUFFICIENT_MEMORY": ("server_error", 507),
    }

    error_type, status_code = error_mapping.get(
        exc.code, ("server_error", 500)
    )

    return create_openai_error(
        message=exc.message,
        error_type=error_type,
        code=exc.code,
        status_code=status_code
    )
```

### Dual-Format Error Routing

The exception handler uses a dual-format routing pattern that returns the appropriate error format based on the request path:

```python
def _is_openai_route(request: Request) -> bool:
    """Check if the request targets an OpenAI-compatible endpoint."""
    return request.url.path.startswith("/v1/")

# ERROR_STATUS_MAP maps error codes to HTTP status + error type
ERROR_STATUS_MAP = {
    "MODEL_NOT_LOADED": (503, "server_error"),
    "MODEL_NOT_FOUND": (404, "invalid_request_error"),
    "VALIDATION_ERROR": (400, "invalid_request_error"),
    "INSUFFICIENT_MEMORY": (507, "server_error"),
    "SAE_NOT_ATTACHED": (400, "invalid_request_error"),
    "CONTEXT_LENGTH_EXCEEDED": (400, "invalid_request_error"),
}
```

- `/v1/*` routes return OpenAI format: `{"error": {"message": ..., "type": ..., "param": ..., "code": ...}}`
- `/api/*` routes return management format: `{"success": false, "data": null, "error": {"code": ..., "message": ..., "details": ...}}`

---

## 6. Component Architecture

### Backend Structure

```
millm/
├── api/
│   └── routes/
│       └── openai/
│           ├── __init__.py         # Router aggregation
│           ├── chat.py             # /v1/chat/completions
│           ├── completions.py      # /v1/completions
│           ├── models.py           # /v1/models
│           ├── embeddings.py       # /v1/embeddings
│           └── errors.py           # OpenAI error handling
│
├── services/
│   └── inference_service.py        # Generation logic
│
└── ml/
    └── generation_config.py        # Generation parameters
```

### InferenceService Design

```python
# millm/services/inference_service.py

from typing import Optional, AsyncGenerator
import uuid
from datetime import datetime

import torch
from transformers import TextIteratorStreamer

from millm.ml.model_loader import LoadedModelState
from millm.services.steering_service import SteeringService
from millm.api.schemas.openai import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionChunk,
    EmbeddingRequest,
    EmbeddingResponse,
)


class InferenceService:
    """
    Handles model inference for OpenAI-compatible endpoints.
    Integrates with steering when SAE is attached.
    """

    def __init__(
        self,
        model_state: LoadedModelState,
        steering_service: Optional[SteeringService] = None,
    ):
        self._model_state = model_state
        self._steering_service = steering_service
        self._request_queue = asyncio.Queue(maxsize=5)

    def is_model_loaded(self) -> bool:
        """Check if a model is currently loaded."""
        return self._model_state.is_loaded

    @property
    def _model(self):
        return self._model_state.current.model

    @property
    def _tokenizer(self):
        return self._model_state.current.tokenizer

    async def create_chat_completion(
        self,
        request: ChatCompletionRequest
    ) -> ChatCompletionResponse:
        """
        Create non-streaming chat completion.
        Returns complete response with usage stats.
        """
        completion_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
        created = int(datetime.now().timestamp())

        # Format messages to prompt
        prompt = self._format_chat_messages(request.messages)

        # Tokenize
        inputs = self._tokenizer(prompt, return_tensors="pt").to("cuda")
        prompt_tokens = inputs.input_ids.shape[1]

        # Generate
        with torch.no_grad():
            # Apply steering if active
            if self._steering_service and self._steering_service.is_active:
                self._steering_service.apply_steering()

            outputs = self._model.generate(
                **inputs,
                max_new_tokens=request.max_tokens or 512,
                temperature=request.temperature if request.temperature > 0 else 1.0,
                top_p=request.top_p,
                do_sample=request.temperature > 0,
                pad_token_id=self._tokenizer.eos_token_id,
            )

        # Decode
        generated_ids = outputs[0][prompt_tokens:]
        completion_text = self._tokenizer.decode(
            generated_ids,
            skip_special_tokens=True
        )

        completion_tokens = len(generated_ids)

        return ChatCompletionResponse(
            id=completion_id,
            created=created,
            model=self._model_state.current.model_id,
            choices=[{
                "index": 0,
                "message": {"role": "assistant", "content": completion_text},
                "finish_reason": "stop"
            }],
            usage={
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens
            }
        )

    def _format_chat_messages(self, messages: list) -> str:
        """
        Format chat messages into prompt string.
        Uses model-specific chat template if available.
        """
        if hasattr(self._tokenizer, "apply_chat_template"):
            return self._tokenizer.apply_chat_template(
                [{"role": m.role, "content": m.content} for m in messages],
                tokenize=False,
                add_generation_prompt=True
            )

        # Fallback: simple concatenation
        parts = []
        for msg in messages:
            if msg.role == "system":
                parts.append(f"System: {msg.content}")
            elif msg.role == "user":
                parts.append(f"User: {msg.content}")
            elif msg.role == "assistant":
                parts.append(f"Assistant: {msg.content}")
        parts.append("Assistant:")
        return "\n".join(parts)

    async def create_embeddings(
        self,
        request: EmbeddingRequest
    ) -> EmbeddingResponse:
        """
        Generate embeddings for input text(s).
        Uses model's hidden states as embeddings.
        """
        inputs = request.input if isinstance(request.input, list) else [request.input]

        embeddings = []
        total_tokens = 0

        for i, text in enumerate(inputs):
            # Tokenize
            tokens = self._tokenizer(text, return_tensors="pt").to("cuda")
            total_tokens += tokens.input_ids.shape[1]

            # Get hidden states
            with torch.no_grad():
                outputs = self._model(
                    **tokens,
                    output_hidden_states=True
                )

            # Use last hidden state, mean pooled
            hidden = outputs.hidden_states[-1]
            embedding = hidden.mean(dim=1).squeeze().cpu().tolist()

            embeddings.append({
                "object": "embedding",
                "index": i,
                "embedding": embedding
            })

        return EmbeddingResponse(
            data=embeddings,
            model=self._model_state.current.model_id,
            usage={
                "prompt_tokens": total_tokens,
                "total_tokens": total_tokens
            }
        )
```

---

## 7. State Management

### Request Queue

```python
# millm/services/request_queue.py

import asyncio
from contextlib import asynccontextmanager
from typing import Optional


class RequestQueue:
    """
    Manages concurrent inference requests.
    Ensures only one request uses GPU at a time.
    """

    def __init__(self, max_concurrent: int = 1, max_pending: int = 5):
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._pending = 0
        self._max_pending = max_pending
        self._lock = asyncio.Lock()

    @asynccontextmanager
    async def acquire(self, timeout: Optional[float] = None):
        """
        Acquire a slot in the request queue.
        Raises if queue is full.
        """
        async with self._lock:
            if self._pending >= self._max_pending:
                raise QueueFullError(
                    f"Request queue full ({self._pending} pending)"
                )
            self._pending += 1

        try:
            acquired = await asyncio.wait_for(
                self._semaphore.acquire(),
                timeout=timeout
            )
            yield
        finally:
            self._semaphore.release()
            async with self._lock:
                self._pending -= 1

    @property
    def pending_count(self) -> int:
        return self._pending


class QueueFullError(Exception):
    pass
```

---

## 8. Security Considerations

### Input Validation
- All requests validated via Pydantic schemas
- Message content sanitized (no code execution)
- Token limits enforced to prevent resource exhaustion
- Stop sequences limited to prevent abuse
- **Model name validation:** All `/v1` endpoints validate that `request.model` matches the loaded model name, returning 404 on mismatch
- **Context length validation:** `_check_context_length()` compares `prompt_tokens + max_new_tokens` vs `model.config.max_position_embeddings`, returning an error if exceeded

### Profile-Based Steering via API

When `request.profile` is set on a chat or completion request, the endpoint loads the named profile from the database and applies its steering values before generation. This allows OpenAI API clients to select steering profiles per-request without using the admin UI.

### No Authentication (v1.0)
- Assumes trusted local network
- Architecture supports future API key middleware
- Log user field for tracing (not authentication)

---

## 9. Performance & Scalability

### Performance Targets

| Metric | Target | Strategy |
|--------|--------|----------|
| Time to first token | <500ms | Efficient tokenization, warm GPU |
| Inter-token latency | <50ms | Streaming with TextIteratorStreamer |
| Request queue | 5 pending | Semaphore-based queue |

### Optimization Strategies

```python
# Token counting cache
class TokenCounter:
    """Cache tokenization for repeated prompts."""

    def __init__(self, tokenizer, cache_size=100):
        self._tokenizer = tokenizer
        self._cache = LRUCache(cache_size)

    def count_tokens(self, text: str) -> int:
        if text in self._cache:
            return self._cache[text]

        tokens = len(self._tokenizer.encode(text))
        self._cache[text] = tokens
        return tokens
```

### Memory Management
- Single request at a time for GPU operations
- Clear CUDA cache between requests if memory pressure
- Quantized models recommended for large contexts

---

## 10. Testing Strategy

### Unit Tests

```python
# tests/unit/api/test_openai_schemas.py

def test_chat_completion_request_validation():
    """Test request validation."""
    # Valid request
    request = ChatCompletionRequest(
        model="gemma-2-2b",
        messages=[{"role": "user", "content": "Hello"}]
    )
    assert request.temperature == 1.0  # Default

    # Invalid temperature
    with pytest.raises(ValidationError):
        ChatCompletionRequest(
            model="gemma-2-2b",
            messages=[{"role": "user", "content": "Hello"}],
            temperature=3.0  # Out of range
        )


def test_chat_completion_response_format():
    """Test response matches OpenAI format."""
    response = ChatCompletionResponse(
        id="chatcmpl-test",
        created=1234567890,
        model="test-model",
        choices=[{
            "index": 0,
            "message": {"role": "assistant", "content": "Hi!"},
            "finish_reason": "stop"
        }],
        usage={"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7}
    )

    data = response.model_dump()
    assert data["object"] == "chat.completion"
    assert "choices" in data
    assert "usage" in data
```

### Integration Tests

```python
# tests/integration/test_openai_api.py

class TestChatCompletions:
    async def test_non_streaming_completion(self, client, loaded_model):
        """Test non-streaming chat completion."""
        response = client.post("/v1/chat/completions", json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "Say hello"}],
            "stream": False,
            "max_tokens": 50
        })

        assert response.status_code == 200
        data = response.json()
        assert data["object"] == "chat.completion"
        assert len(data["choices"]) == 1
        assert data["choices"][0]["finish_reason"] in ["stop", "length"]

    async def test_streaming_completion(self, client, loaded_model):
        """Test streaming chat completion via SSE."""
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "test-model",
                "messages": [{"role": "user", "content": "Count to 3"}],
                "stream": True
            },
            headers={"Accept": "text/event-stream"}
        )

        assert response.status_code == 200
        assert "text/event-stream" in response.headers["content-type"]

        # Parse SSE events
        events = list(parse_sse(response.content))
        assert events[-1] == "[DONE]"
        assert len(events) > 2  # At least role + content + done

    async def test_no_model_loaded_error(self, client):
        """Test error when no model is loaded."""
        response = client.post("/v1/chat/completions", json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "Hello"}]
        })

        assert response.status_code == 503
        data = response.json()
        assert data["error"]["code"] == "model_not_loaded"
```

### Compatibility Tests

```python
# tests/compatibility/test_openai_client.py

def test_openai_python_client_compatibility():
    """Test with official OpenAI Python library."""
    from openai import OpenAI

    client = OpenAI(
        base_url="http://localhost:8000/v1",
        api_key="not-needed"
    )

    # Non-streaming
    response = client.chat.completions.create(
        model="gemma-2-2b",
        messages=[{"role": "user", "content": "Hello!"}],
        max_tokens=50
    )

    assert response.choices[0].message.content
    assert response.usage.total_tokens > 0

    # Streaming
    stream = client.chat.completions.create(
        model="gemma-2-2b",
        messages=[{"role": "user", "content": "Count to 5"}],
        stream=True
    )

    content = ""
    for chunk in stream:
        if chunk.choices[0].delta.content:
            content += chunk.choices[0].delta.content

    assert len(content) > 0
```

---

## 11. Deployment & DevOps

### Configuration

```bash
# OpenAI API specific settings
OPENAI_API_TIMEOUT=300           # Request timeout in seconds
OPENAI_MAX_TOKENS_DEFAULT=512    # Default max_tokens if not specified
OPENAI_RATE_LIMIT=0              # 0 = disabled for v1.0
```

### Health Check

```python
# millm/api/routes/openai/models.py

@router.get("/models")
async def list_models(
    inference: InferenceService = Depends(get_inference_service)
):
    """List available models. Returns loaded model only."""
    if not inference.is_model_loaded():
        return ModelListResponse(data=[])

    model = inference.loaded_model
    return ModelListResponse(
        data=[
            ModelObject(
                id=model.name,
                created=int(model.loaded_at.timestamp()),
                owned_by="miLLM"
            )
        ]
    )
```

---

## 12. Risk Assessment

### Technical Risks

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| SSE client disconnect | Medium | Low | Graceful cleanup, thread safety |
| Token counting mismatch | Medium | Low | Use model tokenizer, not tiktoken |
| OOM during generation | Medium | Medium | Request queue, graceful error |
| Streaming format errors | Low | High | Extensive compatibility testing |

### Mitigation Strategies

```python
# Graceful streaming cleanup
async def stream_with_cleanup(generator):
    """Wrap streaming generator with cleanup."""
    try:
        async for chunk in generator:
            yield chunk
    except asyncio.CancelledError:
        logger.info("Client disconnected, cleaning up stream")
        raise
    finally:
        # Cleanup resources
        torch.cuda.empty_cache()
```

---

## 13. Development Phases

### Phase 1: Core Endpoints (2-3 days)
- [ ] Models endpoint (GET /v1/models)
- [ ] Non-streaming chat completions
- [ ] Request/response schema validation
- [ ] Basic error handling

### Phase 2: Streaming (2-3 days)
- [ ] SSE streaming setup
- [ ] TextIteratorStreamer integration
- [ ] Streaming chunk formatting
- [ ] Client disconnect handling

### Phase 3: Additional Endpoints (2 days)
- [ ] Text completions endpoint
- [ ] Embeddings endpoint
- [ ] GET /v1/models/{id}

### Phase 4: Integration & Testing (2-3 days)
- [ ] Steering integration
- [ ] OpenAI Python client testing
- [ ] Open WebUI testing
- [ ] Performance optimization

---

**Document Status:** Complete
**Next Document:** `002_FTID|OpenAI_API.md` (Technical Implementation Document)
**Instruction File:** `@0xcc/instruct/005_create-tid.md`
