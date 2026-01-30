# Technical Implementation Document: OpenAI API Compatibility

## miLLM Feature 2

**Document Version:** 1.0
**Created:** January 30, 2026
**Status:** Draft
**References:**
- Feature PRD: `002_FPRD|OpenAI_API.md`
- Feature TDD: `002_FTDD|OpenAI_API.md`
- ADR: `000_PADR|miLLM.md`

---

## 1. Overview

This Technical Implementation Document provides specific implementation guidance for Feature 2: OpenAI API Compatibility. It translates the technical design into actionable coding patterns and implementation hints for developers.

### Implementation Philosophy
- **Format Fidelity First:** Every response must pass OpenAI client compatibility tests
- **Streaming is Primary:** Optimize for SSE streaming; non-streaming is the simple case
- **Transparent Integration:** Steering hooks apply without changing API contract
- **Graceful Degradation:** Unsupported parameters ignored, errors formatted correctly

---

## 2. File Structure

### Backend Organization

```
millm/
├── api/
│   ├── routes/
│   │   └── openai/
│   │       ├── __init__.py              # Router aggregation, exports openai_router
│   │       ├── chat.py                  # POST /v1/chat/completions
│   │       ├── completions.py           # POST /v1/completions
│   │       ├── models.py                # GET /v1/models, GET /v1/models/{id}
│   │       ├── embeddings.py            # POST /v1/embeddings
│   │       └── errors.py                # OpenAI error format helpers
│   ├── schemas/
│   │   └── openai.py                    # All Pydantic models for OpenAI API
│   └── dependencies.py                  # FastAPI dependencies (get_inference_service)
│
├── services/
│   ├── inference_service.py             # Core generation logic
│   └── request_queue.py                 # Concurrent request management
│
└── ml/
    └── generation_config.py             # Generation parameter mapping
```

### Test Organization

```
tests/
├── unit/
│   └── api/
│       ├── test_openai_schemas.py       # Schema validation tests
│       ├── test_openai_errors.py        # Error formatting tests
│       └── test_generation_config.py    # Parameter mapping tests
│
├── integration/
│   └── api/
│       ├── test_chat_completions.py     # Full chat endpoint tests
│       ├── test_streaming.py            # SSE streaming tests
│       ├── test_completions.py          # Text completions tests
│       ├── test_embeddings.py           # Embeddings tests
│       └── test_models.py               # Models endpoint tests
│
└── compatibility/
    ├── test_openai_client.py            # OpenAI Python library tests
    ├── test_open_webui.py               # Open WebUI integration
    └── conftest.py                      # Shared fixtures (mock models)
```

---

## 3. Schema Implementation Hints

### Pydantic Models Pattern

```python
# millm/api/schemas/openai.py

"""
OpenAI API compatible schemas.

Key implementation notes:
1. Use Literal types for fixed string values
2. model_dump() replaces deprecated .dict()
3. model_dump_json() for SSE chunk serialization
4. Field() with ge/le for range validation
"""

from pydantic import BaseModel, Field, model_validator
from typing import List, Optional, Union, Literal
from datetime import datetime


# === Message Types ===

class ChatMessage(BaseModel):
    """Chat message with role and content."""
    role: Literal["system", "user", "assistant"]
    content: str

    # Allow extra fields (OpenAI clients may send name, etc.)
    model_config = {"extra": "ignore"}


# === Request Schemas ===

class ChatCompletionRequest(BaseModel):
    """
    Chat completion request - OpenAI format.

    Implementation note: Use default values that match OpenAI behavior.
    Ignore unsupported fields with extra="ignore".
    """
    model: str
    messages: List[ChatMessage]
    stream: bool = False
    temperature: float = Field(default=1.0, ge=0.0, le=2.0)
    top_p: float = Field(default=1.0, ge=0.0, le=1.0)
    max_tokens: Optional[int] = Field(default=None, gt=0)
    stop: Optional[Union[str, List[str]]] = None
    frequency_penalty: float = Field(default=0.0, ge=-2.0, le=2.0)
    presence_penalty: float = Field(default=0.0, ge=-2.0, le=2.0)
    user: Optional[str] = None

    # miLLM extension - steering profile override
    profile: Optional[str] = None

    model_config = {"extra": "ignore"}

    @model_validator(mode="after")
    def validate_stop_sequences(self):
        """Limit stop sequences to 4 (OpenAI limit)."""
        if isinstance(self.stop, list) and len(self.stop) > 4:
            raise ValueError("Maximum 4 stop sequences allowed")
        return self


class TextCompletionRequest(BaseModel):
    """Text completion request - OpenAI format."""
    model: str
    prompt: Union[str, List[str]]
    stream: bool = False
    max_tokens: Optional[int] = Field(default=16, gt=0)
    temperature: float = Field(default=1.0, ge=0.0, le=2.0)
    top_p: float = Field(default=1.0, ge=0.0, le=1.0)
    stop: Optional[Union[str, List[str]]] = None

    model_config = {"extra": "ignore"}


class EmbeddingRequest(BaseModel):
    """Embedding request - OpenAI format."""
    model: str
    input: Union[str, List[str]]

    model_config = {"extra": "ignore"}


# === Response Schemas ===

class Usage(BaseModel):
    """Token usage statistics."""
    prompt_tokens: int
    completion_tokens: int = 0
    total_tokens: int

    @model_validator(mode="after")
    def compute_total(self):
        """Auto-compute total if not provided."""
        if self.total_tokens == 0:
            self.total_tokens = self.prompt_tokens + self.completion_tokens
        return self


class ChatCompletionChoice(BaseModel):
    """Single completion choice."""
    index: int
    message: ChatMessage
    finish_reason: Literal["stop", "length", "timeout"]


class ChatCompletionResponse(BaseModel):
    """
    Non-streaming chat completion response.

    Implementation note: `id` format is "chatcmpl-{24 hex chars}"
    """
    id: str
    object: Literal["chat.completion"] = "chat.completion"
    created: int  # Unix timestamp
    model: str
    choices: List[ChatCompletionChoice]
    usage: Usage


# === Streaming Schemas ===

class ChatCompletionChunkDelta(BaseModel):
    """
    Delta for streaming chunks.

    Implementation note:
    - First chunk: role="assistant", content=None
    - Middle chunks: role=None, content="token"
    - Final chunk: role=None, content=None, finish_reason set
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

    Usage: chunk.model_dump_json() for SSE data field
    """
    id: str
    object: Literal["chat.completion.chunk"] = "chat.completion.chunk"
    created: int
    model: str
    choices: List[ChatCompletionChunkChoice]


# === Models Endpoint ===

class ModelObject(BaseModel):
    """Model metadata."""
    id: str
    object: Literal["model"] = "model"
    created: int
    owned_by: str = "miLLM"


class ModelListResponse(BaseModel):
    """List of available models."""
    object: Literal["list"] = "list"
    data: List[ModelObject]


# === Error Schemas ===

class OpenAIError(BaseModel):
    """OpenAI-format error detail."""
    message: str
    type: Literal[
        "invalid_request_error",
        "authentication_error",
        "rate_limit_error",
        "server_error"
    ]
    param: Optional[str] = None
    code: Optional[str] = None


class OpenAIErrorResponse(BaseModel):
    """OpenAI-format error response."""
    error: OpenAIError
```

---

## 4. Routes Implementation

### Router Aggregation Pattern

```python
# millm/api/routes/openai/__init__.py

"""
OpenAI API routes aggregation.

Mounts all OpenAI-compatible endpoints under /v1 prefix.
"""

from fastapi import APIRouter

router = APIRouter(prefix="/v1", tags=["OpenAI API"])

# Import and mount sub-routers
from .chat import router as chat_router
from .completions import router as completions_router
from .models import router as models_router
from .embeddings import router as embeddings_router

router.include_router(chat_router)
router.include_router(completions_router)
router.include_router(models_router)
router.include_router(embeddings_router)

# Expose for main app mounting
openai_router = router
```

### Chat Completions Endpoint

```python
# millm/api/routes/openai/chat.py

"""
Chat completions endpoint.

Implementation notes:
1. Check model loaded FIRST before any processing
2. Streaming uses EventSourceResponse from sse-starlette
3. Non-streaming returns standard JSONResponse
4. Both paths go through InferenceService
"""

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse
from sse_starlette.sse import EventSourceResponse
import logging

from millm.api.schemas.openai import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    OpenAIErrorResponse,
)
from millm.services.inference_service import InferenceService
from millm.api.dependencies import get_inference_service
from .errors import create_openai_error

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post(
    "/chat/completions",
    response_model=ChatCompletionResponse,
    responses={
        503: {"model": OpenAIErrorResponse, "description": "No model loaded"}
    }
)
async def create_chat_completion(
    request: ChatCompletionRequest,
    inference: InferenceService = Depends(get_inference_service),
):
    """
    Create a chat completion - OpenAI-compatible endpoint.

    Supports both streaming (SSE) and non-streaming modes.
    Applies active steering profile if configured.
    """
    # Check model loaded first
    if not inference.is_model_loaded():
        return create_openai_error(
            message="No model loaded. Load a model via admin UI.",
            error_type="server_error",
            code="model_not_loaded",
            status_code=503
        )

    # Log request (user field is for logging only)
    logger.info(
        f"Chat completion request: model={request.model}, "
        f"messages={len(request.messages)}, stream={request.stream}"
    )

    if request.stream:
        # Streaming response via SSE
        return EventSourceResponse(
            inference.stream_chat_completion(request),
            media_type="text/event-stream"
        )
    else:
        # Non-streaming response
        response = await inference.create_chat_completion(request)
        return response
```

### Models Endpoint

```python
# millm/api/routes/openai/models.py

"""
Models listing endpoint.

Implementation notes:
- Returns only the currently loaded model
- Empty list if no model loaded (not an error)
- GET /v1/models/{id} returns specific model or 404
"""

from fastapi import APIRouter, Depends, HTTPException
from datetime import datetime

from millm.api.schemas.openai import ModelObject, ModelListResponse
from millm.services.inference_service import InferenceService
from millm.api.dependencies import get_inference_service

router = APIRouter()


@router.get("/models", response_model=ModelListResponse)
async def list_models(
    inference: InferenceService = Depends(get_inference_service),
):
    """List available models. Returns currently loaded model only."""
    if not inference.is_model_loaded():
        return ModelListResponse(data=[])

    model_info = inference.get_loaded_model_info()

    return ModelListResponse(
        data=[
            ModelObject(
                id=model_info.name,
                created=int(model_info.loaded_at.timestamp()),
                owned_by="miLLM"
            )
        ]
    )


@router.get("/models/{model_id}", response_model=ModelObject)
async def get_model(
    model_id: str,
    inference: InferenceService = Depends(get_inference_service),
):
    """Get specific model details."""
    if not inference.is_model_loaded():
        raise HTTPException(
            status_code=404,
            detail={"error": {
                "message": f"Model '{model_id}' not found",
                "type": "invalid_request_error",
                "code": "model_not_found"
            }}
        )

    model_info = inference.get_loaded_model_info()

    if model_info.name != model_id:
        raise HTTPException(
            status_code=404,
            detail={"error": {
                "message": f"Model '{model_id}' not found. Available: {model_info.name}",
                "type": "invalid_request_error",
                "code": "model_not_found"
            }}
        )

    return ModelObject(
        id=model_info.name,
        created=int(model_info.loaded_at.timestamp()),
        owned_by="miLLM"
    )
```

### Error Helpers

```python
# millm/api/routes/openai/errors.py

"""
OpenAI error format helpers.

Implementation notes:
- All errors must match OpenAI error response format exactly
- HTTP status codes should match OpenAI behavior
- Include actionable error messages
"""

from fastapi import Request
from fastapi.responses import JSONResponse
from typing import Optional

from millm.core.errors import MiLLMError


def create_openai_error(
    message: str,
    error_type: str = "server_error",
    code: Optional[str] = None,
    param: Optional[str] = None,
    status_code: int = 500
) -> JSONResponse:
    """
    Create OpenAI-format error response.

    Args:
        message: Human-readable error message
        error_type: One of invalid_request_error, server_error, etc.
        code: Machine-readable error code
        param: Parameter that caused the error
        status_code: HTTP status code
    """
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


# Error code to HTTP status mapping
ERROR_STATUS_MAP = {
    "model_not_loaded": (503, "server_error"),
    "model_not_found": (404, "invalid_request_error"),
    "validation_error": (400, "invalid_request_error"),
    "context_length_exceeded": (400, "invalid_request_error"),
    "rate_limit_exceeded": (429, "rate_limit_error"),
    "server_error": (500, "server_error"),
}


async def openai_exception_handler(request: Request, exc: MiLLMError) -> JSONResponse:
    """
    Global exception handler for MiLLM errors.

    Register with FastAPI:
        app.add_exception_handler(MiLLMError, openai_exception_handler)
    """
    status_code, error_type = ERROR_STATUS_MAP.get(
        exc.code, (500, "server_error")
    )

    return create_openai_error(
        message=exc.message,
        error_type=error_type,
        code=exc.code,
        status_code=status_code
    )
```

---

## 5. InferenceService Implementation

### Core Service Structure

```python
# millm/services/inference_service.py

"""
Inference service for OpenAI-compatible generation.

Implementation notes:
1. Thread-based streaming (Transformers generate() is blocking)
2. TextIteratorStreamer bridges generate() to async iteration
3. Request queue prevents GPU memory conflicts
4. Steering integration is transparent to API layer
"""

from typing import Optional, AsyncGenerator
import asyncio
import uuid
from datetime import datetime
from threading import Thread
import logging

import torch
from transformers import TextIteratorStreamer

from millm.api.schemas.openai import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionChunk,
    ChatCompletionChunkDelta,
    ChatCompletionChunkChoice,
    ChatCompletionChoice,
    ChatMessage,
    Usage,
    EmbeddingRequest,
    EmbeddingResponse,
)
from millm.ml.generation_config import GenerationConfig
from millm.services.request_queue import RequestQueue

logger = logging.getLogger(__name__)


class InferenceService:
    """
    Handles inference for OpenAI-compatible endpoints.

    Thread safety notes:
    - One generation at a time via request queue
    - Model/tokenizer access is thread-safe for inference
    - Steering values applied via hooks (not thread-local)
    """

    def __init__(
        self,
        model_service,  # Reference to ModelService
        steering_service=None,  # Optional SteeringService
        max_concurrent: int = 1,
        max_pending: int = 5,
    ):
        self._model_service = model_service
        self._steering_service = steering_service
        self._request_queue = RequestQueue(max_concurrent, max_pending)
        self._device = "cuda" if torch.cuda.is_available() else "cpu"

    def is_model_loaded(self) -> bool:
        """Check if a model is currently loaded."""
        return self._model_service.is_loaded()

    def get_loaded_model_info(self):
        """Get info about loaded model."""
        return self._model_service.get_current_model()

    @property
    def _model(self):
        """Get loaded model."""
        return self._model_service.get_model()

    @property
    def _tokenizer(self):
        """Get loaded tokenizer."""
        return self._model_service.get_tokenizer()

    # === Chat Completions ===

    async def create_chat_completion(
        self,
        request: ChatCompletionRequest
    ) -> ChatCompletionResponse:
        """
        Create non-streaming chat completion.

        Implementation notes:
        - Acquires queue slot before generation
        - Applies steering if active
        - Counts tokens for usage stats
        """
        completion_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
        created = int(datetime.now().timestamp())

        # Format messages to prompt
        prompt = self._format_chat_messages(request.messages)

        async with self._request_queue.acquire():
            # Tokenize input
            inputs = self._tokenizer(
                prompt,
                return_tensors="pt"
            ).to(self._device)
            prompt_tokens = inputs.input_ids.shape[1]

            # Build generation config
            gen_config = GenerationConfig.from_request(request)

            # Generate
            with torch.no_grad():
                # Apply steering if active
                if self._steering_service and self._steering_service.is_active:
                    self._steering_service.prepare_generation()

                outputs = self._model.generate(
                    **inputs,
                    max_new_tokens=gen_config.max_new_tokens,
                    temperature=gen_config.temperature if gen_config.do_sample else 1.0,
                    top_p=gen_config.top_p,
                    do_sample=gen_config.do_sample,
                    pad_token_id=self._tokenizer.eos_token_id,
                )

            # Decode output
            generated_ids = outputs[0][prompt_tokens:]
            completion_text = self._tokenizer.decode(
                generated_ids,
                skip_special_tokens=True
            )
            completion_tokens = len(generated_ids)

        return ChatCompletionResponse(
            id=completion_id,
            created=created,
            model=self.get_loaded_model_info().name,
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message=ChatMessage(role="assistant", content=completion_text),
                    finish_reason="stop"
                )
            ],
            usage=Usage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens
            )
        )

    async def stream_chat_completion(
        self,
        request: ChatCompletionRequest
    ) -> AsyncGenerator[str, None]:
        """
        Stream chat completion via SSE.

        Implementation notes:
        - Yields SSE-formatted strings: "data: {json}\n\n"
        - First chunk has role, middle chunks have content, last has finish_reason
        - Always ends with "data: [DONE]\n\n"
        - Generation runs in thread, tokens yielded async
        """
        completion_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
        created = int(datetime.now().timestamp())
        model_name = self.get_loaded_model_info().name

        # Format messages to prompt
        prompt = self._format_chat_messages(request.messages)

        async with self._request_queue.acquire():
            # Tokenize
            inputs = self._tokenizer(
                prompt,
                return_tensors="pt"
            ).to(self._device)

            # Set up streamer
            streamer = TextIteratorStreamer(
                self._tokenizer,
                skip_prompt=True,
                skip_special_tokens=True
            )

            # Build generation kwargs
            gen_config = GenerationConfig.from_request(request)
            generation_kwargs = {
                **{k: v.to(self._device) for k, v in inputs.items()},
                "streamer": streamer,
                "max_new_tokens": gen_config.max_new_tokens,
                "temperature": gen_config.temperature if gen_config.do_sample else 1.0,
                "top_p": gen_config.top_p,
                "do_sample": gen_config.do_sample,
                "pad_token_id": self._tokenizer.eos_token_id,
            }

            # Apply steering if active
            if self._steering_service and self._steering_service.is_active:
                self._steering_service.prepare_generation()

            # Start generation thread
            thread = Thread(
                target=self._generate_in_thread,
                args=(generation_kwargs,)
            )
            thread.start()

            try:
                # Send first chunk with role
                first_chunk = ChatCompletionChunk(
                    id=completion_id,
                    created=created,
                    model=model_name,
                    choices=[
                        ChatCompletionChunkChoice(
                            index=0,
                            delta=ChatCompletionChunkDelta(role="assistant"),
                            finish_reason=None
                        )
                    ]
                )
                yield f"data: {first_chunk.model_dump_json()}\n\n"

                # Stream tokens
                for token in streamer:
                    if token:
                        chunk = ChatCompletionChunk(
                            id=completion_id,
                            created=created,
                            model=model_name,
                            choices=[
                                ChatCompletionChunkChoice(
                                    index=0,
                                    delta=ChatCompletionChunkDelta(content=token),
                                    finish_reason=None
                                )
                            ]
                        )
                        yield f"data: {chunk.model_dump_json()}\n\n"

                # Send final chunk with finish_reason
                final_chunk = ChatCompletionChunk(
                    id=completion_id,
                    created=created,
                    model=model_name,
                    choices=[
                        ChatCompletionChunkChoice(
                            index=0,
                            delta=ChatCompletionChunkDelta(),
                            finish_reason="stop"
                        )
                    ]
                )
                yield f"data: {final_chunk.model_dump_json()}\n\n"
                yield "data: [DONE]\n\n"

            except asyncio.CancelledError:
                logger.info("Client disconnected during streaming")
                raise
            finally:
                thread.join(timeout=5.0)
                if thread.is_alive():
                    logger.warning("Generation thread did not complete cleanly")

    def _generate_in_thread(self, generation_kwargs: dict):
        """
        Run generation in thread for streaming.

        Must be called in separate thread because generate() is blocking.
        """
        try:
            with torch.no_grad():
                self._model.generate(**generation_kwargs)
        except Exception as e:
            logger.error(f"Generation error: {e}")

    def _format_chat_messages(self, messages: list) -> str:
        """
        Format chat messages into prompt string.

        Implementation notes:
        - Use model's chat template if available (most modern models)
        - Fall back to simple format for models without template
        - Chat template handles special tokens, roles, etc.
        """
        # Prefer model's built-in chat template
        if hasattr(self._tokenizer, "apply_chat_template"):
            try:
                return self._tokenizer.apply_chat_template(
                    [{"role": m.role, "content": m.content} for m in messages],
                    tokenize=False,
                    add_generation_prompt=True
                )
            except Exception as e:
                logger.warning(f"Chat template failed, using fallback: {e}")

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
```

### Request Queue Implementation

```python
# millm/services/request_queue.py

"""
Request queue for managing concurrent inference.

Implementation notes:
- Semaphore limits concurrent GPU operations
- Pending counter prevents queue overflow
- Context manager ensures proper cleanup
"""

import asyncio
from contextlib import asynccontextmanager
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class QueueFullError(Exception):
    """Raised when request queue is at capacity."""
    pass


class RequestQueue:
    """
    Manages concurrent inference requests.

    Default: 1 concurrent, 5 pending
    - Single GPU can only run one inference at a time
    - Queue up to 5 requests to prevent overload
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

        Usage:
            async with request_queue.acquire():
                # Run inference here
                result = await generate(...)

        Raises:
            QueueFullError: If queue is at max_pending capacity
            asyncio.TimeoutError: If timeout expires waiting for slot
        """
        # Check pending count
        async with self._lock:
            if self._pending >= self._max_pending:
                raise QueueFullError(
                    f"Request queue full ({self._pending} pending). "
                    "Try again later."
                )
            self._pending += 1

        try:
            # Wait for semaphore (actual GPU slot)
            if timeout:
                await asyncio.wait_for(
                    self._semaphore.acquire(),
                    timeout=timeout
                )
            else:
                await self._semaphore.acquire()

            yield

        finally:
            self._semaphore.release()
            async with self._lock:
                self._pending -= 1

    @property
    def pending_count(self) -> int:
        """Current number of pending requests."""
        return self._pending

    @property
    def is_available(self) -> bool:
        """Check if queue can accept requests."""
        return self._pending < self._max_pending
```

---

## 6. Generation Config Pattern

```python
# millm/ml/generation_config.py

"""
Generation configuration mapping.

Maps OpenAI parameters to Transformers generate() parameters.
"""

from dataclasses import dataclass, field
from typing import Optional, List


@dataclass
class GenerationConfig:
    """
    Configuration for text generation.

    Mapping from OpenAI to Transformers:
    - max_tokens → max_new_tokens
    - temperature → temperature (0 means greedy)
    - top_p → top_p
    - stop → stopping_criteria (custom implementation)
    - frequency_penalty → repetition_penalty (approximate)
    """
    max_new_tokens: int = 512
    temperature: float = 1.0
    top_p: float = 1.0
    do_sample: bool = True
    stop_sequences: Optional[List[str]] = None
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0

    @classmethod
    def from_request(cls, request) -> "GenerationConfig":
        """
        Create from OpenAI-style request.

        Implementation notes:
        - temperature=0 → do_sample=False (greedy decoding)
        - max_tokens=None → use default 512
        - stop can be string or list
        """
        # Normalize stop sequences
        stop_sequences = None
        if request.stop:
            if isinstance(request.stop, str):
                stop_sequences = [request.stop]
            else:
                stop_sequences = list(request.stop)

        return cls(
            max_new_tokens=request.max_tokens or 512,
            temperature=request.temperature,
            top_p=request.top_p,
            do_sample=request.temperature > 0,
            stop_sequences=stop_sequences,
            frequency_penalty=request.frequency_penalty,
            presence_penalty=request.presence_penalty,
        )

    def to_generate_kwargs(self) -> dict:
        """
        Convert to transformers generate() kwargs.

        Note: stop_sequences requires custom StoppingCriteria,
        not handled here (see inference_service).
        """
        kwargs = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": self.do_sample,
        }

        if self.do_sample:
            kwargs["temperature"] = self.temperature
            kwargs["top_p"] = self.top_p

        # Approximate frequency_penalty with repetition_penalty
        if self.frequency_penalty != 0:
            # OpenAI: -2 to 2, Transformers: typically 1.0-2.0
            # Map 0-2 → 1.0-1.5 (conservative)
            kwargs["repetition_penalty"] = 1.0 + (abs(self.frequency_penalty) * 0.25)

        return kwargs
```

---

## 7. Embeddings Implementation

```python
# millm/api/routes/openai/embeddings.py

"""
Embeddings endpoint implementation.

Implementation notes:
- Uses model's hidden states as embeddings
- Mean pooling over sequence length
- Last hidden layer (most semantic)
"""

from fastapi import APIRouter, Depends
from typing import List

import torch

from millm.api.schemas.openai import (
    EmbeddingRequest,
    EmbeddingResponse,
    EmbeddingData,
    Usage,
)
from millm.services.inference_service import InferenceService
from millm.api.dependencies import get_inference_service
from .errors import create_openai_error

router = APIRouter()


@router.post("/embeddings", response_model=EmbeddingResponse)
async def create_embedding(
    request: EmbeddingRequest,
    inference: InferenceService = Depends(get_inference_service),
):
    """
    Generate embeddings for input text(s).

    Uses mean pooling over last hidden layer.
    """
    if not inference.is_model_loaded():
        return create_openai_error(
            message="No model loaded",
            code="model_not_loaded",
            status_code=503
        )

    # Normalize input to list
    inputs = request.input if isinstance(request.input, list) else [request.input]

    embeddings_data: List[EmbeddingData] = []
    total_tokens = 0

    model = inference._model
    tokenizer = inference._tokenizer
    device = inference._device

    for i, text in enumerate(inputs):
        # Tokenize
        tokens = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=tokenizer.model_max_length
        ).to(device)

        total_tokens += tokens.input_ids.shape[1]

        # Get hidden states
        with torch.no_grad():
            outputs = model(
                **tokens,
                output_hidden_states=True
            )

        # Mean pooling over last hidden state
        # Shape: (1, seq_len, hidden_dim) → (hidden_dim,)
        last_hidden = outputs.hidden_states[-1]
        embedding = last_hidden.mean(dim=1).squeeze().cpu().tolist()

        embeddings_data.append(
            EmbeddingData(
                index=i,
                embedding=embedding
            )
        )

    return EmbeddingResponse(
        data=embeddings_data,
        model=inference.get_loaded_model_info().name,
        usage=Usage(
            prompt_tokens=total_tokens,
            completion_tokens=0,
            total_tokens=total_tokens
        )
    )
```

---

## 8. Text Completions Implementation

```python
# millm/api/routes/openai/completions.py

"""
Text completions endpoint (/v1/completions).

Implementation notes:
- Similar to chat but uses raw prompt string
- Response format slightly different (text vs message)
- Supports same streaming mechanism
"""

from fastapi import APIRouter, Depends
from sse_starlette.sse import EventSourceResponse
import uuid
from datetime import datetime

import torch

from millm.api.schemas.openai import TextCompletionRequest
from millm.services.inference_service import InferenceService
from millm.api.dependencies import get_inference_service
from .errors import create_openai_error

router = APIRouter()


@router.post("/completions")
async def create_completion(
    request: TextCompletionRequest,
    inference: InferenceService = Depends(get_inference_service),
):
    """
    Create text completion - OpenAI-compatible.

    Accepts raw prompt string instead of messages array.
    """
    if not inference.is_model_loaded():
        return create_openai_error(
            message="No model loaded",
            code="model_not_loaded",
            status_code=503
        )

    completion_id = f"cmpl-{uuid.uuid4().hex[:24]}"
    created = int(datetime.now().timestamp())

    # Handle prompt as string or first item of list
    prompt = request.prompt
    if isinstance(prompt, list):
        prompt = prompt[0]

    if request.stream:
        return EventSourceResponse(
            _stream_completion(inference, request, prompt, completion_id, created),
            media_type="text/event-stream"
        )
    else:
        return await _create_completion(
            inference, request, prompt, completion_id, created
        )


async def _create_completion(inference, request, prompt, completion_id, created):
    """Non-streaming text completion."""
    tokenizer = inference._tokenizer
    model = inference._model
    device = inference._device

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    prompt_tokens = inputs.input_ids.shape[1]

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=request.max_tokens or 16,
            temperature=request.temperature if request.temperature > 0 else 1.0,
            do_sample=request.temperature > 0,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated_ids = outputs[0][prompt_tokens:]
    completion_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    completion_tokens = len(generated_ids)

    return {
        "id": completion_id,
        "object": "text_completion",
        "created": created,
        "model": inference.get_loaded_model_info().name,
        "choices": [{
            "index": 0,
            "text": completion_text,
            "finish_reason": "stop"
        }],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens
        }
    }


async def _stream_completion(inference, request, prompt, completion_id, created):
    """Streaming text completion."""
    from transformers import TextIteratorStreamer
    from threading import Thread

    tokenizer = inference._tokenizer
    model = inference._model
    device = inference._device
    model_name = inference.get_loaded_model_info().name

    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    streamer = TextIteratorStreamer(
        tokenizer,
        skip_prompt=True,
        skip_special_tokens=True
    )

    generation_kwargs = {
        **{k: v.to(device) for k, v in inputs.items()},
        "streamer": streamer,
        "max_new_tokens": request.max_tokens or 16,
        "temperature": request.temperature if request.temperature > 0 else 1.0,
        "do_sample": request.temperature > 0,
        "pad_token_id": tokenizer.eos_token_id,
    }

    thread = Thread(target=lambda: model.generate(**generation_kwargs))
    thread.start()

    for token in streamer:
        if token:
            chunk = {
                "id": completion_id,
                "object": "text_completion",
                "created": created,
                "model": model_name,
                "choices": [{
                    "index": 0,
                    "text": token,
                    "finish_reason": None
                }]
            }
            yield f"data: {json.dumps(chunk)}\n\n"

    # Final chunk
    final = {
        "id": completion_id,
        "object": "text_completion",
        "created": created,
        "model": model_name,
        "choices": [{
            "index": 0,
            "text": "",
            "finish_reason": "stop"
        }]
    }
    yield f"data: {json.dumps(final)}\n\n"
    yield "data: [DONE]\n\n"

    thread.join()
```

---

## 9. Dependencies Setup

```python
# millm/api/dependencies.py

"""
FastAPI dependencies for OpenAI routes.

Implementation notes:
- Dependencies are cached per-request
- Services are initialized at app startup
- Use app.state for service instances
"""

from fastapi import Request

from millm.services.inference_service import InferenceService
from millm.services.model_service import ModelService


def get_inference_service(request: Request) -> InferenceService:
    """
    Get InferenceService instance.

    Services are stored in app.state during startup.
    """
    return request.app.state.inference_service


def get_model_service(request: Request) -> ModelService:
    """Get ModelService instance."""
    return request.app.state.model_service
```

### App Startup Setup

```python
# millm/main.py (relevant portion)

from fastapi import FastAPI
from contextlib import asynccontextmanager

from millm.api.routes.openai import openai_router
from millm.api.routes.openai.errors import openai_exception_handler
from millm.services.model_service import ModelService
from millm.services.inference_service import InferenceService
from millm.core.errors import MiLLMError


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    App lifespan - initialize services.
    """
    # Initialize services
    model_service = ModelService()
    inference_service = InferenceService(model_service=model_service)

    # Store in app state
    app.state.model_service = model_service
    app.state.inference_service = inference_service

    yield

    # Cleanup
    await model_service.cleanup()


def create_app() -> FastAPI:
    app = FastAPI(
        title="miLLM",
        description="Mechanistic Interpretability LLM Server",
        lifespan=lifespan
    )

    # Mount OpenAI routes
    app.include_router(openai_router)

    # Register error handler
    app.add_exception_handler(MiLLMError, openai_exception_handler)

    return app
```

---

## 10. Testing Implementation

### Unit Test Patterns

```python
# tests/unit/api/test_openai_schemas.py

"""
Tests for OpenAI schema validation.

Test patterns:
1. Valid input acceptance
2. Invalid input rejection
3. Default value application
4. Edge cases (empty, max values)
"""

import pytest
from pydantic import ValidationError

from millm.api.schemas.openai import (
    ChatCompletionRequest,
    ChatMessage,
    ChatCompletionResponse,
    ChatCompletionChunk,
)


class TestChatCompletionRequest:
    """Tests for chat completion request schema."""

    def test_minimal_valid_request(self):
        """Minimal request with required fields only."""
        request = ChatCompletionRequest(
            model="test-model",
            messages=[ChatMessage(role="user", content="Hello")]
        )
        assert request.model == "test-model"
        assert request.stream is False  # Default
        assert request.temperature == 1.0  # Default

    def test_full_request_with_all_params(self):
        """Request with all parameters."""
        request = ChatCompletionRequest(
            model="test-model",
            messages=[
                ChatMessage(role="system", content="You are helpful"),
                ChatMessage(role="user", content="Hello"),
            ],
            stream=True,
            temperature=0.7,
            top_p=0.9,
            max_tokens=100,
            stop=["END"],
            frequency_penalty=0.5,
            presence_penalty=0.5,
            user="test-user",
            profile="test-profile",
        )
        assert request.stream is True
        assert request.temperature == 0.7
        assert request.profile == "test-profile"

    def test_temperature_validation_range(self):
        """Temperature must be 0.0-2.0."""
        # Valid
        ChatCompletionRequest(
            model="test",
            messages=[ChatMessage(role="user", content="Hi")],
            temperature=0.0
        )
        ChatCompletionRequest(
            model="test",
            messages=[ChatMessage(role="user", content="Hi")],
            temperature=2.0
        )

        # Invalid
        with pytest.raises(ValidationError):
            ChatCompletionRequest(
                model="test",
                messages=[ChatMessage(role="user", content="Hi")],
                temperature=-0.1
            )

        with pytest.raises(ValidationError):
            ChatCompletionRequest(
                model="test",
                messages=[ChatMessage(role="user", content="Hi")],
                temperature=2.1
            )

    def test_stop_sequence_limit(self):
        """Maximum 4 stop sequences."""
        # Valid: 4 sequences
        ChatCompletionRequest(
            model="test",
            messages=[ChatMessage(role="user", content="Hi")],
            stop=["a", "b", "c", "d"]
        )

        # Invalid: 5 sequences
        with pytest.raises(ValidationError):
            ChatCompletionRequest(
                model="test",
                messages=[ChatMessage(role="user", content="Hi")],
                stop=["a", "b", "c", "d", "e"]
            )

    def test_extra_fields_ignored(self):
        """Unknown fields should be ignored (not error)."""
        request = ChatCompletionRequest(
            model="test",
            messages=[ChatMessage(role="user", content="Hi")],
            unknown_field="ignored",  # Should not error
            another_field=123,
        )
        assert request.model == "test"
        assert not hasattr(request, "unknown_field")


class TestChatCompletionResponse:
    """Tests for response schemas."""

    def test_response_serialization(self):
        """Response should serialize to OpenAI format."""
        response = ChatCompletionResponse(
            id="chatcmpl-test123",
            created=1706627200,
            model="gemma-2-2b",
            choices=[{
                "index": 0,
                "message": {"role": "assistant", "content": "Hello!"},
                "finish_reason": "stop"
            }],
            usage={
                "prompt_tokens": 5,
                "completion_tokens": 2,
                "total_tokens": 7
            }
        )

        data = response.model_dump()

        assert data["object"] == "chat.completion"
        assert data["id"] == "chatcmpl-test123"
        assert len(data["choices"]) == 1
        assert data["choices"][0]["message"]["role"] == "assistant"


class TestStreamingChunk:
    """Tests for streaming chunk format."""

    def test_chunk_json_serialization(self):
        """Chunk should serialize correctly for SSE."""
        chunk = ChatCompletionChunk(
            id="chatcmpl-test",
            created=1706627200,
            model="test-model",
            choices=[{
                "index": 0,
                "delta": {"content": "Hello"},
                "finish_reason": None
            }]
        )

        json_str = chunk.model_dump_json()

        assert '"object":"chat.completion.chunk"' in json_str
        assert '"content":"Hello"' in json_str
```

### Integration Test Patterns

```python
# tests/integration/api/test_chat_completions.py

"""
Integration tests for chat completions endpoint.

Uses TestClient with mock model for speed.
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch, MagicMock
import json


@pytest.fixture
def mock_model_service():
    """Mock model service for testing."""
    service = Mock()
    service.is_loaded.return_value = True
    service.get_current_model.return_value = Mock(
        name="test-model",
        loaded_at=Mock(timestamp=Mock(return_value=1706627200))
    )
    service.get_model.return_value = Mock()
    service.get_tokenizer.return_value = Mock()
    return service


@pytest.fixture
def client(mock_model_service):
    """Create test client with mocked services."""
    from millm.main import create_app

    app = create_app()
    app.state.model_service = mock_model_service
    app.state.inference_service = Mock()

    return TestClient(app)


class TestChatCompletionsEndpoint:
    """Tests for /v1/chat/completions endpoint."""

    def test_non_streaming_success(self, client):
        """Non-streaming completion returns valid response."""
        # Setup mock
        client.app.state.inference_service.is_model_loaded.return_value = True
        client.app.state.inference_service.create_chat_completion.return_value = {
            "id": "chatcmpl-test",
            "object": "chat.completion",
            "created": 1706627200,
            "model": "test-model",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": "Hello!"},
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 5,
                "completion_tokens": 2,
                "total_tokens": 7
            }
        }

        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "test-model",
                "messages": [{"role": "user", "content": "Hi"}],
                "stream": False
            }
        )

        assert response.status_code == 200
        data = response.json()
        assert data["object"] == "chat.completion"
        assert data["choices"][0]["message"]["content"] == "Hello!"

    def test_no_model_loaded_error(self, client):
        """Returns 503 when no model is loaded."""
        client.app.state.inference_service.is_model_loaded.return_value = False

        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "test-model",
                "messages": [{"role": "user", "content": "Hi"}]
            }
        )

        assert response.status_code == 503
        data = response.json()
        assert data["error"]["code"] == "model_not_loaded"

    def test_invalid_temperature_rejected(self, client):
        """Invalid temperature returns 422."""
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "test-model",
                "messages": [{"role": "user", "content": "Hi"}],
                "temperature": 5.0  # Out of range
            }
        )

        assert response.status_code == 422


class TestStreamingCompletion:
    """Tests for streaming responses."""

    def test_streaming_response_format(self, client):
        """Streaming returns valid SSE format."""
        # Mock streaming generator
        async def mock_stream():
            yield 'data: {"id":"test","object":"chat.completion.chunk"}\n\n'
            yield 'data: [DONE]\n\n'

        client.app.state.inference_service.is_model_loaded.return_value = True
        client.app.state.inference_service.stream_chat_completion.return_value = mock_stream()

        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "test-model",
                "messages": [{"role": "user", "content": "Hi"}],
                "stream": True
            },
            headers={"Accept": "text/event-stream"}
        )

        assert response.status_code == 200
        assert "text/event-stream" in response.headers["content-type"]
```

### Compatibility Test Patterns

```python
# tests/compatibility/test_openai_client.py

"""
Compatibility tests using official OpenAI Python library.

Requires running miLLM server with loaded model.
Mark as integration tests that need live server.
"""

import pytest
from openai import OpenAI


@pytest.fixture
def openai_client():
    """Create OpenAI client pointing to miLLM."""
    return OpenAI(
        base_url="http://localhost:8000/v1",
        api_key="not-needed"
    )


@pytest.mark.integration
@pytest.mark.requires_server
class TestOpenAIPythonClient:
    """Tests with official OpenAI Python library."""

    def test_list_models(self, openai_client):
        """Can list models using OpenAI client."""
        models = openai_client.models.list()

        assert hasattr(models, "data")
        # May be empty if no model loaded

    def test_chat_completion_non_streaming(self, openai_client):
        """Non-streaming chat works with OpenAI client."""
        response = openai_client.chat.completions.create(
            model="gemma-2-2b",  # Must match loaded model
            messages=[{"role": "user", "content": "Say hello"}],
            max_tokens=10,
            stream=False
        )

        assert response.id.startswith("chatcmpl-")
        assert len(response.choices) == 1
        assert response.choices[0].message.role == "assistant"
        assert len(response.choices[0].message.content) > 0
        assert response.usage.total_tokens > 0

    def test_chat_completion_streaming(self, openai_client):
        """Streaming chat works with OpenAI client."""
        stream = openai_client.chat.completions.create(
            model="gemma-2-2b",
            messages=[{"role": "user", "content": "Count to 3"}],
            max_tokens=50,
            stream=True
        )

        chunks = list(stream)

        # First chunk should have role
        assert chunks[0].choices[0].delta.role == "assistant"

        # Middle chunks should have content
        content_chunks = [
            c for c in chunks
            if c.choices[0].delta.content
        ]
        assert len(content_chunks) > 0

        # Last chunk should have finish_reason
        assert chunks[-1].choices[0].finish_reason == "stop"

    def test_embeddings(self, openai_client):
        """Embeddings work with OpenAI client."""
        response = openai_client.embeddings.create(
            model="gemma-2-2b",
            input="Test text for embedding"
        )

        assert len(response.data) == 1
        assert len(response.data[0].embedding) > 0
        assert response.usage.total_tokens > 0
```

---

## 11. Common Patterns and Anti-Patterns

### DO: Use Pydantic's model_dump_json() for SSE

```python
# Correct: Use model_dump_json() for consistent serialization
chunk = ChatCompletionChunk(...)
yield f"data: {chunk.model_dump_json()}\n\n"

# Incorrect: Manual JSON serialization may miss defaults
import json
yield f"data: {json.dumps(chunk.__dict__)}\n\n"  # Wrong!
```

### DO: Check Model Loaded First

```python
# Correct: Check before any processing
if not inference.is_model_loaded():
    return create_openai_error(...)

# Incorrect: Check after processing (wastes resources)
prompt = format_messages(request.messages)  # Wasted work if no model
if not inference.is_model_loaded():
    ...
```

### DO: Use Context Manager for Request Queue

```python
# Correct: Ensures cleanup even on error
async with self._request_queue.acquire():
    result = await generate(...)
return result

# Incorrect: Manual acquire/release can leak
await self._request_queue.acquire()
result = await generate(...)  # If this errors, slot not released!
self._request_queue.release()
```

### DON'T: Modify Request Objects

```python
# Incorrect: Mutating request
request.temperature = 0.0 if request.temperature == 0 else request.temperature

# Correct: Create new config
gen_config = GenerationConfig.from_request(request)
```

### DON'T: Block Async with Synchronous Generation

```python
# Incorrect: Blocking async event loop
async def generate():
    result = model.generate(...)  # Blocks!
    return result

# Correct: Use thread for blocking operations
async def generate():
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, model.generate, ...)
    return result
```

---

## 12. Performance Optimization Hints

### Token Counting Efficiency

```python
# Cache tokenization for repeated prompts
class TokenCounter:
    def __init__(self, tokenizer, cache_size=100):
        self._tokenizer = tokenizer
        self._cache = {}  # Simple dict, could use LRU

    def count(self, text: str) -> int:
        if text not in self._cache:
            self._cache[text] = len(self._tokenizer.encode(text))
        return self._cache[text]
```

### Streaming Chunk Batching

```python
# For very fast models, batch tokens to reduce SSE overhead
async def stream_with_batching(streamer, batch_size=2):
    buffer = []
    for token in streamer:
        buffer.append(token)
        if len(buffer) >= batch_size:
            yield "".join(buffer)
            buffer = []
    if buffer:
        yield "".join(buffer)
```

### Memory Management

```python
# Clear CUDA cache between requests if needed
import torch

async def generate_with_cleanup(...):
    try:
        result = await generate(...)
        return result
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
```

---

## 13. Error Handling Checklist

- [ ] Model not loaded → 503 with `model_not_loaded` code
- [ ] Model ID mismatch → 404 with `model_not_found` code
- [ ] Invalid request params → 422 with validation errors
- [ ] Queue full → 503 with `rate_limit_exceeded` code
- [ ] Generation timeout → Return partial with `timeout` finish_reason
- [ ] OOM during generation → 507 with helpful message
- [ ] Client disconnect → Cleanup gracefully, no error logged
- [ ] Malformed JSON → 400 with parse error

---

**Document Status:** Complete
**Next Document:** `002_FTASKS|OpenAI_API.md` (Task List)
**Instruction File:** `@0xcc/instruct/006_generate-tasks.md`
