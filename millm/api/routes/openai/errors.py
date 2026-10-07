"""
OpenAI error format helpers.

Provides utilities for creating OpenAI-compatible error responses.
All errors must match the OpenAI error response format exactly.

Error response format:
{
    "error": {
        "message": "Human-readable error message",
        "type": "error_type",
        "param": "parameter_name" | null,
        "code": "error_code" | null
    }
}
"""

from typing import Optional

from fastapi import Request
from fastapi.responses import JSONResponse

from millm.core.errors import MiLLMError


def create_openai_error(
    message: str,
    error_type: str = "server_error",
    code: Optional[str] = None,
    param: Optional[str] = None,
    status_code: int = 500,
    retry_after: Optional[int] = None,
) -> JSONResponse:
    """
    Create OpenAI-format error response.

    Args:
        message: Human-readable error message
        error_type: One of invalid_request_error, authentication_error,
                   rate_limit_error, server_error
        code: Machine-readable error code (e.g., "model_not_found")
        param: Parameter that caused the error (e.g., "model")
        status_code: HTTP status code
        retry_after: Seconds for a `Retry-After` HEADER (Feature 29: every 503 carries one,
            from `backpressure.retry_after_for`). Never a body field: the envelope stays
            exactly OpenAI's (FR-29.6.4).

    Returns:
        JSONResponse with OpenAI error format
    """
    return JSONResponse(
        status_code=status_code,
        content={
            "error": {
                "message": message,
                "type": error_type,
                "param": param,
                "code": code,
            }
        },
        headers={"Retry-After": str(retry_after)} if retry_after else None,
    )


class OpenAIRefusal(Exception):
    """A pre-service refusal carrying the EXACT response the synchronous route returns.

    Raised by the route modules' `validate_<endpoint>()` (Feature 26, FTDD §5): the route returns
    `response` unchanged; the batch validator reads the same status and body off it, so a batch
    line is refused with the code and message the synchronous endpoint gives (FR-26.2.4).
    """

    def __init__(self, response: JSONResponse) -> None:
        super().__init__(response.body.decode("utf-8", "replace"))
        self.response = response

    @property
    def status_code(self) -> int:
        return int(self.response.status_code)

    def body(self) -> dict:
        import json

        return json.loads(self.response.body)


# Error code to (HTTP status, OpenAI error type) mapping
ERROR_STATUS_MAP: dict[str, tuple[int, str]] = {
    # Probe monitors (Feature 24). Present so a probe refusal reaching a `/v1` route renders as
    # an OpenAI error rather than a bare 500 — the arm-time refusals are management-plane, but
    # PROBE_NOT_FOUND can surface from a per-request path.
    "PROBE_NOT_FOUND": (404, "invalid_request_error"),
    "PROBE_MODEL_MISMATCH": (409, "invalid_request_error"),
    "PROBE_PARITY_FAILED": (409, "invalid_request_error"),
    "UNVALIDATED_PROBE": (200, "invalid_request_error"),
    "PROBE_LIMIT": (409, "invalid_request_error"),
    "PROBE_SAE_MISSING": (409, "invalid_request_error"),
    "PROBE_SAE_MISMATCH": (409, "invalid_request_error"),
    "PROBE_NO_MODEL_LOADED": (409, "invalid_request_error"),
    "PROBE_HOOK_UNSUPPORTED": (409, "invalid_request_error"),
    "PROBE_SCOPE_UNVERIFIABLE": (409, "invalid_request_error"),
    # Feature 27. The activation refusal reaches /v1 callers; the scoring one is management-plane
    # but is listed so every MiLLMError code has a row.
    "SAE_ACTIVATIONS_REFUSED": (400, "invalid_request_error"),
    "INVALID_PROBE_SCORE_REQUEST": (400, "invalid_request_error"),
    # Model errors
    "MODEL_NOT_LOADED": (503, "server_error"),
    "MODEL_NOT_FOUND": (404, "invalid_request_error"),
    "MODEL_ALREADY_LOADED": (400, "invalid_request_error"),
    "MODEL_LOADING": (503, "server_error"),
    # A load is in progress, or the model the request names is being unloaded:
    # retry once it finishes. Hardware acceptance, 2026-09-14 (item 11): a request
    # during an unload ran on a half-moved model and answered 500.
    "MODEL_BUSY": (503, "server_error"),
    # Validation errors
    "VALIDATION_ERROR": (400, "invalid_request_error"),
    # Steering profile / dial errors (Feature 8/10): unknown resource and
    # bad indices are the caller's problem, not the server's — without these
    # rows the fallback branded them "server_error" (010 R2 find)
    "PROFILE_NOT_FOUND": (404, "invalid_request_error"),
    "INVALID_FEATURE_INDEX": (400, "invalid_request_error"),
    "CONTEXT_LENGTH_EXCEEDED": (400, "invalid_request_error"),
    "AMBIGUOUS_MODEL_NAME": (400, "invalid_request_error"),
    "INVALID_PARAMETER": (400, "invalid_request_error"),
    # The resident engine cannot do this at all (a GGUF file served by
    # llama.cpp has no module tree). The caller must ask for something else, so
    # it is invalid_request_error. Without this row the live handler
    # (millm_error_handler) falls back to (exc.status_code, "server_error") —
    # the right 400 wearing the wrong TYPE, which an OpenAI client reads as a
    # server fault to retry rather than a request to change. Note the fallback
    # in openai_exception_handler below is (500, "server_error"), but that
    # handler is not the one registered in main.py.
    "ENGINE_UNSUPPORTED": (400, "invalid_request_error"),
    # A request naming a Q2 transformers checkpoint auto-loads it, and the load
    # is refused (bitsandbytes has no 2-bit mode). The same reasoning: the caller
    # must name another model, so not a server fault to retry. Review round 2,
    # 2026-09-14.
    "UNSUPPORTED_QUANTIZATION": (400, "invalid_request_error"),
    # Scoring (2026-10-04). Without these rows a scoring 400 — a token id outside the
    # vocabulary — reached OpenAI clients typed "server_error", which their SDKs retry
    # (Feature 25, 025_FTDD §5.5).
    "INVALID_SCORING_REQUEST": (400, "invalid_request_error"),
    "NON_FINITE_LOGITS": (500, "server_error"),
    # Request validation, chat scoring and structured output (Feature 25).
    "FIELD_NOT_HONOURED": (400, "invalid_request_error"),
    "UNUSED_FIELDS_REFUSED": (400, "invalid_request_error"),
    "RESPONSE_FORMAT_UNSUPPORTED": (400, "invalid_request_error"),
    "NO_CHAT_TEMPLATE": (400, "invalid_request_error"),
    "CONSTRAINED_OUTPUT_INVALID": (500, "server_error"),
    # Feature 30: a pooled embedding vector that is non-finite or has a zero norm under
    # `normalize`. The request was valid; the server could not produce a finite vector.
    "EMBEDDING_VECTOR_INVALID": (500, "server_error"),
    # Every other MiLLMError code, so the map is COMPLETE and a /v1 request never meets
    # the fallback (exc.status_code, "server_error") — which types a 4xx as a server
    # fault to retry. test_error_map_complete.py walks MiLLMError.__subclasses__() and
    # fails on a code with no row. Statuses are each class's own; the type follows it
    # (4xx a request to change, 5xx a server fault). The two HF-token codes are 401s
    # about the SERVER's HuggingFace credential, not the caller's API key, so they are
    # not "authentication_error".
    "SENSING_EVENT_NOT_FOUND": (404, "invalid_request_error"),
    "CIRCUIT_SENSING_EVENT_NOT_FOUND": (404, "invalid_request_error"),
    "CIRCUIT_NOT_FOUND": (404, "invalid_request_error"),
    "CIRCUIT_LAYER_CONTENTION": (200, "invalid_request_error"),
    "NO_ACTIVE_CIRCUIT": (200, "invalid_request_error"),
    "UNVALIDATED_CIRCUIT": (200, "invalid_request_error"),
    "MODEL_ALREADY_EXISTS": (409, "invalid_request_error"),
    "MODEL_LOAD_FAILED": (500, "server_error"),
    "MODEL_LOCKED": (409, "invalid_request_error"),
    # Feature 29: the model lease. MODEL_LEASED and MODEL_NOT_RESIDENT reach /v1 (an
    # auto-load under a foreign lease; the refuse-load policy). The three lease-route
    # codes are management-plane, present so the map stays complete.
    "MODEL_LEASED": (409, "invalid_request_error"),
    "MODEL_NOT_RESIDENT": (409, "invalid_request_error"),
    "LEASE_NOT_FOUND": (404, "invalid_request_error"),
    "LEASE_EXPIRED": (409, "invalid_request_error"),
    "INVALID_LEASE_REQUEST": (400, "invalid_request_error"),
    "INVALID_GGUF_TENSOR_SPLIT": (500, "server_error"),
    "SPLIT_NOT_HONOURED": (409, "invalid_request_error"),
    "DOWNLOAD_CANCELLED": (499, "invalid_request_error"),
    "DOWNLOAD_FAILED": (502, "server_error"),
    "GATED_MODEL_NO_TOKEN": (401, "invalid_request_error"),
    "INVALID_HF_TOKEN": (401, "invalid_request_error"),
    "REPO_NOT_FOUND": (404, "invalid_request_error"),
    "INVALID_LOCAL_PATH": (400, "invalid_request_error"),
    "INSUFFICIENT_DISK": (507, "server_error"),
    "GPU_NOT_FOUND": (404, "invalid_request_error"),
    "HUB_UNAVAILABLE": (503, "server_error"),
    "INVALID_PROFILE_FORMAT": (400, "invalid_request_error"),
    "PROFILE_ALREADY_EXISTS": (409, "invalid_request_error"),
    "PROFILE_INCOMPATIBLE": (400, "invalid_request_error"),
    "PROBE_DTYPE_MISMATCH": (409, "invalid_request_error"),
    "PROBE_RECALIBRATION_MISMATCH": (409, "invalid_request_error"),
    "PROBE_THRESHOLD_UNCALIBRATED": (409, "invalid_request_error"),
    "SAE_ALREADY_ATTACHED": (409, "invalid_request_error"),
    "SAE_INCOMPATIBLE": (400, "invalid_request_error"),
    "SAE_LOAD_FAILED": (500, "server_error"),
    "SAE_NOT_ATTACHED": (400, "invalid_request_error"),
    "SAE_NOT_FOUND": (404, "invalid_request_error"),
    "SAE_SET_INCOMPLETE": (422, "invalid_request_error"),
    # Feature 26: the Batch API.
    "INVALID_BATCH_REQUEST": (400, "invalid_request_error"),
    "BATCH_FILE_LIMIT": (400, "invalid_request_error"),
    "FILE_NOT_FOUND": (404, "invalid_request_error"),
    "FILE_EXPIRED": (404, "invalid_request_error"),
    "FILE_DELETED": (404, "invalid_request_error"),
    "FILE_IN_USE": (409, "invalid_request_error"),
    "BATCH_NOT_FOUND": (404, "invalid_request_error"),
    "BATCH_STATE_CONFLICT": (409, "invalid_request_error"),
    # Resource errors
    "INSUFFICIENT_MEMORY": (503, "server_error"),
    "RATE_LIMIT_EXCEEDED": (429, "rate_limit_error"),
    "QUEUE_FULL": (503, "server_error"),
    # Authentication (for future use)
    "AUTHENTICATION_ERROR": (401, "authentication_error"),
    "INVALID_API_KEY": (401, "authentication_error"),
    # Generic errors
    "SERVER_ERROR": (500, "server_error"),
    "INTERNAL_ERROR": (500, "server_error"),
}


async def openai_exception_handler(request: Request, exc: MiLLMError) -> JSONResponse:
    """
    Global exception handler for MiLLM errors on OpenAI endpoints.

    Converts MiLLMError exceptions to OpenAI-compatible error responses.
    Only handles requests to /v1/* endpoints.

    Register with FastAPI:
        app.add_exception_handler(MiLLMError, openai_exception_handler)

    Args:
        request: The FastAPI request object
        exc: The MiLLMError exception

    Returns:
        JSONResponse with OpenAI error format
    """
    # Get status code and error type from mapping
    status_code, error_type = ERROR_STATUS_MAP.get(exc.code, (500, "server_error"))

    return create_openai_error(
        message=exc.message,
        error_type=error_type,
        code=exc.code.lower() if exc.code else None,
        param=None,
        status_code=status_code,
    )


def model_not_loaded_error() -> JSONResponse:
    """Create error response for when no model is loaded."""
    from millm.core.backpressure import retry_after_for

    return create_openai_error(
        message=(
            "No model is currently loaded. Load a model first using the Management API; "
            "a retry succeeds only after a model is loaded."
        ),
        error_type="server_error",
        code="model_not_loaded",
        status_code=503,
        retry_after=retry_after_for("MODEL_NOT_LOADED"),
    )


def model_not_found_error(model_id: str, available_model: Optional[str] = None) -> JSONResponse:
    """Create error response for model not found."""
    if available_model:
        message = f"The model '{model_id}' does not exist. Available: {available_model}"
    else:
        message = (
            f"The model '{model_id}' does not exist or has not been downloaded. "
            "Download it first using the Management API."
        )

    return create_openai_error(
        message=message,
        error_type="invalid_request_error",
        code="model_not_found",
        param="model",
        status_code=404,
    )


# Architectures that produce EMBEDDINGS, not text. A model tagged with one of
# these has no usable language-model head, and asking it to generate returns
# token soup — verified on Nemotron-3-Embed-8B-BF16, which answered a chat
# prompt with several hundred tokens of multilingual fragments.
#
# This became reachable when the OpenAI endpoints started loading whatever model
# a request names: before that a fixed model was pinned and selecting anything
# else in a client did nothing, so the failure could not be triggered.
EMBEDDING_ARCHITECTURES = frozenset({
    # Embedders — no language-model head at all.
    "sentence-similarity",
    "feature-extraction",
    "sentence-transformers",
    "text-embedding",
    # Encoder CLASSIFIERS — they emit label logits, not tokens. NLI and
    # zero-shot models (p-christ/ModernBERT-large-nli returns
    # entailment/neutral/contradiction over 3 labels) are enormously useful for
    # SCORING a label against a passage, and completely unable to write one.
    "zero-shot-classification",
    "text-classification",
    "token-classification",
})


def is_embedding_only(architecture: Optional[str]) -> bool:
    """True when this architecture cannot generate text.

    Covers embedders AND encoder classifiers. Both produce numbers rather than
    tokens; asking either to chat is a category error, not a capacity problem.
    """
    return bool(architecture) and architecture.strip().lower() in EMBEDDING_ARCHITECTURES


def embedding_model_error(model_id: str, architecture: str) -> JSONResponse:
    """Refuse a generation request aimed at an embedding model.

    A clear refusal, not a 500 and emphatically not garbage output: a user who
    picked the wrong entry in a model list needs to be told which entry it was
    and what it is for.
    """
    return create_openai_error(
        message=(
            f"The model '{model_id}' is a non-generative model (architecture: "
            f"{architecture}) and cannot produce text. Embedding models are "
            f"served by /v1/embeddings; classification and NLI models score "
            f"inputs rather than write them. Select a text-generation model "
            f"for chat."
        ),
        error_type="invalid_request_error",
        code="model_not_generative",
        param="model",
        status_code=400,
    )


def model_locked_error(model_id: str, locked_model: str) -> JSONResponse:
    """Create error response when model is locked for steering."""
    return create_openai_error(
        message=f"The model '{model_id}' is not available. "
        f"Model '{locked_model}' is currently locked for steering.",
        error_type="invalid_request_error",
        code="model_locked",
        param="model",
        status_code=409,
    )


def model_not_resident_error(
    requested: str, resident: Optional[str], lease: Optional[dict] = None
) -> JSONResponse:
    """`X-miLLM-Load-Policy: refuse` and the model is not resident: 409, and nothing loads.

    Names the requested and the resident model ("none" when nothing is loaded). When the
    resident model is leased the message says by whom and until when, so the caller sees why
    the model may not change soon (FR-29.4.2, FR-29.4.4). The lease ID is never in it.
    """
    message = (
        f"The model '{requested}' is not resident and this request asked not to load it "
        f"(X-miLLM-Load-Policy: refuse). Resident model: '{resident or 'none'}'."
    )
    if lease:
        message += (
            f" It is leased by '{lease['holder']}' until {lease['expires_at']} "
            f"({lease['reason']})."
        )
    return create_openai_error(
        message=message,
        error_type="invalid_request_error",
        code="model_not_resident",
        param="model",
        status_code=409,
    )


def validation_error(message: str, param: Optional[str] = None) -> JSONResponse:
    """Create error response for validation errors."""
    return create_openai_error(
        message=message,
        error_type="invalid_request_error",
        code="invalid_parameter",
        param=param,
        status_code=400,
    )


def context_length_exceeded_error(
    max_length: int, requested_length: int
) -> JSONResponse:
    """Create error response for context length exceeded."""
    return create_openai_error(
        message=f"This model's maximum context length is {max_length} tokens. "
        f"However, your messages resulted in {requested_length} tokens.",
        error_type="invalid_request_error",
        code="context_length_exceeded",
        status_code=400,
    )


def rate_limit_error(message: str = "Rate limit exceeded") -> JSONResponse:
    """Create error response for rate limiting."""
    return create_openai_error(
        message=message,
        error_type="rate_limit_error",
        code="rate_limit_exceeded",
        status_code=429,
    )


def load_refused_error(model_id: str, exc: MiLLMError) -> JSONResponse:
    """A load this request triggered was refused or failed: the refusal's own status and type.

    The chat, completions and embeddings routes load the model a request names,
    and answered EVERY load error as a 500 `server_error`. That left the rows
    ERROR_STATUS_MAP keeps for load refusals unreachable on the one path that
    raises them from /v1: a Q2 transformers checkpoint (UNSUPPORTED_QUANTIZATION,
    a request to change) went out as a server fault to retry, and a model no card
    or split holds (INSUFFICIENT_MEMORY) as 500 rather than 503. Review round 2
    added the UNSUPPORTED_QUANTIZATION row for exactly this request and tested it
    against millm_error_handler, which a caught exception never reaches. Review
    round 3, 2026-09-14.

    Codes without a row keep their own status, typed server_error — the fallback
    millm_error_handler uses.
    """
    from millm.core.backpressure import retry_after_for

    status_code, error_type = ERROR_STATUS_MAP.get(exc.code, (exc.status_code, "server_error"))
    return create_openai_error(
        message=f"Could not load '{model_id}': {exc}",
        error_type=error_type,
        code=exc.code.lower() if exc.code else None,
        status_code=status_code,
        retry_after=retry_after_for(exc.code, exc.details) if status_code == 503 else None,
    )


def model_busy_error(message: str, details: Optional[dict] = None) -> JSONResponse:
    """A request that will succeed once a load or unload in progress finishes: 503 model_busy.

    The same answer millm_error_handler gives a ModelBusyError raised past the
    route (ERROR_STATUS_MAP), so "busy, retry" reads one way whichever layer
    found it. It was a 500 server_error from the route and, for a request that
    caught its model mid-unload, a 500 device-mismatch error (hardware
    acceptance, 2026-09-14, item 11).

    `details` are the ModelBusyError's: an unload in progress gets the shorter Retry-After.
    """
    from millm.core.backpressure import retry_after_for

    return create_openai_error(
        message=message,
        error_type="server_error",
        code="model_busy",
        status_code=503,
        retry_after=retry_after_for("MODEL_BUSY", details),
    )


def server_error(message: str = "Internal server error") -> JSONResponse:
    """Create generic server error response."""
    return create_openai_error(
        message=message,
        error_type="server_error",
        code="server_error",
        status_code=500,
    )
