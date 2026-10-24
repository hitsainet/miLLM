"""
OpenAI-compatible chat completions endpoint.

POST /v1/chat/completions - Create chat completion

Supports both streaming and non-streaming responses.
Requires a model to already be loaded via the Management API.
"""

import asyncio
from typing import Annotated, Any, Optional, Union

from fastapi import APIRouter, Depends, Header, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse

from millm.api import json_schema_subset
from millm.api.probe_request_meta import (
    ORIGIN_HEADER,
    SESSION_HEADER,
    parse_origin_header,
    parse_session_header,
    resolve_request_meta,
)
from millm.api.probe_request_meta import response_headers as probe_session_headers
from millm.api.dependencies import ModelServiceDep, get_inference_service
from millm.api.request_policy import (
    IGNORED_FIELDS_HEADER,
    STRICT_HEADER,
    Endpoint,
    PolicyResult,
    engine_of,
    model_name_of,
    evaluate,
    ignored_fields_header,
    parse_strict,
    report_unused,
)
from millm.api.routes.openai.load_policy import (
    apply_load_policy,
    parse_load_policy,
    refuse_inline_steering_before_load,
)
from millm.api.routes.openai.errors import (
    OpenAIRefusal,
    refuse_if_non_generative,
    load_refused_error,
    model_busy_error,
    model_locked_error,
    model_not_found_error,
    model_not_loaded_error,
    server_error,
)
from millm.api.routes.openai.model_resolution import (
    load_kwargs,
    resident_names,
    resolve_requested_model,
    served_name_of,
)
from millm.api.schemas.openai import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    OpenAIErrorResponse,
)
from millm.core.errors import (
    MiLLMError,
    ModelBusyError,
    ModelLockedError,
    ResponseFormatUnsupportedError,
)
from millm.core.logging import get_logger
from millm.ml.constrained_decoding import constraint_kind, schema_of
from millm.services.inference_service import (
    InferenceService,
)


def build_probe_verdicts_header(results: list, max_bytes: Optional[int] = None) -> str:
    """`X-miLLM-Probe-Verdicts` as an RFC 8941 structured list (FR-24.7), FROM THE RESULTS (038).

        "<name>";score=2.31;threshold=1.07;verdict=?1;rung=3;window=all;behaviour="<key>";probe="pr_…",
        "<name>";not-scored;reason="<code>";rung=2;window=prompt;behaviour="<key>";probe="pr_…",
        "rule:<rs_id>.<signal>";verdict=?1;count=2;behaviour="<key>";window=response

    ⚠ A PROJECTION OF THE FINALIZED `mi.probe.result/v1` OBJECTS (FR-38.3), never of the verdicts:
    the header, the terminal chunk, the body field and the stored row are one object. `reason` is
    always a CODE (FR-38.5). No `result_id` (D26, OQ-11 (a)).

    039 (FR-39.10, FR-39.12, TD39-30):

    * the ORDER is the results' own — `probe_result_order` sorted them by (behaviour, kind, name,
      window) before they were published; this builder never sorts again;
    * every string goes through ONE serializer, `sf_string` (`\\` and `"` escaped, non-ASCII →
      `?`), and each member names its `behaviour` and `probe` as parameters;
    * the list is BOUNDED (`PROBE_VERDICT_HEADER_MAX_BYTES`): whole behaviour groups, then
      `"...";truncated;total=<n>`. The full list is always in the body, the chunk and the events.

    Returns "" when nothing is armed, and the caller then sets no header at all: an unarmed
    server's response must be byte-identical to what it was before this feature existed.
    """
    from millm.api.structured_header import bounded_members, sf_decimal, sf_string
    from millm.core.config import settings
    from millm.services.probe_result import legacy_projection

    if not results:
        return ""
    limit = int(max_bytes if max_bytes is not None else settings.PROBE_VERDICT_HEADER_MAX_BYTES)
    groups: list[list[str]] = []
    last_key: object = object()
    for result in results:
        behaviour = getattr(result, "behaviour", None)
        key = getattr(behaviour, "key", None) if behaviour is not None else None
        if key != last_key or key is None:
            # A behaviour's members are contiguous (the results' order), so one group per run;
            # a result with no behaviour is a group of its own.
            groups.append([])
            last_key = key
        groups[-1].append(_header_member(result, key, legacy_projection, sf_string, sf_decimal))
    return bounded_members(groups, limit, total=len(results))


def _header_member(result: Any, key: Optional[str], legacy_projection: Any, sf_string: Any,
                   sf_decimal: Any) -> str:
    """One member. The window stays a TOKEN parameter (its values are fixed lowercase words, and a
    v1.6 consumer reads it as one); every free string is an `sf_string`."""
    tail_behaviour = [f"behaviour={sf_string(key)}"] if key else []
    if getattr(result, "kind", "probe") == "rule":
        parts = [sf_string(f"rule:{result.rule_set_id}.{result.signal}")]
        if result.outcome == "completed" and result.fires is not None:
            parts.append(f"verdict={'?1' if result.fires else '?0'}")
            parts.append(f"count={int(result.count or 0)}")
        else:
            parts.append(f"not-scored;reason={sf_string(result.reason or 'unknown')}")
        parts.extend(tail_behaviour)
        parts.append(f"window={result.window}")
        return ";".join(parts)
    entry = legacy_projection(result)
    name = sf_string(entry["name"] if entry["name"] is not None else entry["probe_id"])
    # The window is a PARAMETER, not part of the name token: a consumer that only knows
    # v1.6 keeps reading the name it already reads, and ignores parameters it does not know.
    tail = [f"window={entry['window']}"]
    if entry["provisional"]:
        tail.append("provisional=?1")
    tail.extend(tail_behaviour)
    tail.append(f"probe={sf_string(entry['probe_id'])}")
    if not entry["scored"]:
        reason = sf_string(entry["reason"] or "unknown")
        return f"{name};not-scored;reason={reason};rung={entry['rung']};" + ";".join(tail)
    parts = [name, f"score={sf_decimal(entry['score'])}"]
    if entry["threshold"] is not None:
        parts.append(f"threshold={sf_decimal(entry['threshold'])}")
    # ⚠ `verdict` is omitted entirely when it is None — the probe ranked without deciding,
    # and `?0` would report a decision it never made.
    if entry["verdict"] is not None:
        parts.append(f"verdict={'?1' if entry['verdict'] else '?0'}")
    parts.append(f"rung={entry['rung']}")
    parts.extend(tail)
    return ";".join(parts)

router = APIRouter()
logger = get_logger(__name__)

# Imported after `build_probe_verdicts_header` is defined: provenance reads it from this module.
from millm.api.provenance import (  # noqa: E402
    finish_body,
    post_generation,
    pre_generation,
    seed_header,
)


def validate_chat(
    request: ChatCompletionRequest, model_row: Any, *, strict: bool, cbm_enabled: bool
) -> PolicyResult:
    """Every chat refusal decidable from the request and the model ROW (no load, no slot).

    Called by the route and by the Batch API validator (026 FTDD §5): a batch line is refused
    with the same code and message as the synchronous request. Raises `OpenAIRefusal` for a
    response-shaped refusal and a `MiLLMError` for a policy one.
    """
    # An embedding model cannot generate text. Refuse before loading it —
    # loading takes time and VRAM to reach an answer that is guaranteed to be
    # nonsense. Nemotron-3-Embed-8B-BF16 answered a chat prompt with hundreds of
    # tokens of multilingual fragments rather than failing, which is worse than
    # an error because it looks like output.
    # The row's tag, else the checkpoint's own config.json (a local folder has no Hub tag).
    refuse_if_non_generative(request.model, model_row)

    # Feature 25 request policy: every output-changing field honoured or refused, every
    # unused field reported (or refused under X-miLLM-Strict).
    policy = evaluate(request, Endpoint.CHAT, engine_of(model_row), strict=strict,
        model_name=model_name_of(model_row),
    )

    # Structured output refusals decidable without a load (FR-25.11): a schema keyword outside
    # the enforced subset, and the continuous batching manager. (GGUF rows were refused by the
    # policy table above.)
    if constraint_kind(request.response_format) is not None:
        if constraint_kind(request.response_format) == "json_schema":
            json_schema_subset.check(schema_of(request.response_format))
        if cbm_enabled:
            raise ResponseFormatUnsupportedError(
                "Structured output is refused while continuous batching is enabled on this "
                "server: the batching manager applies no per-request constraint.",
                details={"param": "response_format"},
            )
    return policy


@router.post(
    "/chat/completions",
    response_model=ChatCompletionResponse,
    responses={
        503: {"model": OpenAIErrorResponse, "description": "No model loaded"},
    },
)
async def create_chat_completion(
    request: ChatCompletionRequest,
    service: ModelServiceDep,
    response: Response,
    http_request: Request,
    inference: InferenceService = Depends(get_inference_service),
    x_millm_lease: Annotated[str | None, Header(alias="X-miLLM-Lease")] = None,
    x_millm_load_policy: Annotated[str | None, Header(alias="X-miLLM-Load-Policy")] = None,
    x_millm_session: Annotated[str | None, Header(alias=SESSION_HEADER)] = None,
    x_millm_origin: Annotated[str | None, Header(alias=ORIGIN_HEADER)] = None,
) -> Union[ChatCompletionResponse, StreamingResponse, JSONResponse]:
    """
    Create a chat completion.

    Accepts messages in OpenAI format and returns a completion.
    Supports both streaming (stream=true) and non-streaming responses.
    Auto-loads the requested model if not already loaded.
    """
    # Feature 36: the session header is refused (400 SESSION_ID_INVALID) before anything else —
    # before a model lookup, a load or a queue slot (FR-36.1).
    client_session_id = parse_session_header(x_millm_session)
    # 036 phase 4 (D14): `X-miLLM-Origin: judge` — anything else stated is refused (400).
    origin = parse_origin_header(x_millm_origin)

    # Which model the name means (032 FR-32.6.3): the resident under its served name or its base
    # name without a load, a database row as before, or a refusal (an unknown name answers the
    # same 404 it always did; a checkpoint row is never loaded on demand, FR-32.6.4).
    resolution = await resolve_requested_model(request.model, service, inference)
    if resolution.refusal is not None:
        return resolution.refusal
    model = resolution.row

    # Every refusal decidable from the request and the ROW, before the auto-load below would
    # evict the resident model. One copy, two callers: this route and the batch validator
    # (Feature 26, FR-26.2.4).
    try:
        policy = validate_chat(
            request, model,
            strict=parse_strict(http_request.headers.get(STRICT_HEADER)),
            cbm_enabled=inference.cbm_enabled() is True,
        )
    except OpenAIRefusal as refusal:
        return refusal.response
    report_unused(policy, Endpoint.CHAT, http_request.headers)
    ignored_header = ignored_fields_header(policy)

    # Load the requested model on demand.
    #
    # An OpenAI client — Open WebUI included — selects a model by naming it in
    # the request body. Rejecting anything that is not already loaded makes
    # model selection a no-op: the picker changes, the request 404s, and the
    # user has to go and load the model by hand somewhere else.
    #
    # load_model_and_wait() was written for exactly this ("Used by the
    # OpenAI-compatible endpoints for auto-load on demand") and had NO callers.
    # It returns immediately when the model is already loaded, so the common
    # path costs nothing.
    #
    # A model LOCKED for steering is the one case where the model must not
    # change: swapping the weights out from under an attached SAE would leave
    # the steering vectors pointing at a different model. load_model_and_wait
    # raises ModelLockedError for that, and only that.
    # Feature 29: `X-miLLM-Load-Policy: refuse` promises this request never causes a swap.
    # After the pre-load refusals above, before the auto-load below (FR-29.4.5).
    # Feature 28 (T-82): a non-empty inline set on a non-resident model can never be honoured —
    # refused before the auto-load below would evict the resident model and its SAEs.
    refuse_inline_steering_before_load(request, inference)

    load_policy = parse_load_policy(x_millm_load_policy)
    policy_refusal = await apply_load_policy(load_policy, model, inference, service)
    if policy_refusal is not None:
        return policy_refusal

    if not resolution.resident:
        try:
            # The holder's lease ID lifts a lease refusal on the swap; a foreign lease answers
            # 409 model_leased through load_refused_error below (never model_locked).
            await service.load_model_and_wait(
                model.id, lease_id=x_millm_lease, **load_kwargs(resolution)
            )
        except ModelLockedError as exc:
            locked_name = (exc.details or {}).get("locked_model_name")
            if not locked_name:
                locked = await service.get_locked_model()
                locked_name = locked.name if locked else "unknown"
            return model_locked_error(request.model, locked_name)
        except ModelBusyError as exc:
            return model_busy_error(
                f"{exc.message}. Retry once it finishes before requesting '{request.model}'.",
                exc.details,
            )
        except asyncio.TimeoutError:
            return server_error(
                f"Timed out loading '{request.model}'. The model may still be "
                f"loading; retry shortly."
            )
        except MiLLMError as exc:
            return load_refused_error(request.model, exc)

        # Confirm the switch actually happened rather than assuming it did.
        model_info = inference.get_loaded_model_info()
        if not model_info:
            return model_not_loaded_error()
        if request.model not in resident_names(model_info):
            return model_not_found_error(request.model, served_name_of(model_info))

    # Feature 27: every `return_sae_activations` refusal, after the model is resident and before
    # any slot or generation (FR-27.1i, FR-27.2). A refused request never generated anything.
    from millm.services.request_activations import refuse_before_generation

    refuse_before_generation(request, inference, chat=True)

    # Profile override (request.profile) is applied inside the inference service's
    # request-queue semaphore to prevent concurrent requests from racing on the
    # global SAE steering state.  The previous steering is restored after each
    # generation completes.  See InferenceService._apply_request_steering.

    logger.info(
        "chat_completion_request",
        model=request.model,
        message_count=len(request.messages),
        stream=request.stream,
    )

    # X-miLLM-Backend header lets clients distinguish which inference path served
    # the request (serial queue vs continuous batching) for latency debugging.
    backend = inference.backend_name

    # X-miLLM-Steering-Intensity echoes the resolved lambda back to dial
    # clients (Feature 10) — emitted only when the field was present.  Resolved
    # here (not stashed at apply time) because streaming headers are sent
    # before the generator body runs.  Best-effort by design: the helper
    # returns None (no header) when nothing can apply (no SAE, unknown
    # profile, DB hiccup), and a concurrent profile switch while the request
    # queues can still skew a symbolic echo — documented in the API reference.
    # X-miLLM-Circuit-Rung tells a dial client WHAT it is steering with
    # (Feature 14). The phrase comes from the evidence ladder, never composed
    # here, so the header can never describe a rung<2 circuit as causal.
    # Drop any memoised steering verdict from a previous request sharing this context, and
    # resolve the two echoes that must be known before the first byte (streaming commits its
    # headers then). One function for the route and the Batch API (`millm.api.provenance`).
    pre = await pre_generation(request, inference, chat=True)
    echo_intensity = pre.echo_intensity
    echo_circuit_rung = pre.echo_circuit_rung

    # Feature 36: which conversation this is, decided HERE — after `pre_generation` reset the
    # request context, before a stream commits its headers. Only while a probe is armed (D40);
    # a resolve that fails is counted and the request is served without a session.
    probe_meta = await resolve_request_meta(
        request.messages, client_session_id, served_model=request.model, origin=origin
    )

    # Handle streaming vs non-streaming
    if request.stream:
        # Check queue capacity before committing to a 200 streaming response.
        # If we let QueueFullError propagate from inside the generator, the HTTP
        # status is already 200 and the client sees a malformed stream instead of
        # a proper error response.
        from millm.services.request_queue import QueueFullError

        # Same pre-commit rule for a bad profile name: apply-time validation
        # runs inside the generator (headers already sent), so check the 404
        # case here while we can still return a proper error response. When
        # a dial is present the echo resolution above already verified it.
        if request.profile and request.steering_intensity is None:
            await inference.ensure_profile_exists(request.profile)
        # Feature 28 (FTASKS 6.4): the in-slot inline checks as a dry run — an unattached
        # `sae_id`, an ambiguous selection or an out-of-range index answers a proper 400 here,
        # not a 200 stream that ends in an error chunk.
        inference.check_inline_steering(request)

        # And a prompt past the model's context: a 400 with the error envelope,
        # not a 200 whose stream is cut off (hardware acceptance, 2026-09-14).
        inference.check_stream_admission(request)

        queue = inference.request_queue
        if queue.pending_count >= queue.max_pending:
            raise QueueFullError(
                f"Request queue is full ({queue.pending_count}/{queue.max_pending} pending). "
                "Please retry shortly."
            )
        stream_headers = {"X-miLLM-Backend": backend}
        if ignored_header:
            # Detection finished before the first byte (FR-25.1.7).
            stream_headers[IGNORED_FIELDS_HEADER] = ignored_header
        if request.seed is not None:
            # Streaming headers precede the generator body, so the scope is decided here, from
            # the same rule the service applies (a stream is one conversation, never batched).
            stream_headers["X-miLLM-Seed"] = seed_header(
                request.seed, inference.seed_scope_for(request)
            )
        if echo_intensity is not None:
            stream_headers["X-miLLM-Steering-Intensity"] = echo_intensity
        if echo_circuit_rung is not None:
            stream_headers["X-miLLM-Circuit-Rung"] = echo_circuit_rung
        stream_headers.update(probe_session_headers(probe_meta))
        return StreamingResponse(
            inference.stream_chat_completion(request),
            media_type="text/event-stream",
            headers=stream_headers,
        )
    else:
        # FastAPI's injected Response lets us set custom headers on the auto-serialised
        # Pydantic response without wrapping it manually. Every `X-miLLM-*` value comes from
        # `millm.api.provenance` — the same function a batch output line reads (FR-26.10.1) —
        # and is decided AFTER generation: the dial applies inside generation and can fail, the
        # seed scope is what the path that ran recorded, and the verdicts do not exist before.
        #
        # ⚠ The STREAMING branch above cannot do this — it commits headers before the first
        # byte, which is why streaming carries the verdict in a terminal chunk instead.
        result = await inference.create_chat_completion(request)
        response.headers.update(
            post_generation(
                request, inference, endpoint="chat", pre=pre, ignored_header=ignored_header
            )
        )
        response.headers.update(probe_session_headers(probe_meta))
        finish_body(result, model, inference)
        return result
