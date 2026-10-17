"""Request policy for the OpenAI-compatible `/v1` surface (Feature 25, FR-25.1 – FR-25.3).

THE GUARANTEE: no request field is dropped without a trace, and a field that changes the
output is never ignored — on any endpoint, on either engine, strict mode or not.

THE DEFECT IT PREVENTS: every `/v1` request schema was `extra="ignore"`. A client sending
`response_format`, `seed` or chat `logprobs` got a 200 that silently ignored them, and nothing
in the response said so — a labelling job would record "structured output, seed 7" against rows
produced with neither (BRD-04 §1).

How:

* The request schemas are `extra="allow"`, so pydantic's own parse leaves every undeclared key
  in `model_extra` (top level and per message). Nothing here re-parses the body, so this cannot
  drift from validation.
* `OUTPUT_CHANGING` is THE list of output-changing fields, with one outcome per
  (endpoint, engine) cell. **Every cell is filled; there is no default outcome**, so a new
  endpoint or engine cannot inherit "honoured" by accident (a test enforces completeness).
  Features 26–30 add rows here; they never add a second table.
* A field present with a value that is not its neutral value (FR-25.3.4) is refused with a
  400 naming it when its cell is *refused* — always, not only under strict mode.
* Everything else the path will not use is REPORTED: `X-miLLM-Ignored-Fields` plus one
  `request_fields_unused` warning carrying locations, never values (FR-25.1.8). Under
  `X-miLLM-Strict: true` it is refused instead, naming every location (FR-25.2).
* Everything is decided from the request and the model ROW, so the routes run it before the
  auto-load that would evict the resident model and its SAEs (FR-25.2.3, FR-25.3.8).

Feature 26 note (FR-25.1.1; 026 FR-26.1.6, FR-26.10.1): batch lines construct request objects
per line and MUST call `evaluate(..., strict=True)`; the per-line extension values reuse
`encode_field_list`. Nothing here imports from `services` or `ml`.
"""

from __future__ import annotations

import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from millm.core.errors import (
    FieldNotHonouredError,
    InvalidParameterError,
    UnusedFieldsRefusedError,
)
from millm.core.logging import get_logger

logger = get_logger(__name__)

#: The request header that turns reporting into refusal (FR-25.2; miStudio 034 TD5 sends it).
STRICT_HEADER = "X-miLLM-Strict"
#: The response header that names unused fields (FR-25.1).
IGNORED_FIELDS_HEADER = "X-miLLM-Ignored-Fields"


class Endpoint(StrEnum):
    CHAT = "chat"
    COMPLETIONS = "completions"
    EMBEDDINGS = "embeddings"


class Engine(StrEnum):
    TRANSFORMERS = "transformers"
    LLAMACPP = "llamacpp"


#: Every `/v1` path that takes a body, and the table column it reads. The coverage test
#: enumerates POST paths from `app.openapi()["paths"]` and fails on a path missing here.
ENDPOINT_PATHS: dict[str, Endpoint] = {
    "/v1/chat/completions": Endpoint.CHAT,
    "/v1/completions": Endpoint.COMPLETIONS,
    "/v1/embeddings": Endpoint.EMBEDDINGS,
}


#: `/v1` POST paths that carry NO generation body (Feature 26): the batch and file control plane.
#: They have no column in the output-changing table because nothing they accept changes a model's
#: output; `POST /v1/batches` applies the unknown-field rule through `evaluate_control` instead
#: (FR-26.1.6). The coverage test requires every live `/v1` POST path to be in exactly one of
#: `ENDPOINT_PATHS` and this set, so a new GENERATION path still cannot slip in uncovered.
CONTROL_PATHS: frozenset[str] = frozenset({
    "/v1/files",
    "/v1/batches",
    "/v1/batches/{batch_id}/cancel",
    "/v1/batches/{batch_id}/lease",
})


@dataclass(frozen=True)
class Honoured:
    """The field is served. `refuse_if`, when set, names a request shape in which this
    endpoint still cannot honour it (e.g. a steering profile on a scoring request, which is
    always unsteered — X-09) and returns the reason, or None."""

    refuse_if: Callable[[Any], str | None] | None = None


@dataclass(frozen=True)
class Refused:
    reason: str


Outcome = Honoured | Refused
HONOURED = Honoured()


def refused(reason: str) -> Refused:
    return Refused(reason)


def _wants_scores(request: Any) -> bool:
    fn = getattr(request, "wants_scores", None)
    return bool(fn()) if callable(fn) else False


def _unless_scoring(reason: str) -> Honoured:
    return Honoured(refuse_if=lambda request: reason if _wants_scores(request) else None)


CHAT, COMPLETIONS, EMBEDDINGS = Endpoint.CHAT, Endpoint.COMPLETIONS, Endpoint.EMBEDDINGS
TF, LC = Engine.TRANSFORMERS, Engine.LLAMACPP


def _cells(
    chat_tf: Outcome, chat_lc: Outcome, comp_tf: Outcome, comp_lc: Outcome,
    emb_tf: Outcome, emb_lc: Outcome,
) -> dict[tuple[Endpoint, Engine], Outcome]:
    return {
        (CHAT, TF): chat_tf, (CHAT, LC): chat_lc,
        (COMPLETIONS, TF): comp_tf, (COMPLETIONS, LC): comp_lc,
        (EMBEDDINGS, TF): emb_tf, (EMBEDDINGS, LC): emb_lc,
    }


_NO_DISTRIBUTION = refused(
    "scoring needs the transformers engine; a GGUF model served by llama.cpp exposes no "
    "per-token distribution"
)
_EMB_NO_TOKENS = refused("embeddings return vectors, not token probabilities")
_NO_FUNCTIONS = refused("function calling is not implemented")
#: A cell whose phase of Feature 25 has not landed yet: refused, never silently dropped.
_PENDING = refused("not implemented on this endpoint yet")
_STEER_LC = refused("steering needs forward hooks on a PyTorch module tree; llama.cpp has none")
_NEVER_STEERED = refused("embeddings are never steered")
_ACTIVATIONS_LC = refused(
    "per-request SAE activations need forward hooks on a PyTorch module tree; llama.cpp has none"
)
#: Feature 30, T-91. The route prefixes the model's name, so the message names the model.
DIMENSIONS_REASON = (
    "this model does not declare truncated-embedding support, and no model can: "
    "miLLM records no such declaration (T-91), so vectors are always full width"
)
_DIMENSIONS_UNDECLARED = refused(DIMENSIONS_REASON)
_GGUF_POOLING = refused(
    "llama.cpp fixes pooling when the model is loaded, and miLLM loads GGUF models with mean "
    "pooling; only `pooling: \"mean\"` is served on a GGUF model"
)


def _EMBEDDINGS_ONLY(name: str) -> Refused:  # noqa: N802 - reads as a table constant
    return refused(f"{name} applies to /v1/embeddings")


_SCORING_UNSTEERED = "scoring is always unsteered (X-09), so a steering field cannot be honoured"

#: THE output-changing list (FR-25.3.3). Field -> (endpoint, engine) -> outcome.
OUTPUT_CHANGING: dict[str, dict[tuple[Endpoint, Engine], Outcome]] = {
    "logprobs": _cells(
        HONOURED, _NO_DISTRIBUTION, HONOURED, _NO_DISTRIBUTION, _EMB_NO_TOKENS, _EMB_NO_TOKENS,
    ),
    "top_logprobs": _cells(
        HONOURED, _NO_DISTRIBUTION,
        refused("/v1/completions takes the number of alternatives in `logprobs`"),
        refused("/v1/completions takes the number of alternatives in `logprobs`"),
        _EMB_NO_TOKENS, _EMB_NO_TOKENS,
    ),
    "allowed_token_ids": _cells(
        HONOURED, _NO_DISTRIBUTION, HONOURED, _NO_DISTRIBUTION, _EMB_NO_TOKENS, _EMB_NO_TOKENS,
    ),
    "response_format": _cells(
        _unless_scoring("scoring returns one token's log-probabilities, not a document"),
        refused("structured output on a GGUF model is refused in v1"),
        refused("structured output is served on /v1/chat/completions only"),
        refused("structured output is served on /v1/chat/completions only"),
        refused("embeddings return vectors, not text"),
        refused("embeddings return vectors, not text"),
    ),
    "seed": _cells(
        # llama.cpp honours seed since T-61 passed on hardware (2026-10-06; 025_FTASKS 6.6).
        HONOURED, HONOURED, HONOURED, HONOURED,
        refused("embeddings are deterministic; there is no sampling for a seed to fix"),
        refused("embeddings are deterministic; there is no sampling for a seed to fix"),
    ),
    "n": _cells(
        HONOURED,
        refused("the llama.cpp engine returns one completion per request"),
        refused("n > 1 on /v1/completions is not implemented in v1 (T-56)"),
        refused("n > 1 on /v1/completions is not implemented in v1 (T-56)"),
        refused("/v1/embeddings returns one embedding per input"),
        refused("/v1/embeddings returns one embedding per input"),
    ),
    "dimensions": _cells(
        refused("dimensions applies to /v1/embeddings"),
        refused("dimensions applies to /v1/embeddings"),
        refused("dimensions applies to /v1/embeddings"),
        refused("dimensions applies to /v1/embeddings"),
        # Feature 30 (FR-30.1.1, FR-30.1.6): `dimensions` is honoured only for a model that
        # DECLARES truncated-embedding support, and no model can (T-91: no declaration exists).
        # So it is refused on both engines, at every value — the native width included, since
        # `dimensions` has no neutral value — and never accepted and ignored.
        _DIMENSIONS_UNDECLARED, _DIMENSIONS_UNDECLARED,
    ),
    # Feature 30 (FR-30.2.7, FR-30.2.11): llama.cpp fixes pooling at construction and miLLM
    # builds GGUF models with MEAN, so `last` and `cls` are refused there before any load
    # (`mean` is neutral). `normalize` is post-processing in miLLM, honoured on both engines.
    "pooling": _cells(
        _EMBEDDINGS_ONLY("pooling"), _EMBEDDINGS_ONLY("pooling"),
        _EMBEDDINGS_ONLY("pooling"), _EMBEDDINGS_ONLY("pooling"),
        HONOURED, _GGUF_POOLING,
    ),
    "normalize": _cells(
        _EMBEDDINGS_ONLY("normalize"), _EMBEDDINGS_ONLY("normalize"),
        _EMBEDDINGS_ONLY("normalize"), _EMBEDDINGS_ONLY("normalize"),
        HONOURED, HONOURED,
    ),
    # Feature 28 (FR-28.4.5, FR-25.3.7): inline steering is honoured on transformers chat and
    # completions; scoring is always unsteered (X-09, FR-25.7.2); llama.cpp has no hooks and is
    # refused here, before any auto-load (FR-28.1.9); embeddings are never steered.
    "steering": _cells(
        _unless_scoring(_SCORING_UNSTEERED), _STEER_LC,
        _unless_scoring(_SCORING_UNSTEERED), _STEER_LC,
        _NEVER_STEERED, _NEVER_STEERED,
    ),
    "tools": _cells(*([_NO_FUNCTIONS] * 6)),
    "tool_choice": _cells(*([_NO_FUNCTIONS] * 6)),
    "logit_bias": _cells(*([refused("logit_bias is not implemented in v1")] * 6)),
    "max_completion_tokens": _cells(
        HONOURED, HONOURED, HONOURED, HONOURED,
        refused("embeddings generate no tokens"),
        refused("embeddings generate no tokens"),
    ),
    # Feature 28 (FR-28.4.1): /v1/completions gains `profile` and the dial too.
    "profile": _cells(
        _unless_scoring(_SCORING_UNSTEERED), _STEER_LC,
        _unless_scoring(_SCORING_UNSTEERED), _STEER_LC, _NEVER_STEERED, _NEVER_STEERED,
    ),
    "steering_intensity": _cells(
        _unless_scoring(_SCORING_UNSTEERED), _STEER_LC,
        _unless_scoring(_SCORING_UNSTEERED), _STEER_LC, _NEVER_STEERED, _NEVER_STEERED,
    ),
    # Feature 27 (FR-27.1, FR-27.2g): refused on a GGUF row BEFORE any auto-load — llama.cpp
    # exposes no layer to read — and on embeddings, which generate no positions to report.
    "return_sae_activations": _cells(
        HONOURED, _ACTIVATIONS_LC, HONOURED, _ACTIVATIONS_LC,
        refused("per-request SAE activations apply to chat and text completions"),
        refused("per-request SAE activations apply to chat and text completions"),
    ),
}


def _format_type(value: Any) -> Any:
    if isinstance(value, Mapping):
        return value.get("type")
    return getattr(value, "type", None)


#: Values that are explicitly "no change" (FR-25.3.4). A refused field sent with its neutral
#: value is honoured as no change; every other value of a refused field is refused, including
#: its zero value. An explicit JSON null is "not sent" for every field.
NEUTRAL: dict[str, Callable[[Any], bool]] = {
    "n": lambda v: v == 1 and not isinstance(v, bool),
    "logprobs": lambda v: v is False,
    "response_format": lambda v: _format_type(v) == "text",
    "tools": lambda v: v == [],
    "logit_bias": lambda v: v == {},
    # Feature 30: the defaults, which change nothing (FR-30.2.2).
    "pooling": lambda v: v == "mean",
    "normalize": lambda v: v is False,
}

#: Declared fields an engine path does not consume (FR-25.1.2 case b). On llama.cpp the template
#: is baked into the GGUF file and takes no variables, so `chat_template_kwargs` is unused there
#: (it was logged at info only, `_refuse_unsupported_llamacpp_request`).
ENGINE_UNUSED: dict[tuple[Endpoint, Engine], frozenset[str]] = {
    (CHAT, LC): frozenset({"chat_template_kwargs"}),
}


@dataclass
class PolicyResult:
    """What the route reports: unused field locations, in request order, no duplicates."""

    unused: list[str] = field(default_factory=list)
    endpoint: Endpoint | None = None
    engine: Engine | None = None


def parse_strict(value: str | None) -> bool:
    """`X-miLLM-Strict`: `true`/`1` on; `false`/`0`/absent off; anything else refused.

    A client that wrote `yes` meant strict, and serving it lenient would be the silent drop this
    feature removes (FR-25.2.2).
    """
    if value is None:
        return False
    text = value.strip().lower()
    if text in ("true", "1"):
        return True
    if text in ("false", "0"):
        return False
    raise InvalidParameterError(
        f"{STRICT_HEADER} must be true, 1, false or 0 (case-insensitive); got {value!r}",
        details={"param": STRICT_HEADER},
    )


def engine_of(row: Any) -> Engine:
    """The engine a model row is served by. `gguf_files` is set at download time, so this is
    knowable with nothing resident (Feature 23)."""
    return Engine.LLAMACPP if getattr(row, "gguf_files", None) else Engine.TRANSFORMERS


def _present(request: Any, name: str) -> tuple[bool, Any, bool]:
    """(sent, value, declared). "Sent" means in the body: a defaulted field is not a presence."""
    if name in type(request).model_fields:
        if name in request.model_fields_set:
            return True, getattr(request, name), True
        return False, None, True
    extra = request.model_extra or {}
    if name in extra:
        return True, extra[name], False
    return False, None, False


def _message_locations(request: Any) -> list[str]:
    locations: list[str] = []
    for i, message in enumerate(getattr(request, "messages", None) or []):
        for key in (getattr(message, "model_extra", None) or {}):
            locations.append(f"messages[{i}].{key}")
    for j, conversation in enumerate(getattr(request, "extra_messages", None) or []):
        for i, message in enumerate(conversation or []):
            for key in (getattr(message, "model_extra", None) or {}):
                locations.append(f"extra_messages[{j}][{i}].{key}")
    return locations


def evaluate(
    request: Any,
    endpoint: Endpoint,
    engine: Engine,
    *,
    strict: bool,
    model_name: str | None = None,
) -> PolicyResult:
    """Apply the output-changing table, then find the unused fields.

    Raises:
        FieldNotHonouredError: a listed field whose outcome here is *refused* (always).
        UnusedFieldsRefusedError: strict mode and at least one unused field.
    """
    refusals: list[tuple[str, str]] = []
    handled: set[str] = set()
    for name, cells in OUTPUT_CHANGING.items():
        sent, value, declared = _present(request, name)
        if not sent:
            continue
        handled.add(name)
        if value is None:
            continue  # an explicit null is "not sent"
        outcome = cells[(endpoint, engine)]  # KeyError = an unfilled cell: a defect, not a default
        neutral = NEUTRAL.get(name, lambda _v: False)(value)
        if isinstance(outcome, Refused):
            if not neutral:
                refusals.append((name, outcome.reason))
            continue
        # HONOURED. Honouring needs a schema field to read it: a listed field that arrives as an
        # undeclared extra would be dropped while the table says "honoured" — fail closed.
        if not declared and not neutral:
            refusals.append((name, f"`{name}` is not implemented on this endpoint"))
            continue
        if outcome.refuse_if is not None and not neutral:
            reason = outcome.refuse_if(request)
            if reason:
                refusals.append((name, reason))
    if refusals:
        name, reason = refusals[0]
        logger.info(
            "request_field_refused",
            field=name,
            fields=[n for n, _ in refusals],
            endpoint=endpoint.value,
            engine=engine.value,
            reason=reason,
        )
        others = (
            " (also refused: " + ", ".join(f"'{n}'" for n, _ in refusals[1:]) + ")"
            if refusals[1:] else ""
        )
        # The model is named when the caller knows it (Feature 30, FR-30.1.2: a `dimensions`
        # refusal names the model that declares nothing).
        on_model = f" for model '{model_name}'" if model_name else ""
        raise FieldNotHonouredError(
            f"'{name}' is not honoured on /v1/{_path_name(endpoint)}{on_model} with the "
            f"{engine.value} engine: {reason}{others}",
            details={
                "param": name,
                "fields": [n for n, _ in refusals],
                "endpoint": endpoint.value,
                "engine": engine.value,
            },
        )

    unused: list[str] = []
    for key in (request.model_extra or {}):
        if key not in handled:
            unused.append(key)
    for name in sorted(ENGINE_UNUSED.get((endpoint, engine), frozenset())):
        if name in request.model_fields_set and getattr(request, name, None) is not None:
            unused.append(name)
    unused.extend(_message_locations(request))
    unused = list(dict.fromkeys(unused))

    if unused and strict:
        raise UnusedFieldsRefusedError(
            f"{STRICT_HEADER} is set and this request carries fields the server would not use: "
            + ", ".join(unused),
            details={"param": unused[0], "fields": unused},
        )
    return PolicyResult(unused=unused, endpoint=endpoint, engine=engine)


def evaluate_control(request: Any, path: str, *, strict: bool) -> PolicyResult:
    """The unknown-field rule for a control-plane body (FR-26.1.6): every undeclared key is
    reported, or refused under `X-miLLM-Strict: true`. Nothing here changes an output, so there
    is no table to apply."""
    unused = list(dict.fromkeys(request.model_extra or {}))
    if unused and strict:
        raise UnusedFieldsRefusedError(
            f"{STRICT_HEADER} is set and this request carries fields the server would not use: "
            + ", ".join(unused),
            details={"param": unused[0], "fields": unused},
        )
    if unused:
        logger.warning("request_fields_unused", endpoint=path, fields=unused, count=len(unused))
    return PolicyResult(unused=unused)


def _path_name(endpoint: Endpoint) -> str:
    return {CHAT: "chat/completions", COMPLETIONS: "completions", EMBEDDINGS: "embeddings"}[endpoint]


def _sf_string(location: str) -> str:
    """One RFC 8941 sf-string. Field names are client-chosen JSON keys, so `\\` and `"` are
    escaped and every byte outside printable ASCII — CR and LF included, which would otherwise
    split the header — is percent-encoded, as is `%` itself so the encoding is unambiguous."""
    out: list[str] = []
    for byte in location.encode("utf-8"):
        ch = chr(byte)
        if byte == 0x25 or byte < 0x20 or byte > 0x7E:
            out.append(f"%{byte:02X}")
        elif ch in ('"', "\\"):
            out.append("\\" + ch)
        else:
            out.append(ch)
    return '"' + "".join(out) + '"'


def encode_field_list(locations: list[str], max_bytes: int) -> str:
    """The `X-miLLM-Ignored-Fields` value: an RFC 8941 list of sf-strings, at most `max_bytes`.

    When members are dropped to fit, the last member is `"+N more"` (FR-25.1.6). Returns "" for
    an empty list, and the caller then sets no header (FR-25.1.5).
    """
    members = [_sf_string(loc) for loc in locations]
    for keep in range(len(members), -1, -1):
        parts = members[:keep]
        if keep < len(members):
            parts = parts + [f'"+{len(members) - keep} more"']
        value = ", ".join(parts)
        if len(value) <= max_bytes or keep == 0:
            return value
    return ""  # pragma: no cover - the loop always returns


def log_unused(endpoint: Endpoint, request_id: str, locations: list[str]) -> None:
    """One structured warning per request: endpoint, request id, LOCATIONS. Never values —
    a field's value may be prompt text (FR-25.1.8)."""
    logger.warning(
        "request_fields_unused",
        endpoint=endpoint.value,
        request_id=request_id,
        fields=locations,
        count=len(locations),
    )


def apply_request_policy(
    request: Any, endpoint: Endpoint, row: Any, headers: Mapping[str, str]
) -> PolicyResult:
    """The route's single call: parse strict mode, read the engine from the row, evaluate, log.

    Called after the row lookup and before the auto-load, so every refusal here costs no load.
    """
    strict = parse_strict(headers.get(STRICT_HEADER))
    result = evaluate(
        request, endpoint, engine_of(row), strict=strict, model_name=model_name_of(row),
    )
    report_unused(result, endpoint, headers)
    return result


def model_name_of(row: Any) -> str | None:
    """The row's name for refusal messages (Feature 30), or None when it has none."""
    name = getattr(row, "name", None)
    return name if isinstance(name, str) else None


def report_unused(result: PolicyResult, endpoint: Endpoint, headers: Mapping[str, str]) -> None:
    """The one `request_fields_unused` log line for a request with unused fields."""
    if result.unused:
        request_id = headers.get("X-Request-ID") or uuid.uuid4().hex[:16]
        log_unused(endpoint, request_id, result.unused)


def ignored_fields_header(result: PolicyResult) -> str | None:
    """The header value for a policy result, or None when nothing was ignored."""
    if not result.unused:
        return None
    from millm.core.config import settings

    return encode_field_list(result.unused, int(settings.IGNORED_FIELDS_HEADER_MAX_BYTES))
