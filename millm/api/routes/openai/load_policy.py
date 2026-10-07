"""
`X-miLLM-Load-Policy`: a request may promise never to cause a model swap (Feature 29, FR-29.4).

`auto` (the default, and what Open WebUI relies on) keeps today's behaviour: a `/v1` request
naming a model that is not resident loads it. `refuse` turns that load into a refusal:

* the model is not resident → `409 model_not_resident`, naming the requested and the
  resident model, plus the resident model's lease when one is held (FR-29.4.2, FR-29.4.4);
* the model is being loaded right now → `503 model_loading` with `Retry-After`: it will
  become resident without this request causing anything (FR-29.4.3).

Under `refuse` the lease is not consulted — no load would be attempted. `apply_load_policy` is
THE refuse-policy function, called by the three `/v1` routes after their existing pre-load
refusals (not found, embedding-only, the request policy's GGUF cells) and immediately before
the auto-load; an AST test asserts each route calls it before `load_model_and_wait`.
"""

from __future__ import annotations

from typing import Any, Literal

from fastapi.responses import JSONResponse

from millm.core.errors import InvalidParameterError
from millm.core.logging import get_logger
from millm.db.models.model import ModelStatus

logger = get_logger(__name__)

LOAD_POLICY_HEADER = "X-miLLM-Load-Policy"
LEASE_HEADER = "X-miLLM-Lease"

LoadPolicy = Literal["auto", "refuse"]


def parse_load_policy(raw: str | None) -> LoadPolicy:
    """`auto` or `refuse`, case-insensitive; absent is `auto`; anything else is 400."""
    if raw is None:
        return "auto"
    value = raw.strip().lower()
    if value == "auto":
        return "auto"
    if value == "refuse":
        return "refuse"
    raise InvalidParameterError(
        f"{LOAD_POLICY_HEADER} must be 'auto' or 'refuse', got {raw!r}.",
        details={"param": LOAD_POLICY_HEADER},
    )


async def apply_load_policy(
    policy: LoadPolicy,
    model_row: Any,
    inference: Any,  # noqa: ARG001 - FTID §3.4 signature; residency is read from the loader
    service: Any,
) -> JSONResponse | None:
    """None when the request may continue; otherwise the refusal to return.

    Continues under `auto`, and under `refuse` when the named model is already resident.
    """
    if policy == "auto":
        return None
    from millm.api.routes.openai.errors import create_openai_error, model_not_resident_error
    from millm.api.schemas.lease import lease_summary
    from millm.core.backpressure import retry_after_for
    from millm.services.model_lease import get_lease_registry

    resident_id = service.loader.loaded_model_id
    if resident_id == model_row.id:
        return None

    if model_row.status == ModelStatus.LOADING or service._loading_model_id == model_row.id:
        logger.info("load_policy_refused", model=model_row.name, outcome="model_loading")
        return create_openai_error(
            message=(
                f"The model '{model_row.name}' is being loaded now. Retry once it is resident; "
                "this request (X-miLLM-Load-Policy: refuse) started nothing."
            ),
            error_type="server_error",
            code="model_loading",
            param="model",
            status_code=503,
            retry_after=retry_after_for("MODEL_LOADING", {"loading_model_id": model_row.id}),
        )

    resident_name = service.loader.model_name if resident_id is not None else None
    record = get_lease_registry().current(resident_id)
    lease = lease_summary(record).model_dump(mode="json") if record is not None else None
    logger.info(
        "load_policy_refused",
        model=model_row.name,
        resident_model=resident_name,
        outcome="model_not_resident",
        leased=lease is not None,
    )
    return model_not_resident_error(model_row.name, resident_name, lease)


def refuse_inline_steering_before_load(request: Any, inference: Any) -> None:
    """A non-empty `steering` set naming a model that is not resident: refused with
    `SAE_NOT_ATTACHED` BEFORE any auto-load (Feature 28, FR-28.1.11, T-82).

    Decidable from the request: SAEs attach only to the resident model, and no load path
    re-attaches one (`SAERepository.get_active_attachment` has no caller), so after the load the
    in-slot check would refuse anyway — having evicted the resident model and every SAE attached
    to it to get there. The attach-time model lock would usually refuse the swap first, but that
    lock is best-effort, so this does not rely on it. `steering: {"features": []}` is not
    refused: there is nothing to attach.
    """
    from millm.core.errors import SAENotAttachedError

    steering = getattr(request, "steering", None)
    if steering is None or not steering.features:
        return
    info = inference.get_loaded_model_info()
    resident = info.name if info else None
    if resident == request.model:
        return
    sae_id = steering.sae_id
    raise SAENotAttachedError(
        f"inline steering names model '{request.model}', which is not resident "
        f"({'resident: ' + repr(resident) if resident else 'no model is loaded'}); an SAE "
        + (f"('{sae_id}') " if sae_id else "")
        + "attaches only to the resident model, so nothing could steer this request. "
        "Load the model and attach the SAE first.",
        details={"param": "steering", "sae_id": sae_id, "model": request.model,
                 "resident_model": resident},
    )
