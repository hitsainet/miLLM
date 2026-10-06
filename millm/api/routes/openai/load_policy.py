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
    policy: LoadPolicy, model_row: Any, inference: Any, service: Any
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
