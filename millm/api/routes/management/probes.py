"""`/api/probes` — import, arm, inspect (FR-24.1, 24.2, 24.8, 24.9).

Shaped like `routes/management/sensing.py`: an `APIRouter` with a prefix, the `ApiResponse`
envelope, and `Annotated` query parameters.

⚠ **THE EVENT DETAIL ROUTE IS THE ONLY PLACE CONTEXT TEXT IS SERVED.** The list route and the
socket both omit `context_text` / `context_token_ids` — they are the decoded window around a
firing position, which is user content. A reviewer asks for one event and gets its context; a
dashboard subscribing to everything does not get a feed of prompts.
"""

from __future__ import annotations

import json
from typing import Annotated, Any, Literal, Optional

from fastapi import APIRouter, Body, Path, Query
from pydantic import BaseModel, Field

from millm.api.dependencies import (
    DbSession,
    ProbeArmingDep,
    ProbeEventRepo,
    ProbeEventServiceDep,
    ProbeHubServiceDep,
    ProbeRepo,
    ProbeServiceDep,
)
from millm.api.schemas.common import ApiResponse
from millm.services.probe_event_service import event_summary
from millm.core.errors import ProbeNotFoundError
from millm.core.probe_evidence import probe_rung_language, probe_rung_next_step
from millm.core.config import settings
from millm.services.probe_arm_bridge import (
    build_parity_forward,
    build_probe_encoder,
    loaded_identity,
)
from millm.services.probe_arming import armed_probe_from_row

router = APIRouter(prefix="/api/probes", tags=["probes"])


class ProbeHubImportRequest(BaseModel):
    """One definition from a Hub repo. Shaped like `HubImportRequest` for clusters."""

    repo_id: str = Field(..., min_length=3, max_length=200)
    filename: str = Field(..., min_length=1, max_length=300)
    revision: str | None = None
    on_conflict: Literal["rename", "fail"] = "rename"


class ProbeArmRequest(BaseModel):
    """⚠ `acknowledge_below_rung2` is the OPERATOR's acknowledgement, stored separately from the
    one inside the definition. The person who exported a weak probe and the person arming it on
    live traffic are not necessarily the same, and only the second is choosing to monitor with it.
    """

    acknowledge_below_rung2: bool = False
    reason: str = Field("", max_length=500)


def _probe_summary(probe: Any) -> dict[str, Any]:
    return {
        "id": probe.id,
        "name": probe.name,
        "hf_id": probe.hf_id,
        "layer": probe.layer,
        "rule": probe.rule,
        "scope": probe.scope,
        "basis": probe.basis,
        "streamable": probe.streamable,
        "threshold": probe.threshold,
        "target_fpr": probe.target_fpr,
        "rung": probe.rung,
        # ⚠ Rendered server-side from the shared vocabulary, never derived by a client from the
        # number. miLLM and miStudio must describe the same probe with the same words.
        "rung_language": probe_rung_language(probe.rung),
        "next_step": probe_rung_next_step(probe.rung),
        "armed": probe.armed,
        "paused_reason": probe.paused_reason,
        "parity": probe.parity,
        "created_at": probe.created_at,
    }


#: ⚠ Deliberately the SERVICE's serialiser, not a copy. The socket and this route feed the same
#: UI list, and when each built its own dict the socket's lacked `id` and `created_at`.
_event_summary = event_summary


# ── import ────────────────────────────────────────────────────────────────────


@router.post("/import", response_model=ApiResponse)
async def import_probe(
    payload: Annotated[dict[str, Any], Body(...)],
    service: ProbeServiceDep,
    on_conflict: Annotated[str, Query(pattern="^(rename|fail)$")] = "rename",
) -> ApiResponse:
    """Import a definition from a request body.

    `on_conflict` is `rename|fail`, matching circuits and clusters. There is deliberately no
    `replace`: overwriting a definition in place while its probe is ARMED would change the
    detector underneath a running monitor while every event kept the same `probe_id`.
    """
    raw_bytes = len(json.dumps(payload).encode("utf-8"))
    probe = await service.import_definition(
        payload, raw_bytes=raw_bytes, on_conflict=on_conflict, origin="file"
    )
    return ApiResponse.ok(_probe_summary(probe))


# ── hub ───────────────────────────────────────────────────────────────────────
#
# Anonymous, read-only. The TTL cache and the circuit breaker are inherited from
# `ClusterHubService`, so a flapping Hub becomes `HUB_UNAVAILABLE` rather than a stalled request.


@router.get("/hub/search", response_model=ApiResponse)
async def hub_search(
    hub: ProbeHubServiceDep,
    q: Annotated[Optional[str], Query(description="Free-text query")] = None,
    base_model: Annotated[Optional[str], Query(description="Filter by base model id")] = None,
    limit: Annotated[int, Query(ge=1, le=50)] = 30,
) -> ApiResponse:
    """Repos tagged `mistudio-probe-definition`."""
    return ApiResponse.ok(await hub.search(query=q, base_model=base_model, limit=limit))


@router.get("/hub/{repo_id:path}/definitions", response_model=ApiResponse)
async def hub_definitions(
    hub: ProbeHubServiceDep,
    repo_id: Annotated[str, Path(description="Hub repo id (org/name)")],
    revision: Annotated[Optional[str], Query()] = None,
) -> ApiResponse:
    """`manifest.jsonl` preferred, falling back to loose `*.probe.json` files."""
    return ApiResponse.ok(await hub.list_definitions(repo_id, revision=revision))


@router.post("/hub/import", response_model=ApiResponse)
async def hub_import(
    request: ProbeHubImportRequest,
    hub: ProbeHubServiceDep,
    service: ProbeServiceDep,
) -> ApiResponse:
    """Fetch one definition and import it. `origin` records that it came from the Hub."""
    _definition, raw_payload, _hub_ref = await hub.fetch_definition(
        request.repo_id, request.filename, revision=request.revision
    )
    probe = await service.import_definition(
        raw_payload,
        raw_bytes=len(json.dumps(raw_payload).encode("utf-8")),
        on_conflict=request.on_conflict,
        origin="hub",
    )
    return ApiResponse.ok(_probe_summary(probe))


# ── list / get / delete ───────────────────────────────────────────────────────


@router.get("", response_model=ApiResponse)
async def list_probes(
    repository: ProbeRepo,
    armed: Annotated[Optional[bool], Query()] = None,
) -> ApiResponse:
    probes = await repository.list(armed=armed)
    return ApiResponse.ok([_probe_summary(p) for p in probes])


@router.get("/status", response_model=ApiResponse)
async def probe_status(service: ProbeEventServiceDep) -> ApiResponse:
    """What every armed probe is doing, and why any of them is not scoring."""
    return ApiResponse.ok(await service.status())


@router.get("/events", response_model=ApiResponse)
async def list_events(
    events: ProbeEventRepo,
    probe_id: Annotated[Optional[str], Query()] = None,
    request_id: Annotated[Optional[str], Query()] = None,
    limit: Annotated[int, Query(ge=1, le=500)] = 100,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> ApiResponse:
    """Recent verdicts. **Without context text** — fetch one event for that."""
    rows = await events.list_events(
        probe_id=probe_id, request_id=request_id, limit=limit, offset=offset
    )
    return ApiResponse.ok([_event_summary(e) for e in rows])


@router.get("/events/{event_id}", response_model=ApiResponse)
async def get_event(event_id: int, events: ProbeEventRepo) -> ApiResponse:
    """One event, WITH its decoded context window.

    This is the only route that serves it, and it is a deliberate, per-event request rather than
    anything a dashboard receives by subscribing.
    """
    event = await events.get(event_id)
    if event is None:
        raise ProbeNotFoundError(f"No probe event {event_id}")
    detail = _event_summary(event)
    detail["context_text"] = event.context_text
    detail["context_token_ids"] = event.context_token_ids
    return ApiResponse.ok(detail)


@router.delete("/events", response_model=ApiResponse)
async def clear_events(
    events: ProbeEventRepo, probe_id: Annotated[Optional[str], Query()] = None
) -> ApiResponse:
    removed = await events.clear(probe_id)
    return ApiResponse.ok({"removed": removed})


@router.get("/{probe_id}", response_model=ApiResponse)
async def get_probe(probe_id: str, repository: ProbeRepo) -> ApiResponse:
    probe = await repository.get(probe_id)
    if probe is None:
        raise ProbeNotFoundError(f"No probe {probe_id}")
    detail = _probe_summary(probe)
    detail["definition"] = probe.definition
    return ApiResponse.ok(detail)


@router.delete("/{probe_id}", response_model=ApiResponse)
async def delete_probe(probe_id: str, service: ProbeServiceDep) -> ApiResponse:
    """Delete a probe and its events. Refused while armed."""
    deleted = await service.delete(probe_id)
    if not deleted:
        raise ProbeNotFoundError(f"No probe {probe_id}")
    return ApiResponse.ok({"deleted": probe_id})


# ── arm / disarm ──────────────────────────────────────────────────────────────


@router.post("/{probe_id}/arm", response_model=ApiResponse)
async def arm_probe(
    probe_id: str,
    request: ProbeArmRequest,
    session: DbSession,
    repository: ProbeRepo,
    arming: ProbeArmingDep,
) -> ApiResponse:
    """Run the four gates and arm the probe if they all pass.

    ⚠ **THE ONLY CALLER OF `ProbeArmingService.arm`.** Every gate — the limit, identity, the
    evidence rung, parity — is reachable only through this route, and `test_probe_reachable.py`
    goes red if its registration is removed.

    A refusal is an error in the envelope naming which gate refused and why; the gates run in the
    order they are cheapest to fail, so a wrong-model probe never costs a forward pass.
    """
    probe = await repository.get(probe_id)
    if probe is None:
        raise ProbeNotFoundError(f"No probe {probe_id}")

    identity, model, tokenizer = await loaded_identity(session)
    armed = await arming.arm(
        probe,
        model=model,
        loaded=identity,
        forward=build_parity_forward(model, probe.layer),
        tokenizer=tokenizer,
        acknowledge_below_rung2=request.acknowledge_below_rung2,
        reason=request.reason,
        encoder=await build_probe_encoder(probe),
    )
    # ⚠ `armed` is an `ArmedProbe`, which has NO `basis` — the basis lives on the row. This read
    # `armed.basis` and raised AttributeError AFTER the probe was already armed and hooked, so
    # the operator saw INTERNAL_ERROR for an operation that had fully succeeded. A response
    # serialiser that can fail after the side effect is worse than one that fails before it.
    detail = _probe_summary(probe)
    detail["armed"] = True
    detail["parity"] = probe.parity
    return ApiResponse.ok(detail)


@router.post("/{probe_id}/parity", response_model=ApiResponse)
async def check_parity(
    probe_id: str,
    session: DbSession,
    repository: ProbeRepo,
) -> ApiResponse:
    """Re-run parity without arming, and store the report.

    Useful after a model reload: the same definition against a differently-loaded model is the
    case parity exists to catch, and an operator should be able to ask before arming.
    """
    from millm.services.probe_parity import ProbeParityEngine

    probe = await repository.get(probe_id)
    if probe is None:
        raise ProbeNotFoundError(f"No probe {probe_id}")

    _identity, model, tokenizer = await loaded_identity(session)
    armed = armed_probe_from_row(probe, encoder=await build_probe_encoder(probe))
    tolerance = max(
        float((probe.definition.get("test_vectors") or {}).get("tolerance", 0.0) or 0.0),
        settings.PROBE_PARITY_TOLERANCE,
    )
    report = ProbeParityEngine(build_parity_forward(model, probe.layer)).run(
        armed, probe.definition, tolerance=tolerance, tokenizer=tokenizer
    )
    await repository.update(probe, parity=report.as_details())
    return ApiResponse.ok(report.as_details())


@router.post("/{probe_id}/disarm", response_model=ApiResponse)
async def disarm_probe(
    probe_id: str, repository: ProbeRepo, arming: ProbeArmingDep
) -> ApiResponse:
    probe = await repository.get(probe_id)
    if probe is None:
        raise ProbeNotFoundError(f"No probe {probe_id}")
    await arming.disarm(probe, reason="operator")
    return ApiResponse.ok(_probe_summary(probe))
