"""
Cluster management API endpoints (Feature 8).

Import `mistudio.cluster-definition/v1` documents (file/paste or Hugging Face
consume-only browse), list/activate/deactivate cluster-typed profiles, adjust
the lambda intensity dial, and re-export lossless definitions.
"""

import json
from typing import Annotated, Any

from fastapi import APIRouter, Body, Path, Query, Request
from pydantic import ValidationError as PydanticValidationError

from millm.api.dependencies import ClusterHubServiceDep, ClusterServiceDep
from millm.api.schemas.cluster import (
    BUNDLE_KIND,
    DEFINITION_KIND,
    MAX_IMPORT_BYTES,
    ClusterBundleV1,
    ClusterDefinitionV1,
    ClusterImportItem,
    ClusterImportResult,
    ClusterListResponse,
    HubDefinitionRef,
    HubImportRequest,
    HubRepoInfo,
    SetIntensityRequest,
)
from millm.api.schemas.common import ApiResponse

router = APIRouter(prefix="/api/clusters", tags=["clusters"])

ClusterId = Annotated[str, Path(description="Cluster profile ID")]


@router.get(
    "",
    response_model=ApiResponse[ClusterListResponse],
    summary="List imported clusters",
)
async def list_clusters(service: ClusterServiceDep) -> ApiResponse[ClusterListResponse]:
    """List cluster-typed profiles with bound state, warnings, and intensity."""
    clusters = await service.list_clusters()
    active = next((c.id for c in clusters if c.is_active), None)
    return ApiResponse.ok(
        ClusterListResponse(clusters=clusters, active_cluster_id=active)
    )


@router.post(
    "/import",
    response_model=ApiResponse[ClusterImportResult],
    summary="Import a cluster definition or bundle",
)
async def import_clusters(
    request: Request,
    service: ClusterServiceDep,
    payload: dict[str, Any] = Body(..., description="Definition or bundle JSON"),
    on_conflict: str = Query("rename", pattern="^(rename|fail)$"),
    activate: bool = Query(False, description="Activate after import (single definition only)"),
) -> ApiResponse[ClusterImportResult]:
    """
    Import a `mistudio.cluster-definition/v1` document or a
    `mistudio.cluster-bundle/v1` (per-item isolated). Kind-keyed; strict
    validation against the frozen v1 contract.
    """
    # Size gates — honest scope: FastAPI has ALREADY read+parsed the body by
    # the time any handler code runs, so neither check is pre-parse (a true
    # stream-level limit belongs in middleware/ingress). These bound the
    # DOWNSTREAM work (Pydantic validation, DB row size) and give hostile
    # payloads a structured refusal instead of a stored megarow.
    declared = request.headers.get("content-length")
    if declared and declared.isdigit() and int(declared) > MAX_IMPORT_BYTES:
        return ApiResponse.fail(
            code="PAYLOAD_TOO_LARGE",
            message=f"Import exceeds {MAX_IMPORT_BYTES} bytes",
        )
    if len(json.dumps(payload, separators=(",", ":"))) > MAX_IMPORT_BYTES:
        return ApiResponse.fail(
            code="PAYLOAD_TOO_LARGE",
            message=f"Import exceeds {MAX_IMPORT_BYTES} bytes",
        )

    kind = payload.get("kind")
    try:
        if kind == BUNDLE_KIND:
            bundle = ClusterBundleV1.model_validate(payload)
            result = await service.import_bundle(bundle, raw_payload=payload, on_conflict=on_conflict)
        elif kind == DEFINITION_KIND:
            definition = ClusterDefinitionV1.model_validate(payload)
            item = await service.import_definition(
                definition, raw_payload=payload,
                on_conflict=on_conflict, activate=activate,
            )
            result = ClusterImportResult(
                results=[item],
                imported=int(item.status in ("imported", "imported_unbound")),
                blocked=int(item.status == "blocked"),
                errors=int(item.status == "error"),
            )
        else:
            return ApiResponse.fail(
                code="UNKNOWN_KIND",
                message=(
                    f"kind {kind!r} is not a supported cluster document "
                    f"(expected {DEFINITION_KIND!r} or {BUNDLE_KIND!r})"
                ),
            )
    except PydanticValidationError as e:
        return ApiResponse.fail(
            code="VALIDATION_ERROR",
            message=f"Payload does not match the v1 contract: {e.error_count()} error(s)",
            details={"errors": json.loads(e.json())[:10]},
        )
    return ApiResponse.ok(result)


# ── Hugging Face consume-only ────────────────────────────────────────────────

@router.get(
    "/hub/search",
    response_model=ApiResponse[list[HubRepoInfo]],
    summary="Search public cluster packs on Hugging Face",
)
async def hub_search(
    hub: ClusterHubServiceDep,
    q: str | None = Query(None, description="Free-text query"),
    base_model: str | None = Query(None, description="Filter by base model id"),
    limit: int = Query(30, ge=1, le=50),
) -> ApiResponse[list[HubRepoInfo]]:
    """Anonymous, tag-filtered (`mistudio-cluster-definition`) repo search."""
    return ApiResponse.ok(await hub.search(query=q, base_model=base_model, limit=limit))


@router.get(
    "/hub/{repo_id:path}/definitions",
    response_model=ApiResponse[list[HubDefinitionRef]],
    summary="List a repo's cluster definitions",
)
async def hub_definitions(
    hub: ClusterHubServiceDep,
    repo_id: str = Path(..., description="Hub repo id (org/name)"),
    revision: str | None = Query(None),
) -> ApiResponse[list[HubDefinitionRef]]:
    """manifest.jsonl preferred; falls back to loose *.cluster.json files."""
    return ApiResponse.ok(await hub.list_definitions(repo_id, revision=revision))


@router.post(
    "/hub/import",
    response_model=ApiResponse[ClusterImportItem],
    summary="Import one definition from a Hub repo",
)
async def hub_import(
    request: HubImportRequest,
    hub: ClusterHubServiceDep,
    service: ClusterServiceDep,
) -> ApiResponse[ClusterImportItem]:
    """Fetch, validate, and import a single `.cluster.json` (anonymous)."""
    definition, raw_payload, hub_ref = await hub.fetch_definition(
        request.repo_id, request.filename, revision=request.revision
    )
    item = await service.import_definition(
        definition, raw_payload=raw_payload,
        hub_ref=hub_ref, activate=request.activate,
    )
    return ApiResponse.ok(item)


# ── Activation / intensity / export ─────────────────────────────────────────

@router.post(
    "/{cluster_id}/activate",
    response_model=ApiResponse[dict[str, Any]],
    summary="Activate a cluster (hard compatibility gate)",
)
async def activate_cluster(
    cluster_id: ClusterId, service: ClusterServiceDep
) -> ApiResponse[dict[str, Any]]:
    """All members apply at sign*strength*lambda (clamped). Blocks on
    feature-space mismatch or out-of-bounds member indices."""
    return ApiResponse.ok(await service.activate(cluster_id))


@router.post(
    "/{cluster_id}/deactivate",
    response_model=ApiResponse[dict[str, Any]],
    summary="Deactivate a cluster",
)
async def deactivate_cluster(
    cluster_id: ClusterId, service: ClusterServiceDep
) -> ApiResponse[dict[str, Any]]:
    return ApiResponse.ok(await service.deactivate(cluster_id))


@router.put(
    "/active/intensity",
    response_model=ApiResponse[dict[str, Any]],
    summary="Set the ACTIVE cluster's intensity (GLOBAL dial)",
)
async def set_active_intensity(
    request: SetIntensityRequest, service: ClusterServiceDep
) -> ApiResponse[dict[str, Any]]:
    """
    GLOBAL: persists lambda on the active cluster and re-applies steering.
    Per-request isolation is the OpenAI-API `steering_intensity` extension
    (Feature 10) — this endpoint serves the Admin UI and MCP tools.
    """
    active = await service.get_active_cluster()
    if active is None:
        return ApiResponse.fail(
            code="NO_ACTIVE_CLUSTER",
            message="No cluster profile is currently active",
        )
    return ApiResponse.ok(
        await service.set_intensity(active.id, request.intensity, reapply=request.reapply)
    )


@router.put(
    "/{cluster_id}/intensity",
    response_model=ApiResponse[dict[str, Any]],
    summary="Set a cluster's intensity (lambda)",
)
async def set_intensity(
    cluster_id: ClusterId,
    request: SetIntensityRequest,
    service: ClusterServiceDep,
) -> ApiResponse[dict[str, Any]]:
    return ApiResponse.ok(
        await service.set_intensity(cluster_id, request.intensity, reapply=request.reapply)
    )


@router.get(
    "/{cluster_id}/export",
    summary="Re-export the lossless original definition",
)
async def export_cluster(
    cluster_id: ClusterId, service: ClusterServiceDep
) -> dict[str, Any]:
    """Returns the raw `mistudio.cluster-definition/v1` document (no envelope —
    the response IS the portable artifact). Deliberately NO response_model:
    the mirror's extra="ignore" would strip unknown additive fields from
    newer producers (round-2 find)."""
    return await service.export_definition(cluster_id)
