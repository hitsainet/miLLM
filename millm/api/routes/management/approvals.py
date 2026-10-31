"""`/api/approvals` — miLLM's approval queue (ENH-008 D1, D6; BUILD_SPEC §3.1).

* `GET  /api/approvals?status=&limit=&offset=` — newest first; `status=decided` is every request
  no longer waiting, newest decision first (the History section). `{approvals, count, pending}`.
* `GET  /api/approvals/actions` — the gated actions, each with its label and the routes the LIVE
  app gates with it. miStudio checks this before relaying a gated tool (fail closed).
* `GET  /api/approvals/{id}`
* `POST /api/approvals/{id}/approve` — 202; runs the stored request exactly once.
* `POST /api/approvals/{id}/reject` — body `{reason?}`.
* `POST /api/approvals/approve-batch` — body `{approval_ids: [...], kind?}` with `kind`
  `settings` (default; ENH-006 D79, "Approve all settings changes") or `code_rules` (D81, "Approve
  all code rules"): every id must be a waiting request of that kind or NOTHING is approved (409
  `approval_batch_refused`); kinds never mix. Each is then approved through the single approve,
  exactly once, with a shared `batch_id`. 202 `{batch_id, kind, results: [{approval_id, status,
  error}]}`.

⚠ An agent can never decide (OSD-56): any `X-miLLM-Agent` header on approve or reject is refused
403 `approval_operator_only`, and the row is unchanged. ⚠ No route here returns `secret_payload`.

`/actions` and `/approve-batch` are declared above `/{approval_id}` so neither literal path is read
as an id.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal

from fastapi import APIRouter, Body, Path, Query, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from millm.api.dependencies import DbSession
from millm.api.schemas.common import ApiResponse
from millm.db.models.approval import REASON_MAX_CHARS
from millm.services import approval_service

router = APIRouter(prefix="/api/approvals", tags=["approvals"])

APPROVAL_ID = r"^apr_[0-9a-f]{32}$"
ApprovalId = Annotated[str, Path(pattern=APPROVAL_ID, description="The approval request id")]


class ApproveBatchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    approval_ids: list[Annotated[str, Field(pattern=APPROVAL_ID)]] = Field(
        min_length=1, max_length=approval_service.BATCH_MAX)
    kind: Literal["settings", "code_rules"] = "settings"


class RejectRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    reason: str | None = Field(default=None, max_length=REASON_MAX_CHARS)


@router.get("", response_model=ApiResponse)
async def list_approvals(
    session: DbSession,
    status: Annotated[str | None, Query(max_length=16)] = None,
    limit: Annotated[int, Query(ge=1, le=500)] = 200,
    offset: Annotated[int, Query(ge=0, le=1_000_000)] = 0,
) -> ApiResponse[Any]:
    rows, pending = await approval_service.listing(session, status=status, limit=limit,
                                                   offset=offset)
    approvals = [approval_service.as_payload(r) for r in rows]
    return ApiResponse.ok({"approvals": approvals, "count": len(approvals), "pending": pending})


@router.get("/actions", response_model=ApiResponse)
async def list_approval_actions(request: Request) -> ApiResponse[Any]:
    """Every action this miLLM gates, read from the LIVE route table: an action no route carries
    is not listed, so a client checking for one learns the truth about this deployment."""
    from millm.api.approval_gate import APPROVAL_ACTIONS, gated_routes

    routes: dict[str, list[str]] = {}
    for method, path, action in gated_routes(request.app):
        routes.setdefault(action, []).append(f"{method} {path}")
    return ApiResponse.ok({
        "actions": [
            {"action": name, "label": APPROVAL_ACTIONS[name].label,
             "description": APPROVAL_ACTIONS[name].description,
             "settings_change": APPROVAL_ACTIONS[name].settings_change,
             "batch_kind": APPROVAL_ACTIONS[name].batch_kind, "routes": routes[name]}
            for name in APPROVAL_ACTIONS if name in routes
        ],
    })


@router.post("/approve-batch", response_model=ApiResponse, status_code=202)
async def approve_batch(body: ApproveBatchRequest, request: Request, session: DbSession) -> Any:
    """ENH-006 D79/D81: approve every named waiting request of one kind with one decision.
    Operator only."""
    from millm.api.approval_gate import refuse_agent

    refuse_agent(request)
    batch_id, results = await approval_service.approve_batch(session, body.approval_ids,
                                                             app=request.app, kind=body.kind)
    return JSONResponse(status_code=202, content=ApiResponse.ok(
        {"batch_id": batch_id, "kind": body.kind, "results": results}).model_dump(mode="json"))


@router.get("/{approval_id}", response_model=ApiResponse)
async def get_approval(approval_id: ApprovalId, session: DbSession) -> ApiResponse[Any]:
    return ApiResponse.ok(approval_service.as_payload(
        await approval_service.get(session, approval_id)))


@router.post("/{approval_id}/approve", response_model=ApiResponse, status_code=202)
async def approve_approval(approval_id: ApprovalId, request: Request,
                           session: DbSession) -> Any:
    from millm.api.approval_gate import refuse_agent

    refuse_agent(request)
    row = await approval_service.approve(session, approval_id, app=request.app)
    return JSONResponse(status_code=202, content=ApiResponse.ok(
        approval_service.as_payload(row)).model_dump(mode="json"))


@router.post("/{approval_id}/reject", response_model=ApiResponse)
async def reject_approval(
    approval_id: ApprovalId,
    request: Request,
    session: DbSession,
    body: Annotated[RejectRequest | None, Body()] = None,
) -> ApiResponse[Any]:
    from millm.api.approval_gate import refuse_agent

    refuse_agent(request)
    reason = (body.reason or "").strip() if body is not None else ""
    row = await approval_service.reject(session, approval_id, reason=reason or None)
    return ApiResponse.ok(approval_service.as_payload(row))
