"""Environment settings (Feature 35; ENH-003 §3.2, decision D3): what configures miLLM, and the 30
settings the operator may change while it runs.

    GET    /api/environment                 every setting: in-force value, environment value, source,
                                            group, when a change takes effect, last change
    PUT    /api/environment                 save several at once  {values: {key: value}, reset: [key],
                                            reason?} — validated as a whole, one commit (D75)
    PUT    /api/environment/{key}           save an override  {value, reason?}
    DELETE /api/environment/{key}?reason=   use the environment's value again
    GET    /api/environment/history?limit=  the change history, newest first (default 50)

Only `runtime_settings.RUNTIME_KEYS` are writable: any other setting answers 422
`SETTING_NOT_RUNTIME`, an unknown name 404 `SETTING_NOT_FOUND`, a refused value 422
`SETTING_INVALID` with the rule that refused it. SECRETS ARE NEVER RETURNED: a secret's row carries
only whether it is set. No auth gate (the suite's no-login decision).

ENH-006 D76: a write (any of the three) by an AI agent (`X-miLLM-Agent`) that names a TELEMETRY
setting is not run: it waits in miLLM's approval queue as `telemetry_write` (202,
`X-miLLM-Approval: pending`) and runs once, as written, when the operator approves it. The
operator's own write applies at once. Other keys keep Feature 35's posture (D5).
"""

from typing import Annotated, Any, Optional

from fastapi import APIRouter, Depends, Query, Request
from pydantic import BaseModel, Field

from millm.api.approval_gate import (
    Actor,
    actor_of,
    executing_actor,
    requires_approval_when_agent,
)
from millm.api.dependencies import DbSession
from millm.api.schemas.common import ApiResponse
from millm.services import approval_facts, runtime_settings

router = APIRouter(prefix="/api/environment", tags=["environment"])


class EnvironmentRow(BaseModel):
    key: str
    label: str
    description: Optional[str] = None
    group: str = Field(description="runtime | model_load | restart | environment")
    effect: str = Field(description="When a change takes effect")
    type: Optional[str] = Field(None, description="int | float | choice (writable keys only)")
    secret: bool
    set: Optional[bool] = Field(None, description="Secrets only: whether a value is configured")
    value: Any = Field(None, description="The value in force (null for a secret)")
    env_value: Any = Field(None, description="The environment's value (null for a secret)")
    source: str = Field(description="env | override")
    runtime_writable: bool
    minimum: Optional[float] = None
    maximum: Optional[float] = None
    choices: Optional[list[str]] = None
    updated_at: Optional[str] = None
    updated_by: Optional[str] = None
    reason: Optional[str] = None


class EnvironmentListing(BaseModel):
    settings: list[EnvironmentRow]
    counts: dict[str, int]
    runtime_keys: list[str]


class EnvironmentChange(BaseModel):
    id: int
    key: str
    action: str = Field(description="set | reset")
    previous_value: Any = None
    new_value: Any = None
    actor: str
    approval_id: Optional[str] = None
    reason: Optional[str] = None
    changed_at: Optional[str] = None


class EnvironmentHistory(BaseModel):
    changes: list[EnvironmentChange]


class EnvironmentWrite(BaseModel):
    value: Any = Field(description="The new value, typed as the setting is (number or text)")
    reason: Optional[str] = Field(
        None, max_length=runtime_settings.REASON_MAX_CHARS,
        description="Optional: why, kept in the change history",
    )


class EnvironmentSave(BaseModel):
    """ENH-006 D75: one save of several settings."""

    values: dict[str, Any] = Field(
        default_factory=dict, description="Settings to change: key -> new value")
    reset: list[str] = Field(
        default_factory=list, description="Settings to return to the environment's value")
    reason: Optional[str] = Field(
        None, max_length=runtime_settings.REASON_MAX_CHARS,
        description="Optional: why, kept in the change history",
    )


class EnvironmentSaved(BaseModel):
    settings: list[EnvironmentRow]


@router.get("", response_model=ApiResponse[EnvironmentListing])
async def list_environment() -> ApiResponse[EnvironmentListing]:
    return ApiResponse.ok(EnvironmentListing(**runtime_settings.listing()))


@router.get("/history", response_model=ApiResponse[EnvironmentHistory])
async def environment_history(
    session: DbSession, limit: int = Query(50, ge=1, le=200)
) -> ApiResponse[EnvironmentHistory]:
    changes = await runtime_settings.history(session, limit)
    return ApiResponse.ok(EnvironmentHistory(changes=[EnvironmentChange(**c) for c in changes]))


def request_actor(request: Request) -> Actor:
    """Who sent the request, from its `X-miLLM-Agent` header (rebuilt as the operator when an
    approved request is replayed; `executing_actor()` then names the agent)."""
    return actor_of(request)


RequestActor = Annotated[Actor, Depends(request_actor)]


def _who(sender: Actor) -> tuple[str, Optional[str]]:
    """(actor, approval_id) for the history: an approved agent request records `agent:<name>` and
    its approval (ENH-006 D76); an agent's ungated write (a non-telemetry key) records its name;
    anything else is the operator."""
    running = executing_actor()
    if running is not None:
        return running.agent or "agent", running.approval_id
    if sender.origin == "agent":
        return sender.agent or "agent", None
    return runtime_settings.ACTOR_OPERATOR, None


def _names_telemetry_key(values: dict[str, Any]) -> bool:
    return runtime_settings.is_telemetry_key(values.get("key"))


@router.put("", response_model=ApiResponse[EnvironmentSaved])
@requires_approval_when_agent("telemetry_write", when=approval_facts.names_telemetry,
                              describe=approval_facts.telemetry_write)
async def save_environment(
    body: EnvironmentSave, session: DbSession, sender: RequestActor
) -> ApiResponse[EnvironmentSaved]:
    """ENH-006 D75: several settings in ONE save, validated as a whole and committed together;
    the export pipeline is rebuilt once however many telemetry keys change. An agent's save that
    names a telemetry key waits for the operator (202, `telemetry_write`)."""
    actor, approval_id = _who(sender)
    keys = await runtime_settings.save_changes(
        session, body.values, body.reset, reason=body.reason, actor=actor,
        approval_id=approval_id)
    return ApiResponse.ok(EnvironmentSaved(
        settings=[EnvironmentRow(**runtime_settings.row_for(k)) for k in keys]))


@router.put("/{key}", response_model=ApiResponse[EnvironmentRow])
@requires_approval_when_agent("telemetry_write", when=_names_telemetry_key,
                              describe=approval_facts.telemetry_write)
async def set_environment(
    key: str, body: EnvironmentWrite, session: DbSession, sender: RequestActor
) -> ApiResponse[EnvironmentRow]:
    actor, approval_id = _who(sender)
    row = await runtime_settings.set_override(session, key, body.value, reason=body.reason,
                                              actor=actor, approval_id=approval_id)
    return ApiResponse.ok(EnvironmentRow(**row))


@router.delete("/{key}", response_model=ApiResponse[EnvironmentRow])
@requires_approval_when_agent("telemetry_write", when=_names_telemetry_key,
                              describe=approval_facts.telemetry_write)
async def reset_environment(
    key: str,
    session: DbSession,
    sender: RequestActor,
    reason: Optional[str] = Query(None, max_length=runtime_settings.REASON_MAX_CHARS),
) -> ApiResponse[EnvironmentRow]:
    actor, approval_id = _who(sender)
    row = await runtime_settings.reset_override(session, key, reason=reason, actor=actor,
                                                approval_id=approval_id)
    return ApiResponse.ok(EnvironmentRow(**row))
