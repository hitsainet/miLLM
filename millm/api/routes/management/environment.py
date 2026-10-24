"""Environment settings (Feature 35; ENH-003 §3.2, decision D3): what configures miLLM, and the 30
settings the operator may change while it runs.

    GET    /api/environment                 every setting: in-force value, environment value, source,
                                            group, when a change takes effect, last change
    PUT    /api/environment/{key}           save an override  {value, reason?}
    DELETE /api/environment/{key}?reason=   use the environment's value again
    GET    /api/environment/history?limit=  the change history, newest first (default 50)

Only `runtime_settings.RUNTIME_KEYS` are writable: any other setting answers 422
`SETTING_NOT_RUNTIME`, an unknown name 404 `SETTING_NOT_FOUND`, a refused value 422
`SETTING_INVALID` with the rule that refused it. SECRETS ARE NEVER RETURNED: a secret's row carries
only whether it is set. No auth gate (the suite's no-login decision), and no MCP tool writes here in
v1 (D5): every change is the operator's.
"""

from typing import Any, Optional

from fastapi import APIRouter, Query
from pydantic import BaseModel, Field

from millm.api.dependencies import DbSession
from millm.api.schemas.common import ApiResponse
from millm.services import runtime_settings

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


@router.get("", response_model=ApiResponse[EnvironmentListing])
async def list_environment() -> ApiResponse[EnvironmentListing]:
    return ApiResponse.ok(EnvironmentListing(**runtime_settings.listing()))


@router.get("/history", response_model=ApiResponse[EnvironmentHistory])
async def environment_history(
    session: DbSession, limit: int = Query(50, ge=1, le=200)
) -> ApiResponse[EnvironmentHistory]:
    changes = await runtime_settings.history(session, limit)
    return ApiResponse.ok(EnvironmentHistory(changes=[EnvironmentChange(**c) for c in changes]))


@router.put("/{key}", response_model=ApiResponse[EnvironmentRow])
async def set_environment(
    key: str, body: EnvironmentWrite, session: DbSession
) -> ApiResponse[EnvironmentRow]:
    row = await runtime_settings.set_override(session, key, body.value, reason=body.reason)
    return ApiResponse.ok(EnvironmentRow(**row))


@router.delete("/{key}", response_model=ApiResponse[EnvironmentRow])
async def reset_environment(
    key: str,
    session: DbSession,
    reason: Optional[str] = Query(None, max_length=runtime_settings.REASON_MAX_CHARS),
) -> ApiResponse[EnvironmentRow]:
    row = await runtime_settings.reset_override(session, key, reason=reason)
    return ApiResponse.ok(EnvironmentRow(**row))
