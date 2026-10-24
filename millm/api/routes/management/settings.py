"""Operator settings (Feature 34): the Hugging Face token the Settings page saves.

    GET    /api/settings/hf-token        status: configured, active source, masked value, last check
    PUT    /api/settings/hf-token        save (encrypted; refused 409 SETTINGS_KEY_MISSING with no key)
    DELETE /api/settings/hf-token        clear the saved token
    POST   /api/settings/hf-token/test   whoami → the user name, or INVALID_HF_TOKEN / HF_HUB_UNREACHABLE

NO ROUTE RETURNS THE TOKEN. Each answers `status()` (masked) or the whoami user name. No auth gate:
no management write in miLLM has one (the suite's no-login decision), and this follows them.
"""

from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel, Field

from millm.api.dependencies import DbSession
from millm.api.schemas.common import ApiResponse
from millm.services import hf_token

router = APIRouter(prefix="/api/settings", tags=["settings"])


class HfTokenStatus(BaseModel):
    configured: bool
    source: str = Field(description="settings | env | none — the tier a Hub call uses when the request carries no token")
    masked: str | None = None
    last_verified_at: str | None = None
    user: str | None = None
    stored: bool
    stored_error: str | None = None
    last_check_error: str | None = None
    encryption_available: bool
    env_configured: bool


class HfTokenSaveRequest(BaseModel):
    # `exclude=True`: never part of a dump of this model, like every `hf_token` request field.
    token: str = Field(exclude=True, description="The Hugging Face access token to save")


class HfTokenTestRequest(BaseModel):
    token: str | None = Field(
        default=None, exclude=True, description="A token to check without saving it; absent → the active token"
    )


class HfTokenTestResult(BaseModel):
    ok: bool
    user: str | None = None
    source: str
    verified_at: str


def _status() -> ApiResponse[HfTokenStatus]:
    return ApiResponse.ok(HfTokenStatus(**hf_token.status()))


@router.get("/hf-token", response_model=ApiResponse[HfTokenStatus])
async def get_hf_token_status() -> ApiResponse[HfTokenStatus]:
    return _status()


@router.put("/hf-token", response_model=ApiResponse[HfTokenStatus])
async def save_hf_token(body: HfTokenSaveRequest, session: DbSession) -> ApiResponse[HfTokenStatus]:
    await hf_token.save_token(session, body.token)
    return _status()


@router.delete("/hf-token", response_model=ApiResponse[HfTokenStatus])
async def clear_hf_token(session: DbSession) -> ApiResponse[HfTokenStatus]:
    await hf_token.clear_token(session)
    return _status()


@router.post("/hf-token/test", response_model=ApiResponse[HfTokenTestResult])
async def test_hf_token(
    session: DbSession, body: HfTokenTestRequest | None = None
) -> ApiResponse[HfTokenTestResult]:
    result: dict[str, Any] = await hf_token.check_token(session, body.token if body else None)
    return ApiResponse.ok(HfTokenTestResult(**result))
