"""
Model lease schemas (Feature 29, 029 FTDD §5.1).

⚠ The lease ID appears in exactly ONE response: `LeaseGrantResponse`, returned by the grant.
No other schema here has a `lease_id` field, and every one forbids extras, so a read cannot
carry it by accident (FR-29.1.6). The status shape is built by ONE serialiser,
`lease_summary`, used by the lease `GET`, `ModelResponse.lease` and `/api/health/detailed`
(memory `one-list-needs-one-serialiser`).

The request schemas declare no bounds and accept any JSON type for their fields: the service
validates, so a refusal is `400 INVALID_LEASE_REQUEST` naming the field and the limit rather
than a bare 422 (029 FTDD TD14).
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:  # importing millm.services at module level would cycle through model_service
    from millm.services.model_lease import EndedLease, LeaseRecord, LeaseRegistry


class LeaseCreateRequest(BaseModel):
    """`POST /api/models/{id}/lease`. Types and bounds are checked by the service (400)."""

    model_config = ConfigDict(extra="forbid")

    holder: Any = Field(None, description="Free-text label for who holds the lease (1-128 chars)")
    reason: Any = Field(None, description="Why the model is pinned (1-512 chars)")
    ttl_seconds: Any = Field(
        None, description="Seconds until the lease expires, 1-7200; default 7200"
    )


class LeaseRenewRequest(BaseModel):
    """`POST /api/models/{id}/lease/renew`. The lease ID travels in `X-miLLM-Lease`."""

    model_config = ConfigDict(extra="forbid")

    ttl_seconds: Any = Field(None, description="New TTL from NOW, 1-7200; default 7200")


class LeaseStatusResponse(BaseModel):
    """A live lease, as every read reports it. Never carries the lease ID."""

    model_config = ConfigDict(extra="forbid")

    model_id: int
    model_name: str
    holder: str
    reason: str
    acquired_at: datetime
    renewed_at: datetime | None = None
    expires_at: datetime
    ttl_seconds: int
    seconds_remaining: int


#: The model list's lease summary is the same shape, from the same serialiser.
LeaseSummary = LeaseStatusResponse


class LeaseGrantResponse(LeaseStatusResponse):
    """The grant: the ONLY response that carries `lease_id`. Store it; it is shown once."""

    lease_id: str = Field(..., description="Send as X-miLLM-Lease; never returned again")


class EndedLeaseResponse(BaseModel):
    """A lease that has ended: released, expired, model_unloaded or restart."""

    model_config = ConfigDict(extra="forbid")

    model_id: int
    model_name: str
    holder: str
    reason: str
    acquired_at: datetime
    expires_at: datetime
    end_reason: str
    ended_at: datetime


class LeaseStatusEnvelope(BaseModel):
    """`GET /api/models/{id}/lease`: the live lease or null, and the last one that ended."""

    model_config = ConfigDict(extra="forbid")

    lease: LeaseStatusResponse | None
    last_ended: EndedLeaseResponse | None


def lease_summary(
    record: LeaseRecord, registry: LeaseRegistry | None = None
) -> LeaseStatusResponse:
    """THE serialiser of a live lease. Reads the record's attributes directly (no defaults)."""
    if registry is None:
        from millm.services.model_lease import get_lease_registry

        registry = get_lease_registry()
    return LeaseStatusResponse(
        model_id=record.model_id,
        model_name=record.model_name,
        holder=record.holder,
        reason=record.reason,
        acquired_at=record.acquired_at,
        renewed_at=record.renewed_at,
        expires_at=record.expires_at,
        ttl_seconds=record.ttl_seconds,
        seconds_remaining=registry.seconds_remaining(record),
    )


def lease_grant_response(lease_id: str, record: LeaseRecord) -> LeaseGrantResponse:
    """The grant response: the summary plus the ID, the one time it is returned."""
    return LeaseGrantResponse(**lease_summary(record).model_dump(), lease_id=lease_id)


def ended_lease_response(ended: EndedLease) -> EndedLeaseResponse:
    record = ended.record
    return EndedLeaseResponse(
        model_id=record.model_id,
        model_name=record.model_name,
        holder=record.holder,
        reason=record.reason,
        acquired_at=record.acquired_at,
        expires_at=record.expires_at,
        end_reason=ended.end_reason,
        ended_at=ended.ended_at,
    )
