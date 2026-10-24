"""The one audit writer for the gated weights calls (FR-31.15; addendum §2.1).

Every gated call, accepted or refused, writes one `weights_audit` line: who (role and label,
both null for an unauthenticated refusal), what (operation, outcome, code, reason), on which
resident model and weights state, and when (UTC).

⚠ TYPED KEYWORDS ONLY, NO `**kwargs`. A free-form field set is how a token, an attachment ID
or a request body ends up in a log line by accident; with a closed signature a caller cannot
pass one without editing this file. `attachment_ref` is the 8-hex ref of the attachment's
digest, never the ID. Features 32 and 33 pass `checkpoint_sha256` / `weights_sha256`,
`adapter`, `mode` and `path` (relative to the checkpoint root); 031's routes pass
`state_before` and `elapsed_ms` (FR-31.15.2, addendum §7 item 6).
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Literal, Optional

from millm.core.logging import get_logger

if TYPE_CHECKING:
    from millm.core.weights_auth import WeightsIdentity

logger = get_logger(__name__)

Outcome = Literal["accepted", "refused"]


def _context() -> tuple[Optional[str], Optional[str]]:
    """The resident model's served name and the barrier's weights state, read for the line.

    An audit line must never turn an accepted operation into a 500, so a failed read records
    None (not recorded) and says why in its own line.
    """
    try:
        from millm.services.resident_identity import served_name
        from millm.services.weight_barrier import get_weight_barrier

        return served_name(), get_weight_barrier().snapshot().weights_state
    except Exception as e:  # noqa: BLE001
        logger.warning("weights_audit_context_unavailable", error_type=type(e).__name__)
        return None, None


def weights_audit(
    operation: str,
    outcome: Outcome,
    identity: Optional["WeightsIdentity"],
    *,
    code: Optional[str] = None,
    reason: Optional[str] = None,
    attachment_ref: Optional[str] = None,
    weights_sha256: Optional[str] = None,
    checkpoint_sha256: Optional[str] = None,
    adapter: Optional[str] = None,
    mode: Optional[str] = None,
    commit_ms: Optional[float] = None,
    steps: Optional[int] = None,
    path: Optional[str] = None,
    state_before: Optional[str] = None,
    elapsed_ms: Optional[float] = None,
) -> None:
    """Write one `weights_audit` line: info when accepted, warning when refused.

    The core fields are always present (null = not applicable or not recorded); the
    operation-specific ones appear only when the caller passes them. `weights_state` is the
    state AFTER the call; `state_before` (logged as `weights_state_before`) and `elapsed_ms`
    are FR-31.15.2's — the barrier's results carry `state_before` for the route to pass.

    Raises:
        ValueError: `outcome` is not `accepted` or `refused` (a programming error).
    """
    if outcome not in ("accepted", "refused"):
        raise ValueError(f"weights_audit outcome must be 'accepted' or 'refused', got {outcome!r}")
    resident_model, weights_state = _context()
    fields: dict[str, Any] = {
        "operation": operation,
        "outcome": outcome,
        "code": code,
        "reason": reason,
        "role": identity.role if identity is not None else None,
        "label": identity.label if identity is not None else None,
        "attachment_ref": attachment_ref,
        "resident_model": resident_model,
        "weights_state": weights_state,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    optional = {
        "weights_sha256": weights_sha256,
        "checkpoint_sha256": checkpoint_sha256,
        "adapter": adapter,
        "mode": mode,
        "commit_ms": commit_ms,
        "steps": steps,
        "path": path,
        "weights_state_before": state_before,
        "elapsed_ms": elapsed_ms,
    }
    fields.update({k: v for k, v in optional.items() if v is not None})
    if outcome == "accepted":
        logger.info("weights_audit", **fields)
    else:
        logger.warning("weights_audit", **fields)
