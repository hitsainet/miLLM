"""
The model lease registry (Feature 29).

A lease pins the RESIDENT model for a named holder until its time to live (TTL) runs out.
While it is live, nobody else can load, unload or swap the model; the holder proves it holds
the lease by sending the lease ID in `X-miLLM-Lease`. The rules (residency, a load in
progress, the lift) live in `ModelService`; this module stores leases and nothing else.

Decisions this module encodes:

* **X-01 — process memory only.** A restart ends every lease. There is no table and no
  migration; `clear_leases_on_startup()` is called from `lifespan` so the reconciliation is
  explicit, tested, and survives a future change that makes leases durable.
* **X-08 — one lease per model.** The registry is keyed by model. Only the resident model can
  be leased (T-85), so at most one lease is live at a time.
* **The lease ID is a bearer secret.** It is returned once, by `grant`. The registry stores its
  SHA-256 digest; logs carry `lease_ref` (the digest's first 8 hex characters), never the ID.
* **Expiry is lazy and monotonic.** Every read compares `time.monotonic()` with the record's
  monotonic deadline (`<=`: at the deadline the lease is gone). A wall-clock step never extends
  or ends a lease; wall-clock times are for display.
* **The read heals itself.** `current()` also ends a lease whose model is no longer the
  loader's resident model (reason `model_unloaded`), covering a forced unload and any path
  that empties the loader without passing through `unload_model`'s success branch.

⚠ **Single-process assumption.** The image runs uvicorn with no `--workers` (`Dockerfile:124`).
Several workers would each hold their own registry, and a lease taken in one would protect
nothing in another. If miLLM ever runs several workers, this registry must move to shared
storage (029 FTDD §9, §12).
"""

from __future__ import annotations

import hashlib
import hmac
import math
import secrets
import threading
import time
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone

from millm.core.errors import LeaseExpiredError, LeaseNotFoundError, ModelLeasedError
from millm.core.logging import get_logger

logger = get_logger(__name__)

#: The sentence every unknown-lease 404 carries (FR-29.1.7; X-01).
UNKNOWN_LEASE_MESSAGE = "unknown lease; a restart ends every lease"

#: End reasons (029 FTDD §4).
END_RELEASED = "released"
END_EXPIRED = "expired"
END_MODEL_UNLOADED = "model_unloaded"
END_RESTART = "restart"


def _digest(lease_id: str) -> str:
    """SHA-256 hex of a lease ID. One of the two places a lease ID is ever hashed."""
    return hashlib.sha256(lease_id.encode("utf-8")).hexdigest()


def _ref(digest: str) -> str:
    """The only form of a lease that appears in a log line."""
    return digest[:8]


@dataclass(frozen=True)
class LeaseRecord:
    """A live lease. Holds the DIGEST of its ID, never the ID."""

    digest: str
    lease_ref: str
    model_id: int
    model_name: str
    holder: str
    reason: str
    ttl_seconds: int
    acquired_at: datetime
    renewed_at: datetime | None
    expires_at: datetime
    expires_mono: float


@dataclass(frozen=True)
class LeaseGrant:
    """What `grant` returns: the record, and the ID — the one time the ID leaves the registry."""

    lease_id: str
    record: LeaseRecord


@dataclass(frozen=True)
class EndedLease:
    """A lease that has ended, remembered so renew/release can say why (409, not 404)."""

    record: LeaseRecord
    end_reason: str
    ended_at: datetime


def _default_resident_model_id() -> int | None:
    from millm.ml.model_loader import LoadedModelState

    return LoadedModelState().loaded_model_id


def _default_now() -> datetime:
    return datetime.now(timezone.utc)


class LeaseRegistry:
    """One live lease per model (X-08). Process memory only (X-01). Holds DIGESTS, never IDs.

    Every method takes a `threading.Lock` for microseconds and never awaits, so a check made
    by a caller and the claim it guards can sit with no `await` between them.
    """

    def __init__(
        self,
        resident_model_id: Callable[[], int | None] = _default_resident_model_id,
        monotonic: Callable[[], float] = time.monotonic,
        now: Callable[[], datetime] = _default_now,
        ended_memory: int | None = None,
    ) -> None:
        from millm.core.config import settings

        self._resident_model_id = resident_model_id
        self.monotonic = monotonic
        self.now = now
        self._lock = threading.Lock()
        self._live: dict[int, LeaseRecord] = {}
        self._ended: OrderedDict[str, EndedLease] = OrderedDict()
        self._last_ended: dict[int, EndedLease] = {}
        self._ended_memory = max(
            ended_memory if ended_memory is not None else settings.LEASE_ENDED_MEMORY, 1
        )

    # --- internal, called with the lock held ------------------------------------------------

    def _end_locked(self, record: LeaseRecord, reason: str) -> EndedLease:
        self._live.pop(record.model_id, None)
        ended = EndedLease(record=record, end_reason=reason, ended_at=self.now())
        self._ended[record.digest] = ended
        self._ended.move_to_end(record.digest)
        while len(self._ended) > self._ended_memory:
            self._ended.popitem(last=False)
        self._last_ended[record.model_id] = ended
        event = {
            END_EXPIRED: "lease_expired",
            END_RELEASED: "lease_released",
        }.get(reason, "lease_ended")
        logger.info(
            event,
            lease_ref=record.lease_ref,
            holder=record.holder,
            model_id=record.model_id,
            reason=record.reason,
            end_reason=reason,
        )
        return ended

    def _current_locked(self, model_id: int) -> LeaseRecord | None:
        record = self._live.get(model_id)
        if record is None:
            return None
        if record.expires_mono <= self.monotonic():
            # `<=`: AT the deadline the lease is gone (FR-29.1.10; mutation control M5).
            self._end_locked(record, END_EXPIRED)
            return None
        if self._resident_model_id() != record.model_id:
            # Self-healing read (FR-29.1.9): the model stopped being resident without passing
            # through unload_model's success branch — a forced unload, a failed load that
            # emptied the loader. A lease on a model that is gone protects nothing.
            self._end_locked(record, END_MODEL_UNLOADED)
            return None
        return record

    def _live_for_id_locked(self, model_id: int, lease_id: str | None) -> LeaseRecord:
        """The live record on `model_id` that `lease_id` proves; raises 404/409 otherwise."""
        digest = _digest(lease_id) if lease_id else ""
        record = self._current_locked(model_id)
        if record is not None and lease_id and hmac.compare_digest(digest, record.digest):
            return record
        ended = self._ended.get(digest) if lease_id else None
        if ended is not None and ended.record.model_id == model_id:
            raise LeaseExpiredError(
                f"The lease on model {model_id} has ended ({ended.end_reason}).",
                details={
                    "model_id": model_id,
                    "end_reason": ended.end_reason,
                    "ended_at": ended.ended_at.isoformat(),
                },
            )
        # Unknown, or a lease on ANOTHER model: the same 404, so the route reveals nothing
        # about other models' leases (029 FTDD §5.1).
        raise LeaseNotFoundError(
            f"Model {model_id}: {UNKNOWN_LEASE_MESSAGE}.", details={"model_id": model_id}
        )

    # --- public API -------------------------------------------------------------------------

    def grant(
        self, model_id: int, model_name: str, holder: str, reason: str, ttl_s: int
    ) -> LeaseGrant:
        """A new lease on `model_id`. Refuses with `ModelLeasedError` while another is live."""
        with self._lock:
            existing = self._current_locked(model_id)
            if existing is not None:
                logger.warning(
                    "lease_refused",
                    lease_ref=existing.lease_ref,
                    holder=existing.holder,
                    model_id=model_id,
                    reason=existing.reason,
                    operation="lease",
                    target_model_id=model_id,
                )
                raise ModelLeasedError.for_lease(
                    model_id=existing.model_id,
                    model_name=existing.model_name,
                    holder=existing.holder,
                    reason=existing.reason,
                    expires_at=existing.expires_at.isoformat(),
                    operation="lease",
                    target_model_id=model_id,
                )
            lease_id = secrets.token_urlsafe(24)
            digest = _digest(lease_id)
            now = self.now()
            record = LeaseRecord(
                digest=digest,
                lease_ref=_ref(digest),
                model_id=model_id,
                model_name=model_name,
                holder=holder,
                reason=reason,
                ttl_seconds=ttl_s,
                acquired_at=now,
                renewed_at=None,
                expires_at=now + timedelta(seconds=ttl_s),
                expires_mono=self.monotonic() + ttl_s,
            )
            self._live[model_id] = record
        logger.info(
            "lease_granted",
            lease_ref=record.lease_ref,
            holder=holder,
            model_id=model_id,
            reason=reason,
            ttl_seconds=ttl_s,
        )
        return LeaseGrant(lease_id=lease_id, record=record)

    def renew(self, model_id: int, lease_id: str | None, ttl_s: int) -> LeaseRecord:
        """New expiry = NOW + ttl, not the old expiry + ttl (FR-29.1.7)."""
        with self._lock:
            record = self._live_for_id_locked(model_id, lease_id)
            now = self.now()
            renewed = replace(
                record,
                ttl_seconds=ttl_s,
                renewed_at=now,
                expires_at=now + timedelta(seconds=ttl_s),
                expires_mono=self.monotonic() + ttl_s,
            )
            self._live[model_id] = renewed
        logger.info(
            "lease_renewed",
            lease_ref=renewed.lease_ref,
            holder=renewed.holder,
            model_id=model_id,
            reason=renewed.reason,
            ttl_seconds=ttl_s,
        )
        return renewed

    def release(self, model_id: int, lease_id: str | None) -> EndedLease:
        """End the lease at once."""
        with self._lock:
            record = self._live_for_id_locked(model_id, lease_id)
            return self._end_locked(record, END_RELEASED)

    def resolve(self, lease_id: str | None) -> LeaseRecord | None:
        """The LIVE record this ID proves, on any model; None otherwise (Feature 26's lookup)."""
        if not lease_id:
            return None
        digest = _digest(lease_id)
        with self._lock:
            for model_id in list(self._live):
                record = self._current_locked(model_id)
                if record is not None and hmac.compare_digest(digest, record.digest):
                    return record
        return None

    def current(self, model_id: int | None) -> LeaseRecord | None:
        """The live lease on `model_id`, after lazy expiry and the residency self-heal."""
        if model_id is None:
            return None
        with self._lock:
            return self._current_locked(model_id)

    def matches(self, record: LeaseRecord, lease_id: str | None) -> bool:
        """Whether `lease_id` proves `record`, compared in constant time."""
        if not lease_id:
            return False
        return hmac.compare_digest(_digest(lease_id), record.digest)

    def end_for_model(self, model_id: int, reason: str) -> EndedLease | None:
        """End whatever lease `model_id` holds (unload success: reason `model_unloaded`)."""
        with self._lock:
            record = self._live.get(model_id)
            if record is None:
                return None
            return self._end_locked(record, reason)

    def last_ended(self, model_id: int) -> EndedLease | None:
        """The most recent lease on `model_id` that ended, if this process remembers one."""
        with self._lock:
            return self._last_ended.get(model_id)

    def seconds_remaining(self, record: LeaseRecord) -> int:
        """Whole seconds until `record` expires, by the monotonic clock; never negative."""
        return max(0, int(math.ceil(record.expires_mono - self.monotonic())))

    def clear(self, reason: str = END_RESTART) -> int:
        """End every live lease with `reason`; returns how many there were."""
        with self._lock:
            live = list(self._live.values())
            for record in live:
                self._end_locked(record, reason)
            return len(live)


_REGISTRY: dict[str, LeaseRegistry | None] = {"registry": None}
_REGISTRY_LOCK = threading.Lock()


def get_lease_registry() -> LeaseRegistry:
    """The process-wide registry, created on first use (like `LoadedModelState`)."""
    registry = _REGISTRY["registry"]
    if registry is None:
        with _REGISTRY_LOCK:
            registry = _REGISTRY["registry"]
            if registry is None:
                registry = LeaseRegistry()
                _REGISTRY["registry"] = registry
    return registry


def set_lease_registry(registry: LeaseRegistry | None) -> None:
    """Replace the process-wide registry (tests inject one with controllable clocks)."""
    with _REGISTRY_LOCK:
        _REGISTRY["registry"] = registry


def clear_leases_on_startup() -> int:
    """End every lease at startup, because no lease survives a restart (X-01). Never raises.

    In production the count is always 0 — a new process starts with an empty registry. The
    call exists so the reconciliation is EXPLICIT: this estate has shipped "a new in-memory
    state left out of startup reconciliation" three times (miLLM's steering lock hid thirteen
    models for three months; probes stayed "armed" with no hook). The same reasoning as
    `disarm_probes_on_startup`: a named function, exercised for real by a test, and a second
    test that fails when `lifespan` stops calling it. `current()`'s residency self-heal keeps
    the READ honest even if this fails.
    """
    try:
        count = get_lease_registry().clear(END_RESTART)
        logger.info(
            "leases_cleared_on_startup",
            count=count,
            detail="a restart ends every lease (X-01); holders re-acquire",
        )
        return count
    except Exception as e:  # noqa: BLE001 - lease bookkeeping must never stop the server starting
        logger.error(
            "lease_startup_clear_failed",
            error=str(e),
            error_type=type(e).__name__,
            exc_info=True,
        )
        return 0
