"""What steering a generation actually ran under — the source of `X-miLLM-Steering` (Feature 28).

⚠ REPORT WHAT RAN, NEVER WHAT WAS ASKED (PADR v1.5 §10, FR-28.3.2). The reader's input is a
SNAPSHOT of hook-visible state — each attached SAE's live values, its enabled flag, the steering
epoch — taken inside the admission slot after generation and before the restore. The request
record only helps choose a LABEL, and only when the snapshot's values EQUAL the record's: a
request that asked for X while the hooks ran Y is reported as Y (as `manual` when nothing else
claims it). An echo of the request could never see an operator write that landed mid-request.

Labelling order per steered entry (FTDD §6.2):

1. the request record (same `(sae_id, layer)`, equal applied set) → `inline`, or
   `profile;source=request` (or `source=active` for a dial over the active profile);
2. a serving circuit whose plan claims the entry → grouped into ONE `circuit` item per circuit;
3. the active profile, when its applied set equals the entry's → `profile;source=active`;
4. otherwise → `manual`.

Equality is EXACT float equality. Both apply paths compute `clamp_steering(float(v) * λ)`, so an
equal set is the same set; an approximate match would let a different setting borrow a profile's
name.

`describe` and `capture` never raise into a request (FR-28.3.7): a failure is `unknown` with a
reason, never an omitted header and never a guess.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Any

from millm.core.logging import get_logger
from millm.core.steering_state import (
    SteeringItem,
    applied_set,
    count_clamped,
    serialize_steering_header,
    steering_set_hash,
)

logger = get_logger(__name__)


@dataclass(frozen=True)
class EntrySnapshot:
    """One attached entry as the hook reads it. `applied` has zeros removed."""

    sae_id: str
    layer: int
    enabled: bool
    applied: dict[int, float]

    @property
    def steered(self) -> bool:
        # Exactly the hook's condition (`sae_wrapper.apply_steering`): enabled AND a delta.
        # Per-thread suppression is never active at capture time on a generation path.
        return self.enabled and bool(self.applied)


@dataclass(frozen=True)
class SteeringSnapshot:
    """Hook-visible steering at one instant. `failed` when it could not be read."""

    epoch: int | None
    entries: tuple[EntrySnapshot, ...] = ()
    failed: bool = False
    error: str | None = None

    @classmethod
    def capture(cls, state: Any = None) -> SteeringSnapshot:
        """Copy every attached entry's live values, enabled flag, and the epoch. Never raises."""
        try:
            if state is None:
                from millm.services.sae_service import AttachedSAEState

                state = AttachedSAEState()
            entries = []
            for entry in state.entries():
                values = entry.sae.get_steering_values()  # already a copy
                entries.append(EntrySnapshot(
                    sae_id=str(entry.sae_id),
                    layer=int(entry.layer),
                    enabled=bool(entry.sae.is_steering_enabled),
                    applied={int(i): float(v) for i, v in values.items() if float(v) != 0.0},
                ))
            return cls(epoch=int(state.steering_epoch), entries=tuple(entries))
        except Exception as exc:  # noqa: BLE001 - a report must never fail a request
            logger.warning("steering_snapshot_failed", error=str(exc),
                           error_type=type(exc).__name__)
            return cls(epoch=None, entries=(), failed=True, error=str(exc))

    def steered(self) -> list[EntrySnapshot]:
        return [e for e in self.entries if e.steered]


@dataclass(frozen=True)
class RequestSteeringRecord:
    """What THIS request applied, for labelling only (never the report's source of truth).

    `kind`: `none` (nothing requested), `inline`, `unsteered`, `profile`, `dial` (a dial over
    live values with no profile), `circuit` (a dial over the serving circuit), `disabled` (λ = 0).
    """

    kind: str
    epoch_at_admission: int | None
    sae_id: str | None = None
    layer: int | None = None
    applied: dict[int, float] | None = None
    clamped: int = 0
    profile_name: str | None = None
    profile_source: str | None = None
    profile_sae_id: str | None = None
    intensity: float | None = None


@dataclass(frozen=True)
class SteeringReport:
    items: tuple[SteeringItem, ...]
    header: str

    @classmethod
    def from_items(cls, items: list[SteeringItem]) -> SteeringReport:
        return cls(items=tuple(items), header=serialize_steering_header(items))

    @classmethod
    def none(cls) -> SteeringReport:
        return cls.from_items([SteeringItem(kind="none")])

    @classmethod
    def unknown(cls, reason: str, changed: bool = False) -> SteeringReport:
        return cls.from_items([SteeringItem(kind="unknown", reason=reason, changed=changed)])


#: Scoring is always unsteered (X-09, FR-28.3.1): its report is this constant.
NONE_REPORT = SteeringReport.none()


class _ClaimsUnreadable(Exception):
    """The circuit claim table could not be read, so `composed` is unknowable (fail closed)."""


class SteeringStateReader:
    """Labels a snapshot. Async because the circuit and active-profile reads hit the database."""

    def __init__(self, inference: Any = None) -> None:
        self._inference = inference

    async def describe(
        self,
        snapshot: SteeringSnapshot,
        record: RequestSteeringRecord | None = None,
        *,
        epoch_at_admission: int | None = None,
        engine: str | None = None,
        request_id: str | None = None,
    ) -> SteeringReport:
        """The report for `snapshot`. Never raises (FR-28.3.7)."""
        try:
            return await self._describe(snapshot, record, epoch_at_admission, engine)
        except _ClaimsUnreadable as exc:
            logger.warning("steering_report_unknown", request_id=request_id,
                           reason="claims_unreadable", error=str(exc))
            return SteeringReport.unknown("claims_unreadable",
                                          changed=self._changed(snapshot, record,
                                                                epoch_at_admission))
        except Exception as exc:  # noqa: BLE001
            logger.warning("steering_report_unknown", request_id=request_id,
                           reason="read_failed", error=str(exc),
                           error_type=type(exc).__name__)
            return SteeringReport.unknown("read_failed")

    @staticmethod
    def _changed(
        snapshot: SteeringSnapshot,
        record: RequestSteeringRecord | None,
        epoch_at_admission: int | None,
    ) -> bool:
        """FR-28.3.6 (T-81): the epoch moved between admission and capture."""
        admitted = record.epoch_at_admission if record is not None else epoch_at_admission
        if admitted is None or snapshot.epoch is None:
            return False
        return admitted != snapshot.epoch

    async def _describe(
        self,
        snapshot: SteeringSnapshot,
        record: RequestSteeringRecord | None,
        epoch_at_admission: int | None,
        engine: str | None,
    ) -> SteeringReport:
        if snapshot.failed:
            logger.warning("steering_report_unknown", reason="read_failed",
                           error=snapshot.error)
            return SteeringReport.unknown("read_failed")
        changed = self._changed(snapshot, record, epoch_at_admission)
        steered = snapshot.steered()
        if not steered:
            return SteeringReport.from_items([SteeringItem(kind="none", changed=changed)])
        if engine == "llamacpp":
            # llama.cpp runs no forward hooks, so an "applied" entry did not apply; nothing here
            # can say what the model ran under beyond that contradiction.
            logger.warning("steering_report_unknown", reason="llamacpp_entries",
                           entries=[(e.sae_id, e.layer) for e in steered])
            return SteeringReport.unknown("llamacpp_entries", changed=changed)

        items: list[SteeringItem] = []
        remaining: list[EntrySnapshot] = []

        # 1. The request record — only when the values the hook read EQUAL what it applied.
        for entry in steered:
            item = self._from_record(entry, record, changed)
            if item is not None:
                items.append(item)
            else:
                remaining.append(entry)

        # 2. Serving circuits: one item per circuit, covering every entry its plan claims.
        if remaining:
            circuit_items, remaining = await self._circuits(remaining, record, changed)
            items.extend(circuit_items)

        # 3. The active profile, by equal values.
        if remaining:
            profile_items, remaining = await self._active_profile(remaining, changed)
            items.extend(profile_items)

        # 4. Everything else: live values nobody claims.
        for entry in remaining:
            items.append(SteeringItem(
                kind="manual", sae=entry.sae_id, layer=entry.layer,
                features=len(entry.applied),
                hash=steering_set_hash(entry.sae_id, entry.applied), changed=changed,
            ))
        return SteeringReport.from_items(items)

    @staticmethod
    def _from_record(
        entry: EntrySnapshot, record: RequestSteeringRecord | None, changed: bool
    ) -> SteeringItem | None:
        if record is None or record.kind not in ("inline", "profile"):
            return None
        if record.sae_id != entry.sae_id or record.layer != entry.layer:
            return None
        if record.applied is None or applied_set(record.applied.items()) != entry.applied:
            return None
        item = SteeringItem(
            kind="inline", sae=entry.sae_id, layer=entry.layer, features=len(entry.applied),
            hash=steering_set_hash(entry.sae_id, entry.applied), clamped=record.clamped,
            changed=changed,
        )
        if record.kind == "inline":
            return item
        _note_profile_mismatch(record.profile_name, record.profile_sae_id, entry)
        return replace(
            item, kind="profile", name=record.profile_name,
            source=record.profile_source or "request",
            intensity=record.intensity if record.intensity is not None else 1.0,
        )

    async def _circuits(
        self,
        entries: list[EntrySnapshot],
        record: RequestSteeringRecord | None,
        changed: bool,
    ) -> tuple[list[SteeringItem], list[EntrySnapshot]]:
        """Every FULL-serving circuit whose plan claims a steered entry.

        ⚠ A STRICT read, not `_steering_circuit()`. That predicate fails OPEN — a database blip
        reads as "no circuit" — which is right for the rung echo's best-effort and wrong here:
        a circuit-steered answer would be labelled `manual`. A failure raises and the report is
        `unknown;reason=read_failed`.
        """
        from millm.api.schemas.circuit import CircuitDefinitionV1
        from millm.db.base import async_session_factory
        from millm.db.repositories.circuit_repository import CircuitRepository
        from millm.ml.circuit_steering import UNSET_INTENSITY, CircuitSteeringEngine
        from millm.services.sae_service import AttachedSAEState

        async with async_session_factory() as session:
            actives = await CircuitRepository(session).list_active()
        full = [c for c in actives if getattr(c, "serving_mode", None) == "full"]
        if not full:
            return [], entries
        engine = CircuitSteeringEngine(AttachedSAEState())
        items: list[SteeringItem] = []
        claimed_keys: set[tuple[str, int]] = set()
        for circuit in sorted(full, key=lambda c: str(c.id)):
            try:
                definition = CircuitDefinitionV1.model_validate(circuit.circuit_meta)
            except Exception:  # noqa: BLE001 - an unparseable circuit steers nothing
                continue
            plan = engine.plan_for(definition, circuit)
            claimed = {(str(e.sae_id), int(e.layer)) for e in plan.claimed_entries}
            mine = [e for e in entries if (e.sae_id, e.layer) in claimed]
            if not mine:
                continue
            claimed_keys.update((e.sae_id, e.layer) for e in mine)
            if record is not None and record.kind == "circuit" and record.intensity is not None:
                lam = float(record.intensity)
            elif plan.intensity is not UNSET_INTENSITY and math.isfinite(plan.intensity):
                lam = float(plan.intensity)
            else:
                lam = float(circuit.intensity)
            items.append(SteeringItem(kind="circuit", circuit_id=str(circuit.id), intensity=lam,
                                      changed=changed))
        if not items:
            return [], entries
        composed = await self._composed_strict()
        if composed:
            items = [SteeringItem(kind="circuit", circuit_id=i.circuit_id, intensity=i.intensity,
                                  composed=True, changed=i.changed) for i in items]
        return items, [e for e in entries if (e.sae_id, e.layer) not in claimed_keys]

    @staticmethod
    async def _composed_strict() -> bool:
        """`_any_layer_composed` without its fail-open: this header is an honesty statement and
        fails CLOSED (FTDD TD11) — the rung echo keeps its own trade-off."""
        try:
            from millm.db.base import async_session_factory
            from millm.services.circuit_claim_registry import CircuitClaimRegistry

            async with async_session_factory() as session:
                claims = await CircuitClaimRegistry(session).live_claims()
            return any(c.composed for c in claims)
        except Exception as exc:  # noqa: BLE001
            raise _ClaimsUnreadable(str(exc)) from exc

    @staticmethod
    async def _active_profile(
        entries: list[EntrySnapshot], changed: bool
    ) -> tuple[list[SteeringItem], list[EntrySnapshot]]:
        from millm.db.base import async_session_factory
        from millm.db.repositories.profile_repository import ProfileRepository

        async with async_session_factory() as session:
            profile = await ProfileRepository(session).get_active()
        if profile is None or not profile.steering:
            return [], entries
        lam = float(profile.intensity) if profile.intensity is not None else 1.0
        scaled = [(int(k), float(v) * lam) for k, v in profile.steering.items()]
        expected = applied_set(scaled)
        items: list[SteeringItem] = []
        rest: list[EntrySnapshot] = []
        for entry in entries:
            if entry.applied == expected:
                _note_profile_mismatch(profile.name, profile.sae_id, entry)
                items.append(SteeringItem(
                    kind="profile", name=profile.name, source="active", intensity=lam,
                    sae=entry.sae_id, layer=entry.layer, features=len(entry.applied),
                    hash=steering_set_hash(entry.sae_id, entry.applied),
                    clamped=count_clamped(scaled), changed=changed,
                ))
            else:
                rest.append(entry)
        return items, rest


def _note_profile_mismatch(name: str | None, profile_sae_id: str | None,
                           entry: EntrySnapshot) -> None:
    """The first-SAE profile defect (FPRD Open Question 1, S3-09) made VISIBLE, not fixed: the
    header names the SAE actually steered, and a disagreement with the profile's own `sae_id` is
    logged."""
    if profile_sae_id and profile_sae_id != entry.sae_id:
        logger.warning("profile_sae_mismatch", profile=name, profile_sae_id=profile_sae_id,
                       applied_sae_id=entry.sae_id, layer=entry.layer)


async def steering_report_for_row(
    snapshot: SteeringSnapshot,
    record: RequestSteeringRecord | None = None,
    *,
    inference: Any = None,
) -> str:
    """The steering value for a finished request (FR-28.3.10) — the SAME string the response
    header and the stream chunk carry, because all three go through `describe` and
    `serialize_steering_header`. Feature 26's batch lines reach it through
    `millm.api.provenance.post_generation`, which reads the report the service published."""
    report = await SteeringStateReader(inference).describe(snapshot, record)
    return report.header
