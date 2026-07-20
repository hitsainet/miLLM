"""
SAE service for orchestrating SAE operations.

This service coordinates between the repository, downloader, loader, and hooker
components to manage SAE lifecycle operations including download, attach, and detach.
"""

import asyncio
import os
import re
import threading
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from millm.api.schemas.circuit import CircuitMember

import structlog
import torch

from millm.core.errors import (
    DownloadCancelledError,
    InsufficientMemoryError,
    ModelNotLoadedError,
    SAEAlreadyAttachedError,
    SAEIncompatibleError,
    SAENotAttachedError,
    SAENotFoundError,
    SAESetIncompleteError,
)
from millm.core.steering_range import STEERING_RANGE, clamp_steering
from millm.db.models.sae import SAE, SAEStatus
from millm.db.repositories.sae_repository import SAERepository
from millm.ml.model_loader import LoadedModelState
from millm.ml.sae_config import SAEConfig
from millm.ml.sae_downloader import SAEDownloader
from millm.ml.sae_hooker import SAEHooker
from millm.ml.sae_loader import SAELoader
from millm.ml.sae_wrapper import LoadedSAE
from millm.sockets.progress import ProgressEmitter

logger = structlog.get_logger()

# Attach-dtype names accepted for the multi-SAE steering weight cache.
_ATTACH_DTYPES: dict[str, "torch.dtype"] = {
    "float16": torch.float16,
    "fp16": torch.float16,
    "half": torch.float16,
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
    "float32": torch.float32,
    "fp32": torch.float32,
}


def _directional_budget(budget: float, sign: int) -> float:
    """Apply the canonical member sign rule (shared with cluster_service).

    A NEGATIVE budget is already directional — the ``sign`` field is redundant
    there and must NOT be multiplied in (doing so double-negates a suppression
    into an amplification). A non-negative budget takes its direction from
    ``sign``. This mirrors ``cluster_service`` so circuits and clusters steer
    identically for the same authored member.
    """
    b = float(budget)
    return b if b < 0 else float(sign) * b


def _resolve_attach_dtype(name: str) -> "torch.dtype":
    """Resolve a configured attach-dtype name to a torch dtype.

    Defaults to fp16 (the measured ~64 MB/SAE footprint) for an unknown name
    rather than raising — a bad config value must not block attachment.
    """
    dtype = _ATTACH_DTYPES.get(str(name).strip().lower())
    if dtype is None:
        logger.warning("unknown_attach_dtype_falling_back_fp16", requested=name)
        return torch.float16
    return dtype


@dataclass
class AttachmentStatus:
    """Current SAE attachment status.

    Back-compat singular view. When several SAEs are attached (Feature 12
    multi-SAE), the singular ``sae_id``/``layer`` reflect the first attached
    entry; callers that need the full set use ``AttachmentStatusSet`` /
    ``SAEService.get_attachment_status_set()``.
    """

    is_attached: bool
    sae_id: Optional[str] = None
    layer: Optional[int] = None
    memory_usage_mb: Optional[int] = None
    steering_enabled: bool = False
    monitoring_enabled: bool = False
    # Diagnostic: number of forward passes in which the steering delta was
    # applied.  Stays 0 if the hook never fires (e.g. compiled graph bypass).
    steering_apply_count: int = 0


@dataclass
class AttachedEntry:
    """One attached SAE, keyed by ``(sae_id, layer)`` in the registry."""

    sae: LoadedSAE
    sae_id: str
    layer: int
    hook_handle: Any


@dataclass
class AttachedEntryStatus:
    """Serializable per-entry attachment status (no live tensors)."""

    sae_id: str
    layer: int
    memory_usage_mb: Optional[int] = None
    steering_enabled: bool = False
    monitoring_enabled: bool = False
    steering_apply_count: int = 0


@dataclass
class AttachmentStatusSet:
    """Plural attachment status across all attached ``(sae_id, layer)`` entries."""

    is_attached: bool
    count: int
    entries: list[AttachedEntryStatus]
    total_memory_usage_mb: Optional[int] = None


@dataclass
class CircuitSteeringResult:
    """Outcome of applying a circuit's members across layers (Feature 12).

    ``applied_per_layer`` maps ``layer -> {feature_idx: effective_strength}``
    actually written to each layer's SAE. ``hazards`` are cross-layer
    compounding/cancellation warnings — SURFACED, never applied.
    ``clamp_warnings`` note members whose ``budget·sign·λ`` exceeded ±200.
    """

    applied_per_layer: dict[int, dict[int, float]]
    hazards: list[dict[str, Any]]
    clamp_warnings: list[str]


@dataclass
class DownloadResult:
    """Result of a download request."""

    sae_id: str
    status: str  # "downloading", "cached", "attached", "already_downloading"
    message: str


@dataclass
class CompatibilityResult:
    """Result of SAE-model compatibility check."""

    compatible: bool
    errors: list[str]
    warnings: list[str]


class AttachedSAEState:
    """
    Singleton managing the currently attached SAE(s).

    This persists SAE attachment state across request boundaries since
    SAEService instances are created per-request via dependency injection.

    Feature 12 (multi-SAE): state is a registry keyed by ``(sae_id, layer)``
    so a cross-layer circuit can attach one SAE per referenced layer, each
    steering through its own decoder. Insertion order is preserved (dict
    ordering). The single-SAE consumers (cluster/monitoring/sensing/health)
    use the back-compat singular properties, which reflect the FIRST attached
    entry — byte-identical behavior when exactly one SAE is attached.

    Thread-safe for access from executor threads.
    """

    _instance: Optional["AttachedSAEState"] = None
    _lock = threading.Lock()

    def __new__(cls) -> "AttachedSAEState":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._entries: dict[tuple[str, int], AttachedEntry] = {}
        return cls._instance

    def _first(self) -> Optional[AttachedEntry]:
        """First attached entry (insertion order), or None. Locked so the
        back-compat singular properties never race a concurrent set/clear."""
        with self._lock:
            return next(iter(self._entries.values()), None)

    @property
    def attached_sae(self) -> Optional[LoadedSAE]:
        """Get the first attached SAE (back-compat singular view)."""
        entry = self._first()
        return entry.sae if entry else None

    @property
    def attached_sae_id(self) -> Optional[str]:
        """Get the ID of the first attached SAE (back-compat singular view)."""
        entry = self._first()
        return entry.sae_id if entry else None

    @property
    def attached_layer(self) -> Optional[int]:
        """Get the layer of the first attached SAE (back-compat singular view)."""
        entry = self._first()
        return entry.layer if entry else None

    @property
    def hook_handle(self) -> Optional[Any]:
        """Get the hook handle of the first attached SAE (back-compat)."""
        entry = self._first()
        return entry.hook_handle if entry else None

    @property
    def is_attached(self) -> bool:
        """Check if at least one SAE is currently attached."""
        with self._lock:
            return bool(self._entries)

    @property
    def count(self) -> int:
        """Number of attached ``(sae_id, layer)`` entries."""
        with self._lock:
            return len(self._entries)

    def entries(self) -> list[AttachedEntry]:
        """Snapshot list of all attached entries, in insertion order."""
        with self._lock:
            return list(self._entries.values())

    def get(self, sae_id: str, layer: int) -> Optional[AttachedEntry]:
        """Fetch the entry attached at ``(sae_id, layer)``, or None."""
        with self._lock:
            return self._entries.get((sae_id, int(layer)))

    def by_layer(self, layer: int) -> Optional[AttachedEntry]:
        """Fetch the unique entry attached at ``layer``.

        Returns None when no SAE — or *more than one* SAE — is attached to
        that layer, so a caller can never silently pick the wrong basis when
        the layer is ambiguous. The scan runs under the registry lock so a
        concurrent attach/detach cannot raise "dict changed size" here.
        """
        layer = int(layer)
        with self._lock:
            matches = [e for e in self._entries.values() if e.layer == layer]
        return matches[0] if len(matches) == 1 else None

    def set(
        self,
        sae: LoadedSAE,
        sae_id: str,
        layer: int,
        hook_handle: Any,
    ) -> None:
        """Attach (or re-attach) the SAE at ``(sae_id, layer)``.

        Removes any existing hook for that SAME key before overwriting so a
        re-attach — or a race that bypasses the is_attached guard — never
        leaves an orphaned hook firing on every forward pass. Other keys are
        untouched.
        """
        key = (sae_id, int(layer))
        with self._lock:
            existing = self._entries.get(key)
            if existing is not None and existing.hook_handle is not None:
                try:
                    existing.hook_handle.remove()
                    logger.warning(
                        "orphaned_hook_removed_before_overwrite",
                        previous_sae_id=sae_id,
                        layer=int(layer),
                    )
                except Exception as e:
                    logger.warning("error_removing_orphaned_hook", error=str(e))
            self._entries[key] = AttachedEntry(
                sae=sae, sae_id=sae_id, layer=int(layer), hook_handle=hook_handle
            )

    def clear(self, sae_id: Optional[str] = None, layer: Optional[int] = None) -> None:
        """Detach attached SAEs.

        With no arguments, detaches ALL entries (back-compat with the
        single-SAE ``clear()``). With both ``sae_id`` and ``layer``, detaches
        only that one entry. Each removed entry's hook is removed first.
        """
        with self._lock:
            if sae_id is not None and layer is not None:
                keys = [(sae_id, int(layer))]
            elif sae_id is not None or layer is not None:
                # Partial key: match all entries with that sae_id OR layer.
                keys = [
                    k
                    for k in self._entries
                    if (sae_id is None or k[0] == sae_id)
                    and (layer is None or k[1] == int(layer))
                ]
            else:
                keys = list(self._entries.keys())
            for key in keys:
                entry = self._entries.pop(key, None)
                if entry is not None and entry.hook_handle is not None:
                    try:
                        entry.hook_handle.remove()
                    except Exception as e:
                        logger.warning("error_removing_hook", error=str(e))


class SAEService:
    """
    Orchestration layer for SAE operations.

    Coordinates between repository, downloader, loader, and hooker components.
    Manages SAE lifecycle: download, attach, steer, monitor, detach.

    Thread Safety:
        Uses _attachment_lock for state mutations during attach/detach.
        Forward pass through SAE is thread-safe.
    """

    def __init__(
        self,
        repository: SAERepository,
        cache_dir: str,
        emitter: Optional[ProgressEmitter] = None,
        inference_service: Optional[Any] = None,
    ) -> None:
        """
        Initialize the SAE service.

        Args:
            repository: SAE database repository.
            cache_dir: Directory for SAE cache.
            emitter: Progress event emitter for WebSocket updates.
            inference_service: Optional InferenceService ref used during detach
                to wait for in-flight requests.  When provided, avoids the
                import of millm.api.dependencies (layer violation).
        """
        self.repository = repository
        self.emitter = emitter
        self._inference_service = inference_service

        # Initialize components
        self._downloader = SAEDownloader(cache_dir)
        self._loader = SAELoader()
        self._hooker = SAEHooker()

        # Attachment state singleton (persists across requests)
        self._sae_state = AttachedSAEState()

        # Track active downloads for cancellation
        self._active_downloads: dict[str, asyncio.Task] = {}
        self._cancelled_downloads: set[str] = set()

        logger.debug("SAEService initialized", cache_dir=cache_dir)

    # =========================================================================
    # Listing Methods
    # =========================================================================

    async def list_saes(self) -> list[SAE]:
        """
        Get all SAEs from the database.

        Returns:
            List of all SAEs ordered by creation date descending.
        """
        return await self.repository.get_all()

    async def get_sae(self, sae_id: str) -> SAE:
        """
        Get a single SAE by ID.

        Args:
            sae_id: The SAE's database ID.

        Returns:
            The SAE if found.

        Raises:
            SAENotFoundError: If SAE doesn't exist.
        """
        sae = await self.repository.get(sae_id)
        if not sae:
            raise SAENotFoundError(
                f"SAE with ID '{sae_id}' not found",
                details={"sae_id": sae_id},
            )
        return sae

    def get_attachment_status(self) -> AttachmentStatus:
        """
        Get current SAE attachment status.

        Returns:
            AttachmentStatus with current state.
        """
        if not self._sae_state.is_attached:
            return AttachmentStatus(is_attached=False)

        sae = self._sae_state.attached_sae
        return AttachmentStatus(
            is_attached=True,
            sae_id=self._sae_state.attached_sae_id,
            layer=self._sae_state.attached_layer,
            memory_usage_mb=int(sae.estimate_memory_mb()) if sae else None,
            steering_enabled=sae.is_steering_enabled if sae else False,
            monitoring_enabled=sae.is_monitoring_enabled if sae else False,
            steering_apply_count=sae.steering_apply_count if sae else 0,
        )

    def get_attachment_status_set(self) -> AttachmentStatusSet:
        """
        Get the plural attachment status across all attached ``(sae_id, layer)``
        entries (Feature 12 multi-SAE).

        Returns:
            AttachmentStatusSet with one AttachedEntryStatus per attached SAE
            and the summed memory footprint.
        """
        entries = self._sae_state.entries()
        if not entries:
            return AttachmentStatusSet(
                is_attached=False, count=0, entries=[], total_memory_usage_mb=None
            )
        statuses: list[AttachedEntryStatus] = []
        total_mb = 0
        for entry in entries:
            sae = entry.sae
            mb = int(sae.estimate_memory_mb()) if sae else None
            if mb is not None:
                total_mb += mb
            statuses.append(
                AttachedEntryStatus(
                    sae_id=entry.sae_id,
                    layer=entry.layer,
                    memory_usage_mb=mb,
                    steering_enabled=sae.is_steering_enabled if sae else False,
                    monitoring_enabled=sae.is_monitoring_enabled if sae else False,
                    steering_apply_count=sae.steering_apply_count if sae else 0,
                )
            )
        return AttachmentStatusSet(
            is_attached=True,
            count=len(statuses),
            entries=statuses,
            total_memory_usage_mb=total_mb,
        )

    # =========================================================================
    # Circuit serving (Feature 12): apply members across layers under one λ
    # =========================================================================

    def set_circuit_steering(
        self,
        members: list["CircuitMember"],
        intensity: float,
        *,
        edges: Optional[list[dict[str, Any]]] = None,
    ) -> CircuitSteeringResult:
        """Serve a circuit: apply every member through ITS OWN layer's SAE at
        the member's frozen budget scaled by one global intensity (λ).

        Each member's layer must resolve to exactly one attached SAE (via
        ``by_layer``); otherwise ``SAESetIncompleteError`` (422) is raised with
        the full offender list and NOTHING is applied — a member is never
        steered through a mismatched SAE. Effective strengths are clamped
        through the shared ±200 gate. Cross-layer hazards are detected and
        returned but never applied (detection, not correction).

        Args:
            members: circuit members to serve.
            intensity: global λ (dial); typically clamped to [0, 2] upstream.
            edges: optional circuit edges carrying validated effect sizes /
                weight priors, used to quantify hazards (Feature 13 supplies
                these; absent here means heuristic sign-only hazards).

        Returns:
            CircuitSteeringResult with per-layer applied strengths, hazards,
            and clamp warnings.
        """
        from millm.core.config import settings

        # 0. Clamp the global λ to the configured circuit-intensity envelope so a
        #    rogue value can neither invert the circuit (negative) nor blow past
        #    the dial ceiling. The apply-time ±200 clamp is a separate gate.
        lo = float(settings.CIRCUIT_INTENSITY_MIN)
        hi = float(settings.CIRCUIT_INTENSITY_MAX)
        if lo > hi:
            # Misconfigured envelope — fall back to the safe [0, 2] default
            # rather than silently pinning every serve to `lo`.
            logger.warning("circuit_intensity_bounds_inverted", min=lo, max=hi)
            lo, hi = 0.0, 2.0
        intensity = max(lo, min(hi, float(intensity)))

        # 1. Resolve every member's layer to a UNIQUE attached SAE ONCE, under a
        #    single consistent snapshot, and collect all offenders first so the
        #    error reports the complete set (fail-closed) — never re-resolve
        #    by_layer in the apply loop (that would be a TOCTOU wrong-basis risk).
        # An empty member set means the circuit is OFF — clear + disable every
        # attached layer rather than silently leaving the PREVIOUS circuit armed.
        if not members:
            cleared = self.clear_circuit_steering()
            logger.info("circuit_steering_cleared_empty_members", layers=cleared)
            return CircuitSteeringResult(
                applied_per_layer={}, hazards=[], clamp_warnings=[]
            )

        # Resolution cache keyed by the RESOLUTION IDENTITY (declared sae_id +
        # layer), NOT by layer alone: two SAEs may be attached on one layer, and
        # keying by layer would let the first member's SAE capture every later
        # member on that layer (a wrong-basis serve).
        resolved_cache: dict[tuple[Optional[str], int], "AttachedEntry"] = {}
        validated: list[tuple["CircuitMember", "AttachedEntry"]] = []
        offenders: list[dict[str, Any]] = []
        substitutions: list[str] = []
        seen_members: set[tuple[int, int]] = set()
        for m in members:
            # Duplicate (layer, feature_idx) members would silently last-write-
            # win — reject them instead.
            mkey = (m.layer, m.feature_idx)
            if mkey in seen_members:
                offenders.append(
                    {
                        "feature_idx": m.feature_idx,
                        "layer": m.layer,
                        "reason": "duplicate_member",
                    }
                )
                continue
            seen_members.add(mkey)

            ckey = (m.sae_id or None, m.layer)
            entry = resolved_cache.get(ckey)
            if entry is None:
                # Prefer an exact (sae_id, layer) match when the member names
                # its SAE — this disambiguates a layer with two attached SAEs
                # AND serves the member through the basis it was authored
                # against.
                if m.sae_id:
                    entry = self._sae_state.get(m.sae_id, m.layer)
                    if entry is None:
                        # The declared SAE is not attached at this layer. Fall
                        # back to the layer's unique SAE if there is one, but
                        # record the substitution — a different SAE means a
                        # different feature basis, which the caller must see.
                        entry = self._sae_state.by_layer(m.layer)
                        if entry is not None:
                            substitutions.append(
                                f"feature {m.feature_idx}@L{m.layer}: declared SAE "
                                f"'{m.sae_id}' not attached; served through "
                                f"'{entry.sae_id}' (different feature basis)"
                            )
                else:
                    entry = self._sae_state.by_layer(m.layer)
            if entry is None:
                offenders.append(
                    {
                        "feature_idx": m.feature_idx,
                        "layer": m.layer,
                        "sae_id": m.sae_id,
                        "reason": "missing_or_ambiguous_sae",
                    }
                )
                continue
            resolved_cache[ckey] = entry
            if not (0 <= m.feature_idx < entry.sae.d_sae):
                offenders.append(
                    {
                        "feature_idx": m.feature_idx,
                        "layer": m.layer,
                        "sae_id": entry.sae_id,
                        "reason": "index_out_of_bounds",
                        "d_sae": entry.sae.d_sae,
                    }
                )
                continue
            validated.append((m, entry))
        if offenders:
            raise SAESetIncompleteError(offenders)

        # 2. Group by the RESOLVED SAE (sae_id, layer) so two SAEs on one layer
        #    each get their own batch. Compute effective (clamped) strengths
        #    under one λ; γ=0 ⇒ B = B_dir. Iterate the VALIDATED list so the
        #    dedup/bounds guarantees are carried in the data, not in a
        #    raise-before-reach argument.
        per_entry: dict[tuple[str, int], dict[int, float]] = {}
        entry_by_key: dict[tuple[str, int], "AttachedEntry"] = {}
        clamp_warnings: list[str] = []
        for m, entry in validated:
            raw = _directional_budget(m.budget, m.sign) * intensity
            eff = clamp_steering(raw)
            if abs(raw) > STEERING_RANGE:
                clamp_warnings.append(
                    f"feature {m.feature_idx}@L{m.layer} clamped "
                    f"{raw:+.3g}→{eff:+.3g} (±{STEERING_RANGE:g})"
                )
            key = (entry.sae_id, entry.layer)
            entry_by_key[key] = entry
            per_entry.setdefault(key, {})[m.feature_idx] = eff

        # 3. Apply each group through ITS OWN resolved SAE (never a second
        #    by_layer call). Clear that SAE's prior steering first so each serve
        #    is authoritative and no stale features from a previous
        #    circuit/cluster/manual set leak in. At λ=0 the circuit is OFF —
        #    clear and leave steering disabled rather than reporting N features
        #    "active" at zero strength.
        disabled = intensity == 0 or all(
            v == 0 for s in per_entry.values() for v in s.values()
        )
        for key, steering in per_entry.items():
            sae = entry_by_key[key].sae
            sae.clear_steering()
            if disabled:
                sae.enable_steering(False)
            else:
                sae.set_steering_batch(steering)  # bounds already gated in step 1
                sae.enable_steering(True)

        # Report per-LAYER for the caller-facing result (merging any same-layer
        # groups), preserving the documented applied_per_layer shape.
        per_layer: dict[int, dict[int, float]] = {}
        for (_sid, layer), steering in per_entry.items():
            per_layer.setdefault(layer, {}).update(steering)

        hazards = self._cross_layer_hazards(members, intensity, edges=edges)
        if substitutions:
            for note in substitutions:
                logger.warning("circuit_member_sae_substituted", detail=note)
            clamp_warnings.extend(substitutions)
        logger.info(
            "circuit_steering_applied",
            layers=sorted(per_layer.keys()),
            member_count=len(members),
            intensity=intensity,
            hazard_count=len(hazards),
            clamped=len(clamp_warnings),
        )
        return CircuitSteeringResult(
            applied_per_layer=per_layer,
            hazards=hazards,
            clamp_warnings=clamp_warnings,
        )

    def clear_circuit_steering(self, layers: Optional[list[int]] = None) -> list[int]:
        """Clear steering on every participating layer (or all attached layers).

        Returns the list of layers actually cleared.
        """
        if layers is None:
            targets = [e.layer for e in self._sae_state.entries()]
        else:
            targets = list(dict.fromkeys(int(l) for l in layers))
        cleared: list[int] = []
        for layer in targets:
            entry = self._sae_state.by_layer(layer)
            if entry is not None:
                entry.sae.clear_steering()
                entry.sae.enable_steering(False)
                cleared.append(layer)
        return cleared

    def _cross_layer_hazards(
        self,
        members: list["CircuitMember"],
        intensity: float,
        *,
        edges: Optional[list[dict[str, Any]]] = None,
    ) -> list[dict[str, Any]]:
        """Detect cross-layer compounding/cancellation hazards.

        For every upstream→downstream feature-member pair (up.layer <
        down.layer) that is co-steered, emit a hazard: same effective sign ⇒
        compounding, opposite ⇒ cancellation. When a matching circuit edge at
        rung ≥ 2 with an effect_size is supplied, the hazard is QUANTIFIED and
        labeled ``validated:ES=…`` (a validated negative edge flips
        compounding↔cancellation); otherwise it is labeled ``heuristic`` from
        the edge weight_prior if present, else a plain sign heuristic. Warnings
        are SURFACED only — the steering config is never mutated here.
        """
        if intensity == 0:
            return []
        # Index edges by (up_layer, up_idx, down_layer, down_idx) for lookup.
        edge_index: dict[tuple, dict[str, Any]] = {}
        for e in edges or []:
            try:
                up = e["up"]
                down = e["down"]
                key = (
                    int(up["layer"]),
                    int(up["feature_idx"]),
                    int(down["layer"]),
                    int(down["feature_idx"]),
                )
            except (KeyError, TypeError, ValueError):
                continue
            edge_index[key] = e

        hazards: list[dict[str, Any]] = []
        seen: set[tuple] = set()
        for up in members:
            for down in members:
                if up.layer >= down.layer:
                    continue
                pair = (up.layer, up.feature_idx, down.layer, down.feature_idx)
                if pair in seen:
                    continue
                seen.add(pair)

                up_sign = 1 if _directional_budget(up.budget, up.sign) >= 0 else -1
                down_sign = 1 if _directional_budget(down.budget, down.sign) >= 0 else -1
                same_sign = up_sign == down_sign

                edge = edge_index.get(pair)
                es = None
                weight_prior = None
                rung = 0
                if edge is not None:
                    es = edge.get("effect_size")
                    weight_prior = edge.get("weight_prior")
                    rung = int(edge.get("rung", 0) or 0)

                if es is not None and rung >= 2:
                    # A validated NEGATIVE edge flips the interaction sense.
                    effective_same = same_sign if es >= 0 else (not same_sign)
                    label = f"validated:ES={float(es):.3g}"
                    hazard_rung = rung
                elif weight_prior is not None:
                    effective_same = same_sign
                    label = f"heuristic:weight_prior={float(weight_prior):.3g}"
                    hazard_rung = rung
                else:
                    effective_same = same_sign
                    label = "heuristic:co-steer-sign"
                    hazard_rung = 0

                kind = "compounding" if effective_same else "cancellation"
                hazards.append(
                    {
                        "kind": kind,
                        "up": {"layer": up.layer, "feature_idx": up.feature_idx},
                        "down": {"layer": down.layer, "feature_idx": down.feature_idx},
                        "label": label,
                        "rung": hazard_rung,
                    }
                )
        return hazards

    async def preview_repository(
        self,
        repository_id: str,
        revision: str = "main",
        token: str | None = None,
    ) -> dict:
        """
        Preview SAE files in a HuggingFace repository without downloading.

        Args:
            repository_id: HuggingFace repo (e.g., "google/gemma-scope-2b-pt-res").
            revision: Git revision (branch, tag, commit).
            token: HuggingFace access token for gated repositories.

        Returns:
            Dictionary with repository info and available SAE files.
        """
        return await self._downloader.list_repository_files(repository_id, revision, token)

    # =========================================================================
    # Download Methods
    # =========================================================================

    async def start_download(
        self,
        repository_id: str,
        revision: str = "main",
        file_path: str | None = None,
        token: str | None = None,
    ) -> DownloadResult:
        """
        Start downloading an SAE from HuggingFace.

        Creates a database record and starts the download asynchronously.

        Args:
            repository_id: HuggingFace repo (e.g., "jbloom/gemma-2-2b-res-jb").
            revision: Git revision (branch, tag, commit).
            file_path: Specific SAE file to download (e.g., "layer_12/width_16k/average_l0_50/params.npz").
                       If provided, only downloads that specific SAE directory.

        Returns:
            DownloadResult with SAE ID, status, and message.

        Raises:
            ValueError: If SAE already exists with same repo/revision.
        """
        # Generate SAE ID first (includes file_path for uniqueness)
        sae_id = self._downloader.generate_sae_id(repository_id, revision, file_path)

        # Check for existing SAE with this specific ID
        existing = await self.repository.get(sae_id)
        if existing:
            if existing.status == SAEStatus.CACHED:
                logger.info(
                    "sae_already_cached",
                    sae_id=existing.id,
                    repository_id=repository_id,
                )
                return DownloadResult(
                    sae_id=existing.id,
                    status="cached",
                    message="SAE is already downloaded and cached",
                )
            elif existing.status == SAEStatus.ATTACHED:
                logger.info(
                    "sae_already_attached",
                    sae_id=existing.id,
                    repository_id=repository_id,
                )
                return DownloadResult(
                    sae_id=existing.id,
                    status="attached",
                    message="SAE is already downloaded and attached to model",
                )
            elif existing.status == SAEStatus.DOWNLOADING:
                logger.info(
                    "sae_already_downloading",
                    sae_id=existing.id,
                    repository_id=repository_id,
                )
                return DownloadResult(
                    sae_id=existing.id,
                    status="already_downloading",
                    message="SAE download is already in progress",
                )
            elif existing.status == SAEStatus.ERROR:
                # Delete the failed SAE and retry download
                logger.info(
                    "sae_retrying_failed_download",
                    sae_id=existing.id,
                    repository_id=repository_id,
                )
                await self.repository.delete(existing.id)

        # Create database record in downloading state
        await self.repository.create_downloading(
            sae_id=sae_id,
            repository_id=repository_id,
            revision=revision,
            cache_path="",  # Updated after download
        )

        logger.info(
            "sae_download_started",
            sae_id=sae_id,
            repository_id=repository_id,
            revision=revision,
        )

        # Start background download and track it
        task = asyncio.create_task(self._download_task(sae_id, repository_id, revision, file_path, token))
        self._active_downloads[sae_id] = task

        return DownloadResult(
            sae_id=sae_id,
            status="downloading",
            message=f"Download started for {repository_id}",
        )

    async def _download_task(
        self,
        sae_id: str,
        repository_id: str,
        revision: str,
        file_path: str | None = None,
        token: str | None = None,
    ) -> None:
        """
        Background task for downloading SAE.

        Updates database on completion or error.
        """
        try:
            # Check if cancelled before starting
            if sae_id in self._cancelled_downloads:
                self._cancelled_downloads.discard(sae_id)
                raise DownloadCancelledError("Download was cancelled")

            # Download SAE
            cache_path = await self._downloader.download(
                repository_id=repository_id,
                revision=revision,
                file_path=file_path,
                progress_callback=self._make_progress_callback(sae_id),
                token=token,
            )

            # Check if cancelled after download
            if sae_id in self._cancelled_downloads:
                self._cancelled_downloads.discard(sae_id)
                raise DownloadCancelledError("Download was cancelled")

            # When downloading a specific file, the actual SAE is in a subdirectory
            # e.g., file_path="layer_20/width_16k/average_l0_71/params.npz"
            # cache_path is the root snapshot, but SAE files are in the subdirectory
            if file_path:
                # Extract directory from file path (e.g., "layer_20/width_16k/average_l0_71")
                sae_subdir = os.path.dirname(file_path)
                if sae_subdir:
                    cache_path = os.path.join(cache_path, sae_subdir)
                    logger.debug(
                        "sae_download_adjusted_path",
                        sae_id=sae_id,
                        file_path=file_path,
                        adjusted_cache_path=cache_path,
                    )

            # Load config to get dimensions
            config = self._loader.load_config(cache_path)

            # Calculate file size
            file_size = sum(
                os.path.getsize(os.path.join(cache_path, f))
                for f in os.listdir(cache_path)
                if os.path.isfile(os.path.join(cache_path, f))
            )

            # Extract width and average_l0 from file_path
            width = None
            average_l0 = None
            if file_path:
                width, average_l0 = self._parse_sae_path_metadata(file_path)

            # Update database with downloaded info
            await self.repository.update_downloaded(
                sae_id=sae_id,
                cache_path=cache_path,
                d_in=config.d_in,
                d_sae=config.d_sae,
                trained_on=config.model_name,
                trained_layer=config.hook_layer,
                file_size_bytes=file_size,
                width=width,
                average_l0=average_l0,
            )

            logger.info(
                "sae_download_complete",
                sae_id=sae_id,
                cache_path=cache_path,
                d_in=config.d_in,
                d_sae=config.d_sae,
            )

            # Emit completion event
            if self.emitter:
                await self.emitter.emit_sae_download_complete(sae_id=sae_id)

        except DownloadCancelledError:
            logger.info("sae_download_cancelled", sae_id=sae_id)
            await self.repository.update_status(
                sae_id=sae_id,
                status=SAEStatus.ERROR,
                error_message="Download cancelled by user",
            )
            if self.emitter:
                await self.emitter.emit_sae_download_error(
                    sae_id=sae_id,
                    error="Download cancelled by user",
                )

        except asyncio.CancelledError:
            logger.info("sae_download_cancelled", sae_id=sae_id)
            await self.repository.update_status(
                sae_id=sae_id,
                status=SAEStatus.ERROR,
                error_message="Download cancelled by user",
            )
            if self.emitter:
                await self.emitter.emit_sae_download_error(
                    sae_id=sae_id,
                    error="Download cancelled by user",
                )

        except Exception as e:
            logger.error(
                "sae_download_failed",
                sae_id=sae_id,
                error=str(e),
            )

            await self.repository.update_status(
                sae_id=sae_id,
                status=SAEStatus.ERROR,
                error_message=str(e),
            )

            if self.emitter:
                await self.emitter.emit_sae_download_error(
                    sae_id=sae_id,
                    error=str(e),
                )

        finally:
            # Clean up tracking
            self._active_downloads.pop(sae_id, None)
            self._cancelled_downloads.discard(sae_id)

    def _parse_sae_path_metadata(self, file_path: str) -> tuple[str | None, int | None]:
        """
        Extract width and average_l0 from SAE file path.

        Args:
            file_path: Path like "layer_20/width_16k/average_l0_38/params.npz"

        Returns:
            Tuple of (width, average_l0). E.g., ("16k", 38)
        """
        width = None
        average_l0 = None

        # Match width pattern: width_16k, width_65k, etc.
        width_match = re.search(r"width[_-]?(\d+k?)", file_path, re.IGNORECASE)
        if width_match:
            width = width_match.group(1)

        # Match average_l0 pattern: average_l0_38, l0_38, etc.
        l0_match = re.search(r"(?:average_)?l0[_-]?(\d+)", file_path, re.IGNORECASE)
        if l0_match:
            average_l0 = int(l0_match.group(1))

        return width, average_l0

    def _make_progress_callback(self, sae_id: str):
        """
        Create progress callback for download.

        The callback may be called from a thread executor, so we use
        run_coroutine_threadsafe to safely schedule async operations.
        """
        # Capture the event loop at callback creation time (when we're in async context)
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        def callback(progress: dict[str, Any]) -> None:
            if self.emitter and progress.get("status") == "downloading":
                coro = self.emitter.emit_sae_download_progress(
                    sae_id=sae_id,
                    percent=progress.get("percent", 0),
                )
                if loop and loop.is_running():
                    # Schedule from thread to main event loop
                    asyncio.run_coroutine_threadsafe(coro, loop)
                else:
                    # Fallback: try to create task if we're in async context
                    try:
                        asyncio.create_task(coro)
                    except RuntimeError:
                        # No event loop available, skip progress emission
                        logger.debug(
                            "skipping_progress_emit",
                            reason="no_event_loop",
                            sae_id=sae_id,
                        )
        return callback

    async def cancel_download(self, sae_id: str) -> SAE:
        """
        Cancel an in-progress SAE download.

        Args:
            sae_id: The SAE's ID.

        Returns:
            The SAE with updated status.

        Raises:
            SAENotFoundError: If SAE doesn't exist.
        """
        sae = await self.get_sae(sae_id)

        # Only cancel if actually downloading
        if sae.status != SAEStatus.DOWNLOADING:
            return sae

        # Mark as cancelled
        self._cancelled_downloads.add(sae_id)

        # Cancel the task if it exists
        task = self._active_downloads.get(sae_id)
        if task and not task.done():
            task.cancel()

        # Update database status
        await self.repository.update_status(
            sae_id=sae_id,
            status=SAEStatus.ERROR,
            error_message="Download cancelled by user",
        )

        logger.info("sae_download_cancelled", sae_id=sae_id)

        # Return updated SAE
        return await self.get_sae(sae_id)

    # =========================================================================
    # Compatibility Methods
    # =========================================================================

    async def check_compatibility(
        self,
        sae_id: str,
        layer: int,
    ) -> CompatibilityResult:
        """
        Check if SAE is compatible with currently loaded model.

        Args:
            sae_id: The SAE's ID.
            layer: Target layer to attach.

        Returns:
            CompatibilityResult with compatibility status and any issues.

        Raises:
            SAENotFoundError: If SAE doesn't exist.
            ModelNotLoadedError: If no model is loaded.
        """
        sae = await self.get_sae(sae_id)

        # Check model is loaded
        model_state = LoadedModelState()
        if not model_state.is_loaded:
            raise ModelNotLoadedError(
                "No model loaded. Load a model before checking SAE compatibility.",
            )

        errors: list[str] = []
        warnings: list[str] = []

        # Check SAE status
        if sae.status != SAEStatus.CACHED:
            errors.append(f"SAE is not ready (status: {sae.status.value})")

        # Check dimension compatibility
        model = model_state.current.model
        if hasattr(model, "config"):
            hidden_size = getattr(model.config, "hidden_size", None)
            if hidden_size and sae.d_in != hidden_size:
                errors.append(
                    f"Dimension mismatch: SAE d_in={sae.d_in}, "
                    f"model hidden_size={hidden_size}"
                )

        # Check layer range
        try:
            num_layers = self._hooker.get_layer_count(model)
            if not 0 <= layer < num_layers:
                errors.append(
                    f"Layer {layer} out of range [0, {num_layers-1}]"
                )
        except ValueError as e:
            warnings.append(f"Could not verify layer count: {e}")

        # Check trained layer match
        if sae.trained_layer is not None and sae.trained_layer != layer:
            warnings.append(
                f"SAE was trained on layer {sae.trained_layer}, "
                f"but attaching to layer {layer}"
            )

        # Check trained model match
        if sae.trained_on:
            model_name = getattr(model.config, "name_or_path", "") if hasattr(model, "config") else ""
            if model_name and sae.trained_on not in model_name and model_name not in sae.trained_on:
                warnings.append(
                    f"SAE was trained on '{sae.trained_on}', "
                    f"current model is '{model_name}'"
                )

        return CompatibilityResult(
            compatible=len(errors) == 0,
            errors=errors,
            warnings=warnings,
        )

    # =========================================================================
    # Attachment Methods
    # =========================================================================

    async def attach_sae(
        self,
        sae_id: str,
        layer: int,
    ) -> dict[str, Any]:
        """
        Attach an SAE to the loaded model.

        Args:
            sae_id: The SAE's ID.
            layer: Layer to attach SAE to.

        Returns:
            Dict with attachment status and memory info.

        Raises:
            SAENotFoundError: If SAE doesn't exist.
            SAEAlreadyAttachedError: If an SAE is already attached.
            SAEIncompatibleError: If SAE is incompatible with model.
            ModelNotLoadedError: If no model is loaded.
        """
        sae = await self.get_sae(sae_id)

        # Check model is loaded
        model_state = LoadedModelState()
        if not model_state.is_loaded:
            raise ModelNotLoadedError(
                "No model loaded. Load a model before attaching SAE.",
            )

        # Reject only re-attaching THIS exact (sae_id, layer). Multi-SAE
        # circuit serving (attach_set) legitimately populates the registry with
        # SAEs on other layers; the single-attach path must not treat "some SAE
        # is attached somewhere" as "already attached" (that wrongly blocked a
        # standalone attach on a different layer once a circuit was served).
        if self._sae_state.get(sae_id, layer) is not None:
            raise SAEAlreadyAttachedError(
                f"SAE '{sae_id}' is already attached at layer {layer}. "
                "Detach it first before re-attaching.",
                details={"sae_id": sae_id, "layer": layer},
            )

        # Check compatibility
        compat = await self.check_compatibility(sae_id, layer)
        if not compat.compatible:
            raise SAEIncompatibleError(
                f"SAE incompatible with model: {compat.errors[0]}",
                details={"errors": compat.errors, "warnings": compat.warnings},
            )

        # Log warnings
        for warning in compat.warnings:
            logger.warning("sae_compatibility_warning", warning=warning)

        # Check available GPU memory before loading
        if torch.cuda.is_available():
            free_mb = torch.cuda.mem_get_info()[0] / (1024 * 1024)
            estimated_mb = (sae.file_size_bytes or 0) / (1024 * 1024) * 1.2  # 20% overhead
            if estimated_mb > 0 and estimated_mb > free_mb:
                logger.warning(
                    "sae_memory_warning",
                    estimated_mb=int(estimated_mb),
                    available_mb=int(free_mb),
                )

        # Load SAE weights
        logger.info("loading_sae", sae_id=sae_id, cache_path=sae.cache_path)
        loaded_sae = self._loader.load(
            cache_path=sae.cache_path,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )

        # Match SAE dtype to model dtype to avoid per-forward-pass casts
        model = model_state.current.model
        model_dtype = getattr(model, "dtype", None)
        if model_dtype is None and hasattr(model, "config"):
            model_dtype = getattr(model.config, "torch_dtype", None)
        if model_dtype is not None and loaded_sae.W_enc.dtype != model_dtype:
            logger.info(
                "sae_dtype_cast",
                from_dtype=str(loaded_sae.W_enc.dtype),
                to_dtype=str(model_dtype),
            )
            loaded_sae.W_enc = loaded_sae.W_enc.to(model_dtype)
            loaded_sae.b_enc = loaded_sae.b_enc.to(model_dtype)
            loaded_sae.W_dec = loaded_sae.W_dec.to(model_dtype)
            loaded_sae.b_dec = loaded_sae.b_dec.to(model_dtype)

        # Install hook
        model = model_state.current.model
        handle = self._hooker.install(model, layer, loaded_sae)

        # Defense-in-depth against torch.compile bypassing the freshly-installed
        # hook.  model_loader sets skip_nnmodule_hook_guards=False before
        # compiling, which should already invalidate the cached graph on hook
        # registration; resetting Dynamo here guarantees the next forward pass
        # re-traces with the hook present even if that config was not applied
        # (e.g. compile disabled, or a future torch version changes semantics).
        self._reset_dynamo_for_hook_change()

        # Update state in singleton
        self._sae_state.set(loaded_sae, sae_id, layer, handle)

        # Update database
        await self.repository.update_status(sae_id, SAEStatus.ATTACHED)
        await self.repository.create_attachment(
            sae_id=sae_id,
            model_id=model_state.loaded_model_id,
            layer=layer,
            memory_usage_mb=int(loaded_sae.estimate_memory_mb()),
        )

        memory_mb = int(loaded_sae.estimate_memory_mb())

        # Auto-lock the model for steering
        try:
            from millm.db.base import async_session_factory
            from millm.db.repositories.model_repository import ModelRepository

            async with async_session_factory() as session:
                model_repo = ModelRepository(session)
                await model_repo.update(model_state.loaded_model_id, locked=True)
                logger.info("model_auto_locked", model_id=model_state.loaded_model_id, sae_id=sae_id)
        except Exception as e:
            logger.warning("model_auto_lock_failed", error=str(e))

        # Sensing lifecycle (011 R1): re-arm when the ACTIVE profile has
        # sensing enabled — activate-then-attach (and detach/re-attach
        # cycles) previously left sensing silently dark while the UI toggle
        # showed it on. Best-effort: never fails the attach.
        try:
            import millm.api.dependencies as deps
            from millm.db.base import async_session_factory
            from millm.db.repositories.profile_repository import ProfileRepository

            async with async_session_factory() as session:
                active = await ProfileRepository(session).get_active()
            if (active is not None
                    and getattr(active, "source_kind", None) == "cluster"
                    and bool(getattr(active, "sensing_enabled", False))):
                try:
                    deps.get_sensing_service().arm_for_profile(
                        active, self._sae_state.attached_sae
                    )
                    logger.info("sensing_rearmed_on_attach",
                                profile_id=active.id)
                except ValueError as arm_error:
                    # Surface the refusal in the attach response (011 R3:
                    # log-only left the toggle on with sensing silently
                    # dark — the same class R2 fixed for activation).
                    compat.warnings.append(
                        f"sensing enabled for '{active.name}' but could not "
                        f"arm: {arm_error}"
                    )
                    logger.warning("sensing_rearm_on_attach_refused",
                                   profile_id=active.id, error=str(arm_error))
        except Exception as e:
            logger.warning("sensing_rearm_on_attach_failed", error=str(e))

        logger.info(
            "sae_attached",
            sae_id=sae_id,
            layer=layer,
            memory_mb=memory_mb,
        )

        return {
            "status": "attached",
            "sae_id": sae_id,
            "layer": layer,
            "memory_usage_mb": memory_mb,
            "warnings": compat.warnings,
            "layer_module_path": getattr(
                self._hooker, "last_resolved_module_path", None
            ),
        }

    async def attach_set(
        self,
        sae_layers: list[tuple[str, int]],
    ) -> dict[str, Any]:
        """Attach several SAEs at once for multi-SAE circuit serving (Feature 12).

        Loads ONLY the referenced SAEs (referenced-only loading), each in the
        configured attach dtype (fp16 by default — the measured ~64 MB/SAE
        footprint), and installs one forward hook per ``(sae_id, layer)`` bound
        to that SAE's own decoder. Idempotent per key: re-attaching a
        ``(sae_id, layer)`` already present is skipped. Reports the total
        attached-set VRAM and a warning when it exceeds the configured
        envelope.

        Unlike ``attach_sae`` this does NOT enforce the single-attach guard —
        it is the deliberate multi-SAE path. It coexists with a previously
        single-attached SAE (that SAE stays in the registry).

        Args:
            sae_layers: list of ``(sae_id, layer)`` pairs to attach.

        Returns:
            Dict with per-entry status, total_memory_usage_mb, the envelope,
            and a vram_warning flag.

        Raises:
            ModelNotLoadedError: If no model is loaded.
            SAENotFoundError: If a referenced SAE does not exist.
            SAEIncompatibleError: If a referenced SAE is incompatible.
        """
        from millm.core.config import settings

        model_state = LoadedModelState()
        if not model_state.is_loaded:
            raise ModelNotLoadedError(
                "No model loaded. Load a model before attaching SAEs.",
            )

        # Dedup requested keys, preserving order.
        requested: list[tuple[str, int]] = list(
            dict.fromkeys((sid, int(layer)) for sid, layer in sae_layers)
        )
        attach_dtype = _resolve_attach_dtype(settings.MULTISAE_ATTACH_DTYPE)
        model = model_state.current.model

        # Split into to-attach (new keys) and already-attached (idempotent skip).
        to_attach: list[tuple[str, int]] = []
        results: list[dict[str, Any]] = []
        for sae_id, layer in requested:
            if self._sae_state.get(sae_id, layer) is not None:
                results.append(
                    {"sae_id": sae_id, "layer": layer, "status": "already_attached"}
                )
            else:
                to_attach.append((sae_id, layer))

        # PRE-VALIDATE every new key (existence + compatibility) BEFORE loading
        # anything, so a bad key fails the whole call without a partial attach.
        prepared: list[tuple[str, int, Any, "CompatibilityResult"]] = []
        for sae_id, layer in to_attach:
            sae = await self.get_sae(sae_id)  # raises SAENotFoundError
            compat = await self.check_compatibility(sae_id, layer)
            if not compat.compatible:
                raise SAEIncompatibleError(
                    f"SAE '{sae_id}' incompatible with model: {compat.errors[0]}",
                    details={"errors": compat.errors, "warnings": compat.warnings},
                )
            for warning in compat.warnings:
                logger.warning("sae_compatibility_warning", sae_id=sae_id, warning=warning)
            prepared.append((sae_id, layer, sae, compat))

        # Free-VRAM pre-check: refuse the whole set if the projected cumulative
        # footprint would not fit, rather than OOM'ing mid-load. Estimate each
        # SAE's fp16 steering-weight footprint from its dimensions (W_enc+W_dec
        # = 2·d_in·d_sae params × 2 bytes) — robust to a NULL file_size_bytes,
        # which would otherwise collapse the projection to 0 and defeat the gate.
        attach_bytes = torch.finfo(attach_dtype).bits // 8
        if torch.cuda.is_available() and prepared:
            free_mb = torch.cuda.mem_get_info()[0] / (1024 * 1024)

            def _projected_mb(sae) -> float:
                by_dims = 2 * int(sae.d_in) * int(sae.d_sae) * attach_bytes / (1024 * 1024)
                by_file = ((sae.file_size_bytes or 0) / (1024 * 1024)) * 0.6
                # Use the larger of the dim-based and file-based estimates so an
                # unknown/under-reported file size never under-projects.
                return max(by_dims, by_file) * 1.1  # 10% headroom for buffers

            projected_mb = sum(_projected_mb(sae) for _, _, sae, _ in prepared)
            if projected_mb > free_mb:
                raise InsufficientMemoryError(
                    f"Attaching {len(prepared)} SAE(s) needs ~{int(projected_mb)} MB "
                    f"but only {int(free_mb)} MB is free.",
                    details={"projected_mb": int(projected_mb), "free_mb": int(free_mb)},
                )

        # Attach, tracking keys added in THIS call so we can roll them all back
        # if a later load/install throws — never leave a partial attach or a
        # loaded-but-unregistered SAE leaking on the GPU.
        attached_keys: list[tuple[str, int]] = []
        try:
            for sae_id, layer, sae, compat in prepared:
                loaded_sae = self._loader.load(
                    cache_path=sae.cache_path,
                    device="cuda" if torch.cuda.is_available() else "cpu",
                    dtype=attach_dtype,
                )
                try:
                    handle = self._hooker.install(model, layer, loaded_sae)
                except Exception:
                    # The SAE loaded but the hook failed — free it before it leaks.
                    try:
                        loaded_sae.to_cpu()
                    except Exception:
                        pass
                    raise
                self._reset_dynamo_for_hook_change()
                self._sae_state.set(loaded_sae, sae_id, layer, handle)
                attached_keys.append((sae_id, layer))
                results.append(
                    {
                        "sae_id": sae_id,
                        "layer": layer,
                        "status": "attached",
                        "memory_usage_mb": int(loaded_sae.estimate_memory_mb()),
                        "warnings": compat.warnings,
                    }
                )
        except Exception:
            # Roll back everything attached in THIS call (leave pre-existing
            # attachments untouched).
            for sae_id, layer in attached_keys:
                entry = self._sae_state.get(sae_id, layer)
                if entry is not None and entry.sae is not None:
                    try:
                        entry.sae.to_cpu()
                    except Exception:
                        pass
                self._sae_state.clear(sae_id=sae_id, layer=layer)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.warning("attach_set_rolled_back", rolled_back=len(attached_keys))
            raise

        status_set = self.get_attachment_status_set()
        envelope = int(settings.MULTISAE_VRAM_ENVELOPE_MB)
        total_mb = status_set.total_memory_usage_mb or 0
        vram_warning = total_mb > envelope
        if vram_warning:
            logger.warning(
                "multisae_vram_over_envelope",
                total_mb=total_mb,
                envelope_mb=envelope,
                attached=status_set.count,
            )
        logger.info(
            "sae_set_attached",
            requested=len(requested),
            attached=status_set.count,
            total_mb=total_mb,
        )
        return {
            "status": "attached",
            "entries": results,
            "attached_count": status_set.count,
            "total_memory_usage_mb": total_mb,
            "vram_envelope_mb": envelope,
            "vram_warning": vram_warning,
        }

    async def detach_sae(self, sae_id: str) -> dict[str, Any]:
        """
        Detach an SAE from the model.

        Args:
            sae_id: The SAE's ID.

        Returns:
            Dict with detachment status and freed memory info.

        Raises:
            SAENotFoundError: If SAE doesn't exist.
            SAENotAttachedError: If SAE is not attached.
        """
        sae = await self.get_sae(sae_id)

        # Multi-SAE aware: the SAE is attached if ANY registry entry carries
        # this sae_id (not only when it is the first/singular entry).
        attached_ids = {e.sae_id for e in self._sae_state.entries()}
        if sae_id not in attached_ids:
            raise SAENotAttachedError(
                f"SAE '{sae_id}' is not attached",
                details={"attached_sae_ids": sorted(attached_ids)},
            )

        # Drain in-flight inference requests before removing the hook.
        # If a generation is running when the hook is removed, the PyTorch
        # forward-hook machinery may call a deallocated closure, causing a crash
        # or silently applying stale steering.  We wait up to 30 s (300 × 0.1 s)
        # for the queue to drain, then proceed with a visible error log.
        try:
            inference_svc = self._inference_service
            if inference_svc is None:
                from millm.api.dependencies import get_inference_service
                inference_svc = get_inference_service()

            def _active_count() -> int:
                # Serial path requests wait in the RequestQueue; CBM requests
                # bypass it entirely, so pending_count is 0 during CBM
                # generation.  Sum both so detach drains either backend.
                count = inference_svc.request_queue.pending_count
                cbm = getattr(inference_svc, "_cbm_backend", None)
                if cbm is not None:
                    count += getattr(cbm, "inflight_count", 0)
                return count

            active = _active_count()
            if active > 0:
                logger.info("detach_waiting_for_pending_requests", pending=active)
                for _ in range(300):  # Wait up to 30 seconds
                    if _active_count() == 0:
                        break
                    await asyncio.sleep(0.1)
                remaining = _active_count()
                if remaining > 0:
                    logger.error(
                        "detach_timeout_with_active_requests",
                        pending=remaining,
                        hint="Detaching anyway — hook removed while generation may be in progress. "
                             "Restart the server if inference behaves unexpectedly.",
                    )
        except Exception:
            pass  # Don't block detach if queue check fails

        # Resolve THIS sae_id's registry entries (multi-SAE aware — a sae_id
        # may be attached on more than one layer). ALL of them must be cleaned
        # up, not just the first.
        own_entries = [e for e in self._sae_state.entries() if e.sae_id == sae_id]

        # Get memory before cleanup (sum across this sae_id's layers). Use an
        # explicit branch (not `or`) so a genuine 0-MB sum isn't overridden by
        # the singular fallback.
        if own_entries:
            memory_freed_mb = sum(
                int(e.sae.estimate_memory_mb()) for e in own_entries if e.sae
            )
        else:
            fallback = self._sae_state.attached_sae
            memory_freed_mb = int(fallback.estimate_memory_mb()) if fallback else 0

        # Remove this sae_id's hook(s).
        for e in own_entries:
            if e.hook_handle:
                self._hooker.remove(e.hook_handle)

        # Reset Dynamo so any compiled graph that captured the hooked path is
        # invalidated — otherwise a stale cached graph could keep applying the
        # (now removed) steering after detach.
        self._reset_dynamo_for_hook_change()

        # Clear steering + monitoring + sensing and move to CPU for EVERY SAE
        # this sae_id owns (a multi-layer sae_id has one LoadedSAE per layer);
        # cleaning only the first would leave the rest armed and resident.
        detach_saes = [e.sae for e in own_entries if e.sae]
        if not detach_saes and self._sae_state.attached_sae is not None:
            detach_saes = [self._sae_state.attached_sae]
        for detach_sae in detach_saes:
            detach_sae.clear_steering()
            detach_sae.enable_monitoring(False)
            # Sensing lifecycle (Feature 11): detaching the SAE disarms —
            # the cached encoder slice is about to move to CPU with the SAE.
            try:
                import millm.api.dependencies as deps

                deps.get_sensing_service().disarm(detach_sae)
            except Exception:
                logger.warning("sensing_disarm_on_detach_failed", exc_info=False)
            detach_sae.to_cpu()

        # Clear CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Clear this sae_id's entries in the singleton (also removes any hook
        # not already removed above). Other attached SAEs are untouched.
        self._sae_state.clear(sae_id=sae_id)

        # Capture model_id before clearing state
        model_state = LoadedModelState()
        locked_model_id = model_state.loaded_model_id if model_state.is_loaded else None

        # Update database
        await self.repository.update_status(sae_id, SAEStatus.CACHED)
        await self.repository.deactivate_attachment(sae_id)

        # Auto-unlock the model ONLY when no SAEs remain attached. In a
        # multi-SAE circuit, detaching one SAE must not unlock the model while
        # others are still hooked and steering (a concurrent unload would tear
        # them out mid-generation).
        if locked_model_id and not self._sae_state.is_attached:
            try:
                from millm.db.base import async_session_factory
                from millm.db.repositories.model_repository import ModelRepository

                async with async_session_factory() as session:
                    model_repo = ModelRepository(session)
                    await model_repo.update(locked_model_id, locked=False)
                    logger.info("model_auto_unlocked", model_id=locked_model_id, sae_id=sae_id)
            except Exception as e:
                logger.warning("model_auto_unlock_failed", error=str(e))

        logger.info(
            "sae_detached",
            sae_id=sae_id,
            memory_freed_mb=memory_freed_mb,
        )

        return {
            "status": "detached",
            "sae_id": sae_id,
            "memory_freed_mb": memory_freed_mb,
        }

    @staticmethod
    def _reset_dynamo_for_hook_change() -> None:
        """Reset TorchDynamo so compiled graphs re-trace after a hook change.

        SAE steering/monitoring works by a forward hook on an inner decoder
        layer.  When the model was compiled with torch.compile, a cached graph
        may not reflect a hook that is added or removed afterwards.  Clearing
        Dynamo's cache forces the next forward pass to re-trace with the current
        hook set.  Best-effort: never raises.
        """
        try:
            import torch._dynamo as _dynamo

            _dynamo.reset()
            logger.debug("dynamo_reset_after_hook_change")
        except Exception as e:
            logger.debug("dynamo_reset_skipped", error=str(e))

    # =========================================================================
    # Steering Methods
    # =========================================================================

    def _check_feature_idx(self, feature_idx: int, sae: Any) -> None:
        """Validate a feature index against the attached SAE's dimension.

        Raises InvalidFeatureIndexError (400) instead of ValueError (500) so the
        client gets a meaningful error response rather than an internal server error.
        """
        from millm.core.errors import InvalidFeatureIndexError
        if not 0 <= feature_idx < sae.d_sae:
            raise InvalidFeatureIndexError(
                f"Feature index {feature_idx} is out of range [0, {sae.d_sae}) "
                f"for the attached SAE.",
                details={"feature_idx": feature_idx, "d_sae": sae.d_sae},
            )

    def set_steering(self, feature_idx: int, value: float) -> None:
        """
        Set steering value for a feature.

        Args:
            feature_idx: Feature index (validated against attached SAE's d_sae).
            value: Steering strength.

        Raises:
            SAENotAttachedError: If no SAE is attached.
            InvalidFeatureIndexError: If feature index is out of range (returns 400).
        """
        if not self._sae_state.is_attached:
            raise SAENotAttachedError("No SAE attached")
        sae = self._sae_state.attached_sae
        self._check_feature_idx(feature_idx, sae)
        sae.set_steering(feature_idx, value)

    def set_steering_batch(self, steering: dict[int, float]) -> None:
        """
        Set multiple steering values at once.

        Args:
            steering: Dict mapping feature indices to values.

        Raises:
            SAENotAttachedError: If no SAE is attached.
            InvalidFeatureIndexError: If any feature index is out of range (returns 400).
        """
        if not self._sae_state.is_attached:
            raise SAENotAttachedError("No SAE attached")
        sae = self._sae_state.attached_sae
        for idx in steering:
            self._check_feature_idx(idx, sae)
        sae.set_steering_batch(steering)

    def clear_steering(self, feature_idx: Optional[int] = None) -> None:
        """
        Clear steering for one or all features.

        Args:
            feature_idx: Specific feature to clear (None = clear all).

        Raises:
            SAENotAttachedError: If no SAE is attached.
        """
        if not self._sae_state.is_attached:
            raise SAENotAttachedError("No SAE attached")
        self._sae_state.attached_sae.clear_steering(feature_idx)

    def enable_steering(self, enabled: bool = True) -> None:
        """
        Enable or disable steering.

        Args:
            enabled: Whether to enable steering.

        Raises:
            SAENotAttachedError: If no SAE is attached.
        """
        if not self._sae_state.is_attached:
            raise SAENotAttachedError("No SAE attached")
        self._sae_state.attached_sae.enable_steering(enabled)

    def get_steering_values(self) -> dict[int, float]:
        """
        Get current steering values.

        Returns:
            Dict mapping feature indices to steering values.

        Raises:
            SAENotAttachedError: If no SAE is attached.
        """
        if not self._sae_state.is_attached:
            raise SAENotAttachedError("No SAE attached")
        return self._sae_state.attached_sae.get_steering_values()

    # =========================================================================
    # Monitoring Methods
    # =========================================================================

    def enable_monitoring(
        self,
        enabled: bool = True,
        features: Optional[list[int]] = None,
    ) -> None:
        """
        Enable or disable feature monitoring.

        Args:
            enabled: Whether to capture activations.
            features: Specific features to monitor (None = all). Each index is
                validated against the attached SAE's d_sae.

        Raises:
            SAENotAttachedError: If no SAE is attached.
            InvalidFeatureIndexError: If any feature index is out of range (400).
        """
        if not self._sae_state.is_attached:
            raise SAENotAttachedError("No SAE attached")
        sae = self._sae_state.attached_sae
        # Validate indices before enabling: an out-of-range index would raise
        # inside the forward hook on *every* forward pass (feature_acts[..., idx]),
        # breaking all inference until monitoring is reconfigured.  Reject up
        # front with a 400 instead.
        if enabled and features is not None:
            for idx in features:
                self._check_feature_idx(idx, sae)
        sae.enable_monitoring(enabled, features)

    def get_last_activations(self) -> Optional[Any]:
        """
        Get feature activations from last forward pass.

        Returns:
            Activations tensor or None.

        Raises:
            SAENotAttachedError: If no SAE is attached.
        """
        if not self._sae_state.is_attached:
            raise SAENotAttachedError("No SAE attached")
        return self._sae_state.attached_sae.get_last_feature_activations()

    # =========================================================================
    # Delete Methods
    # =========================================================================

    async def delete_sae(self, sae_id: str) -> dict[str, Any]:
        """
        Delete an SAE from database and disk.

        Args:
            sae_id: The SAE's ID.

        Returns:
            Dict with deletion status and freed disk space.

        Raises:
            SAENotFoundError: If SAE doesn't exist.
            SAEAlreadyAttachedError: If SAE is currently attached.
        """
        sae = await self.get_sae(sae_id)

        # Check SAE is not attached
        if self._sae_state.attached_sae_id == sae_id:
            raise SAEAlreadyAttachedError(
                f"Cannot delete SAE '{sae_id}' while it is attached. "
                "Detach it first.",
                details={"sae_id": sae_id},
            )

        # Delete cache files
        freed_mb = 0.0
        if sae.cache_path:
            freed_mb = await self._downloader.delete(sae.cache_path)

        # Delete from database
        await self.repository.delete(sae_id)

        logger.info(
            "sae_deleted",
            sae_id=sae_id,
            freed_mb=freed_mb,
        )

        return {
            "status": "deleted",
            "sae_id": sae_id,
            "freed_disk_mb": freed_mb,
        }

    # =========================================================================
    # Cleanup
    # =========================================================================

    def shutdown(self) -> None:
        """Clean up resources on application shutdown."""
        if self._sae_state.is_attached:
            # Detach every attached SAE (multi-SAE aware), each hook removed and
            # each SAE moved to CPU before the registry is cleared.
            for entry in self._sae_state.entries():
                if entry.hook_handle:
                    self._hooker.remove(entry.hook_handle)
                if entry.sae:
                    entry.sae.to_cpu()
            self._sae_state.clear()

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        logger.info("SAEService shutdown complete")
