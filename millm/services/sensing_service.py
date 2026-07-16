"""
SensingService (Feature 11): cluster co-activation sensing lifecycle,
post-generation event recording (context decode + persistence + WS), and
status.

Arm/disarm lifecycle: cluster activate (when profiles.sensing_enabled) arms;
deactivate / SAE detach disarms; the enable/disable endpoints toggle the
column and live-arm/disarm when that cluster is active.
"""

import math
from datetime import datetime
from typing import Any, Optional

import torch

from millm.core.config import settings
from millm.core.logging import get_logger
from millm.ml.sae_wrapper import LoadedSAE, SensedHit, SensingConfig

logger = get_logger(__name__)

CONTEXT_TOKENS_HARD_MAX = 64


class SensingService:
    """Singleton (via api.dependencies) owning sensing runtime state."""

    def __init__(self) -> None:
        self._armed_profile_id: Optional[str] = None
        self._armed_profile_name: Optional[str] = None
        self._armed_config: Optional[SensingConfig] = None
        self._display_token: str = ""
        self._member_labels: dict[int, str] = {}
        self._last_request_overhead_ms: float = 0.0
        self._events_recorded: int = 0
        # -inf: the FIRST flush must always emit — 0.0 would throttle it
        # on platforms where monotonic() starts near zero (011 R2).
        self._last_ws_emit_ts: float = float("-inf")
        self._ws_dropped: int = 0

    _WS_MAX_PER_FLUSH = 5
    _WS_MIN_INTERVAL_S = 0.1

    # ==========================================================================
    # Config build + lifecycle
    # ==========================================================================

    def build_config(self, profile: Any) -> SensingConfig:
        """
        SensingConfig from the profile's raw cluster document.

        Thresholds: theta_i = max(theta_floor, epsilon * max_activation_i);
        members missing max_activation use the floor alone. When EVERY member
        lacks it, threshold_mode='floor_only' is recorded for the status API
        (EC-11.4) so operators know detection quality is degraded.
        Optional overrides ride the document under a 'sensing' key.
        """
        meta = profile.cluster_meta or {}
        members = meta.get("members", [])
        overrides = meta.get("sensing", {}) or {}

        def _override(key: str, default: float) -> float:
            try:
                return float(overrides.get(key, default))
            except (TypeError, ValueError):
                return default

        eps = _override("epsilon", settings.SENSING_EPSILON)
        floor = _override("theta_floor", settings.SENSING_THETA_FLOOR)

        indices: list[int] = []
        thetas: list[float] = []
        missing = 0
        for member in members:
            indices.append(int(member["feature_idx"]))
            max_act = member.get("max_activation")
            try:
                max_act = float(max_act) if max_act is not None else None
            except (TypeError, ValueError):
                max_act = None
            if max_act is None:
                missing += 1
                # No activation scale for this member: with a positive
                # configured floor it fires above the floor; with the
                # default floor of 0 it would fire on ANY positive
                # activation (011 R1: degenerate — every token co-'fires'
                # and inflates min_k), so it gets an infinite threshold
                # and simply never contributes.
                thetas.append(floor if floor > 0 else float("inf"))
            else:
                thetas.append(max(floor, eps * max_act))

        if not indices:
            raise ValueError("cluster has no members to sense")
        if all(theta == float("inf") for theta in thetas):
            raise ValueError(
                "no member has max_activation data and no positive "
                "theta_floor is configured — sensing has no usable "
                "thresholds (set sensing.theta_floor in the definition, or "
                "re-export from miStudio with activation statistics)"
            )

        mode = "floor_only" if missing == len(indices) else "epsilon_max"
        min_k = int(_override("min_k", max(2, math.ceil(0.3 * len(indices)))))
        min_k = max(1, min(min_k, len(indices)))
        context_tokens = int(_override(
            "context_tokens", settings.SENSING_CONTEXT_TOKENS))
        context_tokens = max(0, min(context_tokens, CONTEXT_TOKENS_HARD_MAX))

        return SensingConfig(
            profile_id=profile.id,
            member_indices=indices,
            thresholds=torch.tensor(thetas, dtype=torch.float32),
            threshold_mode=mode,
            min_k=min_k,
            context_tokens=context_tokens,
            max_events_per_request=settings.SENSING_MAX_EVENTS_PER_REQUEST,
        )

    def arm_for_profile(self, profile: Any, sae: LoadedSAE) -> None:
        """Arm sensing for an active cluster profile (idempotent).

        Applies the same declared-feature-space gate as steering activation
        (011 R1): arming must never index_select out of range — on CUDA
        that's a device-side assert that poisons the context for the whole
        process, not just an exception.
        """
        declared = ((profile.cluster_meta or {}).get("sae") or {}).get("n_features")
        if declared is not None and int(declared) != sae.d_sae:
            raise ValueError(
                f"cluster declares an SAE with {declared} features; the "
                f"attached SAE has {sae.d_sae} — refusing to arm sensing"
            )
        config = self.build_config(profile)
        bad = [i for i in config.member_indices if not 0 <= i < sae.d_sae]
        if bad:
            raise ValueError(
                f"member feature indices {bad} out of range "
                f"[0, {sae.d_sae}) for the attached SAE"
            )
        sae.arm_sensing(config)
        self._armed_profile_id = profile.id
        self._armed_profile_name = profile.name
        self._armed_config = config
        meta = profile.cluster_meta or {}
        self._display_token = meta.get("display_token") or profile.name
        self._member_labels = {}
        for member in meta.get("members", []):
            label = member.get("label")
            if label:
                self._member_labels[int(member["feature_idx"])] = str(label)

    def disarm(self, sae: Optional[LoadedSAE]) -> None:
        """Disarm sensing (idempotent; sae may already be detached)."""
        if sae is not None:
            sae.disarm_sensing()
        self._armed_profile_id = None
        self._armed_profile_name = None
        self._armed_config = None
        self._display_token = ""
        self._member_labels = {}

    @property
    def is_armed(self) -> bool:
        return self._armed_profile_id is not None

    @property
    def armed_profile_id(self) -> Optional[str]:
        return self._armed_profile_id

    # ==========================================================================
    # Recording (post-generation, off the hot path)
    # ==========================================================================

    async def record(
        self,
        request_id: str,
        hits: list[SensedHit],
        truncated: bool,
        full_ids: Optional[torch.Tensor],
        tokenizer: Any,
        ambient_counts: Optional[dict[int, int]] = None,
        profile_id: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        """
        Decode context windows, persist bounded events, emit WS updates.

        ambient_counts maps event index -> full-SAE fired count (best-effort,
        filled by the caller only when un-compacted monitoring co-ran).
        profile_id is the BEGIN-time snapshot from the inference path — a
        mid-request re-arm must not attribute these hits to the newly armed
        cluster (011 R1). Falls back to the currently armed id.
        Returns the persisted events as API-shaped dicts.
        """
        profile_id = profile_id or self._armed_profile_id
        if not hits or profile_id is None:
            return []
        # A re-arm between begin and flush swapped the armed identity: the
        # summary formatter (display token, member labels) now belongs to a
        # DIFFERENT cluster than these hits. Persist under the snapshot id
        # with neutral formatting rather than mis-branding (011 R2).
        if profile_id != self._armed_profile_id:
            self._display_token = profile_id
            self._member_labels = {}
        config = self._armed_config
        k = config.context_tokens if config else 0

        rows: list[dict[str, Any]] = []
        for i, hit in enumerate(hits):
            context_text, context_ids = self._context(full_ids, hit, k, tokenizer)
            rows.append({
                "profile_id": profile_id,
                "request_id": request_id,
                "phase": hit.phase,
                "pos_start": hit.pos_start,
                "pos_end": hit.pos_end,
                "fired_members": [[idx, round(act, 4)]
                                  for idx, act in hit.fired],
                "fired_count": hit.fired_count,
                "score": round(hit.score, 4),
                "ambient_fired_count": (ambient_counts or {}).get(i),
                "context_text": context_text,
                "context_token_ids": context_ids,
                "summary": self._summary(hit),
                # Only the LAST event marks truncation — that's where the
                # per-request cap actually cut (011 R1: stamping every row
                # made the cut point unrecoverable).
                "truncated": truncated and i == len(hits) - 1,
            })

        from millm.db.base import async_session_factory
        from millm.db.repositories.sensing_repository import SensingRepository

        async with async_session_factory() as session:
            repo = SensingRepository(session)
            persisted = await repo.create_many(rows)
            await repo.prune(
                profile_id,
                cap=settings.SENSING_MAX_EVENTS_PER_CLUSTER,
                max_age_days=settings.SENSING_MAX_AGE_DAYS,
            )
            payloads = [row.to_dict(include_context=True) for row in persisted]
            await session.commit()

        self._events_recorded += len(payloads)
        self._emit_events(payloads)
        return payloads

    @staticmethod
    def _context(
        full_ids: Optional[torch.Tensor],
        hit: SensedHit,
        k: int,
        tokenizer: Any,
    ) -> tuple[Optional[str], Optional[list[int]]]:
        """±K token window around the span; K=0 keeps events without text."""
        if k == 0 or full_ids is None or tokenizer is None:
            return None, None
        try:
            ids = full_ids[0] if full_ids.dim() == 2 else full_ids
            lo = max(0, hit.pos_start - k)
            hi = min(int(ids.shape[-1]), hit.pos_end + 1 + k)
            window = ids[lo:hi].tolist()
            text = tokenizer.decode(window, skip_special_tokens=True)
            return text, window
        except Exception:
            logger.warning("sensing_context_decode_failed", exc_info=False)
            return None, None

    def _summary(self, hit: SensedHit) -> str:
        """Human-readable one-liner, hard-capped at 300 chars (SEN-R4)."""
        m = len(self._armed_config.member_indices) if self._armed_config else 0
        peak_idx, peak_act = max(hit.fired, key=lambda p: p[1]) if hit.fired \
            else (0, 0.0)
        label = self._member_labels.get(peak_idx)
        peak = f"F{peak_idx}" + (f" '{label}'" if label else "")
        span = (f"@ {hit.pos_start}" if hit.pos_start == hit.pos_end
                else f"@ {hit.pos_start}–{hit.pos_end}")
        text = (f"{self._display_token}: {hit.fired_count}/{m} members fired "
                f"(peak {peak} {hit.score:.1f}×θ) during {hit.phase} {span}")
        return text[:300]

    def _emit_events(self, payloads: list[dict[str, Any]]) -> None:
        """Fire-and-forget WS emission (payload excludes context — user
        content and size; the UI fetches detail via REST).

        Throttled like monitoring (SEN-P4): at most _WS_MAX_PER_FLUSH
        emissions per flush and a minimum interval between flushes — the DB
        rows are complete regardless, and the UI reconciles on refetch."""
        import time as _time

        now = _time.monotonic()
        if now - self._last_ws_emit_ts < self._WS_MIN_INTERVAL_S:
            self._ws_dropped += len(payloads)
            return
        self._last_ws_emit_ts = now
        if len(payloads) > self._WS_MAX_PER_FLUSH:
            self._ws_dropped += len(payloads) - self._WS_MAX_PER_FLUSH
            payloads = payloads[: self._WS_MAX_PER_FLUSH]
        try:
            from millm.sockets.progress import progress_emitter as emitter

            for payload in payloads:
                slim = {key: value for key, value in payload.items()
                        if key not in ("context_text", "context_token_ids")}
                emitter.emit_sensing_event(slim)
        except Exception as exc:
            logger.warning("sensing_ws_emit_failed", error=str(exc))

    # ==========================================================================
    # Status
    # ==========================================================================

    def note_request_overhead(self, overhead_ms: float) -> None:
        self._last_request_overhead_ms = overhead_ms
        if overhead_ms > settings.SENSING_MAX_OVERHEAD_MS:
            logger.warning(
                "sensing_overhead_above_threshold",
                overhead_ms=round(overhead_ms, 3),
                threshold_ms=settings.SENSING_MAX_OVERHEAD_MS,
            )

    def status(self) -> dict[str, Any]:
        config = self._armed_config
        return {
            "armed": self.is_armed,
            "profile_id": self._armed_profile_id,
            "profile_name": self._armed_profile_name,
            "member_count": len(config.member_indices) if config else 0,
            "min_k": config.min_k if config else None,
            "threshold_mode": config.threshold_mode if config else None,
            "context_tokens": config.context_tokens if config else None,
            "last_request_overhead_ms": round(
                self._last_request_overhead_ms, 3),
            "overhead_warn_threshold_ms": settings.SENSING_MAX_OVERHEAD_MS,
            "events_recorded_since_start": self._events_recorded,
            "ws_events_dropped": self._ws_dropped,
            "retention": {
                "max_events_per_cluster": settings.SENSING_MAX_EVENTS_PER_CLUSTER,
                "max_age_days": settings.SENSING_MAX_AGE_DAYS,
            },
        }
