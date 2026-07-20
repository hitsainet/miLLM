"""Circuit edge sensing service (Feature 15).

Feature 11's sensing service arms ONE SAE for a cluster. A circuit spans
layers, so this service arms N SAEs for one circuit against a shared
``EdgeFireRing`` and reconciles them as a set: an edge is sensable only when
BOTH its endpoints resolve to an attached SAE with a usable threshold, and
every edge that fails that test is recorded with a reason rather than silently
dropped.

The honesty rule this service must not break: observing an edge fire says the
upstream member fired and the downstream partner then fired within the lag
window. It is never evidence about the edge's rung. ``rung_language`` is
rendered from the ladder and carried verbatim.
"""

import time
from typing import Any, Optional

from millm.core.circuit_evidence import rung_language
from millm.core.config import settings
from millm.core.logging import get_logger
from millm.ml.sae_wrapper import (
    CircuitSensingConfig,
    EdgeFireRing,
    EdgeSpec,
    LoadedSAE,
)

logger = get_logger(__name__)

#: Hard ceiling mirroring SensingService's — context capture is off the hot
#: path but the decoded window still rides every event row.
CONTEXT_TOKENS_HARD_MAX = 64
#: A lag window wider than this stops being an attribution and becomes a
#: coincidence detector.
MAX_TOKEN_LAG_HARD_MAX = 64


class UnsensableEdge:
    """An edge that cannot be observed, with the reason surfaced to the UI."""

    __slots__ = ("edge_key", "reason", "detail")

    def __init__(self, edge_key: str, reason: str, detail: str = ""):
        self.edge_key = edge_key
        self.reason = reason
        self.detail = detail

    def to_dict(self) -> dict[str, Any]:
        return {
            "edge_key": self.edge_key,
            "reason": self.reason,
            "detail": self.detail,
        }


def edge_key_for(up_layer: int, up_idx: Any, down_layer: int, down_idx: Any) -> str:
    """Stable identity for an edge within its circuit.

    circuit-definition/v1 edges carry no id of their own, so the key is
    synthesised — and it must be stable across arm cycles because the shared
    ring keys upstream fires on it.
    """
    return f"{up_idx}@{up_layer}->{down_idx}@{down_layer}"


class CircuitSensingService:
    """Arms a circuit's SAE set for edge sensing and records what it observes."""

    def __init__(self) -> None:
        self._circuit_id: Optional[str] = None
        self._circuit_name: Optional[str] = None
        self._armed_layers: list[int] = []
        self._configs: dict[int, CircuitSensingConfig] = {}
        self._ring: Optional[EdgeFireRing] = None
        self._unsensable: list[UnsensableEdge] = []
        self._max_token_lag: int = settings.CIRCUIT_SENSING_MAX_TOKEN_LAG
        self._last_request_overhead_ms: float = 0.0
        self._events_recorded: int = 0
        #: -inf, not 0.0 — the FIRST flush must be allowed to emit.
        self._last_ws_emit_ts: float = float("-inf")
        self._ws_dropped: int = 0

    _WS_MAX_PER_FLUSH = 5
    _WS_MIN_INTERVAL_S = 0.1

    # ------------------------------------------------------------------
    # Arming
    # ------------------------------------------------------------------

    def build_configs(
        self,
        circuit: Any,
        definition: Any,
        layer_saes: dict[int, LoadedSAE],
    ) -> tuple[dict[int, CircuitSensingConfig], list[UnsensableEdge]]:
        """Resolve a circuit's edges against the SAEs actually attached.

        Returns per-layer configs plus the edges that cannot be sensed. An
        edge is dropped — with a reason — when either endpoint lacks a feature
        index (a cluster supernode has none), when its layer has no
        unambiguously attached SAE, or when the member has no usable
        activation threshold.
        """
        meta = circuit.circuit_meta or {}
        overrides = dict((meta.get("sensing") or {}))
        overrides.update((meta.get("sensing_overrides") or {}))

        def _override(key: str, default: float, minimum: Optional[float] = None):
            try:
                value = float(overrides.get(key, default))
            except (TypeError, ValueError):
                return default
            # Out-of-range authored overrides degrade to the default rather
            # than clamping — a negative epsilon would resurrect the
            # fire-on-anything degenerate case (011 R3 #2).
            if minimum is not None and value < minimum:
                return default
            return value

        eps = _override("epsilon", settings.CIRCUIT_SENSING_EPSILON, minimum=1e-9)
        floor = _override(
            "theta_floor", settings.CIRCUIT_SENSING_THETA_FLOOR, minimum=0.0
        )
        lag = int(
            _override(
                "max_token_lag", settings.CIRCUIT_SENSING_MAX_TOKEN_LAG, minimum=1
            )
        )
        lag = max(1, min(MAX_TOKEN_LAG_HARD_MAX, lag))
        self._max_token_lag = lag

        ctx = int(
            _override(
                "context_tokens", settings.CIRCUIT_SENSING_CONTEXT_TOKENS, minimum=0
            )
        )
        ctx = max(0, min(CONTEXT_TOKENS_HARD_MAX, ctx))

        # (layer, feature_idx) -> max_activation, from the circuit's members.
        stats = self._member_stats(definition)

        unsensable: list[UnsensableEdge] = []
        # layer -> {feature_idx -> theta}; only features that participate in a
        # sensable edge get armed.
        wanted: dict[int, dict[int, float]] = {}
        resolved: list[tuple[Any, int, int, int, int]] = []

        for edge in getattr(definition, "edges", []) or []:
            up, down = edge.up, edge.down
            key = edge_key_for(
                up.layer, up.feature_idx, down.layer, down.feature_idx
            )
            if up.feature_idx is None or down.feature_idx is None:
                # A cluster-supernode endpoint has no single feature index, so
                # there is no activation to threshold. Not covered by EC-15.4
                # or EC-15.6 — recorded as its own reason.
                unsensable.append(
                    UnsensableEdge(
                        key,
                        "endpoint_not_a_feature",
                        "an endpoint is a cluster supernode with no feature index",
                    )
                )
                continue
            missing_layers = [
                lay for lay in (up.layer, down.layer) if lay not in layer_saes
            ]
            if missing_layers:
                # EC-15.4: slice-fallback serves one layer, so nearly every
                # cross-layer edge lands here.
                unsensable.append(
                    UnsensableEdge(
                        key,
                        "layer_not_attached",
                        f"no unambiguously attached SAE on layer(s) {missing_layers}",
                    )
                )
                continue

            theta_up = self._theta(stats.get((up.layer, up.feature_idx)), eps, floor)
            theta_down = self._theta(
                stats.get((down.layer, down.feature_idx)), eps, floor
            )
            if theta_up == float("inf") or theta_down == float("inf"):
                # EC-15.6: without an activation scale a member either never
                # fires (inf) or fires on anything (theta 0). Both make the
                # edge unobservable; refuse rather than emit noise.
                unsensable.append(
                    UnsensableEdge(
                        key,
                        "no_activation_threshold",
                        "an endpoint has no usable max_activation and no positive floor",
                    )
                )
                continue

            wanted.setdefault(up.layer, {})[up.feature_idx] = theta_up
            wanted.setdefault(down.layer, {})[down.feature_idx] = theta_down
            resolved.append(
                (edge, up.layer, up.feature_idx, down.layer, down.feature_idx)
            )

        configs = self._assemble(
            circuit, resolved, wanted, layer_saes, lag, ctx, floor
        )
        return configs, unsensable

    @staticmethod
    def _theta(max_act: Optional[float], eps: float, floor: float) -> float:
        """theta = max(floor, epsilon * max_activation).

        A zero/negative max_activation is as degenerate as a missing one
        (theta would be 0 and the member would fire on anything), so both
        yield the floor — or infinity when no positive floor is configured,
        which means "never fires" rather than "always fires".
        """
        try:
            value = float(max_act) if max_act is not None else None
        except (TypeError, ValueError):
            value = None
        if value is not None and value <= 0:
            value = None
        if value is None:
            return floor if floor > 0 else float("inf")
        return max(floor, eps * value)

    @staticmethod
    def _member_stats(definition: Any) -> dict[tuple[int, int], Optional[float]]:
        """(layer, feature_idx) -> max_activation, over the EXPANDED members.

        Mirrors the serving path's expansion: a cluster_ref contributes its
        frozen expanded_members AND its own feature when both are present.
        """
        out: dict[tuple[int, int], Optional[float]] = {}
        for member in getattr(definition, "members", []) or []:
            sources = list(getattr(member, "expanded_members", None) or [])
            own = getattr(member, "feature", None)
            if own is not None:
                sources.append(own)
            for feat in sources:
                idx = getattr(feat, "feature_idx", None)
                if idx is None:
                    continue
                out.setdefault(
                    (member.layer, int(idx)), getattr(feat, "max_activation", None)
                )
        return out

    def _assemble(
        self,
        circuit: Any,
        resolved: list[tuple[Any, int, int, int, int]],
        wanted: dict[int, dict[int, float]],
        layer_saes: dict[int, LoadedSAE],
        lag: int,
        ctx: int,
        floor: float,
    ) -> dict[int, CircuitSensingConfig]:
        """Turn resolved edges into one CircuitSensingConfig per layer."""
        import torch

        # Deterministic column ordering per layer.
        order: dict[int, list[int]] = {
            layer: sorted(feats) for layer, feats in wanted.items()
        }
        col_of: dict[int, dict[int, int]] = {
            layer: {idx: col for col, idx in enumerate(feats)}
            for layer, feats in order.items()
        }

        specs_by_layer: dict[int, list[EdgeSpec]] = {lay: [] for lay in order}
        for edge, up_layer, up_idx, down_layer, down_idx in resolved:
            key = edge_key_for(up_layer, up_idx, down_layer, down_idx)
            rung = int(getattr(edge, "rung", 0) or 0)
            base = dict(
                edge_key=key,
                up_layer=up_layer,
                up_feature_idx=up_idx,
                down_layer=down_layer,
                down_feature_idx=down_idx,
                rung=rung,
                # Rendered from the ladder, never composed here.
                rung_language=rung_language(rung),
                edge_type=getattr(edge, "type", None),
            )
            # Each SAE gets the whole spec but only the column belonging to
            # its own layer; -1 means "not my half".
            for layer in {up_layer, down_layer}:
                specs_by_layer[layer].append(
                    EdgeSpec(
                        **base,
                        up_col=(
                            col_of[up_layer][up_idx] if layer == up_layer else -1
                        ),
                        down_col=(
                            col_of[down_layer][down_idx]
                            if layer == down_layer
                            else -1
                        ),
                    )
                )

        configs: dict[int, CircuitSensingConfig] = {}
        for layer, feats in order.items():
            thetas = [wanted[layer][idx] for idx in feats]
            configs[layer] = CircuitSensingConfig(
                circuit_id=circuit.id,
                layer=layer,
                member_indices=list(feats),
                thresholds=torch.tensor(thetas, dtype=torch.float32),
                threshold_mode="floor_only" if floor > 0 and all(
                    t == floor for t in thetas
                ) else "epsilon_max",
                edges=specs_by_layer[layer],
                max_token_lag=lag,
                context_tokens=ctx,
                max_events_per_request=(
                    settings.CIRCUIT_SENSING_MAX_EVENTS_PER_REQUEST
                ),
            )
        return configs

    def arm_for_circuit(
        self, circuit: Any, definition: Any, layer_saes: dict[int, LoadedSAE]
    ) -> list[UnsensableEdge]:
        """Arm every layer of the circuit against one shared ring."""
        configs, unsensable = self.build_configs(circuit, definition, layer_saes)
        if not configs:
            self.disarm(layer_saes)
            self._unsensable = unsensable
            logger.warning(
                "circuit_sensing_no_sensable_edges",
                circuit_id=circuit.id,
                unsensable=len(unsensable),
            )
            return unsensable

        ring = EdgeFireRing(self._max_token_lag)
        armed: list[int] = []
        try:
            for layer, config in configs.items():
                layer_saes[layer].arm_edge_sensing(config, ring)
                armed.append(layer)
        except Exception:
            # Partial arming would sense half a circuit and report edges whose
            # other endpoint was never watched. Roll back to nothing.
            for layer in armed:
                try:
                    layer_saes[layer].disarm_edge_sensing()
                except Exception:
                    logger.exception("circuit_sensing_rollback_failed", layer=layer)
            raise

        self._circuit_id = circuit.id
        self._circuit_name = getattr(circuit, "name", None)
        self._armed_layers = sorted(configs)
        self._configs = configs
        self._ring = ring
        self._unsensable = unsensable
        logger.info(
            "circuit_sensing_armed",
            circuit_id=circuit.id,
            layers=self._armed_layers,
            sensable_edges=sum(
                1 for s in configs[self._armed_layers[0]].edges if s.up_col >= 0
            ),
            unsensable_edges=len(unsensable),
        )
        return unsensable

    def disarm(self, layer_saes: Optional[dict[int, LoadedSAE]] = None) -> None:
        for sae in (layer_saes or {}).values():
            try:
                sae.disarm_edge_sensing()
            except Exception:
                logger.exception("circuit_sensing_disarm_failed")
        self._circuit_id = None
        self._circuit_name = None
        self._armed_layers = []
        self._configs = {}
        self._ring = None
        self._unsensable = []

    # ------------------------------------------------------------------
    # Request lifecycle
    # ------------------------------------------------------------------

    @property
    def is_armed(self) -> bool:
        return self._circuit_id is not None

    @property
    def armed_circuit_id(self) -> Optional[str]:
        return self._circuit_id

    def begin_request(self, request_id: str, layer_saes: dict[int, LoadedSAE]) -> bool:
        """Open one boundary across every armed layer.

        The ring is cleared HERE, once for the whole circuit — clearing it in
        each SAE's begin would wipe upstream fires recorded by a sibling that
        began first.
        """
        if not self.is_armed:
            return False
        if self._ring is not None:
            self._ring.clear()
        began = False
        for layer in self._armed_layers:
            sae = layer_saes.get(layer)
            if sae is None or not sae.is_edge_sensing_armed:
                continue
            sae.begin_edge_sensing_request(request_id)
            began = True
        return began

    def collect_edges(
        self, layer_saes: dict[int, LoadedSAE]
    ) -> tuple[str, list[Any], bool]:
        """Drain every armed layer and merge, ordered by downstream position.

        Overhead is SUMMED across the SAEs: one circuit request touches N
        hooks, so a per-SAE assignment would under-report by a factor of N.
        """
        request_id = ""
        merged: list[Any] = []
        truncated = False
        overhead = 0.0
        for layer in self._armed_layers:
            sae = layer_saes.get(layer)
            if sae is None:
                continue
            rid, edges, trunc = sae.collect_sensed_edges()
            request_id = request_id or rid
            merged.extend(edges)
            truncated = truncated or trunc
            overhead += float(getattr(sae, "_edge_overhead_ms", 0.0) or 0.0)
        self._last_request_overhead_ms = overhead
        if overhead > settings.CIRCUIT_SENSING_MAX_OVERHEAD_MS:
            logger.warning(
                "circuit_sensing_overhead_high",
                overhead_ms=round(overhead, 2),
                threshold_ms=settings.CIRCUIT_SENSING_MAX_OVERHEAD_MS,
                layers=self._armed_layers,
            )
        merged.sort(key=lambda e: (e.down_pos, e.up_pos))
        if self._ring is not None:
            self._ring.clear()
        return request_id, merged, truncated

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    async def record(
        self,
        request_id: str,
        edges: list[Any],
        truncated: bool,
        full_ids: Any,
        tokenizer: Any,
    ) -> list[dict[str, Any]]:
        """Persist observed edges and broadcast them (without prompt text)."""
        from millm.db.base import async_session_factory
        from millm.db.repositories.circuit_edge_sensing_repository import (
            CircuitEdgeSensingRepository,
        )

        circuit_id = self._circuit_id
        if not circuit_id or not edges:
            return []
        ctx_tokens = 0
        if self._armed_layers and self._configs:
            ctx_tokens = self._configs[self._armed_layers[0]].context_tokens

        rows: list[dict[str, Any]] = []
        for edge in edges:
            text, window, parts = self._context(full_ids, edge, ctx_tokens, tokenizer)
            rows.append(
                dict(
                    circuit_id=circuit_id,
                    request_id=request_id,
                    phase=edge.phase,
                    edge_key=edge.edge_key,
                    up_layer=edge.up_layer,
                    up_feature_idx=edge.up_feature_idx,
                    up_pos=edge.up_pos,
                    up_act=edge.up_act,
                    down_layer=edge.down_layer,
                    down_feature_idx=edge.down_feature_idx,
                    down_pos=edge.down_pos,
                    down_act=edge.down_act,
                    token_lag=edge.token_lag,
                    edge_rung=edge.rung,
                    # Carried verbatim from the arm-time render — NOT
                    # re-derived, so the row keeps describing the evidence
                    # that was true when it was observed.
                    edge_rung_language=edge.rung_language,
                    edge_type=edge.edge_type,
                    context_text=text,
                    context_token_ids=window,
                    context_parts=parts,
                    summary=self.summarize(edge),
                    truncated=truncated,
                )
            )

        payloads: list[dict[str, Any]] = []
        try:
            async with async_session_factory() as session:
                repo = CircuitEdgeSensingRepository(session)
                saved = await repo.create_many(rows)
                payloads = [r.to_dict(include_context=False) for r in saved]
                await repo.prune(
                    circuit_id,
                    cap=settings.CIRCUIT_SENSING_MAX_EVENTS_PER_CIRCUIT,
                    max_age_days=settings.CIRCUIT_SENSING_MAX_AGE_DAYS,
                )
                await session.commit()
        except Exception:
            logger.exception("circuit_sensing_persist_failed")
            return []

        self.note_events_recorded(len(payloads))
        self._emit(payloads)
        return payloads

    @staticmethod
    def _context(
        full_ids: Any, edge: Any, k: int, tokenizer: Any
    ) -> tuple[Optional[str], Optional[list[int]], Optional[dict[str, str]]]:
        """±K token window spanning up_pos..down_pos, with a highlight split.

        Segmentation uses PREFIX decodes and length slicing. Decoding each
        segment independently glues words on SentencePiece models (a
        segment-leading '▁piece' loses its space) — the bug Feature 11 R1
        found and fixed; the FTID's sketch for this method reintroduced it.
        Specials are kept so the three prefixes stay aligned.
        """
        if k == 0 or full_ids is None or tokenizer is None:
            return None, None, None
        try:
            ids = full_ids[0] if full_ids.dim() == 2 else full_ids
            total = int(ids.shape[-1])
            start, end = edge.up_pos, edge.down_pos
            if start >= total:
                # Span entirely beyond the available ids (decode-phase event
                # with prompt-only fallback): an empty box with a zero-width
                # highlight is worse than no context at all.
                return None, None, None
            lo = max(0, start - k)
            hi = min(total, end + 1 + k)
            window = ids[lo:hi].tolist()
            text = tokenizer.decode(window, skip_special_tokens=True)
            span_lo = max(start - lo, 0)
            span_hi = min(end + 1 - lo, len(window))
            d_before = tokenizer.decode(window[:span_lo], skip_special_tokens=False)
            d_through = tokenizer.decode(window[:span_hi], skip_special_tokens=False)
            d_all = tokenizer.decode(window, skip_special_tokens=False)
            if not (d_through.startswith(d_before) and d_all.startswith(d_through)):
                # Byte-level BPE can split a multi-byte character at the span
                # boundary, so length-slicing would misplace the highlight.
                # Plain text beats a wrong mark.
                return text, window, None
            return (
                text,
                window,
                {
                    "before": d_before,
                    "span": d_through[len(d_before):],
                    "after": d_all[len(d_through):],
                },
            )
        except Exception:
            logger.warning("circuit_sensing_context_decode_failed", exc_info=False)
            return None, None, None

    def _emit(self, payloads: list[dict[str, Any]]) -> None:
        """Broadcast at most _WS_MAX_PER_FLUSH events, throttled."""
        if not payloads:
            return
        if not self.should_emit():
            self.note_ws_dropped(len(payloads))
            return
        try:
            from millm.sockets.progress import progress_emitter

            for payload in payloads[: self._WS_MAX_PER_FLUSH]:
                progress_emitter.emit_circuit_sensing_event(payload)
            overflow = len(payloads) - self._WS_MAX_PER_FLUSH
            if overflow > 0:
                self.note_ws_dropped(overflow)
        except Exception:
            logger.warning("circuit_sensing_emit_failed", exc_info=False)

    def summarize(self, edge: Any) -> str:
        """One-line human summary. The rung phrase is carried VERBATIM.

        Never composes evidence language: at rung<2 the phrase is
        "associated"/"suggested (attribution-supported)" and the summary must
        not upgrade it by describing the observation as causal.
        """
        text = (
            f"edge {edge.up_feature_idx}@L{edge.up_layer} → "
            f"{edge.down_feature_idx}@L{edge.down_layer} fired "
            f"{edge.token_lag} token(s) apart "
            f"[{edge.rung_language}]"
        )
        return text[:300]

    def status(self, layer_saes: Optional[dict[int, LoadedSAE]] = None) -> dict:
        """Runtime state, self-reconciled against the SAEs actually armed."""
        armed = self.is_armed
        if armed and layer_saes is not None:
            live = [
                layer
                for layer in self._armed_layers
                if (sae := layer_saes.get(layer)) is not None
                and sae.is_edge_sensing_armed
            ]
            if len(live) != len(self._armed_layers):
                # Reporting armed forever after a swallowed disarm is exactly
                # the failure F11's status reconciliation exists to prevent.
                logger.info(
                    "circuit_sensing_state_reconciled",
                    expected=self._armed_layers,
                    live=live,
                )
                self.disarm(layer_saes)
                armed = False
        sensable = 0
        if self._configs:
            first = self._configs[self._armed_layers[0]] if self._armed_layers else None
            if first is not None:
                sensable = sum(1 for s in first.edges if s.up_col >= 0)
        return {
            "armed": armed,
            "circuit_id": self._circuit_id,
            "circuit_name": self._circuit_name,
            "layers": list(self._armed_layers),
            "sensable_edges": sensable,
            "unsensable_edges": [u.to_dict() for u in self._unsensable],
            "max_token_lag": self._max_token_lag,
            "last_request_overhead_ms": round(self._last_request_overhead_ms, 3),
            "events_recorded": self._events_recorded,
            "ws_dropped": self._ws_dropped,
        }

    def note_events_recorded(self, count: int) -> None:
        self._events_recorded += int(count)

    def should_emit(self) -> bool:
        """WS throttle: at most one flush per _WS_MIN_INTERVAL_S."""
        now = time.monotonic()
        if (now - self._last_ws_emit_ts) < self._WS_MIN_INTERVAL_S:
            return False
        self._last_ws_emit_ts = now
        return True

    def note_ws_dropped(self, count: int) -> None:
        self._ws_dropped += int(count)
