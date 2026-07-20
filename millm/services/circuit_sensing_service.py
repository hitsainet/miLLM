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
from millm.ml.edge_sensing import EdgeSensingRequestContext
from millm.ml.sae_wrapper import (
    CircuitSensingConfig,
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
        # The CURRENT request's context, or None outside a boundary. Replaces
        # the long-lived `_ring`: rings are per (request, circuit) now and are
        # owned by the context, so they cannot outlive the request.
        self._ctx: Optional[EdgeSensingRequestContext] = None
        self._armed_saes: dict[int, LoadedSAE] = {}
        #: SAEs actually bound for the OPEN request. `_armed_saes` can be
        #: swapped under us by a re-arm, so it is not a reliable release list
        #: on its own (R3-02).
        self._bound_saes: list[LoadedSAE] = []
        #: Identity of the circuit that owns the OPEN request boundary.
        self._request_circuit_id: Optional[str] = None
        self._request_context_tokens: int = 0
        #: Why an armed circuit is nonetheless not observing (R3: an operator
        #: saw armed=true and zero events with nothing explaining it).
        self._paused_reason: Optional[str] = None
        #: True when `_paused_reason` was set during the CURRENT request, so a
        #: caller's stale-clear must not wipe it (R2-02).
        self._pause_is_current: bool = False
        #: Fires among the armed circuit's OWN members in the last request.
        #: Deliberately NOT ambient_fired_count — see _ambient_fired_count().
        self._last_request_member_fires: int = 0
        #: Layers that dropped events in the last drained request (BR-006).
        self._last_request_truncated_layers: list[int] = []
        #: Armed layers with no usable SAE at begin time — dark for the
        #: current request, and reported as incomplete rather than complete.
        self._request_dark_layers: list[int] = []
        #: Whether the OPEN request has already been counted as truncated, so a
        #: second drain of the same boundary cannot count it again (R3-07).
        self._request_counted_truncated: bool = False
        self._unsensable: list[UnsensableEdge] = []
        self._max_token_lag: int = settings.CIRCUIT_SENSING_MAX_TOKEN_LAG
        self._last_request_overhead_ms: float = 0.0
        self._events_recorded: int = 0
        #: R1-06: how many request boundaries this armed circuit has actually
        #: observed. Without it, "quiet traffic" and "no request ever reached
        #: sensing" are the same reading — armed, zero events, no reason — and
        #: the second is a wiring failure the operator cannot see.
        self._requests_sensed: int = 0
        #: -inf, not 0.0 — the FIRST flush must be allowed to emit.
        self._last_ws_emit_ts: float = float("-inf")
        self._ws_dropped: int = 0
        #: R2-11: events the per-flush cap DECLINED to send. Distinct from
        #: `ws_dropped`, which counts delivery FAILURES. Both are zero on a
        #: healthy quiet system; only this one is non-zero on a healthy BUSY
        #: one, which is the difference an operator needs to read the panel.
        self._ws_throttled: int = 0
        #: R2-13: how many requests have truncated since arming. The
        #: per-request `truncated_layers` describes only the LAST drained
        #: request, so a fast-arriving next request supersedes it before an
        #: operator polls — correct for a "last request" field, but it means a
        #: rare truncation can never be seen. This one is cumulative and cannot
        #: be raced away.
        self._requests_truncated: int = 0

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
        # R2: this used to assign self._max_token_lag here, BEFORE arming could
        # fail — so a circuit that never armed still changed the reported lag,
        # and the next EdgeFireRing was built from it. The value is now
        # returned on the config and committed only by a successful arm.

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
                key = (member.layer, int(idx))
                value = getattr(feat, "max_activation", None)
                # R1: setdefault let a None from expanded_members mask the
                # member's own real max_activation, declaring a perfectly
                # sensable edge unsensable depending on iteration order. Keep
                # the LARGEST usable value seen for a key.
                try:
                    value = float(value) if value is not None else None
                except (TypeError, ValueError):
                    value = None
                if value is not None and value > 0:
                    prior = out.get(key)
                    out[key] = value if prior is None else max(prior, value)
                else:
                    out.setdefault(key, None)
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

        # R1: arming never released the PREVIOUS set, so re-arming (a second
        # circuit, or the same one after an SAE change) left the old SAEs
        # hooked forever — leaking GPU work and letting a stale layer keep
        # writing into a ring nobody reads. This makes several other R1
        # findings reachable in practice rather than in theory.
        if self.is_armed:
            self.disarm(self._armed_saes or layer_saes)

        # Commit the lag only now that we know we can arm.
        lag = next(iter(configs.values())).max_token_lag
        self._max_token_lag = lag
        # No ring is built here any more (F17). Arming is per deployment; the
        # ring is per (request, circuit) and is created by the request context
        # on first use. A ring built at arm time outlives every request that
        # uses it, which is what let one request's upstream fires match a
        # later request's downstream fire.
        armed: list[int] = []
        try:
            for layer, config in configs.items():
                layer_saes[layer].arm_edge_sensing(config)
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
        # Remember the exact SAEs armed: a later disarm must reach THESE, not
        # whatever happens to be attached at that moment.
        self._armed_saes = {lay: layer_saes[lay] for lay in configs}
        self._configs = configs
        self._unsensable = unsensable
        logger.info(
            "circuit_sensing_armed",
            circuit_id=circuit.id,
            layers=self._armed_layers,
            sensable_edges=len(
                {spec.edge_key for cfg in configs.values() for spec in cfg.edges}
            ),
            unsensable_edges=len(unsensable),
        )
        return unsensable

    def disarm(self, layer_saes: Optional[dict[int, LoadedSAE]] = None) -> None:
        # Union of what was armed and what the caller offers: an SAE swapped
        # out from under us must still be released.
        targets = dict(self._armed_saes)
        targets.update(layer_saes or {})
        for sae in targets.values():
            try:
                sae.disarm_edge_sensing()
            except Exception:
                logger.exception("circuit_sensing_disarm_failed")
        self._circuit_id = None
        self._circuit_name = None
        self._armed_layers = []
        self._configs = {}
        self._armed_saes = {}
        self._unsensable = []
        self._request_circuit_id = None
        self._request_context_tokens = 0
        # R2-01: disarm did NOT release an open boundary, so R1-03's
        # concurrency guard turned a hung request into a PERMANENT outage —
        # verified: after one request that never closed, every subsequent
        # begin was refused, and disarm + re-arm did not clear it. A guard that
        # cannot be recovered from is worse than the race it prevents.
        #
        # Disarming is an explicit operator action that ends the circuit's
        # observation entirely, so any boundary it owned is over by definition.
        if self._ctx is not None:
            try:
                self._ctx.close()
            except Exception:
                logger.exception("circuit_sensing_context_close_failed")
        self._ctx = None
        # R1-04: these survived disarm, so status reported `layers: []` with
        # `truncated_layers: [13]` — accusing a layer the armed circuit does
        # not contain, and falsifying the contract's "an empty list means every
        # armed layer reported completely". Verified by execution.
        self._last_request_truncated_layers = []
        self._request_dark_layers = []
        # R2-18: ALL per-arming counters reset together. `requests_sensed` and
        # `requests_truncated` reset while `events_recorded`, `ws_dropped` and
        # `ws_throttled` persisted, so a re-armed circuit read
        # `requests_sensed: 0, events_recorded: 99` — which the schema defines
        # as the WIRING-FAILURE signature ("zero while armed means no request
        # reached sensing at all"). A healthy re-arm looked broken.
        self._requests_sensed = 0
        self._requests_truncated = 0
        self._events_recorded = 0
        self._ws_dropped = 0
        self._ws_throttled = 0
        # R2: every other field was cleared but this one, so a circuit with no
        # override silently inherited the previous circuit's lag window.
        self._max_token_lag = settings.CIRCUIT_SENSING_MAX_TOKEN_LAG

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

        A fresh context is built per request and bound to every armed SAE, so
        nothing needs clearing: the previous request's rings went out of scope
        with its context. The old shape — one long-lived ring cleared here by
        convention — meant correctness depended on every participant
        remembering NOT to clear it (a sibling that cleared on begin wiped
        upstream fires recorded by a sibling that began first).
        """
        if not self.is_armed:
            return False
        # R3-14: `_armed_layers` without `_configs` is an inconsistent state,
        # and it began SUCCESSFULLY with context capture silently off
        # (ctx_tokens falls back to 0) and the budget on a magic literal (20,
        # not the configured value). Two silent degradations at once with
        # `paused_reason` null — an operator sees a healthy armed circuit
        # producing context-free events against a cap nobody chose.
        #
        # Refuse and say why. Sensing that cannot be configured correctly must
        # not run in a half-state.
        if self._armed_layers and not self._configs:
            logger.warning(
                "circuit_sensing_config_missing",
                layers=self._armed_layers,
                detail=(
                    "armed layers have no configs; refusing rather than "
                    "sensing with context capture off and a default cap"
                ),
            )
            self.note_paused("config_missing")
            return False
        # Snapshot the circuit this boundary belongs to. R2: record() read
        # self._circuit_id at DRAIN time, so a re-arm between begin and flush
        # persisted circuit A's observations under circuit B's id — confidently
        # wrong data, not merely lost sensing. request_id was already
        # snapshotted per-SAE; identity was not.
        self._request_circuit_id = self._circuit_id
        # R2-16: this used `_armed_layers[0]` — the same order-dependence R2-08
        # fixed for the budget cap, left behind on the sibling field one
        # expression away. Measured with configs {10: 0, 13: 32}: layer order
        # [10,13] gave 0 and [13,10] gave 32.
        #
        # The consequence is worse than the cap's. `ctx_tokens == 0` hits the
        # `k == 0` early return in `_context`, so ALL context capture — the
        # decoded window on every event row — is silently disabled by whichever
        # layer happened to sort first.
        #
        # MAX here, not min. The cap bounds a shared resource, so the most
        # restrictive layer wins; context capture is per-row enrichment, and
        # taking the min would let a single layer configured to 0 silence the
        # whole circuit. The hard ceiling still applies per config at build
        # time.
        #
        # R3-15, recorded rather than fixed: NEITHER aggregate is right in
        # principle. `context_tokens` is a per-layer setting being collapsed to
        # one request-scoped scalar, so a layer that opted out with 0 has its
        # opt-out overridden by a sibling. Not reachable today — `build_configs`
        # derives ONE `ctx` from a single circuit-level override and gives it to
        # every layer, so the values cannot diverge — but F19's per-circuit
        # configs make it live, and the honest fix is to carry the value
        # per-layer into `_context` rather than to pick a better aggregate.
        # Tracked as F19 debt; changing the shape now would be speculative.
        ctx_tokens = [
            c.context_tokens
            for c in (self._configs or {}).values()
            if c.context_tokens is not None
        ]
        self._request_context_tokens = max(ctx_tokens) if ctx_tokens else 0
        # R1-03: an already-open boundary means two generations interleaved.
        # `MAX_CONCURRENT_REQUESTS` MUST be 1 for this service to attribute
        # observations correctly, and that was enforced only by a comment in
        # config.py. Verified by execution with a second begin: the first
        # request's context was orphaned (its rings leaked, nothing would ever
        # close them), its edges were never drained, and `collect_edges` then
        # reported BOTH requests' events under the second request's id.
        #
        # Fabricated attribution on an evidence surface is categorically worse
        # than lost observations, so the second boundary is refused and the
        # reason is surfaced. Sensing degrades to "not observing, and here is
        # why", never to confidently wrong data.
        if self._ctx is not None and not self._ctx.is_closed:
            logger.warning(
                "circuit_sensing_concurrent_request_refused",
                open_request=self._request_circuit_id,
                new_request=request_id,
                detail=(
                    "a request boundary is already open — MAX_CONCURRENT_"
                    "REQUESTS must be 1 for circuit sensing to attribute "
                    "observations correctly"
                ),
            )
            self.note_paused("concurrent_request")
            # R2-01: RECLAIM the stale boundary instead of leaving it open.
            #
            # Refusing alone made a hung request a PERMANENT outage: every
            # later begin was refused and even disarm+re-arm did not clear it
            # (verified). A guard that cannot be recovered from is worse than
            # the race it prevents.
            #
            # Reclaiming is safe because generation is serialised on the
            # request queue: reaching `begin_request` at all means the previous
            # request is no longer running, so its boundary is stale rather
            # than concurrent. This request is still refused — its observations
            # would be unattributable — but the NEXT one starts clean.
            #
            # If MAX_CONCURRENT_REQUESTS is ever raised above 1, this becomes
            # the wrong call: it would then be a genuinely concurrent request
            # and reclaiming would corrupt the live one. That setting is the
            # documented invariant this whole guard exists to enforce.
            stale, self._ctx = self._ctx, None
            _stale_bound, self._bound_saes = self._bound_saes, []
            # R2-15: the stale request may have observed edges it never drained.
            # They are correctly DISCARDED rather than leaked — the next
            # request's `begin_edge_sensing_request` resets each buffer, so
            # they can never be attributed to it (verified: a later request
            # drained 0 edges and none of the stale observations). But a silent
            # discard on an evidence surface still has to be counted, or the
            # operator sees a clean circuit that quietly lost a request's data.
            undrained = 0
            for sae in list(self._armed_saes.values()):
                try:
                    undrained += len(getattr(sae, "_sensed_edges", ()) or ())
                except Exception:
                    pass
            if undrained:
                self._requests_truncated += 1
                self._request_counted_truncated = True
                logger.warning(
                    "circuit_sensing_stale_observations_discarded",
                    events=undrained,
                    stale_request=getattr(stale, "request_id", None),
                    detail=(
                        "a request boundary was reclaimed before its "
                        "observations were drained; they are discarded, never "
                        "attributed to another request"
                    ),
                )
            try:
                stale.close()
            except Exception:
                logger.exception("circuit_sensing_stale_close_failed")
            # R3-02: unbind the union of the ARMED set and the caller's map.
            # An SAE swapped out of `_armed_saes` since begin kept a reference
            # to the dead context and then self-bound a PRIVATE solo one on its
            # next begin — sensing into a ring no sibling can read. `disarm`
            # already unions for exactly this reason; the reclaim path did not.
            for sae in self._unbind_targets(layer_saes, extra=_stale_bound):
                try:
                    sae.bind_context(None)
                except Exception:
                    logger.exception("circuit_sensing_stale_unbind_failed")
            return False

        # One context for this request, shared by every armed layer. Built over
        # the armed circuit SET rather than a single id: Feature 19 lifts the
        # single-active invariant, and building for one circuit now would
        # repeat the assumption this feature exists to remove.
        # R2-08: the circuit budget used `_armed_layers[0]` — whichever layer
        # sorts first. Verified with divergent configs: {10:5, 20:500} gave 5,
        # and reordering to [20,10] gave 500. Not reachable today (every config
        # takes the same setting), but it is precisely the order-dependence
        # R1-15 fixed for the ambient count, one function away, and F19's
        # per-circuit configs make it live.
        #
        # MIN, not first: the budget bounds the whole circuit, so the most
        # restrictive layer's intent is the safe reading.
        caps = [
            c.max_events_per_request
            for c in (self._configs or {}).values()
            if c.max_events_per_request is not None
        ]
        cap = min(caps) if caps else 20
        # R2-09: `cap=0` meant "armed, latched, and reporting truncation on
        # every layer" — sensing that looks on and observes nothing. A
        # misconfigured zero should mean OFF. There is no lower-bound
        # validation on CIRCUIT_SENSING_MAX_EVENTS_PER_REQUEST, so clamp here
        # and say so rather than degrade into a confusing half-state.
        if cap < 1:
            logger.warning(
                "circuit_sensing_cap_too_low",
                cap=cap,
                detail=(
                    "max_events_per_request below 1 would arm sensing and then "
                    "immediately truncate every layer; clamped to 1 — set the "
                    "circuit inactive to disable sensing"
                ),
            )
            cap = 1
        ctx = EdgeSensingRequestContext(
            request_id=request_id,
            circuit_ids=frozenset(
                {self._request_circuit_id} if self._request_circuit_id else set()
            ),
            cap=cap,
        )
        self._ctx = ctx
        # R2-10: tell the ring how many layers will report, so it prunes to the
        # slowest without guessing from a count. Without this a single-layer
        # circuit never pruned (512 retained fires per edge instead of 4), and
        # naively dropping the old `len < 2` guard let the FIRST layer to
        # report prune past fires the second still needed — R1-01 again.

        # A new boundary: reasons from earlier requests are stale from here.
        self._pause_is_current = False
        # R2-06: and so is the previous request's truncation report. It was
        # only rebuilt at `collect_edges`, so for the whole span between begin
        # and drain the status named LAST request's dark layers as
        # untrustworthy — accusing a layer that has fully recovered. Given the
        # field's contract ("empty means every armed layer reported
        # completely"), accusing a healthy layer is the same class of
        # dishonesty as R1-04's stale-after-disarm, inverted.
        self._last_request_truncated_layers = []
        self._request_dark_layers = []
        self._request_counted_truncated = False
        began = False
        # R1-02: a layer that cannot be bound is DARK for this request, and
        # `began` used to be True if ANY layer began. Verified by execution
        # with one layer absent from `layer_saes`: it was never bound, never
        # begun, observed nothing — and the drain reported
        # `truncated_layers: []`, which this service's own status contract
        # defines as "every armed layer reported completely". The operator was
        # told the circuit was quiet while half of it was blind.
        #
        # Reachable whenever `_circuit_sensing_layer_saes()` drops a layer:
        # two SAEs on one layer (ambiguous), or a detach between arm and
        # request. A false completeness claim is worse than a missing one, so
        # the dark layers are named in `truncated_layers` and the reason is
        # surfaced.
        dark: list[int] = []
        for layer in self._armed_layers:
            sae = layer_saes.get(layer)
            if sae is None or not sae.is_edge_sensing_armed:
                dark.append(layer)
                continue
            sae.bind_context(ctx)
            sae.begin_edge_sensing_request(request_id)
            # Remember it: this is the authoritative record of who holds the
            # context, and it survives a later swap of `_armed_saes` (R3-02).
            self._bound_saes.append(sae)
            began = True
        self._request_dark_layers = sorted(dark)
        if dark:
            logger.warning(
                "circuit_sensing_layer_unavailable",
                layers=self._request_dark_layers,
                request=request_id,
                detail=(
                    "armed layers had no usable SAE at request time; their "
                    "view of this request is incomplete"
                ),
            )
            self.note_paused("layer_unavailable")
        # R2-12: tell the ring how many layers will ACTUALLY report, which is
        # the ones that began — not the armed count. A DARK layer never reports
        # progress, so expecting it made pruning wait forever: measured, a
        # circuit with one dark sibling retained 512 fires instead of 8. R2-10
        # (prune to the slowest) colliding with R1-02 (a layer can be dark).
        if began and self._request_circuit_id:
            try:
                ctx.ring(
                    self._request_circuit_id, self._max_token_lag
                ).expect_layers(len(self._armed_layers) - len(dark))
            except Exception:
                logger.exception("circuit_sensing_expect_layers_failed")
        if not began:
            # R2-04/R2-05: no layer opened, so there is no boundary. Leaving
            # the context assigned orphaned it — nothing closes it, because the
            # caller returns None on False and `_notify_circuit_sensing` early-
            # returns — and the NEXT request was then refused by the
            # concurrency guard. One healthy request lost per orphan, flapping
            # forever under a persistently dark condition.
            #
            # And the count must not include it: `requests_sensed` promises
            # "ZERO while armed means no request reached sensing at all", so
            # counting a boundary that observed nothing reports activity on
            # exactly the wiring-failure path the field exists to expose.
            try:
                ctx.close()
            except Exception:
                logger.exception("circuit_sensing_context_close_failed")
            self._ctx = None
            return False
        self._requests_sensed += 1
        return began

    def collect_edges(
        self, layer_saes: dict[int, LoadedSAE]
    ) -> tuple[str, list[Any], bool]:
        """Drain every armed layer and merge, ordered by downstream position.

        Overhead is SUMMED across the SAEs: one circuit request touches N
        hooks, so a per-SAE assignment would under-report by a factor of N.
        """
        # Which SAEs actually opened this boundary. `_edge_began` is already
        # cleared by `collect_sensed_edges` below, so the record kept at bind
        # time (R3-02) is the only reliable answer here (R3-11).
        begun = {id(s) for s in self._bound_saes}
        request_id = ""
        merged: list[Any] = []
        truncated = False
        truncated_layers: list[int] = []
        overhead = 0.0
        member_fires = 0
        for layer in self._armed_layers:
            sae = layer_saes.get(layer)
            if sae is None:
                continue
            rid, edges, trunc = sae.collect_sensed_edges()
            request_id = request_id or rid
            merged.extend(edges)
            truncated = truncated or trunc
            # BR-006: WHICH layer truncated, not merely that something did. A
            # request-wide boolean tells an operator their view is incomplete
            # without telling them whether the gap is where they are looking —
            # so a layer that observed everything is indistinguishable from one
            # that dropped events, and the honest reading of any empty result
            # becomes "maybe".
            if trunc:
                truncated_layers.append(layer)
            # R1: zeroed only at begin, so a layer that missed begin (the
            # `continue` in begin_request) re-contributed its stale overhead to
            # the next request, inflating the number that drives the warning.
            #
            # R3-11: and R1's fix zeroed it AFTER summing, so the stale value
            # was still counted ONCE before being cleared — the very number the
            # comment says it prevents. Measured: a layer absent from begin
            # carried 8.0ms into the next request's total and tripped
            # `circuit_sensing_overhead_high`. Only layers that BEGAN this
            # request contribute; a layer that missed begin has no overhead
            # belonging to this request, so its stale value is discarded rather
            # than summed.
            layer_overhead = float(getattr(sae, "_edge_overhead_ms", 0.0) or 0.0)
            # Belt and braces: `_reset_edge_buffer` already clears this at
            # begin for any layer that begins, so mutating this line away is
            # harmless TODAY. Kept because a layer that misses begin has no
            # other point of clearing, and its stale value would otherwise grow
            # without bound across requests. Recorded rather than pinned — a
            # test asserting a redundant line would pass for the wrong reason.
            sae._edge_overhead_ms = 0.0
            if id(sae) in begun:
                overhead += layer_overhead
            # R3-12: the identical missed-begin bug as the overhead above,
            # one line down. A layer absent from begin never runs
            # `_reset_edge_buffer`, so its stale fire count was re-counted
            # against this request — measured: 42 fires from request 1 reported
            # again under request 2. Only layers that BEGAN contribute.
            layer_fires = int(getattr(sae, "_edge_member_fires", 0) or 0)
            sae._edge_member_fires = 0
            if id(sae) in begun:
                member_fires += layer_fires
        self._last_request_overhead_ms = overhead
        self._last_request_member_fires = member_fires
        if overhead > settings.CIRCUIT_SENSING_MAX_OVERHEAD_MS:
            logger.warning(
                "circuit_sensing_overhead_high",
                overhead_ms=round(overhead, 2),
                threshold_ms=settings.CIRCUIT_SENSING_MAX_OVERHEAD_MS,
                layers=self._armed_layers,
            )
        merged.sort(key=lambda e: (e.down_pos, e.up_pos))
        # The tuple stays THREE wide on purpose. Widening it would break every
        # `a, b, c = collect_edges(...)` call site — silently by position at
        # the two that ignore the flag — and task 5.4 requires the integration
        # workflow to pass unchanged as the outside-boundary preservation
        # proof. The per-layer detail rides alongside instead, read by the
        # status route.
        # A layer that was DARK for this request is incomplete for exactly the
        # same operator-facing reason as one that shed: its view is partial.
        # Merging them here keeps `truncated_layers` a single honest answer to
        # "which layers should I not trust for this request".
        # R3-13: the CIRCUIT budget is read here too. `EventBudget.
        # truncated_layers()` had ZERO production readers — R1-07 and R2-03
        # both did real work keeping the two truncation sources in agreement,
        # and one of them was never consulted. A source of truth nobody reads
        # is not a source of truth; it is the declared-but-unwired pattern this
        # arc has produced five times.
        #
        # Union of all three: layers that shed (per-SAE flag), layers the
        # shared budget refused, and layers that were dark. Each is a distinct
        # way a layer's view can be incomplete, and the operator's question is
        # the same for all of them — "which layers should I not trust".
        budget_truncated: set[int] = set()
        ctx = self._ctx
        if ctx is not None and self._request_circuit_id:
            try:
                budget_truncated = set(
                    ctx.budget.truncated_layers(self._request_circuit_id)
                )
            except Exception:
                logger.exception("circuit_sensing_budget_truncation_read_failed")
        self._last_request_truncated_layers = sorted(
            set(truncated_layers)
            | set(self._request_dark_layers)
            | budget_truncated
        )
        # R3-07: count once per REQUEST, not once per drain. `collect_edges`
        # can run more than once for one boundary (a retry, a second flush),
        # and `_request_dark_layers` is reset only at begin — so one request
        # with a dark layer, drained three times, read 1 -> 2 -> 3. A counter
        # on an evidence surface manufacturing data loss that never happened.
        #
        # R2-20 tested the drain-vs-reclaim pair for double-counting and missed
        # this one: the same defect through the repeated-drain door.
        if self._last_request_truncated_layers and not self._request_counted_truncated:
            self._request_counted_truncated = True
            self._requests_truncated += 1
        return request_id, merged, truncated

    @property
    def last_request_truncated_layers(self) -> list[int]:
        """Layers that dropped events in the last drained request (BR-006).

        Empty means every armed layer reported completely — which is a
        different statement from "no events were observed", and the reason
        this is a list of layers rather than a boolean.
        """
        return list(self._last_request_truncated_layers)

    def _unbind_targets(self, layer_saes, extra=None):
        """Every SAE that might hold this service's context.

        The union of the ARMED set, the caller's map, and any SAE bound during
        the request. `_armed_saes` alone is not enough: an SAE swapped out by a
        re-arm keeps its reference, and on its next begin self-binds a private
        solo context whose observations no sibling can read (R3-02). `disarm`
        already unioned for this reason; the request paths did not.
        """
        targets = dict(self._armed_saes)
        targets.update(layer_saes or {})
        # ...and every SAE this service actually BOUND for the open request.
        # `close_request` takes no map, so without this record a swapped-out
        # SAE is unreachable from either source and keeps a closed context.
        seen = {id(s): s for s in targets.values()}
        for sae in list(self._bound_saes) + list(extra or ()):
            seen.setdefault(id(sae), sae)
        return list(seen.values())

    def close_request(self) -> None:
        """Release the boundary snapshot. R3: _request_circuit_id survived both
        collect_edges and disarm, so a drain arriving after a disarm attributed
        rows to a circuit that was no longer armed — R2-04 narrowed the
        mis-attribution window without closing it."""
        self._request_circuit_id = None
        self._request_context_tokens = 0
        # Close the context and unbind every SAE. Closing makes a late write
        # from a hung generate thread log and return -1 rather than land in the
        # next request's accounting (CTX-L2, EC-17.5); unbinding means a stale
        # reference cannot resurrect a closed context's rings. Both, because
        # either alone leaves a path where the next request inherits this one's
        # state — the failure mode this whole feature exists to remove.
        ctx, self._ctx = self._ctx, None
        bound, self._bound_saes = self._bound_saes, []
        if ctx is not None:
            try:
                ctx.close()
            except Exception:
                logger.exception("circuit_sensing_context_close_failed")
        # R3-02: same union as the reclaim path and `disarm` — an SAE swapped
        # out since begin must still be released, or it keeps a closed context.
        # `bound` is the authoritative list: it was captured above, before
        # `_bound_saes` was cleared, and it is the only record that survives a
        # re-arm swapping `_armed_saes` under us.
        for sae in self._unbind_targets(None, extra=bound):
            try:
                sae.bind_context(None)
            except Exception:
                logger.exception("circuit_sensing_unbind_failed")

    # F17 task 3.5: `prune_ring`, `safe_prune_boundary` and
    # `prune_between_passes` were DELETED here. They were R2's design for
    # request-level pruning and had ZERO production callers — R2 fixed R1's
    # "declared a mechanism and never wired it" finding by declaring a
    # mechanism and never wiring it. R3 superseded both with the ring tracking
    # layer progress itself (`note_layer_progress`), which works precisely
    # because no caller needs to know about siblings. Carrying two pruning
    # designs, one live and one dead, is how the next reader picks the wrong
    # one.

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

        # Use the identity captured when the boundary OPENED — see
        # begin_request. Reading self._circuit_id here would attribute these
        # observations to whatever is armed NOW.
        circuit_id = self._request_circuit_id or self._circuit_id
        if not circuit_id or not edges:
            return []
        ctx_tokens = self._request_context_tokens

        # EDGE-R2 alone-vs-within, to the SAME contract Feature 11 uses:
        # whole-SAE fired count, ONLY when un-compacted monitoring co-ran,
        # NULL otherwise. Never estimated — a number that looked like the
        # signal but measured the circuit's own members would be compared
        # against F11 rows as though it were the same quantity.
        ambient = self._ambient_fired_count()

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
                    ambient_fired_count=ambient,
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

    def _ambient_fired_count(self) -> Optional[int]:
        """Whole-SAE fired count for the alone-vs-within signal, or None.

        Mirrors ``InferenceService._ambient_counts`` (Feature 11) and the
        ``millm_sensing_events`` MCP contract: this is how many features fired
        across the ENTIRE SAE, which is only knowable when un-compacted
        monitoring co-ran. Anything else stays None — **never estimated**.

        The circuit's own member-fire total is deliberately NOT used here: it
        answers "how busy was this circuit", not "was this firing distinctive
        against everything else", and writing it to this column would make an
        F15 row incomparable with an F11 row carrying the same field name.
        """
        # R1-15: this returned the FIRST armed layer's count. With two
        # monitored layers the answer depended silently on layer ordering —
        # measured, the identical state produced 3 or 9 purely by reordering
        # `_armed_layers`. The field is documented as the count across the
        # ENTIRE SAE and "never estimated"; an arbitrary layer's count IS an
        # estimate, and one that changes under a stable state.
        #
        # A circuit spans layers, so "the entire SAE" has no single answer
        # here. Answer only when exactly ONE layer can supply it; when several
        # can, they are different SAEs measuring different feature spaces and
        # picking one would be a fabricated number on a comparison column that
        # F11 rows also use. None means "not knowable", which is the contract.
        counts: list[int] = []
        for layer in self._armed_layers:
            sae = self._armed_saes.get(layer)
            if sae is None:
                continue
            try:
                if (not getattr(sae, "is_monitoring_enabled", False)
                        or getattr(sae, "_monitored_features", None) is not None):
                    continue
                acts = sae.get_feature_activations_for_item(0)
                if acts is None:
                    continue
                counts.append(int((acts[-1] > 0).sum().item()))
            except Exception:
                continue
        if len(counts) == 1:
            return counts[0]
        if len(counts) > 1:
            logger.info(
                "circuit_ambient_count_ambiguous",
                layers=len(counts),
                detail=(
                    "several armed layers can supply an ambient count; they "
                    "measure different feature spaces, so none is reported"
                ),
            )
        return None

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
        """Broadcast at most _WS_MAX_PER_FLUSH events per flush.

        Everything not delivered is COUNTED. R1: an emit failure was swallowed
        without incrementing ws_dropped, so status reported events recorded,
        ws_dropped=0, and the UI showed nothing — the discrepancy was
        unobservable, which is the "silently dark" mode this feature exists to
        avoid.
        """
        if not payloads:
            return
        self.should_emit()
        # R1-14: `undelivered` was `len(payloads) - sent`, but the loop only
        # ever ATTEMPTS the last `_WS_MAX_PER_FLUSH`. A perfectly healthy
        # 20-event flush therefore reported `ws_dropped: 15` — measured. That
        # conflates the intentional per-flush cap with delivery FAILURE, and
        # raises a dropped-events alarm on a system that is working. The
        # counter exists to make a real discrepancy observable; inflating it on
        # the happy path destroys exactly that signal.
        #
        # `ws_dropped` now counts only what was ATTEMPTED and did not land. The
        # events the cap declined to send are a deliberate throttle, not a
        # loss: they are persisted and readable through the events API.
        attempted = payloads[-self._WS_MAX_PER_FLUSH :]
        # R2-11: R1-14 correctly stopped counting these as DROPPED — they are
        # persisted and readable through the events API, so nothing is lost.
        # But it left them invisible: with a 5-per-flush cap and a 20-event
        # request, 75% of a busy request's events never reach the live panel
        # and no field said so. An operator comparing the panel with the events
        # API had no way to explain the gap.
        throttled = len(payloads) - len(attempted)
        if throttled > 0:
            self._ws_throttled += throttled
        sent = 0
        try:
            from millm.sockets.progress import progress_emitter

            # R2: this kept the FIRST 5. collect_edges sorts by down_pos, so
            # the panel always showed a request's EARLIEST edges and never its
            # most recent — the opposite of the ring's own "recent history is
            # what matters" policy, and wrong for a live-observation surface.
            for payload in attempted:
                progress_emitter.emit_circuit_sensing_event(payload)
                sent += 1
        except Exception:
            logger.warning("circuit_sensing_emit_failed", exc_info=False)
        undelivered = len(attempted) - sent
        if undelivered > 0:
            self.note_ws_dropped(undelivered)

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
        # R1: this counted only the LOWEST armed layer's upstream specs, so a
        # circuit whose edges all flow from a higher layer (L13->L20) reported
        # sensable_edges=0 while sensing perfectly — an operator would read
        # that as "sensing is broken". Count DISTINCT edge keys across layers.
        sensable = len(
            {
                spec.edge_key
                for cfg in self._configs.values()
                for spec in cfg.edges
            }
        )
        return {
            "armed": armed,
            "paused_reason": self._paused_reason,
            "circuit_id": self._circuit_id,
            "circuit_name": self._circuit_name,
            "layers": list(self._armed_layers),
            "sensable_edges": sensable,
            "unsensable_edges": [u.to_dict() for u in self._unsensable],
            "max_token_lag": self._max_token_lag,
            "last_request_overhead_ms": round(self._last_request_overhead_ms, 3),
            # BR-006. An empty list means every armed layer reported
            # completely — a different claim from "no events were observed",
            # which a bare `truncated: true/false` could not make. Named
            # layers let an operator tell "the circuit was quiet" from "this
            # layer's view is incomplete".
            "truncated_layers": list(self._last_request_truncated_layers),
            # R1-06: distinguishes "quiet traffic" from "sensing never ran".
            # Zero while armed means no request reached the boundary at all —
            # check `paused_reason`.
            "requests_sensed": self._requests_sensed,
            # Cumulative since arming. `truncated_layers` only describes the
            # LAST drained request and is superseded by the next one, so a rare
            # truncation could otherwise vanish before an operator polls.
            "requests_truncated": self._requests_truncated,
            "events_recorded": self._events_recorded,
            "ws_dropped": self._ws_dropped,
            # Declined by the per-flush cap, NOT lost — they are in the events
            # API. Non-zero here with ws_dropped zero means "the panel is
            # showing a sample of a busy request", which is working as designed.
            "ws_throttled": self._ws_throttled,
        }

    def note_paused(self, reason: Optional[str]) -> None:
        """Record why an armed circuit is not observing (or None to clear)."""
        self._paused_reason = reason
        # Reasons set for the CURRENT request must survive `clear_stale_pause`
        # (R2-02). A bare `note_paused(None)` from the caller was wiping the
        # `layer_unavailable` this same request had just recorded.
        #
        # R2-14: "current" means set while a boundary is OPEN, not merely set.
        # This used to be `reason is not None`, so a reason recorded OUTSIDE a
        # request — the speculative-decoding and no-attached-SAEs skips, which
        # by definition never open one — survived the next request's clear and
        # showed one request late. Found by attacking R2-02's own fix.
        self._pause_is_current = reason is not None and self._ctx is not None

    def clear_stale_pause(self) -> None:
        """Clear a pause reason left over from a PREVIOUS request.

        R2-02: the caller used `note_paused(None)` on the success path, which
        cleared unconditionally. `begin_request` returns True when SOME layers
        began, so a partially dark circuit reached that line and had its
        `layer_unavailable` reason erased — R1-06's "say why sensing is
        degraded" deleted by R1-02's "say which layers are dark".

        A reason recorded during THIS request is current, not stale, and stays.
        """
        if getattr(self, "_pause_is_current", False):
            self._pause_is_current = False
            return
        self._paused_reason = None

    def note_events_recorded(self, count: int) -> None:
        self._events_recorded += int(count)

    def should_emit(self) -> bool:
        """Kept for compatibility; the flush-level time throttle is gone.

        R1: this dropped ENTIRE flushes inside the interval, so two back-to-back
        requests lost the second one's events wholesale — an F15 invention that
        diverged from Feature 11, which only caps count-per-flush and never
        discards a flush on timing. Live observation is the point of the panel;
        silently withholding a whole request's events defeats it.
        """
        self._last_ws_emit_ts = time.monotonic()
        return True

    def note_ws_dropped(self, count: int) -> None:
        self._ws_dropped += int(count)
