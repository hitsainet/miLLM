"""
LoadedSAE wrapper for inference.

Handles encoding, decoding, steering, and monitoring for attached SAEs.

Steering Formula (miStudio/Neuronpedia compatible):
    modified_activations = original_activations + Σ(strength_i × decoder_direction_i)

Where decoder_direction_i = W_dec[feature_idx_i, :] is the decoder column for feature i.
This applies steering directly to the residual stream, uniformly to all token positions.
"""

import logging
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator, Optional

import torch
from torch import Tensor

# The edge-sensing primitives are DEFINED in millm.ml.edge_sensing and
# re-exported here.
#
# EdgeSpec / SensedEdge / EdgeFireRing and the two shed constants used to be
# declared in BOTH modules with identical field names — two definitions that
# read as interchangeable, were not the same type, and would have diverged the
# moment either gained a field. Nine call sites import these names from this
# module, so the names stay put while the definitions do not.
from millm.ml.edge_sensing import (
    _EDGE_FIRE_BUDGET_MIN,
    _EDGE_SHED_POSITIONS_PER_COL,
    EdgeFireRing,
    EdgeSensingRequestContext,
    EdgeSpec,
    SensedEdge,
    _match_edges_impl,
)
from millm.ml.sae_config import SAEConfig

logger = logging.getLogger(__name__)


@dataclass
class SensingConfig:
    """Armed co-activation sensing parameters (Feature 11).

    thresholds holds theta_i = max(theta_floor, epsilon * max_activation_i)
    per member, aligned with member_indices. threshold_mode records whether
    the epsilon*max rule could be applied ('epsilon_max') or every member
    lacked max_activation ('floor_only') — surfaced in the status API
    (EC-11.4) so operators know the thresholds are degraded.
    """

    profile_id: str
    member_indices: list[int]              # <= 20 (contract cap)
    thresholds: Tensor                     # (m,)
    threshold_mode: str                    # 'epsilon_max' | 'floor_only'
    min_k: int
    context_tokens: int
    max_events_per_request: int = 20


@dataclass
class CircuitSensingConfig:
    """Armed edge-sensing parameters for ONE SAE of a circuit (Feature 15).

    ``member_indices`` are this SAE's own features that participate in any
    sensable edge; ``thresholds`` is aligned with it, built by the same
    theta = max(floor, epsilon * max_activation) rule Feature 11 uses.
    """

    circuit_id: str
    layer: int
    member_indices: list[int]
    thresholds: Tensor                     # (m,)
    threshold_mode: str                    # 'epsilon_max' | 'floor_only'
    edges: list[EdgeSpec]
    max_token_lag: int
    context_tokens: int
    max_events_per_request: int = 20



@dataclass
class SensedHit:
    """A debounced co-activation span within one request.

    Positions are ABSOLUTE token indices from the start of the request's
    sequence (prompt + generated); the event attaches to the token being
    READ at each position — the sampled token is unknowable in-pass.
    fired carries (REAL feature_idx, peak activation) pairs.
    """

    pos_start: int
    pos_end: int
    phase: str                             # 'prefill' | 'decode'
    fired: list[tuple[int, float]]
    fired_count: int
    score: float                           # max(act_i / theta_i) over fired


class LoadedSAE:
    """
    Loaded SAE with encoder and decoder weights.

    Implements direct residual stream steering (miStudio/Neuronpedia compatible)
    and optional SAE encode/decode for monitoring.

    Steering approach:
    - Direct steering: Add steering delta directly to hidden states
    - Delta = Σ (strength × decoder_column) for all configured features
    - Applied uniformly to ALL token positions
    - Neuronpedia-compatible strength semantics (0=none, 1=1x, 80=strong)

    Thread-safety notes:
    - Forward pass is thread-safe (no mutation)
    - Steering modification should use external lock if concurrent
    - Monitoring capture creates new tensor (safe)

    Memory layout:
    - W_enc: (d_in, d_sae) - encoder weights
    - b_enc: (d_sae,) - encoder bias
    - W_dec: (d_sae, d_in) - decoder weights
    - b_dec: (d_in,) - decoder bias

    Attributes:
        W_enc: Encoder weight matrix.
        b_enc: Encoder bias vector.
        W_dec: Decoder weight matrix.
        b_dec: Decoder bias vector.
        config: SAE configuration.
        device: Current device (cpu/cuda).
        d_in: Input dimension (hidden_size).
        d_sae: SAE feature dimension.
    """

    def __init__(
        self,
        W_enc: Tensor,
        b_enc: Tensor,
        W_dec: Tensor,
        b_dec: Tensor,
        config: SAEConfig,
        device: str = "cpu",
    ) -> None:
        """
        Initialize LoadedSAE with weight tensors.

        Args:
            W_enc: Encoder weights (d_in, d_sae).
            b_enc: Encoder bias (d_sae,).
            W_dec: Decoder weights (d_sae, d_in).
            b_dec: Decoder bias (d_in,).
            config: SAE configuration.
            device: Target device.

        Raises:
            AssertionError: If tensor dimensions don't match config.
        """
        self.W_enc = W_enc.to(device)
        self.b_enc = b_enc.to(device)
        self.W_dec = W_dec.to(device)
        self.b_dec = b_dec.to(device)
        self.config = config
        self.device = device

        # Extract dimensions from weights
        self.d_in = W_enc.shape[0]
        self.d_sae = W_enc.shape[1]

        # Validate dimensions match config
        assert self.d_in == config.d_in, (
            f"d_in mismatch: weights have {self.d_in}, config has {config.d_in}"
        )
        assert self.d_sae == config.d_sae, (
            f"d_sae mismatch: weights have {self.d_sae}, config has {config.d_sae}"
        )

        # When True, the forward hook applies neither steering nor monitoring
        # capture.  Used to run "plain" forward passes (e.g. /v1/embeddings)
        # through the same hooked model without perturbing their hidden states
        # or clobbering the last-captured activations.  See suppressed().
        self._suppressed: bool = False

        # Steering state (direct residual stream steering)
        self._steering_values: dict[int, float] = {}
        self._steering_enabled: bool = False
        # Pre-computed steering delta in residual stream space (d_in,)
        self._steering_delta: Optional[Tensor] = None

        # Observability: counts how many times apply_steering actually applied a
        # non-trivial steering delta.  Used to verify that the forward hook is
        # firing (e.g. that torch.compile did not silently bypass it).  Not
        # thread-locked — it is a best-effort diagnostic counter.
        self._steering_apply_count: int = 0

        # Sensing state (Feature 11) — armed-only observation path,
        # deliberately independent of monitoring (which compacts columns
        # positionally and keeps only the last pass).
        self._sensing: Optional[SensingConfig] = None
        self._sensing_thresholds_cpu: list[float] = []
        self._W_enc_m: Optional[Tensor] = None
        self._b_enc_m: Optional[Tensor] = None
        self._sensed_hits: list[SensedHit] = []
        self._sensing_token_offset: int = 0
        self._sensing_phase: str = "prefill"
        self._sensing_done: bool = False
        self._sensing_truncated: bool = False
        self._sensing_began: bool = False
        self._sensing_request_id: str = ""
        # History-dedup boundary (goal item 2): positions BELOW this were
        # part of a previous request's sequence (longest common prefix) and
        # were already reported then — sensed passes still advance offsets,
        # but hits anchored before the boundary are suppressed.
        self._sensing_report_from: int = 0
        self._sensing_batch_warned: bool = False
        # Cumulative per-request overhead accumulator (ms) — read by the
        # sensing status endpoint (SEN-S2); reset at begin.
        self._sensing_overhead_ms: float = 0.0

        # --- Feature 15: circuit edge sensing (independent of F11 above) ---
        self._edge_sensing: Optional["CircuitSensingConfig"] = None
        self._W_enc_e: Optional[Tensor] = None
        self._b_enc_e: Optional[Tensor] = None
        self._sensed_edges: list["SensedEdge"] = []
        # The request-scoped context (F17). Owns the per-circuit rings; the
        # ring is reached through the `_edge_ring` property below rather than
        # stored, so there is no second copy to go stale when a request ends.
        self._edge_ctx: Optional["EdgeSensingRequestContext"] = None
        self._edge_token_offset: int = 0
        self._edge_phase: str = "prefill"
        self._edge_done: bool = False
        self._edge_truncated: bool = False
        self._edge_began: bool = False
        self._edge_request_id: str = ""
        self._edge_batch_warned: bool = False
        self._edge_saturation_warned: bool = False
        self._edge_member_fires: int = 0
        self._edge_overhead_ms: float = 0.0

        # Monitoring state
        self._monitoring_enabled: bool = False
        self._monitored_features: Optional[list[int]] = None
        # Per-batch-item activations: index i → (seq_len, d_sae) tensor for request i.
        # Serial path always produces a list of one item.
        # CBM batches produce one item per batched request (index ≠ request_id).
        self._last_feature_acts_per_item: list[Tensor] = []

        logger.debug(
            f"LoadedSAE initialized: d_in={self.d_in}, d_sae={self.d_sae}, "
            f"device={device}"
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass through SAE (encode -> decode).

        Note: This performs SAE reconstruction but NOT steering.
        Steering is applied directly via apply_steering() for miStudio compatibility.

        Args:
            x: Input activations (batch, seq_len, d_in).

        Returns:
            Reconstructed activations (batch, seq_len, d_in).
        """
        # Store original dtype for output conversion
        original_dtype = x.dtype

        # Cast input to SAE weight dtype if different
        if x.dtype != self.W_enc.dtype:
            x = x.to(self.W_enc.dtype)

        # Encode: x @ W_enc + b_enc with ReLU
        feature_acts = torch.relu(x @ self.W_enc + self.b_enc)

        # Capture for monitoring
        if self._monitoring_enabled:
            self._capture_activations(feature_acts)

        # Decode: feature_acts @ W_dec + b_dec
        reconstructed = feature_acts @ self.W_dec + self.b_dec

        # Cast back to original dtype
        if reconstructed.dtype != original_dtype:
            reconstructed = reconstructed.to(original_dtype)

        return reconstructed

    def apply_steering(self, hidden_states: Tensor) -> Tensor:
        """
        Apply direct residual stream steering (miStudio/Neuronpedia compatible).

        Formula: modified = original + Σ(strength_i × decoder_direction_i)

        This adds the steering delta uniformly to ALL token positions.
        The steering delta is pre-computed from decoder columns.

        Args:
            hidden_states: Model activations (batch, seq_len, d_in).

        Returns:
            Modified activations with steering applied.
        """
        if self._suppressed or not self._steering_enabled or self._steering_delta is None:
            return hidden_states

        # Ensure steering delta matches hidden states dtype/device.
        # On the first call with a new dtype/device (e.g. after the SAE weights
        # were cast to bfloat16 at attach time), update the cached delta in place
        # so that every subsequent call is a no-op cast — one cast total instead
        # of one cast per token.
        delta = self._steering_delta
        if delta.device != hidden_states.device or delta.dtype != hidden_states.dtype:
            self._steering_delta = delta.to(
                device=hidden_states.device, dtype=hidden_states.dtype
            )
            delta = self._steering_delta

        # Broadcast delta to all tokens: [d_in] -> [1, 1, d_in] -> [batch, seq_len, d_in]
        # Use in-place add for Gemma-2 compatibility (some architectures require this)
        batch_size, seq_len, _ = hidden_states.shape
        delta_expanded = delta.unsqueeze(0).unsqueeze(0).expand(batch_size, seq_len, -1)

        # Add the steering delta to every token's residual stream.  This is an
        # out-of-place add (a new tensor is returned); the hook substitutes it
        # for the layer's original output.
        hidden_states = hidden_states + delta_expanded

        self._steering_apply_count += 1

        return hidden_states

    def get_decoder_direction(self, feature_idx: int) -> Tensor:
        """
        Get the decoder direction (column) for a feature.

        This is the direction in residual stream space that the feature represents.

        Args:
            feature_idx: Feature index (0 to d_sae-1).

        Returns:
            Decoder direction vector (d_in,).

        Raises:
            ValueError: If feature_idx is out of range.
        """
        if not 0 <= feature_idx < self.d_sae:
            raise ValueError(
                f"Feature index {feature_idx} out of range [0, {self.d_sae})"
            )
        # W_dec shape is (d_sae, d_in), so W_dec[feature_idx, :] gives (d_in,)
        return self.W_dec[feature_idx, :]

    def encode(self, x: Tensor) -> Tensor:
        """
        Encode activations to feature space.

        Useful for monitoring and analysis without full reconstruction.

        Args:
            x: Input activations (batch, seq_len, d_in).

        Returns:
            Feature activations (batch, seq_len, d_sae).
        """
        return torch.relu(x @ self.W_enc + self.b_enc)

    def decode(self, feature_acts: Tensor) -> Tensor:
        """
        Decode feature activations to input space.

        Args:
            feature_acts: Feature activations (batch, seq_len, d_sae).

        Returns:
            Reconstructed activations (batch, seq_len, d_in).
        """
        return feature_acts @ self.W_dec + self.b_dec

    # ==========================================================================
    # Steering Methods
    # ==========================================================================

    def set_steering(self, feature_idx: int, value: float) -> None:
        """
        Set steering value for a feature.

        Args:
            feature_idx: Feature index (0 to d_sae-1).
            value: Steering strength (positive=amplify, negative=suppress).

        Raises:
            ValueError: If feature_idx is out of range.
        """
        if not 0 <= feature_idx < self.d_sae:
            raise ValueError(
                f"Feature index {feature_idx} out of range [0, {self.d_sae})"
            )

        self._steering_values[feature_idx] = value
        self._rebuild_steering_delta()

    def set_steering_batch(self, steering: dict[int, float]) -> None:
        """
        Set multiple steering values at once.

        Args:
            steering: Dictionary mapping feature indices to steering values.

        Raises:
            ValueError: If any feature index is out of range.
        """
        for idx in steering.keys():
            if not 0 <= idx < self.d_sae:
                raise ValueError(f"Feature index {idx} out of range [0, {self.d_sae})")

        for idx, val in steering.items():
            self._steering_values[idx] = val
        self._rebuild_steering_delta()

    def clear_steering(self, feature_idx: Optional[int] = None) -> None:
        """
        Clear steering for one or all features.

        Args:
            feature_idx: Specific feature to clear (None = clear all).
        """
        if feature_idx is None:
            self._steering_values.clear()
        elif feature_idx in self._steering_values:
            del self._steering_values[feature_idx]

        self._rebuild_steering_delta()

    def get_steering_values(self) -> dict[int, float]:
        """Get current steering values (copy)."""
        return dict(self._steering_values)

    def enable_steering(self, enabled: bool = True) -> None:
        """Enable or disable steering application."""
        self._steering_enabled = enabled

    @property
    def is_steering_enabled(self) -> bool:
        """Check if steering is enabled."""
        return self._steering_enabled

    @property
    def steering_apply_count(self) -> int:
        """Number of forward passes in which the steering delta was applied.

        Zero while steering is enabled and requests have been served indicates
        the hook is not firing (e.g. bypassed by a compiled graph).
        """
        return self._steering_apply_count

    @property
    def steering_delta(self) -> Optional[Tensor]:
        """Get the pre-computed steering delta (for hook access)."""
        return self._steering_delta

    def _rebuild_steering_delta(self) -> None:
        """
        Rebuild pre-computed steering delta from values.

        Computes: delta = Σ (strength_i × decoder_direction_i)
        where decoder_direction_i = W_dec[feature_idx_i, :]

        The result is in residual stream space (d_in dimensions).
        """
        if not self._steering_values:
            self._steering_delta = None
            return

        # Accumulate steering vectors from all features
        # Result shape: (d_in,) - in residual stream space
        delta = torch.zeros(self.d_in, device=self.device, dtype=self.W_dec.dtype)

        for feature_idx, strength in self._steering_values.items():
            if strength == 0:
                continue
            # Get decoder direction for this feature: W_dec[feature_idx, :] -> (d_in,)
            decoder_direction = self.W_dec[feature_idx, :]
            # Accumulate: strength × decoder_direction
            delta = delta + (strength * decoder_direction)

        self._steering_delta = delta

        logger.debug(
            f"Rebuilt steering delta: {len(self._steering_values)} features, "
            f"delta norm={delta.norm().item():.4f}"
        )


    # ==========================================================================
    # Monitoring Methods
    # ==========================================================================

    def enable_monitoring(
        self,
        enabled: bool = True,
        features: Optional[list[int]] = None,
    ) -> None:
        """
        Enable feature activation monitoring.

        Args:
            enabled: Whether to capture activations.
            features: Specific features to monitor (None = all).
                      Monitoring specific features reduces memory usage.
        """
        self._monitoring_enabled = enabled
        self._monitored_features = features

        if not enabled:
            self._last_feature_acts_per_item = []

    def get_last_feature_activations(self) -> Optional[Tensor]:
        """
        Get feature activations for batch item 0 from the last forward pass.

        For the serial inference path (batch_size=1), this is always the correct
        single-request activation tensor of shape (seq_len, d_sae).

        For CBM batches (batch_size > 1), this returns item 0's activations;
        use get_feature_activations_for_item(idx) to retrieve other items.

        Returns:
            Activations tensor (seq_len, d_sae) or None if monitoring disabled
            or no forward pass has occurred yet.
        """
        if not self._last_feature_acts_per_item:
            return None
        return self._last_feature_acts_per_item[0]

    def get_feature_activations_for_item(self, item_idx: int) -> Optional[Tensor]:
        """
        Get feature activations for a specific batch item.

        For the serial path, item_idx is always 0. For CBM batches, item_idx
        corresponds to position in the batch, not to a request ID (batch
        composition is managed internally by ContinuousBatchingManager).

        Args:
            item_idx: Batch item index (0-indexed).

        Returns:
            Activations tensor (seq_len, d_sae) or None if index out of range.
        """
        if not self._last_feature_acts_per_item or item_idx >= len(self._last_feature_acts_per_item):
            return None
        return self._last_feature_acts_per_item[item_idx]

    def get_last_batch_size(self) -> int:
        """
        Get the batch size from the last captured activation.

        Returns 0 if no activation has been captured yet (or monitoring was
        cleared by enable_monitoring(False)).
        """
        return len(self._last_feature_acts_per_item)

    @property
    def is_monitoring_enabled(self) -> bool:
        """Check if monitoring is enabled (and not currently suppressed)."""
        return self._monitoring_enabled and not self._suppressed

    # ==========================================================================
    # Co-activation sensing (Feature 11)
    # ==========================================================================

    def arm_sensing(self, config: SensingConfig) -> None:
        """
        Arm the sensing path: cache the member-only encoder slice so the
        per-pass predicate is a (seq, d_in) @ (d_in, m<=20) matmul.

        Idempotent: re-arming replaces the previous config and clears any
        buffered state. dtype/device follow the SAE weights exactly as
        encode() casts inputs.
        """
        idx = torch.tensor(config.member_indices, dtype=torch.long,
                           device=self.W_enc.device)
        self._W_enc_m = self.W_enc.index_select(1, idx).contiguous()
        self._b_enc_m = self.b_enc.index_select(0, idx).contiguous()
        config.thresholds = config.thresholds.to(
            device=self.W_enc.device, dtype=self.W_enc.dtype)
        # CPU copy for score math off the hot path (one sync at arm, none
        # per token). float32 cast first: inf survives, fp16 would overflow.
        self._sensing_thresholds_cpu = [
            float(v) for v in config.thresholds.to("cpu", torch.float32)
        ]
        self._sensing = config
        self._reset_sensing_buffer()
        self._sensing_began = False
        logger.info(
            "sensing_armed: profile=%s members=%d min_k=%d mode=%s",
            config.profile_id, len(config.member_indices),
            config.min_k, config.threshold_mode,
        )

    def disarm_sensing(self) -> None:
        """Disarm and drop all sensing state (config, caches, buffer)."""
        if self._sensing is not None:
            logger.info("sensing_disarmed: profile=%s",
                        self._sensing.profile_id)
        self._sensing = None
        self._sensing_thresholds_cpu = []
        self._W_enc_m = None
        self._b_enc_m = None
        self._reset_sensing_buffer()
        self._sensing_began = False

    @property
    def is_sensing_armed(self) -> bool:
        return self._sensing is not None

    def _reset_sensing_buffer(self) -> None:
        self._sensed_hits = []
        self._sensing_token_offset = 0
        self._sensing_phase = "prefill"
        self._sensing_done = False
        self._sensing_truncated = False
        self._sensing_overhead_ms = 0.0
        # Pre-existing (F11): never reset, so after one warning a later
        # independent batching violation went unlogged for the SAE's lifetime.
        self._sensing_batch_warned = False

    def begin_sensing_request(self, request_id: str) -> None:
        """
        Open a request boundary (called inside the serial queue semaphore).

        MUST reset every piece of per-request state — a missed begin on an
        unsensed path must yield an empty collect, never stale hits from a
        prior request (FTID pitfall 1).
        """
        self._reset_sensing_buffer()
        self._sensing_request_id = request_id
        self._sensing_began = True
        self._sensing_report_from = 0

    def set_sensing_report_from(self, boundary: int) -> None:
        """Suppress event creation for absolute positions below `boundary`
        (the longest common prefix with the previous request) — re-read chat
        history re-fires the same co-activations every turn otherwise
        (goal item 2). Offsets/phases still advance normally."""
        if self._sensing_began:
            self._sensing_report_from = max(0, int(boundary))

    def collect_sensing_hits(self) -> tuple[str, list["SensedHit"], bool]:
        """
        Close the request boundary: return (request_id, hits, truncated)
        and clear the began flag so a stray later pass cannot append to a
        flushed request. Without a begin, returns an empty result.
        """
        if not self._sensing_began:
            return ("", [], False)
        hits = self._sensed_hits
        truncated = self._sensing_truncated
        request_id = self._sensing_request_id
        self._sensed_hits = []
        self._sensing_began = False
        return (request_id, hits, truncated)

    def _sense(self, hidden_states: Tensor) -> None:
        """
        Per-forward-pass co-activation predicate over the armed members.

        Called from the hook BEFORE apply_steering so positions reflect the
        pre-steer residual read. Never raises into the forward pass.
        """
        if (self._suppressed or self._sensing is None
                or not self._sensing_began or self._W_enc_m is None):
            return
        import time as _time

        started = _time.perf_counter()
        config = self._sensing
        if hidden_states.dim() == 3 and hidden_states.shape[0] > 1:
            # Batched pass while a boundary is open: positions can't be
            # attributed to a request (011 R1). Routing should prevent this
            # (armed forces serial); make the violation observable and skip
            # rather than silently sensing row 0.
            if not self._sensing_batch_warned:
                self._sensing_batch_warned = True
                logger.warning(
                    "sensing_skipped_batched_pass: batch=%d — armed sensing "
                    "expects the serial path", hidden_states.shape[0],
                )
            return
        x = hidden_states[0] if hidden_states.dim() == 3 else hidden_states
        seq_len = x.shape[0]
        try:
            if self._sensing_done:
                return  # cap hit — still advance the offset in finally
            if x.dtype != self._W_enc_m.dtype:
                x = x.to(self._W_enc_m.dtype)
            acts = torch.relu(x @ self._W_enc_m + self._b_enc_m)  # (seq, m)
            fired = acts > config.thresholds                       # (seq, m)
            counts = fired.sum(dim=-1)                             # (seq,)
            hot = (counts >= config.min_k).nonzero(as_tuple=True)[0]
            if hot.numel():
                # ONE device->host transfer per pass: the per-element
                # float()/tolist() pattern cost a CUDA sync per fired
                # member per hot position (011 R1 — hot prefills could
                # burn tens of ms inside the forward hook).
                hot_list = hot.tolist()
                acts_hot = acts[hot].detach().to("cpu", non_blocking=False)
                fired_hot = fired[hot].detach().to("cpu", non_blocking=False)
                self._append_sensing_hits(hot_list, acts_hot, fired_hot)
        except Exception:
            # An observation path must never break generation.
            logger.exception("sensing_pass_failed")
        finally:
            self._sensing_token_offset += seq_len
            if self._sensing_phase == "prefill":
                self._sensing_phase = "decode"
            self._sensing_overhead_ms += (
                (_time.perf_counter() - started) * 1000.0)

    def _append_sensing_hits(
        self, hot_positions: list[int], acts_hot: Tensor, fired_hot: Tensor
    ) -> None:
        """Debounce hot positions into spans and merge with the buffer tail.

        acts_hot/fired_hot are CPU tensors indexed by hot-position ROW (not
        sequence position). Consecutive absolute positions extend one span —
        including across pass boundaries during decode (position p in one
        pass, p+1 in the next: FTID pitfall 3). New spans beyond the
        per-request cap set the truncated flag and stop further sensing for
        the request.
        """
        assert self._sensing is not None
        config = self._sensing
        thresholds = self._sensing_thresholds_cpu

        for row, pos in enumerate(hot_positions):
            abs_pos = self._sensing_token_offset + pos
            if abs_pos < self._sensing_report_from:
                # Re-read history (common prefix with the previous request)
                # — this moment was reported when it first occurred
                # (goal item 2).
                continue
            member_mask = fired_hot[row]
            member_acts = acts_hot[row].tolist()
            fired_pairs: dict[int, float] = {}
            score = 0.0
            for j in member_mask.nonzero(as_tuple=True)[0].tolist():
                real_idx = config.member_indices[j]
                act = float(member_acts[j])
                fired_pairs[real_idx] = act
                theta = thresholds[j]
                score = max(score, act / theta if theta > 0 else act)

            tail = self._sensed_hits[-1] if self._sensed_hits else None
            if tail is not None and abs_pos == tail.pos_end + 1:
                # Extend the span (peaks and score are running maxima)
                merged = dict(tail.fired)
                for idx, act in fired_pairs.items():
                    merged[idx] = max(merged.get(idx, 0.0), act)
                tail.pos_end = abs_pos
                tail.fired = sorted(merged.items())
                # Union count — must agree with fired_members (011 R1: the
                # peak-simultaneous count disagreed with the member list).
                tail.fired_count = len(merged)
                tail.score = max(tail.score, score)
                continue

            if len(self._sensed_hits) >= config.max_events_per_request:
                self._sensing_truncated = True
                self._sensing_done = True
                return

            self._sensed_hits.append(SensedHit(
                pos_start=abs_pos,
                pos_end=abs_pos,
                phase=self._sensing_phase,
                fired=sorted(fired_pairs.items()),
                fired_count=len(fired_pairs),
                score=score,
            ))

    # ------------------------------------------------------------------
    # Feature 15: circuit edge sensing
    # ------------------------------------------------------------------

    def arm_edge_sensing(self, config: "CircuitSensingConfig") -> None:
        """Arm this SAE for one layer of a circuit's edge sensing.

        Arming no longer takes a ring (F17 task 3.3). The ring is per
        (request, circuit) and now lives on the request context, obtained via
        ``bind_context``. Arming is a long-lived configuration step and a
        request is not; binding the two together is what let one request's
        upstream fires be visible to the next, and what made a shared ring
        cross-match two circuits that happened to use the same edge_key.
        """
        if config.member_indices:
            bad = [i for i in config.member_indices if not 0 <= i < self.d_sae]
            if bad:
                # An out-of-range index_select on CUDA is a device-side assert
                # that poisons the context for the whole process — refuse here
                # where it is a clean error. More exposed than F11: a circuit's
                # members are keyed by layer, so a mis-keyed lookup would index
                # into the WRONG SAE's feature space.
                raise ValueError(
                    f"edge sensing members out of range for layer {config.layer} "
                    f"(d_sae={self.d_sae}): {bad[:5]}"
                )
            idx = torch.tensor(
                config.member_indices, dtype=torch.long, device=self.W_enc.device
            )
            self._W_enc_e = self.W_enc.index_select(1, idx).contiguous()
            self._b_enc_e = self.b_enc.index_select(0, idx).contiguous()
            config.thresholds = config.thresholds.to(
                device=self.W_enc.device, dtype=self.W_enc.dtype
            )
            # F17 task 3.4: `_edge_thresholds_cpu` was built here and read by
            # NOTHING — dead since F15 R1-14, re-recorded in R2 and R3, and
            # kept alive only by two tests asserting its contents, which lent
            # it false legitimacy. Deleted rather than carried through the
            # extraction. (Feature 11's equivalent IS consumed, for its
            # act/theta score; F15 computes no such score.)
        # R1 CRITICAL: an out-of-range up_col/down_col raised IndexError inside
        # the matcher, which the broad `except` swallowed — abandoning the
        # ENTIRE pass (every edge, including upstream recording) rather than
        # one bad spec. Refuse at arm time where it is a clean error.
        width = len(config.member_indices)
        bad_cols = [
            spec.edge_key
            for spec in config.edges
            # -1 is the legitimate "not my half" sentinel; anything lower is
            # a bug that would silently skip the edge rather than raise.
            if not (-1 <= spec.up_col < width and -1 <= spec.down_col < width)
        ]
        if bad_cols:
            raise ValueError(
                f"edge sensing column out of range for layer {config.layer} "
                f"(slice width={width}): {bad_cols[:5]}"
            )

        self._edge_sensing = config
        self._reset_edge_buffer()
        logger.info(
            "edge sensing armed: circuit=%s layer=%d members=%d edges=%d",
            config.circuit_id, config.layer,
            len(config.member_indices), len(config.edges),
        )

    @property
    def _edge_ring(self) -> Optional["EdgeFireRing"]:
        """This circuit's ring for the CURRENT request, or None when unbound.

        Derived, never stored. A stored ring outlives the request that owns it,
        which is how one request's upstream fires stayed visible to the next.
        Returns None when unarmed or unbound, so every existing guard against a
        missing ring keeps working unchanged.
        """
        ctx = self._edge_ctx
        cfg = self._edge_sensing
        if ctx is None or cfg is None or ctx.is_closed:
            return None
        return ctx.ring(cfg.circuit_id, cfg.max_token_lag)

    def bind_context(self, ctx: Optional["EdgeSensingRequestContext"]) -> None:
        """Attach the request-scoped context this SAE senses into.

        One context is shared by every SAE of every armed circuit for the
        duration of one request; it owns the absolute position, the per-circuit
        rings and the event budget. Binding to None detaches (used on disarm
        and at request close).

        Deliberately separate from ``arm_edge_sensing``: arming is per
        deployment, binding is per request. See the arm docstring for why
        conflating them was the defect.
        """
        self._edge_ctx = ctx

    def disarm_edge_sensing(self) -> None:
        self._edge_sensing = None
        self._W_enc_e = None
        self._b_enc_e = None
        self._edge_ctx = None
        self._reset_edge_buffer()

    @property
    def is_edge_sensing_armed(self) -> bool:
        """Deliberately distinct from is_sensing_armed — a deployment may run
        cluster sensing and circuit edge sensing at the same time."""
        return self._edge_sensing is not None

    def _reset_edge_buffer(self) -> None:
        self._sensed_edges = []
        self._edge_token_offset = 0
        self._edge_phase = "prefill"
        self._edge_done = False
        self._edge_truncated = False
        self._edge_began = False
        self._edge_request_id = ""
        self._edge_overhead_ms = 0.0
        # R1: these were never reset, so after one warning a LATER independent
        # violation went unlogged for the SAE's lifetime. (Feature 11 has the
        # identical latent bug in _sensing_batch_warned — fixed there too.)
        self._edge_batch_warned = False
        self._edge_saturation_warned = False
        self._edge_member_fires = 0

    def begin_edge_sensing_request(self, request_id: str) -> None:
        """Open a request boundary.

        Nothing is cleared here. Rings belong to the request context, so the
        previous request's went out of scope with it — the old convention
        ("the CALLER clears the shared ring once for the whole circuit") only
        existed because one long-lived ring was shared by every SAE, and it
        failed the moment a participant forgot which half of the rule it was on.

        If no context is bound, a solo one is created here. Arming, beginning
        and then sensing without a context would otherwise find no ring and
        record NOTHING — sensing that reports a clean empty result while
        observing nothing at all, which is the silent-failure mode this
        feature exists to remove. A single-SAE circuit is also a legitimate
        configuration, so this is the correct behaviour rather than a
        convenience.
        """
        self._reset_edge_buffer()
        if self._edge_ctx is None or self._edge_ctx.is_closed:
            cfg = self._edge_sensing
            self._edge_ctx = EdgeSensingRequestContext(
                request_id=request_id,
                circuit_ids=frozenset({cfg.circuit_id} if cfg else set()),
                cap=cfg.max_events_per_request if cfg else 20,
            )
        self._edge_began = True
        self._edge_request_id = request_id

    def collect_sensed_edges(self) -> tuple[str, list["SensedEdge"], bool]:
        """Drain this SAE's edges and close the boundary."""
        if not self._edge_began:
            # F11 parity: draining without an open boundary would surface stale
            # edges attributed to an empty request_id.
            return "", [], False
        request_id = self._edge_request_id
        edges = self._sensed_edges
        truncated = self._edge_truncated
        self._sensed_edges = []
        self._edge_began = False
        self._edge_request_id = ""
        return request_id, edges, truncated

    def _sense_edges(self, hidden_states: Tensor) -> None:
        """Per-pass edge predicate: record upstream fires, match downstream.

        Called from the hook BEFORE apply_steering so positions reflect the
        pre-steer residual read. Never raises into the forward pass.
        """
        # F17 task 3.2: ONE advance, above every guard.
        #
        # The offset advance used to be triplicated — once in each early-return
        # branch and once in the `finally`. That shape caused F15 R1-03: a
        # suppressed or unarmed pass took a branch whose copy was missing, so
        # THIS SAE's offset fell behind its siblings', and because the ring is
        # keyed on absolute position and shared, one layer's coordinates
        # silently shifted relative to another's for the rest of the request.
        # F15 R3's own fix then inherited the same shape one level down, with
        # `note_layer_progress` sitting below the returns.
        #
        # There is now no path that reaches sensing without advancing, and no
        # second copy to forget.
        seq = (
            hidden_states.shape[1]
            if hidden_states.dim() == 3
            else hidden_states.shape[0]
        )
        base = self._advance_edge_position(seq)
        if (self._suppressed or self._edge_sensing is None
                or not self._edge_began or self._W_enc_e is None
                or self._edge_ring is None):
            # A suppressed pass still reports progress, or `_progress` stays
            # under the ring's len<2 guard and pruning never runs (EC-17.1).
            self._report_edge_progress()
            return
        import time as _time

        started = _time.perf_counter()
        config = self._edge_sensing
        if hidden_states.dim() == 3 and hidden_states.shape[0] > 1:
            # Batched pass while a boundary is open: positions cannot be
            # attributed to a request. Routing forces serial when armed; make
            # the violation observable rather than sensing row 0 silently.
            if not self._edge_batch_warned:
                self._edge_batch_warned = True
                logger.warning(
                    "edge_sensing_skipped_batched_pass: batch=%d — armed edge "
                    "sensing expects the serial path", hidden_states.shape[0],
                )
            # This return is ABOVE the try, so it skips the `finally` and would
            # report no progress at all — the same EC-17.1 stall as the
            # suppressed path, on a second path. Found by execution while
            # extracting: offset advanced to 5, `_progress` stayed {}.
            self._report_edge_progress()
            return
        x = hidden_states[0] if hidden_states.dim() == 3 else hidden_states
        seq_len = x.shape[0]
        try:
            # R3: this used to `return`, so a layer that hit its cap stopped
            # recording UPSTREAM fires for the rest of the request — silently
            # blinding every uncapped sibling. That is R2-03's starvation bug
            # reached through the cap instead of the shed. Upstream recording
            # is a dict append and siblings depend on it, so the cap must
            # suppress only the downstream append (see _match_edges).
            if x.dtype != self._W_enc_e.dtype:
                x = x.to(self._W_enc_e.dtype)
            acts = torch.relu(x @ self._W_enc_e + self._b_enc_e)   # (seq, m)
            fired = acts > config.thresholds                        # (seq, m)
            if not bool(fired.any()):
                return
            # ONE device->host transfer per pass. Reading float(acts[p, c])
            # inside the per-position x per-edge loop below would cost a CUDA
            # sync EACH TIME — the regression 011 R1 fixed for _sense.
            fired_cpu = fired.detach().to("cpu", non_blocking=False)
            acts_cpu = acts.detach().to("cpu", torch.float32, non_blocking=False)
            self._match_edges(base, seq_len, acts_cpu, fired_cpu)
        except Exception:
            # An observation path must never break generation.
            logger.exception("edge_sensing_pass_failed")
        finally:
            # Position advanced ONCE at the top, above every guard. Progress is
            # reported HERE, after the match — reporting it earlier lets this
            # layer's advance prune the ring out from under a sibling that has
            # not read it yet (caught by the characterization gate).
            self._report_edge_progress()
            self._edge_overhead_ms += ((_time.perf_counter() - started) * 1000.0)

    def _advance_edge_position(self, seq: int) -> int:
        """Advance past `seq` tokens and report progress. Returns the position
        this pass STARTS at.

        Called unconditionally at the top of `_sense_edges`, before any guard,
        so no return path can skip it. Advancing in three places (two early
        returns plus a `finally`) is what produced F15 R1-03 and then survived
        one level down into R3's own fix.
        """
        # PER-LAYER, deliberately — not ctx.advance().
        #
        # Wiring this to a single shared ctx.position looked like the obvious
        # completion of the context design, and it is wrong. Every layer's
        # hook sees the SAME tokens: with one counter the upstream layer
        # advances it to 12, then the downstream layer senses those same 12
        # tokens starting at 12, so the two layers' coordinates diverge and no
        # cross-layer edge can ever match. Verified by execution — the gate
        # caught it, and `ring._fires` showed upstream positions 2..11 against
        # a downstream layer that began at 12.
        #
        # Absolute position is shared BY CONSTRUCTION instead: every layer
        # counts the same tokens from 0, so their coordinates agree without a
        # shared counter. What genuinely is per-request-and-shared — the rings,
        # the budget, the pruning boundary — lives on the context.
        base = self._edge_token_offset
        self._edge_token_offset += seq
        if self._edge_phase == "prefill":
            self._edge_phase = "decode"
        return base

    def _report_edge_progress(self) -> None:
        """Tell the ring how far THIS layer has walked, so it prunes to the
        slowest.

        Deliberately separate from `_advance_edge_position` and called AFTER
        matching. The characterization gate caught this: reporting progress
        before the match let an upstream layer's advance prune the ring out
        from under the downstream layer that had not read it yet — resurrecting
        F15 R1-01, the exact defect that made cross-layer sensing go dark.
        Position must advance before the guards; progress must be reported
        after the work.
        """
        cfg = self._edge_sensing
        if cfg is None:
            return
        ctx = self._edge_ctx
        if ctx is not None:
            # The context reports to every ring it owns, so a circuit with
            # more than one ring cannot have one of them silently unpruned.
            ctx.report_progress(
                cfg.layer, self._edge_token_offset,
                circuit_id=cfg.circuit_id, max_lag=cfg.max_token_lag,
            )
            return
        if self._edge_ring is not None:
            try:
                self._edge_ring.note_layer_progress(
                    cfg.layer, self._edge_token_offset
                )
            except Exception:
                logger.exception("edge_ring_progress_failed")

    def _match_edges(
        self, base: int, seq_len: int, acts_cpu: Tensor, fired_cpu: Tensor
    ) -> None:
        """Record upstream fires and match downstream ones, in position order.

        Both halves run per position and IN ORDER so that within a single
        prefill pass an upstream fire at position p is visible to a downstream
        fire at p+1 — the intra-pass case that a record-all-then-match-all
        split would miss for same-layer edges.
        """
        config = self._edge_sensing
        ring = self._edge_ring
        if config is None or ring is None:
            return
        phase = self._edge_phase

        # R1 CRITICAL: the previous shape was a positions x edges Python loop
        # doing a scalar tensor read per edge per position — measured at 1430ms
        # inside the forward hook on a 4096-token prefill against a 5ms budget
        # (286x). Vectorise: find the fired positions PER COLUMN once, then
        # iterate only over actual fires. Cost now scales with the number of
        # fires, not with sequence_length x edge_count.
        n_cols = fired_cpu.shape[-1] if fired_cpu.dim() > 1 else 0

        # Shed load BEFORE building the event list. The per-request cap bounds
        # the OUTPUT, but the cost of finding fires is paid first — a
        # pathologically low threshold on a long prefill fired on nearly every
        # (position, member) pair and cost 189ms inside the forward hook even
        # though only 20 events survived. When a pass is this saturated the
        # thresholds are miscalibrated and the observations are noise, so
        # skipping is strictly better than stalling generation to collect it.
        total_fires = int(fired_cpu.sum())
        # Fires among the ARMED CIRCUIT MEMBERS in this pass. NOT the
        # contract's alone-vs-within signal: `ambient_fired_count` is defined
        # (F11, and the millm_sensing_events MCP contract) as the count across
        # the WHOLE SAE, populated only when un-compacted monitoring co-ran and
        # left NULL otherwise — "never estimated". This number answers a
        # different question and must never be written to that column.
        self._edge_member_fires += total_fires
        budget = max(
            config.max_events_per_request * 8, _EDGE_FIRE_BUDGET_MIN
        )
        shed = total_fires > budget
        if shed:
            if not self._edge_saturation_warned:
                self._edge_saturation_warned = True
                logger.warning(
                    "edge_sensing_pass_saturated: fires=%d budget=%d seq=%d — "
                    "thresholds are likely miscalibrated; matching only the "
                    "upstream half of this pass", total_fires, budget, seq_len,
                )
            self._edge_truncated = True
            # R2: R1 returned here, recording NOTHING. Shedding is decided per
            # SAE per pass, so a saturated UPSTREAM layer silently blinded a
            # quiet downstream sibling that did not shed — and the truncated
            # flag landed on the layer that shed, not the layer that lost data.
            # The operator saw a clean, empty result: exactly the silently-dark
            # mode R1-01 existed to eliminate, reintroduced by the R1-02 fix.
            # Upstream recording is cheap (a dict append) and is what siblings
            # depend on, so keep it and skip only the expensive matching.

        # The MATCHING half is delegated. This method keeps only the
        # load-shedding accounting above, because the counters it feeds
        # (_edge_member_fires, the saturation latch, the shed truncation flag)
        # are per-SAE and have no counterpart on the context. Moving the whole
        # body would have silently zeroed _edge_member_fires, which
        # circuit_sensing_service.py:503 reads for an operator-facing counter —
        # a metric going quietly to zero, not a crash.
        _match_edges_impl(
            ring=ring,
            config=config,
            phase=phase,
            base=base,
            acts_cpu=acts_cpu,
            fired_cpu=fired_cpu,
            out=self._sensed_edges,
            n_cols=n_cols,
            shed=shed,
            capped=self._edge_done,
            on_cap=self._note_edge_cap,
            # R1-01: the per-CIRCUIT budget, which was built, unit-tested and
            # never wired — so an N-layer circuit still emitted N x its cap
            # (measured: cap 3, three layers, nine events, `spent` 0). This is
            # the guarantee FPRD §9 criterion 3 requires and F19 depends on.
            try_spend=self._try_spend_circuit_budget,
        )

    def _try_spend_circuit_budget(self, spec) -> bool:
        """Claim one slot from the CIRCUIT's shared budget.

        False means this event is dropped and the layer is recorded as
        truncated — the caller CONTINUES, so upstream recording keeps feeding
        sibling layers (R2-03/R3-02). Returns True when unbound, so an
        unbound SAE degrades to the per-SAE cap rather than to silence.
        """
        ctx = self._edge_ctx
        cfg = self._edge_sensing
        if ctx is None or cfg is None or ctx.is_closed:
            return True
        return ctx.budget.try_spend(cfg.circuit_id, spec.down_layer)

    def _note_edge_cap(self) -> None:
        """The per-request cap was reached. Latches so later passes skip the
        downstream half entirely rather than re-walking it to reject every
        event (the latch is a performance property, not a correctness one)."""
        self._edge_truncated = True
        self._edge_done = True

    @contextmanager
    def suppressed(self) -> Iterator[None]:
        """Temporarily disable steering application and monitoring capture.

        For the duration of the context, the forward hook is effectively inert:
        apply_steering() returns the hidden states unchanged and monitoring does
        not capture.  Used to run forward passes that must see the *unsteered*
        model (e.g. embeddings) through the same hooked model without perturbing
        their output or overwriting the last-captured activations.
        """
        previous = self._suppressed
        self._suppressed = True
        try:
            yield
        finally:
            self._suppressed = previous

    def _capture_activations(self, feature_acts: Tensor) -> None:
        """
        Capture activations per batch item for monitoring.

        Splits the batch dimension so each captured item corresponds to a single
        request. The serial path always has batch_size=1 (one item). CBM batches
        have batch_size > 1 but batch composition is opaque at this level.

        Args:
            feature_acts: Feature activations (batch_size, seq_len, d_sae) or
                          (seq_len, d_sae) for unbatched inputs.
        """
        if self._monitored_features is not None:
            selected = feature_acts[..., self._monitored_features].detach()
        else:
            selected = feature_acts.detach()

        if selected.dim() >= 2:
            batch_size = selected.shape[0]
            self._last_feature_acts_per_item = [
                selected[i].clone() for i in range(batch_size)
            ]
        else:
            # Unbatched (1D) — treat as single item
            self._last_feature_acts_per_item = [selected.clone()]

    # ==========================================================================
    # Memory Management
    # ==========================================================================

    def estimate_memory_mb(self) -> float:
        """
        Estimate current GPU memory usage.

        Returns:
            Estimated memory in MB.
        """
        return self.config.estimate_memory_mb()

    def to_device(self, device: str) -> None:
        """
        Move all tensors to device.

        Rebuilds the steering delta on the new device if steering is active.
        This is important when to_device() is called after steering values have
        already been set: the delta must live on the same device as the weights
        so that apply_steering() does not incur a cross-device transfer.

        Args:
            device: Target device (e.g., "cuda", "cpu", "cuda:0").
        """
        self.W_enc = self.W_enc.to(device)
        self.b_enc = self.b_enc.to(device)
        self.W_dec = self.W_dec.to(device)
        self.b_dec = self.b_dec.to(device)

        self.device = device

        # Rebuild delta on the new device so apply_steering() is zero-copy.
        # This handles both the common case (delta is None, nothing to do) and
        # the edge case where to_device() is called after set_steering().
        if self._steering_values:
            self._rebuild_steering_delta()
        else:
            self._steering_delta = None

        # Sensing tensors move with the weights (011 R3: a device move on an
        # armed SAE left the member slices behind — every _sense pass then
        # threw and was swallowed, sensing silently dark while status said
        # armed).
        if self._W_enc_m is not None:
            self._W_enc_m = self._W_enc_m.to(device)
        if self._b_enc_m is not None:
            self._b_enc_m = self._b_enc_m.to(device)
        if self._sensing is not None:
            self._sensing.thresholds = self._sensing.thresholds.to(device)

        # Same for the edge-sensing slices (Feature 15) — an armed SAE that
        # left these behind would throw on every pass and be swallowed.
        if self._W_enc_e is not None:
            self._W_enc_e = self._W_enc_e.to(device)
        if self._b_enc_e is not None:
            self._b_enc_e = self._b_enc_e.to(device)
        if self._edge_sensing is not None:
            self._edge_sensing.thresholds = self._edge_sensing.thresholds.to(device)

        logger.debug(f"LoadedSAE moved to {device}")

    def to_cpu(self) -> None:
        """Move all tensors to CPU (for cleanup)."""
        self.to_device("cpu")

    def __repr__(self) -> str:
        return (
            f"LoadedSAE(d_in={self.d_in}, d_sae={self.d_sae}, "
            f"device={self.device}, steering={self._steering_enabled})"
        )
