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
        self._sensing_batch_warned: bool = False
        # Cumulative per-request overhead accumulator (ms) — read by the
        # sensing status endpoint (SEN-S2); reset at begin.
        self._sensing_overhead_ms: float = 0.0

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

        logger.debug(f"LoadedSAE moved to {device}")

    def to_cpu(self) -> None:
        """Move all tensors to CPU (for cleanup)."""
        self.to_device("cpu")

    def __repr__(self) -> str:
        return (
            f"LoadedSAE(d_in={self.d_in}, d_sae={self.d_sae}, "
            f"device={self.device}, steering={self._steering_enabled})"
        )
