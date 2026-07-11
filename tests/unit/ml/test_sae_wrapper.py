"""
Unit tests for LoadedSAE wrapper.
"""

import pytest
import torch

from millm.ml.sae_config import SAEConfig
from millm.ml.sae_wrapper import LoadedSAE


@pytest.fixture
def small_sae():
    """Create a small SAE for testing."""
    d_in, d_sae = 64, 128
    config = SAEConfig(
        d_in=d_in,
        d_sae=d_sae,
        model_name="test",
        hook_name="test",
        hook_layer=0,
    )
    return LoadedSAE(
        W_enc=torch.randn(d_in, d_sae),
        b_enc=torch.zeros(d_sae),
        W_dec=torch.randn(d_sae, d_in),
        b_dec=torch.zeros(d_in),
        config=config,
        device="cpu",
    )


class TestLoadedSAEForward:
    """Tests for SAE forward pass."""

    def test_forward_preserves_shape(self, small_sae):
        """Forward pass preserves input shape."""
        x = torch.randn(2, 10, 64)  # batch=2, seq=10, d_in=64
        out = small_sae.forward(x)
        assert out.shape == x.shape

    def test_encode_produces_features(self, small_sae):
        """Encode produces d_sae features."""
        x = torch.randn(1, 5, 64)
        features = small_sae.encode(x)
        assert features.shape == (1, 5, 128)  # d_sae=128

    def test_decode_restores_dimension(self, small_sae):
        """Decode restores d_in dimension."""
        features = torch.randn(1, 5, 128)
        output = small_sae.decode(features)
        assert output.shape == (1, 5, 64)  # d_in=64

    def test_features_are_non_negative(self, small_sae):
        """Feature activations are non-negative (ReLU)."""
        x = torch.randn(1, 5, 64)
        features = small_sae.encode(x)
        assert (features >= 0).all()


class TestLoadedSAESteering:
    """Tests for steering functionality."""

    def test_steering_modifies_output(self, small_sae):
        """apply_steering changes hidden states when steering is enabled."""
        x = torch.randn(1, 5, 64)

        # Baseline: steering disabled — apply_steering returns input unchanged
        small_sae.enable_steering(False)
        out_baseline = small_sae.apply_steering(x.clone())

        # Steered: set a non-zero steering vector and enable
        small_sae.set_steering(0, 10.0)
        small_sae.enable_steering(True)
        out_steered = small_sae.apply_steering(x.clone())

        assert not torch.allclose(out_baseline, out_steered)

    def test_steering_apply_count_tracks_applications(self, small_sae):
        """steering_apply_count increments only when the delta is actually applied."""
        x = torch.randn(1, 5, 64)

        # Disabled: no application, counter stays at 0.
        small_sae.enable_steering(False)
        small_sae.apply_steering(x.clone())
        assert small_sae.steering_apply_count == 0

        # Enabled with a non-zero delta: each call increments the counter.
        small_sae.set_steering(0, 10.0)
        small_sae.enable_steering(True)
        small_sae.apply_steering(x.clone())
        small_sae.apply_steering(x.clone())
        assert small_sae.steering_apply_count == 2

    def test_set_steering_single(self, small_sae):
        """Can set steering for a single feature."""
        small_sae.set_steering(42, 5.0)
        values = small_sae.get_steering_values()
        assert values == {42: 5.0}

    def test_set_steering_batch(self, small_sae):
        """Can set steering for multiple features at once."""
        steering = {0: 1.0, 10: 2.0, 50: -1.5}
        small_sae.set_steering_batch(steering)
        values = small_sae.get_steering_values()
        assert values == steering

    def test_clear_steering_single(self, small_sae):
        """Can clear steering for a single feature."""
        small_sae.set_steering(0, 1.0)
        small_sae.set_steering(1, 2.0)
        small_sae.clear_steering(0)

        values = small_sae.get_steering_values()
        assert values == {1: 2.0}

    def test_clear_steering_all(self, small_sae):
        """Can clear all steering."""
        small_sae.set_steering(0, 1.0)
        small_sae.set_steering(1, 2.0)
        small_sae.clear_steering()

        values = small_sae.get_steering_values()
        assert values == {}

    def test_rejects_invalid_feature_index(self, small_sae):
        """Rejects feature index out of range."""
        with pytest.raises(ValueError):
            small_sae.set_steering(-1, 1.0)

        with pytest.raises(ValueError):
            small_sae.set_steering(128, 1.0)  # d_sae=128, max index is 127

    def test_steering_disabled_by_default(self, small_sae):
        """Steering is disabled by default."""
        assert not small_sae.is_steering_enabled


class TestLoadedSAEMonitoring:
    """Tests for monitoring functionality."""

    def test_monitoring_captures_activations(self, small_sae):
        """Monitoring captures feature activations as per-item tensors."""
        x = torch.randn(1, 5, 64)

        small_sae.enable_monitoring(True)
        small_sae.forward(x)

        acts = small_sae.get_last_feature_activations()
        assert acts is not None
        # Item 0: shape (seq_len, d_sae) — batch dim is stripped per-item
        assert acts.shape == (5, 128)

    def test_monitoring_specific_features(self, small_sae):
        """Can monitor only specific features."""
        x = torch.randn(1, 5, 64)

        small_sae.enable_monitoring(True, features=[0, 1, 2])
        small_sae.forward(x)

        acts = small_sae.get_last_feature_activations()
        assert acts.shape == (5, 3)  # Only 3 features, batch dim stripped

    def test_monitoring_disabled_returns_none(self, small_sae):
        """When disabled, get_last_feature_activations returns None."""
        x = torch.randn(1, 5, 64)

        small_sae.enable_monitoring(False)
        small_sae.forward(x)

        acts = small_sae.get_last_feature_activations()
        assert acts is None

    def test_monitoring_disabled_by_default(self, small_sae):
        """Monitoring is disabled by default."""
        assert not small_sae.is_monitoring_enabled

    def test_get_last_batch_size_no_capture(self, small_sae):
        """Batch size is 0 before any forward pass with monitoring."""
        assert small_sae.get_last_batch_size() == 0

    def test_get_last_batch_size_serial(self, small_sae):
        """Serial path (batch=1) produces a batch size of 1."""
        x = torch.randn(1, 5, 64)
        small_sae.enable_monitoring(True)
        small_sae.forward(x)
        assert small_sae.get_last_batch_size() == 1

    def test_get_last_batch_size_batched(self, small_sae):
        """Batched input (batch=3) produces a batch size of 3."""
        x = torch.randn(3, 5, 64)
        small_sae.enable_monitoring(True)
        small_sae.forward(x)
        assert small_sae.get_last_batch_size() == 3

    def test_get_feature_activations_for_item_serial(self, small_sae):
        """Serial path: item 0 matches get_last_feature_activations()."""
        x = torch.randn(1, 5, 64)
        small_sae.enable_monitoring(True)
        small_sae.forward(x)
        assert torch.equal(
            small_sae.get_last_feature_activations(),
            small_sae.get_feature_activations_for_item(0),
        )

    def test_get_feature_activations_for_item_batched(self, small_sae):
        """Batched input: each item is stored independently."""
        x = torch.randn(3, 5, 64)
        small_sae.enable_monitoring(True)
        small_sae.forward(x)
        for idx in range(3):
            item_acts = small_sae.get_feature_activations_for_item(idx)
            assert item_acts is not None
            assert item_acts.shape == (5, 128)

    def test_get_feature_activations_for_item_out_of_range(self, small_sae):
        """Out-of-range item index returns None."""
        x = torch.randn(1, 5, 64)
        small_sae.enable_monitoring(True)
        small_sae.forward(x)
        assert small_sae.get_feature_activations_for_item(5) is None

    def test_enable_monitoring_false_clears_capture(self, small_sae):
        """Disabling monitoring clears the captured per-item buffer."""
        x = torch.randn(1, 5, 64)
        small_sae.enable_monitoring(True)
        small_sae.forward(x)
        assert small_sae.get_last_batch_size() == 1

        small_sae.enable_monitoring(False)
        assert small_sae.get_last_batch_size() == 0
        assert small_sae.get_last_feature_activations() is None


class TestLoadedSAEMemory:
    """Tests for memory management."""

    def test_estimate_memory(self, small_sae):
        """Can estimate memory usage."""
        memory_mb = small_sae.estimate_memory_mb()
        assert memory_mb > 0

    def test_to_device(self, small_sae):
        """Can move tensors to device."""
        small_sae.to_device("cpu")
        assert small_sae.device == "cpu"
        assert small_sae.W_enc.device.type == "cpu"

    def test_dimensions_match_config(self, small_sae):
        """Dimensions match the config."""
        assert small_sae.d_in == 64
        assert small_sae.d_sae == 128
        assert small_sae.d_in == small_sae.config.d_in
        assert small_sae.d_sae == small_sae.config.d_sae


# =============================================================================
# Tests: delta dtype caching and to_device (Fix 6+7)
# =============================================================================


class TestDeltaDtypeAndDevice:
    """Verify delta dtype is cached after first cast and to_device keeps it in sync."""

    def test_apply_steering_caches_cast_delta(self, small_sae):
        """After first apply_steering with a different dtype, delta is cached."""
        small_sae.set_steering(0, 5.0)
        small_sae.enable_steering(True)

        # Simulate bfloat16 hidden states (common with modern models)
        hidden = torch.randn(1, 4, 64, dtype=torch.bfloat16)
        small_sae.apply_steering(hidden)

        # The cached delta should now be bfloat16
        assert small_sae._steering_delta is not None
        assert small_sae._steering_delta.dtype == torch.bfloat16

    def test_apply_steering_subsequent_calls_use_cached_delta(self, small_sae):
        """After the first call, the delta is not re-cast on subsequent calls."""
        small_sae.set_steering(0, 5.0)
        small_sae.enable_steering(True)

        hidden = torch.randn(1, 4, 64, dtype=torch.bfloat16)
        small_sae.apply_steering(hidden)
        delta_after_first = small_sae._steering_delta

        # Second call — delta object should be the same (no new allocation)
        small_sae.apply_steering(hidden)
        delta_after_second = small_sae._steering_delta
        assert delta_after_first is delta_after_second

    def test_to_device_rebuilds_delta_when_steering_active(self, small_sae):
        """to_device() rebuilds the delta on the new device when steering is set."""
        small_sae.set_steering(0, 3.0)
        small_sae.enable_steering(True)
        assert small_sae._steering_delta is not None

        # Move to CPU (already there, but exercises the rebuild path)
        small_sae.to_device("cpu")
        assert small_sae._steering_delta is not None
        assert str(small_sae._steering_delta.device) == "cpu"

    def test_to_device_clears_delta_when_no_steering(self, small_sae):
        """to_device() leaves delta as None when no steering values are set."""
        assert not small_sae._steering_values
        small_sae.to_device("cpu")
        assert small_sae._steering_delta is None

    def test_steering_output_matches_expected_dtype(self, small_sae):
        """apply_steering output preserves the hidden_states dtype."""
        small_sae.set_steering(0, 5.0)
        small_sae.enable_steering(True)

        for dtype in [torch.float32, torch.bfloat16, torch.float16]:
            hidden = torch.randn(1, 4, 64, dtype=dtype)
            result = small_sae.apply_steering(hidden)
            assert result.dtype == dtype, f"Expected {dtype}, got {result.dtype}"


# =============================================================================
# Tests: stale steering cleared on detach path (Fix 8)
# =============================================================================


class TestSteeringClearedOnDetach:
    """Verify clear_steering() removes state so a re-attached SAE starts clean."""

    def test_clear_steering_resets_all_values(self, small_sae):
        """After clear_steering, no values and delta is None."""
        small_sae.set_steering(0, 10.0)
        small_sae.set_steering(1, -5.0)
        small_sae.enable_steering(True)

        small_sae.clear_steering()

        assert small_sae.get_steering_values() == {}
        assert small_sae._steering_delta is None
        # Enabled flag is NOT cleared by clear_steering — that's enable_steering's job
        # The detach path calls both clear_steering() and enable_monitoring(False)

    def test_enable_monitoring_false_clears_activations(self, small_sae):
        """enable_monitoring(False) clears any captured activations."""
        x = torch.randn(1, 5, 64)
        small_sae.enable_monitoring(True)
        small_sae.forward(x)
        assert small_sae.get_last_batch_size() > 0

        small_sae.enable_monitoring(False)
        assert small_sae.get_last_batch_size() == 0
        assert small_sae.get_last_feature_activations() is None

    def test_reattach_starts_with_clean_state(self, small_sae):
        """After clear+disable sequence (what detach does), SAE is pristine."""
        # Simulate a session: set steering, capture activations
        small_sae.set_steering(0, 10.0)
        small_sae.enable_steering(True)
        small_sae.enable_monitoring(True)
        small_sae.forward(torch.randn(1, 4, 64))

        # Simulate detach sequence
        small_sae.clear_steering()
        small_sae.enable_steering(False)
        small_sae.enable_monitoring(False)

        # State should be clean
        assert small_sae.get_steering_values() == {}
        assert small_sae._steering_delta is None
        assert not small_sae.is_steering_enabled
        assert not small_sae.is_monitoring_enabled
        assert small_sae.get_last_batch_size() == 0


@pytest.mark.gpu
class TestSteeringUnderTorchCompile:
    """Regression guard for C1: a torch.compile'd model must still run the SAE
    steering hook that is registered *after* compilation.

    With TorchDynamo's default skip_nnmodule_hook_guards=True the compiled graph
    would ignore the later-registered hook and steering would silently no-op.
    model_loader flips that flag to False before compiling; this test proves the
    hook fires end-to-end on a real (tiny) compiled model.
    """

    def test_hook_fires_after_compile(self):
        pytest.importorskip("transformers")
        if not torch.cuda.is_available():
            pytest.skip("CUDA required")

        from transformers import AutoModelForCausalLM, AutoConfig
        from millm.ml.sae_config import SAEConfig
        from millm.ml.sae_hooker import SAEHooker

        # Tiny GPT-2-like model so the test is fast.
        config = AutoConfig.for_model(
            "gpt2", n_layer=2, n_embd=64, n_head=2, vocab_size=128, n_positions=64
        )
        model = AutoModelForCausalLM.from_config(config).to("cuda").eval()
        d_in = config.n_embd

        # Match model_loader: enable hook guards, then compile forward.
        import torch._dynamo as _dynamo

        _dynamo.config.skip_nnmodule_hook_guards = False
        _dynamo.reset()
        model.forward = torch.compile(model.forward, fullgraph=False)

        # Warm the compiled graph *without* a hook (mirrors model_loader warmup).
        # If the Inductor backend cannot build kernels in this environment
        # (e.g. missing Python dev headers), skip rather than fail — the guard
        # under test is the hook-guard config, not the codegen toolchain.
        ids = torch.zeros(1, 4, dtype=torch.long, device="cuda")
        try:
            with torch.no_grad():
                model(input_ids=ids, use_cache=False)
        except Exception as e:
            pytest.skip(f"torch.compile backend unavailable in this env: {e}")

        # Build an SAE with a large, obvious steering delta and attach it.
        sae_config = SAEConfig(
            d_in=d_in, d_sae=32, model_name="gpt2", hook_name="h.1", hook_layer=1
        )
        sae = LoadedSAE(
            W_enc=torch.randn(d_in, 32),
            b_enc=torch.zeros(32),
            W_dec=torch.randn(32, d_in) * 100.0,
            b_dec=torch.zeros(d_in),
            config=sae_config,
            device="cuda",
        )
        sae.set_steering(0, 100.0)
        sae.enable_steering(True)

        hooker = SAEHooker()
        _dynamo.reset()  # mirror sae_service attach behavior
        handle = hooker.install(model, layer=1, sae=sae)
        try:
            with torch.no_grad():
                model(input_ids=ids, use_cache=False)
        finally:
            handle.remove()

        # The hook must have fired at least once — this is the C1 guarantee.
        assert sae.steering_apply_count > 0, (
            "SAE steering hook did not fire under torch.compile — "
            "skip_nnmodule_hook_guards guard regressed"
        )
