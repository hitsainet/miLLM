"""The private encoder slice a k-sparse probe reads.

The two tests that matter most:

* **the slice equals a full encode restricted to the chosen columns** — and the reference for
  "full encode" is miStudio's (normalization + the architecture's activation), NOT
  `LoadedSAE.encode`, which is a bare relu with no normalization and no JumpReLU threshold;
* **`AttachedSAEState` is never touched** — the slice is private, so attaching or steering with
  any SAE, including this one, cannot move what the probe reads.
"""

from __future__ import annotations

import hashlib

import pytest
import torch

from millm.services.probe_sae_slice import (
    NO_NORMALIZATION,
    SaeFeatureSlice,
    normalize,
    sha256_of,
)

D_MODEL, N_FEATURES, K = 6, 12, 4
INDICES = [1, 4, 7, 9]


@pytest.fixture
def state():
    torch.manual_seed(5)
    return {
        "W_enc": torch.randn(D_MODEL, N_FEATURES),
        "b_enc": torch.randn(N_FEATURES),
        "threshold": torch.rand(N_FEATURES) * 0.5,
    }


def full_encode(state, x, *, architecture, mode):
    """miStudio's encode, unsliced — the reference this slice must reproduce."""
    hidden = normalize(x, mode)
    pre = hidden @ state["W_enc"] + state["b_enc"]
    if architecture == "jumprelu":
        return torch.where(pre > state["threshold"], pre, torch.zeros_like(pre))
    return torch.relu(pre)


class TestTheSliceEqualsTheFullEncode:
    @pytest.mark.parametrize("architecture", ["jumprelu", "relu"])
    @pytest.mark.parametrize("mode", [NO_NORMALIZATION, "constant_norm_rescale"])
    def test_it_matches_the_reference_restricted_to_idx(self, state, architecture, mode):
        torch.manual_seed(11)
        x = torch.randn(3, D_MODEL) * 2.0
        slice_ = SaeFeatureSlice.from_state_dict(
            state, INDICES, architecture=architecture, normalization_mode=mode
        )
        expected = full_encode(state, x, architecture=architecture, mode=mode)[:, INDICES]
        assert torch.allclose(slice_.encode(x), expected, atol=1e-6)

    def test_it_keeps_only_k_columns(self, state):
        slice_ = SaeFeatureSlice.from_state_dict(state, INDICES, architecture="jumprelu")
        assert slice_.k == K
        assert slice_.d_model == D_MODEL
        assert slice_.weight.shape == (D_MODEL, K)
        assert slice_.bias.shape == (K,)
        assert slice_.thresholds.shape == (K,)

    def test_the_columns_are_the_ones_asked_for(self, state):
        slice_ = SaeFeatureSlice.from_state_dict(state, INDICES, architecture="relu")
        assert torch.equal(slice_.weight, state["W_enc"][:, INDICES])
        assert torch.equal(slice_.bias, state["b_enc"][INDICES])


class TestJumpReLU:
    def test_a_relu_is_NOT_a_jumprelu(self):
        """⚠ miLLM's SAE loader has never read a JumpReLU threshold, so this is the difference
        that would otherwise be silent: relu keeps everything above 0, JumpReLU above θ.

        ⚠ THE FIRST VERSION OF THIS TEST USED THE RANDOM FIXTURE AND FAILED — not because the code
        was wrong, but because the fixture agreed by construction: every pre-activation above 0
        happened also to be above its θ, so relu and JumpReLU were identical on that data. The two
        differ ONLY in the band 0 < pre <= θ, so the fixture has to put a value there on purpose.
        Here pre = 0.25 against θ = 0.5: relu keeps it, JumpReLU zeroes it.
        """
        state = {
            "W_enc": torch.tensor([[0.25]]),      # d_model=1, n_features=1
            "b_enc": torch.tensor([0.0]),
            "threshold": torch.tensor([0.5]),
        }
        x = torch.tensor([[1.0]])                  # pre = 0.25, strictly between 0 and θ
        jump = SaeFeatureSlice.from_state_dict(state, [0], architecture="jumprelu")
        relu = SaeFeatureSlice.from_state_dict(state, [0], architecture="relu")
        assert relu.encode(x).item() == pytest.approx(0.25)
        assert jump.encode(x).item() == 0.0

    def test_jumprelu_without_thresholds_REFUSES_rather_than_falling_back(self):
        """Falling back to relu would encode in a different basis than the probe was fitted in —
        plausible features with different meanings, invisible in every metric."""
        slice_ = SaeFeatureSlice(
            weight=torch.randn(D_MODEL, K),
            bias=torch.zeros(K),
            architecture="jumprelu",
            thresholds=None,
        )
        with pytest.raises(ValueError, match="needs its learned thresholds"):
            slice_.encode(torch.randn(2, D_MODEL))

    def test_below_threshold_is_exactly_zero_not_merely_small(self, state):
        state = dict(state)
        state["threshold"] = torch.full((N_FEATURES,), 1e9)
        slice_ = SaeFeatureSlice.from_state_dict(state, INDICES, architecture="jumprelu")
        assert torch.equal(slice_.encode(torch.randn(3, D_MODEL)), torch.zeros(3, K))


class TestNormalization:
    def test_the_two_rescale_modes_are_the_same_operation(self):
        """⚠ MIS-E2E-085: `anthropic_rescale` is an ALIAS, not a second method.

        Treating them as different is what the contract's `mode` field would otherwise invite.
        """
        torch.manual_seed(1)
        x = torch.randn(5, 8) * 3.0
        a = normalize(x, "constant_norm_rescale")
        b = normalize(x, "anthropic_rescale")
        assert torch.allclose(a, b, atol=1e-6)

    def test_it_rescales_each_row_to_sqrt_d(self):
        x = torch.randn(4, 9) * 7.0
        out = normalize(x, "constant_norm_rescale")
        assert torch.allclose(out.norm(dim=-1), torch.full((4,), 3.0), atol=1e-5)

    def test_none_is_a_no_op(self):
        x = torch.randn(3, 5)
        assert torch.equal(normalize(x, NO_NORMALIZATION), x)

    def test_an_all_zero_row_is_left_alone_not_divided_by_zero(self):
        x = torch.zeros(2, 4)
        out = normalize(x, "constant_norm_rescale")
        assert torch.equal(out, x)
        assert torch.isfinite(out).all()

    def test_an_unknown_mode_raises(self):
        with pytest.raises(ValueError, match="unknown normalization mode"):
            normalize(torch.randn(2, 3), "whatever")


class TestLoading:
    def test_the_sha_is_verified_before_the_basis_is_built(self, state, tmp_path):
        """⚠ A dictionary that is not the one the probe was fitted against produces features that
        are plausible and mean something else. The failure has no symptom."""
        from safetensors.torch import save_file

        path = tmp_path / "sae.safetensors"
        save_file({k: v.contiguous() for k, v in state.items()}, str(path))
        with pytest.raises(ValueError, match="sha256"):
            SaeFeatureSlice.load(
                path, INDICES, architecture="jumprelu", expected_sha256="0" * 64
            )

    def test_the_matching_sha_loads(self, state, tmp_path):
        from safetensors.torch import save_file

        path = tmp_path / "sae.safetensors"
        save_file({k: v.contiguous() for k, v in state.items()}, str(path))
        slice_ = SaeFeatureSlice.load(
            path, INDICES, architecture="jumprelu", expected_sha256=sha256_of(path)
        )
        assert slice_.k == K

    def test_a_missing_file_names_the_path(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="SAE weights not found"):
            SaeFeatureSlice.load(tmp_path / "nope.safetensors", INDICES, architecture="relu")

    def test_a_transposed_export_is_detected(self):
        """Getting the orientation backwards silently encodes noise."""
        state = {
            "W_enc": torch.randn(N_FEATURES, D_MODEL),  # transposed
            "b_enc": torch.randn(N_FEATURES),
        }
        slice_ = SaeFeatureSlice.from_state_dict(state, INDICES, architecture="relu")
        assert slice_.d_model == D_MODEL and slice_.k == K

    def test_missing_encoder_weights_are_refused(self):
        with pytest.raises(ValueError, match="no encoder"):
            SaeFeatureSlice.from_state_dict({"W_dec": torch.randn(4, 4)}, [0], architecture="relu")


class TestItIsPrivate:
    def test_encoding_never_touches_AttachedSAEState(self, state, monkeypatch):
        """⚠ The slice is private to the probe: attaching, detaching or steering with any SAE —
        including this one — must not move what the probe reads."""
        import millm.services.sae_service as sae_service

        touched = []
        original = sae_service.AttachedSAEState.__new__

        def spy(cls, *a, **k):
            touched.append(1)
            return original(cls)

        monkeypatch.setattr(sae_service.AttachedSAEState, "__new__", spy)
        slice_ = SaeFeatureSlice.from_state_dict(state, INDICES, architecture="jumprelu")
        slice_.encode(torch.randn(3, D_MODEL))
        assert touched == [], "the slice reached into the attached-SAE registry"

    def test_the_wrong_width_is_refused_rather_than_broadcast(self, state):
        slice_ = SaeFeatureSlice.from_state_dict(state, INDICES, architecture="relu")
        with pytest.raises(ValueError, match="d_model=3 but this slice is d_model=6"):
            slice_.encode(torch.randn(2, 3))
