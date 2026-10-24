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
from dataclasses import replace

import pytest
import torch

from millm.services.probe_sae_slice import (
    FLIP_RISK_SIGMAS,
    NO_NORMALIZATION,
    PRODUCER_PRECISION_GAP,
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


class TestTheEncodeMatchesMiStudiosFormulation:
    """⚠ THE SLICE ENCODED IN AN UNCENTERED BASIS, AND NOTHING ABOUT THE OUTPUT SAID SO.

    miStudio's SAE is `z = ReLU(W_enc @ (x - b_dec) + b_enc)` — the standard Bricken et al. 2023
    formulation, centering by the DECODER bias so the encoder sees zero-mean residuals. The
    slice omitted `- b_dec`.

    The result was sparse, plausible, well-behaved features that meant something else. The file
    loaded, its sha256 matched the probe's pin, and roughly the right number of features fired.
    Only the parity gate caught it, on hardware: the recorded scores diverged with a per-token
    median of 58.3, where the dense probe's was 0.96.

    ⚠ **And the precision hypothesis was WRONG, which is why this is a test and not a comment.**
    I assumed bf16-vs-fp16 noise crossing JumpReLU thresholds explained the divergence. Measured:
    **1 flip in 23,040 features (0.004%)**, and where both precisions were active the max
    difference was 0.0711. Precision was never the story; the basis was.
    """

    @staticmethod
    def _state(d_model: int = 8, d_sae: int = 6, seed: int = 0):
        torch.manual_seed(seed)
        return {
            "W_enc": torch.randn(d_model, d_sae),
            "b_enc": torch.randn(d_sae) * 0.1,
            # ⚠ NON-ZERO, deliberately. A zero b_dec makes centered and uncentered identical,
            # so a fixture with one would agree by construction with the defect — the single
            # commonest reason a suite in this estate stays green over a real bug.
            "b_dec": torch.randn(d_model),
        }

    def _mistudio_encode(self, state, x, indices):
        """miStudio's formulation, written out, over the FULL dictionary then selected."""
        z = torch.relu((x - state["b_dec"]) @ state["W_enc"] + state["b_enc"])
        return z[..., indices]

    def test_the_slice_agrees_with_the_full_centered_encode(self, tmp_path):
        from safetensors.torch import save_file

        state = self._state()
        path = tmp_path / "sae_weights.safetensors"
        save_file(state, str(path))
        indices = [0, 2, 4]

        slice_ = SaeFeatureSlice.load(path, indices, architecture="standard")
        x = torch.randn(5, 8)
        assert torch.allclose(
            slice_.encode(x), self._mistudio_encode(state, x, indices), atol=1e-5
        ), "the slice does not reproduce miStudio's encode restricted to its features"

    def test_a_fixture_with_a_ZERO_decoder_bias_cannot_tell_the_difference(self, tmp_path):
        """Specificity, stated as a test so the trap stays visible.

        With `b_dec = 0` the centered and uncentered forms are identical, so this fixture
        passes against the DEFECT. It is here to prove the test above is doing work that this
        one cannot.
        """
        from safetensors.torch import save_file

        state = self._state()
        state["b_dec"] = torch.zeros(8)
        path = tmp_path / "zero.safetensors"
        save_file(state, str(path))

        slice_ = SaeFeatureSlice.load(path, [0, 1], architecture="standard")
        x = torch.randn(3, 8)
        uncentered = torch.relu(x @ state["W_enc"] + state["b_enc"])[..., [0, 1]]
        assert torch.allclose(slice_.encode(x), uncentered, atol=1e-5)

    def test_the_decoder_bias_is_loaded_and_kept_whole(self, tmp_path):
        """It is subtracted in d_model space, so slicing it to k would be wrong."""
        from safetensors.torch import save_file

        state = self._state()
        path = tmp_path / "w.safetensors"
        save_file(state, str(path))
        slice_ = SaeFeatureSlice.load(path, [0, 3], architecture="standard")
        assert slice_.decoder_bias is not None
        assert slice_.decoder_bias.shape == (8,), "b_dec must stay d_model-wide, not sliced to k"

    def test_it_survives_a_device_move(self, tmp_path):
        from safetensors.torch import save_file

        state = self._state()
        path = tmp_path / "w.safetensors"
        save_file(state, str(path))
        slice_ = SaeFeatureSlice.load(path, [0, 1], architecture="standard")
        moved = slice_.to_device(torch.device("cpu"))
        assert moved.decoder_bias is not None

    def test_a_dictionary_with_no_b_dec_still_loads(self, tmp_path):
        """Not every published SAE carries one; the absence must not crash the load."""
        from safetensors.torch import save_file

        state = self._state()
        del state["b_dec"]
        path = tmp_path / "nob.safetensors"
        save_file(state, str(path))
        slice_ = SaeFeatureSlice.load(path, [0, 1], architecture="standard")
        assert slice_.decoder_bias is None
        slice_.encode(torch.randn(2, 8))  # no centering, and no crash

class TestFlipRisk:
    """Which positions a JumpReLU gate cannot be reproduced at, computed from OUR side alone.

    ⚠ THE FIXTURE MUST NOT AGREE WITH THE DEFECT BY CONSTRUCTION. `b_dec` is non-zero
    throughout, because with `b_dec = 0` a centred and an uncentered encode are the same
    function and every assertion about the basis passes against the defect that omits it.
    """

    @staticmethod
    def _slice(*, architecture="jumprelu", thresholds=True, centre=True, d=32, k=8):
        g = torch.Generator().manual_seed(4)
        weight = torch.randn(d, k, generator=g) / d**0.5
        b_dec = torch.randn(d, generator=g) * 0.6 + 0.4          # NOT zero
        x = torch.randn(40, d, generator=g) * 0.05 + 0.02
        centred = normalize(x, "constant_norm_rescale") - (b_dec if centre else 0)
        theta = torch.quantile(centred @ weight, 0.95, dim=0).clamp_min(1e-4)
        return (
            SaeFeatureSlice(
                weight=weight.contiguous(),
                bias=torch.zeros(k),
                architecture=architecture,
                normalization_mode="constant_norm_rescale",
                thresholds=theta.contiguous() if thresholds else None,
                feature_indices=tuple(range(k)),
                decoder_bias=b_dec.contiguous() if centre else None,
            ),
            x,
        )

    def test_a_relu_basis_has_no_unreproducible_positions(self):
        """Its gate is at 0, so crossing it moves the feature by ~0. Nothing to set aside."""
        slice_, x = self._slice(architecture="standard", thresholds=False)
        assert slice_.is_stepped is False
        risk = slice_.flip_risk(x)
        assert risk.shape == (40,)
        assert not bool(risk.any())

    def test_a_jumprelu_basis_is_stepped(self):
        slice_, _ = self._slice()
        assert slice_.is_stepped is True

    def test_a_relu_basis_CARRYING_thresholds_is_still_not_stepped(self):
        """⚠ `from_state_dict` reads `threshold` from the file whatever the architecture says, so
        a relu SAE published beside a threshold tensor arrives with one. Deciding on the presence
        of the tensor instead of the architecture would set positions aside for a basis with no
        step in it — weakening the parity gate for every such probe, silently.

        A mutation control found this: the first version of this class only ever built a relu
        slice with `thresholds=None`, so it agreed with that defect by construction.
        """
        stepped, x = self._slice(architecture="jumprelu")
        flat = replace(stepped, architecture="standard")
        assert flat.thresholds is not None
        assert flat.is_stepped is False
        assert not bool(flat.flip_risk(x).any())

    def test_a_feature_sitting_on_its_threshold_is_at_risk(self):
        """The mechanism, stated directly: move one threshold onto one token's pre-activation."""
        slice_, x = self._slice()
        pre = slice_.pre_activations(x)
        token, feature = 7, 2
        theta = slice_.thresholds.clone()
        theta[feature] = pre[token, feature]
        on_the_edge = replace(slice_, thresholds=theta)
        assert bool(on_the_edge.flip_risk(x)[token])

    def test_a_feature_far_from_its_threshold_is_not(self):
        slice_, x = self._slice()
        theta = torch.full_like(slice_.thresholds, 1e6)   # unreachable: nothing can flip
        assert not bool(replace(slice_, thresholds=theta).flip_risk(x).any())

    def test_the_band_scales_with_the_precision_gap(self):
        """A wider producer/consumer gap puts MORE positions out of reach, never fewer."""
        slice_, x = self._slice()
        narrow = int(slice_.flip_risk(x, precision_gap=1e-9).sum())
        wide = int(slice_.flip_risk(x, precision_gap=0.5).sum())
        assert wide > narrow

    def test_it_is_judged_per_position_not_on_a_corpus_average(self):
        """A token whose residual is ten times larger has a ten-times wider band.

        A single corpus-wide slack would mark the small-residual positions as risky and the
        large ones as safe, which is backwards.
        """
        slice_, x = self._slice()
        big = x.clone()
        big[0] *= 10.0
        pre = slice_.pre_activations(big)
        # put every threshold a hair outside the band that position 1 alone would admit
        theta = pre[0] + 0.02
        risk = replace(slice_, thresholds=theta.contiguous()).flip_risk(big)
        assert bool(risk[0]), "the position the thresholds were placed against must be at risk"

    def test_flip_risk_and_encode_read_the_same_pre_activations(self):
        """One source for the pre-activation, so a threshold cannot be judged against a number
        the encoder does not use — including the `- b_dec` centring."""
        slice_, x = self._slice()
        pre = slice_.pre_activations(x)
        encoded = slice_.encode(x)
        gated = torch.where(pre > slice_.thresholds, pre, torch.zeros_like(pre))
        assert torch.equal(encoded, gated)

    def test_pre_activations_are_centred_by_the_decoder_bias(self):
        """With a non-zero b_dec the centred and uncentered pre-activations must differ."""
        slice_, x = self._slice()
        uncentered = replace(slice_, decoder_bias=None)
        assert not torch.allclose(slice_.pre_activations(x), uncentered.pre_activations(x))

    def test_one_sigma_lets_real_flips_escape_and_three_does_not(self):
        """Why `FLIP_RISK_SIGMAS` is 3 and not 1 — measured, on a real flip.

        `W_enc_j . dx` is a random variable whose standard deviation is
        `||W_enc_j|| ||dx|| / sqrt(d_model)`, so a one-sigma band is exceeded by a large minority
        of features. A flip outside the band lands in the subset the parity gate trusts and
        refuses a correct build. 400 positions at the reference probe's own firing rate (0.6%):
        two of nine real flips escape one sigma, none escapes three.
        """
        d, k, t, rate = 64, 16, 400, 0.006
        g = torch.Generator().manual_seed(4)
        weight = torch.randn(d, k, generator=g) / d**0.5
        b_dec = torch.randn(d, generator=g) * 0.6 + 0.4
        consumer = torch.randn(t, d, generator=g) * 0.05 + 0.02
        producer = consumer * (
            1.0 + PRODUCER_PRECISION_GAP * torch.randn(consumer.shape, generator=g)
        )
        scaffold = SaeFeatureSlice(
            weight=weight.contiguous(),
            bias=torch.zeros(k),
            architecture="jumprelu",
            normalization_mode="constant_norm_rescale",
            thresholds=torch.zeros(k),
            feature_indices=tuple(range(k)),
            decoder_bias=b_dec.contiguous(),
        )
        theta = torch.quantile(scaffold.pre_activations(consumer), 1 - rate, dim=0).clamp_min(1e-4)
        slice_ = replace(scaffold, thresholds=theta.contiguous())

        flipped = (
            (slice_.encode(consumer) != 0) != (slice_.encode(producer) != 0)
        ).any(dim=1)
        assert int(flipped.sum()) > 0, "no flip in the fixture; the test would prove nothing"
        escaped_at_one = int((flipped & ~slice_.flip_risk(consumer, sigmas=1.0)).sum())
        escaped_at_three = int((flipped & ~slice_.flip_risk(consumer, sigmas=3.0)).sum())
        assert escaped_at_one > 0
        assert escaped_at_three == 0
        assert FLIP_RISK_SIGMAS >= 3.0

    def test_three_sigmas_still_leaves_most_positions_comparable(self):
        """A band that swallows the sequence would make the gate vacuous rather than tight."""
        slice_, x = self._slice(d=64, k=16)
        assert float((~slice_.flip_risk(x)).float().mean()) > 0.5
