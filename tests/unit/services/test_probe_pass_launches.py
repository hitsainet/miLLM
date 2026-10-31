"""A probe pass launches only the kernels its arithmetic needs (024 10.4, 2026-10-17).

Profiled on the RTX 3090 (LFM2.5-1.2B, a k-sparse + a dense probe on L11): ~26 kernel launches
per decode pass on a launch-bound model, of which four were bookkeeping: the bf16 row cast to
float32 TWICE (by the slice and by the dense head), a `ones_like` and a `zeros_like` that `where`
only selects from, and a fresh all-true window every pass. Each is removed with every value
bit-identical; these tests pin both halves — the values, and the ops that are gone.
"""

from __future__ import annotations

import torch
from torch.profiler import ProfilerActivity, profile

from millm.ml.probe_head import ProbeHead
from millm.services.probe_runtime import ArmedProbe, ProbeRequestContext
from millm.services.probe_sae_slice import SaeFeatureSlice, normalize

D, K = 16, 6


def _slice() -> SaeFeatureSlice:
    torch.manual_seed(1)
    return SaeFeatureSlice(weight=torch.randn(D, K), bias=torch.randn(K) * 0.1,
                           architecture="jumprelu", normalization_mode="constant_norm_rescale",
                           thresholds=torch.rand(K) * 0.5, feature_indices=tuple(range(K)),
                           decoder_bias=torch.randn(D) * 0.1)


def _probes() -> list[ArmedProbe]:
    common = dict(rule="mean", scope="all", layer=1, rung=2, rung_language="detects on unseen tasks")
    return [
        ArmedProbe(probe_id="dense", name="d", head=ProbeHead(weight=torch.randn(D),
                   mean=torch.zeros(D), std=torch.ones(D)), **common),
        ArmedProbe(probe_id="sparse", name="s", head=ProbeHead(weight=torch.randn(K),
                   mean=torch.zeros(K), std=torch.ones(K)), encoder=_slice(), **common),
    ]


def test_normalize_is_bit_identical_to_the_ones_like_form():
    x = torch.randn(5, D)
    x[2] = 0.0   # the zero-norm row the `where` exists for
    d = x.shape[-1]
    norm = x.norm(dim=-1, keepdim=True)
    old = x * torch.where(norm > 0, (d**0.5) / norm.clamp_min(torch.finfo(x.dtype).tiny),
                          torch.ones_like(norm))
    assert torch.equal(normalize(x, "constant_norm_rescale"), old)


def test_jumprelu_is_bit_identical_to_the_zeros_like_form():
    s = _slice()
    x = torch.randn(7, D)
    pre = s._pre_on_device(s._centred_on_device(normalize(x, s.normalization_mode)))
    old = torch.where(pre > s.thresholds, pre, torch.zeros_like(pre))
    new = s.encode(x)
    assert torch.equal(new, old) and not torch.signbit(new[new == 0]).any()


def test_scores_equal_each_probe_scoring_the_row_on_its_own():
    hidden = torch.randn(1, 4, D).to(torch.bfloat16)
    probes = _probes()
    ctx = ProbeRequestContext("r", probes)
    ctx.observe(1, hidden)
    row = hidden[0]
    assert torch.equal(ctx._scores["dense"][0], probes[0].head.token_scores(row))
    assert torch.equal(ctx._scores["sparse"][0], probes[1].head.token_scores(probes[1].encoder(row)))


def _ops_in_one_pass(ctx: ProbeRequestContext, hidden: torch.Tensor) -> dict[str, int]:
    """Op counts for one pass. `_to_copy` counts only TENSOR casts: a Python scalar handed to
    `where` is wrapped as a 0-dim CPU number, which launches no device kernel."""
    with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
        ctx.observe(1, hidden)
    ops = {e.key: e.count for e in prof.key_averages()}
    ops["aten::_to_copy"] = sum(
        1 for e in prof.events()
        if e.name == "aten::_to_copy" and e.input_shapes and e.input_shapes[0]
    )
    return ops


def test_the_row_is_cast_once_and_no_fill_is_spent_on_a_selection():
    ctx = ProbeRequestContext("r", _probes())
    hidden = torch.randn(1, 1, D).to(torch.bfloat16)
    ctx.observe(1, hidden)                      # first pass builds the window
    ops = _ops_in_one_pass(ctx, hidden)         # a decode pass
    assert ops.get("aten::_to_copy", 0) == 1, ops   # one shared float32 cast, not one per probe
    assert "aten::ones_like" not in ops and "aten::zeros_like" not in ops, ops
    assert "aten::ones" not in ops, "the all-true window was rebuilt on a decode pass"


def test_a_pre_activation_exactly_at_its_threshold_is_gated_off():
    """miStudio's JumpReLU is `z * (z > θ)` — strict (`src/ml/sparse_autoencoder.py`,
    `JumpReLUFunction.forward`). Found unpinned 2026-10-17: `>` -> `>=` left all 1,111 probe tests
    green, because random pre-activations never land exactly on θ."""
    base = _slice()
    x = torch.randn(1, D)
    pre = base._pre_on_device(base._centred_on_device(normalize(x, base.normalization_mode)))[0]
    at = SaeFeatureSlice(weight=base.weight, bias=base.bias, architecture="jumprelu",
                         normalization_mode=base.normalization_mode, thresholds=pre.clone(),
                         feature_indices=base.feature_indices, decoder_bias=base.decoder_bias)
    assert torch.equal(at.encode(x), torch.zeros(1, K)), "z == θ must be gated off (strict >)"
    below = SaeFeatureSlice(weight=base.weight, bias=base.bias, architecture="jumprelu",
                            normalization_mode=base.normalization_mode,
                            thresholds=torch.nextafter(pre, torch.full_like(pre, -1e9)),
                            feature_indices=base.feature_indices, decoder_bias=base.decoder_bias)
    assert torch.equal(below.encode(x)[0], pre), "just above θ passes through unchanged"
