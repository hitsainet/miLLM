"""Armed probes, the per-request context, and the properties that keep a verdict honest.

Three things here are load-bearing and easy to get subtly wrong:

* **One hook per layer.** Arming eight probes on one layer must install ONE hook, or the
  device-to-host budget is blown eight times over.
* **`fires=None` is not `fires=False`.** A probe with no threshold ranks without deciding, and a
  probe that was not scored said nothing at all. Reporting either as "did not fire" is a verdict
  it never gave.
* **Every armed probe produces a verdict**, scored or not. Silence and a negative are
  indistinguishable to a reader, and only one of them is true.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

from millm.ml.probe_head import ProbeHead
from millm.services.probe_runtime import ArmedProbe, ProbeRequestContext, ProbeRuntimeState


class TinyLayer(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.linear = nn.Linear(d, d, bias=False)
        with torch.no_grad():
            self.linear.weight.copy_(torch.eye(d))

    def forward(self, x):
        return (self.linear(x), None)


class TinyModel(nn.Module):
    def __init__(self, d: int = 4, n: int = 3):
        super().__init__()
        inner = nn.Module()
        inner.layers = nn.ModuleList([TinyLayer(d) for _ in range(n)])
        self.model = inner

    def forward(self, x):
        for layer in self.model.layers:
            x = layer(x)[0]
        return x


def make_probe(probe_id="pr_1", *, layer=1, rule="mean", threshold=1.0, d=4, name=None):
    return ArmedProbe(
        probe_id=probe_id,
        name=name or probe_id,
        head=ProbeHead(weight=torch.ones(d), bias=0.0, layer=layer),
        rule=rule,
        scope="all",
        layer=layer,
        rung=2,
        rung_language="detects on unseen tasks",
        threshold=threshold,
    )


@pytest.fixture(autouse=True)
def clean_state():
    ProbeRuntimeState.reset_for_tests()
    yield
    ProbeRuntimeState.reset_for_tests()


class TestHookOwnership:
    def test_arming_several_probes_on_one_layer_installs_ONE_hook(self):
        """The 5 ms budget is only reachable with one device-to-host copy per pass."""
        model = TinyModel()
        state = ProbeRuntimeState()
        for i in range(4):
            state.arm(make_probe(f"pr_{i}", layer=1), model)
        assert len(state._handles) == 1
        assert len(model.model.layers[1]._forward_hooks) == 1

    def test_probes_on_different_layers_get_their_own_hooks(self):
        model = TinyModel()
        state = ProbeRuntimeState()
        state.arm(make_probe("pr_a", layer=0), model)
        state.arm(make_probe("pr_b", layer=2), model)
        assert state.layers() == {0, 2}
        assert len(state._handles) == 2

    def test_the_hook_survives_until_the_LAST_probe_on_its_layer_leaves(self):
        model = TinyModel()
        state = ProbeRuntimeState()
        state.arm(make_probe("pr_a", layer=1), model)
        state.arm(make_probe("pr_b", layer=1), model)
        state.disarm("pr_a")
        assert len(model.model.layers[1]._forward_hooks) == 1, "disarming one removed the shared hook"
        state.disarm("pr_b")
        assert len(model.model.layers[1]._forward_hooks) == 0

    def test_disarm_all_removes_every_hook(self):
        model = TinyModel()
        state = ProbeRuntimeState()
        state.arm(make_probe("pr_a", layer=0), model)
        state.arm(make_probe("pr_b", layer=2), model)
        assert state.disarm_all("model_changed") == ["pr_a", "pr_b"]
        assert state.has_armed() is False
        assert all(not layer._forward_hooks for layer in model.model.layers)

    def test_disarming_something_unarmed_is_false_not_an_error(self):
        assert ProbeRuntimeState().disarm("pr_nope") is False


class TestTheRequestSlot:
    def test_no_context_when_nothing_is_armed(self):
        assert ProbeRuntimeState().begin_request("chatcmpl-a") is None

    def test_a_second_concurrent_begin_RAISES_rather_than_overwriting(self):
        """⚠ Silently replacing the slot would attribute one request's activations to another's
        verdict. Serial execution should make this impossible; the guard proves it."""
        model = TinyModel()
        state = ProbeRuntimeState()
        state.arm(make_probe(), model)
        state.begin_request("chatcmpl-a")
        with pytest.raises(RuntimeError, match="already open"):
            state.begin_request("chatcmpl-b")

    def test_end_request_releases_the_slot(self):
        model = TinyModel()
        state = ProbeRuntimeState()
        state.arm(make_probe(), model)
        state.begin_request("chatcmpl-a")
        assert state.end_request().request_id == "chatcmpl-a"
        assert state.current_request() is None
        state.begin_request("chatcmpl-b")  # must not raise


class TestScoring:
    def test_a_forward_pass_reaches_the_context(self):
        model = TinyModel()
        state = ProbeRuntimeState()
        state.arm(make_probe(threshold=0.0), model)
        context = state.begin_request("chatcmpl-a")
        model(torch.ones(1, 5, 4))
        verdicts = state.end_request().finish()
        assert len(verdicts) == 1
        assert verdicts[0].scored is True
        assert verdicts[0].n_scored_tokens == 5
        # weight=ones, bias=0, activations=ones over d=4 -> 4.0 per token, mean 4.0
        assert verdicts[0].score == pytest.approx(4.0)
        assert verdicts[0].fires is True

    def test_below_the_threshold_fires_false(self):
        model = TinyModel()
        state = ProbeRuntimeState()
        state.arm(make_probe(threshold=100.0), model)
        state.begin_request("chatcmpl-a")
        model(torch.ones(1, 3, 4))
        assert state.end_request().finish()[0].fires is False

    def test_NO_threshold_means_fires_is_None_not_False(self):
        """⚠ The probe ranks but does not decide. `False` would be a verdict it never gave."""
        model = TinyModel()
        state = ProbeRuntimeState()
        state.arm(make_probe(threshold=None), model)
        state.begin_request("chatcmpl-a")
        model(torch.ones(1, 3, 4))
        verdict = state.end_request().finish()[0]
        assert verdict.scored is True
        assert verdict.score is not None
        assert verdict.fires is None

    def test_top_positions_are_the_highest_scoring(self):
        probe = make_probe(threshold=0.0)
        context = ProbeRequestContext("chatcmpl-a", [probe])
        hidden = torch.zeros(1, 6, 4)
        hidden[0, 2] = 5.0
        hidden[0, 4] = 3.0
        context.observe(1, hidden)
        verdict = context.finish()[0]
        assert verdict.top_positions[:2] == [2, 4]

    def test_every_armed_probe_gets_a_verdict(self):
        model = TinyModel()
        state = ProbeRuntimeState()
        state.arm(make_probe("pr_a", layer=0), model)
        state.arm(make_probe("pr_b", layer=2), model)
        state.begin_request("chatcmpl-a")
        model(torch.ones(1, 3, 4))
        assert {v.probe_id for v in state.end_request().finish()} == {"pr_a", "pr_b"}


class TestNotScored:
    def test_a_batched_pass_is_not_scored_with_a_reason(self):
        probe = make_probe()
        context = ProbeRequestContext("chatcmpl-a", [probe])
        context.observe(1, torch.ones(4, 5, 4))
        verdict = context.finish()[0]
        assert verdict.scored is False
        assert verdict.not_scored_reason == "batched_request"
        assert verdict.score is None and verdict.fires is None

    def test_the_FIRST_reason_wins(self):
        """A request that hit speculative decoding and then batched is best described by what
        happened first; overwriting reports the last symptom rather than the cause."""
        context = ProbeRequestContext("chatcmpl-a", [make_probe()])
        context.mark_not_scored("speculative_decoding")
        context.mark_not_scored("batched_request")
        assert context.not_scored_reason == "speculative_decoding"

    def test_a_mask_that_selects_nothing_is_not_scored_rather_than_zero(self):
        """The probe never looked. A score of 0 would be a measurement it did not make."""
        probe = make_probe()
        context = ProbeRequestContext("chatcmpl-a", [probe])
        context.observe(1, torch.ones(1, 4, 4), mask=[False] * 4)
        verdict = context.finish()[0]
        assert verdict.scored is False
        assert verdict.not_scored_reason == "no_scored_tokens"

    def test_a_head_of_the_wrong_width_declines_rather_than_scoring_noise(self):
        probe = make_probe(d=8)
        context = ProbeRequestContext("chatcmpl-a", [probe])
        context.observe(1, torch.ones(1, 4, 4))
        assert context.finish()[0].not_scored_reason == "head_mismatch"

    def test_once_not_scored_further_passes_are_ignored(self):
        context = ProbeRequestContext("chatcmpl-a", [make_probe()])
        context.mark_not_scored("speculative_decoding")
        context.observe(1, torch.ones(1, 4, 4))
        assert context.finish()[0].not_scored_reason == "speculative_decoding"


class TestMasking:
    def test_only_masked_positions_are_scored(self):
        probe = make_probe(threshold=0.0)
        context = ProbeRequestContext("chatcmpl-a", [probe])
        hidden = torch.zeros(1, 4, 4)
        hidden[0, 0] = 100.0      # excluded by the mask
        hidden[0, 3] = 1.0
        context.observe(1, hidden, mask=[False, False, False, True])
        verdict = context.finish()[0]
        assert verdict.n_scored_tokens == 1
        assert verdict.score == pytest.approx(4.0)

    def test_a_mask_shorter_than_the_pass_pads_closed_not_open(self):
        """Padding with True would score positions nobody vouched for."""
        probe = make_probe(threshold=0.0)
        context = ProbeRequestContext("chatcmpl-a", [probe])
        context.observe(1, torch.ones(1, 5, 4), mask=[True, True])
        assert context.finish()[0].n_scored_tokens == 2

class TestFlipRiskCollection:
    """The unreproducible-position collector is OFF on the hot path and ON for parity only."""

    class _Encoder:
        """An encoder that records whether its flip_risk was asked for."""

        def __init__(self, d, k):
            self.weight = torch.eye(d, k)
            self.asked = 0

        def __call__(self, x):
            return x[..., : self.weight.shape[1]]

        def flip_risk(self, x):
            self.asked += 1
            return torch.zeros(x.shape[0], dtype=torch.bool)

    def test_a_served_request_never_computes_it(self):
        """It re-derives the pre-activations, which a 5 ms budget cannot afford per request."""
        encoder = self._Encoder(4, 4)
        probe = make_probe()
        probe = ArmedProbe(**{**probe.__dict__, "encoder": encoder})
        context = ProbeRequestContext("r1", [probe])
        context.observe(1, torch.ones(1, 3, 4))
        assert encoder.asked == 0
        assert context.flip_risk_for(probe.probe_id) == []

    def test_parity_asks_for_it(self):
        encoder = self._Encoder(4, 4)
        probe = make_probe()
        probe = ArmedProbe(**{**probe.__dict__, "encoder": encoder})
        context = ProbeRequestContext("r1", [probe], collect_flip_risk=True)
        context.observe(1, torch.ones(1, 3, 4))
        assert encoder.asked == 1
        assert context.flip_risk_for(probe.probe_id) == [False, False, False]

    def test_an_encoder_that_cannot_say_records_FALSE_not_TRUE(self):
        """⚠ Defaulting to True would set every position aside on no evidence, which is how a
        gate stops gating. A dense probe has no encoder at all and must contribute nothing."""
        probe = make_probe()
        context = ProbeRequestContext("r1", [probe], collect_flip_risk=True)
        context.observe(1, torch.ones(1, 3, 4))
        assert context.flip_risk_for(probe.probe_id) == [False, False, False]

    def test_it_is_aligned_with_the_scored_positions(self):
        """Same positions, same order, as `token_scores_for` — the parity engine pairs them."""
        encoder = self._Encoder(4, 4)
        probe = make_probe()
        probe = ArmedProbe(**{**probe.__dict__, "encoder": encoder})
        context = ProbeRequestContext("r1", [probe], collect_flip_risk=True)
        context.observe(1, torch.ones(1, 4, 4), mask=[True, False, True, False])
        assert len(context.flip_risk_for(probe.probe_id)) == len(
            context.token_scores_for(probe.probe_id)
        ) == 2
