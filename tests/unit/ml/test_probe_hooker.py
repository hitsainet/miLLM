"""The read-only prepended hook.

The load-bearing test is `TestItReadsBeforeSteering`. `prepend=True` has no precedent anywhere in
this repository, so nothing else would notice if it were dropped — and dropping it is invisible:
the probe still scores, still produces plausible numbers, and is simply reading the residual *after*
steering has moved it. A monitor whose reading can be moved by the thing it monitors is worse than
no monitor, because it reports confidently.

⚠ That test steers the SAME layer the probe reads. A test that steers a different layer passes
whether or not the hook is prepended, which would be the classic "verified for the wrong reason".
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

from millm.ml.probe_hooker import ProbeHooker, extract_hidden_states, is_single_row


class TinyLayer(nn.Module):
    """A decoder layer that returns a tuple, as HF layers do."""

    def __init__(self, d: int):
        super().__init__()
        self.linear = nn.Linear(d, d, bias=False)
        with torch.no_grad():
            self.linear.weight.copy_(torch.eye(d))

    def forward(self, x):
        return (self.linear(x), None)


class TinyModel(nn.Module):
    """`model.model.layers[i]` — the first pattern `get_layer` tries."""

    def __init__(self, d: int = 4, n: int = 3):
        super().__init__()
        inner = nn.Module()
        inner.layers = nn.ModuleList([TinyLayer(d) for _ in range(n)])
        self.model = inner

    def forward(self, x):
        for layer in self.model.layers:
            x = layer(x)[0]
        return x


@pytest.fixture
def model():
    return TinyModel()


@pytest.fixture
def x():
    return torch.ones(1, 5, 4)


class TestItReadsBeforeSteering:
    def test_the_probe_sees_the_pre_steer_residual(self, model, x):
        """An SAE steering the same layer must not move what the probe reads."""
        seen: list[torch.Tensor] = []
        hooker = ProbeHooker()
        probe_handle = hooker.install(model, 1, lambda h: seen.append(h.clone()))

        # A steering hook on the SAME layer, registered the way SAEHooker does it: no prepend.
        def steer(_m, _i, output):
            return (output[0] + 100.0, output[1])

        steer_handle = model.model.layers[1].register_forward_hook(steer)
        try:
            out = model(x)
        finally:
            steer_handle.remove()
            hooker.remove(probe_handle)

        assert len(seen) == 1
        assert seen[0].abs().max().item() == pytest.approx(1.0), (
            "the probe read a steered value — the hook is not running first"
        )
        # And steering did happen, so the test is not passing because nothing steered.
        assert out.abs().max().item() == pytest.approx(101.0)

    def test_the_hook_is_registered_ahead_of_an_existing_hook(self, model):
        """Order is the mechanism, so assert it directly as well as by effect."""
        order: list[str] = []
        model.model.layers[0].register_forward_hook(lambda m, i, o: order.append("sae") or None)
        hooker = ProbeHooker()
        h = hooker.install(model, 0, lambda _h: order.append("probe"))
        try:
            model(torch.ones(1, 2, 4))
        finally:
            hooker.remove(h)
        assert order[:2] == ["probe", "sae"]


class TestItIsReadOnly:
    def test_the_model_output_is_unchanged(self, model, x):
        baseline = model(x).clone()
        hooker = ProbeHooker()
        h = hooker.install(model, 1, lambda _h: None)
        try:
            with_hook = model(x)
        finally:
            hooker.remove(h)
        assert torch.equal(baseline, with_hook)

    def test_a_callback_that_raises_does_not_break_generation(self, model, x):
        """A probe must never take the forward pass down with it."""
        hooker = ProbeHooker()
        h = hooker.install(model, 1, lambda _h: (_ for _ in ()).throw(RuntimeError("boom")))
        try:
            out = model(x)  # must not raise
        finally:
            hooker.remove(h)
        assert out.shape == x.shape

    def test_removal_actually_detaches(self, model, x):
        seen = []
        hooker = ProbeHooker()
        h = hooker.install(model, 1, seen.append)
        model(x)
        hooker.remove(h)
        model(x)
        assert len(seen) == 1, "the callback fired after removal"


class TestOneHookServesEveryProbeOnALayer:
    def test_the_tensor_is_handed_over_once_per_pass(self, model, x):
        """The D2H budget is one copy per pass; that is only possible if the tensor arrives once.

        Several probes on a layer share this single callback and read the same object.
        """
        calls: list[int] = []
        probes = []
        hooker = ProbeHooker()
        h = hooker.install(model, 2, lambda t: (calls.append(1), probes.append(t.sum().item())))
        try:
            model(x)
        finally:
            hooker.remove(h)
        assert len(calls) == 1


class TestOutputShapes:
    def test_a_bare_tensor_is_accepted(self):
        t = torch.ones(1, 3, 4)
        assert extract_hidden_states(t) is t

    def test_a_tuple_yields_its_first_element(self):
        t = torch.ones(1, 3, 4)
        assert extract_hidden_states((t, None)) is t

    def test_last_hidden_state_is_accepted(self):
        class Out:
            last_hidden_state = torch.ones(1, 3, 4)

        assert extract_hidden_states(Out()) is Out.last_hidden_state

    def test_an_unrecognised_shape_returns_none_rather_than_guessing(self):
        """Guessing which element is the residual is silent and plausible when wrong."""
        assert extract_hidden_states({"hidden": torch.ones(1, 3, 4)}) is None
        assert extract_hidden_states((None, torch.ones(1, 3, 4))) is None
        assert extract_hidden_states(()) is None

    def test_an_unrecognised_output_does_not_break_the_pass(self, x):
        class Odd(nn.Module):
            def forward(self, t):
                return {"hidden": t}

        m = TinyModel()
        m.model.layers[1] = Odd()
        seen = []
        hooker = ProbeHooker()
        h = hooker.install(m, 1, seen.append)
        try:
            m.model.layers[1](x)
        finally:
            hooker.remove(h)
        assert seen == [], "an unrecognised output must not be handed to a probe"


class TestBatchDetection:
    def test_one_row_is_scoreable(self):
        assert is_single_row(torch.ones(1, 5, 4)) is True

    def test_a_batched_pass_is_not(self):
        """Scoring row 0 and calling it the request's verdict would attribute another request's
        activations to this one."""
        assert is_single_row(torch.ones(4, 5, 4)) is False

    def test_an_unexpected_rank_is_not(self):
        assert is_single_row(torch.ones(5, 4)) is False
