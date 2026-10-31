"""The arm-time cost measurer (Feature 37, FR-37.2, FR-37.9). CPU only; timers injected.

Mutation controls (037 FTDD §10):
* drop the fixed part from `project`                     → `test_the_fixed_part_is_charged_once_per_layer_and_device` red
* remove the per-device head memo (`probe_head.py`)       → `test_one_head_copy_per_device` red
"""

from __future__ import annotations

import pytest
import torch

from millm.ml.probe_head import ProbeHead
from millm.services import probe_cost
from millm.services.probe_cost import ProbeCost, device_label, measure, project
from millm.services.probe_runtime import ArmedProbe


def armed(probe_id="pr_a", layer=3, d=8) -> ArmedProbe:
    return ArmedProbe(
        probe_id=probe_id, name=probe_id, head=ProbeHead(weight=torch.ones(d), bias=0.0, layer=layer),
        rule="mean", scope="all", layer=layer, rung=2, rung_language="x", threshold=1.0,
        windows=("all",),
    )


def cost(layer, device, decode, fixed, prefill=0.0, prefill_fixed=0.0) -> ProbeCost:
    return ProbeCost(decode_ms=decode, prefill_ms_per_1k=prefill, layer_fixed_ms=fixed,
                     layer_fixed_prefill_ms_per_1k=prefill_fixed, layer=layer, device=device,
                     k=20, measured_at="t")


def test_measure_times_both_shapes_alone_and_twice_on_the_given_device(monkeypatch):
    seen = []

    def fake_observe(candidate, probes, hidden):
        seen.append((tuple(hidden.shape), len(probes), hidden.device.type, hidden.dtype))
        return ("fn", tuple(hidden.shape), len(probes))

    figures = {((1, 1, 8), 1): 0.03, ((1, 1, 8), 2): 0.05,
               ((1, 1024, 8), 1): 0.30, ((1, 1024, 8), 2): 0.50}
    timer_calls = []

    def fake_timer(fn, k, device):
        timer_calls.append((fn[1], fn[2], k, str(device)))
        return figures[(fn[1], fn[2])]

    monkeypatch.setattr(probe_cost, "_observe_fn", fake_observe)
    monkeypatch.setattr(probe_cost, "_timer", fake_timer)
    result = measure(armed(), device=torch.device("cpu"), d_model=8, k=7, dtype=torch.bfloat16)

    assert [s[:2] for s in seen] == [((1, 1, 8), 1), ((1, 1, 8), 2), ((1, 1024, 8), 1),
                                     ((1, 1024, 8), 2)]
    assert {s[2] for s in seen} == {"cpu"} and {s[3] for s in seen} == {torch.bfloat16}
    assert len(timer_calls) == 4 and {c[2] for c in timer_calls} == {7}
    assert result.decode_ms == pytest.approx(0.02)
    assert result.layer_fixed_ms == pytest.approx(0.01)
    assert result.prefill_ms_per_1k == pytest.approx(0.20 * 1000 / 1024)
    assert result.layer_fixed_prefill_ms_per_1k == pytest.approx(0.10 * 1000 / 1024)
    assert (result.layer, result.device, result.k) == (3, "cpu", 7)


def test_measure_runs_the_serving_function_for_real_on_cpu(monkeypatch):
    """With the REAL timer: what is timed is `ProbeRequestContext.observe` scoring the candidate."""
    monkeypatch.setattr(probe_cost, "_timer", probe_cost._default_timer)
    result = measure(armed(), device="cpu", d_model=8, k=3)
    assert result.decode_ms >= 0 and result.layer_fixed_ms >= 0 and result.device == "cpu"


def test_an_unscorable_probe_cannot_be_measured(monkeypatch):
    monkeypatch.setattr(probe_cost, "_timer", probe_cost._default_timer)
    with pytest.raises(Exception):
        measure(armed(d=4), device="cpu", d_model=8, k=2)  # head width 4 against d_model 8


def test_the_default_timer_really_times():
    calls = []
    ms = probe_cost._default_timer(lambda: calls.append(sum(range(2000))), 5, torch.device("cpu"))
    assert ms > 0 and len(calls) == 6  # one warm call, then k


def test_the_fixed_part_is_charged_once_per_layer_and_device():
    a, b = cost(3, "cuda:GPU-a", 0.01, 0.02), cost(3, "cuda:GPU-a", 0.10, 0.02)
    c = cost(20, "cuda:GPU-b", 0.01, 0.03)
    assert project([a]) == {"decode": pytest.approx(0.03), "prefill": 0.0}
    assert project([a, b])["decode"] == pytest.approx(0.01 + 0.10 + 0.02)
    assert project([a, b, c])["decode"] == pytest.approx(0.01 + 0.10 + 0.02 + 0.01 + 0.03)


def test_a_card_is_labelled_by_uuid_never_by_index(monkeypatch):
    asked = []
    monkeypatch.setattr(probe_cost, "_uuid_for_index",
                        lambda i: asked.append(i) or f"GPU-{'abcd'[i]}")
    assert device_label(torch.device("cuda", 1)) == "cuda:GPU-b"
    assert asked == [1]
    assert device_label("cpu") == "cpu" and device_label("meta") == "meta"


def test_one_head_copy_per_device():
    """The per-device memo (024 10.4): each device gets ONE copy, reused, never re-uploaded."""
    head = ProbeHead(weight=torch.ones(8), bias=0.0, layer=0)
    first = head.to_device(torch.device("meta"))
    assert head.to_device(torch.device("meta")) is first
    assert head.to_device(torch.device("cpu")) is head
    assert first.weight.device.type == "meta"


def test_the_cuda_timer_takes_the_larger_of_gpu_and_host_time(monkeypatch):
    """037 hardware finding: during decode the card waits on the host, so a probe whose kernels
    take 0.01 ms of GPU time but 2 ms of Python dispatch costs 2 ms. Fake events, real host clock."""
    import time as _time

    class FakeEvent:
        def __init__(self, enable_timing=True):
            pass

        def record(self, stream):
            pass

        def synchronize(self):
            pass

        def elapsed_time(self, other):
            return 0.01

    monkeypatch.setattr(torch.cuda, "Event", FakeEvent)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda device=None: None)
    ms = probe_cost._cuda_timer(lambda: _time.sleep(0.002), 3, torch.device("cuda", 0))
    assert ms >= 2.0


def test_the_cuda_timer_keeps_gpu_time_when_it_is_larger(monkeypatch):
    class SlowEvent:
        def __init__(self, enable_timing=True):
            pass

        def record(self, stream):
            pass

        def synchronize(self):
            pass

        def elapsed_time(self, other):
            return 5.0

    monkeypatch.setattr(torch.cuda, "Event", SlowEvent)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda device=None: None)
    assert probe_cost._cuda_timer(lambda: None, 3, torch.device("cuda", 0)) == 5.0


def test_the_timed_call_goes_through_a_forward_hook(monkeypatch):
    """037 hardware finding: timing `observe` alone under-predicted a lone probe; the timed call
    is a module forward whose hook extracts the hidden states — the served dispatch."""
    import millm.ml.probe_hooker as hooker

    calls = []
    real = hooker.extract_hidden_states
    monkeypatch.setattr(hooker, "extract_hidden_states", lambda out: calls.append(1) or real(out))
    a = armed()
    fn = probe_cost._observe_fn(a, [a], torch.randn(1, 2, 8))
    fn()
    fn()
    assert len(calls) == 2
