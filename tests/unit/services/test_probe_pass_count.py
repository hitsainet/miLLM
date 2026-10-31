"""A forward pass is counted ONCE however many layers carry probes (037 hardware finding).

The hook fires on every hooked layer. Counting each call divided the per-pass cost by the number of
layers: on the 3090, eight probes over five layers reported 0.155 ms per "pass" against 0.775 ms per
forward pass. Mutation control: counting on every layer again turns this red.
"""

import torch

from millm.ml.probe_head import ProbeHead
from millm.services.probe_runtime import ArmedProbe, ProbeRequestContext


def probe(pid, layer):
    return ArmedProbe(probe_id=pid, name=pid, head=ProbeHead(weight=torch.ones(4), bias=0.0,
                      layer=layer), rule="mean", scope="all", layer=layer, rung=2,
                      rung_language="x", threshold=1.0, windows=("all",))


def test_three_hooked_layers_three_forward_passes_count_three():
    context = ProbeRequestContext("r", [probe("a", 7), probe("b", 2), probe("c", 11)])
    for _ in range(3):  # each forward pass visits the hooked layers in order
        for layer in (2, 7, 11):
            context.observe(layer, torch.zeros(1, 1, 4))
    assert context.n_passes == 3


def test_one_layer_counts_each_pass():
    context = ProbeRequestContext("r", [probe("a", 5), probe("b", 5)])
    for _ in range(4):
        context.observe(5, torch.zeros(1, 1, 4))
    assert context.n_passes == 4
