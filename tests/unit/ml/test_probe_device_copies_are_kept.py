"""A probe's device copies are uploaded ONCE per device, not once per pass or per request (024 10.4).

On the RTX 3090 (2026-10-17, LFM2.5-1.2B, a 2,985-token prompt) the honest per-pass timer showed
86 ms of HOST time on the prefill pass of a k-sparse probe: `SaeFeatureSlice.encode` rebuilt its
device copy on every pass, a pageable host-to-device copy that blocks the host until the stream
reaches it, and the dense `ProbeHead` was re-uploaded once per request the same way. `meta` stands
in for the GPU so the copy path runs without one.
"""

from __future__ import annotations

import torch

from millm.ml.probe_head import ProbeHead
from millm.services.probe_sae_slice import SaeFeatureSlice

META = torch.device("meta")


def test_the_head_is_moved_once_per_device():
    head = ProbeHead(weight=torch.randn(8), mean=torch.zeros(8), std=torch.ones(8))
    first = head.to_device(META)
    assert first is not head and first.weight.device.type == "meta"
    assert head.to_device(META) is first, "a second request re-uploaded the head"
    assert head.to_device(torch.device("cpu")) is head


def test_the_slice_is_moved_once_per_device():
    slice_ = SaeFeatureSlice(weight=torch.randn(8, 4), bias=torch.zeros(4), architecture="jumprelu",
                             thresholds=torch.zeros(4), feature_indices=(1, 2, 3, 4),
                             decoder_bias=torch.zeros(8))
    first = slice_.to_device(META)
    assert first.weight.device.type == "meta" and first.thresholds.device.type == "meta"
    assert slice_.to_device(META) is first, "every pass re-uploaded the slice"


def test_the_memo_is_not_part_of_equality():
    a = ProbeHead(weight=torch.ones(4))
    b = ProbeHead(weight=torch.ones(4))
    a.to_device(META)
    assert a.__dict__.get("_device_copies") and not b.__dict__.get("_device_copies")
    assert [f for f in a.__dataclass_fields__] == [f for f in b.__dataclass_fields__]
