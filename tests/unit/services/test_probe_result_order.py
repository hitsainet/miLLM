"""The contract order and the `set` block (039 FR-39.10, FR-39.4; TD39-30; FTASKS 4.1, 4.2).

Results are built by 038's builder from real `Verdict`s over armed probes whose descriptors carry
their behaviour; `order_results` is what `_probe_publish_results` applies before publishing.

Control: sort by name alone → `test_a_behaviours_results_are_contiguous` red (the names are chosen
so alphabetical order interleaves the two behaviours).
"""

from __future__ import annotations

from types import SimpleNamespace

import torch

from millm.ml.probe_head import ProbeHead
from millm.services import probe_selection
from millm.services.probe_result import build_probe_results
from millm.services.probe_result_order import order_results, result_sort_key
from millm.services.probe_runtime import ArmedProbe, ProbeDescriptor, Verdict
from tests.support import behaviour_factory as bf


def _armed(pid, name, key, version=1, digest=None):
    return ArmedProbe(
        probe_id=pid, name=name, head=ProbeHead(weight=torch.ones(2)), rule="mean", scope="all",
        layer=0, rung=2, rung_language="detects on unseen tasks", threshold=0.5,
        descriptor=ProbeDescriptor(behaviour_key=key, behaviour_version=version,
                                   detector_digest=digest),
    )


def _verdicts(probes, windows=("all", "prompt")):
    return [Verdict(probe_id=p.probe_id, name=p.name, rung=2,
                    rung_language="detects on unseen tasks", scored=True, score=1.0 + i,
                    threshold=0.5, fires=True, n_scored_tokens=3, window=w)
            for i, p in enumerate(probes) for w in windows]


def _results(probes):
    context = SimpleNamespace(request_id="req-order", probes=probes, session=None, trace=None,
                              interrupted=None)
    return build_probe_results(_verdicts(probes), context)


def test_a_behaviours_results_are_contiguous():
    a, b = "b-aaa" + bf.random_key("x")[1:6], "b-zzz" + bf.random_key("y")[1:6]
    probes = [_armed("pr_000000000001", "alpha", a), _armed("pr_000000000002", "beta", b),
              _armed("pr_000000000003", "gamma", a), _armed("pr_000000000004", "zeta", None)]
    ordered = order_results(_results(probes), probes)
    keys = [r.behaviour.key if r.behaviour else None for r in ordered]
    assert keys == [a] * 4 + [b] * 2 + [None] * 2, keys
    assert [(r.probe_name, r.window) for r in ordered[:4]] == [
        ("alpha", "all"), ("alpha", "prompt"), ("gamma", "all"), ("gamma", "prompt")]
    assert ordered == sorted(ordered, key=result_sort_key)


def test_the_set_block_counts_armed_against_imported():
    """FR-39.4: three of a four-member set armed → `armed_members: 3` against 4, the missing one
    named; a behaviour-less result carries no set."""
    key = bf.random_key()
    ids = [f"pr_00000000000{i}" for i in range(1, 5)]
    probe_selection._rebuild_groups({pid: ("tiny/llama", key, 1) for pid in ids})
    try:
        probes = [_armed(pid, f"p{i}", key, digest=f"{i}" * 64) for i, pid in enumerate(ids[:3])]
        loose = _armed("pr_0000000000ff", "loose", None)
        ordered = order_results(_results(probes + [loose]), probes + [loose])
        blocks = {r.probe_id: r.set_ for r in ordered}
        block = blocks[ids[0]]
        assert (block.armed_members, block.total_members, block.complete) == (3, 4, False)
        assert block.missing_probe_ids == [ids[3]]
        assert len({b.members_digest for pid, b in blocks.items() if pid in ids}) == 1
        assert blocks[loose.probe_id] is None
        assert block.attached_rule_sets == []
    finally:
        probe_selection.forget_for_tests()


def test_the_members_digest_moves_with_a_members_detector():
    from millm.services.probe_result_order import members_digest

    one = members_digest({"pr_a": "1" * 64, "pr_b": "2" * 64})
    assert one == members_digest({"pr_b": "2" * 64, "pr_a": "1" * 64}), "probe-id order"
    assert one != members_digest({"pr_a": "1" * 64, "pr_b": "3" * 64})
