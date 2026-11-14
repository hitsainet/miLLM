"""G38 review round 3 (miLLM): lines no earlier test caught (mutation survivors), pinned.

* Chunked prefill: the round-2 decode shortcut (`_position >= skip` → the cached all-true window) is
  correct only for passes that START at or after P. Every streaming test prefilled the whole prompt in
  one pass, so a shortcut keyed on "not the first pass" passed them all while scoring the prefix of a
  second prefill chunk. Each split below starts a pass INSIDE the prefix after the first.
* Two probes on one request, the plain one FIRST: the per-pass window cache is keyed by (scope, P).
* Arming: an explicit `false` in a stored document is not skipping (absent and false are one value).
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from miserver.ml.probe_head import ProbeHead
from miserver.services.probe_runtime import ArmedProbe, ProbeRequestContext

D = 4
VALUES = [50, 40, 30, 20, 3, 9, 1, 4, 100, 2, 6]      # P = 4 (a descending prefix), prompt = 8
P, PROMPT = 4, 8


def _probe(probe_id, skip, rule="max", params=None):
    return ArmedProbe(
        probe_id=probe_id, name=probe_id,
        head=ProbeHead(weight=torch.tensor([1.0, 0.0, 0.0, 0.0]), bias=0.0),
        rule=rule, scope="all", layer=1, rung=2, rung_language="held-out", threshold=None,
        windows=("all", "prompt", "response"), skip_template_prefix=skip, rule_params=params or {},
    )


def _acts(values):
    row = torch.zeros(1, len(values), D)
    row[0, :, 0] = torch.tensor([float(v) for v in values])
    return row


def _run(probes, passes):
    ctx = ProbeRequestContext("req", probes)
    ctx.set_prompt_length(PROMPT)
    ctx.set_template_prefix(P)
    start = 0
    for size in passes:
        ctx.observe(1, _acts(VALUES[start: start + size]))
        start += size
    assert start == len(VALUES)
    return {(v.probe_id, v.window): (v.scored, v.score, v.n_scored_tokens) for v in ctx.finish()}


@pytest.mark.parametrize("rule,params", [("max", None), ("mean", None), ("rolling_mean_max", {"window": 2})])
@pytest.mark.parametrize("passes", [
    [2, 6, 1, 1, 1],          # second prefill chunk starts at 2, inside P
    [1, 1, 1, 5, 1, 1, 1],    # three chunks inside the prefix
    [3, 2, 3, 1, 1, 1],       # a chunk straddles P
])
def test_chunked_prefill_equals_one_pass(rule, params, passes):
    def probes():
        return [_probe("plain", False, rule, params), _probe("skip", True, rule, params)]

    batch = _run(probes(), [len(VALUES)])
    chunked = _run(probes(), passes)
    assert chunked == batch
    # The fixture separates the two behaviours: the prefix holds the maximum.
    assert batch[("skip", "all")][2] == len(VALUES) - P
    assert batch[("plain", "all")][2] == len(VALUES)
    if rule == "max":
        assert batch[("plain", "all")][1] == 100 and batch[("skip", "prompt")][1] == 9


@pytest.mark.parametrize("order", ["plain_first", "skip_first"])
def test_a_plain_and_a_skipping_probe_on_one_request_read_their_own_windows(order):
    plain, skip = _probe("plain", False), _probe("skip", True)
    probes = [plain, skip] if order == "plain_first" else [skip, plain]
    for passes in ([len(VALUES)], [PROMPT, 1, 1, 1]):
        got = _run(probes, passes)
        assert got[("plain", "all")][2] == len(VALUES) and got[("plain", "prompt")] == (True, 50, PROMPT)
        assert got[("skip", "all")][2] == len(VALUES) - P and got[("skip", "prompt")] == (True, 9, PROMPT - P)


def test_arming_reads_an_explicit_false_as_not_skipping():
    from tests.unit.probe_fixtures import probe_definition

    from miserver.services.probe_arming import armed_probe_from_row

    document = probe_definition()
    document["aggregation"]["skip_template_prefix"] = False
    row = SimpleNamespace(id="pr_1", name="p", definition=document, rule="max", scope="all", layer=1,
                          rung=2, threshold=None, threshold_revision=1)
    assert armed_probe_from_row(row).skip_template_prefix is False
