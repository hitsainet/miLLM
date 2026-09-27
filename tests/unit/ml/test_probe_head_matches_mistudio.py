"""miLLM's probe readout must equal miStudio's, numerically, on every rule.

⚠ WHY THIS IS SEPARATE FROM THE PARITY GATE. FR-24.4 re-scores a definition's test vectors at arm
time, which is the right runtime guard — but it only covers the inputs a particular definition
happens to carry, and miStudio samples those vectors from evaluation data. A padding side, a
degenerate standardisation channel, or a rolling window longer than a short row may never appear in
a sample, so the two implementations can diverge in a region parity never visits and the divergence
only shows up as a wrong verdict on live traffic.

This compares the two modules directly, on inputs chosen to contain exactly those regions.

Measured 2026-09-27 on the first run: **0.000e+00 on all six batch rules, all five online rules,
token scores and attention logits.** Not "within tolerance" — identical.

Skipped when miStudio is not checked out beside miLLM, and REQUIRED under
`MILLM_REQUIRE_CROSS_REPO_CHECKS=1` so CI cannot pass by silently skipping.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import pytest
import torch

from millm.ml import probe_head as mil

MISTUDIO_MODEL = Path(
    os.environ.get(
        "MISTUDIO_REPO", "/home/x-sean/app/miStudio"
    )
) / "backend/src/ml/probe_monitor_model.py"

REQUIRED = os.environ.get("MILLM_REQUIRE_CROSS_REPO_CHECKS") == "1"


def _load_mistudio():
    if not MISTUDIO_MODEL.exists():
        if REQUIRED:
            pytest.fail(
                f"MILLM_REQUIRE_CROSS_REPO_CHECKS=1 but miStudio's probe model is not at "
                f"{MISTUDIO_MODEL}. Set MISTUDIO_REPO or check the repo out beside miLLM."
            )
        pytest.skip(f"miStudio not checked out at {MISTUDIO_MODEL}")
    spec = importlib.util.spec_from_file_location("mistudio_probe_model", MISTUDIO_MODEL)
    module = importlib.util.module_from_spec(spec)
    # Dataclass processing reads sys.modules[cls.__module__], so register before executing.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def mis():
    return _load_mistudio()


@pytest.fixture(scope="module")
def fixture():
    """Activations and a head whose regions parity sampling is unlikely to cover."""
    torch.manual_seed(42)
    B, T, D = 3, 29, 8
    acts = torch.randn(B, T, D) * 2.5
    mask = torch.ones(B, T, dtype=torch.bool)
    mask[1, :4] = False    # LEFT padding — the `last` rule's known trap
    mask[2, 20:] = False   # right padding
    std = torch.rand(D) + 0.4
    std[3] = 0.0           # a degenerate channel — zeroed, not clamped
    return {
        "acts": acts, "mask": mask,
        "weight": torch.randn(D), "bias": 0.37,
        "mean": torch.randn(D), "std": std, "query": torch.randn(D),
    }


def _heads(mis, f):
    kw = dict(weight=f["weight"], bias=f["bias"], mean=f["mean"], std=f["std"],
              attention_query=f["query"])
    return mis.ProbeHead(**kw), mil.ProbeHead(**kw)


class TestTheContractsAgree:
    def test_the_rule_lists_are_identical(self, mis):
        assert tuple(mis.RULES) == tuple(mil.RULES)

    def test_the_same_rules_are_streamable(self, mis):
        """If miStudio marks a rule streamable and miLLM does not, a definition built there is
        unservable here — and the mismatch would surface as a refusal nobody can explain."""
        assert set(mis.STREAMABLE) == set(mil.STREAMABLE)

    def test_last_is_non_streamable_in_both(self, mis):
        assert "last" not in mis.STREAMABLE and "last" not in mil.STREAMABLE


class TestTheArithmeticAgrees:
    def test_token_scores_are_identical(self, mis, fixture):
        a, b = _heads(mis, fixture)
        diff = (a.token_scores(fixture["acts"]) - b.token_scores(fixture["acts"])).abs().max()
        assert diff.item() == 0.0, f"token_scores diverge by {diff.item():.3e}"

    def test_attention_logits_are_identical(self, mis, fixture):
        a, b = _heads(mis, fixture)
        diff = (a.attention_logits(fixture["acts"]) - b.attention_logits(fixture["acts"])).abs().max()
        assert diff.item() == 0.0, f"attention_logits diverge by {diff.item():.3e}"

    @pytest.mark.parametrize("rule", list(mil.RULES))
    def test_every_batch_rule_is_identical(self, mis, fixture, rule):
        a, b = _heads(mis, fixture)
        s = a.token_scores(fixture["acts"])
        lg = a.attention_logits(fixture["acts"])
        got_mis = mis.combine(rule, s, mask=fixture["mask"], attention_logits=lg, tau=0.7, window=5)
        got_mil = mil.combine(rule, s, mask=fixture["mask"], attention_logits=lg, tau=0.7, window=5)
        diff = (got_mis - got_mil).abs().max().item()
        assert diff == 0.0, f"{rule} diverges by {diff:.3e}"

    @pytest.mark.parametrize("rule", sorted(mil.STREAMABLE))
    def test_every_online_rule_is_identical(self, mis, fixture, rule):
        a, b = _heads(mis, fixture)
        s = a.token_scores(fixture["acts"])
        lg = a.attention_logits(fixture["acts"])
        row, mask = 1, fixture["mask"]      # row 1 is the LEFT-padded one

        om = mis.OnlineRule(rule, tau=0.7, window=5)
        ol = mil.OnlineRule(rule, tau=0.7, window=5)
        for t in range(s.shape[1]):
            if not mask[row, t]:
                continue
            kw = {"attention_logit": lg[row, t].item()} if rule == "attention" else {}
            om.update(s[row, t].item(), **kw)
            ol.update(s[row, t].item(), **kw)
        assert abs(om.value - ol.value) == 0.0, (
            f"{rule} online diverges by {abs(om.value - ol.value):.3e}"
        )
