"""miStudio's decision-precedence cases, run through miLLM's own bar selection (039 FTASKS 4.9).

`docs/schemas/decision-precedence-cases.json` is miStudio's (feature 040 S2,
`services/probe_monitor_metrics.deployed_decision`), vendored byte-identically: each case states a
probe's global bar, its length bands, its scope, the window judged, that window's own entry and the
scored-token count, and the bar that must judge it. Every case is run through
`ProbeRequestContext._verdict_for` — the one place this runtime chooses a bar — on a real context
whose window selects exactly `n_scored` tokens, and the verdict's `threshold` must equal the case's.
miStudio's offline evaluation and miLLM's live verdict therefore cannot disagree about which bar a
result was judged against.

Vendored from miStudio 040 S1–S2 as REVIEWED (2026-10-24, sha256 `7517a5dd…`, 29 cases). The byte pin runs against `$MISTUDIO_REPO` (or the sibling checkout) and is required
under `MILLM_REQUIRE_CROSS_REPO_CHECKS=1`; it skips loudly otherwise.

Control: flip one byte of the vendored file → the byte pin goes red.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
import torch

from millm.ml.probe_head import ProbeHead
from millm.services.probe_arming import bands_from_block
from millm.services.probe_runtime import ArmedProbe, ProbeRequestContext
from tests.support.sibling_repos import mistudio_repo

VENDORED = Path(__file__).resolve().parents[3] / "docs" / "schemas" / "decision-precedence-cases.json"
REQUIRED = os.environ.get("MILLM_REQUIRE_CROSS_REPO_CHECKS") == "1"
CASES = json.loads(VENDORED.read_text())["cases"]
D = 4
#: Tokens around the judged window, so a window never selects the whole request by accident.
EXTRA = 3


def _probe(case: dict) -> ArmedProbe:
    p, entry = case["probe"], case["window_entry"]
    window = case["window"]
    window_thresholds, window_bands = {}, {}
    if window is not None and entry is not None:
        if entry.get("threshold") is not None:
            window_thresholds[window] = float(entry["threshold"])
            bands = bands_from_block(entry.get("length_bands"))
            if bands:
                window_bands[window] = bands
    return ArmedProbe(
        probe_id="pr_case", name="case", rule="mean", scope=p["scope"], layer=0, rung=2,
        rung_language="detects on unseen tasks",
        head=ProbeHead(weight=torch.ones(D), bias=0.0, mean=torch.zeros(D), std=torch.ones(D),
                       layer=0),
        threshold=p.get("threshold"), windows=(window or p["scope"],),
        window_thresholds=window_thresholds,
        length_bands=bands_from_block(p.get("length_bands")),
        window_length_bands=window_bands,
    )


def _verdict(case: dict):
    probe = _probe(case)
    window = case["window"] or probe.scope
    n = int(case["n_scored"])
    context = ProbeRequestContext("case", [probe])
    if window == "prompt":
        total, prompt = n + EXTRA, n
    elif window == "response":
        total, prompt = n + EXTRA, EXTRA
    else:  # all: the whole request is the window
        total, prompt = n, max(n - 1, 0)
    context.set_prompt_length(prompt)
    context.observe(0, torch.ones(1, total, D))
    return next(v for v in context.finish() if v.window == window)


#: X-1 CLOSED (miStudio 040 review, 2026-10-24): the two cases miLLM disagreed with were wrong in
#: the case file. It now states miLLM's whole-or-nothing band-table rule (`bands_from_block`: a
#: null-threshold band or a closed last band drops the whole table) and adds three
#: unusable-table cases. Every case passes through the real parser path; none is expected to fail.


@pytest.mark.parametrize("case", CASES, ids=[c["name"] for c in CASES])
def test_every_case_is_judged_against_the_case_bar(case):
    verdict = _verdict(case)
    assert verdict.scored, verdict.not_scored_reason
    assert verdict.n_scored_tokens == case["n_scored"]
    expected = case["expect"]["threshold"]
    if expected is None:
        assert verdict.threshold is None and verdict.fires is None, "no bar: does not fire"
    else:
        assert verdict.threshold == pytest.approx(expected), (case["name"], case["expect"]["rule"])


def test_the_vendored_file_is_byte_identical_to_mistudios():
    source = mistudio_repo() / "docs" / "schemas" / "decision-precedence-cases.json"
    if not source.exists():
        message = f"{source} is not checked out — the cross-repo byte pin did NOT run"
        if REQUIRED:
            pytest.fail(message + " and MILLM_REQUIRE_CROSS_REPO_CHECKS=1")
        pytest.skip(message)
    assert VENDORED.read_bytes() == source.read_bytes(), (
        "the vendored decision-precedence cases differ from miStudio's; re-copy them, never edit "
        "the copy"
    )
