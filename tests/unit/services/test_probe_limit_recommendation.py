"""Each model's explained recommendation (039 FR-39.8, D13, D32; TD39-1 – TD39-3; FTASKS 3.2).

From the two measurements of the 2026-10-21 hardware record (production ids 5 and 6, fixture
`tests/fixtures/probe_capacity/measurements_2026-10-21.json`), `recommend` reproduces the D32
starting values — LFM2.5-1.2B: 1 behaviour; Llama-3.1-8B: 2 behaviours, 0.98 / 2.49 ms — and the
reason text names the numbers it used.

⚠ ONE D32 FIGURE DIFFERS AND IS RECORDED, NOT FORCED (039 controls record, discrepancy X-2):
LFM2.5's prefill budget computes to 0.6354 / 0.9 = 0.706 ms, which the FTID's rule (round UP to
0.01 ms, so the budget always admits the sets it was computed for) makes 0.71; D32 wrote 0.70, a
value D32 itself marks provisional "until the S1 deploy computes them".
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from millm.core.config import settings
from millm.services.probe_limit_recommendation import recommend, settings_snapshot

FIXTURE = (Path(__file__).resolve().parents[2] / "fixtures" / "probe_capacity"
           / "measurements_2026-10-21.json")
MEASUREMENTS = {m["repo_id"]: SimpleNamespace(**m)
                for m in json.loads(FIXTURE.read_text())["measurements"]}
LFM = "LiquidAI/LFM2.5-1.2B-Instruct"
LLAMA = "meta-llama/Llama-3.1-8B-Instruct"


def _rec(repo: str, **settings_over):
    snapshot = settings_snapshot(settings)
    snapshot.update(settings_over)
    return recommend(SimpleNamespace(id=7, repo_id=repo), MEASUREMENTS[repo], snapshot)


def test_lfm25_reproduces_its_starting_values():
    rec = _rec(LFM)
    assert rec["values"] == {"max_behaviours": 1, "decode_budget_ms": 0.42,
                             "prefill_budget_ms": 0.71}  # D32 wrote 0.70 (X-2)
    assert rec["achieved_sets"] == 1 and rec["target_sets"] == 2
    text = " ".join(rec["reasons"])
    # Two sets would take 0.8367 ms per pass: 12.5% of the 6.69 ms step, over the 10% ceiling.
    assert "12.5% of this model's 6.69 ms decode step" in text
    assert "ceiling of 10%" in text and "Measurement 5" in text
    assert "GPU-247aa582-0d1b-e161-8156-983ed1fefc57" in text
    assert rec["estimated"] is True and "estimated from the dense line" in text
    assert "prefill step time is not measured" in text


def test_llama_reproduces_its_starting_values():
    rec = _rec(LLAMA)
    assert rec["values"] == {"max_behaviours": 2, "decode_budget_ms": 0.98,
                             "prefill_budget_ms": 2.49}
    assert rec["achieved_sets"] == 2
    assert "Measurement 6" in " ".join(rec["reasons"])


def test_the_digest_covers_values_measurement_and_settings():
    a, b = _rec(LLAMA), _rec(LLAMA)
    assert a["recommendation_digest"] == b["recommendation_digest"], "pure: same inputs, same digest"
    assert _rec(LLAMA, headroom=0.8)["recommendation_digest"] != a["recommendation_digest"]
    assert _rec(LLAMA, max_decode_share=0.2)["recommendation_digest"] != a["recommendation_digest"]


def test_never_below_one_set_and_says_so():
    rec = _rec(LFM, max_decode_share=0.01)
    assert rec["values"]["max_behaviours"] == 1 and rec["achieved_sets"] == 0
    assert any("Even one reference set breaks a ceiling" in r for r in rec["reasons"])


def test_measured_rule_lines_replace_the_estimate():
    lines = {r: dict(MEASUREMENTS[LLAMA].dense) for r in ("mean", "max", "attention")}
    lines["attention"] = {**lines["attention"], "decode_own_ms": 0.1}
    measured = SimpleNamespace(**{**vars(MEASUREMENTS[LLAMA]), "lines": lines})
    rec = recommend(SimpleNamespace(id=7), measured, settings_snapshot(settings))
    assert rec["estimated"] is False
    # Three attention probes now cost 0.1 each instead of 0.0236.
    assert rec["set_cost"]["decode_ms"] == pytest.approx(6 * 0.0236 + 3 * 0.1 + 3 * 0.0757)


def test_no_measurement_no_recommendation():
    assert recommend(SimpleNamespace(id=1), None, settings_snapshot(settings)) is None
