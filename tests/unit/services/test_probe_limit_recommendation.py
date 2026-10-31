"""Each model's explained recommendation (039 FR-39.8; ENH-006 D83 since 2026-10-29).

D83 replaced the count-based recommendation (K reference sets within 10% / 5% share ceilings,
written as absolute ms) with the SLOWDOWN BUDGET: probes may add at most
`PROBE_SLOWDOWN_BUDGET_PCT` (5) per cent of the model's own measured time. The recommendation is
that percentage, plus what fits under it — costed with admission's own functions
(`test_probe_slowdown_budget.py` proves the recommended budget admits what it reports).

From the two measurements of the 2026-10-21 hardware record (production ids 5 and 6, fixture
`tests/fixtures/probe_capacity/measurements_2026-10-21.json`), which measured no prefill step: that
half falls back to the absolute 0.5 ms, and the reasons say how to get 5% instead.
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


def test_lfm25_recommends_five_percent_of_its_own_time():
    rec = _rec(LFM)
    assert rec["values"] == {"max_behaviours": None, "decode_budget_pct": 5.0,
                             "prefill_budget_pct": 5.0, "decode_budget_ms": 0.3345,
                             "prefill_budget_ms": None}
    assert rec["shares"] == {"decode_pct": 5.0, "prefill_pct": None}
    text = " ".join(rec["reasons"])
    assert "5% of the measured 6.69 ms per generated token is 0.3345 ms" in text
    assert "Measurement 5" in text and "GPU-247aa582-0d1b-e161-8156-983ed1fefc57" in text
    assert "prefill step time is not measured" in text and "probe-capacity/measure" in text
    assert rec["estimated"] is True and "upper bound" in text


def test_llama_recommends_five_percent_of_its_own_time():
    rec = _rec(LLAMA)
    assert rec["values"]["decode_budget_ms"] == round(0.05 * 22.68, 6)
    assert "Measurement 6" in " ".join(rec["reasons"])


def test_the_digest_covers_values_measurement_settings_and_the_cost_model():
    a, b = _rec(LLAMA), _rec(LLAMA)
    assert a["recommendation_digest"] == b["recommendation_digest"], "pure: same inputs, same digest"
    assert _rec(LLAMA, headroom=0.8)["recommendation_digest"] != a["recommendation_digest"]
    assert _rec(LLAMA, slowdown_budget_pct=4.0)["recommendation_digest"] != \
        a["recommendation_digest"]
    assert a["cost_model"].startswith("probe_cost.project + probe_arming.over_budget")


def test_measured_rule_lines_replace_the_estimate():
    lines = {r: dict(MEASUREMENTS[LLAMA].dense) for r in ("mean", "max", "attention")}
    lines["attention"] = {**lines["attention"], "decode_own_ms": 0.1}
    measured = SimpleNamespace(**{**vars(MEASUREMENTS[LLAMA]), "lines": lines,
                                  "prefill_ms_per_1k": 200.0})
    rec = recommend(SimpleNamespace(id=7), measured, settings_snapshot(settings))
    assert rec["estimated"] is False and rec["capacity"]["upper_bound"] is False
    # Three attention probes now cost 0.1 each instead of 0.0236: the set's decode cost says so.
    assert rec["set_cost"]["decode_ms"] == pytest.approx(6 * 0.0236 + 3 * 0.1 + 3 * 0.0757)


def test_no_measurement_no_recommendation():
    assert recommend(SimpleNamespace(id=1), None, settings_snapshot(settings)) is None
