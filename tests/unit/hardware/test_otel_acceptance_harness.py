"""The Feature 038 benchmark harness (FTASKS 13.1) on the CPU: its pure parts, and the whole flow
against the fake bench pod (`--dry-run`). The run itself needs an operator-approved GPU window.

What must hold, and why each test exists:

* the profile is D38.52's: 16 cells, 10 warm-up then 60 measured per arm, A-B-B-A in blocks of 10;
* the p95 interval is a real bootstrap (resampling WITH replacement, seeded), and the change
  between arms is B relative to A;
* each FR-38.29 target is judged at its FPRD limit, never a relaxed one, and every unmet or
  unmeasured target becomes a release decision for the operator in the record;
* the harness never loads a model (the load-policy header on every request, a residency check
  before every block) and never runs against a server without the bench routes;
* the checks that read the product's state bite: an arm switch that did not take, an enqueue
  timer that saw nothing, a leaked sentinel, a wrong trace, an unpermitted stateless export.
"""

from __future__ import annotations

import copy
import inspect
import json
import random
import re
import sys
import time
from pathlib import Path

import httpx
import pytest

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "tests" / "hardware"))
import otel_acceptance as h  # noqa: E402
import otel_fake_millm as fake_mod  # noqa: E402

TEMPLATE = json.loads((REPO / "tests" / "fixtures" / "lfm2_probe_definition.json").read_text())
FPRD = REPO / "0xcc" / "prds" / "038_FPRD|Probe_Result_Contract_And_OpenTelemetry_Export.md"
NINE = [f"p-{c}" for c in "abcdefghi"]


# ── the cell grid and the A-B-B-A order ──────────────────────────────────────────────────────


def test_the_grid_is_two_models_by_two_sets_by_two_lengths_by_streaming():
    grid = h.cell_grid(["m-small", "m-large"], ["two", "nine"])
    assert len(grid) == 16
    assert len({c.slug for c in grid}) == 16
    assert {c.prompt_tokens for c in grid} == {200, 4000}
    assert {c.stream for c in grid} == {False, True}
    with pytest.raises(ValueError):
        h.cell_grid(["m"], ["two"], prompts=["medium"])


def test_the_schedule_is_ten_warmup_then_sixty_per_arm_in_abba_blocks_of_ten():
    steps = h.schedule()
    warm = [s for s in steps if s.phase == "warmup"]
    measured = [s for s in steps if s.phase == "measured"]
    assert len(warm) == 10 and [s.arm for s in warm] == ["off"] * 5 + ["on"] * 5
    assert steps[:10] == warm  # warm-up comes first
    assert sum(s.arm == "off" for s in measured) == 60 == sum(s.arm == "on" for s in measured)
    blocks = h.group_blocks(measured)
    assert [len(b) for b in blocks] == [10] * 12
    assert [b[0].arm for b in blocks] == ["off", "on", "on", "off"] * 3
    assert [b[0].block for b in blocks] == list(range(12))
    assert [s.seq for s in steps] == list(range(130))


@pytest.mark.parametrize("per_arm,block", [(60, 10), (50, 10), (20, 10), (30, 5)])
def test_abba_is_balanced_for_any_even_number_of_blocks(per_arm, block):
    measured = [s for s in h.schedule(0, per_arm, block) if s.phase == "measured"]
    assert sum(s.arm == "on" for s in measured) == per_arm == sum(s.arm == "off" for s in measured)
    arms = [b[0].arm for b in h.group_blocks(measured)]
    assert arms[:4] in (["off", "on", "on", "off"], arms[:4])  # pattern starts A B B A
    assert all(arms[i] == ("off" if i % 4 in (0, 3) else "on") for i in range(len(arms)))


def test_the_schedule_refuses_a_shape_it_cannot_balance():
    with pytest.raises(ValueError):
        h.schedule(10, 55, 10)
    with pytest.raises(ValueError):
        h.schedule(10, 60, 0)
    with pytest.raises(ValueError):
        h.schedule(10, 60, 10, arms=("on", "on"))


# ── the bootstrap p95 interval ─────────────────────────────────────────────────────────────────


def test_the_percentile_interval_takes_the_nearest_ranks():
    assert h._ci([float(i) for i in range(2000)], 0.05) == (50.0, 1949.0)
    assert h._ci([float(i) for i in range(100)], 0.10) == (5.0, 94.0)


def test_the_p95_interval_is_seeded_contains_the_point_and_has_width():
    rng = random.Random(1)
    values = [rng.gauss(100, 15) for _ in range(60)]
    point, lo, hi = h.bootstrap_p95(values)
    assert h.bootstrap_p95(values) == (point, lo, hi)  # seeded: the same evidence, same interval
    assert lo <= point <= hi and hi - lo > 1.0
    assert point == h.pctl(values, 95)
    assert h.bootstrap_p95([7.0] * 60) == (7.0, 7.0, 7.0)


def test_the_p95_interval_covers_the_true_p95_most_of_the_time():
    """With replacement, a 95 % percentile interval on n = 60 covers the true p95 of a normal
    distribution in most trials. A resample WITHOUT replacement returns the sample itself every
    time, a zero-width interval that almost never covers it."""
    true_p95 = 100 + 1.6449 * 15
    covered = 0
    for trial in range(60):
        rng = random.Random(1000 + trial)
        values = [rng.gauss(100, 15) for _ in range(60)]
        _, lo, hi = h.bootstrap_p95(values, n_boot=400, seed=trial)
        covered += lo <= true_p95 <= hi
    assert covered >= 42, covered


def test_the_change_is_b_relative_to_a_with_an_interval():
    rng = random.Random(2)
    a = [rng.gauss(100, 5) for _ in range(300)]
    b = [x * 1.10 for x in a]
    point, lo, hi = h.bootstrap_relative_delta(a, b, h.p95)
    assert point == pytest.approx(0.10, abs=1e-9)
    assert 0.0 < lo <= point <= hi
    same_point, same_lo, same_hi = h.bootstrap_relative_delta(a, list(a), h.p95)
    assert same_point == 0.0 and same_lo < 0.0 < same_hi


def test_throughput_is_tokens_over_block_wall_time():
    blocks = [{"tokens": 100, "wall_s": 2.0}, {"tokens": 300, "wall_s": 2.0}]
    assert h.block_throughput(blocks) == 100.0
    point, lo, hi = h.bootstrap_relative_delta(blocks, [{"tokens": 95, "wall_s": 1.0}] * 2,
                                               h.block_throughput)
    assert point == pytest.approx(-0.05)


# ── targets: judged at the FPRD's limits, never relaxed ─────────────────────────────────────────


@pytest.mark.parametrize("key,value,status", [
    ("latency_p95", 0.05, "met"), ("latency_p95", 0.0500001, "unmet"), ("latency_p95", -0.2, "met"),
    ("throughput", -0.05, "met"), ("throughput", -0.0500001, "unmet"), ("throughput", 0.3, "met"),
    ("enqueue_p95", 2.0, "met"), ("enqueue_p95", 2.0001, "unmet"),
    ("arrival_5s", 0.99, "met"), ("arrival_5s", 0.9899, "unmet"), ("arrival_5s", 1.0, "met"),
    ("outage", 0, "met"), ("outage", 1, "unmet"),
    ("latency_p95", None, "not_measured"), ("arrival_5s", float("nan"), "not_measured"),
])
def test_each_target_is_judged_at_its_limit(key, value, status):
    assert h.judge(h.TARGET_BY_KEY[key], value) == status


def test_the_targets_are_the_fprd_table_word_for_word():
    if not FPRD.exists():
        pytest.skip(f"LOUD SKIP: {FPRD.name} is not in this tree (0xcc/ stripped); the target "
                    "texts cannot be pinned here")
    text = FPRD.read_text()
    table = text[text.index("**Provisional targets:**"):text.index("**Rules.**", text.index(
        "**Provisional targets:**"))]
    rows = [tuple(c.strip() for c in line.strip("|").split("|"))
            for line in table.splitlines() if line.startswith("|") and "---" not in line]
    assert rows[0] == ("Measure", "Target")
    assert [(t.measure, t.text) for t in h.TARGETS] == rows[1:]


def test_nothing_can_pass_a_different_limit_to_the_report():
    assert list(inspect.signature(h.render_report).parameters) == ["profile", "cells", "outage", "at"]
    assert list(inspect.signature(h.judge).parameters) == ["target", "value"]


# ── arrival, enqueue ───────────────────────────────────────────────────────────────────────────


def test_arrival_counts_a_missing_record_against_the_target():
    s = 1_000_000_000
    arrivals = {"a": {"finalized_ns": 0, "accepted_ns": 1 * s},
                "b": {"finalized_ns": 0, "accepted_ns": 5 * s},       # exactly 5 s: within
                "c": {"finalized_ns": 0, "accepted_ns": 5 * s + 1}}   # just over
    out = h.arrival_summary(["a", "b", "c", "d", "a"], arrivals)
    assert out["expected"] == 4 and out["received"] == 3 and out["missing"] == 1
    assert out["within"] == 2 and out["fraction_within"] == 0.5
    assert out["max_s"] == pytest.approx(5.000000001)
    assert h.arrival_summary([], {})["fraction_within"] is None


def test_enqueue_summary_is_in_milliseconds():
    out = h.enqueue_summary({"enqueue_log": [float(i) for i in range(1, 101)]})  # µs
    assert out["enqueue_log"]["p95_ms"] == pytest.approx(0.095)
    assert out["enqueue_log"]["n"] == 100


# ── the report: every unmet or unmeasured target is a release decision ────────────────────────


def _cell_doc(off_ms: float, on_ms: float, *, enqueue_us: float = 50.0,
              arrival_s: float = 0.5, arrival: bool = True, kind: str = "telemetry",
              arms: tuple[str, str] = ("off", "on")) -> dict:
    rng = random.Random(7)
    blocks, arrivals = [], {}
    for b in range(12):
        arm = h.abba_arm(b, arms)
        base = off_ms if arm == arms[0] else on_ms
        reqs = []
        for i in range(10):
            rid = f"r{b}-{i}"
            reqs.append({"ok": True, "latency_ms": base * (1 + rng.uniform(-0.01, 0.01)),
                         "completion_tokens": 256, "result_ids": [rid] if arm == arms[1] else []})
            if arm == arms[1]:
                arrivals[rid] = {"finalized_ns": 0, "accepted_ns": int(arrival_s * 1e9)}
        blocks.append({"phase": "measured", "arm": arm, "block": b, "requests": reqs,
                       "wall_s": sum(r["latency_ms"] for r in reqs) / 1000.0,
                       "tokens": 2560, "gpu_used_mib": 9000, "queue_used": {"logs": 1},
                       "counter_series": 12, "enqueue_us": {"enqueue_log": [enqueue_us] * 30}
                       if arm == arms[1] else {}})
    cell = h.Cell("m", "nine", "short", False)
    return {"cell": {"slug": cell.slug}, "kind": kind, "arms": list(arms), "blocks": blocks,
            "arrivals": arrivals, "arrival_measured": arrival}


def test_a_cell_within_every_target_reads_met():
    rows = {r["target"]: r for r in h.evaluate_cell(h.summarise_cell(_cell_doc(100.0, 102.0)))}
    assert {k: r["status"] for k, r in rows.items()} == {
        "latency_p95": "met", "throughput": "met", "enqueue_p95": "met", "arrival_5s": "met"}
    assert rows["latency_p95"]["value"] == pytest.approx(0.02, abs=0.015)


def test_unmet_targets_become_release_decisions_quoting_the_fprd_target():
    slow = _cell_doc(100.0, 112.0, enqueue_us=3000.0, arrival_s=6.0)
    rows = {r["target"]: r["status"] for r in h.evaluate_cell(h.summarise_cell(slow))}
    assert rows == {"latency_p95": "unmet", "throughput": "unmet", "enqueue_p95": "unmet",
                    "arrival_5s": "unmet"}
    text = h.render_report({"date_et": "2026-10-25 10:00:00 EDT"}, [slow],
                           {"completed": True, "failures_attributable": 0}, None)
    decisions = text[text.index("## Release decisions for the operator"):]
    for t in ("≤ +5 %", "≥ −5 %", "p95 ≤ 2 ms", "≥ 99 %"):
        assert f"provisional target **{t}**" in decisions
    assert decisions.count("Release decision RD-") == 4
    assert "never relaxed" in text and "The target is not relaxed" in decisions
    assert "3.000 ms" in decisions and "0.00 %" in decisions  # the measured values are stated


def test_an_unmeasured_target_is_a_decision_not_a_pass():
    no_arrival = _cell_doc(100.0, 101.0, arrival=False)
    text = h.render_report({}, [no_arrival], None, None)
    decisions = text[text.index("## Release decisions for the operator"):]
    assert "*Records received within 5 s of finalization*" in decisions
    assert "*A 10-minute receiver outage* in `outage` was NOT MEASURED" in decisions
    assert "every target was measured and met" not in decisions


def test_with_every_target_met_the_record_says_so():
    text = h.render_report({}, [_cell_doc(100.0, 101.0)],
                           {"completed": True, "failures_attributable": 0, "model": "m"}, None)
    assert text.rstrip().endswith("None: every target was measured and met.")
    assert "| p95 end-to-end latency | ≤ +5 % |" in text


def test_an_outage_with_failures_is_a_release_decision():
    text = h.render_report({}, [_cell_doc(100.0, 101.0)],
                           {"completed": True, "failures_attributable": 3}, None)
    assert "Zero inference failures attributable to export** (FR-38.29)" in text
    assert "measured 3 against" in text


def test_the_pre038_pair_reads_within_or_outside_noise():
    same = _cell_doc(100.0, 100.0, kind="pre038", arms=("pre038", "038_off"), arrival=False)
    text = h.render_report({}, [same], {"completed": True, "failures_attributable": 0}, None)
    assert "**within_noise**" in text and "pre-038 pair" not in text.split("## Release")[1]
    worse = _cell_doc(100.0, 120.0, kind="pre038", arms=("pre038", "038_off"), arrival=False)
    text = h.render_report({}, [worse], {"completed": True, "failures_attributable": 0}, None)
    assert "**outside_noise**" in text and "the pre-038 pair" in text.split("## Release")[1]
    assert h.noise_verdict([None, None, None]) == "not_measured"


# ── server facts: residency, arms, relay, cards ───────────────────────────────────────────────


def test_a_model_that_is_not_resident_is_refused_with_the_approval_to_request():
    models = [{"id": 1, "name": "small"}, {"id": 7, "name": "large"}]
    assert h.residency_refusal({"model_loaded": True, "model_name": "large"}, models, "large") is None
    msg = h.residency_refusal({"model_loaded": True, "model_name": "small"}, models, "large")
    assert "millm_load_model(model_id=7)" in msg and "X-miLLM-Load-Policy: refuse" in msg
    assert "resident: small" in msg and "approves it in miStudio" in msg
    assert "no model row" in h.residency_refusal({"model_loaded": False}, models, "absent")


def test_every_request_carries_the_refuse_load_policy():
    assert h.LOAD_POLICY_HEADERS == {"X-miLLM-Load-Policy": "refuse"}
    client = h.Client("http://127.0.0.1:9")
    assert client.http.headers["X-miLLM-Load-Policy"] == "refuse"


def test_the_arm_must_be_in_force_in_the_products_health_block():
    off = {"enabled": False, "signals": {s: {"state": "disabled"} for s in ("traces", "logs")}}
    on = {"enabled": True, "signals": {"traces": {"state": "healthy"}, "logs": {"state": "degraded"}}}
    assert h.arm_mismatch("off", off) is None and h.arm_mismatch("on", on) is None
    assert h.arm_mismatch("on", off) and h.arm_mismatch("off", on)
    assert h.arm_mismatch("on", {"enabled": True, "signals": {"logs": {"state": "unavailable"}}})
    assert h.arm_mismatch("on", {"enabled": True, "signals": {}})


def test_the_endpoint_must_be_the_relay():
    assert h.relay_mismatch({"config": {"endpoint": "http://127.0.0.1:4319"}},
                            "http://127.0.0.1:4319/") is None
    assert "not the relay" in h.relay_mismatch(
        {"config": {"endpoint": "http://otel-collector.millm-otel.svc:4318"}}, "http://127.0.0.1:4319")


def test_the_card_is_chosen_by_uuid_from_the_visible_list_never_by_index():
    smi = ("GPU 0: NVIDIA GeForce RTX 3090 (UUID: GPU-247aa582-0d1b-e161-8156-983ed1fefc57)\n"
           "GPU 1: NVIDIA GeForce RTX 3080 Ti (UUID: GPU-f47ba814-49a2-603f-3595-275284140251)\n")
    uuids = h.parse_uuids(smi)
    assert h.choose_card(uuids, "GPU-f47ba814-49a2-603f-3595-275284140251") == uuids[1]
    for bad in (None, "", "0", "1", "GPU-00000000-0000-0000-0000-000000000000"):
        with pytest.raises(ValueError):
            h.choose_card(uuids, bad)
    with pytest.raises(ValueError):  # one visible card is still named, not assumed
        h.choose_card(uuids[:1], None)
    assert h.parse_smi_memory("GPU-a, 123\nGPU-b, 9\nnot, 1\n") == {"GPU-a": 123, "GPU-b": 9}


def test_prompts_are_neutral_deterministic_and_rescaled_from_a_measurement():
    p = h.make_prompt(50, "short")
    assert p == h.make_prompt(50, "short") and p != h.make_prompt(50, "long")
    body = p[len(h.INSTRUCTION):]
    words = re.findall(r"[a-z]+", body.lower())
    assert len(words) == 50 and set(words) <= set(h.WORDS)
    assert not re.search(r"\d", p)
    assert h.rescale_words(100, 4000, 2000) == 200 and h.rescale_words(100, 200, 0) == 100


def test_result_ids_are_found_on_every_surface_shape():
    body = {"millm_probe_verdicts": [{"result_id": "a"}, {"result_id": "b", "x": [{"result_id": "c"}]}]}
    assert h.find_result_ids(body) == ["a", "b", "c"]
    assert [e["result_id"] for e in h.verdict_entries(body)] == ["a", "b"]


def test_sentinels_are_found_whole_or_in_part():
    s = h._sentinels(1, seed=3)[0]
    assert h.sentinel_hits(b"nothing here", [s]) == {s: 0}
    assert h.sentinel_hits(f"xx{s}yy".encode(), [s])[s] > 0
    assert h.sentinel_hits(f"..{s[5:14]}..".encode(), [s])[s] > 0
    assert h.sentinel_hits(f"..{s[5:11]}..".encode(), [s])[s] == 0  # under 8 characters


# ── 13.2: the nine probes as ordinary definitions + a two-probe set ───────────────────────────


def _nine() -> list[dict]:
    docs = []
    for layer in (0, 1, 4):
        for rule in ("mean", "max", "attention"):
            d = copy.deepcopy(TEMPLATE)
            d["name"] = f"pmr_x-L{layer}-{rule}"
            d["read"] = {**d["read"], "layer": layer}
            d["aggregation"] = {"rule": rule, "params": {}, "streamable": True}
            if rule == "attention":
                d["head"]["attention_query"] = [0.01] * len(d["head"]["weights"])
            docs.append(d)
    return docs


def test_the_plan_gives_every_probe_one_behaviour_key_and_changes_nothing_else():
    from millm.services.probe_digest import decision_digest, detector_digest

    docs = _nine()
    plan = h.plan_probe_sets(docs, "bench-behaviour", "Bench behaviour")
    assert plan["sets"]["nine"] == [d["name"] for d in docs]
    assert plan["sets"]["two"] == ["pmr_x-L4-attention", "pmr_x-L4-max"]  # deepest layer
    for before, after in zip(docs, plan["documents"], strict=True):
        assert after["behaviour"] == {"key": "bench-behaviour", "name": "Bench behaviour",
                                      "version": None}
        assert {k: v for k, v in after.items() if k != "behaviour"} == before
        assert detector_digest(after) == detector_digest(before)
        assert decision_digest(after["decision"]) == decision_digest(before["decision"])
    assert plan["layers_rules"] == sorted({(lay, r) for lay in (0, 1, 4)
                                           for r in ("mean", "max", "attention")})


def test_the_plan_refuses_a_set_it_cannot_vouch_for():
    docs = _nine()
    with pytest.raises(ValueError):
        h.plan_probe_sets(docs[:8], "k1")
    with pytest.raises(ValueError):
        h.plan_probe_sets(docs, "Not A Slug")
    with pytest.raises(ValueError):
        h.plan_probe_sets(docs, "k1", pair=["pmr_x-L4-max", "nope"])
    mixed = copy.deepcopy(docs)
    mixed[0]["model"] = {**mixed[0]["model"], "hf_id": "other/model"}
    with pytest.raises(ValueError):
        h.plan_probe_sets(mixed, "k1")
    assert h.plan_probe_sets(docs, "k1", pair=["pmr_x-L0-mean", "pmr_x-L1-max"])["sets"]["two"] == [
        "pmr_x-L0-mean", "pmr_x-L1-max"]


@pytest.mark.parametrize("rule", ["mean", "max", "attention"])
async def test_an_adapted_synthetic_probe_passes_the_real_parity_engine(test_session, rule):
    """The second model's probes: same layer and rule, zero head with bias 1.0, so parity is exact
    by construction for every rule the nine use (FTASKS 13.4)."""
    import torch

    from millm.api.schemas.probe import ProbeDefinitionV1
    from millm.db.repositories.probe_repository import ProbeRepository
    from millm.services.probe_arming import armed_probe_from_row
    from millm.services.probe_parity import ProbeParityEngine
    from millm.services.probe_service import ProbeService

    src = next(d for d in _nine() if d["name"] == f"pmr_x-L4-{rule}")
    src["identity"] = {"algorithm": "mistudio.probe-identity/1", "detector_digest": "0" * 64,
                       "decision_digest": "0" * 64}
    doc = h.adapt_synthetic(src, hf_id="org/other-model", d_model=64, n_layers=8,
                            architecture="llama")
    ProbeDefinitionV1.model_validate(doc)
    assert "identity" not in doc and doc["read"]["layer"] == 4
    assert doc["aggregation"]["rule"] == rule and doc["name"].startswith("otelbench-syn-")
    assert all(w == 0.0 for w in doc["head"]["weights"]) and len(doc["head"]["weights"]) == 64
    assert (doc["head"]["attention_query"] == [0.0] * 64) if rule == "attention" else (
        doc["head"]["attention_query"] is None)
    row = await ProbeService(ProbeRepository(test_session)).import_definition(
        doc, raw_bytes=1, on_conflict="fail", origin="file")
    armed = armed_probe_from_row(row, encoder=None, windows=[])
    torch.manual_seed(0)

    def forward(input_ids, context):
        context.observe(4, torch.randn(1, input_ids.shape[1], 64) * 7)

    report = ProbeParityEngine(forward).run(armed, row.definition, tolerance=1e-6)
    assert report.passed, report.as_details()
    with pytest.raises(ValueError):
        h.adapt_synthetic(src, hf_id="o/m", d_model=64, n_layers=4, architecture="llama")


def test_plan_probes_dry_run_writes_the_plan_and_imports_nothing(tmp_path, capsys):
    defs = []
    for d in _nine():
        p = tmp_path / "in" / f"{d['name']}.json"
        p.parent.mkdir(exist_ok=True)
        p.write_text(json.dumps(d))
        defs.append(str(p))
    out = tmp_path / "plan"
    assert h.main(["plan-probes", "--definitions", *defs, "--behaviour-key", "bench-key",
                   "--out", str(out)]) == 0
    sets = json.loads((out / "probe_sets.json").read_text())
    assert len(sets["sets"]["nine"]) == 9 and len(sets["sets"]["two"]) == 2
    assert len(list((out / "definitions").glob("*.json"))) == 9
    assert "DRY RUN" in capsys.readouterr().out
    syn = tmp_path / "syn"
    assert h.main(["plan-probes", "--definitions", *defs, "--behaviour-key", "bench-key",
                   "--adapt", "org/other,64,8,llama", "--out", str(syn)]) == 0
    syn_sets = json.loads((syn / "probe_sets.json").read_text())
    assert syn_sets["synthetic"] and all(n.startswith("otelbench-syn-")
                                         for n in syn_sets["sets"]["nine"])


def test_plan_probes_apply_imports_into_a_bench_pod_only(tmp_path):
    defs = []
    for d in _nine():
        p = tmp_path / f"{d['name']}.json"
        p.write_text(json.dumps(d))
        defs.append(str(p))
    live_like = fake_mod.FakeMillm("m", [], bench=False).start()
    bench = fake_mod.FakeMillm("m", [], bench=True).start()
    try:
        with pytest.raises(SystemExit, match="not a bench pod"):
            h.main(["plan-probes", "--definitions", *defs, "--behaviour-key", "k1", "--apply",
                    "--base", live_like.base, "--out", str(tmp_path / "a")])
        assert live_like.probes == {}
        assert h.main(["plan-probes", "--definitions", *defs, "--behaviour-key", "k1", "--apply",
                       "--base", bench.base, "--out", str(tmp_path / "b")]) == 0
        assert sorted(r["name"] for r in bench.probes.values()) == sorted(d["name"] for d in _nine())
    finally:
        live_like.stop()
        bench.stop()


# ── the relay ──────────────────────────────────────────────────────────────────────────────────


def test_the_relay_forwards_byte_identically_and_times_the_collectors_acceptance():
    coll = fake_mod.FakeCollector().start()
    relay = h.Relay(0, coll.url).start()
    try:
        rid = "01a16ebe-e265-7000-9be5-d1d727185be1"
        body = fake_mod._otlp_logs([{"event_name": "mi.probe.result", "time_ns": time.time_ns(),
                                     "trace_id": "ab" * 16,
                                     "attributes": {"mi.probe.result_id": rid}}])
        r = httpx.post(relay.url + "/v1/logs", content=body,
                       headers={"Content-Type": "application/x-protobuf"})
        assert r.status_code == 200
        assert coll.bodies == [("/v1/logs", body)]
        got = relay.arrivals()[rid]
        assert got["accepted_ns"] >= got["finalized_ns"]
        assert relay.results[rid]["trace_id"] == "ab" * 16
        port = relay.port
        relay.stop()  # the outage: connection refused
        with pytest.raises(httpx.ConnectError):
            httpx.post(f"http://127.0.0.1:{port}/v1/logs", content=body)
        relay.start()
        assert relay.port == port
        coll.stop()  # the Collector itself down: the relay answers 503 (retry), never 200
        late = "01a16ebe-e265-7000-9be5-d1d727185be2"
        body2 = fake_mod._otlp_logs([{"event_name": "mi.probe.result", "time_ns": time.time_ns(),
                                      "attributes": {"mi.probe.result_id": late}}])
        assert httpx.post(relay.url + "/v1/logs", content=body2).status_code == 503
        assert late not in relay.arrivals()  # not accepted by the Collector: not arrived
    finally:
        relay.stop()
        coll.stop()


# ── the whole flow against the fake bench pod (--dry-run) ─────────────────────────────────────


@pytest.fixture
def sets_file(tmp_path):
    p = tmp_path / "probe_sets.json"
    p.write_text(json.dumps({"sets": {"two": NINE[:2], "nine": NINE}}))
    return p


@pytest.fixture
def capture_fake(monkeypatch):
    created: list = []
    real = fake_mod.FakeStack

    def make(*args, **kwargs):
        defects = kwargs.pop("defects", frozenset()) | capture_fake_defects
        stack = real(*args, defects=defects, **kwargs)
        created.append(stack)
        return stack

    capture_fake_defects: set = set()
    monkeypatch.setattr(fake_mod, "FakeStack", make)
    return created, capture_fake_defects


SWEEP = ["sweep", "--dry-run", "--model", "fake-model", "--probe-sets", "two", "--prompts",
         "short", "--warmup", "2", "--measured", "20", "--block", "10", "--max-tokens", "8",
         "--relay-port", "0"]


def test_a_dry_run_sweep_runs_the_whole_flow_and_never_loads_a_model(tmp_path, sets_file,
                                                                     capture_fake):
    created, _ = capture_fake
    out = tmp_path / "ev"
    assert h.main([*SWEEP, "--sets", str(sets_file), "--out", str(out)]) == 0
    stack = created[0]
    assert stack.a.v1_requests > 0 and stack.a.v1_without_policy == 0
    assert stack.a.auto_loaded == []
    cells = sorted(out.glob("cell-telemetry-*.json"))
    assert [p.name for p in cells] == ["cell-telemetry-fake-model-two-short-nostream.json",
                                       "cell-telemetry-fake-model-two-short-stream.json"]
    doc = json.loads(cells[1].read_text())
    measured = [b for b in doc["blocks"] if b["phase"] == "measured"]
    assert [b["arm"] for b in measured] == ["off", "on", "on", "off"]
    assert all(b["gpu_used_mib"] == 2048 for b in doc["blocks"])
    s = h.summarise_cell(doc)
    assert s["arrival"]["expected"] == 2 * 3 * 20 and s["arrival"]["missing"] == 0
    assert s["enqueue"]["enqueue_log"]["n"] > 0
    assert all(b["queue_used"] == {"traces": 0, "metrics": 0, "logs": 0}
               for b in measured if b["arm"] == "off")
    profile = json.loads((out / "profile.json").read_text())
    assert profile["probe_sets"] == {"fake-model:two": NINE[:2]}
    assert profile["card_uuid"] == fake_mod.FAKE_CARD
    report = tmp_path / "otel_benchmark.md"
    assert h.main(["report", "--out", str(out), "--write", str(report)]) == 0
    text = report.read_text()
    assert "fake-model-two-short-stream" in text and "## Release decisions for the operator" in text


def test_a_model_unloaded_mid_run_stops_the_harness_before_the_next_block(tmp_path, sets_file,
                                                                          capture_fake):
    created, defects = capture_fake
    defects.add("unload_after_12")
    with pytest.raises(SystemExit, match="is not resident"):
        h.main([*SWEEP, "--sets", str(sets_file), "--out", str(tmp_path / "ev")])
    stack = created[0]
    assert stack.a.auto_loaded == []
    assert stack.a.v1_requests <= 13  # the calibration request + the 12 before the unload


def test_a_server_without_the_bench_routes_is_refused(tmp_path, sets_file, capture_fake):
    created, defects = capture_fake
    defects.add("no_bench")
    with pytest.raises(SystemExit, match="not a bench pod"):
        h.main([*SWEEP, "--sets", str(sets_file), "--out", str(tmp_path / "ev")])
    assert created[0].a.v1_requests == 0


def test_an_arm_switch_that_did_not_take_is_refused(tmp_path, sets_file, capture_fake):
    _, defects = capture_fake
    defects.add("arm_ignored")
    with pytest.raises(RuntimeError, match="arm switch not in force"):
        h.main([*SWEEP, "--sets", str(sets_file), "--out", str(tmp_path / "ev")])


def test_an_arm_that_did_not_take_on_a_probe_is_refused(tmp_path, sets_file, capture_fake):
    created, defects = capture_fake
    defects.add("arm_noop")
    with pytest.raises(SystemExit, match="is not the requested"):
        h.main([*SWEEP, "--sets", str(sets_file), "--out", str(tmp_path / "ev")])
    assert created[0].a.v1_requests == 0


def test_an_enqueue_timer_that_saw_nothing_is_refused(tmp_path, sets_file, capture_fake):
    _, defects = capture_fake
    defects.add("no_enqueue")
    with pytest.raises(SystemExit, match="enqueue timer saw no call"):
        h.main([*SWEEP, "--sets", str(sets_file), "--out", str(tmp_path / "ev")])


def test_the_pre038_pair_runs_against_two_servers(tmp_path, sets_file, capture_fake):
    created, _ = capture_fake
    out = tmp_path / "ev"
    assert h.main(["sweep", "--dry-run", "--arm-switch", "servers", "--model", "fake-model",
                   "--sets", str(sets_file), "--probe-sets", "two", "--prompts", "short",
                   "--streams", "nostream", "--warmup", "2", "--measured", "20", "--block", "10",
                   "--max-tokens", "8", "--no-relay", "--out", str(out)]) == 0
    stack = created[0]
    assert stack.a.v1_requests > 0 and stack.b.v1_requests > 0
    assert stack.a.v1_without_policy == 0 == stack.b.v1_without_policy
    doc = json.loads((out / "cell-pre038-fake-model-two-short-nostream.json").read_text())
    assert doc["arms"] == ["pre038", "038_off"] and doc["arm_switch"] == "servers"
    assert "Off means free" in h.render_report({}, [doc], None, None)


def test_a_dry_run_outage_counts_failures_and_reads_the_counters(tmp_path, sets_file, capture_fake):
    created, _ = capture_fake
    out = tmp_path / "ev"
    assert h.main(["outage", "--dry-run", "--model", "fake-model", "--sets", str(sets_file),
                   "--probe-set", "nine", "--before-s", "0.3", "--outage-s", "0.6",
                   "--after-s", "0.3", "--drain-timeout-s", "5", "--max-tokens", "4",
                   "--relay-port", "0", "--out", str(out)]) == 0
    doc = json.loads((out / "outage.json").read_text())
    assert doc["completed"] and doc["valid"] and doc["failures_attributable"] == 0
    assert set(doc["failures_by_phase"]) == {"before", "outage", "after"}
    failed_after = sum(v["failed"] for v in doc["telemetry"]["after"]["signals"].values())
    assert failed_after > 0  # the exports during the outage failed, and were counted
    assert h.evaluate_outage(doc)["status"] == "met"


AT = ["at", "--dry-run", "--model", "fake-model", "--probe-set", "two", "--relay-port", "0",
      "--settle-s", "0.3", "--template", str(REPO / "tests" / "fixtures" /
                                              "lfm2_probe_definition.json")]


def test_the_at_rerun_passes_against_a_correct_server(tmp_path, sets_file, capture_fake):
    out = tmp_path / "ev"
    assert h.main([*AT, "--sets", str(sets_file), "--out", str(out)]) == 0
    checks = json.loads((out / "at.json").read_text())["checks"]
    assert {k: v["result"] for k, v in checks.items()} == {
        "AT02": "pass", "AT04": "pass", "AT07": "pass", "AT08": "pass", "AT10": "pass",
        "AT12": "pass", "AT03": "manual", "AT05": "manual", "AT06": "manual"}


@pytest.mark.parametrize("defect,check", [("wrong_trace", "AT02"), ("leak", "AT10"),
                                          ("stateless_leak", "AT07")])
def test_each_at_check_bites_on_its_defect(tmp_path, sets_file, capture_fake, defect, check):
    _, defects = capture_fake
    defects.add(defect)
    out = tmp_path / "ev"
    assert h.main([*AT, "--sets", str(sets_file), "--out", str(out)]) == 1
    checks = json.loads((out / "at.json").read_text())["checks"]
    assert checks[check]["result"] == "fail"
    if defect == "wrong_trace":  # each AT02 case is judged on its own, not only the conjunction
        facts = checks["AT02"]["facts"]
        assert facts["known"] is False and facts["interleaved"] is False


# ── review R1 (2026-10-25): checks that no test exercised ──────────────────────────────────────
# Each test below was written for a mutation that survived the suite; the mutation table and its
# negative controls are in `0xcc/reviews/038_13_1_review_R1_2026-10-25.md`.


def _with_failed(doc: dict, arm: str, latency_ms: float) -> dict:
    """The cell with one failed request per block of `arm`, far outside the served latencies."""
    doc = copy.deepcopy(doc)
    for b in doc["blocks"]:
        if b["arm"] == arm:
            b["requests"].append({"ok": False, "latency_ms": latency_ms, "completion_tokens": 0,
                                  "result_ids": []})
    return doc


def test_failed_requests_are_counted_and_kept_out_of_the_latency():
    doc = _with_failed(_cell_doc(100.0, 101.0), "on", 1.0)
    s = h.summarise_cell(doc)
    on_ok = [r["latency_ms"] for b in doc["blocks"] if b["arm"] == "on" for r in b["requests"]
             if r["ok"]]
    assert s["per_arm"]["on"]["failed"] == 6 and s["per_arm"]["off"]["failed"] == 0
    assert s["per_arm"]["on"]["p50_ms"] == h.median(on_ok)
    assert s["per_arm"]["on"]["p95_ms"] == h.p95(on_ok)


def test_a_stream_ending_in_an_error_event_or_without_done_is_a_failed_request():
    for defect, code in (("stream_error", "streaming_error"),
                         ("stream_truncated", "stream_ended_without_done")):
        server = fake_mod.FakeMillm("m", [], defects=frozenset({defect})).start()
        try:
            r = h.Client(server.base).chat("m", "hello there", True, max_tokens=8)
        finally:
            server.stop()
        assert r["status"] == 200 and r["ok"] is False and r["error"] == code, (defect, r)
    good = fake_mod.FakeMillm("m", []).start()
    try:
        r = h.Client(good.base).chat("m", "hello there", True, max_tokens=8)
    finally:
        good.stop()
    assert r["ok"] is True and r["completion_tokens"] == 8 and "error" not in r


def test_a_dry_run_outage_that_breaks_inference_is_unmet(tmp_path, sets_file, capture_fake):
    _, defects = capture_fake
    defects.add("outage_fails")
    out = tmp_path / "ev"
    assert h.main(["outage", "--dry-run", "--model", "fake-model", "--sets", str(sets_file),
                   "--probe-set", "nine", "--before-s", "0.3", "--outage-s", "0.6",
                   "--after-s", "0.3", "--drain-timeout-s", "5", "--max-tokens", "4",
                   "--relay-port", "0", "--out", str(out)]) == 0
    doc = json.loads((out / "outage.json").read_text())
    assert doc["valid"] and doc["failures_by_phase"]["before"] == 0
    assert doc["failures_attributable"] == (doc["failures_by_phase"]["outage"]
                                            + doc["failures_by_phase"]["after"]) > 0
    assert h.evaluate_outage(doc)["status"] == "unmet"
    text = h.render_report({}, [], doc, None)
    assert "Release decision RD-1" in text and "*A 10-minute receiver outage* in `outage` measured" in text


def test_an_outage_run_with_a_broken_baseline_is_invalid_never_met(tmp_path, sets_file,
                                                                   capture_fake):
    created, defects = capture_fake
    defects.add("chat_fails")
    out = tmp_path / "ev"
    assert h.main(["outage", "--dry-run", "--model", "fake-model", "--sets", str(sets_file),
                   "--probe-set", "nine", "--before-s", "0.2", "--outage-s", "0.2",
                   "--after-s", "0.2", "--drain-timeout-s", "2", "--max-tokens", "4",
                   "--relay-port", "0", "--out", str(out)]) == 0
    doc = json.loads((out / "outage.json").read_text())
    assert doc["failures_by_phase"]["before"] > 0 and doc["valid"] is False
    # Even with no failure in the outage or after it, an invalid run is not judged met.
    doc["failures_attributable"] = 0
    row = h.evaluate_outage(doc)
    assert row["status"] == "not_measured" and row["valid"] is False
    text = h.render_report({}, [], doc, None)
    assert "run invalid" in text and "was NOT MEASURED" in text
    assert created[0].a.preset == "off"  # the outage's `finally` switched the arm back off


def test_the_outage_and_at_runs_leave_telemetry_off(tmp_path, sets_file, capture_fake):
    created, _ = capture_fake
    assert h.main(["outage", "--dry-run", "--model", "fake-model", "--sets", str(sets_file),
                   "--probe-set", "two", "--before-s", "0.1", "--outage-s", "0.1",
                   "--after-s", "0.1", "--drain-timeout-s", "2", "--max-tokens", "4",
                   "--relay-port", "0", "--out", str(tmp_path / "o")]) == 0
    assert created[-1].a.preset == "off"
    assert h.main([*AT, "--sets", str(sets_file), "--out", str(tmp_path / "a")]) == 0
    assert created[-1].a.preset == "off"  # AT12 ends on "on": only the `finally` turns it off


def test_a_sweep_stopped_in_an_on_block_switches_telemetry_back_off(tmp_path, sets_file,
                                                                    capture_fake):
    """Calibration (1) + warm-up off (1) + warm-up on (1): the model goes after request 3, so the
    residency check before the first measured block stops the run with the arm ON."""
    created, defects = capture_fake
    defects.add("unload_after_3")
    with pytest.raises(SystemExit, match="is not resident"):
        h.main([*SWEEP, "--sets", str(sets_file), "--out", str(tmp_path / "ev")])
    assert created[0].a.preset == "off"


def test_raised_limits_are_restored_after_the_sweep(tmp_path, sets_file, capture_fake):
    created, _ = capture_fake
    assert h.main([*SWEEP, "--streams", "nostream", "--raise-limits", "--sets", str(sets_file),
                   "--out", str(tmp_path / "ev")]) == 0
    a = created[0].a
    assert a.env_overrides == {}
    assert [c[:2] for c in a.env_changes] == [("PUT", k) for k in h.RAISED] + [
        ("DELETE", k) for k in h.RAISED]


@pytest.mark.parametrize("defect,match", [
    ("mcr_4", "MAX_CONCURRENT_REQUESTS is 4"),
    ("endpoint_elsewhere", "not the relay"),
    ("wrong_model", "responses named another model"),
])
def test_the_sweep_refuses_a_server_off_the_profile(tmp_path, sets_file, capture_fake, defect,
                                                     match):
    created, defects = capture_fake
    defects.add(defect)
    with pytest.raises(SystemExit, match=match):
        h.main([*SWEEP, "--sets", str(sets_file), "--out", str(tmp_path / "ev")])
    if defect != "wrong_model":  # refused in the preflight, before any /v1 request
        assert created[0].a.v1_requests == 0


def test_arming_a_set_disarms_every_other_probe(capture_fake):
    server = fake_mod.FakeMillm("m", NINE).start()
    try:
        client = h.Client(server.base)
        stray = next(r for r in server.probes.values() if r["name"] == "p-i")
        stray["armed"] = True
        ids = h.ensure_armed(client, NINE[:2])
        assert stray["armed"] is False
        assert sorted(r["id"] for r in server.probes.values() if r["armed"]) == ids
    finally:
        server.stop()


@pytest.mark.parametrize("defect,check", [
    ("disconnect_completed", "AT04"), ("sampler_ignored", "AT08"), ("metric_ids", "AT08"),
    ("refusal_wrong_code", "AT12"), ("drop_sentinel_requests", "AT10"),
    ("outcome_mismatch", "AT04"), ("stateless_partial", "AT07"),
    ("leak_stream", "AT10"), ("leak_score", "AT10"), ("auth_leak", "AT10"),
])
def test_each_review_at_check_bites_on_its_defect(tmp_path, sets_file, capture_fake, defect,
                                                  check):
    _, defects = capture_fake
    defects.add(defect)
    out = tmp_path / "ev"
    assert h.main([*AT, "--sets", str(sets_file), "--out", str(out)]) == 1
    checks = json.loads((out / "at.json").read_text())["checks"]
    assert checks[check]["result"] == "fail", checks[check]
    others = {k for k, v in checks.items() if v["result"] == "fail"}
    assert others == {check}, others  # the defect fails its own check, and only that one


def test_a_corrupt_gzip_body_is_still_forwarded_and_answered():
    coll = fake_mod.FakeCollector().start()
    relay = h.Relay(0, coll.url).start()
    try:
        r = httpx.post(relay.url + "/v1/logs", content=b"not gzip at all",
                       headers={"Content-Encoding": "gzip"})
        assert r.status_code == 200  # the Collector's answer reaches miLLM: no retry, no duplicate
        assert coll.bodies == [("/v1/logs", b"not gzip at all")]
        assert relay.requests_by_path.get("undecodable") == 1
    finally:
        relay.stop()
        coll.stop()


def test_at10_fails_when_the_score_route_exported_nothing(tmp_path, sets_file, capture_fake):
    """Without a record from the score route, its sentinels were never put at risk: a clean
    capture there proves nothing (review R1: the score half of AT10 had passed vacuously)."""
    _, defects = capture_fake
    defects.add("score_never_exports")
    out = tmp_path / "ev"
    assert h.main([*AT, "--sets", str(sets_file), "--out", str(out)]) == 1
    at10 = json.loads((out / "at.json").read_text())["checks"]["AT10"]
    assert at10["result"] == "fail" and at10["facts"]["score_records_exported"] == 0
    assert at10["facts"]["hits"] == 0 and at10["facts"]["bytes_searched"] > 0  # else clean
