#!/usr/bin/env python3
"""Feature 038 hardware harness (FTASKS 13.1) — telemetry's cost and timeliness, and the AT re-run.

Run INSIDE the throwaway bench pod in an operator-approved GPU window (13.3), never by the unit
suite. The pod runs the deployed image with the branch under test on PYTHONPATH (as Features
31–33 and 37 did) and starts miLLM through `otel_bench_server.py`, which adds the in-process
telemetry arm switch (its docstring records why: every `TELEMETRY_*` setting is environment-only,
and a restart per block of 10 would reload the model 13 times per cell). FR-38.29, FR-38.30,
D38.52; FTDD §9. All times written are US Eastern (ET).

SUBCOMMANDS
  cards        the cards by UUID, from `nvidia-smi -L` (never an index; the 1 + n GPUs rule)
  plan-probes  13.2: the nine probes of a miStudio run as ordinary definitions (one behaviour key
               each) + a two-probe set; DRY RUN by default, `--apply` imports into a BENCH pod only
  sweep        13.4: the cells of the RESIDENT model (2 models × 2 probe sets × ~200/~4,000-token
               prompts × streaming or not = 16), 10 warm-up then 60 measured per arm, off/on A-B-B-A
               in blocks of 10. `--arm-switch servers` runs the one pre-038 pair (two servers)
  outage       13.5: a 10-minute receiver outage under traffic; counters before and after
  at           12.5's remaining halves on the reference deployment: AT02, AT04, AT07, AT08 (result
               halves), AT10 (score route), AT12 (model mismatch). AT03, AT05, AT06 are listed as
               manual (they need crafted probes, engine paths or a recalibration)
  report       13.6: the record (Markdown) from the evidence files; every unmet target is written
               as a release decision for the operator, never relaxed

GATES THIS HARNESS NEVER BYPASSES
* It never loads, unloads or swaps a model. Every request carries `X-miLLM-Load-Policy: refuse`
  (Feature 29: a `/v1` request naming a model that is not resident is refused instead of loading
  it), and before every block it checks that the cell's model is the resident one. When it is not,
  it prints the exact approval to request (miStudio MCP `millm_load_model`, approved in miStudio's
  banner, on the live server) and exits.
* `sweep`, `outage`, `at` and `plan-probes --apply` refuse a server without the bench routes, so
  they cannot run against the live miLLM (which has no bench route) — and the live pod's telemetry
  stays off (12.5, 13.3: no deploy). Since ENH-006 D75 the live miLLM can switch export from its
  Settings page; this harness never does that: it switches only through the bench routes.
* Settings changes (`--raise-limits`, for the nine-probe set above `PROBE_MAX_ARMED`) go through
  the product's own `PUT /api/environment/{key}` on the bench pod and are restored in a `finally`.

ARRIVAL DELAY AT THE REFERENCE COLLECTOR. The bench pod's `TELEMETRY_ENDPOINT` points at a relay
this harness runs on loopback (`--relay-port`, default 4319). The relay timestamps each OTLP
request, forwards the body BYTE-IDENTICALLY to the reference Collector (`--collector`), returns
the Collector's status to miLLM, and records each `mi.probe.result` record's finalization time
(the record's own timestamp) and the moment the Collector ACCEPTED it (its 2xx). Delay = accepted
minus finalized; the target is ≥ 99 % within 5 s. The relay is also the outage switch (it closes
its socket: connection refused), so 13.5 needs no change in `millm-otel`; `--outage-mode operator`
instead prints the scale-down/up commands for the operator and waits.

GPU WINDOW SEQUENCE (13.3 – 13.6), each step's approval in brackets:
  0. [operator: GPU window + the bench pod beside the live one] check no GPU work is in flight.
  1. cards; plan-probes (dry run, then --apply on the bench pod)            [window approval]
  2. load LFM2.5 on the bench pod                                          [window approval]
     sweep --model <LFM name> --sets probe_sets.json --card <UUID>
  3. load Llama-3.1-8B on the bench pod; sweep again (its synthetic sets)  [window approval]
     outage --model <8B name> --probe-set nine; at ...
  4. the pre-038 pair: a second bench pod on the pre-038 image, LFM resident on both
     sweep --arm-switch servers --base-b <pre-038 URL> ...                 [window approval]
  5. report --write 0xcc/reviews/otel_benchmark_<date>.md
`--dry-run` runs any of sweep/outage/at against an in-process fake miLLM and fake Collector
(`otel_fake_millm.py`): the whole flow, no GPU — the CPU tests run it in CI.
"""

from __future__ import annotations

import argparse
import copy
import gzip
import json
import math
import random
import re
import string
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from hw_common import now_et, pctl  # noqa: E402
from probe_scaling_acceptance import parse_uuids  # noqa: E402

# ── the declared profile (FPRD FR-38.29; FTDD §9; D38.52) ──────────────────────────────────────

#: Every request the harness sends carries this: a `/v1` request never loads a model (Feature 29).
LOAD_POLICY_HEADERS = {"X-miLLM-Load-Policy": "refuse"}
PROMPT_TOKENS = {"short": 200, "long": 4000}
OUTPUT_TOKENS = 256
#: "concurrency 1 plus a queued backlog of 8": the server serves one request at a time
#: (MAX_CONCURRENT_REQUESTS stays 1), the client keeps 1 + 8 in flight.
BACKLOG = 8
WARMUP = 10
MEASURED_PER_ARM = 60
BLOCK = 10
ARRIVAL_WINDOW_S = 5.0
BOOTSTRAP_N = 2000
BOOTSTRAP_SEED = 38
BENCH_PREFIX = "/__otel_bench"
DEFAULT_RELAY_PORT = 4319
#: Settings raised on the BENCH pod for a probe set above the count backstop, restored after.
RAISED = {"PROBE_MAX_ARMED": 16, "PROBE_MAX_OVERHEAD_MS_PER_PASS": 10.0,
          "PROBE_MAX_PREFILL_MS_PER_1K": 10.0}


@dataclass(frozen=True)
class Target:
    """One provisional target of FR-38.29, exactly as the FPRD's table states it.

    ⚠ NEVER RELAXED. There is no parameter anywhere that changes a limit; the report renders this
    tuple, and `tests/unit/hardware/test_otel_acceptance_harness.py` pins `text` to the FPRD row.
    """

    key: str
    measure: str
    text: str
    op: str          # "le" | "ge"
    limit: float
    unit: str


TARGETS: tuple[Target, ...] = (
    Target("latency_p95", "p95 end-to-end latency", "≤ +5 %", "le", 0.05, "relative change"),
    Target("throughput", "Generation throughput", "≥ −5 %", "ge", -0.05, "relative change"),
    Target("enqueue_p95", "Export enqueue", "p95 ≤ 2 ms", "le", 2.0, "ms"),
    Target("arrival_5s", "Records received within 5 s of finalization", "≥ 99 %", "ge", 0.99,
           "fraction"),
    Target("outage", "A 10-minute receiver outage",
           "Zero inference failures attributable to export", "le", 0.0, "failures"),
)
TARGET_BY_KEY = {t.key: t for t in TARGETS}


@dataclass(frozen=True)
class Cell:
    model: str
    probe_set: str
    prompt: str        # short | long
    stream: bool

    @property
    def prompt_tokens(self) -> int:
        return PROMPT_TOKENS[self.prompt]

    @property
    def slug(self) -> str:
        model = re.sub(r"[^A-Za-z0-9._-]+", "_", self.model)
        return f"{model}-{self.probe_set}-{self.prompt}-{'stream' if self.stream else 'nostream'}"


@dataclass(frozen=True)
class Step:
    phase: str         # warmup | measured
    arm: str
    block: int         # -1 for warm-up
    seq: int


# ── pure parts (CPU-tested in tests/unit/hardware/test_otel_acceptance_harness.py) ────────────


def cell_grid(models: Sequence[str], probe_sets: Sequence[str],
              prompts: Sequence[str] = ("short", "long"),
              streams: Sequence[bool] = (False, True)) -> list[Cell]:
    """2 models × 2 probe sets × 2 prompt lengths × streaming or not = 16 cells."""
    for p in prompts:
        if p not in PROMPT_TOKENS:
            raise ValueError(f"unknown prompt length {p!r}; known: {sorted(PROMPT_TOKENS)}")
    return [Cell(m, s, p, st) for m in models for s in probe_sets for p in prompts for st in streams]


def abba_arm(block_index: int, arms: Sequence[str]) -> str:
    """A-B-B-A: blocks 0 and 3 of every four are arm A, blocks 1 and 2 arm B."""
    return arms[0] if block_index % 4 in (0, 3) else arms[1]


def schedule(warmup: int = WARMUP, measured_per_arm: int = MEASURED_PER_ARM, block: int = BLOCK,
             arms: Sequence[str] = ("off", "on")) -> list[Step]:
    """Warm-up (first half arm A, second half arm B, so both arms' paths are warm), then
    `2 × measured_per_arm / block` measured blocks in A-B-B-A order."""
    if len(arms) != 2 or arms[0] == arms[1]:
        raise ValueError("two distinct arms")
    if block <= 0 or measured_per_arm <= 0 or measured_per_arm % block:
        raise ValueError("measured_per_arm must be a positive multiple of block")
    if warmup < 0:
        raise ValueError("warmup must be >= 0")
    steps: list[Step] = []
    seq = 0
    for i in range(warmup):
        steps.append(Step("warmup", arms[0] if i < warmup // 2 else arms[1], -1, seq))
        seq += 1
    for b in range(2 * measured_per_arm // block):
        arm = abba_arm(b, arms)
        for _ in range(block):
            steps.append(Step("measured", arm, b, seq))
            seq += 1
    return steps


def group_blocks(steps: Sequence[Step]) -> list[list[Step]]:
    """Contiguous runs of one (phase, block, arm): the unit the runner executes and switches on."""
    out: list[list[Step]] = []
    for s in steps:
        if out and (out[-1][0].phase, out[-1][0].block, out[-1][0].arm) == (s.phase, s.block, s.arm):
            out[-1].append(s)
        else:
            out.append([s])
    return out


def p95(values: Sequence[float]) -> float:
    return pctl(list(values), 95)


def median(values: Sequence[float]) -> float:
    return pctl(list(values), 50)


def _ci(replicates: list[float], alpha: float) -> tuple[float, float]:
    """The percentile interval: the replicates at ranks alpha/2 and 1 − alpha/2 (nearest rank)."""
    reps = sorted(replicates)
    n = len(reps)
    lo = reps[min(n - 1, max(0, math.floor(round(alpha / 2 * n, 9))))]
    hi = reps[min(n - 1, max(0, math.ceil(round((1 - alpha / 2) * n, 9)) - 1))]
    return lo, hi


def bootstrap_ci(items: Sequence[Any], stat: Callable[[Sequence[Any]], float],
                 n_boot: int = BOOTSTRAP_N, seed: int = BOOTSTRAP_SEED,
                 alpha: float = 0.05) -> tuple[float, float, float]:
    """(point, lo, hi): the statistic and its percentile bootstrap interval, resampling WITH
    replacement, seeded (a report re-rendered from the same evidence gives the same interval)."""
    items = list(items)
    if not items:
        return float("nan"), float("nan"), float("nan")
    rng = random.Random(seed)
    reps = [stat(rng.choices(items, k=len(items))) for _ in range(n_boot)]
    lo, hi = _ci(reps, alpha)
    return stat(items), lo, hi


def bootstrap_p95(values: Sequence[float], **kw: Any) -> tuple[float, float, float]:
    return bootstrap_ci(values, p95, **kw)


def bootstrap_relative_delta(a: Sequence[Any], b: Sequence[Any],
                             stat: Callable[[Sequence[Any]], float],
                             n_boot: int = BOOTSTRAP_N, seed: int = BOOTSTRAP_SEED,
                             alpha: float = 0.05) -> tuple[float, float, float]:
    """stat(b) / stat(a) − 1 (B relative to A), with an interval from resampling A and B
    independently in each replicate."""
    a, b = list(a), list(b)
    if not a or not b:
        return float("nan"), float("nan"), float("nan")
    rng = random.Random(seed)
    reps = []
    for _ in range(n_boot):
        sa = stat(rng.choices(a, k=len(a)))
        sb = stat(rng.choices(b, k=len(b)))
        reps.append(sb / sa - 1.0 if sa else float("nan"))
    lo, hi = _ci(reps, alpha)
    point_a = stat(a)
    return (stat(b) / point_a - 1.0 if point_a else float("nan")), lo, hi


def block_throughput(blocks: Sequence[dict[str, Any]]) -> float:
    """Generation throughput of an arm: completion tokens over block wall time (the server is
    saturated by the backlog, so this is the server's rate, not a client's)."""
    wall = sum(float(b["wall_s"]) for b in blocks)
    return sum(int(b["tokens"]) for b in blocks) / wall if wall > 0 else float("nan")


def arrival_summary(expected_ids: Iterable[str], arrivals: dict[str, dict[str, int]],
                    within_s: float = ARRIVAL_WINDOW_S) -> dict[str, Any]:
    """Records accepted by the Collector within `within_s` of their finalization. A record that
    never arrived counts AGAINST the target (it was not received within 5 s)."""
    expected = sorted(set(expected_ids))
    delays = [(arrivals[r]["accepted_ns"] - arrivals[r]["finalized_ns"]) / 1e9
              for r in expected if r in arrivals]
    within = sum(1 for d in delays if d <= within_s)
    return {
        "expected": len(expected), "received": len(delays), "within": within,
        "missing": len(expected) - len(delays),
        "fraction_within": (within / len(expected)) if expected else None,
        "p50_s": pctl(delays, 50) if delays else None,
        "p95_s": pctl(delays, 95) if delays else None,
        "max_s": max(delays) if delays else None,
    }


def enqueue_summary(samples_us: dict[str, Sequence[float]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for kind, vals in samples_us.items():
        vals = list(vals)
        out[kind] = {"n": len(vals), "p50_ms": pctl(vals, 50) / 1000.0 if vals else None,
                     "p95_ms": pctl(vals, 95) / 1000.0 if vals else None,
                     "max_ms": max(vals) / 1000.0 if vals else None}
    return out


def judge(target: Target, value: float | None) -> str:
    """`met`, `unmet` or `not_measured`. Nothing else, and no tolerance beyond the limit."""
    if value is None or value != value:  # None or NaN
        return "not_measured"
    if target.op == "le":
        return "met" if value <= target.limit else "unmet"
    if target.op == "ge":
        return "met" if value >= target.limit else "unmet"
    raise ValueError(target.op)


# -- prompts (synthetic, neutral text; no identifier-shaped value ever) ---------------------------

WORDS = (
    "river", "garden", "morning", "quiet", "window", "lantern", "harbor", "meadow", "pencil",
    "orchard", "valley", "station", "kitchen", "library", "bridge", "canvas", "thunder", "ladder",
    "planet", "market", "pocket", "forest", "candle", "basket", "mirror", "island", "village",
    "summer", "autumn", "winter", "spring", "teacher", "painter", "farmer", "sailor", "doctor",
    "walks", "builds", "carries", "watches", "follows", "gathers", "paints", "reads", "writes",
    "slowly", "gently", "often", "always", "under", "beside", "across", "toward", "through",
    "bright", "simple", "careful", "patient", "golden", "silver", "narrow", "distant", "early",
)
TOKENS_PER_WORD = 1.35
INSTRUCTION = "Continue the following passage in the same plain style, at length.\n\n"


def make_prompt(n_words: int, salt: str = "") -> str:
    rng = random.Random(f"otel-bench-{salt}")
    words = [rng.choice(WORDS) for _ in range(max(1, n_words))]
    sentences = [" ".join(words[i:i + 12]).capitalize() + "." for i in range(0, len(words), 12)]
    return INSTRUCTION + " ".join(sentences)


def initial_words(target_tokens: int) -> int:
    return max(1, round(target_tokens / TOKENS_PER_WORD))


def rescale_words(words: int, target_tokens: int, measured_tokens: int) -> int:
    """The word count that would have hit the target, from one measured prompt length."""
    if measured_tokens <= 0:
        return words
    return max(1, round(words * target_tokens / measured_tokens))


# -- server facts -------------------------------------------------------------------------------


def residency_refusal(health: dict[str, Any], models: Sequence[dict[str, Any]],
                      expected: str) -> str | None:
    """None when `expected` is the resident model; otherwise the refusal, naming the approval."""
    if health.get("model_loaded") and health.get("model_name") == expected:
        return None
    resident = health.get("model_name") if health.get("model_loaded") else None
    row = next((m for m in models if m.get("name") == expected), None)
    if row is None:
        return (f"REFUSED: no model row named {expected!r} on this server (resident: "
                f"{resident or 'none'}). Nothing was loaded.")
    mid = row.get("id")
    return (
        f"REFUSED: {expected!r} (model id {mid}) is not resident (resident: {resident or 'none'}). "
        "This harness never loads a model, and every request it sends carries "
        "`X-miLLM-Load-Policy: refuse`, so nothing was loaded.\n"
        f"  Live miLLM: request miStudio MCP `millm_load_model(model_id={mid})`; the operator "
        "approves it in miStudio's approvals banner.\n"
        f"  Bench pod (its own database, no MCP gate in front): the operator's GPU-window approval "
        f"(13.3) must cover this load; then `POST /api/models/{mid}/load` on the bench pod.\n"
        "Re-run once it is resident."
    )


def env_value(listing: dict[str, Any], key: str) -> Any:
    for row in listing.get("settings", []):
        if row.get("key") == key:
            return row.get("value")
    return None


def arm_mismatch(preset: str, telemetry: dict[str, Any]) -> str | None:
    """The product's own health block must agree with the arm the harness just set."""
    signals = telemetry.get("signals") or {}
    states = {k: (v or {}).get("state") for k, v in signals.items()}
    if preset == "off":
        if telemetry.get("enabled") or any(s != "disabled" for s in states.values()):
            return f"arm off but telemetry reads enabled={telemetry.get('enabled')} {states}"
        return None
    if not telemetry.get("enabled") or not states or any(
            s in (None, "disabled", "unavailable") for s in states.values()):
        return f"arm {preset} but telemetry reads enabled={telemetry.get('enabled')} {states}"
    return None


def relay_mismatch(telemetry: dict[str, Any], relay_url: str) -> str | None:
    endpoint = ((telemetry.get("config") or {}).get("endpoint") or "").rstrip("/")
    if endpoint != relay_url.rstrip("/"):
        return (f"TELEMETRY_ENDPOINT is {endpoint or 'unset'!r}, not the relay {relay_url!r}: "
                "arrival cannot be measured. Set the bench pod's TELEMETRY_ENDPOINT to the relay "
                "(the relay forwards to the reference Collector) or pass --no-relay.")
    return None


def choose_card(uuids: Sequence[str], requested: str | None) -> str:
    """The card named by UUID, checked against `nvidia-smi -L` at run time. Never an index."""
    if not requested:
        raise ValueError(f"pass --card <UUID>; visible: {', '.join(uuids) or 'none'}")
    if requested not in uuids:
        raise ValueError(f"card {requested} is not visible; visible: {', '.join(uuids) or 'none'}")
    return requested


def parse_smi_memory(csv: str) -> dict[str, int]:
    """`nvidia-smi --query-gpu=uuid,memory.used --format=csv,noheader,nounits` → {uuid: MiB}."""
    out: dict[str, int] = {}
    for line in csv.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) == 2 and parts[0].startswith("GPU-"):
            try:
                out[parts[0]] = int(parts[1])
            except ValueError:
                continue
    return out


def find_result_ids(obj: Any) -> list[str]:
    """Every `result_id` anywhere in a response (body field, terminal chunk, score route)."""
    found: list[str] = []
    if isinstance(obj, dict):
        rid = obj.get("result_id")
        if isinstance(rid, str):
            found.append(rid)
        for v in obj.values():
            if isinstance(v, (dict, list)):
                found.extend(find_result_ids(v))
    elif isinstance(obj, list):
        for v in obj:
            found.extend(find_result_ids(v))
    return found


def verdict_entries(obj: Any) -> list[dict[str, Any]]:
    return [e for e in (obj or {}).get("millm_probe_verdicts") or [] if isinstance(e, dict)]


# -- summaries, targets, report -------------------------------------------------------------------


def summarise_cell(cell_doc: dict[str, Any]) -> dict[str, Any]:
    """Per arm and between arms, from the raw evidence of one cell (one source for the report)."""
    arms = cell_doc["arms"]
    a_name, b_name = arms
    measured = [b for b in cell_doc["blocks"] if b["phase"] == "measured"]
    per: dict[str, Any] = {}
    lat: dict[str, list[float]] = {}
    blocks_by: dict[str, list[dict[str, Any]]] = {}
    for arm in arms:
        blocks = [b for b in measured if b["arm"] == arm]
        reqs = [r for b in blocks for r in b["requests"]]
        ok = [r for r in reqs if r.get("ok")]
        lat[arm] = [r["latency_ms"] for r in ok]
        blocks_by[arm] = blocks
        point, lo, hi = bootstrap_p95(lat[arm]) if lat[arm] else (None, None, None)
        decode = [r["decode_tps"] for r in ok if r.get("decode_tps")]
        gpu = [b.get("gpu_used_mib") for b in blocks if b.get("gpu_used_mib") is not None]
        queues = [q for b in blocks for q in (b.get("queue_used") or {}).values() if q is not None]
        series = [b.get("counter_series") for b in blocks if b.get("counter_series") is not None]
        rss = [b.get("rss_kib") for b in blocks if b.get("rss_kib") is not None]
        per[arm] = {
            "requests": len(reqs), "failed": len(reqs) - len(ok),
            "p50_ms": median(lat[arm]) if lat[arm] else None,
            "p95_ms": point, "p95_ci_ms": [lo, hi],
            "throughput_tps": block_throughput(blocks) if blocks else None,
            "decode_tps_median": median(decode) if decode else None,
            "gpu_used_mib_max": max(gpu) if gpu else None,
            "queue_used_max": max(queues) if queues else None,
            "counter_series_max": max(series) if series else None,
            "rss_kib_max": max(rss) if rss else None,
            "prompt_tokens": sorted({r.get("prompt_tokens") for r in ok if r.get("prompt_tokens")}),
        }
    lat_delta = (bootstrap_relative_delta(lat[a_name], lat[b_name], p95)
                 if lat[a_name] and lat[b_name] else (None, None, None))
    tp_delta = (bootstrap_relative_delta(blocks_by[a_name], blocks_by[b_name], block_throughput)
                if blocks_by[a_name] and blocks_by[b_name] else (None, None, None))
    enq_samples: dict[str, list[float]] = {}
    for b in measured:
        if b["arm"] == b_name:
            for kind, vals in (b.get("enqueue_us") or {}).items():
                enq_samples.setdefault(kind, []).extend(vals)
    enqueue = enqueue_summary(enq_samples)
    on_ids = [rid for b in measured if b["arm"] == b_name for r in b["requests"]
              for rid in r.get("result_ids", [])]
    arrival = (arrival_summary(on_ids, cell_doc.get("arrivals") or {})
               if cell_doc.get("arrival_measured") else None)
    return {
        "cell": cell_doc["cell"], "arms": arms, "kind": cell_doc.get("kind", "telemetry"),
        "per_arm": per,
        "latency_p95_delta": list(lat_delta), "throughput_delta": list(tp_delta),
        "enqueue": enqueue, "arrival": arrival,
    }


def evaluate_cell(summary: dict[str, Any]) -> list[dict[str, Any]]:
    """The four per-cell targets for a telemetry off/on cell."""
    enq = (summary.get("enqueue") or {}).get("enqueue_log") or {}
    arrival = summary.get("arrival") or {}
    values = {
        "latency_p95": (summary["latency_p95_delta"][0], summary["latency_p95_delta"][1:]),
        "throughput": (summary["throughput_delta"][0], summary["throughput_delta"][1:]),
        "enqueue_p95": (enq.get("p95_ms") if enq.get("n") else None, None),
        "arrival_5s": (arrival.get("fraction_within"), None),
    }
    out = []
    for key, (value, ci) in values.items():
        t = TARGET_BY_KEY[key]
        out.append({"target": key, "measure": t.measure, "text": t.text, "value": value,
                    "ci": ci, "status": judge(t, value)})
    return out


def evaluate_outage(outage: dict[str, Any]) -> dict[str, Any]:
    """The outage target. A run whose BEFORE phase already failed requests is invalid (review R1,
    2026-10-25): its later failures cannot be told apart from a broken baseline, so it is
    `not_measured` (a release decision), never `met`."""
    t = TARGET_BY_KEY["outage"]
    valid = outage.get("valid") is not False
    value = (outage.get("failures_attributable")
             if outage.get("completed") and valid else None)
    return {"target": t.key, "measure": t.measure, "text": t.text, "value": value, "ci": None,
            "status": judge(t, value), "valid": valid}


def noise_verdict(delta: Sequence[float | None]) -> str:
    """The pre-038 pair: within noise when the interval of the change contains zero."""
    point, lo, hi = (list(delta) + [None, None, None])[:3]
    if point is None or lo is None or hi is None or point != point:
        return "not_measured"
    return "within_noise" if lo <= 0.0 <= hi else "outside_noise"


def _fmt(v: Any, pct: bool = False, digits: int = 1) -> str:
    if v is None or (isinstance(v, float) and v != v):
        return "—"
    if pct:
        return f"{v * 100:+.{digits}f} %"
    if isinstance(v, float):
        return f"{v:.{digits}f}"
    return str(v)


def _mib(kib: Any) -> str:
    return "—" if kib is None else f"{kib / 1024:.0f}"


def _fmt_value(row: dict[str, Any]) -> str:
    t = TARGET_BY_KEY[row["target"]]
    v = row["value"]
    if t.unit == "relative change":
        ci = row.get("ci") or [None, None]
        return f"{_fmt(v, pct=True)} [{_fmt(ci[0], pct=True)}, {_fmt(ci[1], pct=True)}]"
    if t.unit == "fraction":
        return "—" if v is None else f"{v * 100:.2f} %"
    if t.unit == "ms":
        return "—" if v is None else f"{v:.3f} ms"
    return _fmt(v)


def release_decision(n: int, where: str, row: dict[str, Any]) -> str:
    """One unmet (or unmeasured) target as a decision for the operator. The target is quoted as
    the FPRD states it; nothing here can move it."""
    t = TARGET_BY_KEY[row["target"]]
    state = ("was NOT MEASURED" if row["status"] == "not_measured"
             else f"measured {_fmt_value(row)}")
    return (
        f"- **Release decision RD-{n} (operator):** *{t.measure}* in `{where}` {state} against the "
        f"provisional target **{t.text}** (FR-38.29). The target is not relaxed. The switch-on "
        "(14.0) stays blocked until you decide: (a) accept this result and switch on, recording "
        "the cost; (b) keep export off and fix first; or (c) re-measure (state why)."
    )


def render_report(profile: dict[str, Any], cells: Sequence[dict[str, Any]],
                  outage: dict[str, Any] | None = None,
                  at: dict[str, Any] | None = None) -> str:
    """The 13.6 record. Every unmet or unmeasured target becomes a release decision."""
    lines: list[str] = []
    date = profile.get("date_et") or now_et()
    lines.append(f"# OpenTelemetry benchmark — {date} (Feature 038, FR-38.29)")
    lines.append("")
    lines.append("All times US Eastern (ET). Generated by `tests/hardware/otel_acceptance.py report` "
                 "from the evidence files; the targets are the FPRD's, never relaxed.")
    lines.append("")
    lines.append("## Profile")
    lines.append("")
    for key in ("card_uuid", "card_name", "image", "code", "max_concurrent_requests", "backlog",
                "output_tokens", "warmup", "measured_per_arm", "block", "relay", "collector",
                "dry_run"):
        if key in profile:
            lines.append(f"- `{key}`: {profile[key]}")
    for key, val in sorted((profile.get("probe_sets") or {}).items()):
        lines.append(f"- probe set `{key}`: {len(val)} probes ({', '.join(val)})")
    lines.append("")
    lines.append("## Targets (FPRD FR-38.29, provisional)")
    lines.append("")
    lines.append("| Measure | Target |")
    lines.append("|---|---|")
    for t in TARGETS:
        lines.append(f"| {t.measure} | {t.text} |")
    lines.append("")

    decisions: list[str] = []
    telemetry_cells = [c for c in cells if c.get("kind", "telemetry") == "telemetry"]
    pair_cells = [c for c in cells if c.get("kind") == "pre038"]
    if telemetry_cells:
        lines.append("## Matched pairs: telemetry off (A) vs on (B), probes on in both")
        lines.append("")
        lines.append("| Cell | p50 A / B ms | p95 A [95 % CI] ms | p95 B [95 % CI] ms | p95 change "
                     "[CI] | throughput A / B tok/s | change [CI] | enqueue p95 | within 5 s | "
                     "arrival p95 | GPU MiB A / B | RSS MiB A / B | queue max | series max | failed A / B |")
        lines.append("|" + "---|" * 15)
        for doc in telemetry_cells:
            s = summarise_cell(doc)
            a, b = s["arms"]
            pa, pb = s["per_arm"][a], s["per_arm"][b]
            rows = {r["target"]: r for r in evaluate_cell(s)}
            arrival = s.get("arrival") or {}
            lines.append(
                f"| `{s['cell']['slug']}` | {_fmt(pa['p50_ms'])} / {_fmt(pb['p50_ms'])} | "
                f"{_fmt(pa['p95_ms'])} [{_fmt(pa['p95_ci_ms'][0])}, {_fmt(pa['p95_ci_ms'][1])}] | "
                f"{_fmt(pb['p95_ms'])} [{_fmt(pb['p95_ci_ms'][0])}, {_fmt(pb['p95_ci_ms'][1])}] | "
                f"{_fmt_value(rows['latency_p95'])} | "
                f"{_fmt(pa['throughput_tps'])} / {_fmt(pb['throughput_tps'])} | "
                f"{_fmt_value(rows['throughput'])} | {_fmt_value(rows['enqueue_p95'])} | "
                f"{_fmt_value(rows['arrival_5s'])} | {_fmt(arrival.get('p95_s'), digits=3)} s | "
                f"{_fmt(pa['gpu_used_mib_max'])} / {_fmt(pb['gpu_used_mib_max'])} | "
                f"{_mib(pa['rss_kib_max'])} / {_mib(pb['rss_kib_max'])} | "
                f"{_fmt(pb['queue_used_max'])} | {_fmt(pb['counter_series_max'])} | "
                f"{pa['failed']} / {pb['failed']} |")
        lines.append("")
        lines.append("### Each target, per cell")
        lines.append("")
        lines.append("| Cell | Measure | Target | Measured | Status |")
        lines.append("|---|---|---|---|---|")
        for doc in telemetry_cells:
            s = summarise_cell(doc)
            for row in evaluate_cell(s):
                lines.append(f"| `{s['cell']['slug']}` | {row['measure']} | {row['text']} | "
                             f"{_fmt_value(row)} | **{row['status']}** |")
                if row["status"] != "met":
                    decisions.append(release_decision(len(decisions) + 1, s["cell"]["slug"], row))
        lines.append("")
    if pair_cells:
        lines.append("## Off means free: pre-038 image (A) vs this image, telemetry off (B), probes on")
        lines.append("")
        lines.append("| Cell | p95 A / B ms | p95 change [CI] | throughput change [CI] | verdict |")
        lines.append("|---|---|---|---|---|")
        for doc in pair_cells:
            s = summarise_cell(doc)
            a, b = s["arms"]
            ld, td = s["latency_p95_delta"], s["throughput_delta"]
            verdict = noise_verdict(ld)
            lines.append(f"| `{s['cell']['slug']}` | {_fmt(s['per_arm'][a]['p95_ms'])} / "
                         f"{_fmt(s['per_arm'][b]['p95_ms'])} | {_fmt(ld[0], pct=True)} "
                         f"[{_fmt(ld[1], pct=True)}, {_fmt(ld[2], pct=True)}] | "
                         f"{_fmt(td[0], pct=True)} [{_fmt(td[1], pct=True)}, {_fmt(td[2], pct=True)}]"
                         f" | **{verdict}** |")
            if verdict != "within_noise":
                n = len(decisions) + 1
                decisions.append(
                    f"- **Release decision RD-{n} (operator):** the pre-038 pair `{s['cell']['slug']}`"
                    f" is **{verdict}** (p95 change {_fmt(ld[0], pct=True)}, interval "
                    f"[{_fmt(ld[1], pct=True)}, {_fmt(ld[2], pct=True)}]): FPRD §8 says the S1 "
                    "builder adds no measurable cost with telemetry off. Decide before 14.0.")
        lines.append("")
    lines.append("## Receiver outage (13.5)")
    lines.append("")
    if outage:
        row = evaluate_outage(outage)
        lines.append(f"- model `{outage.get('model')}`, probe set `{outage.get('probe_set')}`, mode "
                     f"`{outage.get('mode')}`, outage {outage.get('outage_s')} s; requests "
                     f"{outage.get('requests')} — failed by phase {outage.get('failures_by_phase')}")
        for phase in ("before", "after"):
            snap = (outage.get("telemetry") or {}).get(phase) or {}
            lines.append(f"- counters {phase}: " + json.dumps(
                {k: {f: (v or {}).get(f) for f in ("state", "exported", "failed", "dropped")}
                 for k, v in (snap.get("signals") or {}).items()}, sort_keys=True))
        if not row["valid"]:
            lines.append(f"- **run invalid:** {outage.get('failures_by_phase', {}).get('before')} "
                         "request(s) failed BEFORE the outage; the target cannot be judged")
        lines.append(f"- target *{row['measure']}* — **{row['text']}**: measured "
                     f"{_fmt(row['value'])} → **{row['status']}**")
        if row["status"] != "met":
            decisions.append(release_decision(len(decisions) + 1, "outage", row))
    else:
        row = evaluate_outage({})
        lines.append("- not run")
        decisions.append(release_decision(len(decisions) + 1, "outage", row))
    lines.append("")
    if at:
        lines.append("## AT re-run on the reference deployment (12.5's remaining halves)")
        lines.append("")
        lines.append("| Test | Result | Facts |")
        lines.append("|---|---|---|")
        for name, res in sorted(at.get("checks", {}).items()):
            facts = json.dumps(res.get("facts", {}), sort_keys=True)[:300]
            lines.append(f"| {name} | **{res.get('result')}** | `{facts}` |")
        lines.append("")
    lines.append("## Release decisions for the operator")
    lines.append("")
    if decisions:
        lines.extend(decisions)
    else:
        lines.append("None: every target was measured and met.")
    lines.append("")
    return "\n".join(lines)


# ── 13.2: probe-set planning (pure; `--apply` imports into a BENCH pod only) ───────────────────


def definition_layer_rule(doc: dict[str, Any]) -> tuple[int, str]:
    return int((doc.get("read") or {}).get("layer", -1)), str((doc.get("aggregation") or {}).get("rule"))


def with_behaviour(doc: dict[str, Any], key: str, name: str | None = None) -> dict[str, Any]:
    """The definition as an ORDINARY document with one behaviour key. `behaviour` is outside the
    detector and decision projections (`probe_digest.DETECTOR_V1`), so a carried `identity` stays
    valid; nothing else changes."""
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,62}[a-z0-9]", key):
        raise ValueError(f"behaviour key {key!r} is not a valid slug")
    out = copy.deepcopy(doc)
    out["behaviour"] = {"key": key, "name": name or key, "version": None}
    return out


def adapt_synthetic(doc: dict[str, Any], *, hf_id: str, d_model: int, n_layers: int,
                    architecture: str, prefix: str = "otelbench-syn-") -> dict[str, Any]:
    """The definition re-pointed at ANOTHER model as a synthetic probe (FTASKS 13.4's second
    model: the miStudio run trained on one model only). Same layer, rule and window, a head of
    zeros with bias 1.0 — so every position scores exactly 1.0 and parity is exact by
    construction (probe_scaling_acceptance's method) — while the arithmetic per pass is a real
    probe's. The identity block is dropped (its digests named the other model's detector)."""
    out = copy.deepcopy(doc)
    out["name"] = prefix + str(doc.get("name", "probe"))
    out["model"] = {"hf_id": hf_id, "revision": "main", "d_model": int(d_model),
                    "n_layers": int(n_layers), "architecture": architecture}
    layer = int((out.get("read") or {}).get("layer", 0))
    if layer >= int(n_layers):
        raise ValueError(f"layer {layer} does not exist on a {n_layers}-layer model")
    out.pop("identity", None)
    out.pop("sae", None)
    out.pop("render", None)
    head = out["head"]
    head["weights"] = [0.0] * int(d_model)
    head["bias"] = 1.0
    head["norm_mean"] = [0.0] * int(d_model)
    head["norm_std"] = [1.0] * int(d_model)
    if (out.get("aggregation") or {}).get("rule") == "attention":
        head["attention_query"] = [0.0] * int(d_model)
    else:
        head["attention_query"] = None
    out["decision"] = {**out.get("decision", {}), "threshold": 2.0}
    vectors = []
    for v in (out.get("test_vectors") or {}).get("vectors", [])[:2]:
        ids = list(v["token_ids"])
        vectors.append({**v, "token_scores": [1.0] * len(ids), "score": 1.0, "verdict": False})
    out["test_vectors"] = {"tolerance": 1e-6, "vectors": vectors}
    out.setdefault("provenance", {})["harness"] = (
        "otel_acceptance: synthetic zero head (bias 1.0) re-pointed at another model")
    return out


def default_pair(docs: Sequence[dict[str, Any]]) -> list[str]:
    """The two-probe set unless `--pair` names one: the deepest layer's probes, by rule name."""
    deepest = max(definition_layer_rule(d)[0] for d in docs)
    at_layer = sorted((d for d in docs if definition_layer_rule(d)[0] == deepest),
                      key=lambda d: definition_layer_rule(d)[1])
    pool = at_layer if len(at_layer) >= 2 else sorted(docs, key=lambda d: str(d.get("name")))
    return [str(d["name"]) for d in pool[:2]]


def plan_probe_sets(docs: Sequence[dict[str, Any]], behaviour_key: str,
                    behaviour_name: str | None = None, pair: Sequence[str] | None = None,
                    expected: int = 9, strip_identity: bool = False) -> dict[str, Any]:
    """13.2: the nine documents (one behaviour key each) and the two-probe set, as a plan."""
    if len(docs) != expected:
        raise ValueError(f"expected {expected} definitions, got {len(docs)}")
    names = [str(d.get("name")) for d in docs]
    if len(set(names)) != len(names):
        raise ValueError("definition names must be distinct")
    models = {(d.get("model") or {}).get("hf_id") for d in docs}
    if len(models) != 1:
        raise ValueError(f"the set spans models {sorted(map(str, models))}")
    shaped = []
    for d in docs:
        doc = with_behaviour(d, behaviour_key, behaviour_name)
        if strip_identity:
            doc.pop("identity", None)
        shaped.append(doc)
    two = list(pair) if pair else default_pair(docs)
    missing = [n for n in two if n not in names]
    if len(two) != 2 or missing:
        raise ValueError(f"the two-probe set must name two of the nine; unknown: {missing}")
    return {
        "model": next(iter(models)), "behaviour": behaviour_key,
        "sets": {"nine": names, "two": two},
        "layers_rules": sorted({definition_layer_rule(d) for d in docs}),
        "documents": shaped,
    }


# ── the relay (in front of the reference Collector) ──────────────────────────────────────────


def _any_value(v: Any) -> Any:
    kind = v.WhichOneof("value")
    return getattr(v, kind) if kind else None


def decode_logs(body: bytes) -> list[dict[str, Any]]:
    from opentelemetry.proto.collector.logs.v1.logs_service_pb2 import ExportLogsServiceRequest

    out = []
    req = ExportLogsServiceRequest.FromString(body)
    for rl in req.resource_logs:
        for sl in rl.scope_logs:
            for rec in sl.log_records:
                out.append({"event_name": rec.event_name, "time_unix_nano": rec.time_unix_nano,
                            "trace_id": rec.trace_id.hex(), "span_id": rec.span_id.hex(),
                            "attributes": {kv.key: _any_value(kv.value) for kv in rec.attributes}})
    return out


def decode_span_count(body: bytes) -> int:
    from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import ExportTraceServiceRequest

    req = ExportTraceServiceRequest.FromString(body)
    return sum(len(ss.spans) for rs in req.resource_spans for ss in rs.scope_spans)


def decode_metric_points(body: bytes) -> list[dict[str, Any]]:
    from opentelemetry.proto.collector.metrics.v1.metrics_service_pb2 import (
        ExportMetricsServiceRequest,
    )

    out = []
    req = ExportMetricsServiceRequest.FromString(body)
    for rm in req.resource_metrics:
        for sm in rm.scope_metrics:
            for metric in sm.metrics:
                kind = metric.WhichOneof("data")
                for point in getattr(metric, kind).data_points:
                    out.append({"name": metric.name,
                                "attributes": {kv.key: _any_value(kv.value)
                                               for kv in point.attributes}})
    return out


class Relay:
    """OTLP/HTTP pass-through on loopback: timestamps, forwards byte-identically, decodes."""

    def __init__(self, port: int, upstream: str | None, capture_raw: bool = False) -> None:
        self.port = port
        self.upstream = upstream.rstrip("/") if upstream else None
        self.capture_raw = capture_raw
        self.lock = threading.Lock()
        self.results: dict[str, dict[str, Any]] = {}
        self.lifecycle: list[dict[str, Any]] = []
        self.span_count = 0
        self.metric_points: list[dict[str, Any]] = []
        self.requests_by_path: dict[str, int] = {}
        self.upstream_status: dict[str, int] = {}
        self.raw: list[bytes] = []
        self.bytes_seen = 0
        self._server: ThreadingHTTPServer | None = None
        self._client: Any = None

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def start(self) -> Relay:
        import httpx

        relay = self
        if self._client is None:
            self._client = httpx.Client(timeout=10.0, trust_env=False)

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args: Any) -> None:  # noqa: ARG002 - quiet
                return None

            def do_POST(self) -> None:  # noqa: N802
                rx_ns = time.time_ns()
                body = self.rfile.read(int(self.headers.get("Content-Length") or 0))
                status, payload = relay._forward(self.path, body, dict(self.headers.items()))
                accepted_ns = time.time_ns()
                relay._observe(self.path, body, self.headers.get("Content-Encoding"),
                               rx_ns, accepted_ns, status)
                self.send_response(status)
                self.send_header("Content-Type", "application/x-protobuf")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

        server = ThreadingHTTPServer(("127.0.0.1", self.port), Handler)
        server.daemon_threads = True
        self.port = server.server_address[1]
        self._server = server
        threading.Thread(target=server.serve_forever, daemon=True).start()
        return self

    def stop(self) -> None:
        """Close the socket: the receiver is gone (connection refused). The port is kept."""
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._server = None

    @property
    def running(self) -> bool:
        return self._server is not None

    def _forward(self, path: str, body: bytes, headers: dict[str, str]) -> tuple[int, bytes]:
        if not self.upstream:
            return 200, b""
        keep = {k: v for k, v in headers.items()
                if k.lower() in ("content-type", "content-encoding", "authorization")}
        try:
            r = self._client.post(self.upstream + path, content=body, headers=keep)
            return r.status_code, r.content
        except Exception:  # noqa: BLE001 - the Collector is down: tell miLLM to retry
            return 503, b""

    def _observe(self, path: str, body: bytes, encoding: str | None, rx_ns: int,
                 accepted_ns: int, status: int) -> None:
        ok = 200 <= status < 300
        try:
            data = gzip.decompress(body) if (encoding or "").lower() == "gzip" else body
        except (OSError, EOFError):  # a corrupt body: still forwarded and answered, counted
            data = b""
            with self.lock:
                self.requests_by_path["undecodable"] = self.requests_by_path.get("undecodable", 0) + 1
        with self.lock:
            self.requests_by_path[path] = self.requests_by_path.get(path, 0) + 1
            self.upstream_status[str(status)] = self.upstream_status.get(str(status), 0) + 1
            self.bytes_seen += len(data)
            if self.capture_raw:
                self.raw.append(data)
        try:
            if path.endswith("/v1/logs"):
                for rec in decode_logs(data):
                    a = rec["attributes"]
                    with self.lock:
                        if rec["event_name"] == "mi.probe.result" and ok:
                            rid = a.get("mi.probe.result_id")
                            if rid and rid not in self.results:
                                self.results[rid] = {
                                    "finalized_ns": rec["time_unix_nano"], "rx_ns": rx_ns,
                                    "accepted_ns": accepted_ns, "trace_id": rec["trace_id"],
                                    "attributes": a}
                        elif rec["event_name"] == "mi.probe.lifecycle" and ok:
                            self.lifecycle.append({"rx_ns": rx_ns, "attributes": a})
            elif path.endswith("/v1/traces") and ok:
                n = decode_span_count(data)
                with self.lock:
                    self.span_count += n
            elif path.endswith("/v1/metrics") and ok:
                points = decode_metric_points(data)
                with self.lock:
                    self.metric_points.extend(points)
        except Exception:  # noqa: BLE001 - an undecodable body is still forwarded and counted
            with self.lock:
                self.requests_by_path["undecodable"] = self.requests_by_path.get("undecodable", 0) + 1

    def arrivals(self) -> dict[str, dict[str, int]]:
        with self.lock:
            return {k: {"finalized_ns": v["finalized_ns"], "accepted_ns": v["accepted_ns"]}
                    for k, v in self.results.items()}

    def wait_for(self, ids: Iterable[str], timeout_s: float) -> bool:
        ids = set(ids)
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            with self.lock:
                if ids <= set(self.results):
                    return True
            time.sleep(0.1)
        return False

    def all_bytes(self) -> bytes:
        with self.lock:
            return b"".join(self.raw)


# ── the server side (HTTP) ──────────────────────────────────────────────────────────────────


class Client:
    """A thin client: every request carries `X-miLLM-Load-Policy: refuse`."""

    def __init__(self, base: str, timeout: float = 900.0) -> None:
        import httpx

        self.base = base.rstrip("/")
        self.http = httpx.Client(base_url=self.base, timeout=timeout, trust_env=False,
                                 headers=LOAD_POLICY_HEADERS)

    def call(self, method: str, path: str, body: Any = None,
             headers: dict[str, str] | None = None) -> tuple[int, Any, Any]:
        r = self.http.request(method, path, json=body, headers=headers)
        try:
            j = r.json()
        except ValueError:
            j = {"_text": r.text[:500]}
        return r.status_code, j, r.headers

    def ok(self, method: str, path: str, body: Any = None) -> Any:
        status, j, _ = self.call(method, path, body)
        if status >= 300 or (isinstance(j, dict) and j.get("success") is False):
            raise RuntimeError(f"{method} {path} -> {status}: {json.dumps(j)[:600]}")
        return j.get("data", j) if isinstance(j, dict) and "success" in j else j

    def health(self) -> dict[str, Any]:
        _, j, _ = self.call("GET", "/api/health/detailed")
        return j

    def models(self) -> list[dict[str, Any]]:
        data = self.ok("GET", "/api/models")
        return data if isinstance(data, list) else []

    def probes(self) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        cursor = None
        while True:
            path = "/api/probes?limit=1000" + (f"&cursor={cursor}" if cursor else "")
            status, j, _ = self.call("GET", path)
            if status >= 300:
                raise RuntimeError(f"GET /api/probes -> {status}")
            out.extend(j.get("data") or [])
            cursor = (j.get("meta") or {}).get("next")
            if not cursor:
                return out

    def bench_present(self) -> bool:
        try:
            status, _, _ = self.call("GET", f"{BENCH_PREFIX}/state?samples=0")
        except Exception:  # noqa: BLE001
            return False
        return status == 200

    def chat(self, model: str, prompt: str, stream: bool, max_tokens: int = OUTPUT_TOKENS,
             headers: dict[str, str] | None = None, stop_after_chunks: int | None = None
             ) -> dict[str, Any]:
        """One timed request. Streaming reads SSE lines; `stop_after_chunks` disconnects early."""
        body = {"model": model, "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens, "temperature": 0, "stream": stream}
        t0 = time.perf_counter()
        out: dict[str, Any] = {"ok": False, "status": None, "result_ids": [], "entries": []}
        if not stream:
            r = self.http.post("/v1/chat/completions", json=body, headers=headers)
            t1 = time.perf_counter()
            out["status"] = r.status_code
            try:
                j = r.json()
            except ValueError:
                j = {}
            usage = j.get("usage") or {}
            out.update(ok=r.status_code == 200, latency_ms=(t1 - t0) * 1000.0,
                       completion_tokens=int(usage.get("completion_tokens") or 0),
                       prompt_tokens=int(usage.get("prompt_tokens") or 0),
                       request_id=j.get("id"), model=j.get("model"),
                       entries=verdict_entries(j), result_ids=find_result_ids(
                           {"v": j.get("millm_probe_verdicts")}),
                       error=(j.get("error") or {}).get("code") if r.status_code != 200 else None)
            out["decode_tps"] = None
            return out
        t_first = t_last = None
        usage: dict[str, Any] = {}
        entries: list[dict[str, Any]] = []
        chunks = 0
        request_id = None
        saw_done = False
        stream_error = None
        with self.http.stream("POST", "/v1/chat/completions", json=body, headers=headers) as r:
            out["status"] = r.status_code
            if r.status_code != 200:
                r.read()
                out.update(latency_ms=(time.perf_counter() - t0) * 1000.0, error=r.status_code)
                return out
            for line in r.iter_lines():
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    saw_done = True
                    break
                try:
                    chunk = json.loads(data)
                except ValueError:
                    continue
                if isinstance(chunk.get("error"), dict):
                    # miLLM reports a failure after the 200 as an SSE error event, then `[DONE]`
                    # (`inference_service._stream_error_event`): a FAILED request (review R1).
                    stream_error = chunk["error"].get("code") or "stream_error"
                    continue
                request_id = request_id or chunk.get("id")
                if chunk.get("usage"):
                    usage = chunk["usage"]
                if chunk.get("millm_probe_verdicts"):
                    entries = verdict_entries(chunk)
                for choice in chunk.get("choices") or []:
                    if (choice.get("delta") or {}).get("content"):
                        now = time.perf_counter()
                        t_first = t_first or now
                        t_last = now
                        chunks += 1
                if stop_after_chunks is not None and chunks >= stop_after_chunks:
                    out.update(disconnected=True)
                    break
        t1 = time.perf_counter()
        n = int(usage.get("completion_tokens") or 0)
        # ok only for a stream that ended with `[DONE]` and carried no error event: a stream cut
        # off without `[DONE]` or ending in an error is a failed request (review R1, 2026-10-25).
        ok = saw_done and stream_error is None and not out.get("disconnected")
        if not ok and not out.get("disconnected"):
            out["error"] = stream_error or "stream_ended_without_done"
        out.update(ok=ok, latency_ms=(t1 - t0) * 1000.0,
                   ttft_ms=((t_first - t0) * 1000.0) if t_first else None,
                   completion_tokens=n, prompt_tokens=int(usage.get("prompt_tokens") or 0),
                   decode_tps=((n - 1) / (t_last - t_first)) if (t_first and t_last
                                                                 and t_last > t_first and n > 1)
                   else None,
                   request_id=request_id, entries=entries,
                   result_ids=[e["result_id"] for e in entries if e.get("result_id")])
        return out


class BenchArms:
    """The arm switch on ONE bench server (otel_bench_server.py), verified by the product's health."""

    kind = "bench"

    def __init__(self, client: Client, labels: tuple[str, str] = ("off", "on")) -> None:
        self.client = client
        self.labels = labels
        self.current: str | None = None

    def server(self, arm: str) -> Client:  # noqa: ARG002 - the arm interface
        return self.client

    def set(self, arm: str) -> dict[str, Any]:
        state = self.client.ok("POST", f"{BENCH_PREFIX}/arm", {"preset": arm})
        problem = arm_mismatch(arm, (self.client.health() or {}).get("telemetry") or {})
        if problem:
            raise RuntimeError(f"arm switch not in force: {problem}")
        self.current = arm
        return state

    def reset_timer(self, arm: str) -> None:  # noqa: ARG002 - the arm interface
        self.client.ok("POST", f"{BENCH_PREFIX}/reset")

    def state(self, arm: str, samples: bool = True) -> dict[str, Any]:  # noqa: ARG002 - the arm interface
        return self.client.ok("GET", f"{BENCH_PREFIX}/state?samples={1 if samples else 0}")


class ServerArms:
    """Two servers, one per arm (the pre-038 pair). Nothing is switched; requests are routed."""

    kind = "servers"

    def __init__(self, a: Client, b: Client, labels: tuple[str, str] = ("pre038", "038_off"),
                 b_preset: str = "off") -> None:
        self.clients = {labels[0]: a, labels[1]: b}
        self.labels = labels
        self.b_preset = b_preset
        self.current: str | None = None

    def server(self, arm: str) -> Client:
        return self.clients[arm]

    def set(self, arm: str) -> dict[str, Any]:
        self.current = arm
        if arm == self.labels[1] and self.clients[arm].bench_present():
            return self.clients[arm].ok("POST", f"{BENCH_PREFIX}/arm", {"preset": self.b_preset})
        return {}

    def reset_timer(self, arm: str) -> None:
        if self.clients[arm].bench_present():
            self.clients[arm].ok("POST", f"{BENCH_PREFIX}/reset")

    def state(self, arm: str, samples: bool = True) -> dict[str, Any]:
        c = self.clients[arm]
        if c.bench_present():
            return c.ok("GET", f"{BENCH_PREFIX}/state?samples={1 if samples else 0}")
        return {"telemetry": (c.health() or {}).get("telemetry") or {}}


@dataclass
class Context:
    out: Path
    card: str | None
    relay: Relay | None
    backlog: int
    dry_run: bool
    gpu_memory: Callable[[str | None], int | None]
    profile: dict[str, Any]
    max_tokens: int = OUTPUT_TOKENS
    #: How long the AT checks wait for a pipeline to flush before reading the relay.
    settle_s: float = 6.0


def _gpu_memory(card: str | None) -> int | None:  # pragma: no cover - needs a GPU
    if not card:
        return None
    out = subprocess.run(["nvidia-smi", "--query-gpu=uuid,memory.used",
                          "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout
    return parse_smi_memory(out).get(card)


def _cards() -> list[str]:  # pragma: no cover - needs a GPU
    out = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, check=True).stdout
    return parse_uuids(out)


def preflight(client: Client, model: str, *, need_bench: bool = True,
              relay: Relay | None = None) -> dict[str, Any]:
    """Refuse unless: bench routes present, `model` resident, MAX_CONCURRENT_REQUESTS == 1, and
    (with a relay) the telemetry endpoint is the relay. Returns the facts it checked."""
    if need_bench and not client.bench_present():
        raise SystemExit(
            f"REFUSED: {client.base} has no {BENCH_PREFIX} routes — it is not a bench pod "
            "(otel_bench_server.py). This harness never runs against the live miLLM.")
    health = client.health()
    refusal = residency_refusal(health, client.models(), model)
    if refusal:
        raise SystemExit(refusal)
    listing = client.ok("GET", "/api/environment")
    mcr = env_value(listing, "MAX_CONCURRENT_REQUESTS")
    if mcr != 1:
        raise SystemExit(f"REFUSED: MAX_CONCURRENT_REQUESTS is {mcr!r}; the profile needs 1 (13.3).")
    if relay is not None:
        problem = relay_mismatch(health.get("telemetry") or {}, relay.url)
        if problem:
            raise SystemExit(f"REFUSED: {problem}")
    return {"model": model, "max_concurrent_requests": mcr, "version": health.get("version"),
            "placement": health.get("model_placement")}


def ensure_armed(client: Client, names: Sequence[str]) -> list[str]:
    """Arm exactly `names` (disarming any other armed probe); returns the ids. Refuses loudly."""
    probes = client.probes()
    by_name = {p["name"]: p for p in probes}
    missing = [n for n in names if n not in by_name]
    if missing:
        raise SystemExit(f"REFUSED: probes not imported on this server: {missing} "
                         "(run plan-probes --apply on the bench pod first)")
    want = {by_name[n]["id"] for n in names}
    for p in probes:
        if p.get("armed") and p["id"] not in want:
            client.ok("POST", f"/api/probes/{p['id']}/disarm")
    for n in names:
        p = by_name[n]
        if not p.get("armed"):
            status, j, _ = client.call("POST", f"/api/probes/{p['id']}/arm", {"windows": []})
            if status >= 300 or not j.get("success", status < 300):
                code = (j.get("error") or {}).get("code")
                hint = (" — the set exceeds the bench pod's probe limits; re-run with "
                        "--raise-limits (bench pod only, restored after)") if code and (
                    "BUDGET" in code or "MAX" in code or "LIMIT" in code) else ""
                raise SystemExit(f"REFUSED: arming {n} -> {status} {code}{hint}")
    armed = {p["id"] for p in client.probes() if p.get("armed")}
    if armed != want:
        raise SystemExit(f"REFUSED: armed set {sorted(armed)} is not the requested {sorted(want)}")
    return sorted(want)


def _block_state(arms: Any, arm: str) -> dict[str, Any]:
    try:
        state = arms.state(arm, samples=True)
    except Exception:  # noqa: BLE001 - evidence collection records what failed
        return {}
    telemetry = state.get("telemetry") or {}
    out: dict[str, Any] = {
        "queue_used": {k: ((v or {}).get("queue") or {}).get("used")
                       for k, v in (telemetry.get("signals") or {}).items()},
        "signals": {k: {f: (v or {}).get(f) for f in ("state", "exported", "failed", "dropped")}
                    for k, v in (telemetry.get("signals") or {}).items()},
    }
    store = state.get("counter_store") or {}
    if store:
        out["counter_series"] = sum((store.get("series") or {}).values())
        out["series_overflow"] = store.get("series_overflow")
    if state.get("rss_kib") is not None:
        out["rss_kib"] = state["rss_kib"]
    enq = state.get("enqueue") or {}
    if enq:
        out["enqueue_calls"] = enq.get("calls")
        out["enqueue_us"] = enq.get("samples_us") or {}
    return out


def run_block(ctx: Context, arms: Any, cell: Cell, steps: Sequence[Step], prompt: str,
              max_tokens: int) -> dict[str, Any]:
    arm = steps[0].arm
    client = arms.server(arm)
    arms.reset_timer(arm)
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=1 + ctx.backlog) as ex:
        reqs = list(ex.map(lambda _s: client.chat(cell.model, prompt, cell.stream, max_tokens),
                           steps))
    wall = time.perf_counter() - t0
    for r in reqs:
        r.pop("entries", None)
    block = {"phase": steps[0].phase, "arm": arm, "block": steps[0].block, "wall_s": wall,
             "tokens": sum(int(r.get("completion_tokens") or 0) for r in reqs if r.get("ok")),
             "requests": reqs, "gpu_used_mib": ctx.gpu_memory(ctx.card), "at_et": now_et(),
             **_block_state(arms, arm)}
    wrong = {r.get("model") for r in reqs if r.get("ok") and r.get("model") not in (None, cell.model)}
    if wrong:
        raise SystemExit(f"REFUSED: responses named another model {sorted(wrong)}")
    return block


def calibrate_prompt(client: Client, cell: Cell, words: int) -> tuple[int, int]:
    """One warm-up request measures the prompt length; returns (words, measured tokens)."""
    r = client.chat(cell.model, make_prompt(words, cell.prompt), False, max_tokens=1)
    if not r.get("ok"):
        raise SystemExit(f"REFUSED: calibration request failed: {r.get('status')} {r.get('error')}")
    measured = int(r.get("prompt_tokens") or 0)
    return rescale_words(words, cell.prompt_tokens, measured), measured


def run_cell(ctx: Context, arms: Any, cell: Cell, steps: Sequence[Step], *, kind: str,
             residency: Callable[[], None], on_arm: str | None) -> dict[str, Any]:
    words = initial_words(cell.prompt_tokens)
    words, first_measured = calibrate_prompt(arms.server(arms.labels[0]), cell, words)
    prompt = make_prompt(words, cell.prompt)
    blocks = []
    for group in group_blocks(steps):
        residency()
        arm = group[0].arm
        if arms.current != arm:
            arms.set(arm)
        block = run_block(ctx, arms, cell, group, prompt, ctx.max_tokens)
        if (kind == "telemetry" and arm == on_arm and group[0].phase == "measured"
                and not (block.get("enqueue_calls") or {}).get("enqueue_log")):
            raise SystemExit("REFUSED: the enqueue timer saw no call in a measured 'on' block — "
                             "the bench patch is not wired, or no result was emitted")
        blocks.append(block)
        print(f"{cell.slug} {group[0].phase} block {group[0].block} arm {arm}: "
              f"{len(block['requests'])} req, {block['wall_s']:.2f} s, {block['tokens']} tok",
              flush=True)
    doc: dict[str, Any] = {
        "cell": {**asdict(cell), "slug": cell.slug}, "kind": kind, "arms": list(arms.labels),
        "arm_switch": arms.kind, "prompt_words": words, "calibration_prompt_tokens": first_measured,
        "blocks": blocks, "at_et": now_et(), "arrival_measured": False,
    }
    if ctx.relay is not None and on_arm is not None:
        ids = [rid for b in blocks if b["arm"] == on_arm and b["phase"] == "measured"
               for r in b["requests"] for rid in r.get("result_ids", [])]
        ctx.relay.wait_for(ids, timeout_s=15.0)
        arrivals = ctx.relay.arrivals()
        doc["arrivals"] = {rid: arrivals[rid] for rid in ids if rid in arrivals}
        doc["arrival_measured"] = True
    return doc


def _with_raised_limits(client: Client, raise_limits: bool) -> Callable[[], None]:
    if not raise_limits:
        return lambda: None
    for key, value in RAISED.items():
        client.ok("PUT", f"/api/environment/{key}",
                  {"value": value, "reason": "otel benchmark (bench pod), restored after"})

    def restore() -> None:
        for key in RAISED:
            client.call("DELETE", f"/api/environment/{key}?reason=otel+benchmark+done")

    return restore


def _residency_check(client: Client, model: str) -> Callable[[], None]:
    def check() -> None:
        refusal = residency_refusal(client.health(), client.models(), model)
        if refusal:
            raise SystemExit(refusal)

    return check


def write_json(path: Path, doc: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=1, default=str))


def cmd_sweep(args: argparse.Namespace, ctx: Context, client: Client,
              client_b: Client | None) -> int:
    sets = json.loads(Path(args.sets).read_text())["sets"] if args.sets else {}
    set_names = args.probe_sets.split(",")
    for s in set_names:
        if s not in sets:
            raise SystemExit(f"REFUSED: probe set {s!r} is not in {args.sets}")
    models = args.models.split(",") if args.models else [args.model]
    grid = cell_grid(models, set_names, args.prompts.split(","),
                     [x == "stream" for x in args.streams.split(",")])
    mine = [c for c in grid if c.model == args.model]
    pending = [c.slug for c in grid if c.model != args.model]
    if pending:
        print(f"pending (needs another resident model, a separate run): {pending}", flush=True)
    steps = schedule(args.warmup, args.measured, args.block,
                     arms=("pre038", "038_off") if args.arm_switch == "servers" else ("off", "on"))
    ctx.profile["probe_sets"] = {f"{args.model}:{s}": sets[s] for s in set_names}
    if args.arm_switch == "servers":
        if client_b is None:
            raise SystemExit("REFUSED: --arm-switch servers needs --base-b")
        # Arm A is the pre-038 server (no bench routes there), arm B this image with export off.
        preflight(client_b, args.model, need_bench=True)
        preflight(client, args.model, need_bench=False)
        arms: Any = ServerArms(client, client_b)
        kind, on_arm = "pre038", None
    else:
        preflight(client, args.model, relay=ctx.relay)
        arms = BenchArms(client)
        kind, on_arm = "telemetry", "on"
    restore = _with_raised_limits(arms.server(arms.labels[1]), args.raise_limits)
    try:
        for cell in mine:
            for c in {id(arms.server(a)): arms.server(a) for a in arms.labels}.values():
                ensure_armed(c, sets[cell.probe_set])
            checks = [_residency_check(arms.server(a), cell.model) for a in arms.labels]

            def residency(checks: list[Callable[[], None]] = checks) -> None:
                for check in checks:
                    check()

            doc = run_cell(ctx, arms, cell, steps, kind=kind, residency=residency, on_arm=on_arm)
            write_json(ctx.out / f"cell-{kind}-{cell.slug}.json", doc)
            s = summarise_cell(doc)
            print(json.dumps({"cell": cell.slug, "latency_p95_delta": s["latency_p95_delta"],
                              "throughput_delta": s["throughput_delta"],
                              "arrival": s["arrival"]}, default=str), flush=True)
    finally:
        if isinstance(arms, BenchArms) and arms.current not in (None, "off"):
            arms.set("off")
        restore()
    write_json(ctx.out / "profile.json", merge_profile(ctx.out / "profile.json", ctx.profile))
    return 0


def merge_profile(path: Path, profile: dict[str, Any]) -> dict[str, Any]:
    """One profile across runs (one sweep per resident model): probe sets accumulate."""
    old = json.loads(path.read_text()) if path.exists() else {}
    sets = {**(old.get("probe_sets") or {}), **(profile.get("probe_sets") or {})}
    return {**old, **profile, "probe_sets": sets}


def _telemetry(client: Client) -> dict[str, Any]:
    return (client.health() or {}).get("telemetry") or {}


def _prometheus_probe_lines(client: Client) -> list[str]:
    r = client.http.get("/api/health/metrics/prometheus")
    return [line for line in r.text.splitlines() if line.startswith("millm_probe_")]


def cmd_outage(args: argparse.Namespace, ctx: Context, client: Client) -> int:
    sets = json.loads(Path(args.sets).read_text())["sets"]
    preflight(client, args.model, relay=ctx.relay if args.outage_mode == "relay" else None)
    if args.outage_mode == "relay" and ctx.relay is None:
        raise SystemExit("REFUSED: --outage-mode relay needs the relay (drop --no-relay)")
    arms = BenchArms(client)
    restore = _with_raised_limits(client, args.raise_limits)
    cell = Cell(args.model, args.probe_set, args.prompt, False)
    phases = [("before", args.before_s), ("outage", args.outage_s), ("after", args.after_s)]
    doc: dict[str, Any] = {"model": args.model, "probe_set": args.probe_set,
                           "mode": args.outage_mode, "outage_s": args.outage_s,
                           "telemetry": {}, "prometheus": {}, "phases": {}, "completed": False}
    check = _residency_check(client, args.model)
    try:
        ensure_armed(client, sets[args.probe_set])
        arms.set("on")
        prompt = make_prompt(initial_words(cell.prompt_tokens), cell.prompt)
        doc["telemetry"]["start"] = _telemetry(client)
        doc["prometheus"]["start"] = _prometheus_probe_lines(client)
        for phase, seconds in phases:
            if phase == "outage":
                doc["telemetry"]["before"] = _telemetry(client)
                doc["prometheus"]["before"] = _prometheus_probe_lines(client)
                _outage_begin(args, ctx)
                doc["outage_began_et"] = now_et()
            check()
            reqs: list[dict[str, Any]] = []
            t_end = time.monotonic() + seconds
            while time.monotonic() < t_end:
                with ThreadPoolExecutor(max_workers=1 + ctx.backlog) as ex:
                    batch = list(ex.map(
                        lambda _i: _safe_chat(client, cell, prompt, args.max_tokens),
                        range(1 + ctx.backlog)))
                reqs.extend(batch)
            if phase == "outage":
                _outage_end(args, ctx)
                doc["outage_ended_et"] = now_et()
            doc["phases"][phase] = {
                "requests": len(reqs), "failed": [r for r in reqs if not r.get("ok")][:50],
                "n_failed": sum(1 for r in reqs if not r.get("ok")), "at_et": now_et()}
            print(f"outage phase {phase}: {len(reqs)} requests, "
                  f"{doc['phases'][phase]['n_failed']} failed", flush=True)
        deadline = time.monotonic() + args.drain_timeout_s
        while time.monotonic() < deadline:
            sig = _telemetry(client).get("signals") or {}
            if all(((v or {}).get("queue") or {}).get("used", 0) == 0 for v in sig.values()):
                break
            time.sleep(1.0)
        doc["telemetry"]["after"] = _telemetry(client)
        doc["prometheus"]["after"] = _prometheus_probe_lines(client)
        doc["requests"] = sum(p["requests"] for p in doc["phases"].values())
        doc["failures_by_phase"] = {k: v["n_failed"] for k, v in doc["phases"].items()}
        # Conservative: every failure while the receiver was down or recovering counts as
        # attributable to export unless the evidence shows otherwise (the operator reads it).
        doc["failures_attributable"] = (doc["phases"]["outage"]["n_failed"]
                                        + doc["phases"]["after"]["n_failed"])
        doc["valid"] = doc["phases"]["before"]["n_failed"] == 0
        doc["completed"] = True
    finally:
        if ctx.relay is not None and not ctx.relay.running:
            ctx.relay.start()
        arms.set("off")
        restore()
        write_json(ctx.out / "outage.json", doc)
    return 0


def _safe_chat(client: Client, cell: Cell, prompt: str, max_tokens: int) -> dict[str, Any]:
    try:
        r = client.chat(cell.model, prompt, False, max_tokens)
    except Exception as exc:  # noqa: BLE001 - a transport failure is a failed request, recorded
        return {"ok": False, "error": type(exc).__name__}
    return {"ok": r.get("ok"), "status": r.get("status"), "error": r.get("error"),
            "latency_ms": r.get("latency_ms")}


def _outage_begin(args: argparse.Namespace, ctx: Context) -> None:
    if args.outage_mode == "relay":
        assert ctx.relay is not None
        ctx.relay.stop()
        return
    _operator_wait("Stop the receiver (operator action, cluster):\n"
                   "  kubectl -n millm-otel scale deployment/otel-collector --replicas=0\n"
                   "Press Enter once no Collector pod is Ready.")


def _outage_end(args: argparse.Namespace, ctx: Context) -> None:
    if args.outage_mode == "relay":
        assert ctx.relay is not None
        ctx.relay.start()
        return
    _operator_wait("Restore the receiver (operator action, cluster):\n"
                   "  kubectl -n millm-otel scale deployment/otel-collector --replicas=1\n"
                   "Press Enter once the Collector pod is Ready.")


def _operator_wait(message: str) -> None:  # pragma: no cover - interactive
    print(message, flush=True)
    sys.stdin.readline()


# -- AT re-run (12.5's remaining halves) ------------------------------------------------------------

AT_MANUAL = {
    "AT03": "above/below/on-threshold, no threshold and non-finite failure need crafted probes; "
            "the CPU suite covers them (test_probe_result_outcomes); run by hand if required",
    "AT05": "GGUF, batched, continuous-batching and speculative paths need those engines loaded",
    "AT06": "needs a recalibration of an unchanged detector (POST /api/probes/{id}/recalibrate "
            "with a matching cut from miStudio)",
}
FORBIDDEN_METRIC_KEYS = frozenset({"request_id", "result_id", "trace_id", "span_id",
                                   "session_id", "batch_id"})


def _sentinels(n: int = 3, seed: int | None = None) -> list[str]:
    rng = random.Random(seed)
    return ["qz" + "".join(rng.choice(string.ascii_lowercase) for _ in range(22)) for _ in range(n)]


def sentinel_hits(raw: bytes, sentinels: Sequence[str], min_len: int = 8) -> dict[str, int]:
    """Occurrences of each sentinel, whole or as any substring of `min_len` characters or more."""
    hits = {}
    for s in sentinels:
        subs = {s[i:j] for i in range(len(s)) for j in range(i + min_len, len(s) + 1)}
        hits[s] = sum(raw.count(sub.encode()) for sub in subs if raw.count(sub.encode()))
    return hits


def _new_trace() -> tuple[str, str]:
    return ("".join(random.choice("0123456789abcdef") for _ in range(32)),
            "".join(random.choice("0123456789abcdef") for _ in range(16)))


def at_checks(client: Client, arms: BenchArms, relay: Relay, model: str,
              probe_ids: Sequence[str], template: dict[str, Any] | None,
              wait_s: float = 20.0, settle_s: float = 6.0) -> dict[str, Any]:
    checks: dict[str, Any] = {}
    prompt = make_prompt(40, "at")

    def chat(**kw: Any) -> dict[str, Any]:
        return client.chat(model, kw.pop("content", prompt), kw.pop("stream", False),
                           kw.pop("max_tokens", 16), **kw)

    arms.set("on")
    # AT02 (result half): known, missing and malformed trace context; interleaved requests.
    trace, span = _new_trace()
    known = chat(headers={"traceparent": f"00-{trace}-{span}-01"})
    missing = chat()
    malformed = chat(headers={"traceparent": "00-zzzz-yyyy-01"})
    pairs = [_new_trace() for _ in range(4)]
    with ThreadPoolExecutor(max_workers=4) as ex:
        inter = list(ex.map(lambda p: chat(headers={"traceparent": f"00-{p[0]}-{p[1]}-01"}), pairs))
    ids = (known["result_ids"] + missing["result_ids"] + malformed["result_ids"]
           + [i for r in inter for i in r["result_ids"]])
    relay.wait_for(ids, wait_s)
    rec = relay.results

    def traces_of(r: dict[str, Any]) -> set[str]:
        return {e.get("trace_id") for e in r["entries"]}

    def exported_traces(r: dict[str, Any]) -> set[str]:
        return {rec[i]["trace_id"] for i in r["result_ids"] if i in rec}

    ok_known = bool(known["result_ids"]) and traces_of(known) == {trace} == exported_traces(known)
    t_missing = traces_of(missing)
    ok_missing = (bool(missing["result_ids"]) and len(t_missing) == 1
                  and trace not in t_missing and exported_traces(missing) == t_missing)
    t_mal = traces_of(malformed)
    ok_mal = (bool(malformed["result_ids"]) and len(t_mal) == 1
              and all(t and re.fullmatch(r"[0-9a-f]{32}", t) for t in t_mal)
              and exported_traces(malformed) == t_mal)
    ok_inter = all(bool(r["result_ids"]) and traces_of(r) == {p[0]} == exported_traces(r)
                   for r, p in zip(inter, pairs, strict=True))
    checks["AT02"] = {"result": "pass" if ok_known and ok_missing and ok_mal and ok_inter else "fail",
                      "facts": {"known": ok_known, "missing": ok_missing, "malformed": ok_mal,
                                "interleaved": ok_inter, "results": len(ids),
                                "exported": sum(1 for i in ids if i in rec)}}

    # AT04 (result half): a completed stream and a client disconnect agree on both surfaces.
    done = chat(stream=True, max_tokens=24)
    cut = chat(stream=True, max_tokens=OUTPUT_TOKENS, stop_after_chunks=2)
    time.sleep(settle_s)
    events: list[dict[str, Any]] = []
    deadline = time.monotonic() + wait_s
    while cut.get("request_id") and time.monotonic() < deadline:
        _, j, _ = client.call("GET", f"/api/probes/events?request_id={cut['request_id']}")
        events = j.get("data") or []
        if events:
            break
        time.sleep(0.5)
    cut_ids = [e["result_id"] for e in events if e.get("result_id")]
    relay.wait_for(done["result_ids"] + cut_ids, wait_s)
    api_outcomes = {e["result_id"]: e.get("outcome") for e in events if e.get("result_id")}
    exp_outcomes = {i: rec[i]["attributes"].get("mi.probe.outcome") for i in cut_ids if i in rec}
    done_outcomes = {rec[i]["attributes"].get("mi.probe.outcome") for i in done["result_ids"]
                     if i in rec}
    ok04 = (bool(cut_ids) and api_outcomes == exp_outcomes
            and any(o != "completed" for o in api_outcomes.values())
            and done_outcomes == {"completed"})
    checks["AT04"] = {"result": "pass" if ok04 else "fail",
                      "facts": {"completed_stream": sorted(map(str, done_outcomes)),
                                "disconnect_api": sorted(map(str, api_outcomes.values())),
                                "disconnect_exported": sorted(map(str, exp_outcomes.values()))}}

    # AT07: the stateless score route — default never exports; permitted export marks `stateless`.
    def stateless_records() -> int:
        with relay.lock:
            return sum(1 for v in relay.results.values()
                       if v["attributes"].get("mi.probe.origin") == "stateless")

    def newest_event() -> int | None:
        _, j, _ = client.call("GET", "/api/probes/events?limit=1")
        data = j.get("data") or []
        return data[0].get("id") if data else None

    score_body = {"probe_ids": list(probe_ids), "inputs": [{"text": prompt}], "export_results": True}
    ev0, st0 = newest_event(), stateless_records()
    _, s1, _ = client.call("POST", "/api/probes/score", score_body)
    time.sleep(settle_s)
    st1, ev1 = stateless_records(), newest_event()
    arms.set("on_stateless")
    _, s2, _ = client.call("POST", "/api/probes/score", score_body)
    n2 = len(find_result_ids(s2.get("data")))
    relay.wait_for(find_result_ids(s2.get("data")), wait_s)
    st2, ev2 = stateless_records(), newest_event()
    arms.set("on")
    ok07 = st1 == st0 and ev1 == ev0 and n2 > 0 and st2 - st1 == n2 and ev2 == ev0
    checks["AT07"] = {"result": "pass" if ok07 else "fail",
                      "facts": {"not_permitted_exported": st1 - st0, "permitted_results": n2,
                                "permitted_exported": st2 - st1, "live_events_created":
                                [ev0, ev1, ev2], "export_summary_default":
                                (s1.get("data") or {}).get("export") if isinstance(
                                    s1.get("data"), dict) else None}}

    # AT08 (result half): sampler 0 — no traces; result logs and metrics still arrive; no ids
    # or free text as metric dimensions.
    arms.set("on_sampler0")
    interval = float(((_telemetry(client).get("config") or {}).get("metric_interval_s")) or 15.0)
    spans0, points0 = relay.span_count, len(relay.metric_points)
    r8 = chat()
    relay.wait_for(r8["result_ids"], wait_s)
    time.sleep(interval + settle_s)
    arms.set("on")  # the switch drains the sampler-0 pipeline (metrics included)
    with relay.lock:
        new_points = relay.metric_points[points0:]
        spans_seen = relay.span_count - spans0
    bad_keys = sorted({k for p in new_points for k in p["attributes"] if k in FORBIDDEN_METRIC_KEYS})
    id_values = set(r8["result_ids"])
    leaked = sorted({str(v) for p in new_points for v in p["attributes"].values() if v in id_values})
    ok08 = (spans_seen == 0 and bool(r8["result_ids"])
            and all(i in relay.results for i in r8["result_ids"]) and bool(new_points)
            and not bad_keys and not leaked)
    checks["AT08"] = {"result": "pass" if ok08 else "fail",
                      "facts": {"spans": spans_seen, "result_records": len(r8["result_ids"]),
                                "metric_points": len(new_points), "forbidden_keys": bad_keys,
                                "id_values": leaked}}

    # AT10: sentinels never leave, whole or in part, through every route the FPRD names — the
    # chat route, the STREAM, credential headers, and the score route WITH export permitted and
    # requested (review R1, 2026-10-25: the score call exported nothing, the stream and headers
    # were never sent, so those halves passed vacuously). Substrings of 8+ characters, not the
    # FPRD's 4: a random lowercase 4-gram matches ~1 in 26^4 positions of the lowercase attribute
    # text a capture holds, i.e. several false hits per 50 KB; the CPU AT10 matches whole strings.
    sentinels = _sentinels(6)
    relay.capture_raw = True
    start = len(relay.raw)
    st_before = stateless_records()
    chat(content=f"{prompt} {sentinels[0]}",
         headers={"Authorization": f"Bearer {sentinels[3]}", "X-Api-Key": sentinels[4]})
    chat(content=f"{prompt} {sentinels[1]}", stream=True, max_tokens=8)
    arms.set("on_stateless")
    client.call("POST", "/api/probes/score", {
        "probe_ids": list(probe_ids), "export_results": True,
        "inputs": [{"text": f"{prompt} {sentinels[2]}"},
                   {"messages": [{"role": "user", "content": sentinels[5]}]}]})
    time.sleep(settle_s)
    arms.set("on")  # drain
    with relay.lock:
        raw = b"".join(relay.raw[start:])
    relay.capture_raw = False
    score_exported = stateless_records() - st_before
    hits = sentinel_hits(raw, sentinels)
    checks["AT10"] = {"result": "pass" if raw and score_exported > 0 and not any(hits.values())
                      else "fail",
                      "facts": {"bytes_searched": len(raw), "hits": sum(hits.values()),
                                "score_records_exported": score_exported}}

    # AT12 (model mismatch): a refused arm is one lifecycle record carrying the refusal's code.
    if template is None:
        checks["AT12"] = {"result": "not_run", "facts": {"why": "no --template given"}}
    else:
        doc = copy.deepcopy(template)
        doc["name"] = "otelbench-mismatch-" + "".join(random.choice(string.ascii_lowercase)
                                                      for _ in range(6))
        doc["model"] = {**doc["model"], "hf_id": "otelbench/not-the-resident-model",
                        "revision": "main"}
        doc.pop("identity", None)
        n0 = len(relay.lifecycle)
        status, j, _ = client.call("POST", "/api/probes/import?on_conflict=rename", doc)
        pid = (j.get("data") or {}).get("id") if status < 300 else None
        code = None
        if pid:
            _, ja, _ = client.call("POST", f"/api/probes/{pid}/arm", {"windows": []})
            code = None if ja.get("success") else (ja.get("error") or {}).get("code")
            client.call("DELETE", f"/api/probes/{pid}")
        arms.set("on")  # drain
        with relay.lock:
            recs = [r["attributes"] for r in relay.lifecycle[n0:]]
        refused = [r for r in recs if r.get("mi.lifecycle.action") == "arm_refused"
                   or r.get("mi.lifecycle.outcome") == "refused"]
        ok12 = bool(pid) and bool(code) and any(r.get("mi.lifecycle.code") == code for r in refused)
        checks["AT12"] = {"result": "pass" if ok12 else "fail",
                          "facts": {"import_status": status, "arm_code": code,
                                    "lifecycle_actions": [r.get("mi.lifecycle.action")
                                                          for r in recs]}}
    for name, why in AT_MANUAL.items():
        checks[name] = {"result": "manual", "facts": {"why": why}}
    return checks


def cmd_at(args: argparse.Namespace, ctx: Context, client: Client) -> int:
    if ctx.relay is None:
        raise SystemExit("REFUSED: the AT re-run reads the exported records at the relay")
    sets = json.loads(Path(args.sets).read_text())["sets"]
    preflight(client, args.model, relay=ctx.relay)
    arms = BenchArms(client)
    try:
        ids = ensure_armed(client, sets[args.probe_set])
        template = json.loads(Path(args.template).read_text()) if args.template else None
        checks = at_checks(client, arms, ctx.relay, args.model, ids, template,
                           settle_s=ctx.settle_s)
    finally:
        arms.set("off")
    doc = {"model": args.model, "probe_set": args.probe_set, "at_et": now_et(), "checks": checks}
    write_json(ctx.out / "at.json", doc)
    for name, res in sorted(checks.items()):
        print(f"{name}: {res['result']} {json.dumps(res['facts'], default=str)[:200]}", flush=True)
    return 0 if all(r["result"] in ("pass", "manual") for r in checks.values()) else 1


# -- plan-probes (13.2) ---------------------------------------------------------------------------


def cmd_plan_probes(args: argparse.Namespace, out: Path) -> int:
    paths = sorted(Path(p) for p in args.definitions)
    docs = [json.loads(p.read_text()) for p in paths]
    plan = plan_probe_sets(docs, args.behaviour_key, args.behaviour_name,
                           args.pair.split(",") if args.pair else None,
                           expected=args.expected, strip_identity=args.strip_identity)
    sets_doc: dict[str, Any] = {"model": plan["model"], "behaviour": plan["behaviour"],
                                "sets": dict(plan["sets"]), "layers_rules": plan["layers_rules"]}
    documents = list(plan["documents"])
    if args.adapt:
        hf_id, d_model, n_layers, arch = args.adapt.split(",")
        synth = [adapt_synthetic(d, hf_id=hf_id, d_model=int(d_model), n_layers=int(n_layers),
                                 architecture=arch) for d in documents]
        rename = {d["name"]: s["name"] for d, s in zip(documents, synth, strict=True)}
        sets_doc = {"model": hf_id, "behaviour": plan["behaviour"], "synthetic": True,
                    "sets": {k: [rename[n] for n in v] for k, v in plan["sets"].items()},
                    "layers_rules": plan["layers_rules"]}
        documents = synth
    out.mkdir(parents=True, exist_ok=True)
    for d in documents:
        write_json(out / "definitions" / f"{d['name']}.json", d)
    write_json(out / "probe_sets.json", sets_doc)
    print(json.dumps(sets_doc, indent=1), flush=True)
    if not args.apply:
        print(f"DRY RUN: {len(documents)} definitions written under {out}; nothing imported. "
              "Re-run with --apply --base <bench pod URL> in the GPU window.", flush=True)
        return 0
    client = Client(args.base)
    if not client.bench_present():
        raise SystemExit(f"REFUSED: {args.base} is not a bench pod; plan-probes --apply imports "
                         "into a bench pod only (never the live miLLM)")
    existing = {p["name"] for p in client.probes()}
    for d in documents:
        if d["name"] in existing:
            print(f"present: {d['name']}", flush=True)
            continue
        data = client.ok("POST", "/api/probes/import?on_conflict=fail", d)
        print(f"imported: {d['name']} -> {data.get('id')}", flush=True)
    return 0


# -- report ----------------------------------------------------------------------------------------


def load_evidence(out: Path) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any] | None,
                                      dict[str, Any] | None]:
    profile = json.loads((out / "profile.json").read_text()) if (out / "profile.json").exists() else {}
    cells = [json.loads(p.read_text()) for p in sorted(out.glob("cell-*.json"))]
    outage = json.loads((out / "outage.json").read_text()) if (out / "outage.json").exists() else None
    at = json.loads((out / "at.json").read_text()) if (out / "at.json").exists() else None
    return profile, cells, outage, at


def cmd_report(args: argparse.Namespace, out: Path) -> int:
    profile, cells, outage, at = load_evidence(out)
    text = render_report(profile, cells, outage, at)
    if args.write:
        Path(args.write).write_text(text)
        print(f"written {args.write}", flush=True)
    else:
        print(text)
    return 0


# -- CLI ---------------------------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument("command", choices=["cards", "plan-probes", "sweep", "outage", "at", "report"])
    p.add_argument("--base", default="http://127.0.0.1:8000", help="the bench pod's miLLM")
    p.add_argument("--base-b", help="the second server (pre-038 pair: arm B, this image)")
    p.add_argument("--out", default=None, help="evidence directory (default $HWTEST_OUT/otel)")
    p.add_argument("--card", help="GPU UUID (from `cards`); checked against nvidia-smi -L")
    p.add_argument("--model", help="the RESIDENT model's served name")
    p.add_argument("--models", help="every model of the grid (default: --model)")
    p.add_argument("--sets", help="probe_sets.json from plan-probes")
    p.add_argument("--probe-sets", default="two,nine")
    p.add_argument("--probe-set", default="nine")
    p.add_argument("--prompts", default="short,long")
    p.add_argument("--prompt", default="short", help="outage: prompt length")
    p.add_argument("--streams", default="nostream,stream")
    p.add_argument("--warmup", type=int, default=WARMUP)
    p.add_argument("--measured", type=int, default=MEASURED_PER_ARM)
    p.add_argument("--block", type=int, default=BLOCK)
    p.add_argument("--backlog", type=int, default=BACKLOG)
    p.add_argument("--max-tokens", type=int, default=OUTPUT_TOKENS)
    p.add_argument("--arm-switch", choices=["bench", "servers"], default="bench")
    p.add_argument("--raise-limits", action="store_true",
                   help="bench pod only: raise PROBE_MAX_ARMED and budgets, restored after")
    p.add_argument("--relay-port", type=int, default=DEFAULT_RELAY_PORT)
    p.add_argument("--collector", default="http://otel-collector.millm-otel.svc:4318")
    p.add_argument("--no-relay", action="store_true", help="no arrival measurement")
    p.add_argument("--outage-mode", choices=["relay", "operator"], default="relay")
    p.add_argument("--before-s", type=float, default=120.0)
    p.add_argument("--outage-s", type=float, default=600.0)
    p.add_argument("--after-s", type=float, default=300.0)
    p.add_argument("--drain-timeout-s", type=float, default=120.0)
    p.add_argument("--template", help="at: a definition for the model-mismatch check (AT12)")
    p.add_argument("--definitions", nargs="*", default=[], help="plan-probes: definition files")
    p.add_argument("--behaviour-key", help="plan-probes: the behaviour key every probe carries")
    p.add_argument("--behaviour-name")
    p.add_argument("--pair", help="plan-probes: the two-probe set, two names")
    p.add_argument("--expected", type=int, default=9)
    p.add_argument("--strip-identity", action="store_true")
    p.add_argument("--adapt", help="plan-probes: hf_id,d_model,n_layers,architecture (synthetic)")
    p.add_argument("--apply", action="store_true", help="plan-probes: import into the bench pod")
    p.add_argument("--write", help="report: write the Markdown record here")
    p.add_argument("--dry-run", action="store_true",
                   help="against an in-process fake miLLM and fake Collector (no GPU)")
    p.add_argument("--fake-defects", default="", help=argparse.SUPPRESS)
    p.add_argument("--settle-s", type=float, default=6.0,
                   help="at: seconds to wait for a pipeline flush before reading the relay")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    from hw_common import OUT

    out = Path(args.out) if args.out else OUT / "otel"
    if args.command == "cards":  # pragma: no cover - needs a GPU
        print("\n".join(_cards()))
        return 0
    if args.command == "report":
        return cmd_report(args, out)
    if args.command == "plan-probes":
        if not args.behaviour_key:
            raise SystemExit("plan-probes needs --behaviour-key (the catalogue's key)")
        return cmd_plan_probes(args, out)
    fake = None
    if args.dry_run:
        from otel_fake_millm import FakeStack

        names = sorted({n for v in (json.loads(Path(args.sets).read_text())["sets"].values()
                                    if args.sets else []) for n in v})
        fake = FakeStack(model=args.model or "fake-model", probe_names=names,
                         second=args.arm_switch == "servers",
                         defects=frozenset(filter(None, args.fake_defects.split(",")))).start()
        args.base, args.base_b = fake.base, fake.base_b
        card: str | None = fake.card
        args.collector = fake.collector_url
        gpu: Callable[[str | None], int | None] = fake.gpu_memory
    else:  # pragma: no cover - needs a GPU
        card = choose_card(_cards(), args.card)
        gpu = _gpu_memory
    if not args.model:
        raise SystemExit("--model (the resident model's served name) is required")
    relay = None if args.no_relay else Relay(args.relay_port, args.collector).start()
    if fake is not None and relay is not None:
        fake.point_telemetry_at(relay.url)
    ctx = Context(out=out, card=card, relay=relay, backlog=args.backlog, dry_run=args.dry_run,
                  gpu_memory=gpu, profile={
                      "date_et": now_et(), "card_uuid": card, "backlog": args.backlog,
                      "output_tokens": args.max_tokens, "warmup": args.warmup,
                      "measured_per_arm": args.measured, "block": args.block,
                      "max_concurrent_requests": 1, "relay": relay.url if relay else None,
                      "collector": args.collector, "dry_run": args.dry_run},
                  max_tokens=args.max_tokens, settle_s=args.settle_s)
    client = Client(args.base)
    client_b = Client(args.base_b) if args.base_b else None
    try:
        if args.command == "sweep":
            if args.arm_switch == "servers":
                # Arm A = the pre-038 server (--base), arm B = this image (--base-b).
                return cmd_sweep(args, ctx, client, client_b)
            return cmd_sweep(args, ctx, client, None)
        if args.command == "outage":
            return cmd_outage(args, ctx, client)
        return cmd_at(args, ctx, client)
    finally:
        if relay is not None:
            relay.stop()
        if fake is not None:
            fake.stop()


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
