#!/usr/bin/env python3
"""Feature 37 hardware harness — run INSIDE the miLLM test pod in a GPU window, NOT by the unit suite.

Measures what N armed probes cost per forward pass, per card (chosen by UUID) and on a model split
across cards, and checks the arm-time PROJECTION (`GET /api/probes/status` → `projected`) against
the observed wall-clock overhead (armed minus disarmed, median of `--repeats` identical greedy
requests). FR-37.1, FR-37.8; plan in the suite spec `research/probe-cost-measurement.md`.

SYNTHETIC PROBES, AND WHY THEIR PARITY IS EXACT (recorded in every evidence file): each probe is a
clone of a REAL definition for the model (`--template`, `--sae-template`) with its head weights set
to ZERO and its bias to 1.0, so every position scores exactly 1.0 whatever the residual is. Its test
vectors keep the template's `token_ids` and record `token_scores = [1.0] * n`, so parity passes by
construction while the ARITHMETIC — the cast, the matvec over d_model or the k-sparse SAE slice, the
dispatch — is exactly a real probe's. The scores do not matter here; only the cost does. Each kind
also runs one REAL imported probe (the template itself) as a control case.

Subcommands:
  cards     list the cards by UUID (from `nvidia-smi -L`, never a document)
  sweep     for a model and each card UUID: N in --ns, kinds dense|sparse|mixed, spread one|many
  split     the same on a model loaded across every card (`gpu: "all"`)
  report    summarise evidence files: projection vs observed, and every ratio > 1.25 (a finding)

The per-device head memo control (FR-37.1): run `sweep` once on an image with the memo removed
(`ProbeHead.to_device` returning a fresh copy) and `report --baseline <normal> <memo-removed>`;
the harness must report a regression. This needs a patched image and is therefore a manual step.

Budget settings are raised for the run (so every N can arm) and RESTORED in a `finally`.
"""

from __future__ import annotations

import argparse
import copy
import json
import re
import statistics
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))

#: A finding (FR-37.8): observed above projection by more than this.
RATIO_LIMIT = 1.25
UUID_RE = re.compile(r"\(UUID:\s*(GPU-[0-9a-fA-F-]+)\)")
HARNESS_PREFIX = "hwscale-"
RAISED = {"PROBE_MAX_ARMED": 64, "PROBE_MAX_OVERHEAD_MS_PER_PASS": 10.0,
          "PROBE_MAX_PREFILL_MS_PER_1K": 10.0, "PROBE_BUDGET_HEADROOM": 1.0}


@dataclass(frozen=True)
class Case:
    model: str
    card: str          # a GPU UUID, or "all" for the split case
    n: int
    kind: str          # dense | sparse | mixed
    spread: str        # one | many

    @property
    def slug(self) -> str:
        card = "split" if self.card == "all" else self.card[:12]
        return f"{self.model.replace('/', '_')}-{card}-{self.kind}-{self.spread}-n{self.n}"


# ── pure parts (CPU-tested in tests/unit/hardware/test_probe_scaling_harness.py) ──────────────


def parse_uuids(nvidia_smi_l: str) -> list[str]:
    """Card UUIDs from `nvidia-smi -L`, in its order. Never an index (1 + n GPUs rule)."""
    return UUID_RE.findall(nvidia_smi_l)


def case_grid(model: str, cards: Iterable[str], ns: Iterable[int], kinds: Iterable[str],
              spreads: Iterable[str]) -> list[Case]:
    return [Case(model, c, n, k, s) for c in cards for n in ns for k in kinds for s in spreads]


def layers_for(n: int, spread: str, n_layers: int, base_layer: int) -> list[int]:
    """`one`: every probe on the template's layer. `many`: spread over the middle half."""
    if spread == "one":
        return [base_layer] * n
    lo, hi = n_layers // 4, max(n_layers // 4 + 1, (3 * n_layers) // 4)
    span = list(range(lo, hi))
    return [span[i % len(span)] for i in range(n)]


def kinds_for(n: int, kind: str) -> list[str]:
    if kind == "mixed":
        return ["sparse" if i % 2 else "dense" for i in range(n)]
    return [kind] * n


def adapt(template: dict[str, Any], *, hf_id: str, d_model: int, n_layers: int) -> dict[str, Any]:
    """The template re-pointed at another model (the second model of the sweep): its identity
    block, a head of that width, and NO pinned chat-template hash or revision SHA (`main`, which
    arming records as REVISION_UNVERIFIED and allows). Only valid with `synthetic` (zero head)."""
    doc = copy.deepcopy(template)
    doc["model"] = {**doc["model"], "hf_id": hf_id, "d_model": int(d_model),
                    "n_layers": int(n_layers), "revision": "main"}
    doc["model"].pop("chat_template_sha256", None)
    doc["model"].pop("load_dtype", None)
    doc.pop("sae", None)
    doc["head"]["weights"] = [0.0] * int(d_model)
    return doc


def synthetic(template: dict[str, Any], name: str, layer: int) -> dict[str, Any]:
    """A clone that scores exactly 1.0 everywhere: zero head, bias 1.0, matching test vectors."""
    doc = copy.deepcopy(template)
    doc["name"] = name
    doc["read"] = {**doc.get("read", {}), "layer": int(layer)}
    head = doc["head"]
    head["weights"] = [0.0] * len(head["weights"])
    head["bias"] = 1.0
    head["norm_mean"] = [0.0] * len(head["weights"])
    head["norm_std"] = [1.0] * len(head["weights"])
    head["attention_query"] = None
    doc["aggregation"] = {"rule": "mean", "params": {}, "streamable": True}
    doc["scope"] = "all"
    doc["decision"] = {**doc.get("decision", {}), "threshold": 2.0}
    doc.pop("behaviour", None)
    # Identity pins the harness cannot vouch for on a live cache are relaxed (warnings, not refusals):
    doc["model"] = {**doc["model"], "revision": "main"}
    doc["model"].pop("chat_template_sha256", None)
    vectors = []
    for v in (doc.get("test_vectors") or {}).get("vectors", [])[:2]:
        ids = list(v["token_ids"])
        vectors.append({**v, "token_scores": [1.0] * len(ids), "score": 1.0, "verdict": False})
    doc["test_vectors"] = {"tolerance": 1e-6, "vectors": vectors}
    doc.setdefault("provenance", {})["harness"] = "probe_scaling_acceptance: synthetic zero head"
    return doc


def observed_per_pass(armed_ms: list[float], disarmed_ms: list[float], n_passes: int) -> float:
    """Wall-clock overhead per forward pass: median(armed) − median(disarmed), over the passes."""
    return (statistics.median(armed_ms) - statistics.median(disarmed_ms)) / max(n_passes, 1)


def ratio(observed: float, projected: float) -> Optional[float]:
    return None if projected <= 0 else observed / projected


def findings(records: list[dict[str, Any]], limit: float = RATIO_LIMIT) -> list[dict[str, Any]]:
    """Every case whose observed decode cost exceeds its projection by more than `limit`."""
    return [r for r in records if (r.get("ratio_decode") or 0.0) > limit]


def regression(baseline: list[dict[str, Any]], candidate: list[dict[str, Any]],
               limit: float = RATIO_LIMIT) -> list[str]:
    """Cases whose observed cost grew by more than `limit` between two runs (the memo control)."""
    base = {r["case"]: r for r in baseline}
    out = []
    for r in candidate:
        b = base.get(r["case"])
        if b and b["observed_decode_ms"] > 0 and r["observed_decode_ms"] > limit * b["observed_decode_ms"]:
            out.append(r["case"])
    return out


# ── the pod side ────────────────────────────────────────────────────────────────────────────


def _api():  # pragma: no cover - needs the pod
    from hw_common import Api

    return Api()


def _cards() -> list[str]:  # pragma: no cover - needs a GPU
    out = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, check=True).stdout
    return parse_uuids(out)


def _set(api: Any, key: str, value: Any) -> None:  # pragma: no cover
    api.ok("PUT", f"/api/environment/{key}", body={"value": value, "reason": "probe scaling harness"})


def _reset(api: Any, key: str) -> None:  # pragma: no cover
    api.call("DELETE", f"/api/environment/{key}?reason=probe+scaling+harness+done")


def _clear(api: Any) -> None:  # pragma: no cover
    for p in api.ok("GET", "/api/probes"):
        if p["name"].startswith(HARNESS_PREFIX) or p["armed"]:
            if p["armed"]:
                api.ok("POST", f"/api/probes/{p['id']}/disarm")
            if p["name"].startswith(HARNESS_PREFIX):
                api.ok("DELETE", f"/api/probes/{p['id']}")


def _load_on(api: Any, model: str, card: str) -> None:  # pragma: no cover
    row = api.model_row(model)
    res = api.resident()
    if res is not None:
        api.unload(res["name"])
    api.ok("POST", f"/api/models/{row['id']}/load", body={"gpu": card})
    for _ in range(600):
        if api.model_row(model)["status"] == "loaded":
            return
        time.sleep(0.5)
    raise TimeoutError(f"load {model} on {card}")


def _timed_chats(api: Any, model: str, repeats: int, max_tokens: int,
                 hooks: Optional[list] = None) -> tuple[list[float], int]:  # pragma: no cover
    """Wall times of identical greedy requests; with `hooks`, also each request's product-timer
    figure (read after it, so the per-pass number is a MEDIAN, not the last request's alone)."""
    times, passes = [], 0
    for _ in range(repeats):
        t = time.perf_counter()
        status, body, _ = api.chat(model, "Count slowly from one to fifty, in words.",
                                   max_tokens=max_tokens)
        times.append((time.perf_counter() - t) * 1000.0)
        if status != 200:
            raise RuntimeError(f"chat -> {status}: {json.dumps(body)[:300]}")
        passes = 1 + int(body["usage"]["completion_tokens"])
        if hooks is not None:
            hooks.append(api.ok("GET", "/api/probes/status").get("last_overhead_ms_per_pass"))
    return times, passes


def run_case(api: Any, case: Case, templates: dict[str, dict], out: Path, repeats: int,
             max_tokens: int, n_layers: int) -> dict[str, Any]:  # pragma: no cover
    _clear(api)
    base_layer = int(templates["dense"]["read"]["layer"])
    disarmed, n_passes = _timed_chats(api, case.model, repeats, max_tokens)
    arm_ms = []
    for i, (kind, layer) in enumerate(zip(kinds_for(case.n, case.kind),
                                          layers_for(case.n, case.spread, n_layers, base_layer))):
        if kind == "sparse":
            # A k-sparse probe reads through ITS SAE, which exists for one layer: it stays there.
            layer = int(templates["sparse"]["read"]["layer"])
        doc = synthetic(templates[kind], f"{HARNESS_PREFIX}{kind}-{i}", layer)
        probe = api.ok("POST", "/api/probes/import?on_conflict=rename", body=doc)
        t = time.perf_counter()
        api.ok("POST", f"/api/probes/{probe['id']}/arm", body={"windows": []})
        arm_ms.append((time.perf_counter() - t) * 1000.0)
    status = api.ok("GET", "/api/probes/status")
    hooks: list = []
    armed, _ = _timed_chats(api, case.model, repeats, max_tokens, hooks=hooks)
    observed = observed_per_pass(armed, disarmed, n_passes)
    hook = statistics.median([h for h in hooks if h is not None]) if any(
        h is not None for h in hooks) else None
    record = {
        "case": case.slug, **asdict(case), "n_passes": n_passes,
        "projected_decode_ms": status["projected"]["decode"],
        "projected_prefill_ms_per_1k": status["projected"]["prefill"],
        "observed_decode_ms": observed,
        "product_timer_ms_per_pass": hook,
        "product_timer_samples": hooks,
        #: The budget is PER PASS of hook work, which the product's own timer measures. The wall
        #: clock difference also carries PER-REQUEST bookkeeping (event rows, context windows,
        #: retention), spread over the passes; it is reported, and judged separately.
        "ratio_decode": ratio(hook or 0.0, status["projected"]["decode"]),
        "ratio_wall": ratio(observed, status["projected"]["decode"]),
        "wall_overhead_ms_per_request": statistics.median(armed) - statistics.median(disarmed),
        "arm_ms_median": statistics.median(arm_ms) if arm_ms else None,
        "costs": {p["name"]: p.get("cost") for p in status["armed"]},
        "armed_ms": armed, "disarmed_ms": disarmed,
        "synthetic": "zero head, bias 1.0 — parity exact by construction; cost = a real probe's",
    }
    (out / f"{case.slug}.json").write_text(json.dumps(record, indent=1))
    print(f"{case.slug}: projected {record['projected_decode_ms']:.4f} hook {hook} wall "
          f"{observed:.4f} ms/pass ratio {record['ratio_decode']}", flush=True)
    return record


def parity_control(api: Any, model: str, template: dict[str, Any], out: Path,
                   label: str) -> dict[str, Any]:  # pragma: no cover
    """A REAL definition, imported as-is and armed: parity against miStudio's recorded scores."""
    _clear(api)
    doc = copy.deepcopy(template)
    doc["name"] = f"{HARNESS_PREFIX}real-{label}"
    status, body, _ = api.call("POST", "/api/probes/import?on_conflict=rename", body=doc)
    record: dict[str, Any] = {"case": f"parity-{label}", "model": model, "import_status": status}
    if status < 300:
        pid = body["data"]["id"]
        status, body, _ = api.call("POST", f"/api/probes/{pid}/arm", body={"windows": []})
        record.update(arm_status=status, success=body.get("success"),
                      error=(body.get("error") or {}),
                      parity=((body.get("data") or {}).get("parity")
                              or api.ok("GET", f"/api/probes/{pid}").get("parity")))
    else:
        record["error"] = body
    (out / f"parity-{label}.json").write_text(json.dumps(record, indent=1, default=str))
    print(json.dumps({k: record.get(k) for k in ("case", "arm_status", "success")}), flush=True)
    _clear(api)
    return record


def main(argv: Optional[list[str]] = None) -> int:  # pragma: no cover - needs the pod
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("command", choices=["cards", "sweep", "split", "parity", "report"])
    parser.add_argument("--model-key", default="fp16", help="hw_common.ROWS key to register")
    parser.add_argument("--template", type=Path)
    parser.add_argument("--sae-template", type=Path)
    parser.add_argument("--adapt", help="hf_id,d_model,n_layers: re-point the template (dense)")
    parser.add_argument("--ns", default="1,8,16,32,64")
    parser.add_argument("--kinds", default="dense,sparse,mixed")
    parser.add_argument("--spreads", default="one,many")
    parser.add_argument("--n-layers", type=int, default=16)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--out", default="probe_scaling")
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("files", nargs="*", type=Path)
    args = parser.parse_args(argv)
    from hw_common import OUT, ROWS, seed_rows

    out = OUT / args.out
    out.mkdir(parents=True, exist_ok=True)
    if args.command == "cards":
        print("\n".join(_cards()))
        return 0
    if args.command == "report":
        records = [json.loads(p.read_text()) for p in (args.files or sorted(out.glob("*.json")))
                   if not p.name.startswith("parity-")]
        for r in records:
            print(f"{r['case']}: projected {r['projected_decode_ms']:.4f} observed "
                  f"{r['observed_decode_ms']:.4f} ratio {r['ratio_decode']}")
        bad = findings(records)
        print(f"FINDINGS (> {RATIO_LIMIT}x): {[r['case'] for r in bad]}")
        if args.baseline:
            base = [json.loads(p.read_text()) for p in sorted(args.baseline.glob("*.json"))
                    if not p.name.startswith("parity-")]
            print(f"REGRESSION vs baseline: {regression(base, records)}")
        return 0
    seed_rows([args.model_key])
    model = ROWS[args.model_key]["name"]
    templates = {"dense": json.loads(args.template.read_text())}
    if args.adapt:
        hf_id, d_model, n_layers = args.adapt.split(",")
        templates["dense"] = adapt(templates["dense"], hf_id=hf_id, d_model=int(d_model),
                                   n_layers=int(n_layers))
    templates["sparse"] = (json.loads(args.sae_template.read_text()) if args.sae_template
                           else templates["dense"])
    kinds = args.kinds.split(",") if args.sae_template else ["dense"]
    cards = ["all"] if args.command == "split" else _cards()
    api = _api()
    try:
        for key, value in RAISED.items():
            _set(api, key, value)
        for card in cards:
            _load_on(api, model, card)
            if args.command == "parity":
                parity_control(api, model, json.loads(args.template.read_text()), out, "dense")
                if args.sae_template:
                    parity_control(api, model, templates["sparse"], out, "sparse")
                continue
            for case in case_grid(model, [card], [int(x) for x in args.ns.split(",")],
                                  kinds, args.spreads.split(",")):
                run_case(api, case, templates, out, args.repeats, args.max_tokens, args.n_layers)
    finally:
        _clear(api)
        for key in RAISED:
            _reset(api, key)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
