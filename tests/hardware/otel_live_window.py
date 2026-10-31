#!/usr/bin/env python3
"""Feature 038 13.2 – 13.5 and 12.5 on the LIVE miLLM, in A-B-A form (ENH-006 D75 – D79).

Why this file exists beside `otel_acceptance.py`. The 13.1 harness was written for a throwaway
bench pod that switches telemetry in-process, per block of 10 (A-B-B-A). Since D75 the live miLLM
switches export at runtime through its gated Settings route, and every agent change waits for the
operator (D76). Interleaving would need ~64 approvals, so the operator chose A-B-A (D79): every
cell with telemetry OFF (A1), ONE approved change turns it ON, every cell again (B), ONE approved
change turns it OFF, then a drift sample with it off (A2). This driver runs that design against
the live server. It reuses the reviewed harness's pure parts (prompts, bootstrap, relay, report),
and adds only what the live server needs:

* it NEVER switches telemetry, loads, unloads or changes a setting. Every request carries
  `X-miLLM-Load-Policy: refuse` AND `X-miLLM-Agent`, so anything gated is queued for the operator,
  never executed. It WAITS (polling) for the operator's approved state — the resident model and
  the telemetry state its phase needs — and verifies that state from `/api/health/detailed`
  before every block; a block that finds the wrong state aborts the run;
* the relay (`otel_acceptance.Relay`) runs in this process on the pod's loopback, forwards every
  OTLP body byte-identically to the reference Collector and timestamps the Collector's 2xx; the
  live `TELEMETRY_ENDPOINT` points at it for B only (the approved ON change). The B process keeps
  the relay up until telemetry reads OFF again, so the OFF drain also reaches the Collector;
* the 13.5 outage closes the relay (connection refused, as a dead receiver): no cluster change;
* the enqueue is NOT timed: the product does not time itself and the bench-only timer cannot be
  installed in the live process. The enqueue target is reported `not_measured`, never guessed.

Run INSIDE the live backend container (`kubectl exec`), from a copy under /tmp. Phases:
  calibrate  --model M          prompt word counts per (model, length), shared by every phase
  phase      --phase A1|B|A2 --plan model:set,set;model:set ...   (cells run in plan order)
             B adds: --outage-after <model> (13.5 on that model, nine-probe set) and --at-on <model>
  merge      A1 + B -> cell-telemetry-*.json (the 13.1 report's format) + drift.json (A1 vs A2)
All times written are US Eastern (ET).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import otel_acceptance as oa  # noqa: E402
from hw_common import now_et  # noqa: E402

AGENT_HEADERS = {"X-miLLM-Agent": "agent-6b-038-live-window"}
ARM_OF_PHASE = {"A1": "off", "B": "on", "A2": "off"}


class LiveClient(oa.Client):
    def __init__(self, base: str, timeout: float = 1800.0) -> None:
        super().__init__(base, timeout)
        self.http.headers.update(AGENT_HEADERS)


def log(msg: str) -> None:
    print(f"[{now_et()}] {msg}", flush=True)


def server_pid() -> int | None:
    """The miLLM server process in this container (for RSS), by its command line."""
    for d in Path("/proc").iterdir():
        if not d.name.isdigit() or int(d.name) == os.getpid():
            continue
        try:
            cmd = (d / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
        except OSError:
            continue
        if ("uvicorn" in cmd or "millm.main" in cmd) and "otel_live_window" not in cmd:
            return int(d.name)
    return None


def rss_kib(pid: int | None) -> int | None:
    if pid is None:
        return None
    try:
        for line in Path(f"/proc/{pid}/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1])
    except OSError:
        return None
    return None


def gpu_all() -> dict[str, int]:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=uuid,memory.used",
                              "--format=csv,noheader,nounits"], capture_output=True, text=True,
                             timeout=20).stdout
    except (OSError, subprocess.SubprocessError):
        return {}
    return oa.parse_smi_memory(out)


def gpu_used(card: str) -> int | None:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=uuid,memory.used",
                              "--format=csv,noheader,nounits"], capture_output=True, text=True,
                             timeout=20).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    return oa.parse_smi_memory(out).get(card)


def state_problem(client: oa.Client, model: str, arm: str, relay_url: str | None,
                  receiver_down_ok: bool = False) -> str | None:
    """None when the server is in the state this phase needs; otherwise what differs.
    `receiver_down_ok`: right after an outage the signals read `unavailable` until their next
    successful export; export is still ON (enabled, endpoint the relay), which is what matters."""
    health = client.health()
    if not (health.get("model_loaded") and health.get("model_name") == model):
        return f"resident is {health.get('model_name') if health.get('model_loaded') else None!r}, need {model!r}"
    tel = health.get("telemetry") or {}
    if receiver_down_ok and arm == "on":
        if not tel.get("enabled"):
            return f"telemetry reads enabled={tel.get('enabled')}"
        return oa.relay_mismatch(tel, relay_url) if relay_url else None
    problem = oa.arm_mismatch(arm, tel)
    if problem:
        return problem
    if arm == "on" and relay_url:
        return oa.relay_mismatch(tel, relay_url)
    return None


def wait_state(client: oa.Client, model: str, arm: str, relay_url: str | None,
               timeout_s: float) -> None:
    """Poll until the operator's approved state is in force (never acts to get there)."""
    deadline = time.monotonic() + timeout_s
    last = None
    while True:
        problem = state_problem(client, model, arm, relay_url)
        if problem is None:
            log(f"state ok: {model} resident, telemetry {arm}")
            return
        if problem != last:
            log(f"waiting for the operator's approved state: {problem}")
            last = problem
        if time.monotonic() > deadline:
            raise SystemExit(f"TIMEOUT waiting for state ({problem})")
        time.sleep(30)


def block_state(client: oa.Client, pid: int | None) -> dict[str, Any]:
    tel = (client.health() or {}).get("telemetry") or {}
    return {
        "queue_used": {k: ((v or {}).get("queue") or {}).get("used")
                       for k, v in (tel.get("signals") or {}).items()},
        "signals": {k: {f: (v or {}).get(f) for f in ("state", "exported", "failed", "dropped")}
                    for k, v in (tel.get("signals") or {}).items()},
        "series_overflow": tel.get("series_overflow"),
        "rss_kib": rss_kib(pid),
    }


def run_block(client: oa.Client, model: str, prompt: str, stream: bool, n: int, backlog: int,
              max_tokens: int, phase: str, arm: str, block: int, card: str,
              pid: int | None) -> dict[str, Any]:
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=1 + backlog) as ex:
        reqs = list(ex.map(lambda _i: client.chat(model, prompt, stream, max_tokens), range(n)))
    wall = time.perf_counter() - t0
    for r in reqs:
        r.pop("entries", None)
    wrong = {r.get("model") for r in reqs if r.get("ok") and r.get("model") not in (None, model)}
    if wrong:
        raise SystemExit(f"REFUSED: responses named another model {sorted(wrong)}")
    return {"phase": phase, "arm": arm, "block": block, "wall_s": wall,
            "tokens": sum(int(r.get("completion_tokens") or 0) for r in reqs if r.get("ok")),
            "requests": reqs, "gpu_used_mib": gpu_used(card), "gpu_all_mib": gpu_all(),
            "at_et": now_et(),
            **block_state(client, pid)}


def parse_plan(text: str) -> list[tuple[str, list[str]]]:
    out = []
    for part in filter(None, text.split(";")):
        model, sets = part.rsplit(":", 1)
        out.append((model, sets.split(",")))
    return out


def load_sets(args: argparse.Namespace) -> dict[str, dict[str, list[str]]]:
    """{model served name: {set name: [probe names]}} from `--sets model=probe_sets.json,...`."""
    out = {}
    for item in args.sets.split(","):
        model, path = item.split("=", 1)
        out[model] = json.loads(Path(path).read_text())["sets"]
    return out


def cmd_calibrate(args: argparse.Namespace, client: oa.Client, out: Path) -> int:
    path = out / "prompts.json"
    prompts = json.loads(path.read_text()) if path.exists() else {}
    wait_state(client, args.model, "off", None, args.wait_s)
    for length in args.prompts.split(","):
        key = f"{args.model}|{length}"
        if key in prompts:
            continue
        cell = oa.Cell(args.model, "none", length, False)
        words, measured = oa.calibrate_prompt(client, cell, oa.initial_words(cell.prompt_tokens))
        check = client.chat(args.model, oa.make_prompt(words, length), False, max_tokens=1)
        prompts[key] = {"words": words, "first_measured": measured,
                        "prompt_tokens": check.get("prompt_tokens"), "at_et": now_et()}
        log(f"calibrated {key}: {prompts[key]}")
    oa.write_json(path, prompts)
    return 0


def run_cell(args: argparse.Namespace, client: oa.Client, cell: oa.Cell, phase: str,
             prompts: dict[str, Any], names: list[str], relay: oa.Relay | None,
             pid: int | None, measured: int) -> dict[str, Any]:
    arm = ARM_OF_PHASE[phase]
    key = f"{cell.model}|{cell.prompt}"
    if key not in prompts:
        if phase != "A1":
            raise SystemExit(f"REFUSED: no calibrated prompt for {key}; every phase must reuse A1's")
        calib = oa.Cell(cell.model, "none", cell.prompt, False)
        words, measured_tokens = oa.calibrate_prompt(client, calib, oa.initial_words(calib.prompt_tokens))
        check = client.chat(cell.model, oa.make_prompt(words, cell.prompt), False, max_tokens=1)
        prompts[key] = {"words": words, "first_measured": measured_tokens,
                        "prompt_tokens": check.get("prompt_tokens"), "at_et": now_et()}
        oa.write_json(Path(args.out) / "prompts.json", prompts)
        log(f"calibrated {key}: {prompts[key]}")
    words = prompts[key]["words"]
    prompt = oa.make_prompt(words, cell.prompt)
    ids = oa.ensure_armed(client, names)
    blocks = []
    plan = [("warmup", -1, args.warmup)] if args.warmup else []
    plan += [("measured", b, args.block) for b in range(measured // args.block)]
    for kind, b, n in plan:
        problem = state_problem(client, cell.model, arm, relay.url if relay else None)
        if problem:
            raise SystemExit(f"ABORT {cell.slug} {phase}: state changed under the run: {problem}")
        blk = run_block(client, cell.model, prompt, cell.stream, n, args.backlog, args.max_tokens,
                        kind, arm, b, args.card, pid)
        blocks.append(blk)
        log(f"{phase} {cell.slug} {kind} block {b}: {len(blk['requests'])} req, "
            f"{blk['wall_s']:.1f} s, {blk['tokens']} tok, failed "
            f"{sum(1 for r in blk['requests'] if not r.get('ok'))}")
    doc: dict[str, Any] = {"cell": {**oa.asdict(cell), "slug": cell.slug}, "phase": phase,
                           "arm": arm, "probe_ids": ids, "prompt_words": words, "blocks": blocks,
                           "placement": (client.health() or {}).get("model_placement"),
                           "card": args.card, "at_et": now_et()}
    if relay is not None:
        rids = [rid for blk in blocks if blk["phase"] == "measured" for r in blk["requests"]
                for rid in r.get("result_ids", [])]
        relay.wait_for(rids, timeout_s=15.0)
        arr = relay.arrivals()
        doc["arrivals"] = {rid: arr[rid] for rid in rids if rid in arr}
        doc["expected_result_ids"] = len(rids)
    return doc


def cmd_phase(args: argparse.Namespace, client: oa.Client, out: Path) -> int:
    phase = args.phase
    arm = ARM_OF_PHASE[phase]
    prompts = json.loads((out / "prompts.json").read_text()) if (out / "prompts.json").exists() else {}
    sets = load_sets(args)
    pid = server_pid()
    log(f"phase {phase} (telemetry {arm}); server pid {pid}")
    relay = oa.Relay(args.relay_port, args.collector).start() if arm == "on" else None
    if relay:
        log(f"relay {relay.url} -> {args.collector}")
    try:
        for model, set_names in parse_plan(args.plan):
            wait_state(client, model, arm, relay.url if relay else None, args.wait_s)
            mcr = oa.env_value(client.ok("GET", "/api/environment"), "MAX_CONCURRENT_REQUESTS")
            if mcr != 1:
                raise SystemExit(f"REFUSED: MAX_CONCURRENT_REQUESTS is {mcr!r}; needs 1")
            for cell in oa.cell_grid([model], set_names, args.prompts.split(","),
                                     [x == "stream" for x in args.streams.split(",")]):
                path = out / f"phase-{phase}-{cell.slug}.json"
                if path.exists():
                    log(f"present, skipped: {path.name}")
                    continue
                doc = run_cell(args, client, cell, phase, prompts, sets[model][cell.probe_set],
                               relay, pid, args.measured)
                oa.write_json(path, doc)
            if relay and args.at_on == model:
                cmd_at_live(args, client, relay, model, sets[model], out)
            if relay and args.outage_after == model:
                run_outage(args, client, relay, model, sets[model], prompts, out, pid)
        if relay:
            log("phase B done: keeping the relay up until telemetry reads OFF (the OFF request)")
            while True:
                tel = (client.health() or {}).get("telemetry") or {}
                if oa.arm_mismatch("off", tel) is None:
                    log("telemetry reads OFF; relay closing")
                    break
                time.sleep(15)
            oa.write_json(out / "relay-B.json", {
                "requests_by_path": relay.requests_by_path, "upstream_status": relay.upstream_status,
                "results": len(relay.results), "lifecycle": len(relay.lifecycle),
                "spans": relay.span_count, "metric_points": len(relay.metric_points),
                "at_et": now_et()})
    finally:
        if relay:
            relay.stop()
    return 0


def run_outage(args: argparse.Namespace, client: oa.Client, relay: oa.Relay, model: str,
               sets: dict[str, list[str]], prompts: dict[str, Any], out: Path,
               pid: int | None) -> None:
    """13.5: a receiver outage (the relay closes: connection refused) under traffic."""
    path = out / "outage.json"
    if path.exists():
        log("outage present, skipped")
        return
    cell = oa.Cell(model, "nine", "short", False)
    prompt = oa.make_prompt(prompts[f"{model}|short"]["words"], "short")
    doc: dict[str, Any] = {"model": model, "probe_set": "nine", "mode": "relay",
                           "outage_s": args.outage_s, "telemetry": {}, "prometheus": {},
                           "phases": {}, "completed": False}
    oa.ensure_armed(client, sets["nine"])
    try:
        doc["telemetry"]["start"] = oa._telemetry(client)
        doc["prometheus"]["start"] = oa._prometheus_probe_lines(client)
        for phase, seconds in (("before", args.before_s), ("outage", args.outage_s),
                               ("after", args.after_s)):
            problem = state_problem(client, model, "on", relay.url, receiver_down_ok=True)
            if problem and phase != "outage":
                raise SystemExit(f"ABORT outage: {problem}")
            if phase == "outage":
                doc["telemetry"]["before"] = oa._telemetry(client)
                doc["prometheus"]["before"] = oa._prometheus_probe_lines(client)
                relay.stop()
                doc["outage_began_et"] = now_et()
                log("outage: relay closed (receiver refuses connections)")
            reqs: list[dict[str, Any]] = []
            t_end = time.monotonic() + seconds
            while time.monotonic() < t_end:
                with ThreadPoolExecutor(max_workers=1 + args.backlog) as ex:
                    reqs.extend(ex.map(lambda _i: oa._safe_chat(client, cell, prompt,
                                                                args.max_tokens),
                                       range(1 + args.backlog)))
            if phase == "outage":
                doc["telemetry"]["during_end"] = oa._telemetry(client)
                relay.start()
                doc["outage_ended_et"] = now_et()
                log("outage over: relay listening again")
            doc["phases"][phase] = {"requests": len(reqs),
                                    "failed": [r for r in reqs if not r.get("ok")][:50],
                                    "n_failed": sum(1 for r in reqs if not r.get("ok")),
                                    "at_et": now_et(), "rss_kib": rss_kib(pid)}
            log(f"outage phase {phase}: {len(reqs)} requests, {doc['phases'][phase]['n_failed']} failed")
        deadline = time.monotonic() + args.drain_timeout_s
        while time.monotonic() < deadline:
            sig = oa._telemetry(client).get("signals") or {}
            if all(((v or {}).get("queue") or {}).get("used", 0) == 0 for v in sig.values()):
                break
            time.sleep(1.0)
        doc["telemetry"]["after"] = oa._telemetry(client)
        doc["prometheus"]["after"] = oa._prometheus_probe_lines(client)
        doc["requests"] = sum(p["requests"] for p in doc["phases"].values())
        doc["failures_by_phase"] = {k: v["n_failed"] for k, v in doc["phases"].items()}
        doc["failures_attributable"] = (doc["phases"]["outage"]["n_failed"]
                                        + doc["phases"]["after"]["n_failed"])
        doc["valid"] = doc["phases"]["before"]["n_failed"] == 0
        doc["completed"] = True
    finally:
        if not relay.running:
            relay.start()
        oa.write_json(path, doc)


def cmd_at_live(args: argparse.Namespace, client: oa.Client, relay: oa.Relay, model: str,
                sets: dict[str, list[str]], out: Path) -> None:
    """12.5's result halves that the approved ON state can run (no further settings change):
    AT02, AT04, AT07 (not-permitted half), AT10 (chat, stream, credential headers; score route
    with export NOT permitted), AT12. AT08 needs sampler 0 and AT07's permitted half / AT10's
    score-export half need stateless export permitted: each is a further telemetry approval, so
    they are recorded `not_run` with that reason, never passed."""
    path = out / "at.json"
    if path.exists():
        log("at present, skipped")
        return
    probe_ids = oa.ensure_armed(client, sets["two"])
    checks: dict[str, Any] = {}
    prompt = oa.make_prompt(40, "at")
    wait_s, settle_s = 20.0, 8.0

    def chat(**kw: Any) -> dict[str, Any]:
        return client.chat(model, kw.pop("content", prompt), kw.pop("stream", False),
                           kw.pop("max_tokens", 16), **kw)

    # AT02
    trace, span = oa._new_trace()
    known = chat(headers={"traceparent": f"00-{trace}-{span}-01"})
    missing = chat()
    malformed = chat(headers={"traceparent": "00-zzzz-yyyy-01"})
    pairs = [oa._new_trace() for _ in range(4)]
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
    ok_missing = (bool(missing["result_ids"]) and len(t_missing) == 1 and trace not in t_missing
                  and exported_traces(missing) == t_missing)
    t_mal = traces_of(malformed)
    ok_mal = (bool(malformed["result_ids"]) and len(t_mal) == 1
              and all(t and oa.re.fullmatch(r"[0-9a-f]{32}", t) for t in t_mal)
              and exported_traces(malformed) == t_mal)
    ok_inter = all(bool(r["result_ids"]) and traces_of(r) == {p[0]} == exported_traces(r)
                   for r, p in zip(inter, pairs, strict=True))
    checks["AT02"] = {"result": "pass" if ok_known and ok_missing and ok_mal and ok_inter else "fail",
                      "facts": {"known": ok_known, "missing": ok_missing, "malformed": ok_mal,
                                "interleaved": ok_inter, "results": len(ids),
                                "exported": sum(1 for i in ids if i in rec)}}
    # AT04
    done = chat(stream=True, max_tokens=24)
    cut = chat(stream=True, max_tokens=oa.OUTPUT_TOKENS, stop_after_chunks=2)
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

    # AT07, not-permitted half
    def stateless_records() -> int:
        with relay.lock:
            return sum(1 for v in relay.results.values()
                       if v["attributes"].get("mi.probe.origin") == "stateless")

    def newest_event() -> Any:
        _, j, _ = client.call("GET", "/api/probes/events?limit=1")
        data = j.get("data") or []
        return data[0].get("id") if data else None

    score_body = {"probe_ids": list(probe_ids), "inputs": [{"text": prompt}], "export_results": True}
    ev0, st0 = newest_event(), stateless_records()
    st_code, s1, _ = client.call("POST", "/api/probes/score", score_body)
    time.sleep(settle_s)
    st1, ev1 = stateless_records(), newest_event()
    n1 = len(oa.find_result_ids(s1.get("data")))
    ok07 = st_code < 300 and n1 > 0 and st1 == st0 and ev1 == ev0
    checks["AT07"] = {"result": "pass (not-permitted half only)" if ok07 else "fail",
                      "facts": {"score_status": st_code, "score_results": n1,
                                "exported_when_not_permitted": st1 - st0,
                                "live_events_created": ev1 != ev0,
                                "export_summary": (s1.get("data") or {}).get("export")
                                if isinstance(s1.get("data"), dict) else None,
                                "permitted_half": "not_run: needs TELEMETRY_STATELESS_EXPORT_"
                                                  "PERMITTED=true, a further telemetry approval"}}
    checks["AT08"] = {"result": "not_run",
                      "facts": {"why": "needs TELEMETRY_TRACES_SAMPLER_RATIO=0, a further "
                                       "telemetry approval (the window had two: ON and OFF)"}}
    # AT10
    sentinels = oa._sentinels(6)
    relay.capture_raw = True
    start = len(relay.raw)
    chat(content=f"{prompt} {sentinels[0]}",
         headers={"Authorization": f"Bearer {sentinels[3]}", "X-Api-Key": sentinels[4]})
    chat(content=f"{prompt} {sentinels[1]}", stream=True, max_tokens=8)
    client.call("POST", "/api/probes/score", {
        "probe_ids": list(probe_ids), "export_results": True,
        "inputs": [{"text": f"{prompt} {sentinels[2]}"},
                   {"messages": [{"role": "user", "content": sentinels[5]}]}]})
    time.sleep(settle_s + 6.0)
    with relay.lock:
        raw = b"".join(relay.raw[start:])
    relay.capture_raw = False
    hits = oa.sentinel_hits(raw, sentinels)
    checks["AT10"] = {"result": ("pass (chat, stream, headers; score route not exported)"
                                 if raw and not any(hits.values()) else "fail"),
                      "facts": {"bytes_searched": len(raw), "hits": sum(hits.values()),
                                "score_export": "not permitted in this window: the score "
                                                "route's export half is not run"}}
    # AT12
    tpl_path = Path(args.at_template) if args.at_template else None
    if tpl_path is None:
        checks["AT12"] = {"result": "not_run", "facts": {"why": "no --at-template"}}
    else:
        doc = json.loads(tpl_path.read_text())
        doc["name"] = "otelbench-mismatch-" + "".join(oa.random.choice(oa.string.ascii_lowercase)
                                                      for _ in range(6))
        doc["model"] = {**doc["model"], "hf_id": "otelbench/not-the-resident-model",
                        "revision": "main"}
        doc.pop("identity", None)
        n0 = len(relay.lifecycle)
        status, j, _ = client.call("POST", "/api/probes/import?on_conflict=rename", doc)
        pid = (j.get("data") or {}).get("id") if status < 300 else None
        code = None
        deleted = None
        if pid:
            log(f"AT12 imported throwaway probe {pid}")
            _, ja, _ = client.call("POST", f"/api/probes/{pid}/arm", {"windows": []})
            code = None if ja.get("success") else (ja.get("error") or {}).get("code")
            deleted, _, _ = client.call("DELETE", f"/api/probes/{pid}")
            log(f"AT12 deleted throwaway probe {pid}: {deleted}")
        time.sleep(settle_s)
        with relay.lock:
            recs = [r["attributes"] for r in relay.lifecycle[n0:]]
        refused = [r for r in recs if r.get("mi.lifecycle.action") == "arm_refused"
                   or r.get("mi.lifecycle.outcome") == "refused"]
        ok12 = bool(pid) and bool(code) and any(r.get("mi.lifecycle.code") == code for r in refused)
        checks["AT12"] = {"result": "pass" if ok12 else "fail",
                          "facts": {"import_status": status, "probe_id": pid, "arm_code": code,
                                    "delete_status": deleted,
                                    "lifecycle_actions": [r.get("mi.lifecycle.action")
                                                          for r in recs]}}
    for name, why in oa.AT_MANUAL.items():
        checks[name] = {"result": "manual", "facts": {"why": why}}
    oa.write_json(path, {"model": model, "probe_set": "two", "at_et": now_et(), "checks": checks})
    for name, res in sorted(checks.items()):
        log(f"{name}: {res['result']} {json.dumps(res['facts'], default=str)[:240]}")


# -- merge (A1 + B -> the 13.1 report format; A1 vs A2 drift) -----------------------------------


def _lat(doc: dict[str, Any]) -> list[float]:
    return [r["latency_ms"] for b in doc["blocks"] if b["phase"] == "measured"
            for r in b["requests"] if r.get("ok")]


def _measured(doc: dict[str, Any]) -> list[dict[str, Any]]:
    return [b for b in doc["blocks"] if b["phase"] == "measured"]


def cmd_merge(args: argparse.Namespace, out: Path) -> int:
    ev = Path(args.evidence)
    ev.mkdir(parents=True, exist_ok=True)
    docs: dict[str, dict[str, dict[str, Any]]] = {}
    for p in sorted(out.glob("phase-*.json")):
        d = json.loads(p.read_text())
        docs.setdefault(d["cell"]["slug"], {})[d["phase"]] = d
    drift = []
    for slug, by in sorted(docs.items()):
        if "A1" in by and "B" in by:
            a, b = by["A1"], by["B"]
            blocks = ([{**blk, "arm": "off"} for blk in a["blocks"]]
                      + [{**blk, "arm": "on"} for blk in b["blocks"]])
            oa.write_json(ev / f"cell-telemetry-{slug}.json", {
                "cell": a["cell"], "kind": "telemetry", "arms": ["off", "on"],
                "arm_switch": "live A-B-A (D79)", "prompt_words": a["prompt_words"],
                "blocks": blocks, "arrivals": b.get("arrivals") or {},
                "arrival_measured": "arrivals" in b, "at_et": b["at_et"],
                "a_at_et": a["at_et"]})
        if "A1" in by and "A2" in by:
            a, c = by["A1"], by["A2"]
            lat_d = oa.bootstrap_relative_delta(_lat(a), _lat(c), oa.p95)
            tp_d = oa.bootstrap_relative_delta(_measured(a), _measured(c), oa.block_throughput)
            drift.append({"slug": slug, "a1_p95_ms": oa.p95(_lat(a)), "a2_p95_ms": oa.p95(_lat(c)),
                          "p95_delta": list(lat_d), "throughput_delta": list(tp_d),
                          "a1_n": len(_lat(a)), "a2_n": len(_lat(c)),
                          "verdict": oa.noise_verdict(lat_d), "a1_at_et": a["at_et"],
                          "a2_at_et": c["at_et"]})
    oa.write_json(ev / "drift.json", drift)
    for name in ("outage.json", "at.json", "profile.json"):
        if (out / name).exists():
            (ev / name).write_text((out / name).read_text())
    print(json.dumps(drift, indent=1, default=str))
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument("command", choices=["calibrate", "phase", "merge"])
    p.add_argument("--base", default="http://127.0.0.1:8000")
    p.add_argument("--out", default="/tmp/otel038/out")
    p.add_argument("--evidence", default="/tmp/otel038/evidence")
    p.add_argument("--card", default="GPU-247aa582-0d1b-e161-8156-983ed1fefc57")
    p.add_argument("--model")
    p.add_argument("--phase", choices=list(ARM_OF_PHASE))
    p.add_argument("--plan", help="model:set,set;model:set,...  (cells in this order)")
    p.add_argument("--sets", help="model=probe_sets.json,model=probe_sets.json")
    p.add_argument("--prompts", default="short,long")
    p.add_argument("--streams", default="nostream,stream")
    p.add_argument("--warmup", type=int, default=oa.WARMUP)
    p.add_argument("--measured", type=int, default=oa.MEASURED_PER_ARM)
    p.add_argument("--block", type=int, default=oa.BLOCK)
    p.add_argument("--backlog", type=int, default=oa.BACKLOG)
    p.add_argument("--max-tokens", type=int, default=oa.OUTPUT_TOKENS)
    p.add_argument("--relay-port", type=int, default=oa.DEFAULT_RELAY_PORT)
    p.add_argument("--collector", default="http://otel-collector.millm-otel.svc:4318")
    p.add_argument("--wait-s", type=float, default=4 * 3600.0)
    p.add_argument("--outage-after", help="B: run 13.5 after this model's cells")
    p.add_argument("--at-on", help="B: run the AT halves after this model's cells")
    p.add_argument("--at-template", help="AT12: a definition to re-point at a missing model")
    p.add_argument("--before-s", type=float, default=120.0)
    p.add_argument("--outage-s", type=float, default=600.0)
    p.add_argument("--after-s", type=float, default=300.0)
    p.add_argument("--drain-timeout-s", type=float, default=120.0)
    args = p.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.command == "merge":
        return cmd_merge(args, out)
    client = LiveClient(args.base)
    if args.command == "calibrate":
        return cmd_calibrate(args, client, out)
    return cmd_phase(args, client, out)


if __name__ == "__main__":
    sys.exit(main())
