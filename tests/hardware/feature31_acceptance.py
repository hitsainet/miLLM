#!/usr/bin/env python3
"""Feature 31 hardware acceptance — run INSIDE the miLLM pod, NOT by the unit suite.

Weight attachment and the step barrier (031_FTASKS 0.2, 9.2–9.9; FTID §8.5). The trainer is this
process (or a child it spawns) in the SAME container as miLLM: addendum D-02, in-container
topology. Every subcommand prints a JSON verdict per check (also appended to
$HWTEST_OUT/verdicts.jsonl) and exits non-zero on a failed criterion.

Subcommands:
  seed        register the model rows in the pod's own database (FP16, Q4, GGUF, tiny Llama)
  evidence    9.3: GPU UUIDs, kernel module and user-space driver versions, kernel, torch/CUDA,
              trainer placement, NVreg_EnableGpuFirmware=0
  spike       0.2 / 9.9: the attachable load (D-03) — `_share_cuda_` works on an attachable load
              and is refused on an ordinary one; the parameter pool holds parameters only (D2);
              allocations land in the pool from the loader's worker thread (R31 8e); a second
              process in the same container rebuilds one parameter, writes a value, and the loader
              process reads it back
  handles     9.2 / acceptance 1: attach, rebuild every parameter, compare next-token
              log-probabilities with miLLM's (top-1 identical; explicit token set within 1e-2)
  watchdog    9.4: N bracketed full-parameter in-place updates under concurrent chat load;
              commit_ms p50/p99/p99.9/max from the responses and (separately) the audit log
  step        9.5 / acceptance 2, 6, 8 (+ D7): one parameter zeroed inside the bracket; a hold
              timeout with every parameter's SHA-256 unchanged; a commit held past the watchdog
  kill        9.6 / acceptance 3, 7, 10 (+ D8): kill a trainer with no commit open, then
              mid-commit; every /v1 call in the torn window refused; recover with a row load
  limbo       9.9 / D8: a SIGKILLed trainer vs one that detaches and exits; the test server's own
              GPU memory after unload against a clean load/unload baseline
  quantized   9.7 / acceptance 5: attach on Q4 bitsandbytes and on GGUF → 400 NOT_ATTACHABLE
  restart-prep / restart-verify   9.8: leave an attachment held and the set torn; restart the
              pod (outside); then status is clean, nothing attached, steps 0 after a load

Examples (in the pod):
  python /hw/src/tests/hardware/feature31_acceptance.py evidence
  python /hw/src/tests/hardware/feature31_acceptance.py watchdog --n 10000
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import signal
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hw_common import (  # noqa: E402
    LFM_DIR, OPERATOR, OUT, ROWS, TRAINER, Api, Skeleton, compare_logprobs, gpu_used_mib,
    hash_tensors, now_et, rebuild, seed_rows, sh, summary, verdict, write_tiny_llama,
)

FP16 = ROWS["fp16"]["name"]
CHAT_PROMPT = "Write two sentences about the history of the city of Lyon."


def _attach(api: Api, token: str = TRAINER) -> dict:
    return api.ok("POST", "/api/weights/attach", token=token)


def _detach(api: Api, att: str, token: str = TRAINER) -> Any:
    return api.call("POST", "/api/weights/detach", token=token, attachment=att)


def _ensure_clean_fp16(api: Api) -> dict:
    st = api.status()
    if st["attached"]:
        raise RuntimeError(f"something is attached: {st['attachment']}")
    if st["weights_state"] != "clean":
        raise RuntimeError(f"weights_state is {st['weights_state']}")
    return api.load(FP16)


# --------------------------------------------------------------------------------------------


def cmd_seed(a: argparse.Namespace) -> int:
    write_tiny_llama()
    ids = seed_rows(["fp16", "q4", "gguf", "tiny"])
    return verdict("seed", True, rows=ids)


def cmd_evidence(a: argparse.Namespace) -> int:
    import torch

    facts = {
        "nvidia_smi_L": sh("nvidia-smi -L"),
        "proc_driver_nvidia_version": sh("cat /proc/driver/nvidia/version"),
        "nvidia_smi_driver": sh("nvidia-smi --query-gpu=driver_version,name,uuid,pci.bus_id "
                                "--format=csv,noheader"),
        "uname_r": sh("uname -r"),
        "torch": torch.__version__, "torch_cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "device_uuid_seen_by_torch": str(torch.cuda.get_device_properties(0).uuid),
        "trainer_placement": "in-container (D-02), same container as miLLM, same card",
        "NVIDIA_VISIBLE_DEVICES": os.environ.get("NVIDIA_VISIBLE_DEVICES"),
        "hostname": sh("hostname"),
    }
    # NVreg_EnableGpuFirmware is a host file; the pod cannot see /etc/modprobe.d. The operator
    # session records it from the node (`a.host_firmware_evidence`), and the module parameter as
    # loaded is readable here when /proc/driver/nvidia/params is present.
    facts["proc_driver_nvidia_params_EnableGpuFirmware"] = sh(
        "grep -i EnableGpuFirmware /proc/driver/nvidia/params")
    ok = "247aa582-0d1b-e161-8156-983ed1fefc57" in facts["nvidia_smi_L"]
    return verdict("evidence_9_3", ok, **facts)


# --------------------------------------------------------------------------------------------
# 0.2 / 9.9: the attachable-load spike (standalone, in this container)

_CHILD = r"""
import pickle, sys, torch
from torch.multiprocessing.reductions import rebuild_cuda_tensor
args = pickle.load(open(sys.argv[1], "rb"))
t = rebuild_cuda_tensor(*args)
before = float(t[0])
t[0] = 123.0
torch.cuda.synchronize()
print(f"child: before={before} wrote=123.0 now={float(t[0])}", flush=True)
"""


def _pool_report(loaded: Any) -> dict:
    import torch

    pools = getattr(loaded, "attachable_mem_pools", []) or []
    pool_ids = {tuple(p.id) for p in pools}
    snap = torch.cuda.memory_snapshot()
    in_pool = [s for s in snap if tuple(s.get("segment_pool_id", (0, 0))) in pool_ids]
    model = loaded.model
    ranges = []
    for _, p in model.named_parameters(remove_duplicate=True):
        st = p.untyped_storage()
        ranges.append((st.data_ptr(), st.data_ptr() + st.nbytes()))
    buf_ranges = []
    for name, b in model.named_buffers():
        if b.is_cuda:
            st = b.untyped_storage()
            buf_ranges.append((name, st.data_ptr(), st.data_ptr() + st.nbytes()))

    def owner(addr: int) -> str:
        if any(lo <= addr < hi for lo, hi in ranges):
            return "parameter"
        for name, lo, hi in buf_ranges:
            if lo <= addr < hi:
                return f"buffer:{name}"
        return "other"

    blocks = {"parameter": 0, "other": 0}
    buffers: dict[str, int] = {}
    other_bytes = 0
    for seg in in_pool:
        for blk in seg["blocks"]:
            if blk["state"] != "active_allocated":
                continue
            o = owner(blk["address"])
            if o.startswith("buffer:"):
                buffers[o] = buffers.get(o, 0) + 1
            else:
                blocks[o] += 1
                if o == "other":
                    other_bytes += blk["size"]
    seg_ranges = [(s["address"], s["address"] + s["total_size"]) for s in in_pool]
    params_outside = sum(1 for lo, _ in ranges
                         if not any(a <= lo < b for a, b in seg_ranges))
    return {"pools": len(pools), "pool_segments": len(in_pool),
            "pool_bytes": sum(s["total_size"] for s in in_pool),
            "active_blocks_parameter": blocks["parameter"], "active_blocks_buffer": buffers,
            "active_blocks_other": blocks["other"], "other_bytes": other_bytes,
            "parameter_storages": len(ranges), "parameter_storages_outside_pool": params_outside}


def cmd_spike(a: argparse.Namespace) -> int:
    import concurrent.futures as cf

    import millm  # noqa: F401 - sets expandable_segments exactly as the server does
    import torch
    from millm.ml.model_loader import ModelLoadContext

    def _load(attachable: bool) -> Any:
        with ModelLoadContext(1, "spike") as ctx:
            return ctx.load(LFM_DIR, "FP16", attachable=attachable)

    facts: dict[str, Any] = {"alloc_env": {k: v for k, v in os.environ.items() if "ALLOC" in k}}
    ex = cf.ThreadPoolExecutor(1)  # the service loads on an executor thread, so does the spike
    loaded = ex.submit(_load, True).result()
    facts["attachable_allocation_flag"] = loaded.attachable_allocation
    facts["pool"] = _pool_report(loaded)
    name, param = next((n, p) for n, p in loaded.model.named_parameters() if p.dim() == 1)
    try:
        param.untyped_storage()._share_cuda_()
        facts["share_on_attachable"] = "ok"
    except Exception as exc:  # noqa: BLE001
        facts["share_on_attachable"] = f"{type(exc).__name__}: {exc}"

    # The second process, same container: rebuild one parameter, write, read back here.
    from torch.multiprocessing.reductions import reduce_tensor

    _, args = reduce_tensor(param.data)
    shm_before = sorted(os.listdir("/dev/shm"))
    with tempfile.TemporaryDirectory() as td:
        argf, childf = Path(td, "args.pkl"), Path(td, "child.py")
        pickle.dump(args, open(argf, "wb"))
        childf.write_text(_CHILD)
        shm_during = sorted(os.listdir("/dev/shm"))
        before = float(param.data[0])
        r = subprocess.run([sys.executable, "-I", str(childf), str(argf)], capture_output=True,
                           text=True, timeout=120)
        torch.cuda.synchronize()
        after = float(param.data[0])
    facts.update({
        "ipc_param": name, "parent_before": before, "parent_after": after,
        "child_rc": r.returncode, "child_out": r.stdout.strip(), "child_err": r.stderr.strip()[-800:],
        "dev_shm_before": shm_before, "dev_shm_during": shm_during,
        "dev_shm_after": sorted(os.listdir("/dev/shm")),
    })
    ipc_ok = r.returncode == 0 and after == 123.0
    param.data[0] = before

    # An ordinary load (expandable segments on). torch 2.14 EXPORTS it; what must fail is the
    # second process's rebuild (measured: "pidfd_getfd: Operation not permitted", the pod has no
    # ptrace capability), which is why attach refuses such a load (`allocator`).
    ordinary = ex.submit(_load, False).result()
    p2 = next(p for _, p in ordinary.model.named_parameters() if p.dim() == 1)
    try:
        p2.untyped_storage()._share_cuda_()
        facts["export_on_ordinary"] = "ok"
        _, args2 = reduce_tensor(p2.data)
        with tempfile.TemporaryDirectory() as td:
            argf, childf = Path(td, "args.pkl"), Path(td, "child.py")
            pickle.dump(args2, open(argf, "wb"))
            childf.write_text(_CHILD)
            r2 = subprocess.run([sys.executable, "-I", str(childf), str(argf)],
                                capture_output=True, text=True, timeout=120)
        facts["rebuild_on_ordinary"] = {"rc": r2.returncode, "out": r2.stdout.strip(),
                                        "err": r2.stderr.strip().splitlines()[-1:]}
        refused = r2.returncode != 0
    except Exception as exc:  # noqa: BLE001
        facts["export_on_ordinary"] = f"{type(exc).__name__}: {exc}"
        refused = True
    pool = facts["pool"]
    isolated = pool["active_blocks_other"] == 0 and pool["parameter_storages_outside_pool"] == 0
    ok = facts["share_on_attachable"] == "ok" and ipc_ok and refused and isolated
    return verdict("spike_0_2_9_9", ok, ipc_write_read_back=ipc_ok, ordinary_refused=refused,
                   pool_isolated=isolated, **facts)


# --------------------------------------------------------------------------------------------
# 9.2: acceptance 1


def cmd_handles(a: argparse.Namespace) -> int:
    api = Api()
    _ensure_clean_fp16(api)
    body = _attach(api)
    att = body["attachment_id"]
    try:
        tensors = rebuild(body)
        skel = Skeleton(LFM_DIR, tensors)
        res = compare_logprobs(api, FP16, skel, extra_ids=[1, 7, 100, 1000, 30000])
        shared = {p["name"]: p["storage"] for p in body["parameters"]}
        facts = {"parameters": len(body["parameters"]), "storages": len(body["storages"]),
                 "producer": body.get("producer"), "model": body.get("model"),
                 "missing_parameters_in_skeleton": skel.missing_parameters,
                 "missing_state_keys": skel.missing_state, "unexpected": skel.unexpected,
                 "tied_lm_head_shares_storage":
                     shared.get("lm_head.weight") == shared.get("model.embed_tokens.weight")
                     if "lm_head.weight" in shared else "lm_head not exported separately",
                 **res}
        ok = res["pass"] and not skel.missing_parameters
        return verdict("acceptance_1_handles_9_2", ok, **facts)
    finally:
        _detach(api, att)


# --------------------------------------------------------------------------------------------
# 9.4: watchdog measurement


class _Load:
    """Keeps /v1/chat/completions busy: `threads` clients in a loop."""

    def __init__(self, api_factory: Any, threads: int = 2, max_tokens: int = 16) -> None:
        self.stop = threading.Event()
        self.ok = 0
        self.bad: list[Any] = []
        self.lock = threading.Lock()
        self.threads = [threading.Thread(target=self._run, args=(api_factory(), max_tokens),
                                         daemon=True) for _ in range(threads)]

    def _run(self, api: Api, max_tokens: int) -> None:
        i = 0
        while not self.stop.is_set():
            i += 1
            status, j, _ = api.chat(FP16, f"{CHAT_PROMPT} ({i})", max_tokens=max_tokens)
            with self.lock:
                if status == 200:
                    self.ok += 1
                else:
                    self.bad.append((status, json.dumps(j)[:200]))

    def __enter__(self) -> "_Load":
        for t in self.threads:
            t.start()
        return self

    def __exit__(self, *exc: Any) -> None:
        self.stop.set()
        for t in self.threads:
            t.join(timeout=120)


class _Heartbeat:
    def __init__(self, att: str, token: str = TRAINER, every: float = 5.0) -> None:
        self.att, self.token, self.every = att, token, every
        self.stop = threading.Event()
        self.t = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        api = Api()
        while not self.stop.wait(self.every):
            api.call("POST", "/api/weights/heartbeat", token=self.token, attachment=self.att)

    def __enter__(self) -> "_Heartbeat":
        self.t.start()
        return self

    def __exit__(self, *exc: Any) -> None:
        self.stop.set()
        self.t.join(timeout=30)


def cmd_watchdog(a: argparse.Namespace) -> int:
    import torch

    api = Api()
    _ensure_clean_fp16(api)
    body = _attach(api)
    att = body["attachment_id"]
    server_ms: list[float] = []
    harness_ms: list[float] = []
    write_ms: list[float] = []
    quiesce_ms: list[float] = []
    overran = 0
    failures: list[Any] = []
    try:
        tensors = rebuild(body)
        uniq = list({t.untyped_storage().data_ptr(): t for t in tensors.values()}.values())
        params = [t for t in uniq]
        h0 = hash_tensors(tensors)
        snap = [t.clone() for t in params]
        torch.cuda.synchronize()
        with _Heartbeat(att), _Load(Api, threads=a.load_threads, max_tokens=a.load_tokens) as load:
            t_start = time.monotonic()
            for i in range(a.n):
                t0 = time.perf_counter()
                st, j, _ = api.call("POST", "/api/weights/quiesce", token=TRAINER, attachment=att)
                if st != 200:
                    failures.append(("quiesce", i, st, j))
                    continue
                t1 = time.perf_counter()
                quiesce_ms.append((t1 - t0) * 1000)
                st, j, _ = api.call("POST", "/api/weights/commit-begin", token=TRAINER,
                                    attachment=att)
                if st != 200:
                    failures.append(("commit-begin", i, st, j))
                    api.call("POST", "/api/weights/resume", token=TRAINER, attachment=att)
                    continue
                tb = time.perf_counter()
                torch._foreach_add_(params, a.delta)
                torch._foreach_copy_(params, snap)
                torch.cuda.synchronize()
                tw = time.perf_counter()
                st, j, _ = api.call("POST", "/api/weights/commit-end", token=TRAINER, attachment=att)
                te = time.perf_counter()
                if st != 200:
                    failures.append(("commit-end", i, st, j))
                else:
                    d = j["data"]
                    server_ms.append(float(d["commit_ms"]))
                    overran += bool(d["overran"])
                    harness_ms.append((te - tb) * 1000)
                    write_ms.append((tw - tb) * 1000)
                api.call("POST", "/api/weights/resume", token=TRAINER, attachment=att)
                if a.progress and (i + 1) % a.progress == 0:
                    print(json.dumps({"i": i + 1, "elapsed_s": round(time.monotonic() - t_start, 1),
                                      "server_p99_9_so_far": summary(server_ms)["p99_9"],
                                      "chats_ok": load.ok, "chats_bad": len(load.bad)}),
                          flush=True)
            wall = time.monotonic() - t_start
            chats_ok, chats_bad = load.ok, list(load.bad)
        h1 = hash_tensors(tensors)
    finally:
        _detach(api, att)
    srv = summary(server_ms)
    gate = None
    if server_ms:
        import math

        gate = int(math.ceil(10 * max(srv["p99_9"], summary(harness_ms)["p99_9"]) / 100.0) * 100)
    out = {"n_requested": a.n, "commits": len(server_ms), "failures": failures[:20],
           "failure_count": len(failures), "overran": overran,
           "commit_ms_server": srv, "commit_ms_harness": summary(harness_ms),
           "write_ms_harness": summary(write_ms), "quiesce_ms": summary(quiesce_ms),
           "wall_s": wall, "chats_ok": chats_ok, "chats_bad": len(chats_bad),
           "chats_bad_sample": chats_bad[:5], "load_threads": a.load_threads,
           "load_max_tokens": a.load_tokens, "parameters_written": len(params),
           "bytes_per_write": sum(t.numel() * t.element_size() for t in params),
           "sha256_before": h0, "sha256_after": h1, "weights_unchanged": h0 == h1,
           "suggested_watchdog_ms_10x_p99_9": gate,
           "delta": a.delta, "write": "foreach add_(delta) then copy_ from a snapshot (exact)"}
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / f"watchdog_{int(time.time())}.json", "w") as fh:
        json.dump({**out, "server_ms": server_ms, "harness_ms": harness_ms}, fh)
    ok = (len(server_ms) >= a.n and not failures and h0 == h1 and not chats_bad and overran == 0)
    return verdict("watchdog_9_4", ok, **out)


# --------------------------------------------------------------------------------------------
# 9.5: acceptance 2, 6, 8 (and D7)


def cmd_step(a: argparse.Namespace) -> int:
    import torch

    api = Api()
    _ensure_clean_fp16(api)
    body = _attach(api)
    att = body["attachment_id"]
    results: dict[str, Any] = {}
    oks: list[bool] = []
    try:
        tensors = rebuild(body)
        target = a.param
        p = tensors[target]
        original = p.clone()
        mt = a.max_tokens

        def text(r: tuple) -> Any:
            status, j, _ = r
            return j["choices"][0]["message"]["content"] if status == 200 else (status, j)

        base = text(api.chat(FP16, CHAT_PROMPT, max_tokens=mt))

        # Acceptance 2: in-flight completion unaffected, the next one changed.
        inflight: dict[str, Any] = {}

        def run_inflight() -> None:
            inflight["t0"] = time.monotonic()
            inflight["text"] = text(Api().chat(FP16, CHAT_PROMPT, max_tokens=mt))
            inflight["t1"] = time.monotonic()

        th = threading.Thread(target=run_inflight)
        th.start()
        time.sleep(a.inflight_delay)
        tq0 = time.monotonic()
        q = api.ok("POST", "/api/weights/quiesce", token=TRAINER, attachment=att)
        tq1 = time.monotonic()
        api.ok("POST", "/api/weights/commit-begin", token=TRAINER, attachment=att)
        p.zero_()  # immediately after quiesce returned (D7: the hold synchronized first)
        torch.cuda.synchronize()
        ce = api.ok("POST", "/api/weights/commit-end", token=TRAINER, attachment=att)
        api.ok("POST", "/api/weights/resume", token=TRAINER, attachment=att)
        th.join()
        after = text(api.chat(FP16, CHAT_PROMPT, max_tokens=mt))
        st2, j2, h2 = api.chat(FP16, CHAT_PROMPT, max_tokens=4)
        ok2 = (inflight["text"] == base and after != base
               and inflight["t0"] < tq0 and inflight["t1"] <= tq1 + 0.05)
        results["acceptance_2"] = {
            "zeroed": target, "numel": p.numel(), "base_text": base,
            "inflight_text_equals_base": inflight["text"] == base, "next_text": after,
            "next_differs": after != base,
            "inflight_started_before_quiesce_s": round(tq0 - inflight["t0"], 3),
            "inflight_finished_before_quiesce_returned": inflight["t1"] <= tq1 + 0.05,
            "quiesce_returned_s_after_inflight_finished": round(tq1 - inflight["t1"], 3),
            "quiesce": q, "commit_end": ce, "x_millm_model_after": h2.get("X-miLLM-Model")}
        oks.append(ok2)
        # Restore inside a bracket; the restored text must be the base text again.
        api.ok("POST", "/api/weights/quiesce", token=TRAINER, attachment=att)
        api.ok("POST", "/api/weights/commit-begin", token=TRAINER, attachment=att)
        p.copy_(original)
        torch.cuda.synchronize()
        api.ok("POST", "/api/weights/commit-end", token=TRAINER, attachment=att)
        api.ok("POST", "/api/weights/resume", token=TRAINER, attachment=att)
        restored = text(api.chat(FP16, CHAT_PROMPT, max_tokens=mt))
        results["acceptance_2"]["restored_equals_base"] = restored == base
        oks.append(restored == base)

        # Acceptance 6: a quiesce with no commit-begin self-resumes; every parameter unchanged.
        hold_s = float(a.hold_timeout_s)
        h0 = hash_tensors(tensors)
        api.ok("POST", "/api/weights/quiesce", token=TRAINER, attachment=att)
        held = api.status()
        time.sleep(hold_s + 2.0)
        st = api.status()
        c6 = api.chat(FP16, CHAT_PROMPT, max_tokens=4)
        h1 = hash_tensors(tensors)
        ok6 = (held["barrier"] == "quiesced" and st["barrier"] == "serving"
               and st["last_hold_timeout_at"] is not None and c6[0] == 200 and h0 == h1)
        results["acceptance_6"] = {"barrier_while_held": held["barrier"],
                                   "barrier_after": st["barrier"],
                                   "last_hold_timeout_at": st["last_hold_timeout_at"],
                                   "v1_after": c6[0], "sha256_before": h0, "sha256_after": h1,
                                   "unchanged": h0 == h1, "waited_s": hold_s + 2.0}
        oks.append(ok6)

        # Acceptance 8: a commit held open past the watchdog, then ended.
        wd = int(api.ok("POST", "/api/weights/quiesce", token=TRAINER, attachment=att) and 0)
        cb = api.ok("POST", "/api/weights/commit-begin", token=TRAINER, attachment=att)
        wd = int(cb["watchdog_ms"])
        time.sleep(wd / 1000 + 0.75)
        mid = api.status()
        mid_v1 = api.chat(FP16, CHAT_PROMPT, max_tokens=4)
        end = api.ok("POST", "/api/weights/commit-end", token=TRAINER, attachment=att)
        api.ok("POST", "/api/weights/resume", token=TRAINER, attachment=att)
        final = api.status()
        c8 = api.chat(FP16, CHAT_PROMPT, max_tokens=4)
        ok8 = (mid["weights_state"] == "torn" and mid_v1[0] == 503
               and end["weights_state"] == "clean" and end["overran"] is True
               and final["last_overrun"] is not None
               and float(final["last_overrun"].get("commit_ms", 0)) >= wd and c8[0] == 200)
        results["acceptance_8"] = {"watchdog_ms": wd, "state_mid": mid["weights_state"],
                                   "v1_mid": [mid_v1[0], (mid_v1[1].get("error") or {}).get("code")
                                              if isinstance(mid_v1[1], dict) else None,
                                              mid_v1[2].get("Retry-After")],
                                   "commit_end": end, "last_overrun": final["last_overrun"],
                                   "commit_overran": final["commit_overran"],
                                   "state_after": final["weights_state"], "v1_after": c8[0]}
        oks.append(ok8)
    finally:
        _detach(api, att)
    return verdict("acceptance_2_6_8_step_9_5", all(oks), parts_pass=oks, **results)


# --------------------------------------------------------------------------------------------
# 9.6: acceptance 3, 7, 10 (and D8). The trainer is a CHILD process, so it can be SIGKILLed.

_TRAINER = r"""
import json, os, sys, time, signal
sys.path.insert(0, sys.argv[1])
from hw_common import Api, rebuild
api = Api()
token, mode, out = sys.argv[2], sys.argv[3], sys.argv[4]
body = api.ok("POST", "/api/weights/attach", token=token)
att = body["attachment_id"]
tensors = rebuild(body)            # handles held open in this process until it dies
info = {"attachment_id": att, "parameters": len(tensors), "pid": os.getpid()}
if mode == "commit":
    api.ok("POST", "/api/weights/quiesce", token=token, attachment=att)
    api.ok("POST", "/api/weights/commit-begin", token=token, attachment=att)
    tensors[next(iter(tensors))].add_(0)   # a write inside the bracket
    info["commit_open"] = True
json.dump(info, open(out, "w"))
last = time.monotonic()
while True:                        # heartbeat only while idle; mid-commit it just waits to die
    time.sleep(1)
    if mode == "idle" and time.monotonic() - last > 5:
        api.call("POST", "/api/weights/heartbeat", token=token, attachment=att)
        last = time.monotonic()
"""


def _spawn_trainer(mode: str, token: str = TRAINER) -> tuple[subprocess.Popen, dict]:
    td = tempfile.mkdtemp()
    script, out = Path(td, "trainer.py"), Path(td, "info.json")
    script.write_text(_TRAINER)
    proc = subprocess.Popen([sys.executable, str(script), str(Path(__file__).resolve().parent),
                             token, mode, str(out)])
    t0 = time.monotonic()
    while not out.exists():
        if proc.poll() is not None:
            raise RuntimeError(f"trainer exited {proc.returncode}")
        if time.monotonic() - t0 > 180:
            raise TimeoutError("trainer did not attach")
        time.sleep(0.2)
    time.sleep(0.2)
    return proc, json.loads(out.read_text())


def _wait(pred: Any, timeout: float, every: float = 0.5) -> bool:
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        if pred():
            return True
        time.sleep(every)
    return False


def cmd_kill(a: argparse.Namespace) -> int:
    api = Api()
    row = _ensure_clean_fp16(api)
    base_text = api.chat(FP16, CHAT_PROMPT, max_tokens=24)[1]["choices"][0]["message"]["content"]
    mem = {"before_trainer": gpu_used_mib()}
    res: dict[str, Any] = {}
    oks: list[bool] = []
    hb_timeout = float(a.heartbeat_timeout_s)

    # Acceptance 3: killed with no commit open.
    proc, info = _spawn_trainer("idle")
    mem["trainer_attached"] = gpu_used_mib()
    os.kill(proc.pid, signal.SIGKILL)
    proc.wait()
    mem["after_kill_idle"] = gpu_used_mib()
    serving = [api.chat(FP16, CHAT_PROMPT, max_tokens=4)[0] for _ in range(3)]
    stale = _wait(lambda: (api.status().get("attachment") or {}).get("state") == "stale",
                  hb_timeout + 15)
    st = api.status()
    serving_stale = [api.chat(FP16, CHAT_PROMPT, max_tokens=4)[0] for _ in range(3)]
    ok3 = stale and all(s == 200 for s in serving + serving_stale) and st["weights_state"] == "clean"
    res["acceptance_3"] = {"trainer_pid": info["pid"], "v1_right_after_kill": serving,
                           "became_stale": stale, "attachment": st["attachment"],
                           "weights_state": st["weights_state"], "v1_while_stale": serving_stale}
    oks.append(ok3)
    cl = api.call("POST", "/api/weights/clear-stale", token=OPERATOR)
    res["clear_stale_idle"] = [cl[0], cl[1].get("data")]
    mem["after_clear_idle"] = gpu_used_mib()

    # Acceptance 7 and 10: killed mid-commit.
    proc, info = _spawn_trainer("commit")
    os.kill(proc.pid, signal.SIGKILL)
    proc.wait()
    t_kill = time.monotonic()
    mem["after_kill_commit"] = gpu_used_mib()
    window: list[Any] = []
    stop = threading.Event()

    def fire() -> None:
        c = Api(timeout=120)
        while not stop.is_set():
            s, j, h = c.chat(FP16, CHAT_PROMPT, max_tokens=4)
            code = (j.get("error") or {}).get("code") if isinstance(j, dict) else None
            window.append((round(time.monotonic() - t_kill, 2), s, code, h.get("Retry-After")))
            time.sleep(0.2)

    firer = threading.Thread(target=fire, daemon=True)
    firer.start()
    torn = _wait(lambda: api.status()["weights_state"] == "torn", 30)
    st_torn = api.status()
    stale2 = _wait(lambda: (api.status().get("attachment") or {}).get("state") == "stale",
                   hb_timeout + 15)
    cl2 = api.call("POST", "/api/weights/clear-stale", token=OPERATOR)
    after_clear = api.status()
    re = api.call("POST", "/api/weights/attach", token=OPERATOR)
    re_att = (re[1].get("data") or {}).get("attachment_id")
    re_state = (re[1].get("data") or {}).get("weights_state")
    det = _detach(api, re_att, OPERATOR) if re_att else (None, None, None)
    after_detach = api.status()
    time.sleep(2)
    stop.set()
    firer.join(timeout=150)
    answered = [w for w in window if w[1] == 200]
    refused_torn = [w for w in window if w[1] == 503 and w[2] in ("weights_torn", "WEIGHTS_TORN")]
    with_retry_after = [w for w in refused_torn if w[3]]
    mem["torn_before_recovery"] = gpu_used_mib()
    # Recover: POST /api/weights/load {"model_id": N} (D-01).
    rec = api.call("POST", "/api/weights/load", token=OPERATOR, body={"model_id": row["id"]})
    after_rec = api.status()
    rec_text = api.chat(FP16, CHAT_PROMPT, max_tokens=24)
    mem["after_recovery"] = gpu_used_mib()
    rec_ok = rec[0] == 200 and after_rec["weights_state"] == "clean" and rec_text[0] == 200
    ok7 = (torn and st_torn["weights_state"] == "torn" and after_clear["weights_state"] == "torn"
           and re_state == "torn" and after_detach["weights_state"] == "torn" and rec_ok)
    ok10 = (not answered and len(window) > 0 and len(refused_torn) > 0
            and len(with_retry_after) == len(refused_torn))
    res["acceptance_7"] = {
        "trainer_pid": info["pid"], "torn": torn, "status_torn": {
            k: st_torn[k] for k in ("weights_state", "commit", "attached", "attachment")},
        "became_stale": stale2, "clear_stale": [cl2[0], cl2[1].get("data")],
        "state_after_clear": after_clear["weights_state"], "reattach_status": re[0],
        "reattach_weights_state": re_state, "detach": det[0],
        "state_after_detach": after_detach["weights_state"],
        "recovery": [rec[0], rec[1].get("data") or rec[1].get("error")],
        "state_after_recovery": after_rec["weights_state"],
        "steps_after_recovery": after_rec["steps_since_load"],
        "text_after_recovery_equals_base": rec_text[0] == 200 and
            rec_text[1]["choices"][0]["message"]["content"] == base_text}
    res["acceptance_10"] = {"calls_in_window": len(window), "answered_200": len(answered),
                            "refused_503_weights_torn": len(refused_torn),
                            "with_retry_after": len(with_retry_after),
                            "other": [w for w in window if w[1] != 200 and w not in refused_torn][:10],
                            "first": window[:3], "last": window[-3:]}
    res["d8_gpu_memory_mib"] = mem
    oks += [ok7, ok10]
    return verdict("acceptance_3_7_10_kill_9_6", all(oks), parts_pass=oks, **res)


# --------------------------------------------------------------------------------------------
# 9.9 / D8: crashed-trainer IPC limbo


def _server_mib(live_pid: int, exclude: set[int]) -> int:
    """GPU memory of every compute process except the live miLLM's and our own children (the
    pod sees host PIDs in nvidia-smi; the test server is what remains)."""
    out = sh("nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits")
    total = 0
    for line in out.splitlines():
        pid, mib = (int(x) for x in line.split(","))
        if pid != live_pid and pid not in exclude:
            total += mib
    return total


def cmd_limbo(a: argparse.Namespace) -> int:
    """D8: kill a trainer with handles open; is the model's memory returned after clear-stale and
    unload? Control: a trainer that detaches and exits normally. Measured per process: the test
    server's own GPU memory (nvidia-smi), after an unload, against a clean load/unload baseline."""
    api = Api()
    if api.status()["attached"]:
        raise RuntimeError("something is attached")
    st = api.status()
    if st["resident_model"] is not None:
        api.unload(api.resident()["name"])
    out: dict[str, Any] = {"live_pid_excluded": a.live_pid}
    api.load(FP16)
    out["clean_loaded_mib"] = _server_mib(a.live_pid, set())
    api.unload(FP16)
    time.sleep(2)
    base = out["clean_unloaded_mib"] = _server_mib(a.live_pid, set())

    def run(mode: str) -> dict[str, Any]:
        api.load(FP16)
        loaded = _server_mib(a.live_pid, set())
        if mode == "graceful":
            r = subprocess.run([sys.executable, "-c", (
                "import sys; sys.path.insert(0, %r)\n"
                "from hw_common import Api, rebuild\n"
                "api = Api(); b = api.ok('POST', '/api/weights/attach', token=%r)\n"
                "t = rebuild(b)\n"
                "del t\n"
                "import gc, torch; gc.collect(); torch.cuda.ipc_collect()\n"
                "api.call('POST', '/api/weights/detach', token=%r, attachment=b['attachment_id'])\n"
                ) % (str(Path(__file__).resolve().parent), TRAINER, TRAINER)],
                capture_output=True, text=True, timeout=300)
            pid = None
            note = f"rc={r.returncode}"
        else:
            proc, info = _spawn_trainer("idle")
            pid = proc.pid
            os.kill(proc.pid, signal.SIGKILL)
            proc.wait()
            _wait(lambda: (api.status().get("attachment") or {}).get("state") == "stale",
                  a.heartbeat_timeout_s + 15)
            note = f"clear-stale {api.call('POST', '/api/weights/clear-stale', token=OPERATOR)[0]}"
        api.unload(FP16)
        time.sleep(2)
        after = _server_mib(a.live_pid, {pid} if pid else set())
        st = api.status()
        return {"loaded_mib": loaded, "after_unload_mib": after, "retained_mib": after - base,
                "note": note, "status_cuda_memory": st.get("cuda_memory"),
                "status_exports_since_start": st.get("exports_since_start")}

    for i, mode in enumerate(a.modes):
        out[f"{i}_{mode}"] = run(mode)
    kills = [v for k, v in out.items() if k.endswith("_killed")]
    graceful = [v for k, v in out.items() if k.endswith("_graceful")]
    out["limbo_confirmed"] = bool(kills) and all(k["retained_mib"] > 1000 for k in kills)
    # Each graceful cycle is measured against the clean baseline; a one-time constant (the first
    # export's own allocations) shows as the same small figure every time, a leak as growth.
    out["graceful_retained_mib"] = [g["retained_mib"] for g in graceful]
    out["graceful_returns_memory"] = bool(graceful) and max(out["graceful_retained_mib"]) < 1000
    return verdict("d8_ipc_limbo_9_9", out["graceful_returns_memory"], **out)


# --------------------------------------------------------------------------------------------
# 9.7: acceptance 5


def cmd_quantized(a: argparse.Namespace) -> int:
    api = Api()
    out: dict[str, Any] = {}
    oks = []
    for key, want in (("q4", "quantized"), ("gguf", "gguf")):
        name = ROWS[key]["name"]
        row = api.load(name)
        st, j, _ = api.call("POST", "/api/weights/attach", token=TRAINER)
        err = j.get("error") or {}
        reason = (err.get("details") or {}).get("reason")
        out[key] = {"model": name, "dtype": row.get("dtype"), "status": st,
                    "code": err.get("code"), "reason": reason, "message": err.get("message")}
        oks.append(st == 400 and err.get("code") == "NOT_ATTACHABLE" and reason == want)
        if st == 200:
            _detach(api, j["data"]["attachment_id"])
        api.unload(name)
    return verdict("acceptance_5_quantized_9_7", all(oks), **out)


# --------------------------------------------------------------------------------------------
# 9.8: restart


def cmd_restart_prep(a: argparse.Namespace) -> int:
    api = Api()
    _ensure_clean_fp16(api)
    body = _attach(api, OPERATOR)
    att = body["attachment_id"]
    api.ok("POST", "/api/weights/quiesce", token=OPERATOR, attachment=att)
    cb = api.ok("POST", "/api/weights/commit-begin", token=OPERATOR, attachment=att)
    torn = _wait(lambda: api.status()["weights_state"] == "torn", cb["watchdog_ms"] / 1000 + 10)
    st = api.status()
    return verdict("restart_prep_9_8", torn and st["attached"], status_before_restart={
        k: st[k] for k in ("attached", "weights_state", "barrier", "steps_since_load",
                           "resident_model", "commit_overran")}, at_et=now_et())


def cmd_restart_verify(a: argparse.Namespace) -> int:
    api = Api()
    st0 = api.status()
    api.load(FP16)
    st1 = api.status()
    ok = (st0["weights_state"] == "clean" and not st0["attached"] and st0["barrier"] == "serving"
          and st1["weights_state"] == "clean" and not st1["attached"]
          and st1["steps_since_load"] == 0)
    keep = ("attached", "attachment", "weights_state", "barrier", "steps_since_load",
            "last_overrun", "last_hold_timeout_at", "last_save", "commit_overran")
    return verdict("restart_verify_9_8", ok, status_after_restart={k: st0[k] for k in keep},
                   status_after_load={k: st1[k] for k in keep})


def cmd_recover(a: argparse.Namespace) -> int:
    """Return the test pod to a clean FP16 resident after an aborted run (harness utility)."""
    api = Api()
    st = api.status()
    if st["attached"]:
        api.call("POST", "/api/weights/clear-stale", token=OPERATOR)
    st = api.status()
    out: dict[str, Any] = {"before": {k: st[k] for k in ("weights_state", "attached", "barrier")}}
    if st["weights_state"] != "clean" or st["barrier"] != "serving":
        row = api.model_row(FP16)
        r = api.call("POST", "/api/weights/load", token=OPERATOR, body={"model_id": row["id"]})
        out["load"] = [r[0], r[1].get("data") or r[1].get("error")]
    st = api.status()
    out["after"] = {k: st[k] for k in ("weights_state", "attached", "barrier")}
    return verdict("recover_utility", st["weights_state"] == "clean", **out)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("seed", "evidence", "spike", "handles", "quantized", "restart-prep",
                 "restart-verify", "recover"):
        sub.add_parser(name)
    s = sub.add_parser("watchdog")
    s.add_argument("--n", type=int, default=10000)
    s.add_argument("--delta", type=float, default=1e-3)
    s.add_argument("--load-threads", type=int, default=2)
    s.add_argument("--load-tokens", type=int, default=16)
    s.add_argument("--progress", type=int, default=500)
    s = sub.add_parser("step")
    s.add_argument("--param", default="model.layers.8.operator_norm.weight")
    s.add_argument("--max-tokens", type=int, default=48)
    s.add_argument("--inflight-delay", type=float, default=0.25)
    s.add_argument("--hold-timeout-s", type=float, default=5.0)
    s = sub.add_parser("kill")
    s.add_argument("--heartbeat-timeout-s", type=float, default=30.0)
    s = sub.add_parser("limbo")
    s.add_argument("--modes", nargs="+", default=["graceful", "graceful", "killed"])
    s.add_argument("--live-pid", type=int, required=True,
                   help="host PID of the LIVE miLLM process (excluded from the measurement)")
    s.add_argument("--heartbeat-timeout-s", type=float, default=30.0)
    a = ap.parse_args(argv)
    fn = {"seed": cmd_seed, "evidence": cmd_evidence, "spike": cmd_spike,
          "handles": cmd_handles, "watchdog": cmd_watchdog, "step": cmd_step,
          "kill": cmd_kill, "quantized": cmd_quantized, "restart-prep": cmd_restart_prep,
          "restart-verify": cmd_restart_verify, "recover": cmd_recover, "limbo": cmd_limbo}[a.cmd]
    return fn(a)


if __name__ == "__main__":
    sys.exit(main())
