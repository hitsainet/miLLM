#!/usr/bin/env python3
"""Feature 32 hardware acceptance — run INSIDE the miLLM pod, NOT by the unit suite.

Checkpoints, load and lineage (032_FTASKS 11.2–11.5; BRD-01 acceptance 4, 9, 11, 12). Instantiate
(acceptance 13, 14) is deferred (R-01.26) and not exercised. Every subcommand prints a JSON verdict
per check (also appended to $HWTEST_OUT/verdicts.jsonl) and exits non-zero on a failed criterion.

Subcommands:
  save          11.2 first half (acceptance 4): one bracketed step on LFM2.5-1.2B, save at that
                step; `weights_sha256` equals a fresh SHA-256 of the file; records the save-time
                log-probabilities (explicit token set + top 20) to $HWTEST_OUT/acc4_saved.json
  load-fresh    11.2 second half: run in a FRESHLY STARTED miLLM (restart the pod's server first);
                load the saved checkpoint; log-probabilities within 1e-2 of the save-time run
  tiny-save     a differently shaped checkpoint (a random 2-layer Llama) for acceptance 11
  inplace       11.3 (acceptance 11, 12): attached process, in-place load, its log-probabilities vs
                miLLM's; the tiny checkpoint refused while attached; a byte written into the
                checkpoint's weights file DURING the in-place copy leaves `torn`, every /v1 call
                refused, then an in-place load of a good checkpoint recovers
  refusals      11.4 (acceptance 9): save refused while `writing` and while `torn`
Timings (11.5) are in each verdict: snapshot_ms / write_ms (save), verify_ms / write_ms (load).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hw_common import (  # noqa: E402
    LFM_DIR, OPERATOR, OUT, PROMPT, ROWS, TRAINER, Api, Skeleton, compare_logprobs, hash_tensors,
    rebuild, verdict,
)

FP16 = ROWS["fp16"]["name"]
CKPT_ROOT = Path(os.environ.get("WEIGHTS_CHECKPOINT_DIR", "/hw/data/checkpoints"))
SAVED = OUT / "acc4_saved.json"


def _file_sha256(path: Path) -> tuple[str, float]:
    t0 = time.perf_counter()
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(16 << 20), b""):
            h.update(chunk)
    return h.hexdigest(), (time.perf_counter() - t0) * 1000


def _scores(api: Api, model: str, explicit: list[int]) -> dict[str, Any]:
    s = api.score(model, PROMPT, allowed=explicit, top=len(explicit))
    f = api.score(model, PROMPT, allowed=None, top=20)
    return {"explicit": {str(k): v for k, v in s["top"].items()}, "chosen": f["chosen"],
            "top20": {str(k): v for k, v in f["top"].items()}, "model": f["model"],
            "x_millm_model": f["x_millm_model"]}


def _max_diff(a: dict, b: dict) -> float:
    keys = set(a) & set(b)
    if keys != set(a):
        return float("inf")
    return max(abs(a[k] - b[k]) for k in keys)


def _step(api: Api, att: str, tensors: dict, name: str, scale: float) -> dict:
    """One bracketed write: scale one parameter (a visible, real change)."""
    api.ok("POST", "/api/weights/quiesce", token=TRAINER, attachment=att)
    api.ok("POST", "/api/weights/commit-begin", token=TRAINER, attachment=att)
    tensors[name].mul_(scale)
    import torch

    torch.cuda.synchronize()
    ce = api.ok("POST", "/api/weights/commit-end", token=TRAINER, attachment=att)
    api.ok("POST", "/api/weights/resume", token=TRAINER, attachment=att)
    return ce


def _reload_base(api: Api) -> None:
    """The bare base row, reloaded through the weights API (D-01), so the resident is the file's
    weights and not a checkpoint a save renamed it to or a step left behind."""
    r = api.call("POST", "/api/weights/load", token=OPERATOR,
                 body={"model_id": api.model_row(FP16)["id"]})
    if r[0] != 200:
        raise RuntimeError(f"base reload: {r[0]} {r[1]}")


def _attach(api: Api) -> tuple[str, dict]:
    body = api.ok("POST", "/api/weights/attach", token=TRAINER)
    return body["attachment_id"], body


def _save(api: Api, att: str, run_id: str, step: int) -> tuple[int, Any]:
    st, j, _ = api.call("POST", "/api/weights/save", token=TRAINER, attachment=att, body={
        "manifest": {"run_id": run_id, "step": step, "domain": "hardware acceptance",
                     "signal_source": "corpus", "parent": None,
                     "optimizer": {"name": "none", "state": "fresh", "config_hash": "0"}}})
    return st, j


def cmd_save(a: argparse.Namespace) -> int:
    api = Api()
    st0 = api.status()
    if st0["attached"] or st0["weights_state"] != "clean":
        raise RuntimeError(f"not clean: {st0}")
    if api.resident() is None:
        api.load(FP16)
    _reload_base(api)
    att, body = _attach(api)
    try:
        tensors = rebuild(body)
        ce = _step(api, att, tensors, a.param, a.scale)
        skel = Skeleton(LFM_DIR, tensors)
        import torch

        full, _, _ = skel.logprobs(PROMPT)
        explicit = list(dict.fromkeys([int(i) for i in torch.topk(full, 10).indices]
                                      + [1, 7, 100, 1000, 30000]))
        before = _scores(api, FP16, explicit)  # the base row's name until the save renames it
        st, j = _save(api, att, a.run_id, a.step)
        if st != 201 and st != 200:
            return verdict("acceptance_4_save_11_2", False, status=st, body=j)
        d = j["data"]
        weights = CKPT_ROOT / d["path"] / "model.safetensors"
        fresh, hash_ms = _file_sha256(weights)
        after_name = [m["id"] for m in api.ok("GET", "/v1/models")]
        res = {"commit_end": ce, "save": d, "fresh_sha256": fresh,
               "hash_matches": fresh == d["weights_sha256"], "fresh_hash_ms_harness": hash_ms,
               "file_bytes": weights.stat().st_size, "v1_models_after_save": after_name,
               "status_last_save": api.status()["last_save"]}
        SAVED.parent.mkdir(parents=True, exist_ok=True)
        SAVED.write_text(json.dumps({"explicit_ids": explicit, "scores": before,
                                     "save": d, "at": time.time()}))
        ok = fresh == d["weights_sha256"] and d["identity_set"] is True
        return verdict("acceptance_4_save_11_2", ok, **res)
    finally:
        api.call("POST", "/api/weights/detach", token=TRAINER, attachment=att)


def cmd_load_fresh(a: argparse.Namespace) -> int:
    api = Api()
    saved = json.loads(SAVED.read_text())
    st0 = api.status()
    fresh_process = st0["resident_model"] is None and st0["steps_since_load"] is None
    t0 = time.perf_counter()
    st, j, _ = api.call("POST", "/api/weights/load", token=OPERATOR,
                        body={"path": saved["save"]["path"]})
    wall = (time.perf_counter() - t0) * 1000
    if st != 200:
        return verdict("acceptance_4_load_fresh_11_2", False, status=st, body=j)
    d = j["data"]
    name = d["name"]
    now = _scores(api, name, saved["explicit_ids"])
    de = _max_diff(saved["scores"]["explicit"], now["explicit"])
    dt = _max_diff(saved["scores"]["top20"], now["top20"])
    ok = (fresh_process and d["weights_sha256"] == saved["save"]["weights_sha256"]
          and now["chosen"] == saved["scores"]["chosen"] and de <= 1e-2 and dt <= 1e-2)
    return verdict("acceptance_4_load_fresh_11_2", ok, fresh_process=fresh_process,
                   status_before=st0, load=d, load_wall_ms=wall, served_name=name,
                   top1_saved=saved["scores"]["chosen"], top1_now=now["chosen"],
                   max_abs_diff_explicit_set=de, max_abs_diff_top20=dt, tolerance=1e-2,
                   x_millm_model=now["x_millm_model"])


def cmd_tiny_save(a: argparse.Namespace) -> int:
    api = Api()
    api.load(ROWS["tiny"]["name"])
    att, _ = _attach(api)
    try:
        st, j = _save(api, att, "hwtest-tiny", 1)
    finally:
        api.call("POST", "/api/weights/detach", token=TRAINER, attachment=att)
    api.unload(ROWS["tiny"]["name"])
    return verdict("tiny_checkpoint_for_acceptance_11", st in (200, 201), status=st,
                   body=j.get("data") or j.get("error"))


def _fire(window: list, stop: threading.Event, t0: float, model: str) -> None:
    c = Api(timeout=120)
    while not stop.is_set():
        s, j, h = c.chat(model, "Say hello.", max_tokens=4)
        code = (j.get("error") or {}).get("code") if isinstance(j, dict) else None
        window.append((round(time.monotonic() - t0, 2), s, code, h.get("Retry-After")))
        time.sleep(0.1)


def cmd_inplace(a: argparse.Namespace) -> int:
    import torch

    api = Api()
    saved = json.loads(SAVED.read_text())
    good = saved["save"]["path"]
    _reload_base(api)  # the checkpoint differs from the base by one scaled parameter
    st0 = api.status()
    if st0["attached"] or st0["weights_state"] != "clean":
        raise RuntimeError(f"not clean: {st0}")
    att, body = _attach(api)
    res: dict[str, Any] = {}
    oks: list[bool] = []
    try:
        tensors = rebuild(body)
        addrs = {n: t.data_ptr() for n, t in tensors.items()}
        skel = Skeleton(LFM_DIR, tensors)
        h_base = hash_tensors(tensors)
        # Acceptance 11: in-place load while attached.
        st, j, _ = api.call("POST", "/api/weights/load", token=TRAINER, attachment=att,
                            body={"path": good})
        d = j.get("data") or {}
        name = d.get("name", FP16)
        cmp = compare_logprobs(api, name, skel, extra_ids=saved["explicit_ids"][10:])
        h_after = hash_tensors(tensors)
        moved = [n for n, t in tensors.items() if t.data_ptr() != addrs[n]]
        ok11 = (st == 200 and d.get("mode") == "in_place" and cmp["pass"] and not moved
                and h_after != h_base)
        res["acceptance_11_in_place"] = {"status": st, "load": d or j, "served_name": name,
                                         "handles_read_new_values": h_after != h_base,
                                         "addresses_moved": moved, **cmp}
        oks.append(ok11)
        # Differently shaped checkpoint while attached: refused, nothing changed.
        tiny = sorted((CKPT_ROOT / "hwtest-tiny").glob("step-*"))
        st2, j2, _ = api.call("POST", "/api/weights/load", token=TRAINER, attachment=att,
                              body={"path": f"hwtest-tiny/{tiny[-1].name}"})
        err2 = j2.get("error") or {}
        h2 = hash_tensors(tensors)
        ok11b = st2 == 409 and err2.get("code") == "ALREADY_ATTACHED" and h2 == h_after
        res["acceptance_11_shape_refused"] = {"status": st2, "code": err2.get("code"),
                                              "message": err2.get("message"),
                                              "weights_unchanged": h2 == h_after,
                                              "state": api.status()["weights_state"]}
        oks.append(ok11b)
        # Acceptance 12: a second checkpoint, loaded in place, with one byte of its weights file
        # rewritten while the copy runs (an injected failure the fingerprint re-check catches).
        _step(api, att, tensors, a.param, 1.5)
        st3, j3 = _save(api, att, f"hwtest-inject-{int(time.time())}", 1)
        if st3 not in (200, 201):
            raise RuntimeError(f"inject save: {st3} {j3}")
        inj_path = j3["data"]["path"]
        _step(api, att, tensors, a.param, 1 / 1.5)
        target = CKPT_ROOT / inj_path / "model.safetensors"
        os.chmod(target, 0o644)
        result: dict[str, Any] = {}

        def do_load() -> None:
            result["r"] = Api().call("POST", "/api/weights/load", token=TRAINER, attachment=att,
                                     body={"path": inj_path})

        th = threading.Thread(target=do_load)
        th.start()
        injected_at = None
        t_start = time.monotonic()
        while th.is_alive() and time.monotonic() - t_start < 120:
            if api.status()["weights_state"] == "writing":
                fd = os.open(target, os.O_WRONLY)
                try:
                    size = os.fstat(fd).st_size
                    os.pwrite(fd, b"\x00", size - 1)
                finally:
                    os.close(fd)
                injected_at = round(time.monotonic() - t_start, 3)
                break
            time.sleep(0.002)
        th.join()
        st4, j4, _ = result["r"]
        torn_status = api.status()
        window: list[Any] = []
        stop = threading.Event()
        t0 = time.monotonic()
        # The served name is the one the save gave the resident: a torn load never renames it.
        inj_name = j3["data"]["name"]
        firer = threading.Thread(target=_fire, args=(window, stop, t0, inj_name), daemon=True)
        firer.start()
        time.sleep(a.window_s)
        stop.set()
        firer.join(timeout=60)
        answered = [w for w in window if w[1] == 200]
        refused_torn = [w for w in window if w[1] == 503 and w[2] == "weights_torn" and w[3]]
        # Recover in place (the holder, while torn; D-01 recovery by load), from the good file.
        st5, j5, _ = api.call("POST", "/api/weights/load", token=TRAINER, attachment=att,
                              body={"path": good})
        rec = api.status()
        v1 = api.chat((j5.get("data") or {}).get("name", FP16), "Say hello.", max_tokens=4)
        ok12 = (injected_at is not None and st4 >= 400 and torn_status["weights_state"] == "torn"
                and window and not answered and len(refused_torn) == len(window) and st5 == 200 and rec["weights_state"] == "clean"
                and v1[0] == 200)
        res["acceptance_12_interrupted"] = {
            "injected_s_after_load_start": injected_at, "load_status": st4,
            "load_error": j4.get("error"), "state_after": torn_status["weights_state"],
            "served_name_in_window": inj_name,
            "v1_calls_in_window": len(window), "answered_200": len(answered),
            "refused_503_weights_torn_with_retry_after": len(refused_torn),
            "codes": sorted({(w[1], w[2], w[3]) for w in window}),
            "recovery_in_place": [st5, j5.get("data") or j5.get("error")],
            "state_after_recovery": rec["weights_state"], "v1_after_recovery": v1[0]}
        oks.append(ok12)
    finally:
        api.call("POST", "/api/weights/detach", token=TRAINER, attachment=att)
    return verdict("acceptance_11_12_inplace_11_3", all(oks), parts_pass=oks, **res)


def cmd_refusals(a: argparse.Namespace) -> int:
    api = Api()
    _reload_base(api)
    att, _ = _attach(api)
    res = {}
    try:
        api.ok("POST", "/api/weights/quiesce", token=TRAINER, attachment=att)
        cb = api.ok("POST", "/api/weights/commit-begin", token=TRAINER, attachment=att)
        st_w, j_w = _save(api, att, "hwtest-refused", 1)
        time.sleep(cb["watchdog_ms"] / 1000 + 0.75)
        state = api.status()["weights_state"]
        st_t, j_t = _save(api, att, "hwtest-refused", 2)
        api.ok("POST", "/api/weights/commit-end", token=TRAINER, attachment=att)
        api.ok("POST", "/api/weights/resume", token=TRAINER, attachment=att)
        nothing = not (CKPT_ROOT / "hwtest-refused").exists()
        res = {"while_writing": [st_w, (j_w.get("error") or {}).get("code"),
                                 (j_w.get("error") or {}).get("details")],
               "state_second": state,
               "while_torn": [st_t, (j_t.get("error") or {}).get("code"),
                              (j_t.get("error") or {}).get("details")],
               "nothing_written": nothing, "state_after": api.status()["weights_state"]}
        ok = (st_w == 409 and st_t == 409 and state == "torn" and nothing
              and (j_w.get("error") or {}).get("code") == "WEIGHTS_NOT_CLEAN"
              and (j_t.get("error") or {}).get("code") == "WEIGHTS_NOT_CLEAN")
    finally:
        api.call("POST", "/api/weights/detach", token=TRAINER, attachment=att)
    return verdict("acceptance_9_refusals_11_4", ok, **res)


def cmd_timing(a: argparse.Namespace) -> int:
    """11.5: N saves and N in-place loads of LFM2.5-1.2B (FPRD §8: save hold <= 1,000 ms,
    in-place load hold <= 3,000 ms). Each save's directory is removed after its load."""
    import shutil

    from hw_common import summary

    api = Api()
    _reload_base(api)
    att, _ = _attach(api)
    snap, write, verify, hold, fresh_hash = [], [], [], [], []
    try:
        stamp = int(time.time())
        for i in range(a.n):
            st, j = _save(api, att, f"hwtest-timing-{stamp}", i + 1)
            if st not in (200, 201):
                raise RuntimeError(f"save {i}: {st} {j}")
            d = j["data"]
            snap.append(d["snapshot_ms"])
            write.append(d["write_ms"])
            _, ms = _file_sha256(CKPT_ROOT / d["path"] / "model.safetensors")
            fresh_hash.append(ms)
            st, j, _ = api.call("POST", "/api/weights/load", token=TRAINER, attachment=att,
                                body={"path": d["path"]})
            if st != 200 or j["data"]["mode"] != "in_place":
                raise RuntimeError(f"load {i}: {st} {j}")
            verify.append(j["data"]["verify_ms"])
            hold.append(j["data"]["write_ms"])
        shutil.rmtree(CKPT_ROOT / f"hwtest-timing-{stamp}")
    finally:
        api.call("POST", "/api/weights/detach", token=TRAINER, attachment=att)
    out = {"save_snapshot_hold_ms": summary(snap), "save_write_fsync_hash_publish_ms": summary(write),
           "harness_file_sha256_ms": summary(fresh_hash), "load_verify_ms": summary(verify),
           "load_in_place_hold_ms": summary(hold), "raw": {"snapshot": snap, "write": write,
                                                          "verify": verify, "hold": hold},
           "targets": {"save_hold_ms": 1000, "in_place_hold_ms": 3000},
           "save_hold_within_target": max(snap) <= 1000,
           "in_place_hold_within_target": max(hold) <= 3000}
    return verdict("timings_11_5", True, **out)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("save")
    s.add_argument("--run-id", default="hwtest-rt")
    s.add_argument("--step", type=int, default=1)
    s.add_argument("--param", default="model.layers.8.operator_norm.weight")
    s.add_argument("--scale", type=float, default=0.5)
    sub.add_parser("load-fresh")
    sub.add_parser("tiny-save")
    s = sub.add_parser("inplace")
    s.add_argument("--param", default="model.layers.8.operator_norm.weight")
    s.add_argument("--window-s", type=float, default=5.0)
    sub.add_parser("refusals")
    s = sub.add_parser("timing")
    s.add_argument("--n", type=int, default=5)
    a = ap.parse_args(argv)
    return {"save": cmd_save, "load-fresh": cmd_load_fresh, "tiny-save": cmd_tiny_save,
            "inplace": cmd_inplace, "refusals": cmd_refusals, "timing": cmd_timing}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
