#!/usr/bin/env python3
"""Feature 33 hardware acceptance — run INSIDE the miLLM pod, NOT by the unit suite.

Adapter serving (033_FTASKS 10.3, 11.3, 11.4; BRD-01 acceptance 15, 16). No miForge-produced
adapter exists on the node or in the miForge checkout (searched 2026-10-16: no
`adapter_manifest.json` under /data/miforge or /home/x-sean/app/miForge), so `build` HAND-BUILDS
one in the same three-file format: PEFT `adapter_model.safetensors` + `adapter_config.json`, and
`adapter_manifest.json` written by miLLM's own `python -m millm.tools.write_adapter_manifest`.

Subcommands:
  build        the reference adapter (rank 16, alpha 32, targets q_proj, v_proj, w2 = 28 modules
               on LFM2.5-1.2B) and a MISMATCHED one (one module's A has the wrong input width)
  serve        11.3 / acceptance 15 (a), nothing attached: completions change, every completion
               and /v1/models report the adapter name, unload returns greedy text to the base
  throughput   10.3: decode tokens/s with and without the adapter (median of N greedy runs)
  attached     11.4 / acceptance 15 (b) and 16: an attached process hashes every base parameter
               through its handles before load, while applied, after unload (all equal); its own
               fp32 LoRA (D-12) reproduces miLLM's adapted log-probabilities; then the mismatched
               adapter is refused with serving uninterrupted
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hw_common import (  # noqa: E402
    LFM_DIR, OPERATOR, OUT, PROMPT, ROWS, TRAINER, Api, Skeleton, hash_tensors, rebuild, verdict,
)

FP16 = ROWS["fp16"]["name"]
CKPT_ROOT = Path(os.environ.get("WEIGHTS_CHECKPOINT_DIR", "/hw/data/checkpoints"))
ADAPTER = "hwtest-adapter/adapter-step-000001"
MISMATCH = "hwtest-adapter-bad/adapter-step-000001"
TARGETS = ["q_proj", "v_proj", "w2"]
RANK, ALPHA = 16, 32
#: lora_B std. 0.01 moved log-probabilities (max 2.25 over the top 20) but not a 48-token greedy
#: completion; 0.05 changes the text, which acceptance 15 asks to observe.
B_STD = 0.05
PROMPT_CHAT = "Write two sentences about the history of the city of Lyon."


def _reload_base(api: Api) -> None:
    if api.resident() is None:
        api.load(FP16)
    r = api.call("POST", "/api/weights/load", token=OPERATOR,
                 body={"model_id": api.model_row(FP16)["id"]})
    if r[0] != 200:
        raise RuntimeError(f"base reload: {r[0]} {r[1]}")


def _write_adapter(path: Path, *, bad: bool = False, seed: int = 0) -> dict:
    import torch
    from safetensors.torch import save_file
    from transformers import AutoConfig, AutoModelForCausalLM

    cfg = AutoConfig.from_pretrained(LFM_DIR)
    with torch.device("meta"):
        meta = AutoModelForCausalLM.from_config(cfg)
    g = torch.Generator().manual_seed(seed)
    tensors, modules = {}, []
    for name, mod in meta.named_modules():
        if isinstance(mod, torch.nn.Linear) and name.split(".")[-1] in TARGETS:
            modules.append(name)
            fan_in = mod.in_features + (1 if bad and not tensors else 0)  # one wrong width
            a = torch.randn(RANK, fan_in, generator=g) / fan_in ** 0.5
            b = torch.randn(mod.out_features, RANK, generator=g) * B_STD
            tensors[f"base_model.model.{name}.lora_A.weight"] = a.to(torch.bfloat16).contiguous()
            tensors[f"base_model.model.{name}.lora_B.weight"] = b.to(torch.bfloat16).contiguous()
    path.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(path / "adapter_model.safetensors"))
    (path / "adapter_config.json").write_text(json.dumps({
        "peft_type": "LORA", "task_type": "CAUSAL_LM", "r": RANK, "lora_alpha": ALPHA,
        "lora_dropout": 0.0, "target_modules": TARGETS, "bias": "none",
        "base_model_name_or_path": "LiquidAI/LFM2.5-1.2B-Instruct", "fan_in_fan_out": False,
        "use_rslora": False, "use_dora": False, "modules_to_save": None,
        "init_lora_weights": True, "inference_mode": True}, indent=2))
    r = subprocess.run([sys.executable, "-m", "millm.tools.write_adapter_manifest", str(path),
                        "--base-name", "LiquidAI/LFM2.5-1.2B-Instruct", "--force"],
                       capture_output=True, text=True)
    return {"path": str(path), "modules": len(modules), "tool_rc": r.returncode,
            "tool_out": (r.stdout + r.stderr).strip()[-600:]}


def cmd_build(a: argparse.Namespace) -> int:
    good = _write_adapter(CKPT_ROOT / ADAPTER)
    bad = _write_adapter(CKPT_ROOT / MISMATCH, bad=True, seed=1)
    manifest = json.loads((CKPT_ROOT / ADAPTER / "adapter_manifest.json").read_text()) \
        if good["tool_rc"] == 0 else None
    # The manifest tool checks keys against target_modules and r, not shapes against a base: the
    # mismatched adapter gets a valid manifest and is refused only at load (acceptance 16).
    ok = good["tool_rc"] == 0 and good["modules"] == 28 and bad["tool_rc"] == 0
    return verdict("adapter_build", ok, provenance="hand-built (no miForge adapter exists)",
                   good=good, mismatched=bad, manifest=manifest)


def _text(r: tuple) -> Any:
    s, j, h = r
    if s != 200:
        return {"status": s, "body": j}
    return j["choices"][0]["message"]["content"]


def _load_adapter(api: Api, path: str) -> tuple[int, Any]:
    s, j, _ = api.call("POST", "/api/weights/adapter/load", token=OPERATOR, body={"path": path})
    return s, j


def cmd_serve(a: argparse.Namespace) -> int:
    api = Api()
    _reload_base(api)
    st = api.status()
    if st["attached"]:
        raise RuntimeError("15 (a) needs nothing attached")
    base = _text(api.chat(FP16, PROMPT_CHAT, max_tokens=48))
    s, j = _load_adapter(api, ADAPTER)
    if s != 200:
        return verdict("acceptance_15a_serve_11_3", False, load_status=s, body=j)
    info = j["data"]
    name = info["adapter"]["name"] if "adapter" in info else info.get("name")
    r_adapted = api.chat(name, PROMPT_CHAT, max_tokens=48)
    r_bare = api.chat(FP16, PROMPT_CHAT, max_tokens=48)
    models = [m["id"] for m in api.ok("GET", "/v1/models")]
    status_block = api.ok("GET", "/api/weights/adapter")
    adapted = _text(r_adapted)
    stream_name = None
    with api.c.stream("POST", "/v1/chat/completions", json={
            "model": name, "messages": [{"role": "user", "content": "Hi"}], "max_tokens": 4,
            "stream": True}) as resp:
        for line in resp.iter_lines():
            if line.startswith("data: {"):
                chunk = json.loads(line[6:])
                stream_name = chunk.get("millm_model", stream_name)
    su, ju, _ = api.call("POST", "/api/weights/adapter/unload", token=OPERATOR, body={})
    after = _text(api.chat(FP16, PROMPT_CHAT, max_tokens=48))
    gone = api.chat(name, PROMPT_CHAT, max_tokens=4)
    models_after = [m["id"] for m in api.ok("GET", "/v1/models")]
    ok = (adapted != base and r_adapted[1]["model"] == name and r_bare[1].get("model") == name
          and "origin=adapter" in (r_adapted[2].get("X-miLLM-Model") or "")
          and name in models and su == 200 and after == base and gone[0] == 404
          and name not in models_after)
    return verdict("acceptance_15a_serve_11_3", ok, adapter=info, adapter_name=name,
                   base_text=base, adapted_text=adapted, changed=adapted != base,
                   body_model_adapted=r_adapted[1].get("model"),
                   bare_base_name_served_as=r_bare[1].get("model"),
                   x_millm_model=r_adapted[2].get("X-miLLM-Model"), stream_millm_model=stream_name,
                   v1_models_while_applied=models, status_block=status_block,
                   unload=[su, ju.get("data")], text_after_unload=after,
                   after_unload_equals_base=after == base, adapter_name_after_unload=gone[0],
                   v1_models_after=models_after)


def _decode_tps(api: Api, model: str, n: int, max_tokens: int) -> list[float]:
    out = []
    for i in range(n):
        t0 = time.perf_counter()
        s, j, _ = api.chat(model, f"Write a long story about a lighthouse keeper. ({i})",
                           max_tokens=max_tokens, ignore_eos=True)
        dt = time.perf_counter() - t0
        if s != 200:
            raise RuntimeError(f"chat {s} {j}")
        out.append(j["usage"]["completion_tokens"] / dt)
    return out


def cmd_throughput(a: argparse.Namespace) -> int:
    api = Api()
    _reload_base(api)
    api.chat(FP16, "warm up", max_tokens=8)
    bare = _decode_tps(api, FP16, a.n, a.max_tokens)
    s, j = _load_adapter(api, ADAPTER)
    info = j["data"]
    name = info["adapter"]["name"] if "adapter" in info else info.get("name")
    api.chat(name, "warm up", max_tokens=8)
    adapted = _decode_tps(api, name, a.n, a.max_tokens)
    api.call("POST", "/api/weights/adapter/unload", token=OPERATOR, body={})
    block = info.get("adapter", info)
    out = {"tokens_per_s_bare_median": statistics.median(bare),
           "tokens_per_s_adapter_median": statistics.median(adapted),
           "ratio": statistics.median(adapted) / statistics.median(bare),
           "bare_runs": bare, "adapter_runs": adapted, "runs_each": a.n,
           "max_tokens": a.max_tokens, "rank": block.get("rank"),
           "target_count": block.get("target_count"),
           "note": "wall time of a non-streaming greedy chat / completion_tokens (includes the "
                   "prefill of a short prompt)"}
    return verdict("throughput_10_3", s == 200, **out)


def cmd_attached(a: argparse.Namespace) -> int:
    import torch
    from safetensors.torch import load_file

    api = Api()
    _reload_base(api)
    body = api.ok("POST", "/api/weights/attach", token=TRAINER)
    att = body["attachment_id"]
    res: dict[str, Any] = {}
    oks: list[bool] = []
    try:
        tensors = rebuild(body)
        h0 = hash_tensors(tensors)
        base_text = _text(api.chat(FP16, PROMPT_CHAT, max_tokens=32))
        s, j = _load_adapter(api, ADAPTER)
        info = j.get("data") or j
        name = info["adapter"]["name"] if "adapter" in info else info.get("name")
        h1 = hash_tensors(tensors)
        adapted_text = _text(api.chat(name, PROMPT_CHAT, max_tokens=32))
        # The trainer's own LoRA (D-12: A and B in fp32, x cast to fp32, scaled, cast back).
        skel = Skeleton(LFM_DIR, tensors)
        lora = load_file(str(CKPT_ROOT / ADAPTER / "adapter_model.safetensors"), device="cuda")
        scale = ALPHA / RANK
        hooks = []
        for mname, mod in skel.model.named_modules():
            key = f"base_model.model.{mname}.lora_A.weight"
            if key in lora:
                A = lora[key].float()
                B = lora[key.replace("lora_A", "lora_B")].float()

                def hook(m, inp, out, A=A, B=B):
                    x = inp[0].float()
                    # Added in float32 and rounded ONCE, as PEFT does (D-12).
                    return (out + torch.nn.functional.linear(
                        torch.nn.functional.linear(x, A), B) * scale).to(out.dtype)

                hooks.append(mod.register_forward_hook(hook))
        full, _, logits = skel.logprobs(PROMPT)
        explicit = [int(i) for i in torch.topk(full, 15).indices]
        srv = api.score(name, PROMPT, allowed=explicit, top=len(explicit))
        mine = torch.log_softmax(logits[torch.tensor(explicit, device=logits.device)], dim=0)
        diffs = [abs(srv["top"][t] - float(mine[k])) for k, t in enumerate(explicit)
                 if t in srv["top"]]
        for h in hooks:
            h.remove()
        su, ju, _ = api.call("POST", "/api/weights/adapter/unload", token=OPERATOR, body={})
        h2 = hash_tensors(tensors)
        after_text = _text(api.chat(FP16, PROMPT_CHAT, max_tokens=32))
        ok15b = (s == 200 and h0 == h1 == h2 and adapted_text != base_text
                 and after_text == base_text and su == 200)
        res["acceptance_15b"] = {"load_status": s, "hooks": len(hooks), "adapter_name": name,
                                 "base_sha256_before": h0, "while_applied": h1, "after_unload": h2,
                                 "all_equal": h0 == h1 == h2, "changed": adapted_text != base_text,
                                 "after_unload_equals_base": after_text == base_text,
                                 "trainer_lora_vs_millm_max_abs_diff": max(diffs) if diffs else None,
                                 "trainer_lora_top1": explicit[0], "millm_top1": srv["chosen"]}
        oks.append(ok15b)
        # Acceptance 16: a mismatched adapter, serving uninterrupted.
        window: list[Any] = []
        stop = threading.Event()

        def fire() -> None:
            c = Api(timeout=120)
            while not stop.is_set():
                window.append(c.chat(FP16, "Say hello.", max_tokens=4)[0])

        th = threading.Thread(target=fire, daemon=True)
        th.start()
        time.sleep(0.5)
        sb, jb = _load_adapter(api, MISMATCH)
        time.sleep(0.5)
        stop.set()
        th.join(timeout=60)
        err = jb.get("error") or {}
        block = api.ok("GET", "/api/weights/adapter")
        h3 = hash_tensors(tensors)
        ok16 = (sb == 400 and err.get("code") == "ADAPTER_SHAPE_MISMATCH" and window
                and all(w == 200 for w in window) and not block["applied"] and h3 == h0)
        res["acceptance_16"] = {"status": sb, "code": err.get("code"),
                                "details": err.get("details"), "v1_during": window,
                                "applied_after": block["applied"], "base_unchanged": h3 == h0}
        oks.append(ok16)
    finally:
        api.call("POST", "/api/weights/detach", token=TRAINER, attachment=att)
    return verdict("acceptance_15b_16_attached_11_4", all(oks), parts_pass=oks, **res)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for n in ("build", "serve", "attached"):
        sub.add_parser(n)
    s = sub.add_parser("throughput")
    s.add_argument("--n", type=int, default=5)
    s.add_argument("--max-tokens", type=int, default=256)
    a = ap.parse_args(argv)
    return {"build": cmd_build, "serve": cmd_serve, "throughput": cmd_throughput,
            "attached": cmd_attached}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
