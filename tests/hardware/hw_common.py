"""Shared helpers for the attachable-weights hardware harnesses (Features 31, 32, 33).

Run INSIDE the miLLM pod (addendum D-02: in-container trainer topology), never by the unit suite.
Every subcommand prints one JSON verdict per check and exits non-zero on a failed criterion.

Environment (set by `tests/hardware/millm_hwtest_pod.yaml`):
  HWTEST_TRAINER_TOKEN, HWTEST_OPERATOR_TOKEN  bearer tokens whose digests are in WEIGHTS_TOKENS
  HWTEST_BASE                                  miLLM base URL (default http://127.0.0.1:8000)
  HWTEST_OUT                                   where evidence files go (default /hw/out)
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Optional

BASE = os.environ.get("HWTEST_BASE", "http://127.0.0.1:8000")
OUT = Path(os.environ.get("HWTEST_OUT", "/hw/out"))
TRAINER = os.environ.get("HWTEST_TRAINER_TOKEN", "")
OPERATOR = os.environ.get("HWTEST_OPERATOR_TOKEN", "")

#: The fixed prompt every log-probability comparison uses.
PROMPT = "The capital of France is Paris. The capital of Germany is"

#: Model rows the harness registers in the test pod's own database (`seed`).
MODEL_CACHE = os.environ.get("MODEL_CACHE_DIR", "/hw/data/model_cache")
LFM_DIR = f"{MODEL_CACHE}/huggingface/LiquidAI--LFM2.5-1.2B-Instruct--FP16"
GGUF_DIR = f"{MODEL_CACHE}/huggingface/LiquidAI--LFM2.5-1.2B-Instruct-GGUF--Q4--Q4_K_M"
TINY_DIR = "/hw/data/local_models/tiny-llama-hwtest"
ROWS = {
    "fp16": dict(name="LFM2.5-1.2B-Instruct", source="huggingface",
                 repo_id="LiquidAI/LFM2.5-1.2B-Instruct", quantization="FP16",
                 cache_path=LFM_DIR, params="1.2B", architecture="text-generation"),
    "q4": dict(name="LFM2.5-1.2B-Instruct-Q4", source="huggingface",
               repo_id="LiquidAI/LFM2.5-1.2B-Instruct", quantization="Q4",
               cache_path=LFM_DIR, params="1.2B", architecture="text-generation"),
    "gguf": dict(name="LFM2.5-1.2B-Instruct-GGUF:Q4_K_M", source="huggingface",
                 repo_id="LiquidAI/LFM2.5-1.2B-Instruct-GGUF", quantization="Q4",
                 gguf_label="Q4_K_M", gguf_files=["LFM2.5-1.2B-Instruct-Q4_K_M.gguf"],
                 cache_path=GGUF_DIR, params="1.2B", architecture="text-generation"),
    # Feature 37's second model (probe scaling): d_model 4096, 32 layers.
    "llama8b": dict(name="Llama-3.1-8B-Instruct", source="huggingface",
                    repo_id="meta-llama/Llama-3.1-8B-Instruct", quantization="FP16",
                    cache_path=f"{MODEL_CACHE}/huggingface/meta-llama--Llama-3.1-8B-Instruct--FP16",
                    params="8.0B", architecture="text-generation"),
    "tiny": dict(name="tiny-llama-hwtest", source="local", local_path=TINY_DIR,
                 quantization="FP16", cache_path=TINY_DIR, params="4M",
                 architecture="text-generation"),
}


def now_et() -> str:
    """Wall-clock time in US Eastern (the operator reads ET)."""
    from datetime import datetime
    from zoneinfo import ZoneInfo

    return datetime.now(ZoneInfo("America/New_York")).strftime("%Y-%m-%d %H:%M:%S %Z")


def verdict(name: str, ok: bool, **facts: Any) -> int:
    rec = {"check": name, "pass": ok, "at_et": now_et(), "code": code_identity(), **facts}
    print(json.dumps(rec, indent=2, default=str), flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "verdicts.jsonl", "a") as fh:
        fh.write(json.dumps(rec, default=str) + "\n")
    return 0 if ok else 1


def code_identity() -> str:
    """Which commit the server's code came from: the `src-<commit>` directory /hw/src points at."""
    first = (os.environ.get("PYTHONPATH") or "/hw/src").split(":")[0]
    try:
        return Path(first).resolve().name
    except OSError:
        return "unknown"


def pctl(values: list[float], q: float) -> float:
    """Nearest-rank percentile (q in [0, 100]); no interpolation, so p99.9 is a real sample."""
    s = sorted(values)
    if not s:
        return float("nan")
    k = max(0, min(len(s) - 1, int(-(-q * len(s) // 100)) - 1))
    return s[k]


def summary(values: list[float]) -> dict[str, Any]:
    return {"n": len(values), "p50": pctl(values, 50), "p99": pctl(values, 99),
            "p99_9": pctl(values, 99.9), "max": max(values) if values else None,
            "mean": statistics.fmean(values) if values else None}


# --------------------------------------------------------------------------------------------
# HTTP


class Api:
    def __init__(self, base: str = BASE, timeout: float = 600.0) -> None:
        import httpx

        self.c = httpx.Client(base_url=base.rstrip("/"), timeout=timeout)

    def call(self, method: str, path: str, *, token: Optional[str] = None,
             attachment: Optional[str] = None, body: Any = None,
             headers: Optional[dict] = None) -> tuple[int, Any, Any]:
        h = dict(headers or {})
        if token:
            h["Authorization"] = f"Bearer {token}"
        if attachment:
            h["X-miLLM-Attachment"] = attachment
        r = self.c.request(method, path, headers=h, json=body)
        try:
            j = r.json()
        except ValueError:
            j = {"_text": r.text[:500]}
        return r.status_code, j, r.headers

    def ok(self, method: str, path: str, **kw: Any) -> Any:
        status, j, _ = self.call(method, path, **kw)
        if status >= 300:
            raise RuntimeError(f"{method} {path} -> {status}: {json.dumps(j)[:800]}")
        return j.get("data", j) if isinstance(j, dict) else j

    def status(self) -> dict:
        return self.ok("GET", "/api/weights/status")

    def model_row(self, name: str) -> dict:
        rows = self.ok("GET", "/api/models")
        for r in rows:
            if r["name"] == name:
                return r
        raise KeyError(name)

    def resident(self) -> Optional[dict]:
        for r in self.ok("GET", "/api/models"):
            if r["status"] == "loaded":
                return r
        return None

    def load(self, name: str, timeout: float = 300.0) -> dict:
        row = self.model_row(name)
        if row["status"] == "loaded":
            return row
        res = self.resident()
        if res is not None:
            self.unload(res["name"])
        self.ok("POST", f"/api/models/{row['id']}/load", body={})
        t0 = time.monotonic()
        while time.monotonic() - t0 < timeout:
            row = self.model_row(name)
            if row["status"] == "loaded":
                return row
            if row["status"] == "error":
                raise RuntimeError(f"load {name} failed: {row.get('error_message')}")
            time.sleep(0.5)
        raise TimeoutError(f"load {name}")

    def unload(self, name: str, timeout: float = 120.0) -> None:
        row = self.model_row(name)
        if row["status"] != "loaded":
            return
        self.ok("POST", f"/api/models/{row['id']}/unload", body={})
        t0 = time.monotonic()
        while time.monotonic() - t0 < timeout:
            if self.model_row(name)["status"] != "loaded":
                return
            time.sleep(0.5)
        raise TimeoutError(f"unload {name}")

    # -- scoring: log-probabilities of the next token ------------------------------------------

    def score(self, model: str, prompt: str = PROMPT, allowed: Optional[list[int]] = None,
              top: int = 20) -> dict[str, Any]:
        """Next-token log-probabilities from `/v1/completions` scoring mode (temperature 0,
        so no scaling). With `allowed`, renormalised over that explicit set (vLLM semantics)."""
        body: dict[str, Any] = {"model": model, "prompt": prompt, "max_tokens": 1,
                                "temperature": 0, "logprobs": top,
                                "return_tokens_as_token_ids": True}
        if allowed is not None:
            body["allowed_token_ids"] = allowed
        status, j, headers = self.call("POST", "/v1/completions", body=body)
        if status != 200:
            raise RuntimeError(f"score -> {status}: {json.dumps(j)[:600]}")
        lp = j["choices"][0]["logprobs"]
        top_map = {int(k.split(":", 1)[1]): float(v) for k, v in lp["top_logprobs"][0].items()}
        chosen = int(lp["tokens"][0].split(":", 1)[1])
        return {"chosen": chosen, "top": top_map, "prompt_tokens": j["usage"]["prompt_tokens"],
                "model": j.get("model"), "x_millm_model": headers.get("X-miLLM-Model")}

    def chat(self, model: str, content: str, max_tokens: int = 24, **extra: Any) -> tuple[int, Any, Any]:
        return self.call("POST", "/v1/chat/completions", body={
            "model": model, "messages": [{"role": "user", "content": content}],
            "max_tokens": max_tokens, "temperature": 0, **extra})


# --------------------------------------------------------------------------------------------
# Rebuilding the attachment's tensors (the manual's recipe, manual/docs/api/weights.md)


def rebuild(body: dict) -> dict[str, Any]:
    """The manual's recipe (manual/docs/api/weights.md): ONE rebuild per storage entry, then a view
    per parameter. Rebuilding per parameter strands a tied storage (hardware finding 3)."""
    import torch
    from torch.multiprocessing.reductions import rebuild_cuda_tensor

    # The exported parameters keep miLLM's `requires_grad`; every write here is a raw in-place
    # update, never a differentiable one.
    torch.set_grad_enabled(False)
    storages = []
    for s in body["storages"]:

        def b(k: str, s: dict = s) -> Optional[bytes]:
            return base64.b64decode(s[k]) if s[k] is not None else None

        whole = rebuild_cuda_tensor(
            torch.Tensor, (s["storage_size_bytes"],), (1,), 0, torch.UntypedStorage, torch.uint8,
            s["device_index"], b("handle"), s["storage_size_bytes"], s["storage_offset_bytes"],
            False, b("ref_counter_handle"), s["ref_counter_offset"], b("event_handle"),
            s["event_sync_required"],
        )
        storages.append(whole.untyped_storage())
    tensors = {}
    for p in body["parameters"]:
        st = storages[p["storage"]]
        t = torch.empty(0, dtype=getattr(torch, p["dtype"]), device=st.device)
        t.set_(st, p["storage_offset"], tuple(p["shape"]), tuple(p["stride"]))
        tensors[p["name"]] = t
    return tensors


def hash_tensors(tensors: dict[str, Any]) -> str:
    """SHA-256 over every tensor's bytes in name order (read through the handles)."""
    import torch

    h = hashlib.sha256()
    torch.cuda.synchronize()
    for name in sorted(tensors):
        t = tensors[name].detach()
        h.update(name.encode())
        h.update(t.contiguous().view(torch.uint8).cpu().numpy().tobytes())
    return h.hexdigest()


class Skeleton:
    """The resident model's architecture with its parameters REPLACED by the rebuilt handles, so a
    forward pass in this (the trainer's) process reads miLLM's memory."""

    def __init__(self, cache_path: str, tensors: dict[str, Any]) -> None:
        import torch
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

        cfg = AutoConfig.from_pretrained(cache_path)
        dtype = next(iter(tensors.values())).dtype
        with torch.device("cuda"):
            model = AutoModelForCausalLM.from_config(cfg, dtype=dtype)
        result = model.load_state_dict(tensors, strict=False, assign=True)
        params = {n for n, _ in model.named_parameters(remove_duplicate=False)}
        self.missing_parameters = sorted(params - set(tensors))
        self.missing_state = list(result.missing_keys)
        self.unexpected = list(result.unexpected_keys)
        model.eval()
        self.model = model
        self.tok = AutoTokenizer.from_pretrained(cache_path)
        torch.cuda.empty_cache()

    def logprobs(self, prompt: str = PROMPT) -> Any:
        import torch

        enc = self.tok(prompt, return_tensors="pt").to("cuda")
        with torch.no_grad():
            # The call miLLM's scorer makes (`InferenceService._next_token_logits`): no cache, the
            # last position only. A different call (a cache, every position's logits) takes other
            # kernels, and bfloat16 logits of magnitude 16-32 have a 0.0625 spacing, so any
            # difference in the forward shows up as a ~0.06 log-probability step.
            out = self.model(**enc, use_cache=False, logits_to_keep=1)
        logits = out.logits[0, -1].float()
        return torch.log_softmax(logits, dim=0), int(enc.input_ids.shape[1]), logits


def compare_logprobs(api: Api, model_name: str, skel: Skeleton, prompt: str = PROMPT,
                     extra_ids: Iterable[int] = ()) -> dict[str, Any]:
    """Acceptance 1 / 11 (addendum D-27): identical top-1, and log-probabilities over an EXPLICIT
    token set within 1e-2. The set is the trainer's top 10 plus fixed ids; miLLM renormalises
    over it (allowed_token_ids), and so does the comparison here. Also compares miLLM's
    unrestricted top 20 against the trainer's full-vocabulary log-probabilities."""
    import torch

    full, n_prompt, logits = skel.logprobs(prompt)
    trainer_top1 = int(torch.argmax(full))
    explicit = list(dict.fromkeys(
        [int(i) for i in torch.topk(full, 10).indices] + [int(i) for i in extra_ids]))[:20]
    srv_set = api.score(model_name, prompt, allowed=explicit, top=len(explicit))
    trainer_set = torch.log_softmax(logits[torch.tensor(explicit, device=logits.device)], dim=0)
    set_diffs = {i: abs(srv_set["top"][i] - float(trainer_set[k]))
                 for k, i in enumerate(explicit) if i in srv_set["top"]}
    srv_full = api.score(model_name, prompt, allowed=None, top=20)
    full_diffs = {i: abs(v - float(full[i])) for i, v in srv_full["top"].items()}
    max_set = max(set_diffs.values()) if set_diffs else float("inf")
    max_full = max(full_diffs.values()) if full_diffs else float("inf")
    return {
        "prompt_tokens_server": srv_full["prompt_tokens"], "prompt_tokens_trainer": n_prompt,
        "top1_server": srv_full["chosen"], "top1_trainer": trainer_top1,
        "top1_identical": srv_full["chosen"] == trainer_top1,
        "explicit_set": explicit, "explicit_set_returned": len(set_diffs),
        "max_abs_diff_explicit_set": max_set,
        "max_abs_diff_server_top20_vs_trainer_full": max_full,
        "tolerance": 1e-2,
        "pass": (srv_full["chosen"] == trainer_top1 and len(set_diffs) == len(explicit)
                 and max_set <= 1e-2 and max_full <= 1e-2
                 and srv_full["prompt_tokens"] == n_prompt),
    }


def sh(cmd: str) -> str:
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=60).stdout.strip()
    except Exception as exc:  # noqa: BLE001 - evidence collection records what failed
        return f"<{type(exc).__name__}: {exc}>"


def gpu_used_mib() -> Optional[int]:
    out = sh("nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits")
    try:
        return int(out.splitlines()[0])
    except (ValueError, IndexError):
        return None


def seed_rows(keys: Iterable[str]) -> dict[str, int]:
    """Register model rows in THIS pod's database (never the live one) through miLLM's ORM."""
    import asyncio

    async def go() -> dict[str, int]:
        from millm.db.base import async_session_factory
        from millm.db.models.model import Model, ModelSource, ModelStatus, QuantizationType
        from millm.db.repositories.model_repository import ModelRepository

        ids = {}
        async with async_session_factory() as session:
            repo = ModelRepository(session)
            for key in keys:
                spec = dict(ROWS[key])
                existing = await repo.find_by_name(spec["name"])
                if existing is not None:
                    ids[key] = existing.id
                    continue
                spec["source"] = ModelSource(spec["source"])
                spec["quantization"] = QuantizationType(spec["quantization"])
                row = await repo.create(status=ModelStatus.READY, **spec)
                ids[key] = row.id
            await session.commit()
        return ids

    return asyncio.run(go())


def write_tiny_llama(path: str = TINY_DIR) -> None:
    """A 16-bit random Llama, for the differently-shaped checkpoint (acceptance 11)."""
    import torch
    from transformers import AutoTokenizer, LlamaConfig, LlamaForCausalLM

    if Path(path, "model.safetensors").exists():
        return
    torch.manual_seed(0)
    cfg = LlamaConfig(vocab_size=65536, hidden_size=64, intermediate_size=128,
                      num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=4,
                      max_position_embeddings=512, tie_word_embeddings=False, dtype="bfloat16")
    model = LlamaForCausalLM(cfg).to(torch.bfloat16)
    model.save_pretrained(path)
    # Any tokenizer with the same vocabulary size serves; LFM2.5's has 65,536 ids.
    AutoTokenizer.from_pretrained(LFM_DIR).save_pretrained(path)


def run(argv: list[str]) -> int:  # pragma: no cover - convenience for ad hoc checks
    return subprocess.call([sys.executable, *argv])
