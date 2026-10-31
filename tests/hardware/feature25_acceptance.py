#!/usr/bin/env python3
"""Feature 25 hardware acceptance — run in the operator session, NOT by the unit suite.

Written 2026-10-06 on a workstation with no GPU and no access to the deployment, so it has been
syntax-checked and `--help`-checked only; nothing here has been run against a server. Every
subcommand prints a JSON verdict and exits non-zero on a failed criterion.

Subcommands (025_FTASKS 9.3–9.6, 0.1, 0.3):

  parity      BRD-04 acceptance 3 / SC-4: chat scoring of `messages` vs completion scoring of the
              same rendered prompt (`add_special_tokens: false`) — identical chosen token and
              alternatives, every logprob within 1e-5 absolute. The rendered prompt is produced
              by the model's own tokenizer (`--tokenizer`, a local path or HF id), so it is the
              same template the server renders.
  unsteered   BRD-04 acceptance 4 / SC-5: chat scoring with a profile active equals chat scoring
              with no SAE attached. Run once in each state (`--label`) and compare the JSON files.
  structured  BRD-04 acceptance 5 / SC-6: N json_schema requests on a transformers model all parse
              and validate; the per-token time against the same requests unconstrained; the same
              request on a GGUF model returns 400 with no load (the resident model's `loaded_at`
              unchanged, read from GET /api/models).
  seed        BRD-04 acceptance 6 / SC-7: the same sampled request with seed 7 twice gives
              byte-identical text and finish_reason, and echoes X-miLLM-Seed.
  llamacpp-seed  T-61 (0.3): forward a seed to llama.cpp DIRECTLY (not through miLLM, which
              refuses it) on the reference GGUF file, twice; report whether the bytes match.
  tokenizer-spike  0.1: build xgrammar's TokenizerInfo from a tokenizer (gemma-4) and compile the
              FTDD §3.1 schema; require the target string accepted; report timings.
  float32     FR-25.15 / SC-10 (12.x, added 2026-10-18): `scoring_precision: "float32"` on the
              node. Scores --n prompts --repeats times in float32 and once natively over a fixed
              token set (the first float32 answer's top --top ids); checks the echo; computes an
              INDEPENDENT float32 reference in this process from --checkpoint (loaded bfloat16,
              each tensor cast to float32 — miForge verify.py's method) over the CLEAN weights
              and with --scale-param multiplied by --scale (1.01). Passes when the float32
              repeat spread <= 1e-6, |float32 - clean reference| <= --clean-bar (1e-4), and the
              1 % error moves the answer by more than 10x that clean gap. Native's gap to the
              clean reference is reported (the bfloat16 blind spot float32 removes).

Examples:
  python tests/hardware/feature25_acceptance.py parity --base http://k8s-millm.hitsai.local \
      --model JEV-9B-decision --tokenizer autotrust/JEV-9B --prompts prompts.jsonl --n 200
  python tests/hardware/feature25_acceptance.py structured --base ... --model LFM2.5-1.2B-Instruct \
      --gguf-model some-model-GGUF:Q4_K_M --n 100
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from typing import Any

JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "label": {"type": "string", "enum": ["humor", "not_humor"]},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "reasons": {"type": "array", "items": {"type": "string"}, "maxItems": 3},
    },
    "required": ["label", "confidence"],
    "additionalProperties": False,
}
TARGET = '{"label":"humor","confidence":0.75,"reasons":["pun"]}'


def _client(base: str):
    import httpx

    return httpx.Client(base_url=base.rstrip("/"), timeout=600.0)


def _verdict(name: str, ok: bool, **facts: Any) -> int:
    print(json.dumps({"check": name, "pass": ok, **facts}, indent=2, default=str))
    return 0 if ok else 1


def _prompts(path: str | None, n: int) -> list[list[dict]]:
    if path:
        with open(path) as f:
            rows = [json.loads(line) for line in f if line.strip()]
        return [r["messages"] if isinstance(r, dict) else r for r in rows][:n]
    return [[{"role": "user", "content": f"Is this a joke? Item {i}: a pun about cheese."}]
            for i in range(n)]


def cmd_parity(a: argparse.Namespace) -> int:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    c = _client(a.base)
    worst, mismatches = 0.0, []
    convs = _prompts(a.prompts, a.n)
    for i, messages in enumerate(convs):
        common = {"model": a.model, "max_tokens": 1, "temperature": 1.0,
                  "return_tokens_as_token_ids": True}
        if a.allowed:
            common["allowed_token_ids"] = a.allowed
        chat = c.post("/v1/chat/completions", headers={"X-miLLM-Strict": "true"},
                      json={**common, "messages": messages, "logprobs": True,
                            "top_logprobs": a.top}).json()
        rendered = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        comp = c.post("/v1/completions", headers={"X-miLLM-Strict": "true"},
                      json={**common, "prompt": rendered, "logprobs": a.top,
                            "add_special_tokens": False}).json()
        ce = chat["choices"][0]["logprobs"]["content"][0]
        tl = comp["choices"][0]["logprobs"]
        chat_top = {e["token"]: e["logprob"] for e in ce["top_logprobs"]}
        comp_top = tl["top_logprobs"][0]
        if ce["token"] != tl["tokens"][0] or set(chat_top) != set(comp_top):
            mismatches.append(i)
            continue
        worst = max([worst, abs(ce["logprob"] - tl["token_logprobs"][0])]
                    + [abs(chat_top[k] - comp_top[k]) for k in chat_top])
        if chat["usage"]["prompt_tokens"] != comp["usage"]["prompt_tokens"]:
            mismatches.append(i)
    return _verdict("chat_scoring_parity", not mismatches and worst <= 1e-5,
                    prompts=len(convs), token_mismatches=mismatches, worst_abs_logprob_diff=worst)


def cmd_unsteered(a: argparse.Namespace) -> int:
    c = _client(a.base)
    out = []
    for messages in _prompts(a.prompts, a.n):
        r = c.post("/v1/chat/completions", json={
            "model": a.model, "messages": messages, "max_tokens": 1, "logprobs": True,
            "top_logprobs": 5, "return_tokens_as_token_ids": True}).json()
        out.append(r["choices"][0]["logprobs"]["content"][0])
    path = f"unsteered_{a.label}.json"
    with open(path, "w") as f:
        json.dump(out, f)
    if a.compare:
        with open(a.compare) as f:
            other = json.load(f)
        diffs = [abs(x["logprob"] - y["logprob"]) for x, y in zip(out, other, strict=True)]
        same_tokens = all(x["token"] == y["token"] for x, y in zip(out, other, strict=True))
        return _verdict("scoring_unsteered", same_tokens and max(diffs) <= 1e-5,
                        worst_abs_logprob_diff=max(diffs), wrote=path)
    return _verdict("scoring_unsteered_capture", True, wrote=path)


def _loaded_at(c, name: str) -> Any:
    rows = c.get("/api/models").json().get("data") or []
    return {r.get("name"): r.get("loaded_at") for r in rows}.get(name)


def cmd_structured(a: argparse.Namespace) -> int:
    from jsonschema import Draft202012Validator

    c = _client(a.base)
    fmt = {"type": "json_schema", "json_schema": {"name": "judge_v1", "schema": JUDGE_SCHEMA}}
    failures, constrained_tps, plain_tps, headers = [], [], [], set()
    for i, messages in enumerate(_prompts(a.prompts, a.n)):
        body = {"model": a.model, "messages": messages, "max_tokens": a.max_tokens,
                "temperature": 0.7, "seed": i}
        t0 = time.perf_counter()
        r = c.post("/v1/chat/completions", json={**body, "response_format": fmt})
        dt = time.perf_counter() - t0
        j = r.json()
        try:
            choice = j["choices"][0]
            if choice["finish_reason"] != "stop":
                raise ValueError(f"finish_reason {choice['finish_reason']}")
            Draft202012Validator(JUDGE_SCHEMA).validate(json.loads(choice["message"]["content"]))
            constrained_tps.append(j["usage"]["completion_tokens"] / dt)
            headers.add(r.headers.get("X-miLLM-Constrained"))
        except Exception as exc:  # noqa: BLE001 - every failure is reported
            failures.append({"index": i, "status": r.status_code, "error": str(exc)})
        t0 = time.perf_counter()
        p = c.post("/v1/chat/completions", json=body).json()
        plain_tps.append(p["usage"]["completion_tokens"] / (time.perf_counter() - t0))
    gguf = {}
    if a.gguf_model:
        before = _loaded_at(c, a.model)
        r = c.post("/v1/chat/completions", json={"model": a.gguf_model, "response_format": fmt,
                                                 "messages": [{"role": "user", "content": "x"}]})
        gguf = {"status": r.status_code, "param": (r.json().get("error") or {}).get("param"),
                "resident_loaded_at_unchanged": _loaded_at(c, a.model) == before}
    ok = not failures and (not gguf or (gguf["status"] == 400 and gguf["param"] == "response_format"
                                        and gguf["resident_loaded_at_unchanged"]))
    return _verdict("structured_output", ok, requests=a.n, failures=failures,
                    constrained_header=sorted(map(str, headers)),
                    tokens_per_s_constrained=statistics.median(constrained_tps or [0]),
                    tokens_per_s_unconstrained=statistics.median(plain_tps or [0]), gguf=gguf)


def cmd_seed(a: argparse.Namespace) -> int:
    c = _client(a.base)
    body = {"model": a.model, "messages": [{"role": "user", "content": "Write a short poem."}],
            "max_tokens": 64, "temperature": 1.0, "seed": 7}
    r1, r2 = c.post("/v1/chat/completions", json=body), c.post("/v1/chat/completions", json=body)
    x1, x2 = r1.json()["choices"][0], r2.json()["choices"][0]
    r3 = c.post("/v1/chat/completions", json={**body, "seed": 8}).json()["choices"][0]
    return _verdict("seed_repeatability",
                    x1 == x2 and r1.headers.get("X-miLLM-Seed") == '7;scope="request"',
                    identical=x1 == x2, seed_8_differs=r3 != x1,
                    header=r1.headers.get("X-miLLM-Seed"),
                    fingerprint=r1.json().get("system_fingerprint"))


def cmd_llamacpp_seed(a: argparse.Namespace) -> int:
    from llama_cpp import Llama

    llm = Llama(model_path=a.gguf, n_gpu_layers=-1, seed=a.seed, verbose=False)
    msgs = [{"role": "user", "content": "Write a short poem."}]
    outs = [llm.create_chat_completion(messages=msgs, max_tokens=64, temperature=1.0,
                                       seed=a.seed)["choices"][0]["message"]["content"]
            for _ in range(2)]
    return _verdict("t61_llamacpp_seed", outs[0] == outs[1], identical=outs[0] == outs[1],
                    note="pass -> flip the llama.cpp seed cells and forward seed in "
                         "_llamacpp_params (025_FTASKS 6.6); fail -> keep refused")


def cmd_tokenizer_spike(a: argparse.Namespace) -> int:
    import xgrammar as xgr
    from transformers import AutoConfig, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    try:
        cfg = AutoConfig.from_pretrained(a.tokenizer)
        vocab = getattr(cfg, "vocab_size", None) or cfg.text_config.vocab_size
    except Exception:  # noqa: BLE001
        vocab = len(tok)
    t0 = time.perf_counter()
    info = xgr.TokenizerInfo.from_huggingface(tok, vocab_size=vocab)
    grammar = xgr.GrammarCompiler(info).compile_json_schema(json.dumps(JUDGE_SCHEMA))
    compile_s = time.perf_counter() - t0
    accepted = xgr.GrammarMatcher(grammar).accept_string(TARGET)
    matcher, mask = xgr.GrammarMatcher(grammar), xgr.allocate_token_bitmask(1, vocab)
    times = []
    for _ in range(50):
        t = time.perf_counter()
        matcher.fill_next_token_bitmask(mask, 0)
        times.append((time.perf_counter() - t) * 1000)
    return _verdict("tokenizer_spike", bool(accepted), vocab=vocab, compile_s=compile_s,
                    mask_ms_median=statistics.median(times), mask_ms_max=max(times))


def _f32_reference(checkpoint: str, prompts: list[str], ids: list[int], scale_param: str | None,
                   scale: float, add_special_tokens: bool) -> list[dict[int, float]]:
    """Log-softmax over `ids` at the next position, float32 on the CPU, from the bf16 weights."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.set_float32_matmul_precision("highest")
    tok = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForCausalLM.from_pretrained(checkpoint, dtype=torch.bfloat16)
    model = model.float().eval()  # bf16 values, each upcast exactly
    if scale_param:
        with torch.no_grad():
            dict(model.named_parameters())[scale_param].mul_(scale)
    out = []
    for prompt in prompts:
        enc = tok(prompt, return_tensors="pt", add_special_tokens=add_special_tokens)
        with torch.no_grad():
            logits = model(**enc).logits[0, -1].float()
        lp = torch.log_softmax(logits[ids], 0)
        out.append({i: float(v) for i, v in zip(ids, lp, strict=True)})
    return out


def cmd_float32(a: argparse.Namespace) -> int:
    c = _client(a.base)
    prompts = [f"Question {i}: the capital of France is" if not a.prompts else line.strip()
               for i, line in enumerate(open(a.prompts) if a.prompts else range(a.n))][: a.n]
    base = {"model": a.model, "max_tokens": 1, "temperature": 1.0,
            "return_tokens_as_token_ids": True, "add_special_tokens": a.add_special_tokens}
    first = c.post("/v1/completions", headers={"X-miLLM-Strict": "true"},
                   json={**base, "prompt": prompts[0], "logprobs": a.top,
                         "scoring_precision": "float32"})
    if first.status_code != 200:
        return _verdict("float32_scoring", False, status=first.status_code, body=first.text,
                        reason_header=first.headers.get("X-miLLM-Scoring-Precision-Reason"))
    ids = [int(k.split(":", 1)[1]) for k in first.json()["choices"][0]["logprobs"]["top_logprobs"][0]]

    def score(precision: str | None) -> tuple[list[dict[int, float]], list[str | None]]:
        body = {**base, "logprobs": len(ids), "allowed_token_ids": ids}
        if precision:
            body["scoring_precision"] = precision
        rows, echoes = [], []
        for prompt in prompts:
            r = c.post("/v1/completions", headers={"X-miLLM-Strict": "true"},
                       json={**body, "prompt": prompt})
            r.raise_for_status()
            top = r.json()["choices"][0]["logprobs"]["top_logprobs"][0]
            rows.append({int(k.split(":", 1)[1]): v for k, v in top.items()})
            echoes.append(r.headers.get("X-miLLM-Scoring-Precision"))
            if precision and r.json().get("scoring_precision") != precision:
                echoes[-1] = None
        return rows, echoes

    def gap(xs: list[dict[int, float]], ys: list[dict[int, float]]) -> float:
        return max(abs(x[i] - y[i]) for x, y in zip(xs, ys, strict=True) for i in ids)

    runs = [score("float32") for _ in range(a.repeats)]
    native, _ = score(None)
    echo_ok = all(e == "float32" for _, echoes in runs for e in echoes)
    spread = max(gap(runs[0][0], r[0]) for r in runs)
    clean_ref = _f32_reference(a.checkpoint, prompts, ids, None, 1.0, a.add_special_tokens)
    error_ref = _f32_reference(a.checkpoint, prompts, ids, a.scale_param, a.scale,
                               a.add_special_tokens)
    clean = gap(runs[0][0], clean_ref)
    error = gap(runs[0][0], error_ref)
    native_gap = gap(native, clean_ref)
    ok = echo_ok and spread <= 1e-6 and clean <= a.clean_bar and error > 10 * max(clean, 1e-9)
    return _verdict("float32_scoring", ok, prompts=len(prompts), token_ids=len(ids),
                    repeats=a.repeats, echo_ok=echo_ok, float32_repeat_spread=spread,
                    float32_vs_clean_reference=clean, float32_vs_one_percent_error=error,
                    native_vs_clean_reference=native_gap, scale_param=a.scale_param,
                    scale=a.scale)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("parity", "unsteered", "structured", "seed"):
        s = sub.add_parser(name)
        s.add_argument("--base", required=True)
        s.add_argument("--model", required=True)
        s.add_argument("--prompts")
        s.add_argument("--n", type=int, default=200 if name == "parity" else 100)
        if name == "parity":
            s.add_argument("--tokenizer", required=True)
            s.add_argument("--allowed", type=int, nargs="*")
            s.add_argument("--top", type=int, default=2)
        if name == "unsteered":
            s.add_argument("--label", required=True)
            s.add_argument("--compare")
        if name == "structured":
            s.add_argument("--gguf-model")
            s.add_argument("--max-tokens", type=int, default=128)
    s = sub.add_parser("llamacpp-seed")
    s.add_argument("--gguf", required=True)
    s.add_argument("--seed", type=int, default=7)
    s = sub.add_parser("tokenizer-spike")
    s.add_argument("--tokenizer", required=True)
    s = sub.add_parser("float32")
    s.add_argument("--base", required=True)
    s.add_argument("--model", required=True)
    s.add_argument("--checkpoint", required=True, help="local path of the served checkpoint")
    s.add_argument("--scale-param", required=True,
                   help="a parameter name to scale for the 1 %% error, e.g. a mid-layer MLP weight")
    s.add_argument("--scale", type=float, default=1.01)
    s.add_argument("--prompts")
    s.add_argument("--n", type=int, default=8)
    s.add_argument("--repeats", type=int, default=10)
    s.add_argument("--top", type=int, default=20)
    s.add_argument("--clean-bar", type=float, default=1e-4)
    s.add_argument("--add-special-tokens", action=argparse.BooleanOptionalAction, default=True)
    a = ap.parse_args(argv)
    return {"parity": cmd_parity, "unsteered": cmd_unsteered, "structured": cmd_structured,
            "seed": cmd_seed, "llamacpp-seed": cmd_llamacpp_seed,
            "tokenizer-spike": cmd_tokenizer_spike, "float32": cmd_float32}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
