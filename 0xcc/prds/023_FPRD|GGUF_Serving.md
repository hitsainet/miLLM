# Feature PRD: GGUF Serving
## miLLM Feature 23

**Source BRD:** BRD-MILLM-GGUF-001
**Status:** ✅ Implemented and serving in production (2026-09-07/08)
**Supersedes:** the v1.0 out-of-scope entry for GGUF in PPRD §4 and FPRD 001

---

## 1. Overview

miLLM serves `.gguf` files through the same OpenAI-compatible API as its
transformers models. A GGUF model is downloaded, named, loaded, locked, unloaded
and deleted through the existing model-management surface; only the loader
differs.

**User value:** run the models the local-inference ecosystem actually
distributes, including large models that only fit on one card once quantized.
The forcing case was a 31B judge for feature labeling: bf16 does not fit a 24 GB
card, `IQ4_XS` does at ~16 GB.

## 2. Functional requirements

**FR-23.1 — Serve GGUF over the OpenAI surface.** `/v1/chat/completions`,
`/v1/completions`, `/v1/embeddings`, `/v1/models`. No separate endpoint, no
client-side branching.

**FR-23.2 — Download a specific quantization.** `gguf_label` names the file to
fetch (`IQ4_XS`, `Q4_K_M`, …); Preview lists what a repository publishes with
sizes.

**FR-23.3 — Coexisting quantizations.** Several quantizations of one repository
may be present at once, named `repo:QUANT`, each separately selectable in
`/v1/models`. A user-supplied `custom_name` is preserved.

**FR-23.4 — Unambiguous resolution.** A bare repository name resolves while
exactly one quantization carries it. When several do, return **400
`AMBIGUOUS_MODEL_NAME`** listing the tags. Never pick by insertion order.

**FR-23.5 — Derived context window.** Read `n_ctx_train` from the file, predict
what free VRAM holds, and load at the largest window that fits under
`GGUF_CONTEXT_LENGTH` — which is a *ceiling*, not a target. Confirm by loading;
fall back to the ladder if a prediction cannot be made.

**FR-23.6 — KV cache as a deliberate VRAM trade.** `GGUF_KV_CACHE_TYPE` defaults
to `q8_0`; a quantized cache requires flash attention and the loader refuses the
invalid pairing rather than failing later.

**FR-23.7 — Graceful capability degradation.** If embedding support prevents a
load, retry without it and record `supports_embeddings=false` on the model.

**FR-23.8 — Correct client-error semantics.** An oversized prompt is 400
`context_length_exceeded`. `chat_template_kwargs` a GGUF template cannot use is
ignored and logged, not refused. A trailing assistant message is *continued*,
not restarted.

## 3. Out of scope

SAE attachment, steering and probe monitoring on GGUF models. llama.cpp does not
expose the hook points these need; interpretability work remains
transformers-only. Format conversion and GGUF training are also out of scope.

## 4. Acceptance criteria — all met on the RTX 3090 deployment

| # | Criterion | Evidence |
|---|---|---|
| 1 | GGUF and transformers models served together | 17 models in `/v1/models`, 4 GGUF |
| 2 | Two quantizations of one repo, separately selectable | `…-GGUF:IQ4_XS` and `…-GGUF:Q4_K_M` |
| 3 | Ambiguous bare name refused, not guessed | 400 `AMBIGUOUS_MODEL_NAME` naming both tags |
| 4 | Context derived, not guessed | `gemma-4-31b IQ4_XS` loads at 12,288 (was 4,096) |
| 5 | Prediction is accurate | 4 of 4 measurements bracketed correctly |
| 6 | Oversized prompt is a client error | 400 `context_length_exceeded` |
| 7 | Serving survives an unsupported optional feature | Qwen3.8-27B loads without embeddings |
| 8 | Continue resumes | verified in Open WebUI on two GGUF models |
| 9 | Real workload | 31B GGUF judge completed a live 20-feature labeling batch |

## 5. Why this PRD is retroactive

The v1.0 chain put GGUF out of scope on the premise that it "requires a
different inference engine". It requires a different *loader*; everything above
the loader is shared. The work was then done incrementally in response to a real
need, and the chain was never reconciled — so the documentation said one thing
while production did another. This closes that.
