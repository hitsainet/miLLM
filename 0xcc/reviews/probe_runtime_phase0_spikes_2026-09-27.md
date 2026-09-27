# Feature 024 — Phase 0 spikes, 2026-09-27

Four spikes, all four answered. **Two of them changed the design**: OQ-2's planned mechanism could
never have worked, and OQ-3's fallback turns out to be unnecessary.

Baseline before any code: `tests/unit` **2505 passed / 13 skipped / 0 failed** in 87.55s
(`./venv/bin/python -m pytest tests/unit --no-cov`). CLAUDE.md's "2133 passed" is stale — dated
2026-07-21, predating Feature 023.

---

## 0.1 / OQ-2 — resolving the model revision

**The specified mechanism does not work, on any model.** FTID §2 point 9 says to "parse `cache_path`
for `snapshots/<40-hex>`; if absent, `REVISION_UNVERIFIED`". miLLM does not use the HuggingFace hub
cache layout — it downloads with a `local_dir`, so every real `cache_path` looks like:

```
/data/model_cache/huggingface/LiquidAI--LFM2.5-1.2B-Instruct--FP16
```

There is no `snapshots/` segment in any of them, so the parse would fall through to
`REVISION_UNVERIFIED` **every time** and the revision check would be a permanent no-op warning.

**`models.revision` is not a substitute either.** It is written as `revision=request.revision` —
the value the operator *requested*. That may be a commit SHA, a branch name like `main`, or NULL.
Live: NULL on 2 of the 4 models present.

**What does work: the HuggingFace download metadata.** A `local_dir` download leaves
`.cache/huggingface/download/<file>.metadata`, whose **first line is the resolved commit SHA**.
Verified on the node across every model directory:

| model dir | .metadata files | distinct commits |
|---|---|---|
| `LiquidAI--LFM2.5-1.2B-Instruct--FP16` | 11 | `0f604ada3f766f9f257460c4c9f0b5d6f69d431b` |
| `Qwen--Qwen2.5-7B-Instruct--FP16` | 14 | `a09a35458c702b33eeacc393d103063234e8bc28` |
| `bartowski--OLMo-2-1124-13B-Instruct-GGUF--Q4` | 28 | `00f2e4aed2fdcac064c1b9613ce2d520f54369e2` |
| `kushalpatil--jevify-gemma4-e4b--Q8` | 9 | `a6b5a716f5d7dba2949c14f2c9e93227956e935d` |

Every directory is internally consistent — one commit across all its files — and LFM2.5's matches
the `revision` column exactly. **Qwen's commit is recoverable from disk while its DB `revision` is
NULL**, so this takes the check from working on 2 of 4 models to 4 of 4.

**Decision.** Resolve in this order: (1) the commit from any `.metadata` file under `cache_path`;
(2) `models.revision` when it is 40 hex characters; (3) otherwise `REVISION_UNVERIFIED`, warn and
allow, per locked decision 10.

⚠ **Assert agreement, do not sample one file.** A directory re-downloaded file-by-file across two
upstream revisions would hold files from different commits, and a single-file read would report a
confident, wrong SHA. Read at least `config.json` and the weights file and refuse to claim a
revision when they disagree — that disagreement is itself worth surfacing, because it means the
checkout on disk is not any published revision.

---

## 0.2 / OQ-3 — do real clients tolerate a terminal `choices: []` chunk?

**Yes, both of them. The `X-miLLM-Probe-Stream: 1` opt-in fallback is not needed.**

**OpenAI Python SDK 3.19.2** — measured against a mock server emitting miLLM's exact serial
sequence (role chunk, two content chunks, final chunk with `finish_reason` + `usage`, then the
proposed probe chunk, then `[DONE]`):

```
exception            : NONE — the SDK consumed it cleanly
chunks yielded       : 5  (5 emitted)
assembled text       : 'High stakes.'
extension reachable  : YES via ev.model_extra
  -> high-stakes, verdict=True, rung=3
```

**Open WebUI** (`ghcr.io/open-webui/open-webui:latest`, running on the node) — read its stream
parser rather than guessing. `utils/middleware.py:4966` is an explicit `if not choices:` branch that
checks for a provider error and otherwise `continue`s. The unguarded `choices[0]` at `:4991` is
downstream of it and cannot be reached with an empty list. Two further sites (`:4089`, `:4201`) are
independently guarded with `if choices else ''`.

This is unsurprising in hindsight: OpenAI's own `stream_options.include_usage` chunk also carries
`choices: []`, so any client that supports OpenAI already handles the shape.

⚠ **Open WebUI will silently DROP the extension field.** Its empty-choices branch reads only
`error` and `usage`. That is consistent with the FPRD's "Open WebUI outlet display" non-goal and
with the shipped dial filter's own note that it has no outlet hook — the chunk is *safe* there, not
*visible* there. A verdict reaches a human through miLLM's own Probe Monitors page, not the chat.

---

## 0.3 / OQ-1 — parity tolerance

**1e-3 confirmed, and it is generous rather than tight.** miStudio's 033 export acceptance measured
the recorded `token_ids` reproducing the stored scores at **0.000e+00 on all sixteen test vectors**.
The tolerance is not absorbing float drift at fp16; it is absorbing nothing at all on the reference
path.

⚠ **This only holds for the `token_ids` path.** The same acceptance measured re-scoring from the
document's own `messages` missing by up to **1.153** against a 0.05 tolerance, because `messages` is
a reconstruction of plain prose as a single user turn and re-rendering it adds six template tokens.
`ProbeParityEngine` must score from `token_ids` and report `messages` divergence **separately** as
tokenization drift — never fold it into the pass/fail number. A parity check that reports "incorrect"
against a correct consumer is worse than no check, because it is believed the first time.

---

## 0.4 / OQ-4 — is SAE encoding identical in both apps?

**No, and it cannot be — but the contract already carries everything needed to close the gap.**

miLLM's `LoadedSAE.encode` (`ml/sae_wrapper.py:369`) is:

```python
return torch.relu(x @ self.W_enc + self.b_enc)
```

miStudio's `encode_with_training_normalization` (`ml/sparse_autoencoder.py:1486`) differs in two
ways, both load-bearing:

1. **Normalization.** `if sae.normalize_activations != "none": x, _ = sae.normalize(x)` before
   encoding. Its own docstring records why this exists: MIS-E2E-083, where five of six call sites
   reached for bare `encode()` and mined every circuit from activations the dictionary was never
   trained to decode — "the features fire, the numbers are plausible, and the basis is wrong".
2. **The architecture's real activation.** JumpReLU applies `(pre > threshold) * pre`. `torch.relu`
   is not that, and it destroys negative pre-activations before any threshold comparison.

**Normalization needs no new contract fields.** The three modes are `constant_norm_rescale`,
`anthropic_rescale` and `none` — and the first two are **the same operation**, a per-sample rescale
to ‖x‖ = √dim. miStudio measured the two formulations agreeing to **7.2e-7** and records that the
second is an alias, not a second method (MIS-E2E-085). So the runtime needs only the **mode**, which
`sae.normalization = {"mode", "source", ...}` already carries.

**Decision — resolving the normativity conflict flagged in the plan.** FTASKS 4A.4 asserts "the
slice equals the full encode restricted to `idx`". Taken against `LoadedSAE.encode` that assertion
is *satisfiable and wrong*, because it would pin the slice to the un-normalized relu. **The
normative target is miStudio's encode**, since that is the basis the probe's weights were fitted in.
`SaeFeatureSlice.encode` therefore implements normalization and the architecture's activation
itself, and 4A.4 must be reworded to compare against a miStudio-equivalent reference rather than
against `LoadedSAE.encode`.

**Still outstanding, and correctly deferred to acceptance:** the numeric max-abs-diff on a real
LFM2 SAE. That is a measurement, not a design input — the design consequence above is structural
and does not depend on it.

---

## Consequences for the build

| Spike | Changes |
|---|---|
| 0.1 | Identity resolution reads HF download metadata, not `cache_path`. Add a commit-agreement check across files. |
| 0.2 | Drop the `X-miLLM-Probe-Stream: 1` fallback from phase 6. Emit the chunk unconditionally when a probe is armed. |
| 0.3 | Parity scores from `token_ids`; `messages` drift is reported beside the verdict, never inside it. |
| 0.4 | `SaeFeatureSlice.encode` owns normalization + activation. 4A.4's comparison target is miStudio's encode, not `LoadedSAE.encode`. |
