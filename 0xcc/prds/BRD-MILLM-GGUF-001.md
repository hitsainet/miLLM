# BRD: GGUF Serving

**Document ID:** BRD-MILLM-GGUF-001
**Status:** Implemented and serving in production (2026-09-07/08)
**Related:** Feature 23 · files `023_*` · supersedes the v1.0 out-of-scope entry
**Supersedes:** PPRD §4 "Out of Scope (Version 1.0)" — *GGUF model format · Focus on
Transformers ecosystem · v1.1+* — and FPRD 001 §"GGUF format support · Requires a
different inference engine"

---

## 1. Why this document exists at all

**The doc chain said GGUF was out of scope while GGUF models were serving
production traffic.** Four of the seventeen models this deployment currently
serves are GGUF files, including the judge that does miStudio's feature
labeling. The v1.0 scope decision was made on a premise that turned out to be
wrong — that GGUF "requires a different inference engine" and therefore a
different product — and the work was done incrementally without the chain ever
being reconciled.

That gap is the same failure this project keeps recording in the other
direction: documentation and reality disagreeing, with nothing to force the
question. This BRD is written after the fact to close it.

## 2. Why GGUF

miLLM's stated goal is to be *"a fully functional offline inference server"*.
GGUF is how the local-inference ecosystem actually distributes weights — it is
what Ollama serves, what llama.cpp reads, and what most community quantizations
of a large model are published as. A server that cannot open a `.gguf` file is
not offline-capable in the way its users mean.

The concrete forcing case: labeling 53,000 SAE features needs a strong judge on
one 24 GB card. A 31B model at bf16 does not fit; the same model as an `IQ4_XS`
GGUF does, at ~16 GB, leaving room for the KV cache.

## 3. Business requirements

**BR-001 — Serve GGUF files through the same OpenAI-compatible surface** as
HuggingFace checkpoints, with no separate endpoint, client or code path for
callers.

**BR-002 — Several quantizations of one repository may coexist and be
individually selectable.** They are different models with different sizes and
different quality.

**BR-003 — A model is never chosen ambiguously.** Where a name could mean more
than one quantization, the server says so and names the alternatives rather than
picking by insertion order.

**BR-004 — The context window is derived from the model and the hardware,** not
configured by guesswork. A model trained for a large window must not be served a
small one silently.

**BR-005 — VRAM is spent deliberately.** Where a setting materially changes how
much context fits on a given card, the trade is measured and the default is
justified.

**BR-006 — A client error is reported as a client error.** A request that can
never succeed must not be returned as a server fault that invites retries.

**BR-007 — Optional capabilities degrade; serving does not.** If a model cannot
support an optional feature, load it without that feature and say so, rather
than refusing to serve it.

**BR-008 — Standard client affordances work.** Continuing a truncated answer
resumes it.

## 4. Non-goals

Training or fine-tuning GGUF; converting between formats; SAE attachment to a
GGUF model (the hook points differ and llama.cpp does not expose them — steering
and interpretability remain transformers-only).

## 5. Acceptance

Live, on the RTX 3090 deployment, and all met:

* Four GGUF models served concurrently in `/v1/models` alongside transformers
  models, two of them different quantizations of the same repository.
* A 31B GGUF judge completed a real 20-feature labeling batch through the
  OpenAI-compatible API.
* `gemma-4-31b IQ4_XS` loads at a **12,288-token** window on a 24 GB card,
  where the same file managed 4,096 before the KV-cache work.
* An oversized prompt returns 400 `context_length_exceeded`, not 500.
* Open WebUI's **Continue** resumes mid-sentence.

## 6. What this cost, recorded honestly

Every requirement above exists because something failed first.

* **BR-003** — widening the uniqueness constraint without widening the *name*
  gave two rows the same name, and `find_by_name` raised on every OpenAI request
  for a model that was loaded and serving. **Three review rounds missed it**
  because no fixture ever held two rows from one repository.
* **BR-004/BR-005** — the loader searched downward from a fixed ceiling, one
  full model load per rung. The window a card could hold was computable from
  metadata the loader already read.
* **BR-006** — a 500 on an oversized prompt made miStudio's labeling loop retry
  each impossible request three times.
* **BR-007** — a pooling flag added *for* embeddings was refused by an
  architecture and failed context creation at every length, producing advice
  ("try a smaller quantization") that could not work.
