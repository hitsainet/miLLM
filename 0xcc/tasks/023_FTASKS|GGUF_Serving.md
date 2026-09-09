# Tasks: GGUF Serving
## miLLM Feature 23

**Source:** BRD-MILLM-GGUF-001 · FPRD 023 · FTDD 023 · FTID 023
**Status:** ✅ COMPLETE — serving in production since 2026-09-08

---

## Phase 1 — Serve the format at all

- [x] 1.1 llama.cpp loader path behind the existing model-management surface (`005722d`)
- [x] 1.2 `chat_template_kwargs` ignored and logged rather than refused (`005722d`)
- [x] 1.3 Embeddings on GGUF by the same method as transformers (`2bffe08`)

## Phase 2 — Make it loadable

- [x] 2.1 Bound the context, or a large-context model cannot load at all (`6767fb6`)
- [x] 2.2 Load at the largest context that fits instead of failing (`3b26aef`)
- [x] 2.3 Oversized prompt → 400, not 500 (`a61da16`)
- [x] 2.4 Size the context to the model, and drop embeddings rather than the model (`1e1bbf8`)

## Phase 3 — Make it addressable

- [x] 3.1 Name GGUF models `repo:QUANT` (`6ea1888`)
- [x] 3.2 `find_by_name`: exact → bare-if-unique → `AmbiguousModelNameError`
- [x] 3.3 Map the error to 400 `invalid_request_error`
- [x] 3.4 Migration 015 backfill — data-only, idempotent, up/down/up verified
- [x] 3.5 Correct the context-overflow refusal, which named a path that 404s (`e94915e`)

## Phase 4 — Make it fit

- [x] 4.1 Quantize the KV cache — 3× the context on the same card (`95deddb`)
- [x] 4.2 Refuse quantized KV without flash attention
- [x] 4.3 Compute the context that fits instead of discovering it (`9266818`)

## Phase 5 — Make it usable

- [x] 5.1 Continue a truncated answer instead of restarting it (`68906e8`)

## Phase 6 — Documentation

- [x] 6.1 Manual: GGUF woven through hardware, model-management, configuration,
      both API pages, troubleshooting and the Open WebUI tutorial (`1178c4b`)
- [x] 6.2 `0xcc` chain: BRD-MILLM-GGUF-001, PPRD Feature 23, PADR §10 decisions,
      FPRD/FTDD/FTID/FTASKS 023
- [x] 6.3 **Correct the v1.0 scope documents**, which said GGUF was out of scope
      while GGUF served production traffic — PPRD §4 and FPRD 001

---

## Verification

**Measured on the RTX 3090, not estimated.**

| what | result |
|---|---|
| Context prediction vs reality | 4 of 4 measurements bracketed correctly |
| `gemma-4-31b IQ4_XS` window | 4,096 (f16) → **12,288** (q8_0) |
| `q4_0` window | 16,384 — rejected as default on accuracy |
| Embeddings cost | 113.5 → 105.9 tok/s (6.7%), no VRAM change |
| Real workload | 31B GGUF judge completed a 20-feature labeling batch |
| Models served together | 17, of which 4 GGUF, 2 sharing a repository |

**Mutation controls (naming, `6ea1888`)** — five, each verified biting: dropping
the `:QUANT` suffix; returning the first candidate instead of raising; dropping
the bare-name fallback; removing `AMBIGUOUS_MODEL_NAME` from `ERROR_STATUS_MAP`;
unregistering the `MiLLMError` handler.

The first of those is why the naming test runs `download_model` rather than
asserting over two pre-named rows — **that version was true by construction of
its own fixture and survived deleting the naming code entirely.**

## Known limitations

- **No SAE attachment, steering or probe monitoring on GGUF models.** llama.cpp
  does not expose the hook points. This is a boundary, not a backlog item.
- Reasoning-trace handling differs by template; see `f9d8b5a`.
- `q4_0` remains available but is not recommended where the model must
  discriminate.

## Relevant Files

- `millm/ml/model_loader.py` — `declared_context`, `predicted_max_context`,
  `_kv_cache_kwargs`, `_is_context_related`, `_bisect_upward`, `supports_embeddings`
- `millm/services/model_service.py` — `repo:QUANT` naming at download
- `millm/db/repositories/model_repository.py` — `find_by_name` resolution order
- `millm/core/errors.py` — `AmbiguousModelNameError`, `ContextLengthExceeded`
- `millm/core/config.py` — the four `GGUF_*` settings, each with its rationale
- `millm/services/inference_service.py` — `_llamacpp_continuation_prompt`,
  `_render_chat_template`, `_completion_as_chat`, `_completion_chunks_as_chat`
- `millm/db/migrations/versions/015_tag_gguf_model_names.py`
- `tests/unit/db/test_model_name_disambiguation.py` — the fixture whose absence
  let three review rounds miss the collision
- `manual/docs/{getting-started/hardware,features/model-management,reference/configuration,api/models,api/openai-compatible,troubleshooting,tutorials/open-webui}.md`
