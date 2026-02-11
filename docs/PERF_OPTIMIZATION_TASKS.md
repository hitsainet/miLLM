# miLLM Performance Optimization — Complete Task List

> **Created:** 2026-02-10
> **Implemented:** 2026-02-10 — All Phases 1-3 code complete
> **Purpose:** Transform miLLM into a high-performance OpenAI-compatible inference server while preserving SAE steering/monitoring.
> **Hardware:** NVIDIA RTX 3090 (SM86, 24GB VRAM) on K8s host 192.168.244.61
> **Status:** CODE COMPLETE — Phases 1, 2, 3, 4. Remaining: Dockerfile updates, testing, minor UI.

---

## Implementation Summary

### Files Modified
| File | Changes |
|------|---------|
| `millm/core/config.py` | Added all performance settings (concurrency, torch.compile, KV cache, prefix cache, speculative decoding) |
| `millm/ml/model_loader.py` | FlashAttention-2 auto-detect, GPTQ/AWQ detection, torch.compile, `attn_implementation` + `quantization_method` on LoadedModel |
| `millm/services/sae_service.py` | SAE dtype matching on attach (cast W_enc/b_enc/W_dec/b_dec to model dtype) |
| `millm/api/dependencies.py` | Wired all config settings into InferenceService constructor |
| `millm/services/inference_service.py` | Prefix cache, speculative decoding, KV cache mode, queue concurrency |
| `millm/ml/generation_config.py` | Added `cache_implementation` field |
| `millm/services/model_service.py` | Wired torch_compile + torch_compile_mode through to loader |
| `pyproject.toml` | Added `[project.optional-dependencies] perf` group |

### Files Created
| File | Purpose |
|------|---------|
| `millm/ml/prefix_cache.py` | LRU cache for system prompt KV states, steering-aware invalidation |

---

## Phase 1: Free / Near-Free Wins (~2x throughput)

### Task 1.1: Enable FlashAttention-2
**Impact:** 2-3x faster attention, linear memory scaling
**Hook compatible:** YES — FlashAttention replaces attention *sub-module internals*; hook is on the *layer boundary*, never sees the difference
**Requirement:** RTX 3090 is SM86 (Ampere), fully supported

- [x] **1.1.1** Add `flash-attn` to pyproject.toml dependencies
  - Added to `[project.optional-dependencies] perf` group as `"flash-attn>=2.7.0"`

- [x] **1.1.2** Add `attn_implementation` parameter to `ModelLoadContext.load()`
  - Added to BOTH from_pretrained() calls (primary + trust_remote_code fallback)

- [x] **1.1.3** Add graceful fallback if flash-attn is not installed
  - Auto-detects via `import flash_attn`, falls back to `"sdpa"` (PyTorch native SDPA)
  - Logs which attention backend is being used

- [x] **1.1.4** Add `attn_implementation` field to LoadedModel dataclass
  - Added `attn_implementation: str = "unknown"` field, set during load

- [ ] **1.1.5** Update Dockerfile to install flash-attn
  - flash-attn needs CUDA headers at build time
  - Add: `RUN pip install flash-attn --no-build-isolation` (or use pre-built wheel)
  - Test: Docker build completes, flash-attn importable in container

- [ ] **1.1.6** Test: Load model with FlashAttention-2 → verify SAE steering still works
  - Load a model (e.g., gemma-3-4b-it) with flash attention
  - Attach SAE, enable steering on a feature
  - Generate text and verify steering effect
  - Check logs for `attn_implementation=flash_attention_2`

---

### Task 1.2: SAE Dtype Matching (Eliminate per-forward-pass cast)
**Impact:** Removes dtype conversion on EVERY forward pass token
**Hook compatible:** YES — this IS optimizing the hook

- [x] **1.2.1** Match SAE weights to model dtype during attachment
  - In `sae_service.py` at attach time: casts W_enc, b_enc, W_dec, b_dec to match model dtype

- [x] **1.2.2** Verify the dtype cast check in hook becomes a no-op
  - Hook's `if x.dtype != sae.W_enc.dtype` check is now always False after dtype matching

- [x] **1.2.3** Also match steering delta dtype
  - `_rebuild_steering_delta()` uses W_dec.dtype, which now matches model dtype

---

### Task 1.3: Increase Request Queue Concurrency
**Impact:** 30-50% throughput under concurrent load
**Hook compatible:** YES — hooks fire independently per forward pass, steering is shared state

- [x] **1.3.1** Add inference concurrency settings to config
  - Added `MAX_CONCURRENT_REQUESTS: int = 2` and `MAX_PENDING_REQUESTS: int = 10`

- [x] **1.3.2** Wire config into InferenceService initialization
  - Wired via `dependencies.py` → `get_inference_service()`

- [ ] **1.3.3** Expose queue status in health endpoint
  - File: `millm/api/routes/system/health.py`
  - Add `request_queue: { pending: N, max_concurrent: N, max_pending: N }` to detailed health

- [ ] **1.3.4** Test: Two concurrent requests complete without error

---

## Phase 2: Medium Effort, High Impact (~3-4x decode speed)

### Task 2.1: GPTQ/AWQ Pre-Quantized Model Support
**Impact:** 1.5x FP16 speed with Marlin kernels (vs bitsandbytes which is SLOWER than FP16)
**Hook compatible:** YES — GPTQ/AWQ replace linear layer internals, not layer interface

- [x] **2.1.1** Add dependencies to pyproject.toml
  - Added to `perf` optional group: `auto-gptq>=0.7.0`, `autoawq>=0.2.0`, `optimum>=1.21.0`

- [x] **2.1.2** Detect pre-quantized models in ModelLoadContext.load()
  - Auto-detects via `AutoConfig.from_pretrained()` → `quantization_config.quant_method`
  - Skips bitsandbytes for GPTQ/AWQ models, records quant_method on LoadedModel

- [ ] **2.1.3** Add "PRE_QUANTIZED" as a recognized quantization type
  - Currently model DB stores quantization as enum: FP16, Q8, Q4
  - Add GPTQ and AWQ as recognized types
  - File: `millm/db/models/model.py` — check QuantizationType enum

- [ ] **2.1.4** Update model preview to show quantization info
  - File: `millm/ml/model_downloader.py` `get_model_info()` method
  - Extract `quantization_config.quant_method` from model config if present
  - Return it in the preview response so admin UI shows "This model is GPTQ 4-bit"

- [ ] **2.1.5** Update admin UI to show pre-quantized badge
  - File: `admin-ui/src/pages/ModelsPage.tsx`
  - Show "GPTQ-4bit" or "AWQ-4bit" badge if model is pre-quantized

- [ ] **2.1.6** Update Dockerfile to install auto-gptq
  - `pip install auto-gptq optimum`
  - Test: Docker build succeeds, auto_gptq importable

- [ ] **2.1.7** Test: Load a GPTQ model, verify inference + SAE steering

---

### Task 2.2: Static KV Cache + torch.compile
**Impact:** Up to 4x decode speedup (HuggingFace benchmark)
**Hook compatible:** YES — using `fullgraph=False` preserves hook compatibility

#### Sub-task 2.2a: Static KV Cache

- [x] **2.2a.1** Add `cache_implementation="static"` to generate() calls
  - Added to `generation_config.py` as `cache_implementation` field
  - Injected in `inference_service._build_generate_kwargs()` when KV_CACHE_MODE="static"

- [x] **2.2a.2** Add config setting for cache mode
  - Added `KV_CACHE_MODE: str = "static"` to config

- [ ] **2.2a.3** Test: Static cache with various sequence lengths

#### Sub-task 2.2b: torch.compile

- [x] **2.2b.1** Add TORCH_COMPILE config setting
  - Added `TORCH_COMPILE: bool = False` and `TORCH_COMPILE_MODE: str = "reduce-overhead"`

- [x] **2.2b.2** Apply torch.compile in model loading
  - Applied `torch.compile(model.forward, mode=..., fullgraph=False)` in ModelLoadContext.load()
  - `fullgraph=False` allows hooks and dynamic control flow
  - Skipped for bitsandbytes-quantized models (incompatible)
  - Wired through model_service.py → loader.load()

- [~] **2.2b.3** ~~Implement always-on hook with fast no-op path~~
  - **SKIPPED** — used `fullgraph=False` approach instead, which allows dynamic hooks without permanent hook refactoring

- [~] **2.2b.4** ~~Compilation step~~ (merged into 2.2b.2)

- [ ] **2.2b.5** Add warmup step after compilation
  - Run dummy forward pass at load time to trigger compilation upfront
  - Currently compilation happens on first user request

- [ ] **2.2b.6** Test: torch.compile + SAE steering end-to-end
- [ ] **2.2b.7** Test: torch.compile + static cache combined

---

## Phase 3: Latency Improvements (~2x non-steering latency)

### Task 3.1: Prefix Caching for System Prompts
**Impact:** Skip prefill for repeated system prompts (common in Open WebUI)
**Hook compatible:** YES — cached KV states include steering effects; entries tagged with steering hash for invalidation

- [x] **3.1.1** Design prefix cache data structure
  - Created `millm/ml/prefix_cache.py` — LRU OrderedDict with CacheEntry dataclass

- [x] **3.1.2** Implement PrefixCache class
  - Methods: `get()`, `put()`, `invalidate_steering()`, `clear()`, `get_steering_hash()`
  - Cache keys combine prompt text hash + steering delta hash
  - LRU eviction when max_entries exceeded

- [x] **3.1.3** Integrate prefix cache into inference pipeline
  - Integrated into `inference_service.create_chat_completion()` and streaming
  - System prompt extraction → cache lookup → cache miss stores new entry

- [x] **3.1.4** Cache invalidation on steering change
  - **Approach:** Cache entries are tagged with steering hash (computed from SAE delta tensor)
  - `invalidate_steering(old_hash)` removes entries with stale steering state
  - `get_steering_hash()` static method reads from AttachedSAEState singleton

- [x] **3.1.5** Add config settings
  - Added `ENABLE_PREFIX_CACHE: bool = True` and `PREFIX_CACHE_MAX_ENTRIES: int = 5`

- [ ] **3.1.6** Test: Repeated system prompts are faster on 2nd+ request

---

### Task 3.2: Speculative Decoding (When Steering is Off)
**Impact:** 1.5-3x latency reduction for general (non-steering) inference
**Hook compatible:** PARTIAL — auto-disabled when SAE is attached (draft model runs un-steered)

- [x] **3.2.1** Add config settings for speculative decoding
  - Added `SPECULATIVE_MODEL: Optional[str] = None` and `SPECULATIVE_NUM_TOKENS: int = 5`

- [x] **3.2.2** Manage draft model lifecycle
  - Lazy-loaded in InferenceService via `_get_draft_model()` method
  - Auto-loaded on first use, auto-disabled when SAE is attached

- [x] **3.2.3** Pass assistant_model to generate() when steering is OFF
  - In `_build_generate_kwargs()`: checks `_is_sae_attached()`, only adds draft model when SAE is not attached
  - Auto-disables speculative decoding during steering (safety guarantee)

- [ ] **3.2.4** Test: Speculative decoding speedup without steering
- [ ] **3.2.5** Admin UI: Show speculative decoding status

---

## Phase 4: Production-Grade Throughput

> **Status:** CODE COMPLETE — Dual-mode inference implemented. CBM opt-in via `ENABLE_CONTINUOUS_BATCHING=true`.

### Task 4.1: Continuous Batching via ContinuousBatchingManager

**Architecture:** Dual-mode inference (opt-in). CBM disabled by default. When enabled via
`ENABLE_CONTINUOUS_BATCHING=true`, CBM handles all generation (higher throughput, shared
sampling params). When disabled, existing RequestQueue pipeline runs (full per-request params,
all Phase 1-3 optimizations). SAE steering works in both modes.

**Known limitations:**
- Per-request sampling params (temperature, top_p) are fixed at CBM creation time — only `max_new_tokens` varies per request
- CBM permanently modifies model attention to paged implementation on init
- SAE monitoring captures batch-level activations (acceptable for uniform steering)

- [x] **4.1.1** CBM backend class (`millm/services/cbm_backend.py`)
  - `ContinuousBatchingBackend` encapsulates all CBM interaction
  - Non-streaming: `generate()` via `run_in_executor` + `get_result()`
  - Streaming: `generate_stream()` via background thread + asyncio.Queue bridge
  - Clean lifecycle: `start(model, tokenizer)` / `stop()`

- [x] **4.1.2** InferenceService integration (`millm/services/inference_service.py`)
  - `_use_cbm()` mode check, `on_model_loaded()` / `on_model_unloading()` lifecycle
  - CBM delegation in `create_chat_completion`, `stream_chat_completion`, `create_text_completion`
  - `_cbm_chat_completion()`, `_cbm_stream_chat_completion()`, `_cbm_text_completion()`
  - Embeddings unchanged (CBM is generation-only)

- [x] **4.1.3** Configuration and wiring
  - 5 settings in `config.py`: `ENABLE_CONTINUOUS_BATCHING`, `CBM_MAX_QUEUE_SIZE`, `CBM_DEFAULT_TEMPERATURE`, `CBM_DEFAULT_TOP_P`, `CBM_DEFAULT_MAX_TOKENS`
  - `dependencies.py` passes CBM config to InferenceService singleton

- [x] **4.1.4** Model lifecycle hooks (`millm/services/model_service.py`)
  - `_load_worker`: calls `inference_service.on_model_loaded()` after successful load
  - `unload_model`: calls `inference_service.on_model_unloading()` before unload

- [x] **4.1.5** Health endpoint (`millm/api/routes/system/health.py`)
  - Detailed health now reports: `backend` (cbm/queue), `cbm_enabled`, `cbm_running`, queue stats

- [ ] **4.1.6** Test: Throughput under concurrent load
  - Benchmark: 10, 50, 100 concurrent requests
  - Compare tok/s with old RequestQueue vs ContinuousBatchingManager

### Task 4.2: Paged Attention
**Note:** CBM uses paged attention internally (`model.set_attn_implementation("paged|{impl}")` on init). No separate configuration needed.

- [x] **4.2.1** Paged attention handled internally by ContinuousBatchingManager
- [ ] **4.2.2** Test: Long sequence generation (4096+ tokens) with continuous batching

### Task 4.3: Multi-GPU Support
- [ ] **4.3.1** Verify current device_map="auto" handles multi-GPU
- [ ] **4.3.2** Ensure SAE weights are on correct device
- [ ] **4.3.3** Test: SAE steering on multi-GPU model

---

## Remaining Work Summary

### Deployment Tasks (needed for K8s)
| # | Task | Priority |
|---|------|----------|
| 1 | **Dockerfile: install flash-attn** (1.1.5) | HIGH — needed for FlashAttention-2 on K8s |
| 2 | **Dockerfile: install auto-gptq, optimum** (2.1.6) | MEDIUM — needed for GPTQ model support |
| 3 | **Push + CI + k8s_deploy** | HIGH — deploy all Phase 1-3 changes |

### Backend Polish Tasks
| # | Task | Priority |
|---|------|----------|
| 4 | **Health endpoint: queue status** (1.3.3) | LOW — nice-to-have observability |
| 5 | **torch.compile warmup step** (2.2b.5) | LOW — shifts compilation from first request to load time |
| 6 | **DB enum: GPTQ/AWQ quantization types** (2.1.3) | MEDIUM — for proper pre-quantized model tracking |
| 7 | **Model preview: show quant info** (2.1.4) | MEDIUM — helps users identify GPTQ/AWQ models |

### Frontend/UI Tasks
| # | Task | Priority |
|---|------|----------|
| 8 | **Pre-quantized badge in models page** (2.1.5) | LOW |
| 9 | **Speculative decoding status display** (3.2.5) | LOW |

### Testing Tasks
| # | Task | Priority |
|---|------|----------|
| 10 | **FlashAttention-2 + SAE steering** (1.1.6) | HIGH — verify on K8s with GPU |
| 11 | **Concurrent requests** (1.3.4) | MEDIUM |
| 12 | **Static cache various lengths** (2.2a.3) | MEDIUM |
| 13 | **torch.compile + SAE** (2.2b.6) | MEDIUM |
| 14 | **torch.compile + static cache** (2.2b.7) | MEDIUM |
| 15 | **Prefix cache latency** (3.1.6) | MEDIUM |
| 16 | **Speculative decoding speedup** (3.2.4) | MEDIUM |
| 17 | **GPTQ model load + SAE** (2.1.7) | LOW — needs a GPTQ model downloaded |

---

## Configuration Summary

### New Settings (`millm/core/config.py`)

```python
# Performance — Phase 1
MAX_CONCURRENT_REQUESTS: int = 2        # Was hardcoded to 1
MAX_PENDING_REQUESTS: int = 10          # Was hardcoded to 5

# Performance — Phase 2
TORCH_COMPILE: bool = False             # Enable torch.compile
TORCH_COMPILE_MODE: str = "reduce-overhead"
KV_CACHE_MODE: str = "static"           # "static" or "dynamic"

# Performance — Phase 3
ENABLE_PREFIX_CACHE: bool = True
PREFIX_CACHE_MAX_ENTRIES: int = 5
SPECULATIVE_MODEL: Optional[str] = None
SPECULATIVE_NUM_TOKENS: int = 5

# Performance — Phase 4 (Continuous Batching)
ENABLE_CONTINUOUS_BATCHING: bool = False  # Opt-in
CBM_MAX_QUEUE_SIZE: int = 256
CBM_DEFAULT_TEMPERATURE: float = 0.7
CBM_DEFAULT_TOP_P: float = 0.95
CBM_DEFAULT_MAX_TOKENS: int = 512
```

### New Dependencies (`pyproject.toml`)

```toml
[project.optional-dependencies]
perf = [
    "flash-attn>=2.7.0",
    "auto-gptq>=0.7.0",
    "autoawq>=0.2.0",
    "optimum>=1.21.0",
]
```

---

## Steering Compatibility Matrix

| Optimization | Steering ON | Steering OFF | Monitoring ON | Notes |
|---|---|---|---|---|
| FlashAttention-2 | YES | YES | YES | Transparent to hooks |
| SAE dtype match | YES | N/A | YES | Optimizes the hook itself |
| Queue concurrency | YES (shared) | YES | YES | All requests share same steering |
| GPTQ/AWQ models | YES | YES | YES | Linear layer internals only |
| Static KV cache | YES | YES | YES | Internal to attention |
| torch.compile | YES (fullgraph=False) | YES | YES | Allows dynamic hooks |
| Prefix cache | YES (invalidate on change) | YES | YES | Cache includes steering effects |
| Speculative decode | AUTO-DISABLED | YES | AUTO-DISABLED | Draft model is un-steered |
| Continuous batching | YES (uniform) | YES | YES | Per-request needs batch-aware hook |
| Paged attention | YES | YES | YES | Internal to attention |
| Multi-GPU | YES (match device) | YES | YES | SAE on same GPU as target layer |

---

## Verification Checklist (Run After Each Phase)

- [x] Model loads successfully with new optimizations enabled (verified imports + construction)
- [ ] Chat completion (non-streaming) returns correct output
- [ ] Chat completion (streaming) streams tokens correctly
- [ ] Text completion works
- [ ] Embeddings endpoint works
- [ ] SAE can be attached and detached without errors
- [ ] Steering produces visible effect on generation
- [ ] Monitoring captures feature activations
- [ ] Multiple models can be loaded/unloaded (auto-load still works)
- [ ] Admin UI shows correct model status
- [ ] No CUDA OOM under normal workloads
- [ ] Logs show expected optimization features active
- [ ] K8s deployment works (Docker build + deploy)
