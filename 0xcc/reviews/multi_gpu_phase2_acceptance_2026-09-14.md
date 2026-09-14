# Multi-GPU Phase 2 — hardware acceptance (miLLM)

**Date:** 2026-09-14, 16:53–17:25 UTC (12:53–1:25 PM ET). GPU work 16:59–17:25 UTC (12:59–1:25 PM ET), about 27 minutes.
**Node:** mcs-lnxhost02. cuda:0 = RTX 3080 Ti 12 GB (`GPU-f47ba814-49a2-603f-3595-275284140251`), cuda:1 = RTX 3090 24 GB (`GPU-247aa582-0d1b-e161-8156-983ed1fefc57`). Driver 580, `CUDA_DEVICE_ORDER=PCI_BUS_ID`.
**Image:** `hitsai/millm-backend@sha256:9cbca7c7c2fad59f8fb142bb5ed2ab6e04abca6d87ae67935b8971693de21a0f`, origin/main `6ca6121`. One process (pid 1943544) throughout; no restart.
**Deployment settings that bound what could be observed:** `TORCH_COMPILE=false`, `ENABLE_CONTINUOUS_BATCHING=false`, no `SPECULATIVE_MODEL`. `TRANSFORMERS_MIN_CONTEXT` 4096 and `TRANSFORMERS_CUDA_CONTEXT_MB` 500 (both defaults, as logged in every fit).
**Method:** miLLM's API through `kubectl exec … curl localhost:8000`; `nvidia-smi` on the host, including a 200 ms poller during long requests; backend logs. Prompt lengths were sized with each model's own tokenizer and chat template, and the server's `prompt_tokens` matched them exactly. No code, manifest, env or deployment was changed.

**Models used:** LFM2.5-1.2B-Instruct (id 1, pre-existing); Qwen/Qwen2.5-7B-Instruct FP16 (id 2, downloaded 12:58 PM ET, 14,536 MB); allenai/OLMo-2-1124-13B-Instruct FP16 (id 3, downloaded, 26,170 MB). Also a temporary local Q8 row (id 4) registered over OLMo's FP16 directory for item 10, and deleted afterwards (a `source=local` delete removes no files; the directory was verified intact, 26,172 MB). SAEs downloaded through the API: `Geaming/Qwen2.5-7B-Instruct_SAEs` FAST layer 4 and layer 25 JumpReLU (d_sae 28,672), and `chanind/qwen2.5-7B-it-layer-20-saes` lmsys matryoshka (d_sae 65,536). Kept: ids 2 and 3 and the three SAEs.

## Result: 9 PASS (3 with parts NOT RUN), 3 FAIL (items 4, 7, 11)

| # | Check | Result | Evidence (ET) |
|---|---|---|---|
| 1 | CUDA context on each card | **PASS** (measured) | LFM2.5 by UUID on the 3090 at 1:00 PM: process 2,492 MiB for 2,236 MB of weights, so **256 MiB** idle; **334 MiB** after one short chat. The earlier cuda:0 figures were 250 and 330 MiB. After unload the process kept **328 MiB** on cuda:0 (12:59 PM) and **334 MiB** on cuda:1 (1:01 PM). Over the session these leftovers grew to 394 and 400 MiB. |
| 2 | Auto, UUID, index, unknown card | **PASS** | Auto at 1:01 PM: `reason most_free_card_fits, devices [cuda:1]`, and cuda:0 stayed at 328 MiB. A context already existed there from earlier, so this checks "no growth", not "no context". Index 0: `requested_card, requested 0, devices [cuda:0]`. With LFM2.5 loaded, at 1:04 PM on ready row 2: `404 GPU_NOT_FOUND` for `GPU-00000000-…` and for index 5, with both cards listed. LFM2.5 kept serving (200). My first attempt at 1:01 PM hit a still-downloading row (409) and the already-loaded row (400), which run before the card check; I redid it and the redo is what counts. |
| 3 | Equivalence, Qwen2.5-7B FP16, 3090 vs `all` | **PASS**; compile/CBM/draft **NOT RUN** | The 3090 by UUID at 1:04 PM, then `all` at 1:06 PM: `transformers_fit_split_accepted` 20 / 8 layers, `passes 1`. Preflight map 9,930 / 4,596 MiB; landed 9,922 / 4,594. Greedy output (temperature 0, max_tokens 96) was **byte-identical on all 3 prompts**, including token counts (51, 96, 94). The API has no logprobs, so text was the only comparison. Unloading the split at 1:13 PM returned both cards to context only (386 / 392 MiB). The split-specific compile and CBM skip events sit behind settings that are off on this node, so they could not fire. No `torch_compile_*`, `cbm_*` or `draft_*` event appeared all session, and health reports `cbm_enabled: false`. |
| 4 | Capacity, OLMo-2-13B bf16 on Auto | **FAIL** | At 1:14 PM, free 11,767 / 23,976 MiB: `no_single_card_fits`, **15 / 25 layers**, `passes 1`. At these free figures the fit gives cuda:0 need 11,756 of 11,767, **11 MiB spare**. The docs' 14 / 26 layers and 429 MiB spare assumed 11,500 free. Preflight map 10,056 / 16,106 MiB; landed 10,074 / 16,124. nvidia-smi 10,460 / 16,516 = landed + the existing context, within 18 MiB. **A request inside the admitted context, 3,879 prompt + 217 max_tokens, ran cuda:0 out of memory**: `generation_out_of_memory` at 17:14:11Z, "Tried to allocate 76.00 MiB … 52.62 MiB is free", poller peak 12,106 MiB on cuda:0. See Failure 1. `hf_device_map` is not exposed by the API: equality is shown per card in MiB (transformers' own map for the logged `max_memory` equals the landed memory), not layer by layer. |
| 5 | Rebalance loads as logged | **PASS** (on a Q8 split); FP16 **NOT RUN** | The Q8 OLMo row with `gpu: all` at 1:22 PM: `passes 2, lowered_limits_mb {cuda:0: 11225}`, 26 / 14 layers. `split_load_memory_map` used `max_memory {"0": "11225MiB", "1": "23476MiB"}`, the lowered limit, and `split_preflight_mapped` equals the fit's per-card weights exactly (8,847 / 5,216). No FP16 plan needed a re-plan: free on cuda:0 is always 11,767 after an unload. The window where a 15-layer OLMo map lands but its card is short is about [11,536, 11,756) MiB free, and there was no way to reach it within the rules. Loading LFM2.5 first does not help, because `load_model` unloads the resident model before the authoritative check. |
| 6 | Refusals before unload | **PASS** (507); 409 and `rebalance_passes` **NOT RUN** | With the Qwen split resident, at 1:08 PM: OLMo on index 0 → `507 INSUFFICIENT_MEMORY`, `short_devices [cuda:0]`, 18,067 MiB short, `before_loading: true`, with `per_card`, `min_context_tokens` 4096 and `model_max_context_tokens` 4096. Index 1 → 507, cuda:1 5,856 MiB short. Qwen stayed loaded split and served identical greedy output. **Not triggered:** `409 SPLIT_NOT_HONOURED` needs `all` to leave a card empty, but the pre-unload projection returns the resident's memory, so no miLLM model can fill the 3080 Ti. My LFM2.5 `all` attempt at 1:06 PM was honoured (2 layers on cuda:0, 14 on cuda:1), which is correct but refused nothing, and it unloaded Qwen as a real load would. A 507 carrying `rebalance_passes` needs weights that fit the cards while their KV cache does not; none of the models on the node falls in that window. |
| 7 | Context cap → 400 | **FAIL** | Qwen, 32,699 + 512 > 32,768: **HTTP 500 `server_error`** non-streaming (1:07 PM). Streaming (1:08 PM): **HTTP 200, empty body, curl exit 18** (no error event, no `[DONE]`). OLMo, 4,272 + 64 > 4,096: **500** (1:14 PM). See Failure 2. |
| 8 | SAE attach on a split model | **PASS** | Qwen `all`, cuda:0 holds layers 0–19. At 1:11 PM an attach-set of three SAEs on cuda:0 layers → `507 INSUFFICIENT_MEMORY`, `device cuda:0, projected_mb 2218, kv_reserve_mb 160, kv_context_tokens 4096, available_mb 1871`; attachments stayed empty. No single SAE on hand exceeds the room, so the oversized case is a set. The layer-4 SAE at layer 4 (`model.layers.4`, cuda:0) attached, 784 MB. With steering on feature 100 at 8.0, a 3,893-prompt-token request returned 200 (88 tokens generated), poller peak cuda:0 11,572 MiB, **no OOM**. Detach returned cuda:0 to 10,296 MiB. |
| 9 | Out of memory while generating | **PASS** | Qwen `all`, prompt of 15,993 tokens + 256. Non-streaming at 1:09 PM: **503**, `type invalid_request_error, code insufficient_memory`, naming **cuda:0 (RTX 3080 Ti)**, the prompt tokens and max_tokens. Process on cuda:0: 10,294 MiB before, 11,868 MiB peak, **10,288 after**. The next request returned 200. Streaming ×3 at 1:11 PM: each stream ended with the error event and then `data: [DONE]`. cuda:0 was 10,296 MiB after each (+2 MiB), the next request returned 200, and health was 200. |
| 10 | bitsandbytes split | **PASS** | Local Q8 row over OLMo's FP16 files, `gpu: all`, at 1:22 PM. There was no CPU staging refusal. `bitsandbytes: true` in `split_load_memory_map`. The map equals the fit's layout (8,847 / 5,216 MiB, 26 / 14 layers). The model served (200). Two notes, both part of Failure 1: the landed memory is **9,500 / 5,660 MiB, 653 / 444 MiB above the map** (FP16 agreed within 18 MiB); and a request of 3,879 + 16 tokens **ran cuda:0 out of memory** at 1:24 PM (peak 12,136 MiB). |
| 11 | Unload and switch | **FAIL** | Switch OLMo FP16 → Q8 at 1:22 PM: `gpu_memory_cleanup` freed 10,074 / 16,124 MiB, down to 391 / 397. Every unload returned both cards to context only. **A `/v1` request during an unload is not told to retry**: while the Q8 split unloaded (17:24:13–17:24:21Z), requests at +1 s and +3 s both got **HTTP 500 `server_error`**, `RuntimeError: Expected all tensors to be on the same device, but got index is on cuda:0, different from other tensors on cpu` in `embed_tokens`. See Failure 3. Health stayed 200 and memory still returned. |
| 12 | Logs | **PASS** | Sweep of the whole session (6,016 lines): **no** `transformers_fit_falls_back_to_slack`, **no** `*_engine_failed`, **no** `split_preflight_skipped`. Also no `torch_compile_*`, `cbm_*` or `draft_*` (disabled). There were 6 `generation_out_of_memory`: 4 deliberate (item 9), plus the item 4 and item 10 failures. There were 5 `unhandled_exception`: 3 from item 7 and 2 from item 11. |

## Failures

### Failure 1 — the per-card fit admits a context its card cannot run (items 4, 10)

The fit accepts a card when `free ≥ weights + KV(TRANSFORMERS_MIN_CONTEXT) + TRANSFORMERS_CUDA_CONTEXT_MB`. On this node the 500 MB allowance is far below what a long prefill needs beyond the weights and the KV cache.

- **OLMo-2-13B FP16 on Auto (1:14 PM):** cuda:0 was admitted with 11 MiB to spare. A 3,879 + 217 request failed at 17:14:11Z. Torch reported 11.22 GiB allocated and 285 MiB reserved but unallocated, in a process holding 11.82 GiB. So at the failure, the context, prefill transients and fragmentation already came to at least ~900 MiB above the weights and the prompt's KV cache, and the pass wanted more.
- **The same split, three requests that succeeded (1:18–1:20 PM):** peak on cuda:0 minus the weights (10,074) minus the KV cache of the full request on its 15 layers:

  | Request (prompt + max_tokens) | cuda:0 peak | Overhead beyond weights + KV |
  |---|---|---|
  | 977 + 96 | 10,924 MiB | ~536 MiB |
  | 1,981 + 96 | 11,626 MiB | ~943 MiB |
  | 2,987 + 96 | 12,106 MiB | ~1,129 MiB |

  Extrapolated to 4,096 tokens this is ~1,300 MiB.
- **Qwen2.5-7B split, steered ~4k request:** overhead ~525 MiB on cuda:0. So the overhead depends on the model. OLMo-2 has no GQA and an RMSNorm on q/k that upcasts to float32; the failing 76 MiB allocation is the size of one such float32 activation at 3,879 tokens × 5,120 × 4 bytes.
- **OLMo-2-13B Q8 split (1:24 PM):** the landed weights were already 653 MiB above the map on cuda:0, and a 3,879 + 16 request ran out of memory there ("Tried to allocate 104.00 MiB … 22.62 MiB is free").

Every OOM was handled as designed (typed 503, memory released, the next request 200). But the load's promise that each card holds a `TRANSFORMERS_MIN_CONTEXT` context is not kept for these models.

**Also seen:** after a *successful* request the torch cache is not released. cuda:0 went 10,484 → 10,924 → 11,626 → 12,004 MiB across the three requests above, leaving 155 MiB free, until the unload. Only the OOM path empties the cache. Another tenant's Auto placement would not see that memory as free.

### Failure 2 — a request past the model's context is a 500, not a 400 (item 7)

`InferenceService._check_context_length` (`millm/services/inference_service.py:2491`) raises a bare `ValueError`. The `ContextLengthExceededError` (400, `millm/core/errors.py:59`) exists and is used for llama.cpp, but not here. The function even imports `context_length_exceeded_error` and never uses it.

- Non-streaming (`create_chat_completion`, line 3076) reaches the client as **500**.
- Streaming (line 3789) raises inside the generator after the 200 headers, so the client gets **a truncated stream with no error event and no `[DONE]`**.

Log: `unhandled_exception ValueError "Context length exceeded: 32699 prompt + 512 max_tokens = 33211 > 32768"` at 17:07:14Z and 17:08:43Z, and `"4272 prompt + 64 max_tokens = 4336 > 4096"` at 17:14:07Z. The raise dates from `de474c63` (2026-02-07); round 5 (`7571cab`) changed only how the maximum is read. So this predates Phase 2, but the checklist item fails.

### Failure 3 — a `/v1` request during an unload runs on a half-moved model (item 11)

`ModelService.unload_model` keeps the row LOADED, and the loader reporting the model, until `_unload_worker` returns. Meanwhile `LoadedModelState.clear()` (`millm/ml/model_loader.py:303`, via `ModelLoader.unload` at 3659) moves the weights `.to("cpu")` first. accelerate logs "You shouldn't move a model that is dispatched using accelerate hooks." A chat arriving in that window passes the route's `model_info.name == request.model` check. It is dispatched into `generate()` and fails in `embed_tokens` with the device-mismatch `RuntimeError` above, reaching the client as 500.

The "still being unloaded; retry" path added in round 4 (`load_model_and_wait` → `ModelBusyError`) only covers the moment after the loader is cleared and before the row is written. It never covers a request for the model being unloaded.

Log: 17:24:14.146Z and 17:24:16.134Z, request lines at 17:24:14.094Z, `unload_started` at 17:24:13.082Z. Seen on a split model only; a single-card model was not tried.

## CUDA context and `TRANSFORMERS_CUDA_CONTEXT_MB`

| | Idle after load | After a generation | Left after unload |
|---|---|---|---|
| cuda:0 (3080 Ti) | 250 MiB | 330 MiB | 328 MiB, growing to 394 MiB by the end of the session |
| cuda:1 (3090) | 256 MiB | 334 MiB | 334 MiB, growing to 400 MiB |

The CUDA context alone is ~330–400 MiB per card. The allowance also has to cover prefill activations and fragmentation, measured above at ~525 MiB (Qwen2.5-7B, ~4k) up to an extrapolated ~1,300 MiB (OLMo-2-13B, 4,096).

**Recommendation: `TRANSFORMERS_CUDA_CONTEXT_MB=1500` on this node** while `TRANSFORMERS_MIN_CONTEXT` is 4096. At today's free memory:
- OLMo-2-13B on Auto would need 12,756 MiB on cuda:0 against 11,767 free, so it would be re-planned with layers moved to the 3090 (5,370 MiB spare there today).
- Qwen2.5-7B `all` would still be accepted (11,590 of 11,825).

A constant cannot be right for every model size. Sizing the prefill activations per model (hidden and intermediate size × the admitted context, float32 norm upcasts included) would replace the guess.

## Final node state (1:25 PM ET)

- **Loaded:** exactly one model, LFM2.5-1.2B-Instruct (id 1), `requested_card` `GPU-f47ba814-…` (cuda:0), 2,232 MB. It served a chat (200).
- **SAEs:** none attached.
- **Health:** `/api/health/detailed` healthy, `model_loaded: true`, `cbm_enabled: false`.
- **Memory:** miLLM process 2,626 MiB on cuda:0 (2,566 at the start) and **400 MiB on cuda:1**. That is the CUDA context this run created on the 3090; only a process restart frees it, and restarts were out of scope.
- **Rows:** 1 loaded; 2 (Qwen2.5-7B FP16) and 3 (OLMo-2-13B FP16) ready. The failed `google/gemma-4-31B-it` row was not touched.
- **miStudio:** no `train_sae` / `extract` lines all session, and no miStudio process on either card.

## Other observations (not checklist items)

- **SAE compatibility warning compares the SAE's model name with the model's cache path:** "SAE was trained on 'Qwen/Qwen2.5-7B-Instruct', current model is '/data/model_cache/huggingface/Qwen--Qwen2.5-7B-Instruct--FP16'". It fires for the right model.
- **`SAEConfig._parse_config` reads only top-level `hook_layer` / `hook_name`.** chanind's SAELens 6 `cfg.json` nests `hook_name` under `metadata`, so its layer would parse as 0. Read in code, not verified on the node; attach takes an explicit layer.
- **An unload of a split FP16 model takes 8–15 s** (Qwen 8.5 s, OLMo 15.3 s), against 2 s for LFM2.5 on one card.
