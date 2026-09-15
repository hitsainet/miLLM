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
| 12 | Logs | **PASS** (unchanged by the fixes below) | Sweep of the whole session (6,016 lines): **no** `transformers_fit_falls_back_to_slack`, **no** `*_engine_failed`, **no** `split_preflight_skipped`. Also no `torch_compile_*`, `cbm_*` or `draft_*` (disabled). There were 6 `generation_out_of_memory`: 4 deliberate (item 9), plus the item 4 and item 10 failures. There were 5 `unhandled_exception`: 3 from item 7 and 2 from item 11. |

## Fixes (2026-09-14, after this run; deployed as `58082c7` and re-run on the node — see "Re-run on the node" below)

| Finding | Commits | Fix |
|---|---|---|
| Failure 1 (items 4, 10) | `6ce5c38`, `80a9aef` | Each card now also holds a request's **working memory**, traced from the model, and a bitsandbytes load's staging. Torch's unused cache goes back to the cards once the request queue has been idle for `TRANSFORMERS_IDLE_CACHE_RELEASE_S` (5 s). |
| Failure 2 (item 7) | `2c84fe8` | A request past the model's context is a 400 on every route. A stream is refused before its 200 is sent. |
| Failure 3 (item 11) | `ee20b11` | A request for a model being unloaded is refused with 503 `model_busy` ("retry once the unload finishes"), and nothing runs on its moving weights. |

**Failure 1 in figures.** A card's working memory is T + ⌈0.40 × (T + KV)⌉. T is the transient peak of the phases that card runs, traced by running the model's forward pass on the meta device. 0.40 is the caching allocator's share (see `millm/ml/working_memory.py`). The fit's plans at this run's free memory (11,767 / 23,976 MiB), computed with the committed code:
- **OLMo-2-13B FP16 on Auto:** 2 passes, **13 / 27 layers**. cuda:0: 8,846 MiB of weights + 1,040 of KV + 1,151 of working memory + 500 of context, **230 MiB spare** (this run: 15 / 25 layers, 11 spare).
- **OLMo-2-13B Q8 with `all`:** 2 passes, **19 / 21 layers**. cuda:0 counts 805 MiB of staging (the node landed 653 above the map on 26 layers), 634 MiB spare (this run: 26 / 14).
- **Qwen2.5-7B FP16 with `all`:** 1 pass, **18 / 10 layers**, cuda:0 1,241 MiB spare (this run: 20 / 8). Item 3's layout will differ on a re-run.

The fit's allowance against what this run measured on OLMo-2-13B's cuda:0 (overhead beyond weights and KV, CUDA context included): 650 / 946 / 1,245 MiB against 536 / 943 / 1,129 for the three successful requests (+21%, +0.3%, +10%), and 1,545 against the extrapolated ~1,300 at 4,096 tokens (+19%). This adds the smallest measured context (330 MiB) to the fit's working memory. It is never below the node. Qwen2.5-7B's steered request is +120% (1,153 against ~525, measured with an SAE attached), which is unexplained and errs toward refusing. A test pins the three OLMo figures.

The idle cache release takes the request queue's slot, so it never runs during a request. It does not run while continuous batching is on, whose requests hold no queue slot (a defect found and fixed in review before commit). It reads memory only on the model's own cards.

**`TRANSFORMERS_CUDA_CONTEXT_MB` stays 500.** The recommendation of 1,500 below is superseded: it was sized to also cover prefill activations and fragmentation, and those are now counted per card, so 1,500 would count them twice (about 1 GiB a card for OLMo-2-13B). Lowering it to 400 is not recommended either, for two reasons. The context grew from 250 to 400 MiB within this session with no ceiling observed. And at 2,077 tokens the fit's working memory plus the smallest measured context covers the node by only 3 MiB, so the setting's 100–170 MiB above the measured context is the real margin there.

**Re-run on the node after deploying, in this order:**
1. **Item 4.** OLMo-2-13B FP16 on Auto. Expect `transformers_fit_split_accepted` with 13 / 27 layers and 230 MiB planned spare on cuda:0 (at 11,767 / 23,976 free). Expect the same 3,879 + 217-token request to return 200. Then, 5 s after the last request, expect `idle_cache_released` and cuda:0 back near landed weights plus context in nvidia-smi.
2. **Item 7.** Expect Qwen2.5-7B 32,699 + 512 and OLMo-2-13B 4,272 + 64 to return 400, non-streaming and streaming.
3. **Item 10.** OLMo-2-13B Q8 with `all`. Expect 19 / 21 layers, landed memory within the map plus the counted staging, and the 3,879 + 16-token request to return 200.
4. **Item 11.** Unload a split model with `/v1` requests arriving during the unload. Expect 503 `model_busy`, no 500, and memory returned.

## Re-run on the node: items 4, 7, 10 and 11 PASS

**When:** 2026-09-14, 20:41–20:49 UTC (4:41–4:49 PM ET), after `58082c7` deployed through GitOps (CI green on both repos).
**Image:** `hitsai/millm-backend@sha256:ecc509d3113d1dfd1318da1813441aa883050b12686b035ee697a8de903e305a`, pod started 4:40 PM ET. The miStudio SAE training on the 3090 had finished at 4:28 PM, and no GPU lease or job was active.
**Method:** as in the first run. The request bodies are the first run's files, with the same prompt token counts (the server's `prompt_tokens` matched: 3,879, 4,272, 32,699). A 200 ms `nvidia-smi` poller ran on the host for the whole re-run.

| # | Check | Result | Evidence (ET) |
|---|---|---|---|
| 4 | Capacity, OLMo-2-13B bf16 on Auto | **PASS** | At 4:41 PM, free 12,156 / 23,978 MiB (the new pod held no context yet): `no_single_card_fits`, `transformers_fit_split_accepted` **13 / 27 layers**, 4 passes, `lowered_limits_mb {cuda:0: 10235}`. cuda:0: 8,846 weights + 1,040 KV + 1,151 working memory (`method traced`) + 500 context = 11,537 of 12,156, **619 MiB spare**. The planned 230 assumed 11,767 free. Landed 8,848 / 17,320 MiB. **The 3,879 + 217-token request returned 200** in 16.5 s (217 tokens). Poller peak **10,982 MiB** on cuda:0 (first run: 12,106 and out of memory) and 21,491 on cuda:1. `idle_cache_released` at 20:42:54.8Z, **5.8 s after the response**, freed 1,800 / 3,438 MiB. nvidia-smi went back to 9,182 / 18,053 MiB, 78 MiB above the loaded state on each card. |
| 7 | Context cap → 400 | **PASS** | OLMo 4,272 + 64 > 4,096 at 4:43 PM: **400 `invalid_request_error` `context_length_exceeded`**, both non-streaming and streaming. The stream was refused before a 200 was sent. Qwen2.5-7B (`all`) 32,699 + 512 > 32,768 at 4:48 PM: **400 `context_length_exceeded`**, both non-streaming and streaming. A short Qwen request right after returned 200. |
| 10 | bitsandbytes split | **PASS** | A local Q8 row over OLMo's FP16 files (id 5), `gpu: all`, at 4:45 PM: **19 / 21 layers**, 2 passes, `bitsandbytes: true`, no staging refusal. The fit counted 805 / 890 MiB of staging, and cuda:0 need was 11,133 of 11,825. Map 6,729 / 7,334 MiB; **landed 7,132 / 7,940**, which is 403 / 606 above the map and inside the counted staging (first run: 653 above the map with none counted). **The 3,879 + 16-token request returned 200** in 3.0 s. Poller peak **9,792 MiB** on cuda:0 (first run: 12,136 and out of memory). The row was deleted afterwards; the directory is intact (26,172 MB, 6 shards). |
| 11 | Unload and switch | **PASS** | Unloading the OLMo FP16 split at 4:43:49 PM: requests at +1 s and +3 s both got **503 `server_error` `model_busy`** ("being unloaded; retry once the unload finishes") in 10–20 ms. No 500, no `unhandled_exception`. The unload returned 200 after 15.8 s. `gpu_memory_cleanup` freed 8,848 / 17,320 MiB, and the poller shows **334 / 733 MiB** at 20:44:04.9Z, which is context only (733 on cuda:1 includes miStudio's worker context). A request 3 s after the unload loaded the model again on demand (`load_started` 20:44:08Z, same 13 / 27 fit, complete at 20:44:16Z) and returned 200. |

**Also seen:**
- **A log field reported the wrong status.** The `api_error` line for each `model_busy` refusal said `status_code: 409`, while the client got 503. `millm_error_handler` logged the exception's management-API status before mapping it for `/v1`. That affects every mapped `/v1` error (MODEL_NOT_LOADED logged 400 for a 503, for example). The log now carries the status sent. Guard: `tests/unit/api/test_exception_handlers.py::TestTheLoggedStatusIsTheOneSent`. Control LOG-M1 (log `exc.status_code` again) failed both parameters; the file was restored and sha256-verified.
- **Qwen2.5-7B with `all`** placed **19 / 9 layers** at 11,817 / 23,633 MiB free, with 835 MiB spare on cuda:0 (1 pass). The plan above predicted 18 / 10 and 1,241 spare at 11,767 / 23,976. Item 3 was not re-run, so this layout was not compared against a 3090-only load.
- The unload of the FP16 split took 15.8 s. The switches in the first run took about 8 s.

**Final node state (4:50 PM ET):** LFM2.5-1.2B-Instruct loaded by UUID on the 3080 Ti (2,224 MiB, `requested_card`). nvidia-smi showed 2,568 / 741 MiB, both idle. Qwen2.5-7B (id 2) and OLMo-2-13B FP16 (id 3) were ready and not loaded.

## Failures

### Failure 1 — the per-card fit admits a context its card cannot run (items 4, 10)

**Fixed in `6ce5c38` and `80a9aef`** (see Fixes).

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

**Fixed in `2c84fe8`.**

`InferenceService._check_context_length` (`millm/services/inference_service.py:2491`) raises a bare `ValueError`. The `ContextLengthExceededError` (400, `millm/core/errors.py:59`) exists and is used for llama.cpp, but not here. The function even imports `context_length_exceeded_error` and never uses it.

- Non-streaming (`create_chat_completion`, line 3076) reaches the client as **500**.
- Streaming (line 3789) raises inside the generator after the 200 headers, so the client gets **a truncated stream with no error event and no `[DONE]`**.

Log: `unhandled_exception ValueError "Context length exceeded: 32699 prompt + 512 max_tokens = 33211 > 32768"` at 17:07:14Z and 17:08:43Z, and `"4272 prompt + 64 max_tokens = 4336 > 4096"` at 17:14:07Z. The raise dates from `de474c63` (2026-02-07); round 5 (`7571cab`) changed only how the maximum is read. So this predates Phase 2, but the checklist item fails.

### Failure 3 — a `/v1` request during an unload runs on a half-moved model (item 11)

**Fixed in `ee20b11`.**

`ModelService.unload_model` keeps the row LOADED, and the loader reporting the model, until `_unload_worker` returns. Meanwhile `LoadedModelState.clear()` (`millm/ml/model_loader.py:303`, via `ModelLoader.unload` at 3659) moves the weights `.to("cpu")` first. accelerate logs "You shouldn't move a model that is dispatched using accelerate hooks." A chat arriving in that window passes the route's `model_info.name == request.model` check. It is dispatched into `generate()` and fails in `embed_tokens` with the device-mismatch `RuntimeError` above, reaching the client as 500.

The "still being unloaded; retry" path added in round 4 (`load_model_and_wait` → `ModelBusyError`) only covers the moment after the loader is cleared and before the row is written. It never covers a request for the model being unloaded.

Log: 17:24:14.146Z and 17:24:16.134Z, request lines at 17:24:14.094Z, `unload_started` at 17:24:13.082Z. Seen on a split model only; a single-card model was not tried.

## CUDA context and `TRANSFORMERS_CUDA_CONTEXT_MB`

| | Idle after load | After a generation | Left after unload |
|---|---|---|---|
| cuda:0 (3080 Ti) | 250 MiB | 330 MiB | 328 MiB, growing to 394 MiB by the end of the session |
| cuda:1 (3090) | 256 MiB | 334 MiB | 334 MiB, growing to 400 MiB |

The CUDA context alone is ~330–400 MiB per card. The allowance also has to cover prefill activations and fragmentation, measured above at ~525 MiB (Qwen2.5-7B, ~4k) up to an extrapolated ~1,300 MiB (OLMo-2-13B, 4,096).

**Recommendation (superseded by the fix for Failure 1: keep 500, see Fixes): `TRANSFORMERS_CUDA_CONTEXT_MB=1500` on this node** while `TRANSFORMERS_MIN_CONTEXT` is 4096. At today's free memory:
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
