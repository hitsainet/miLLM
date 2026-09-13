# Multi-GPU Phase 1 — hardware acceptance (miLLM)

**Date:** 2026-09-13, 20:56–21:08 UTC (4:56–5:08 PM ET)
**Node:** mcs-lnxhost02 — RTX 3080 Ti 12 GB at CUDA/NVML index 0 (`GPU-f47ba814-49a2-603f-3595-275284140251`), RTX 3090 24 GB at index 1 (`GPU-247aa582-0d1b-e161-8156-983ed1fefc57`)
**Image:** `hitsai/millm-backend@sha256:9a54c39046ffe48b588ae35786c7bfa0d0c7cdfc39927c502320f4c9426abb1d` (commits `f4b497d`..`2f1a351`)
**Model:** id 1, LFM2.5-1.2B-Instruct, FP16 (estimate 2,746 MB)
**Script:** miStudio session scratchpad `main/millm_phase1_acceptance.py` — every check reads the live API and `nvidia-smi` on the host.

## Result: 6 of 6

| # | Check | Evidence |
|---|---|---|
| A1 | Auto puts the model on ONE card, the one with the most free memory | `placement = {mode: single, reason: most_free_card_fits, devices: [cuda:1], capacity_mb: 24112, memory_by_device_mb: {cuda:1: 2236}}` |
| A2 | No miLLM CUDA context on the card the model is not on | miLLM pid 999209 holds 2,492 MiB on the 3090 and nothing on the 3080 Ti (`nvidia-smi --query-compute-apps`) |
| A3 | A card named by UUID is honoured | `reason: requested_card, devices: [cuda:0], capacity_mb: 12156`; pid now 2,486 MiB on the 3080 Ti |
| A4 | An unknown card is refused synchronously and nothing starts loading | `404 GPU_NOT_FOUND "No visible GPU matches 'GPU-00000000-…'"`, details list both cards; model stayed unloaded |
| A5 | A card named by index is honoured (index 1 = the 3090 under `CUDA_DEVICE_ORDER=PCI_BUS_ID`) | `reason: requested_card, requested: 1, devices: [cuda:1]` |
| A6 | Placement is reported by the model list and `/api/health/detailed` | both carry the same placement block |

## Found during acceptance

1. **`GET /api/models/{id}` returned `placement: null`** for the loaded model while the list reported its card, and every listing logged `PydanticSerializationUnexpectedValue` (a raw dict assigned to the `ModelPlacement` field). Fixed in `685ef79`: both routes share `_with_runtime`, which validates the placement into its type. Two mutation controls (`test_model_load_gpu_request.py`). **Not deployed at the time of this record** — it ships with the next push.
2. **Moving a loaded model to another card is unload, then load.** `POST /load` on an already-loaded model returns `400 MODEL_ALREADY_LOADED` whatever `gpu` says. That is the API's contract, not a defect; the first run of the script assumed otherwise and was corrected. The admin UI's card selector applies on Load and Switch, which matches.
3. **A CUDA context outlives a move.** After moving the model from the 3090 to the 3080 Ti the process still held 256 MiB on the 3090 — the CUDA context from the earlier load, released only when the process exits. Expected; recorded because it is memory another job's Auto placement will not see as free.

## Not exercised on hardware

- **Refuse-before-unload** (`2f1a351`): a too-small named card or an unplaceable model is refused before the resident model is unloaded. It needs a second model that does not fit a card; only one model is on the node. Covered by `tests/unit/api/test_load_refusal_keeps_resident_model.py`.
- **GGUF placement** (`split_mode`/`main_gpu`) — no GGUF model on the node.
- **A model larger than one card** (the `all` spread) — Phase 2 territory; no such model on the node.
