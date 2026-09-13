# Multi-GPU Phase 0 — hardware acceptance (miLLM)

**Date:** 2026-09-13, 3:10 PM ET · **Node:** mcs-lnxhost02 · **Image:** `hitsai/millm-backend@sha256:ab5149dd…` (commit `924a6a5`)
**Plan:** miStudio `0xcc/plans/Multi-GPU-Plan.md`, Phase 0

## Hardware

| NVML index | Card | Total |
|---|---|---|
| 0 | RTX 3080 Ti | 12,288 MiB |
| 1 | RTX 3090 | 24,576 MiB |

Container: `NVIDIA_VISIBLE_DEVICES=all`, `CUDA_DEVICE_ORDER=PCI_BUS_ID`.

## What was run

`millm_phase0_acceptance.py` inside the backend pod, against the cached
`LiquidAI--LFM2.5-1.2B-Instruct--FP16`:

1. `get_gpu_metrics()` lists both cards.
2. Load at FP16 with `device_map="auto"` (miLLM's FP16 path). Confirm the model spans both cards and pick a layer on `cuda:1`.
3. Build a random-weight `LoadedSAE` (d_in 2048, d_sae 512, fp16) on `SAEHooker.layer_device(model, layer)`, enable monitoring, install the hook, `generate(max_new_tokens=8, do_sample=False)`.
4. Control: build the same SAE on `cuda:0` (the pre-fix placement), install at the same layer, generate.

## Result — 7/7

```
PASS  GPU metrics list both cards  — 0: NVIDIA GeForce RTX 3080 Ti 12288 MB, 1: NVIDIA GeForce RTX 3090 24576 MB
PASS  FP16 model is split across both cards  — devices ['0', '1']
PASS  a layer lives on cuda:1  — layer 11 of 16
PASS  SAE placed on its layer's card  — cuda:1
PASS  monitoring captured activations during generate  — shape (1, 512) on cuda:1
PASS  generation unchanged by monitoring  — ' Paris. It is the most populous'
PASS  control: SAE loaded on cuda:0 is moved to cuda:1 by the hook  — cuda:1
7/7 checks passed
```

GPU memory returned to 0 MiB on both cards when the process exited.

## Also verified

- Backend Tests (unit, schema guards on PostgreSQL 16, contract ↔ MCP registry) green in CI for `924a6a5`.
- Local unit suite with CI's exclusions: 1783 passed, 12 skipped.
- Mutation controls — each broke one load-bearing line, each turned its test red, each file restored byte-identical:

| # | Mutation | Result |
|---|---|---|
| M1 | parse only the first `nvidia-smi` line | killed (3 failed) |
| M2 | hook no longer moves a misplaced SAE | killed |
| M3 | `layer_device` reads the model, not the layer | killed |
| M4 | `attach_set` loads on bare `"cuda"` | killed (2 failed) |
| M5 | `attach_sae` loads on bare `"cuda"` | killed |
| M6 | memory gate sums every SAE into one bucket | killed (2 failed) |
| M7 | fit check reads GPU 0 for FP16 | killed (2 failed) |
| M8 | total free takes the max instead of the sum | killed |

## Not covered by this run

- Loading an FP16 model larger than 12 GB: no such model is cached on the node. The fit check is covered by `tests/unit/ml/test_model_load_fit_check.py`.
- The attach API end to end with a real SAE: none is downloaded on the fresh node. The service wiring is covered by `tests/unit/services/test_sae_attach_device.py`, including an AST check that both attach paths load on the resolved layer device.
- Small FP16 models are still split across both cards (Phase 1 places a model that fits on one card).
