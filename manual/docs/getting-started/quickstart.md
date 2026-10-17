---
sidebar_position: 2
title: Quickstart
---

# Quickstart

Go from zero to steered inference in about 10 minutes. This path uses Docker Compose, `google/gemma-2-2b`, and a GemmaScope SAE.

## Prerequisites

- Docker with the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
- An NVIDIA GPU with **8 GB+ VRAM** (see [Hardware Requirements](/getting-started/hardware))
- A [HuggingFace token](https://huggingface.co/settings/tokens) with access to `google/gemma-2-2b` (it's a gated model — accept the license on its model page first)

## 1. Start the stack

```bash
git clone https://github.com/hitsainet/miLLM.git
cd miLLM
docker compose up -d
```

This starts PostgreSQL, Redis, the miLLM API on port `8000`, and the Admin UI on port `3000` behind nginx on port `80`.

Verify the API is up:

```bash
curl http://localhost:8000/api/health
# {"status":"healthy","version":"0.5.0",...}
```

Open the Admin UI at [http://localhost](http://localhost) (or `http://localhost:3000` for the direct Vite dev server).

## 2. Download and load a model

In the Admin UI: **Models → Download Model**, enter `google/gemma-2-2b`, paste your HF token, and pick a quantization (`FP16` needs ~5.5 GB VRAM for this model; `Q8` about half).

Or via the API:

```bash
curl -X POST http://localhost:8000/api/models \
  -H "Content-Type: application/json" \
  -d '{
    "source": "huggingface",
    "repo_id": "google/gemma-2-2b",
    "quantization": "FP16",
    "hf_token": "hf_..."
  }'
```

Download progress streams to the UI. When the model shows **Ready**, click **Load** (or `POST /api/models/{id}/load`). Loading takes ~30 s, plus a one-time `torch.compile` warmup (~20 s) on CUDA.

:::tip Base vs instruction-tuned
GemmaScope SAEs were trained on the **base** model (`gemma-2-2b`), not the instruction-tuned `gemma-2-2b-it`. SAEs only produce meaningful features on the model family they were trained for — start with the base model for steering experiments.
:::

## 3. Attach an SAE

In the UI: **SAEs → Download SAE**, enter `google/gemma-scope-2b-pt-res`, click **Preview** to browse available layers, and pick `layer_12/width_16k/average_l0_82/params.npz` (a good default).

Or via the API:

```bash
curl -X POST http://localhost:8000/api/saes/download \
  -H "Content-Type: application/json" \
  -d '{
    "repository_id": "google/gemma-scope-2b-pt-res",
    "file_path": "layer_12/width_16k/average_l0_82/params.npz",
    "hf_token": "hf_..."
  }'
```

When cached, attach it to **layer 12**:

```bash
curl -X POST http://localhost:8000/api/saes/{sae_id}/attach \
  -H "Content-Type: application/json" \
  -d '{"layer": 12}'
```

The response includes `layer_module_path` (e.g. `model.layers.12`) confirming exactly which module the steering hook landed on.

## 4. Steer a feature

Find an interesting feature on [Neuronpedia](https://neuronpedia.org/gemma-2-2b/12-gemmascope-res-16k) — for example, a feature that fires on dogs, the Golden Gate Bridge, or formal language. Note its index, then:

```bash
curl -X POST http://localhost:8000/api/saes/steering \
  -H "Content-Type: application/json" \
  -d '{"feature_idx": 12082, "value": 60}'
```

Steering is now active for **all** inference requests.

## 5. Generate and compare

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gemma-2-2b",
    "messages": [{"role": "user", "content": "Tell me about your day."}],
    "max_tokens": 100
  }'
```

Then clear steering and run the same prompt again:

```bash
curl -X DELETE http://localhost:8000/api/saes/steering
```

The difference between the two outputs is the **causal effect** of that one feature. That's the whole point of miLLM.

## Where to next

- [Tutorial: Steering Gemma](/tutorials/steering-gemma) — the full workflow with feature discovery, calibration, and verification
- [Concepts: How Steering Works](/concepts/steering) — the math and the knobs
- [Probe Monitoring](/features/probe-monitoring) — watch features fire in real time
- [Profiles](/features/profiles) — save the configuration you just built
