---
sidebar_position: 1
title: "Steering Gemma End-to-End"
---

# Tutorial: Steering Gemma End-to-End

A complete, reproducible steering experiment: load Gemma 2 2B, attach a GemmaScope SAE, discover a feature on Neuronpedia, calibrate its strength, and verify the causal effect. Time: ~30 minutes (mostly downloads).

## What you'll build

By the end you will have demonstrated that a single SAE feature causally shifts Gemma's output — the core experiment of activation steering — and saved it as a reusable profile.

**Prerequisites:** a running miLLM instance ([Quickstart](/getting-started/quickstart)), a HuggingFace token with the `google/gemma-2-2b` license accepted, and ~10 GB free VRAM.

## Step 1 — Load the base model

Steering experiments use the **base** model, because GemmaScope SAEs were trained on it (`-pt-` = pretrained):

```bash
# Download (one-time, ~5 GB)
curl -X POST http://localhost:8000/api/models \
  -H "Content-Type: application/json" \
  -d '{"source": "huggingface", "repo_id": "google/gemma-2-2b",
       "quantization": "FP16", "hf_token": "hf_..."}'

# Watch for status "ready" (or watch the UI), then load:
curl http://localhost:8000/api/models          # find the id
curl -X POST http://localhost:8000/api/models/1/load
```

The first request after load pays a one-time `torch.compile` warmup; everything after is fast.

## Step 2 — Attach a GemmaScope SAE at layer 12

```bash
# Preview the repo to see what exists (optional but instructive)
curl -X POST http://localhost:8000/api/saes/preview \
  -H "Content-Type: application/json" \
  -d '{"repository_id": "google/gemma-scope-2b-pt-res", "hf_token": "hf_..."}'

# Download the layer-12, 16k-width SAE
curl -X POST http://localhost:8000/api/saes/download \
  -H "Content-Type: application/json" \
  -d '{"repository_id": "google/gemma-scope-2b-pt-res",
       "file_path": "layer_12/width_16k/average_l0_82/params.npz",
       "hf_token": "hf_..."}'

# When cached, attach at layer 12 (its trained layer)
curl -X POST http://localhost:8000/api/saes/{sae_id}/attach \
  -H "Content-Type: application/json" -d '{"layer": 12}'
```

Check the response: `layer_module_path` should read `model.layers.12`, and `warnings` should be empty. If you see a *trained-on* warning, you loaded `-it` instead of the base model.

## Step 3 — Pick a feature on Neuronpedia

Open the [GemmaScope 2B layer-12 16k dashboard](https://neuronpedia.org/gemma-2-2b/12-gemmascope-res-16k) on Neuronpedia. Search for a concept with an unmistakable signature in text — good first choices: *dogs*, *ocean/sea*, *cooking*, *legal language*.

For each candidate feature, Neuronpedia shows the tokens it fires on. Pick one whose top activations are clearly and narrowly about your concept, and note its **index** (say, feature `12082`).

:::tip Why obvious concepts first
Your first experiment should have an effect you can see without statistics. Subtle features (honesty, sentiment) need careful prompts and A/B discipline — do those second.
:::

## Step 4 — Baseline, then steer

Run an unrelated prompt at temperature 0 (deterministic, so differences are attributable):

```bash
BODY='{"model":"gemma-2-2b","messages":[{"role":"user","content":"Describe your ideal weekend."}],"temperature":0,"max_tokens":120}'

# Baseline
curl -s http://localhost:8000/v1/chat/completions -H "Content-Type: application/json" -d "$BODY" | jq -r '.choices[0].message.content'
```

Now steer the feature at moderate strength and rerun the **same** request:

```bash
curl -X POST http://localhost:8000/api/saes/steering \
  -H "Content-Type: application/json" \
  -d '{"feature_idx": 12082, "value": 40}'

curl -s http://localhost:8000/v1/chat/completions -H "Content-Type: application/json" -d "$BODY" | jq -r '.choices[0].message.content'
```

At strength 40 the concept should visibly intrude on the weekend description. If not, verify before cranking the strength:

```bash
curl -s http://localhost:8000/api/saes/attachment | jq '.data.steering_apply_count'
```

The counter must have advanced during your steered request. If it did, the intervention is live and the feature just needs more strength (or is not what Neuronpedia suggested).

## Step 5 — Calibrate

Sweep the strength and watch the behavior change character:

| Strength | Expect |
|----------|--------|
| 10 | Barely detectable; maybe one word choice shifts |
| 40 | Concept clearly present |
| 80 | Output dominated by the concept |
| 150+ | Coherence degrades — repetition, fixation |
| −60 | With a prompt *about* the concept: the model avoids or talks around it |

```bash
for s in 10 40 80 150; do
  curl -sX POST http://localhost:8000/api/saes/steering \
    -H "Content-Type: application/json" \
    -d "{\"feature_idx\": 12082, \"value\": $s}" > /dev/null
  echo "=== strength $s ==="
  curl -s http://localhost:8000/v1/chat/completions -H "Content-Type: application/json" -d "$BODY" \
    | jq -r '.choices[0].message.content' | head -4
done
```

Find the strength that maximizes concept expression while staying coherent — that's the value worth saving.

## Step 6 — Close the loop with monitoring

Enable monitoring and watch the steered feature (plus everything else) fire:

```bash
curl -X POST http://localhost:8000/api/monitoring/configure \
  -H "Content-Type: application/json" \
  -d '{"enabled": true, "top_k": 10}'
```

Open the **Probe** page, send a few requests, and confirm your feature appears in the top activations — and notice which *other* features co-fire with it. (Records reflect the final token of each generation; see [monitoring semantics](/concepts/monitoring).)

## Step 7 — Save it

```bash
curl -X POST http://localhost:8000/api/profiles/save-current \
  -H "Content-Type: application/json" \
  -d '{"name": "dogs-40", "description": "GemmaScope L12/16k feature 12082 @ 40"}'
```

The profile can now be re-activated in one click, exported as JSON, or applied **per-request** with `"profile": "dogs-40"` in any chat completion — see the [Python scripting tutorial](/tutorials/python-scripting) for building an A/B harness on top of it.

## Cleanup

```bash
curl -X DELETE http://localhost:8000/api/saes/steering    # clear + disable steering
```

Steering values are also cleared automatically if you detach the SAE.
