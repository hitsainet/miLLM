---
sidebar_position: 100
title: Troubleshooting
---

# Troubleshooting

Organized by symptom. For error codes returned by the API, see the [Error Codes reference](/reference/error-codes).

## Steering has no effect

The most important failure mode in miLLM — an intervention that silently isn't happening invalidates your experiment. Check in this order:

1. **Is steering actually enabled with non-zero values?**
   ```bash
   curl -s http://localhost:8000/api/saes/steering | jq '.data'
   ```
2. **Is the hook firing?** Generate a completion, then:
   ```bash
   curl -s http://localhost:8000/api/saes/attachment | jq '.data.steering_apply_count'
   ```
   The counter must increase across a steered generation. If it doesn't move, the hook isn't executing — re-attach the SAE and check server logs. (miLLM guards the known `torch.compile`-swallows-hooks failure automatically, and attach/detach forces a recompile; a non-moving counter after that indicates a genuinely unsupported architecture.)
3. **Right module?** The attach response's `layer_module_path` should name a decoder layer (e.g. `model.layers.12`). On multimodal/exotic architectures the layer-resolution heuristics can land elsewhere — try attaching by a different layer index and compare.
4. **Strength too low or feature not what you think?** Sweep to ±100 on a feature with an obvious signature (see the [tutorial](/tutorials/steering-gemma)); check the feature's meaning on Neuronpedia.
5. **Wrong model variant?** SAEs trained on `gemma-2-2b` are much weaker on `gemma-2-2b-it`. The attach `warnings` array tells you about this mismatch.

## Out-of-memory (OOM)

| Symptom | Cause | Fix |
|---------|-------|-----|
| OOM during model load | Model too large for GPU | More aggressive quantization (Q8/Q4), or let `device_map=auto` offload to CPU |
| OOM during inference | model + KV cache + SAE exceeds VRAM | Reduce `max_tokens`, use a narrower SAE (16k vs 65k+), lower `MAX_CONCURRENT_REQUESTS` |
| OOM only with hybrid/Mamba models | `mamba-ssm` not installed; naive fallback allocates 20 GB+ intermediates | Install `mamba-ssm`, or set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` |
| VRAM used but no model loaded | Leaked memory after a crash | Restart the backend process/pod, then reload |

On OOM with an SAE attached, miLLM degrades gracefully: the SAE is disabled and the base model continues serving — check the attachment status if steering suddenly stops.

```bash
nvidia-smi --query-gpu=memory.used,memory.free,memory.total --format=csv,noheader
nvidia-smi --query-compute-apps=pid,name,used_memory --format=csv,noheader
```

Sizing guidance: [Hardware Requirements](/getting-started/hardware).

## Downloads & model loading

| Symptom | Cause | Fix |
|---------|-------|-----|
| `401 GATED_MODEL_NO_TOKEN` | Gemma/Llama are gated | Accept the license on the HF model page, supply `hf_token` |
| `404 REPO_NOT_FOUND` | Typo, or private repo without token | Check the repo ID; add a token |
| Load fails: "TokenizersBackend" | Model needs a custom tokenizer package | Retry with `trust_remote_code: true` |
| Load fails: BitNet/GPTQ conflicts | Pre-quantized weights + bitsandbytes don't mix | Download with `FP16` (miLLM auto-detects most cases and skips bitsandbytes) |
| First request after load is very slow (~20 s) | One-time `torch.compile` warmup | Expected; subsequent requests are fast. Also occurs once after SAE attach/detach. Disable with `TORCH_COMPILE=false` if warmup matters more than throughput |
| Download stuck | Network or HF rate limiting | Cancel (`POST /api/models/{id}/cancel`) and retry; check `GET /api/health/circuits` for a tripped HuggingFace breaker |

## SAE attachment

| Symptom | Cause | Fix |
|---------|-------|-----|
| `400 SAE_INCOMPATIBLE` (dimension mismatch) | SAE `d_in` ≠ model hidden size | Use the SAE built for this exact model size (2b vs 9b vs 27b) |
| Attach warning: trained on different layer/model | Deliberate mismatch or wrong file | Heed it — features are only meaningful at the trained layer on the trained model |
| `409 SAE_ALREADY_ATTACHED` | One SAE at a time | Detach first |
| Detach appears to hang | Waiting (up to 30 s) for in-flight requests to finish | Expected — the hook is never removed mid-generation |

## Inference & API

| Symptom | Cause | Fix |
|---------|-------|-----|
| `503 model_not_loaded` on `/v1/*` | No model loaded | Load one; consider `AUTO_LOAD_MODEL` for restarts |
| `429` errors | Request queue full (`MAX_PENDING_REQUESTS`) | Raise the limit, or slow the client (Open WebUI's parallel title-generation is a common culprit) |
| `400 context_length_exceeded` | prompt + `max_tokens` > model context | Shorten the prompt or reduce `max_tokens` |
| Streaming stops with an in-stream `error` event | Generation failed after headers were sent (HTTP is already 200) | Check server logs; SSE consumers should watch for `error` objects before `[DONE]` |
| Responses look like completions, not chat | Base (non-`-it`) model | Expected — base models aren't instruction-tuned ([why you might want that](/tutorials/open-webui)) |
| CORS errors in browser | Origin not allowed | Set `CORS_ORIGINS` ([Configuration](/reference/configuration)) |
| 500s on `/v1/chat/completions` after a crash | Leaked GPU state | Restart the backend, reload the model |

## Monitoring

| Symptom | Cause | Fix |
|---------|-------|-----|
| No activation records | Monitoring disabled, or no completions since enabling | `GET /api/monitoring` to check state; records are written once per completed generation |
| Wrong/garbage feature indices in history | (Fixed in current versions) watched-feature list desync | Upgrade; configure via `/api/monitoring/configure` |
| History entry per request, not per token | By design — records capture the final forward pass | See [monitoring semantics](/concepts/monitoring) |
| WebSocket disconnects during long generations | Event loop busy | Normal; the client auto-reconnects — resync state via REST after reconnect |

## Kubernetes-specific

```bash
kubectl get pods -n millm                      # pod states
kubectl logs -n millm deploy/millm-backend     # backend logs
kubectl delete pod -n millm <pod-name>         # restart to clear leaked VRAM
```

GPU scheduling requires the NVIDIA device plugin and a node with the GPU visible; see the [Kubernetes install guide](/getting-started/install-guide-k8s).

## Still stuck?

- Set `LOG_LEVEL=DEBUG` and reproduce — steering/hook activity is logged
- `GET /api/health/detailed` gives a per-component health breakdown
- Open an issue: [github.com/hitsainet/miLLM](https://github.com/hitsainet/miLLM/issues) with logs and your model/SAE combination
