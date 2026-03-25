---
sidebar_position: 100
title: Troubleshooting
---

# Troubleshooting

## Common Issues

| Symptom | Cause | Fix |
|---------|-------|-----|
| OOM during model load | Model too large for GPU | Use more aggressive quantization (Q4 or Q2), or use `device_map=auto` for CPU offload |
| OOM during inference | KV cache + model + SAE exceeds VRAM | Reduce `max_tokens`, use smaller SAE width, or unload unused SAEs |
| OOM with hybrid/Mamba model | `mamba-ssm` not installed, naive fallback creates 22GB+ tensor | Install `mamba-ssm` package, or set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` |
| Steering has no effect | Hook not firing on model architecture | Check logs for `[Steering Hook] FIRED` messages. Some architectures return single tensors instead of tuples. |
| Steering outputs identical at all strengths | Model state not reset between generations | Ensure `disable_cache=True` during steered generation |
| Model load fails with "TokenizersBackend" | Custom tokenizer class not available | Model needs specific tokenizer package. Try with `trust_remote_code=True` |
| Model load fails with "BitNet" | Pre-quantized model conflicts with bitsandbytes | Pre-quantized models (BitNet, GPTQ) should not use bitsandbytes. miLLM auto-detects and skips. |
| 500 errors on `/v1/chat/completions` | Model crashed or leaked GPU memory | Restart the backend pod to free leaked VRAM, then reload the model |
| SAE attach fails "dimension mismatch" | SAE d_in doesn't match model hidden dim at target layer | Ensure the SAE was trained for this model and layer |
| WebSocket disconnects | Long-running inference blocks the event loop | Normal behavior — WebSocket reconnects automatically |
| Labeling job gets 500s from miLLM | Model OOMing during inference | Switch to a smaller model or increase quantization |

## GPU Memory Troubleshooting

Check GPU memory usage:
```bash
nvidia-smi --query-gpu=memory.used,memory.free,memory.total --format=csv,noheader
```

Check what processes hold GPU memory:
```bash
nvidia-smi --query-compute-apps=pid,name,used_memory --format=csv,noheader
```

If memory is leaked (no model shows as loaded but VRAM is used), restart the backend pod:
```bash
kubectl delete pod -n millm <pod-name>
```

## Useful Environment Variables

| Variable | Purpose | Default |
|----------|---------|---------|
| `PYTORCH_CUDA_ALLOC_CONF` | Set to `expandable_segments:True` for hybrid models | Not set |
| `MODEL_CACHE_DIR` | Where models are stored | `/data/model_cache` |
| `SAE_CACHE_DIR` | Where SAEs are stored | `/data/sae_cache` |
| `LOG_LEVEL` | Backend log verbosity | `INFO` |
| `CORS_ORIGINS` | Allowed CORS origins | `http://localhost` |
