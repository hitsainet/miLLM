---
sidebar_position: 2
title: Installation
---

# Installation

## Hardware Requirements

| Tier | VRAM | Capability |
|------|------|-----------|
| **Minimum** | 8 GB | 1B–2B models at Q4 with small SAEs (16k width) |
| **Recommended** | 16–24 GB | 2B–9B models with wide SAEs (131k width) |
| **Multi-GPU** | 2×24 GB+ | Large models (27B+) with CPU offloading |

:::warning GPU Memory Budget
The model, SAE, and KV cache all share GPU memory. A 9B model at Q4 (~6GB) plus a 131k-width SAE (~5GB) plus KV cache (~2-4GB) can fill a 24GB GPU. Monitor VRAM usage in the Dashboard.
:::

## Docker Compose (Recommended)

```bash
# Clone the repository
git clone https://github.com/Onegaishimas/miLLM.git
cd miLLM

# Start all services
docker compose up -d
```

This starts:
- **PostgreSQL** on port 5432
- **Redis** on port 6379
- **Backend** on port 8000
- **Admin UI** on port 3000
- **Nginx** on port 80

Access the admin UI at `http://localhost` or your configured domain.

## Kubernetes

miLLM includes a K8s deployment manifest:

```bash
kubectl apply -f k8s/millm-deployment.yaml
```

:::info Environment Variables
Key configuration:
- `DATABASE_URL` — PostgreSQL connection string
- `MODEL_CACHE_DIR` — Where downloaded models are stored (mount persistent volume)
- `SAE_CACHE_DIR` — Where downloaded SAEs are stored
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` — Required for hybrid SSM models
:::
