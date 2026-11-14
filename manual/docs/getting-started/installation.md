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
git clone https://github.com/hitsainet/miServer.git
cd miServer

# Your own settings: choose a database password (there is no default)
cp .env.example .env
# edit .env and write a password after POSTGRES_PASSWORD=

# Start all services
docker compose up -d
```

Compose refuses to start while `POSTGRES_PASSWORD` is missing or empty, and names the value it needs.

This starts:
- **PostgreSQL** on port 5432
- **Redis** on port 6379
- **Backend** on port 8000
- **Admin UI** on port 3000
- **Nginx** on port 80

Access the admin UI at `http://localhost` or your configured domain.

## Kubernetes

Kubernetes is the recommended deployment method for lab and production environments. The Kustomize base at `k8s/base/` deploys the full miServer stack into a dedicated `miserver` namespace.

### Architecture

```
┌──────────────────────────────────────────────────┐
│  Namespace: miserver                                │
│                                                  │
│  ┌──────────┐  ┌──────────┐                     │
│  │ postgres │  │  redis   │  (persistent storage)│
│  └──────────┘  └──────────┘                     │
│                                                  │
│  ┌──────────────────────────────────────┐        │
│  │  miserver-backend Pod (GPU node)        │        │
│  │  └── backend  (FastAPI + ML :8000)   │        │
│  └──────────────────────────────────────┘        │
│                                                  │
│  ┌────────────────────┐                          │
│  │  miserver-frontend    │  (Admin UI/Nginx :80)    │
│  └────────────────────┘                          │
└──────────────────────────────────────────────────┘
         │
    NGINX Ingress
    ├── /api        → miserver-backend:8000
    ├── /v1         → miserver-backend:8000  (OpenAI API)
    ├── /docs       → miserver-backend:8000  (Swagger)
    ├── /socket.io  → miserver-backend:8000  (WebSocket)
    └── /           → miserver-frontend:80
```

Unlike miWorkbench, miServer runs the backend as a single container — inference, SAE management, and WebSocket streaming all live in one process. An init container runs on startup to create and permission the data directories.

### Prerequisites

**Cluster requirements:**
- Kubernetes 1.25+ (MicroK8s, k3s, or full K8s)
- NGINX Ingress Controller (`ingressClassName: public`)
- [NVIDIA Device Plugin](https://github.com/NVIDIA/k8s-device-plugin) for GPU scheduling
- At least one node with an NVIDIA GPU and the NVIDIA Container Toolkit installed

**Local tooling:**
```bash
# Verify kubectl is connected to your cluster
kubectl cluster-info

# Verify NVIDIA device plugin is running
kubectl get pods -n kube-system | grep nvidia

# Verify GPU is schedulable
kubectl describe node <gpu-node> | grep nvidia.com/gpu
```

### Step 1: Prepare Host Storage

miServer uses `hostPath` volumes for persistent data. Create the required directories on the GPU node before deploying:

```bash
# Run on the GPU node (or via ssh)
sudo mkdir -p /data/miserver/postgres
sudo mkdir -p /data/miserver/redis
sudo mkdir -p /data/miserver/data/model_cache
sudo mkdir -p /data/miserver/data/sae_cache
sudo mkdir -p /data/miserver/data/hf_cache
sudo chown -R 1000:1000 /data/miserver
```

The init container (`fix-permissions`) also creates and permissions the cache directories automatically on each pod start, so this step is mainly to ensure the parent paths exist.

:::tip Storage sizing
`model_cache` holds downloaded model weights — plan for 5–30 GB per model depending on quantization. `sae_cache` holds SAE weights — wide SAEs (131k features) can be 4–6 GB each. Provision 500 GB+ for active use.
:::

### Step 2: Configure the Manifests

The manifests live in `k8s/base/`, one file per component. Update these before applying:

**Node selector** (`k8s/base/backend.yaml`) — pin the GPU pod to your GPU node:
```yaml
nodeSelector:
  kubernetes.io/hostname: your-gpu-node-name   # replace mcs-lnxgpu01
```

**Domain names** — update the ingress hosts (`k8s/base/ingress.yaml`) and the backend's `hostAliases` (`k8s/base/backend.yaml`):
```yaml
# In hostAliases:
- ip: "192.168.x.x"           # Your GPU node IP
  hostnames:
    - "k8s-miserver.yourdomain.com"

# In both Ingress objects (miserver-ingress and miserver-websocket-ingress):
- host: k8s-miserver.yourdomain.com
```

**Secrets** — credentials are not stored in the manifests. Create the `miserver-secrets` Secret the deployments reference:
```bash
kubectl create namespace miserver --dry-run=client -o yaml | kubectl apply -f -
kubectl create secret generic miserver-secrets -n miserver \
  --from-literal=POSTGRES_PASSWORD="your-strong-password" \
  --from-literal=DATABASE_URL="postgresql+asyncpg://millm:your-strong-password@postgres:5432/millm"
```

**CORS** (`k8s/base/backend.yaml`) — add any additional origins that will access the API:
```yaml
- name: CORS_ORIGINS
  value: "http://k8s-miserver.yourdomain.com,http://your-open-webui.yourdomain.com"
```

### Step 3: Deploy

```bash
# Apply the Kustomize base
kubectl apply -k k8s/base

# Watch pods come up
kubectl get pods -n miserver -w
```

:::info GitOps alternative
The reference cluster is managed by ArgoCD instead of manual applies: `k8s/argocd/miserver-app.yaml` defines an Application that watches the `gitops/pilot` branch, and ArgoCD Image Updater rolls new image digests automatically as CI publishes them. See the comments in that file for the one-time setup (repo credential + secret).
:::

Expected output once healthy:
```
NAME                                READY   STATUS    RESTARTS   AGE
miserver-backend-xxxxxxxxx-xxxxx       1/1     Running   0          90s
miserver-frontend-xxxxxxxxx-xxxxx      1/1     Running   0          60s
postgres-xxxxxxxxx-xxxxx            1/1     Running   0          60s
redis-xxxxxxxxx-xxxxx               1/1     Running   0          60s
```

:::info First-start database migration
The backend runs `alembic upgrade head` automatically on startup via the entrypoint script. Check backend logs if the pod restarts — a failed migration is the most common first-boot issue.
:::

### Step 4: Configure DNS

```bash
# On each client machine
echo "192.168.x.x  k8s-miserver.yourdomain.com" | sudo tee -a /etc/hosts
```

Then access the Admin UI at `http://k8s-miserver.yourdomain.com` and the OpenAI-compatible API at `http://k8s-miserver.yourdomain.com/v1`.

### Verifying the Deployment

```bash
# Pod status
kubectl get pods -n miserver

# Backend logs
kubectl logs -n miserver deployment/miserver-backend --tail=50

# Verify GPU is allocated to the backend
kubectl exec -n miserver deployment/miserver-backend -- nvidia-smi

# Confirm API health
curl http://k8s-miserver.yourdomain.com/api/health

# Confirm OpenAI-compatible endpoint
curl http://k8s-miserver.yourdomain.com/v1/models
```

### Connecting Open WebUI

miServer exposes a fully OpenAI-compatible API at `/v1`. To connect Open WebUI or any other compatible client:

1. In Open WebUI → **Settings** → **Connections** → **OpenAI API**
2. Set the URL to `http://k8s-miserver.yourdomain.com/v1`
3. Leave the API key blank (or set any value — miServer does not enforce key auth by default)
4. Toggle the connection on and save

The model name shown in Open WebUI will match whichever model is currently loaded in miServer.

### Updating to New Images

miServer publishes new images to DockerHub on every push to `main`. To update a running cluster:

```bash
# Pull and restart
kubectl rollout restart deployment/miserver-backend -n miserver
kubectl rollout restart deployment/miserver-frontend -n miserver

# Wait for completion
kubectl rollout status deployment/miserver-backend -n miserver --timeout=180s
kubectl rollout status deployment/miserver-frontend -n miserver --timeout=180s
```

:::info Recreate strategy
The backend uses `strategy: Recreate` — the old pod terminates fully before the new one starts. This ensures the GPU and model weights are cleanly released before the new process takes over, preventing CUDA context conflicts.
:::

### WebSocket Configuration

miServer uses Socket.IO for real-time inference progress and steering updates. The manifest includes a dedicated WebSocket ingress (`miserver-websocket-ingress`) for the `/socket.io` path with extended timeouts:

| Annotation | Value | Purpose |
|-----------|-------|---------|
| `websocket-services` | `miserver-backend` | Enables WebSocket upgrade handling |
| `proxy-http-version` | `1.1` | Required for HTTP upgrade |
| `proxy-read-timeout` | `86400` | 24-hour timeout for long inference sessions |
| `proxy-send-timeout` | `86400` | 24-hour timeout for long inference sessions |

:::warning Do not merge the WebSocket ingress
The WebSocket and HTTP ingresses are intentionally separate. Combining them with `configuration-snippet` Upgrade headers causes header duplication and broken connections.
:::

### Environment Variable Reference

| Variable | Default | Description |
|----------|---------|-------------|
| `DATABASE_URL` | `postgresql+asyncpg://millm:millm@postgres:5432/millm` | Async PostgreSQL connection |
| `MODEL_CACHE_DIR` | `/data/model_cache` | Where downloaded model weights are stored |
| `SAE_CACHE_DIR` | `/data/sae_cache` | Where downloaded SAE weights are stored |
| `HF_HOME` | `/data/hf_cache` | HuggingFace cache directory |
| `CORS_ORIGINS` | *(comma-separated URLs)* | Allowed origins for browser API access |
| `LOG_LEVEL` | `INFO` | Logging verbosity: `DEBUG`, `INFO`, `WARNING` |
| `LOG_FORMAT` | `json` | Log format: `json` or `text` |
| `DEBUG` | `false` | Enable FastAPI debug mode |
