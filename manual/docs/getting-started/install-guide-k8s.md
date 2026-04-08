---
sidebar_position: 5
title: "Install Guide: Kubernetes"
description: "Executable Claude Code installation guide for Kubernetes deployment"
---

# Install Guide: Kubernetes

:::info This document is written as executable instructions for Claude Code
Read the entire **Before We Begin** section first and collect the user's answers. Then follow each section in order. Every check must pass (or be acknowledged) before moving to the next section.
:::

## Before We Begin

Present each question to the user and record their answers. Use the recorded answers throughout all subsequent steps.

---

**Q1 — Run Mode**

> "Are you running me directly on the machine that has kubectl access to the cluster, or from a workstation that will SSH into a jump host?"

- **`local`** — `kubectl` is available directly. Use kubectl commands as-is.
- **`remote`** — Ask: *"What is the SSH user and hostname of the machine with kubectl access? (e.g. `sean@192.168.1.100`)"* Record as `SSH_TARGET`. Prefix all kubectl commands accordingly.

---

**Q2 — Missing Prerequisites**

> "If I find a required tool is missing, should I attempt to install it automatically (requires sudo), or report what's missing and stop so you can handle it?"

- **`auto`** — Attempt installation automatically.
- **`diagnose`** — Report with fix instructions and stop.

---

**Q3 — Secrets**

> "Should I generate a secure random database password, or will you provide one?"

- **`generate`** — Claude Code generates a value using `openssl rand`.
- **`provide`** — Ask the user for the value before proceeding.

---

**Q4 — GPU Node**

> "Should I discover the available nodes in your cluster and let you choose the GPU node, or will you provide the node name directly?"

- **`discover`** — Run `kubectl get nodes` and present the list.
- **`provide`** — Ask: *"What is the exact Kubernetes node name of the GPU host?"* Record as `GPU_NODE`.

---

Record answers as `RUN_MODE`, `PREREQ_MODE`, `SECRETS_MODE`, `NODE_MODE`. Confirm with the user before proceeding.

---

## Pre-Flight Checks

Run all checks before any installation steps. For each result:
- **PASS** — continue silently
- **WARN** — print the warning and ask whether to continue
- **FAIL (auto)** — attempt the documented fix, re-check; if still failing, stop
- **FAIL (diagnose)** — print the issue and fix instructions, then stop

### Tooling

**kubectl**
```bash
kubectl version --client --short 2>/dev/null || echo "NOT_FOUND"
```
- PASS: any version
- FAIL auto: `sudo snap install kubectl --classic`
- FAIL diagnose: "Install kubectl: https://kubernetes.io/docs/tasks/tools/"

**Cluster reachable**
```bash
kubectl cluster-info 2>/dev/null | head -2
```
- PASS: returns control plane URL
- FAIL: "Cannot reach Kubernetes cluster. Check `kubectl config current-context` and network/VPN access."

**Ingress controller**
```bash
kubectl get ingressclass 2>/dev/null | grep -c "public" || echo "0"
```
- PASS: returns `1` or higher
- WARN returns `0`: "No ingress class named 'public' found. The manifest uses `ingressClassName: public`. Show available classes with `kubectl get ingressclass` and update the manifest to match, or install an NGINX ingress controller."

**NGINX Ingress pods running**
```bash
kubectl get pods -A | grep ingress | grep -v Terminating
```
- PASS: at least one Running pod
- WARN none found: "Install NGINX Ingress Controller: `kubectl apply -f https://raw.githubusercontent.com/kubernetes/ingress-nginx/main/deploy/static/provider/cloud/deploy.yaml`"

### GPU Node

**If NODE_MODE=discover:**
```bash
kubectl get nodes -o wide
```
Present the full list to the user and ask them to identify the GPU node. Record as `GPU_NODE`.

**If NODE_MODE=provide:** Use the already-recorded `GPU_NODE`.

**Node exists and is Ready**
```bash
kubectl get node $GPU_NODE --no-headers | awk '{print $2}'
```
- PASS: `Ready`
- FAIL: "Node '$GPU_NODE' not found or not Ready. Verify with `kubectl get nodes`."

**NVIDIA Device Plugin**
```bash
kubectl get pods -n kube-system | grep nvidia-device-plugin | grep -c Running || echo "0"
```
- PASS: returns `1` or higher
- WARN returns `0`:
  ```
  "NVIDIA Device Plugin not found. Install with:
  kubectl apply -f https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/v0.14.0/nvidia-device-plugin.yml"
  ```

**GPU schedulable on node**
```bash
kubectl describe node $GPU_NODE | grep -A5 "Capacity:" | grep "nvidia.com/gpu"
```
- PASS: shows `nvidia.com/gpu: 1` or more
- FAIL: "GPU not schedulable on $GPU_NODE. Verify NVIDIA driver and device plugin are installed on that node."

### Storage

**Host storage paths — check or create on GPU node**

If the GPU node is directly accessible (local or via SSH):
```bash
sudo mkdir -p /data/millm/postgres /data/millm/redis /data/millm/data/model_cache /data/millm/data/sae_cache /data/millm/data/hf_cache
sudo chown -R 1000:1000 /data/millm
ls -la /data/millm/
```

If not directly accessible, instruct the user:
> "Please ensure these directories exist on node $GPU_NODE before continuing:
> - `/data/millm/postgres`
> - `/data/millm/redis`
> - `/data/millm/data/model_cache`
> - `/data/millm/data/sae_cache`
> - `/data/millm/data/hf_cache`
>
> With ownership `1000:1000`. The init container will also create subdirectories on first run."

:::info Init container
The backend pod includes an init container (`fix-permissions`) that runs `mkdir -p` and `chown` on `/data/model_cache`, `/data/sae_cache`, and `/data/hf_cache` every time the pod starts. The parent `/data/millm/data` directory must exist beforehand.
:::

**Disk space on GPU node**
If accessible:
```bash
df -BG /data | tail -1 | awk '{print $4}' | tr -d 'G'
```
- PASS ≥ 50 GB free
- WARN 20–49 GB: "Limited disk space. Each model is 1–20 GB. Consider adding storage."
- FAIL < 20 GB: "Less than 20GB free. Provision more storage."

### Images

**Backend image pullable**
```bash
docker pull hitsai/millm-backend:latest 2>&1 | tail -1
```
- PASS: `Status: Downloaded newer image` or `Status: Image is up to date`
- FAIL: "Cannot pull `hitsai/millm-backend:latest`. Check internet access from the cluster node."

---

## Configuration

### GPU Node Name

Confirm `GPU_NODE` is recorded from the Pre-Flight section.

### Domain Name

Ask the user:
> "What hostname should miLLM be accessible at? Press Enter to use the default: `k8s-millm.hitsai.local`"

Record as `DOMAIN`. Default: `k8s-millm.hitsai.local`.

Ask the user:
> "What is the IP address of the GPU node ($GPU_NODE)?"

Record as `GPU_NODE_IP`.

### Secrets

**If SECRETS_MODE=generate:**
```bash
POSTGRES_PASSWORD=$(openssl rand -hex 16)
```
Print: `"Generated POSTGRES_PASSWORD: $POSTGRES_PASSWORD — save this now."`

**If SECRETS_MODE=provide:**
Ask: "What should the PostgreSQL password be?" → `POSTGRES_PASSWORD`

### CORS Origins

Ask the user:
> "Will any external services connect to miLLM's OpenAI-compatible API? (e.g. Open WebUI at a different domain). Provide their base URLs comma-separated, or press Enter for the default."

Record as `CORS_ORIGINS`. Default: `http://$DOMAIN`.

### Optional HuggingFace Token

Ask the user:
> "Do you have a HuggingFace token for downloading gated models? Press Enter to skip."

Record as `HF_TOKEN`. Default: empty.

---

## Prepare the Manifest

Clone the repository from the machine with kubectl access:
```bash
git clone https://github.com/Onegaishimas/miLLM.git
cd miLLM
cp k8s/millm-deployment.yaml k8s/millm-deployment.local.yaml
```

Apply all substitutions to `millm-deployment.local.yaml`:

**Node selector:**
```bash
sed -i "s/mcs-lnxgpu01/$GPU_NODE/g" k8s/millm-deployment.local.yaml
```

**Host IP and domain:**
```bash
sed -i "s/192\.168\.244\.61/$GPU_NODE_IP/g" k8s/millm-deployment.local.yaml
sed -i "s/k8s-millm\.mcslab\.io/$DOMAIN/g" k8s/millm-deployment.local.yaml
sed -i "s/k8s-millm\.hitsai\.net/$DOMAIN/g" k8s/millm-deployment.local.yaml
```

**PostgreSQL password** (update POSTGRES_PASSWORD value and DATABASE_URL):
```bash
sed -i "s/value: millm$/value: $POSTGRES_PASSWORD/" k8s/millm-deployment.local.yaml
sed -i "s|millm:millm@postgres|millm:$POSTGRES_PASSWORD@postgres|g" k8s/millm-deployment.local.yaml
```

**CORS origins:**
```bash
sed -i "s|http://k8s-millm.hitsai.local,http://k8s-millm.hitsai.net,http://localhost:3000|$CORS_ORIGINS|g" k8s/millm-deployment.local.yaml
```

**HuggingFace token** (if provided):
```bash
# Add HF_TOKEN env var to the backend container spec if provided
# Only do this if HF_TOKEN is non-empty
if [ -n "$HF_TOKEN" ]; then
  sed -i "/name: LOG_FORMAT/a\\            - name: HF_TOKEN\\n              value: \"$HF_TOKEN\"" k8s/millm-deployment.local.yaml
fi
```

Verify substitutions before applying:
```bash
grep -E "hostname|POSTGRES_PASSWORD|DATABASE_URL|CORS_ORIGINS|host:" k8s/millm-deployment.local.yaml
```

---

## Deployment

### Step 1 — Apply the manifest

```bash
kubectl apply -f k8s/millm-deployment.local.yaml
```

Expected output:
```
namespace/millm created (or unchanged)
deployment.apps/postgres created
service/postgres created
deployment.apps/redis created
service/redis created
deployment.apps/millm-backend created
service/millm-backend created
deployment.apps/millm-frontend created
service/millm-frontend created
ingress.networking.k8s.io/millm-ingress created
ingress.networking.k8s.io/millm-websocket-ingress created
```

### Step 2 — Wait for pods

```bash
kubectl rollout status deployment/postgres -n millm --timeout=120s
kubectl rollout status deployment/redis -n millm --timeout=120s
kubectl rollout status deployment/millm-backend -n millm --timeout=300s
kubectl rollout status deployment/millm-frontend -n millm --timeout=120s
```

The backend shows `1/1` when ready. On first boot it runs `alembic upgrade head` — allow up to 2 minutes.

### Step 3 — Configure DNS

```bash
grep -q "$DOMAIN" /etc/hosts || echo "$GPU_NODE_IP  $DOMAIN" | sudo tee -a /etc/hosts
```

If `RUN_MODE=remote`, instruct the user to also add this entry on any machine that will access miLLM.

---

## Post-Install Verification

```bash
# Pod status
kubectl get pods -n millm

# Backend logs (check for migration success)
kubectl logs -n millm deployment/millm-backend --tail=40

# GPU allocated to backend
kubectl exec -n millm deployment/millm-backend -- nvidia-smi

# API health
curl -s http://$DOMAIN/api/health

# OpenAI models endpoint (empty list is OK — no model loaded yet)
curl -s http://$DOMAIN/v1/models | python3 -m json.tool

# Frontend reachable
curl -sf http://$DOMAIN > /dev/null && echo "Frontend: OK" || echo "Frontend: FAIL"

# Verify both ingresses are present (important for WebSocket)
kubectl get ingress -n millm
```

Print access summary:
```
✓ miLLM Admin UI:     http://$DOMAIN
✓ OpenAI API:         http://$DOMAIN/v1
✓ API docs (Swagger): http://$DOMAIN/docs
✓ API health:         http://$DOMAIN/api/health
✓ Namespace:          millm
✓ GPU node:           $GPU_NODE
```

### Connecting Open WebUI

Instruct the user:
1. Open WebUI → **Settings** → **Connections** → **OpenAI API**
2. URL: `http://$DOMAIN/v1`
3. API key: leave blank or enter any value
4. Toggle on, save

:::warning WebSocket ingress is separate
The `millm-websocket-ingress` routes `/socket.io` with extended 24-hour timeouts for long inference sessions. Both ingresses must be present — verify with `kubectl get ingress -n millm`.
:::

---

## Updating to New Images

```bash
kubectl rollout restart deployment/millm-backend -n millm
kubectl rollout restart deployment/millm-frontend -n millm
kubectl rollout status deployment/millm-backend -n millm --timeout=180s
kubectl rollout status deployment/millm-frontend -n millm --timeout=180s
```

:::info Recreate strategy
The backend uses `strategy: Recreate`. The old pod terminates fully before the new one starts, ensuring the GPU and model weights are cleanly released before the new process loads.
:::

---

## Troubleshooting Quick Reference

| Symptom | Check | Fix |
|---------|-------|-----|
| Backend pod `0/1` | `kubectl describe pod -n millm -l app=millm-backend` | Check Events — usually GPU not schedulable or image pull failure |
| `ImagePullBackOff` | `kubectl get events -n millm` | Node cannot reach Docker Hub — check internet on GPU node |
| Pod starts but API 503 | `kubectl logs -n millm deployment/millm-backend` | Migration failure or DB not ready — check postgres pod |
| WebSocket disconnects | `kubectl get ingress -n millm` | Both ingresses must be present; check `millm-websocket-ingress` exists |
| Model download hangs | Logs in Admin UI or `kubectl logs` | Network issue on GPU node; check HF_TOKEN for gated models |
| Data lost after pod restart | `ls /data/millm/data` on GPU node | hostPath volume must exist with correct ownership before pod starts |
| Inference returns 503 | Admin UI → Models | No model is currently loaded — load one first |
