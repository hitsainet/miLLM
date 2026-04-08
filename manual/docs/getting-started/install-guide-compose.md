---
sidebar_position: 4
title: "Install Guide: Docker Compose"
description: "Executable Claude Code installation guide for Docker Compose deployment"
---

# Install Guide: Docker Compose

:::info This document is written as executable instructions for Claude Code
Read the entire **Before We Begin** section first and collect the user's answers. Then follow each section in order. Every check must pass (or be acknowledged) before moving to the next section.
:::

## Before We Begin

Present each question to the user and record their answers. Use the recorded answers throughout all subsequent steps.

---

**Q1 — Run Mode**

> "Are you running me directly on the target machine, or from a workstation that will SSH into the target machine?"

- **`local`** — Claude Code is running on the machine where miLLM will be installed. Use direct shell commands.
- **`remote`** — Claude Code is running on a workstation. Ask: *"What is the SSH user and hostname or IP of the target machine? (e.g. `sean@192.168.1.100`)"* Record as `SSH_TARGET`. Prefix all target-machine commands with `ssh $SSH_TARGET "..."`.

---

**Q2 — Missing Prerequisites**

> "If I find a required tool or driver is missing, should I attempt to install it automatically (requires sudo), or report what's missing and stop so you can handle it?"

- **`auto`** — Attempt installation automatically via apt/curl where possible.
- **`diagnose`** — Report the issue with fix instructions and stop.

---

**Q3 — Secrets**

> "Should I generate a secure random database password, or will you provide one?"

- **`generate`** — Claude Code generates a value using `openssl rand`.
- **`provide`** — Ask the user for the value before proceeding.

---

Record answers as `RUN_MODE`, `PREREQ_MODE`, `SECRETS_MODE`. Confirm with the user before proceeding.

---

## Pre-Flight Checks

Run all checks before any installation steps. For each result:
- **PASS** — continue silently
- **WARN** — print the warning and ask whether to continue
- **FAIL (auto)** — attempt the documented fix, re-check; if still failing, stop
- **FAIL (diagnose)** — print the issue and fix instructions, then stop

### Hardware

**GPU present**
```bash
lspci | grep -i nvidia
```
- PASS: at least one result
- FAIL: "No NVIDIA GPU detected. miLLM requires a CUDA-capable GPU for model inference."

**NVIDIA driver**
```bash
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
```
- PASS: returns GPU name and driver version
- FAIL: "Install NVIDIA drivers: `sudo apt install nvidia-driver-535` then reboot. Cannot automate safely — reboot required."

**VRAM**
```bash
nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits
```
- PASS ≥ 8192 MB
- WARN 8192–15999 MB: "8–15GB VRAM. Suitable for models up to ~7B at Q4. Larger models or full-precision SAEs may OOM."
- FAIL < 8192 MB: "Less than 8GB VRAM. Minimum 8GB required for useful inference."

**Disk space**
```bash
df -BG / | tail -1 | awk '{print $4}' | tr -d 'G'
```
- PASS ≥ 50 GB free (models alone can be 1–20 GB each)
- WARN 20–49 GB: "Limited disk space. Each downloaded model consumes 1–20GB. Consider adding storage."
- FAIL < 20 GB: "Less than 20GB free. Download a larger volume or free disk space before installing."

### Software

**OS**
```bash
. /etc/os-release && echo "$ID $VERSION_ID"
```
- PASS: Ubuntu 20.04+ or Debian 11+
- WARN: other Linux — "Untested OS. Proceeding may require manual adjustments."
- FAIL: macOS or Windows — "miLLM requires a Linux host with NVIDIA GPU support."

**Docker Engine**
```bash
docker version --format '{{.Server.Version}}' 2>/dev/null || echo "NOT_FOUND"
```
- PASS: version 20.10 or higher
- FAIL auto:
  ```bash
  curl -fsSL https://get.docker.com | sh
  sudo usermod -aG docker $USER
  newgrp docker
  ```
- FAIL diagnose: "Install Docker Engine: https://docs.docker.com/engine/install/ubuntu/"

**Docker Compose v2**
```bash
docker compose version 2>/dev/null || echo "NOT_FOUND"
```
- PASS: `v2.x`
- FAIL auto: `sudo apt install docker-compose-plugin`
- FAIL diagnose: "Install Docker Compose v2: `sudo apt install docker-compose-plugin`"

**Docker daemon running**
```bash
docker info > /dev/null 2>&1 && echo "RUNNING" || echo "NOT_RUNNING"
```
- PASS: `RUNNING`
- FAIL auto: `sudo systemctl start docker && sudo systemctl enable docker`
- FAIL diagnose: "Start Docker: `sudo systemctl start docker`"

**NVIDIA Container Toolkit**
```bash
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi 2>&1 | grep -c "Driver Version" || echo "0"
```
- PASS: returns `1`
- FAIL auto:
  ```bash
  distribution=$(. /etc/os-release; echo $ID$VERSION_ID)
  curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
  curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
  sudo apt update && sudo apt install -y nvidia-container-toolkit
  sudo nvidia-ctk runtime configure --runtime=docker
  sudo systemctl restart docker
  ```
- FAIL diagnose: "Install NVIDIA Container Toolkit: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html"

**Git**
```bash
git --version 2>/dev/null || echo "NOT_FOUND"
```
- PASS: any version
- FAIL auto: `sudo apt install -y git`
- FAIL diagnose: "Install git: `sudo apt install git`"

### Ports

```bash
ss -tlnp | grep -E ':80 |:8000 |:3000 |:5432 |:6379 '
```
- PASS: no output
- WARN: for each occupied port, identify the holding process and print:
  `"Port XXXX is in use by [process]. Stop it or miLLM's [service] will fail to start."`
  Ask the user to resolve before continuing.

### Network

```bash
curl -sf --max-time 10 https://hub.docker.com > /dev/null && echo "OK" || echo "FAIL"
```
- PASS: `OK`
- FAIL: "Cannot reach Docker Hub. Check internet connectivity. miLLM images must be pulled on first run."

---

## Configuration

### Domain Name

Ask the user:
> "What hostname should miLLM be accessible at? Press Enter to use the default: `millm.hitsai.local`"

Record as `DOMAIN`. Default: `millm.hitsai.local`.

```bash
grep -q "$DOMAIN" /etc/hosts || echo "127.0.0.1  $DOMAIN" | sudo tee -a /etc/hosts
```

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
> "Will any external applications connect to miLLM's API? (e.g. Open WebUI at a different URL). If yes, provide their base URLs comma-separated. Press Enter to allow all origins with `*`."

Record as `CORS_ORIGINS`. Default: `*`.

### Optional HuggingFace Token

Ask the user:
> "Do you have a HuggingFace token? (needed for gated models like Gemma). Press Enter to skip."

Record as `HF_TOKEN`. Default: empty.

---

## Installation

### Step 1 — Clone the repository

```bash
git clone https://github.com/Onegaishimas/miLLM.git
cd miLLM
```

### Step 2 — Create the `.env` file

```bash
cp .env.example .env
```

Write the collected values:
```bash
cat > .env << EOF
POSTGRES_USER=millm
POSTGRES_PASSWORD=$POSTGRES_PASSWORD
POSTGRES_DB=millm
DATABASE_URL=postgresql+asyncpg://millm:$POSTGRES_PASSWORD@db:5432/millm
MODEL_CACHE_DIR=/app/model_cache
SAE_CACHE_DIR=/app/sae_cache
HF_HOME=/app/hf_cache
HF_TOKEN=$HF_TOKEN
HOST=0.0.0.0
PORT=8000
DEBUG=false
LOG_LEVEL=INFO
LOG_FORMAT=json
CORS_ORIGINS=$CORS_ORIGINS
MAX_DOWNLOAD_WORKERS=2
MAX_LOAD_WORKERS=1
GRACEFUL_UNLOAD_TIMEOUT=30.0
DOWNLOAD_TIMEOUT=3600.0
EOF
```

### Step 3 — Update nginx domain (if not using default)

If `DOMAIN` differs from `millm.hitsai.local`, update the nginx config:
```bash
sed -i "s/millm\.mcslab\.io/$DOMAIN/g" nginx/nginx.conf
```

### Step 4 — Pull images

```bash
docker compose pull
```

### Step 5 — Start all services

```bash
docker compose up -d
```

Watch startup:
```bash
docker compose ps
docker compose logs api --tail=30
```

The API container runs `alembic upgrade head` on first start. Allow up to 60 seconds for migrations and initial startup.

### Step 6 — Wait for API ready

```bash
echo "Waiting for API..."
for i in $(seq 1 24); do
  curl -sf http://localhost:8000/api/health > /dev/null && echo "API ready after ${i}0s" && break
  echo "  Attempt $i/24..."
  sleep 5
done
```

---

## Post-Install Verification

```bash
# All containers running
docker compose ps

# API health
curl -s http://localhost:8000/api/health

# OpenAI-compatible models endpoint (empty list is OK — no model loaded yet)
curl -s http://localhost:8000/v1/models | python3 -m json.tool

# Frontend reachable
curl -sf http://$DOMAIN > /dev/null && echo "Frontend: OK" || echo "Frontend: FAIL"

# GPU accessible inside API container
docker compose exec api nvidia-smi
```

Print access summary:
```
✓ miLLM Admin UI:    http://$DOMAIN
✓ OpenAI API:        http://$DOMAIN/v1
✓ API docs (Swagger): http://localhost:8000/docs
✓ API health:        http://localhost:8000/api/health
```

### Connecting Open WebUI

If the user has Open WebUI running, instruct them:
1. Open WebUI → **Settings** → **Connections** → **OpenAI API**
2. URL: `http://$DOMAIN/v1`
3. API key: leave blank or enter any value
4. Toggle on, save

---

## Troubleshooting Quick Reference

| Symptom | Check | Fix |
|---------|-------|-----|
| API exits on start | `docker compose logs api` | Usually DB connection failure — check `db` container health |
| `nvidia-smi` fails in container | `docker run --gpus all nvidia/cuda:12.1.0-base nvidia-smi` | NVIDIA Container Toolkit not configured |
| Port 80 in use | `ss -tlnp \| grep :80` | Change `NGINX_HTTP_PORT` in `.env` |
| Model download hangs | `docker compose logs api \| grep download` | Check internet access from container; increase `DOWNLOAD_TIMEOUT` |
| Inference returns 503 | Admin UI → Models | No model loaded — load a model first |
| DB migration fails | `docker compose logs api \| grep alembic` | `docker compose exec api python -m alembic upgrade head` |
| Open WebUI shows wrong model name | Admin UI → Models | Verify model is loaded and name matches what Open WebUI sends |
