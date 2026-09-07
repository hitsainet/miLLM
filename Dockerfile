# miLLM Backend Dockerfile
# Two targets: 'runtime' (slim, for k8s) and 'cuda' (full CUDA, for local dev)

# =============================================================================
# Runtime Stage — slim Python, GPU via nvidia-container-toolkit on host
# =============================================================================
FROM python:3.11-slim as runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Install minimal system deps and upgrade packages with known CVEs.
# gcc + libc6-dev are required by triton's JIT kernel compiler, which
# torch.compile's reduce-overhead mode uses to compile CUDA kernels at
# first inference. libc6-dev provides stdlib.h and friends that triton
# includes when building cuda_utils.c; without it gcc is present but
# cannot find standard headers.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    curl \
    gcc \
    libc6-dev \
    && apt-get upgrade -y openssl \
    && rm -rf /var/lib/apt/lists/*

# Copy only requirements first for better caching
COPY pyproject.toml ./
COPY millm/__init__.py millm/__init__.py

# Upgrade wheel to fix CVE-2026-24049 (path traversal, fixed in 0.46.2)
RUN pip install --no-cache-dir --upgrade "wheel>=0.46.2"

# Install dependencies (torch bundles its own CUDA runtime)
RUN pip install --no-cache-dir . || pip install --no-cache-dir -e .

# Install mamba-ssm from pre-built wheels (requires CUDA at compile time,
# so we use pre-built wheels that match the torch CUDA version)
# stderr is NOT discarded: a `2>/dev/null` here is why the flash-attn failure
# could not be diagnosed from the build log at all.
RUN pip install --no-cache-dir causal-conv1d mamba-ssm --no-build-isolation || \
    echo "WARN: mamba-ssm not available as pre-built wheel, SSM models will use slow torch fallback"

# llama-cpp-python, for serving GGUF files, from a PRE-BUILT CUDA wheel.
#
# Not built from source, deliberately. This image is python:3.11-slim: it has
# gcc and libc6-dev (for triton's JIT) but no cmake, no ninja, no g++ and no
# nvcc, and llama-cpp-python's scikit-build-core backend needs all of them plus
# the CUDA dev headers to compile with -DGGML_CUDA=on. Adding that toolchain is
# 2-3 GB of image and a 10-30 minute build. See the flash-attn note below: this
# image already has one dependency that cannot be built here, and the lesson
# recorded there is to prove the need before paying that cost.
#
# Failure is tolerated but LOUD, and stderr is not discarded — the same rule as
# the line above, written after a flash-attn failure that could not be
# diagnosed from the build log. Without the wheel, GGUF models refuse to load
# with a clear message; everything else serves normally.
# cu130 MATCHES this image's torch (2.13.0+cu130 — verified in the running
# pod, not assumed). The index publishes py3-none / manylinux_2_35 wheels, so
# they are Python-version agnostic; >=0.3.29 is the floor that carries a proper
# manylinux tag, which pip will actually accept. glibc in this base (bookworm,
# 2.36) satisfies manylinux_2_35.
RUN pip install --no-cache-dir "llama-cpp-python>=0.3.29" \
      --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu130 || \
    echo "WARN: llama-cpp-python not installed; GGUF models will refuse to load"

# NOT installing the flash-attn package. Deliberate, and verified on the box.
#
# It was added on the premise that "SDPA is materially slower". That is FALSE
# for the torch this image ships: PyTorch's own SDPA dispatches to the
# FlashAttention kernel. Checked in the running container (torch 2.13.0+cu130):
#
#   torch.backends.cuda.flash_sdp_enabled()  -> True
#   can_use_flash_attention(<fp16 2048-len>) -> True
#
# The separate package also never installed: the build attempt failed ~6s in and
# hit the fallback below, so the image has been running on SDPA the whole time
# regardless — while the build log implied an optimisation was being attempted.
#
# There is additionally no published flash-attn wheel for cu130 + torch 2.13, so
# pip fell back to a source build that cannot succeed in this image.
#
# If this is revisited: prove the delta by MEASURING against SDPA on this
# hardware first. Do not reintroduce it on the assumption that it is faster.

# Copy application code
COPY millm/ /app/millm/
COPY alembic.ini /app/
COPY docker-entrypoint.sh /app/

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash millm \
    && chown -R millm:millm /app \
    && chmod +x /app/docker-entrypoint.sh

# Create model and SAE cache directories
RUN mkdir -p /app/model_cache /app/sae_cache && chown -R millm:millm /app/model_cache /app/sae_cache

# Switch to non-root user
USER millm

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/health')" || exit 1

# Entrypoint runs migrations before starting the app
ENTRYPOINT ["/app/docker-entrypoint.sh"]

# Default command
CMD ["python", "-m", "uvicorn", "millm.main:app", "--host", "0.0.0.0", "--port", "8000"]

# =============================================================================
# Development Stage
# =============================================================================
FROM runtime as development

USER root

# Install development tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install development dependencies
RUN pip install --no-cache-dir pytest pytest-asyncio pytest-cov ruff mypy

USER millm

# Development command with hot reload
CMD ["python", "-m", "uvicorn", "millm.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
