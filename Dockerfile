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
RUN pip install --no-cache-dir causal-conv1d mamba-ssm --no-build-isolation 2>/dev/null || \
    echo "WARN: mamba-ssm not available as pre-built wheel, SSM models will use slow torch fallback"

# FlashAttention-2 (SM80+; the RTX 3090 this deploys to is SM86).
# Transformers falls back to SDPA without it, which is materially slower on the
# long prompts labeling sends (~2,100 tokens each).
# NON-FATAL by design, exactly like mamba-ssm above: flash-attn has no universal
# pre-built wheel and compiling it from source can take 30+ minutes or fail
# outright. A failure here must not break the image — attention silently falls
# back to SDPA, which is correct, just slower.
RUN pip install --no-cache-dir flash-attn --no-build-isolation 2>/dev/null || \
    echo "WARN: flash-attn unavailable, attention falls back to SDPA (slower on long prompts)"

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
