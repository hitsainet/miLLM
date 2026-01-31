# miLLM Backend Dockerfile
# Supports NVIDIA GPU for model inference

# =============================================================================
# Runtime Stage (Single stage for simplicity with CUDA support)
# =============================================================================
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04 as runtime

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DEBIAN_FRONTEND=noninteractive

WORKDIR /app

# Install Python 3.11 from deadsnakes PPA and dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-venv \
    python3.11-dev \
    python3-pip \
    libpq5 \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.11 /usr/bin/python \
    && ln -sf /usr/bin/python3.11 /usr/bin/python3

# Upgrade pip
RUN python -m pip install --upgrade pip setuptools wheel

# Copy only requirements first for better caching
COPY pyproject.toml ./
COPY millm/__init__.py millm/__init__.py

# Install dependencies
RUN pip install --no-cache-dir . || pip install --no-cache-dir -e .

# Copy application code
COPY millm/ /app/millm/
COPY alembic.ini /app/

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash millm \
    && chown -R millm:millm /app

# Create model and SAE cache directories
RUN mkdir -p /app/model_cache /app/sae_cache && chown -R millm:millm /app/model_cache /app/sae_cache

# Switch to non-root user
USER millm

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/health')" || exit 1

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
