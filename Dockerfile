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

# Install minimal system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy only requirements first for better caching
COPY pyproject.toml ./
COPY millm/__init__.py millm/__init__.py

# Install dependencies (torch bundles its own CUDA runtime)
RUN pip install --no-cache-dir . || pip install --no-cache-dir -e .

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
