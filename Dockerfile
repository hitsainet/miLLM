# miLLM Backend Dockerfile
# Two targets: 'runtime' (slim, for k8s) and 'cuda' (full CUDA, for local dev)

# =============================================================================
# Runtime Stage — slim Python, GPU via nvidia-container-toolkit on host
# =============================================================================
FROM python:3.13-slim AS runtime

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
# libseccomp2: the rule-runner worker installs its OWN seccomp filter through it (D55); already in
# the base image (apt depends on it), named here so nothing can drop it quietly.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    libseccomp2 \
    curl \
    gcc \
    libc6-dev \
    && apt-get upgrade -y openssl \
    && rm -rf /var/lib/apt/lists/*

# Copy only requirements first for better caching.
# constraints.txt is THE LOCK (2026-10-21): every pip install below passes `-c constraints.txt`, the
# same file CI installs from, so the image cannot resolve a version CI did not test. The checker
# runs after the last install and fails the build on any unpinned or mismatched distribution.
# tests/unit/test_dependency_lock.py pins every install line here to the lock.
COPY pyproject.toml constraints.txt ./
COPY docker/check_installed_against_constraints.py docker/check_installed_against_constraints.py
COPY millm/__init__.py millm/__init__.py

# Upgrade wheel to fix CVE-2026-24049 (path traversal, fixed in 0.46.2)
RUN pip install --no-cache-dir -c constraints.txt "wheel==0.48.0"

# Install dependencies (torch bundles its own CUDA runtime)
RUN pip install --no-cache-dir -c constraints.txt . || pip install --no-cache-dir -c constraints.txt -e .

# Install mamba-ssm from pre-built wheels (requires CUDA at compile time,
# so we use pre-built wheels that match the torch CUDA version)
# stderr is NOT discarded: a `2>/dev/null` here is why the flash-attn failure
# could not be diagnosed from the build log at all.
# Versions are the `cuda` extra's (pyproject.toml), held equal by test_dependency_lock.py.
RUN pip install --no-cache-dir -c constraints.txt "causal-conv1d==1.7.0" "mamba-ssm==2.3.2.post1" --no-build-isolation || \
    echo "WARN: mamba-ssm not available as pre-built wheel, SSM models will use slow torch fallback"

# llama-cpp-python, for serving GGUF files, from a PRE-BUILT CUDA wheel.
#
# Not built from source, deliberately. This image is python:3.13-slim: it has
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
# cu130 MATCHES this image's torch (2.14.1+cu130, measured in the running pod
# 2026-10-16; it was 2.13.0+cu130 when this line was first written). The index
# publishes py3-none / manylinux_2_35 wheels, so they are Python-version
# agnostic; 0.3.29 was the first release with a proper manylinux tag. PINNED to
# 0.3.36, the version the live pod runs (2026-10-17), the same as the gguf extra
# in pyproject.toml (tests/unit/test_torch_pin.py holds the two equal). glibc in this base (Debian 13
# trixie, glibc 2.41, python:3.13-slim checked 2026-10-21) satisfies manylinux_2_35. The wheel is
# py3-none (a ctypes binding), so the 3.11 -> 3.13 move needed no new wheel.
# --only-binary is load-bearing, not belt-and-braces. PyPI publishes this
# project as an SDIST ONLY, and with --extra-index-url pip merges both indexes
# and picks the HIGHEST version — wheel-vs-sdist only breaks ties within one
# version. The moment PyPI is a release ahead of the CUDA index (it usually
# is), pip chooses the sdist, tries to compile it without cmake or nvcc, fails,
# and the `|| echo WARN` swallows it: an image that builds green and refuses
# every GGUF model. --only-binary=:all: makes that resolution impossible.
RUN pip install --no-cache-dir -c constraints.txt --only-binary=:all: "llama-cpp-python==0.3.36" \
      --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu130 || \
    echo "WARN: llama-cpp-python not installed; GGUF models will refuse to load"

# The lock, checked where it is installed: an installed distribution that constraints.txt does not
# pin, or pins at another version, fails the build here rather than shipping.
RUN python docker/check_installed_against_constraints.py constraints.txt

# NOT installing the flash-attn package. Deliberate, and verified on the box.
#
# It was added on the premise that "SDPA is materially slower". That is FALSE
# for the torch this image shipped when it was checked: PyTorch's own SDPA
# dispatches to the FlashAttention kernel. Checked in the running container on
# torch 2.13.0+cu130 (the image now runs 2.14.1+cu130; the check was not
# repeated on 2.14.1):
#
#   torch.backends.cuda.flash_sdp_enabled()  -> True
#   can_use_flash_attention(<fp16 2048-len>) -> True
#
# The separate package also never installed: the build attempt failed ~6s in and
# hit the fallback below, so the image has been running on SDPA the whole time
# regardless — while the build log implied an optimisation was being attempted.
#
# There was additionally no published flash-attn wheel for cu130 + torch 2.13
# (2.14.1 not checked), so pip fell back to a source build that cannot succeed
# in this image.
#
# If this is revisited: prove the delta by MEASURING against SDPA on this
# hardware first. Do not reintroduce it on the assumption that it is faster.

# Copy application code
COPY millm/ /app/millm/
# The vendored contract schemas: the adapter and checkpoint manifest validators read them
# at runtime (docs/schemas/ is the only part of docs/ the public mirror keeps).
COPY docs/schemas/ /app/docs/schemas/
COPY alembic.ini /app/
COPY docker-entrypoint.sh /app/

# Create non-root user for security. uid/gid 1000 PINNED, not left to useradd's "first free
# id": miForge's F11 hot-update worker must run as this exact uid while the hostIPC component
# is on, because CUDA/torch create their IPC files in /dev/shm owner-only (0600, regardless of
# umask) and only a same-uid process can open them (verified live 2026-10-17). Pinned by
# tests/unit/test_k8s_hostipc_component.py.
RUN groupadd --gid 1000 millm \
    && useradd --uid 1000 --gid 1000 --create-home --shell /bin/bash millm \
    && chown -R millm:millm /app \
    && chmod +x /app/docker-entrypoint.sh

# Create the cache, batch-file and checkpoint directories. A fresh named volume copies the
# ownership of the directory it mounts over, so one missing here is root-owned and unwritable
# by the millm user (compose mounts batch_files and checkpoints there).
RUN mkdir -p /app/model_cache /app/sae_cache /app/batch_files /app/checkpoints \
    && chown -R millm:millm /app/model_cache /app/sae_cache /app/batch_files /app/checkpoints
# 039 S3: the rule runner's socket directory. A fresh Compose named volume copies this ownership,
# so the worker (uid 1001, group 1000) can create its socket there; Kubernetes mounts an emptyDir over it.
# Review R1 (M1): group-writable, as the worker runs as uid 1001 in miLLM's group 1000.
RUN mkdir -p /run/millm-rules && chown millm:millm /run/millm-rules && chmod 0770 /run/millm-rules

# 039 S3 (FR-39.26; TD39-33): the guest the `rule-runner` sidecar runs operator code in. Fetched from
# the vendored runner contract's `guest.bundle.snapshot.url` and KEPT ONLY IF its sha256 and size
# are the contract's (the published `runner-bundle-v1` bytes; never rebuilt here, never committed),
# then precompiled for this Wasmtime build (`snapshot.cwasm` + its sidecar). Review R1 (L3): AFTER
# `chown -R millm:millm /app`, so the bundle - the native `.cwasm` is loaded without a digest - stays
# root-owned and no runtime uid can rewrite it (018 R1-L5). A node whose CPU
# rejects the precompiled form compiles the verified snapshot in process instead.
# D55: the worker's own seccomp filter must BUILD on this image's libseccomp (built, never loaded
# here); an image that cannot build it never ships.
RUN python -m millm.rules.runner_worker fetch /app/runner-bundle \
    && python -m millm.rules.runner_worker precompile /app/runner-bundle \
    && python -m millm.rules.runner_worker seccomp-check

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
FROM runtime AS development

USER root

# Install development tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install development dependencies
RUN pip install --no-cache-dir -c constraints.txt "pytest==9.1.1" "pytest-asyncio==1.4.0" "pytest-cov==7.1.0" "ruff==0.17.1" "mypy==2.4.0" \
    && python docker/check_installed_against_constraints.py constraints.txt

USER millm

# Development command with hot reload
CMD ["python", "-m", "uvicorn", "millm.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
