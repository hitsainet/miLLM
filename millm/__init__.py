"""
miLLM - Mechanistic Interpretability LLM Server

A server for running LLMs with Sparse Autoencoder (SAE) support
for interpretability research and feature steering.

Deployed via GitOps: images are built selectively by CI and rolled out by
ArgoCD Image Updater (see k8s/argocd/millm-app.yaml).
"""

#: ⚠ KEEP IN STEP WITH THE `VERSION` FILE AT THE REPO ROOT, which `GET /api/health/version`
#: prefers over this constant. They disagreed — VERSION said 0.5.0 while this said 0.5.1 — so the
#: OpenAPI document and the health endpoint reported different versions of the same running
#: server. `test_one_version_everywhere.py` now fails if they drift again.
__version__ = "0.5.0"
__author__ = "miLLM Team"


# ⚠ EXPANDABLE SEGMENTS, OR AN UNLOAD RETURNS NOTHING (2026-10-05). Set here, in the package's first
# lines, because the allocator reads it once, before torch makes its first CUDA allocation — a
# setting applied later is ignored. Measured on JEV-9B (Qwen3.5, 17 GB bf16): after one request,
# unload left 8 MiB ALLOCATED and 17,080 MiB RESERVED, so the card stayed full with no model
# loaded. The 8 MiB is no Python tensor (cuBLAS's workspace is the size and lifetime that fits);
# it sits inside the one block the weights were carved from, and `empty_cache` can only return a
# block that is entirely free. With expandable segments the same unload ends at 20 MiB reserved.
# An operator's own setting wins; only a value that does not mention the option is extended.
import os as _os

from millm.core.cuda_allocator import allocator_env as _allocator_env

_name, _value = _allocator_env(_os.environ)
_os.environ[_name] = _value
del _os, _allocator_env, _name, _value
