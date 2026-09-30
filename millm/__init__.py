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
