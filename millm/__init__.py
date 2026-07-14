"""
miLLM - Mechanistic Interpretability LLM Server

A server for running LLMs with Sparse Autoencoder (SAE) support
for interpretability research and feature steering.

Deployed via GitOps: images are built selectively by CI and rolled out by
ArgoCD Image Updater (see k8s/argocd/millm-app.yaml).
"""

__version__ = "0.5.1"
__author__ = "miLLM Team"
