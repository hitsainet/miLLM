---
sidebar_position: 1
title: Introduction to miLLM
---

# Introduction to miLLM

miLLM is a **Mechanistic Interpretability LLM Server** — an inference server that combines standard LLM serving with SAE-based feature steering and real-time activation monitoring.

## Architecture

miLLM runs as a multi-service stack:

| Service | Purpose |
|---------|---------|
| **FastAPI Backend** | API server with OpenAI-compatible endpoints |
| **PostgreSQL** | Stores model metadata, SAE records, profiles |
| **Redis** | Caching and pub/sub for real-time updates |
| **Nginx** | Reverse proxy and static file serving |
| **React Admin UI** | Web dashboard for management and monitoring |

## How It Fits In

miLLM is designed to work standalone or as part of a larger research stack:

- **With miStudio:** miStudio uses miLLM's OpenAI-compatible API for feature labeling via the "OpenAI Compatible" labeling method
- **With Open WebUI:** Connect Open WebUI to miLLM's `/v1` endpoint for steered chat conversations
- **With custom scripts:** Use the OpenAI Python SDK pointed at miLLM for programmatic access

:::info OpenAI API Compatibility
Any tool that speaks the OpenAI API format can use miLLM. Set `base_url` to your miLLM instance (e.g., `http://millm.mcslab.io/v1`) and it works as a drop-in replacement — with the addition of feature steering.
:::

## Key Concepts

- **Feature Steering:** Adding a scaled decoder direction to the model's residual stream during inference, amplifying or suppressing specific learned concepts
- **SAE Attachment:** Loading a Sparse Autoencoder's weights and hooking it into a specific model layer
- **Probe Monitoring:** Observing which SAE features activate during normal inference, without modifying the output
- **Profiles:** Named configurations of feature steering values that can be saved, shared, and activated on-demand
