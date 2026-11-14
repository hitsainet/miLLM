---
sidebar_position: 1
title: Introduction to miServer
---

# Introduction to miServer

miServer is a **Mechanistic Interpretability LLM Server** — an inference server that combines standard LLM serving with SAE-based feature steering and real-time activation monitoring.

## Architecture

miServer runs as a multi-service stack:

| Service | Purpose |
|---------|---------|
| **FastAPI Backend** | API server with OpenAI-compatible endpoints |
| **PostgreSQL** | Stores model metadata, SAE records, profiles |
| **Redis** | Caching and pub/sub for real-time updates |
| **Nginx** | Reverse proxy and static file serving |
| **React Admin UI** | Web dashboard for management and monitoring |

## How It Fits In

miServer is designed to work standalone or as part of a larger research stack:

- **With miWorkbench:** miWorkbench uses miServer's OpenAI-compatible API for feature labeling via the "OpenAI Compatible" labeling method
- **With Open WebUI:** Connect Open WebUI to miServer's `/v1` endpoint for steered chat conversations
- **With custom scripts:** Use the OpenAI Python SDK pointed at miServer for programmatic access

:::info OpenAI API Compatibility
Any tool that speaks the OpenAI API format can use miServer. Set `base_url` to your miServer instance (e.g., `http://miserver.hitsai.local/v1`) and it works as a drop-in replacement — with the addition of feature steering.
:::

## Key Concepts

- **Feature Steering:** Adding a scaled decoder direction to the model's residual stream during inference, amplifying or suppressing specific learned concepts
- **SAE Attachment:** Loading a Sparse Autoencoder's weights and hooking it into a specific model layer
- **Feature Monitor:** Observing which SAE features activate during normal inference, without modifying the output
- **Profiles:** Named configurations of feature steering values that can be saved, shared, and activated on-demand
