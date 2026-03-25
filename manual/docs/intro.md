---
sidebar_position: 1
slug: /
title: miLLM User Manual
---

# miLLM User Manual

Welcome to the **miLLM** (Mechanistic Interpretability LLM Server) user manual.

miLLM is an OpenAI-compatible LLM inference server with built-in support for **Sparse Autoencoder (SAE) feature steering** and **real-time activation monitoring**. It serves as the inference backbone for mechanistic interpretability research, providing a REST API that any tool — including miStudio, Open WebUI, or custom scripts — can use to run steered inference.

## What miLLM Does

- **Serves LLMs** via an OpenAI-compatible API (`/v1/chat/completions`)
- **Attaches SAEs** to model layers for real-time feature analysis
- **Steers model behavior** by amplifying or suppressing specific features
- **Monitors activations** to observe which features fire during inference
- **Manages profiles** to save and restore steering configurations

## Quick Navigation

| Section | What You'll Learn |
|---------|------------------|
| [Getting Started](/getting-started/introduction) | What miLLM is and how to install it |
| [Model Management](/features/model-management) | Downloading, loading, and quantizing models |
| [SAE Management](/features/sae-management) | Downloading and attaching Sparse Autoencoders |
| [Feature Steering](/features/feature-steering) | Manipulating model behavior via features |
| [Probe Monitoring](/features/probe-monitoring) | Real-time activation observation |
| [Profiles](/features/profiles) | Saving and sharing steering configurations |
| [API Reference](/api/openai-compatible) | OpenAI-compatible and management endpoints |
| [Troubleshooting](/troubleshooting) | Common issues and fixes |
