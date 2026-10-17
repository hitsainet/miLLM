---
sidebar_position: 1
slug: /
title: miLLM User Manual
---

# miLLM User Manual

**miLLM** (Mechanistic Interpretability LLM Server) is an OpenAI-compatible LLM inference server with built-in support for **Sparse Autoencoder (SAE) feature steering** and **real-time activation monitoring**.

It answers a question ordinary inference servers can't: *not just what the model says, but why — and what happens if you change it.* Attach an SAE to a model layer, dial individual features up or down, and observe the causal effect on generated text through the same OpenAI API your existing tools already speak.

![miLLM Dashboard](/img/miLLM_Dashboard_01.jpg)

## What miLLM Does

- **Serves LLMs** via an OpenAI-compatible API (`/v1/chat/completions`, `/v1/completions`, `/v1/embeddings`) — works with the OpenAI SDK, Open WebUI, LangChain, or plain `curl`
- **Attaches SAEs** (SAELens format, e.g. GemmaScope) to any residual-stream layer of the loaded model
- **Steers model behavior** by adding scaled feature directions to the residual stream during inference — Neuronpedia-compatible strengths
- **Monitors activations** in real time to observe which features fire during inference
- **Manages profiles** to save, restore, export, and apply steering configurations — including per-request via the API
- **Runs anywhere** with Docker Compose or Kubernetes, one GPU is enough

## Choose Your Path

| I want to… | Start here |
|------------|-----------|
| Get running in 10 minutes | [Quickstart](/getting-started/quickstart) |
| Understand what SAEs and steering actually are | [Concepts: Interpretability](/concepts/interpretability) |
| Steer Gemma with a GemmaScope SAE, end to end | [Tutorial: Steering Gemma](/tutorials/steering-gemma) |
| Use miLLM as a backend for Open WebUI | [Tutorial: Open WebUI](/tutorials/open-webui) |
| Script experiments in Python | [Tutorial: Python Scripting](/tutorials/python-scripting) |
| Look up an endpoint | [API Reference](/api/overview) |
| Configure the server | [Configuration Reference](/reference/configuration) |
| Fix a problem | [Troubleshooting](/troubleshooting) |

## The Core Workflow

Everything in miLLM revolves around one loop:

1. **Load a model** — download from HuggingFace (optionally quantized), load to GPU. [Model Management →](/features/model-management)
2. **Attach an SAE** — download a matching SAE and hook it to a layer. [SAE Management →](/features/sae-management)
3. **Steer** — set per-feature strengths and watch outputs change. [Feature Steering →](/features/feature-steering)
4. **Probe** — monitor which features activate during inference. [Probe Monitoring →](/features/probe-monitoring)
5. **Save** — capture the configuration as a profile you can re-apply, export, or invoke per-request. [Profiles →](/features/profiles)

## Two APIs, One Server

miLLM exposes two API surfaces on the same port:

- **`/v1/*` — OpenAI-compatible inference.** Drop-in replacement for the OpenAI API. Steering configured on the server applies transparently to every completion; a `profile` request parameter applies a saved steering profile for a single request.
- **`/api/*` — Management.** Models, SAEs, steering, monitoring, profiles, health. Everything the Admin UI does is available here, returning a consistent `{success, data, error}` envelope.

A WebSocket layer (Socket.IO) streams download progress, GPU metrics, steering changes, and live activations to the Admin UI — or to your own clients. [WebSocket Events →](/api/websockets)
