# miLLM — Mechanistic Interpretability LLM Server

miLLM is a self-hosted LLM inference server purpose-built for mechanistic interpretability research. It combines standard LLM serving with Sparse Autoencoder (SAE) integration, real-time activation monitoring, and feature steering — all behind an OpenAI-compatible API that any existing tool can speak to without modification.

The core proposition: run a language model locally, attach an SAE to one of its layers, and then causally intervene in the model's computation by amplifying or suppressing specific learned features — all while observing which features activate in real time during inference.

---

## Why miLLM Exists

Existing local LLM inference servers — Ollama, vLLM, llama.cpp — treat the model as a black box. Tokens go in, tokens come out. If you want to change model behavior, your options are system prompts (which consume context window, are brittle, and don't generalize) or fine-tuning (expensive, inflexible, and requires a new model checkpoint for each behavioral variant). Neither approach gives you insight into *why* the model behaves as it does.

Mechanistic interpretability offers a different path. Sparse Autoencoders, trained on a model's internal activations, decompose the dense residual stream into a sparse set of interpretable features — directions in activation space that correspond to human-understandable concepts. Feature steering then lets you add a scaled copy of a feature's decoder direction directly to the model's residual stream during inference, amplifying or suppressing that concept in real time. The behavioral change happens without touching the model weights, without consuming context, and without fine-tuning — and it can be toggled on and off mid-session.

No existing inference server supports this. miLLM is built around it from the ground up. The OpenAI-compatible API surface means existing tooling — Open WebUI, the OpenAI Python SDK, curl, any OpenAI API client — works without modification. The interpretability layer sits underneath, available whenever you need it.

A typical research workflow: download a model and quantize it for your available VRAM; download an SAE trained on that model from a repository like Gemma Scope; attach the SAE to a specific residual stream layer; use the Probe Monitoring page to observe which features activate during normal inference; identify features of interest; configure Feature Steering to amplify or suppress those features and observe the effect on model output. The result is a causal test: if steering feature 1234 toward "French" causes the model to respond in French, that feature plausibly encodes something about that concept.

This workflow is also the basis for integration with [miStudio](https://github.com/Onegaishimas/miStudio), which uses miLLM's OpenAI-compatible endpoint for feature labeling — automatically generating natural-language descriptions of what each SAE feature detects, closing the loop between SAE training and deployed steering.

---

## Architecture

miLLM runs as a coordinated multi-service stack. Each component has a specific responsibility:

**FastAPI Backend** — The core API server. It exposes two API surfaces: an OpenAI-compatible `/v1` endpoint for standard inference, and a management API for model loading, SAE attachment, steering configuration, and profile management. The backend owns all ML operations: model loading via HuggingFace Transformers, quantization via bitsandbytes, SAE hooking via PyTorch forward hooks, and activation capture during inference.

**React Admin UI** — A web dashboard for interacting with all management features: downloading and loading models, downloading and attaching SAEs, configuring feature steering, watching live activation charts from the probe monitor, and managing profiles. The UI communicates with the backend over both REST and WebSocket, with the WebSocket connection carrying real-time activation data during inference.

**PostgreSQL** — Stores durable state: model metadata, SAE records (including file paths and layer configurations), steering profiles, and probe monitoring history. The database allows miLLM to restart cleanly and restore its configuration without re-downloading anything.

**Redis** — Used for caching and pub/sub messaging. The real-time activation data emitted by the probe monitor during inference flows through Redis to the WebSocket layer before reaching the browser.

**Nginx** — Reverse proxy that routes requests to the backend and serves the Admin UI static files under a single domain. This is what makes miLLM accessible at a single URL (e.g., `http://millm.hitsai.local`) rather than requiring clients to know about separate ports for the API and UI.

### The SAE Integration Layer

The most technically distinctive part of miLLM is how it integrates SAEs into the inference pipeline without modifying the base model weights.

When you attach an SAE, miLLM registers a PyTorch forward hook on the specified model layer. During every forward pass, this hook captures the residual stream activations at that layer and runs them through the SAE encoder. If steering is enabled, the hook adds the steering delta — the sum of `strength × decoder_direction` for each configured feature — directly to the residual stream before returning. If probe monitoring is enabled, the hook records the top-K activating features and emits them via WebSocket.

This hook-based architecture means the intervention is fully transparent to the rest of the model. Layers downstream of the hooked layer see a modified residual stream; layers upstream are unaffected. The original model weights are never altered.

### OpenAI API Compatibility

The `/v1/chat/completions` endpoint is compatible with the OpenAI API specification, including streaming via Server-Sent Events. Any client using the OpenAI Python SDK, the OpenAI TypeScript SDK, or direct HTTP against the OpenAI API can be redirected to miLLM by changing the `base_url` to your miLLM instance. Authentication is not required.

This compatibility is what allows Open WebUI, miStudio, and custom research scripts to use miLLM as a drop-in backend. Steering configuration happens through the management API or Admin UI; from the client's perspective, it simply receives (potentially steered) completions.

---

## Key Features

**Model Management** — Download models from HuggingFace with optional 4-bit (Q4) or 8-bit (Q8) quantization via bitsandbytes. Models are cached locally and loaded to GPU on demand. The Admin UI shows VRAM usage so you can plan memory budgets before loading.

**SAE Management** — Download SAEs from any HuggingFace repository (including Gemma Scope, EleutherAI's SAE suite, and custom uploads). The preview browser groups SAE files by layer and width so you can select the right configuration for your model without guessing. Multiple SAEs can be cached; one can be attached at a time.

**Feature Steering** — Configure per-feature steering strengths (positive to amplify, negative to suppress) and apply them during inference. The steering toggle lets you compare steered and unsteered outputs without reconfiguring. Batch add supports entering multiple feature indices at once, with optional per-feature strength overrides.

**Probe Monitoring** — Observe the top-K activating SAE features during inference in real time, with a live bar chart and running statistics (count, mean, min, max, std per feature). Monitoring runs passively — it does not alter the model's output, making it safe to use during production serving.

**Profiles** — Save steering configurations as named profiles that can be activated, deactivated, edited, and exported as JSON. Profiles record the model and SAE they were created with. Exported profiles can be imported on other miLLM instances, making it easy to share or reproduce specific steering configurations across environments.

**Concept-Based Model (CBM) Backend** — An alternative inference mode using linear probe ensembles for interpretable feature detection, available for research scenarios where SAE-based steering is not the right tool.

---

## Hardware Requirements

miLLM requires an NVIDIA GPU. The minimum for useful work is 8 GB VRAM, which supports 1B–2B parameter models at Q4 quantization with small SAEs. For 2B–9B models with wider SAEs (131k features), 16–24 GB VRAM is the practical target. GPU memory is shared between the model weights, SAE weights, and the KV cache — the Admin UI dashboard shows current allocation.

---

## Installation

miLLM ships with two production deployment paths. The documentation covers both in detail, including hardware configuration, environment variables, and the steps for first-run model and SAE download.

The **Docker Compose** path is recommended for single-machine deployments and local research setups. It starts all five services (backend, admin UI, PostgreSQL, Redis, Nginx) with a single command and handles inter-service networking automatically.

→ [Docker Compose Installation Guide](https://onegaishimas.github.io/miLLM/getting-started/install-guide-compose)

The **Kubernetes** path is designed for shared research infrastructure where miLLM runs alongside other services in a cluster. The provided manifests configure all services, persistent volumes for model and SAE storage, and ingress rules for external access. This is the deployment mode used in the hitsai.local research cluster where miLLM runs alongside miStudio and Neuronpedia.

→ [Kubernetes Installation Guide](https://onegaishimas.github.io/miLLM/getting-started/install-guide-k8s)

If you are new to miLLM, the [Introduction](https://onegaishimas.github.io/miLLM/getting-started/introduction) in the manual walks through the architecture, key concepts, and how miLLM fits into a broader interpretability research stack before you begin installation.

---

## Documentation

The full user manual is hosted at **[onegaishimas.github.io/miLLM](https://onegaishimas.github.io/miLLM/)** and covers every feature in depth with screenshots and worked examples.

| Section | Description |
|---------|-------------|
| [Getting Started](https://onegaishimas.github.io/miLLM/getting-started/introduction) | Architecture overview, hardware requirements, installation |
| [Model Management](https://onegaishimas.github.io/miLLM/features/model-management) | Downloading, quantizing, loading, and unloading models |
| [SAE Management](https://onegaishimas.github.io/miLLM/features/sae-management) | Downloading SAEs, attaching to model layers, configuration |
| [Feature Steering](https://onegaishimas.github.io/miLLM/features/feature-steering) | Configuring steering strengths, batch operations, steering mechanics |
| [Probe Monitoring](https://onegaishimas.github.io/miLLM/features/probe-monitoring) | Real-time activation observation, statistics, history |
| [Profiles](https://onegaishimas.github.io/miLLM/features/profiles) | Saving, sharing, and activating steering configurations |
| [OpenAI-Compatible API](https://onegaishimas.github.io/miLLM/api/openai-compatible) | API endpoints, SDK usage, streaming |
| [Management API](https://onegaishimas.github.io/miLLM/api/management-api) | Model/SAE/steering/profile management endpoints |
| [Troubleshooting](https://onegaishimas.github.io/miLLM/troubleshooting) | Common issues, VRAM errors, connection problems |

---

## Relation to miStudio

miLLM and [miStudio](https://github.com/Onegaishimas/miStudio) are complementary tools in the same interpretability research stack.

miStudio is the full research environment: it handles dataset ingestion, SAE training from scratch, activation extraction, and feature labeling. miLLM is the inference server: it takes a trained SAE and deploys it against a running model for steering and monitoring.

The most direct integration point is miStudio's labeling system, which can use miLLM's OpenAI-compatible endpoint as its language model backend. This means SAEs trained in miStudio can be downloaded to miLLM, and the resulting feature activations observed in miLLM's probe monitor can be fed back into miStudio's labeling pipeline — all on local hardware, without external API calls.

---

## License

Apache 2.0 — see [LICENSE](LICENSE) for details.
