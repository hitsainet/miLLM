# Project PRD: miLLM

## Mechanistic Interpretability LLM Server

**Document Version:** 1.5
**Created:** January 30, 2026
**Status:** Draft
**Reference:** BRD v1.0 (January 29, 2026) · BRD-MILLM-CLUSTERS-001 (July 16, 2026) · BRD-MILLM-CIRCUITS-001 (July 20, 2026) · BRD-MILLM-CIRCUITS-002 (July 20, 2026) · BRD-MILLM-PROBES-001 · BRD-04

### Document Revision History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2026-01-30 | Initial project PRD (Features 1–7, from BRD v1.0) |
| 1.1 | 2026-07-16 | Cluster Runtime increment (BRD-MILLM-CLUSTERS-001): Features 8–11 (Cluster Import, Unified MCP, OWUI Cluster Dial, Co-Activation Sensing), FR-8.x–FR-11.x, NFR-1.4, matrix extension; former future stubs renumbered 12–14 |
| 1.2 | 2026-07-20 | Circuit Runtime increment (BRD-MILLM-CIRCUITS-001): Features 12–15 (Multi-SAE Attach & Circuit Serving, Circuit Import + Slice-Fallback + Evidence Ladder, Circuit-Aware OWUI Dial, Circuit Edge Sensing), FR-12.x–FR-15.x, NFR-1.5, matrix extension; retired the former "Multi-SAE Support" future stub (now specified as Feature 12); remaining future stubs renumbered 16+ |
| 1.3 | July 20, 2026 | Circuit Consolidation increment (BRD-MILLM-CIRCUITS-002): Features 16-20 (steering epoch, request-scoped sensing context, single serving derivation, concurrent circuit serving, MCP circuit surface + reachability assurance), FR-16.x-20.x, matrix columns; future stubs renumbered 21/22. |
| 1.4 | 2026-09-25 | Probe Monitor Runtime increment (BRD-MILLM-PROBES-001): Feature 24 — import `mistudio.probe-definition/v1` (file / HF tag / MCP), strict model-identity check, parity gate on the definition's test vectors, prepended read hook independent of SAE attachment, dense and k-sparse SAE probes (private encoder copy), per-request scoring on the sensing lifecycle, verdicts in an `X-miLLM-Probe-Verdicts` header and a final stream chunk, `probe_events` with a privacy-stripped live feed, Probe Monitors page (the old "Probe" page renamed "Feature Monitor"), MCP `millm_probes` (contract v1.6). Specified in the planning workspace (ENH-001). |
| 1.5 | 2026-10-06 | Dataworks Support increment (BRD-04): Features 25–30 — chat scoring, structured output, seed and request validation (unknown fields reported or refused; output-changing fields never ignored); OpenAI-shaped batch API persisted in PostgreSQL and admitted through `_admit()`; stateless probe scoring, per-request SAE activations and the batched-chat probe-skip fix; inline steering and `X-miLLM-Steering`; model lease, `Retry-After` backpressure and per-card GPU memory; embedding pooling, normalisation and `dimensions`. FR-25.x–FR-30.x (47 BRD-04 requirements, coverage table in §6), matrix footnote. |

---

## 1. Project Overview

### Project Name
**miLLM** - Mechanistic Interpretability LLM Server

### Vision Statement
To provide the first practical inference server that bridges mechanistic interpretability research with real-world LLM applications, enabling users to understand and influence model behavior through Sparse Autoencoder (SAE) feature steering.

### Brief Description
miLLM is a lightweight, OpenAI API-compatible inference server designed to run local large language models with integrated SAE steering capabilities. Unlike existing solutions (Ollama, vLLM, llama.cpp), miLLM enables users to hook SAEs into models at runtime, allowing real-time manipulation of model behavior through feature activation adjustments.

### Problem Statement
Current local LLM inference solutions lack support for mechanistic interpretability techniques:
- No existing inference server supports SAE integration
- Behavioral modification requires extensive system prompts consuming context window space
- Fine-tuning for behavioral changes is resource-intensive and inflexible
- There is no practical way to experiment with feature steering in a production-like environment
- Ollama requires specially packaged models rather than raw Hugging Face weights

### Opportunity
miLLM fills a critical gap in the interpretability tooling ecosystem by making SAE steering accessible and practical. This enables:
- Researchers to test interpretability hypotheses in realistic inference scenarios
- Developers to build applications with fine-grained behavioral control
- The broader community to explore the implications of feature steering

### Success Definition
A successful miLLM v1.0 delivers a complete, polished system where users can:
1. Download and run Hugging Face models with quantization support
2. Attach SAEs and adjust feature strengths to influence outputs
3. Monitor feature activations in real-time
4. Use any OpenAI API-compatible client seamlessly
5. Save and manage steering configurations as profiles

---

## 2. Project Goals & Objectives

### Primary Business Goals

| ID | Goal | Success Indicator |
|----|------|-------------------|
| BO-1 | Enable practical SAE steering in local inference | Users successfully steer model outputs using SAE features |
| BO-2 | Reduce dependency on system prompts for behavioral control | Equivalent modifications achieved with <10% context usage |
| BO-3 | Seamless integration with LLM tooling ecosystem | 100% compatibility with OpenAI API clients |
| BO-4 | Support interpretability research | System demonstrates both monitoring and influence scenarios |
| BO-5 | Foundation for miStudio integration | Defined Management API contract for future miStudio communication |

### Secondary Objectives
- Establish miLLM as a reference implementation for SAE-augmented inference
- Create comprehensive documentation for the interpretability community
- Build architecture that supports future multi-SAE, multi-layer configurations
- Provide educational value demonstrating real-world implications of feature steering

### Success Metrics and KPIs

| Metric | Target | Measurement Method |
|--------|--------|-------------------|
| SAE overhead | <15% latency increase vs base model | Benchmark comparison |
| API compatibility | 100% with OpenAI v1 endpoints | Integration tests with Open WebUI, LibreChat |
| Time to first token | <500ms after model loaded | Performance monitoring |
| Request queue handling | 5+ pending requests without drops | Load testing |
| Feature steering accuracy | Observable behavioral changes | Manual verification with known features |

### Timeline Expectations
- **Development Approach:** Standard development cycle (2-4 months)
- **Quality Priority:** Thorough, accurate, and high-quality implementation
- **Release Strategy:** Complete v1.0 with all specified features before launch

---

## 3. Target Users & Stakeholders

### Primary User Persona: Developer/Researcher

**Profile:** Technical users who want to integrate SAE-steered models into applications or research workflows.

**Characteristics:**
- Comfortable with APIs, Docker, and Python environments
- Seeks fine-grained control over model behavior
- Wants to experiment with interpretability techniques in practical settings

**Needs:**
- Reliable inference server with standard API compatibility
- Easy model and SAE management
- Clear documentation and predictable behavior
- Ability to save and reproduce steering configurations

### Secondary User Personas

#### MI Researchers
**Profile:** Academics and researchers exploring mechanistic interpretability.

**Needs:**
- Detailed activation monitoring capabilities
- Ability to test hypotheses about feature effects
- Export/logging of activation data for analysis
- Precise control over which features to observe

#### Power Users/Hobbyists
**Profile:** Enthusiasts running local LLMs who want advanced behavioral control.

**Needs:**
- Easy setup and integration with existing chat interfaces
- Intuitive UI for feature adjustment
- Pre-configured profiles for common use cases
- Clear feedback on what steering is doing

### Key Stakeholders

| Stakeholder | Interest | Success Criteria |
|-------------|----------|------------------|
| miStudio Team | API integration compatibility | Clean Management API contract |
| Interpretability Community | Reference implementation | Well-documented, reproducible results |
| Open Source Community | Extensibility and contribution | Clear architecture, contribution guidelines |

### User Journey Overview

```
Discovery → Installation → Model Setup → SAE Configuration → Steering Experimentation → Profile Management → Production Use
```

1. **Discovery:** User learns about miLLM's SAE steering capabilities
2. **Installation:** Docker pull or pip install, single command startup
3. **Model Setup:** Preview model metadata from HuggingFace, select quantization (FP32/FP16/Q8/Q4/Q2), download
4. **SAE Configuration:** Download SAE, attach to model layer
5. **Steering Experimentation:** Adjust features, observe effects in real-time
6. **Profile Management:** Save successful configurations for reuse
7. **Production Use:** Connect OpenAI-compatible clients, use in workflows

---

## 4. Project Scope

### In Scope (Version 1.0)

#### API Layer
- OpenAI API-compatible endpoints: `/v1/chat/completions`, `/v1/completions`, `/v1/models`, `/v1/embeddings`
- Streaming response support (SSE) for chat applications
- miLLM Management API for configuration and control

#### Model Management
- Hugging Face model downloading and loading (Transformers format)
- Support for safetensors and pytorch formats
- Multiple quantization levels via bitsandbytes (FP32, FP16, Q8, Q4, Q2)
- Local model caching
- Memory requirement estimation with per-quantization size previews
- Rich model preview with HuggingFace metadata (downloads, likes, tags, license, architecture)

#### SAE Management
- SAE downloading from Hugging Face (SAELens format and compatible formats)
- Single SAE attachment to configurable model layer
- Dynamic attachment/detachment without server restart
- Local SAE caching

#### Feature Steering
- Individual feature activation strength adjustment by index
- Simultaneous adjustment of multiple features
- Positive (amplify) and negative (suppress) steering values
- Real-time adjustment without restart

#### Input Monitoring
- Feature activation capture for incoming requests
- Monitoring API/websocket for activation data
- Configurable feature selection for monitoring
- Monitoring on embeddings endpoint

#### Configuration Management
- Named steering configuration profiles
- Profile persistence and loading
- Profile selection via UI and API
- Import/export capability (miStudio-compatible format)

#### Administrative UI
- Model download and management interface
- SAE download and attachment interface
- Feature value adjustment controls
- Real-time activation monitoring display
- Profile management (create, edit, delete, activate)
- Server status and loaded model information

#### Deployment
- Docker containerization with NVIDIA GPU support
- pip install for development environments
- Environment variable configuration (12-factor app)
- Request queuing for single-user scenarios

### Out of Scope (Version 1.0)

| Item | Rationale | Future Consideration |
|------|-----------|---------------------|
| Multi-user authentication | Assumes trusted local network (like Ollama) | v1.1+ |
| Multiple concurrent SAEs | Architectural complexity | v2.0 |
| ~~GGUF model format~~ | ~~Focus on Transformers ecosystem~~ — **DELIVERED 2026-09-08 as Feature 23.** The premise was wrong: GGUF needs a different *loader*, not a different product, and it shares the whole OpenAI-compatible surface. See BRD-MILLM-GGUF-001 | Delivered |
| Kubernetes deployment | Docker sufficient for target users | v1.1+ |
| Feature discovery/analysis tools | Delegated to miStudio | N/A |
| Neuronpedia API integration | Nice-to-have, not core | v1.1+ |
| Direct miStudio push integration | miStudio developing simultaneously | v1.1+ |

### Future Roadmap Considerations
- Multi-layer SAE support with coordinated feature adjustment
- ~~Additional model format support (GGUF, etc.)~~ — GGUF delivered as Feature 23; other formats remain open
- API key authentication for non-local deployments
- Multi-user request management
- Neuronpedia integration for feature browsing
- miStudio bidirectional sync

### Dependencies and Assumptions

**Dependencies:**
- Hugging Face Transformers library for model loading
- SAELens or compatible framework for SAE operations
- bitsandbytes for quantization
- NVIDIA CUDA for GPU acceleration

**Assumptions:**
- Users have NVIDIA GPU with CUDA support
- Users have sufficient VRAM for model + SAE
- Network access to Hugging Face for downloads
- Single-user local deployment model for v1.0

---

## 5. High-Level Requirements

### Core Functional Requirements

Organized by logical workflow (matching UI structure):

#### Models (FR-1.x)
- FR-1.1: Download models from Hugging Face by identifier
- FR-1.2: Load models in Transformers format (safetensors/pytorch)
- FR-1.3: Support multiple quantization levels (FP32, FP16, Q8, Q4, Q2)
- FR-1.4: Cache downloaded models locally
- FR-1.5: Display memory requirements before loading with per-quantization estimates
- FR-1.6: Support extensible model formats via Transformers
- FR-1.7: Preview model metadata from HuggingFace (downloads, likes, tags, license, architecture) before downloading

#### SAEs (FR-2.x)
- FR-2.1: Download SAEs from Hugging Face by identifier
- FR-2.2: Attach single SAE to specified model layer
- FR-2.3: Detach/reattach SAEs without server restart
- FR-2.4: Cache downloaded SAEs locally
- FR-2.5: Support SAELens format and compatible formats
- FR-2.6: Architecture supports future multi-SAE configurations

#### Steering (FR-3.x)
- FR-3.1: Adjust individual feature activation strengths by index
- FR-3.2: Support simultaneous multiple feature adjustment
- FR-3.3: Apply steering to model output generation
- FR-3.4: Allow adjustments without server restart
- FR-3.5: Support positive (amplify) and negative (suppress) values

#### Profiles (FR-6.x)
- FR-6.1: Persist steering configurations as named profiles
- FR-6.2: Allow profile selection via admin UI
- FR-6.3: Allow profile selection via API parameter
- FR-6.4: Support import/export for miStudio compatibility
- FR-6.5: Follow documented profile format contract

#### Monitor (FR-4.x)
- FR-4.1: Capture feature activations for incoming requests
- FR-4.2: Expose activation data via monitoring API/websocket
- FR-4.3: Support monitoring on embeddings endpoint
- FR-4.4: Allow configurable feature selection for monitoring

#### API Compatibility (FR-5.x)
- FR-5.1: Implement `/v1/chat/completions` per OpenAI spec
- FR-5.2: Implement `/v1/completions` per OpenAI spec
- FR-5.3: Implement `/v1/models` endpoint
- FR-5.4: Implement `/v1/embeddings` endpoint
- FR-5.5: Support streaming responses (SSE)
- FR-5.6: Compatible with OpenAI API clients (Open WebUI, LibreChat, etc.)

#### Administrative UI (FR-7.x)
- FR-7.1: Model download and selection interface
- FR-7.2: SAE download and attachment interface
- FR-7.3: Feature value adjustment interface
- FR-7.4: Real-time activation monitoring display
- FR-7.5: Configurable feature monitoring selection
- FR-7.6: Profile management interface
- FR-7.7: Server status display

#### Cluster Import (FR-8.x) — Increment: Cluster Runtime
- FR-8.1: Import `mistudio.cluster-definition/v1` documents (single) and `mistudio.cluster-bundle/v1` documents (multi) from JSON with strict schema validation
- FR-8.2: Evaluate import compatibility against the attached model+SAE (bind / warn-bind / block / unbound) and report outcomes honestly per item
- FR-8.3: Materialize imported definitions as cluster-typed steering profiles preserving name, narrative, members with tuned strengths/signs, budget metadata (incl. intensity λ), and provenance
- FR-8.4: Activate an imported cluster so ALL members steer together at their stored strengths (λ-scaled, clamped to the steering range) with no manual tuning
- FR-8.5: Browse and import public cluster packs from Hugging Face anonymously (tag convention `mistudio-cluster-definition`), recording hub provenance
- FR-8.6: Treat imported definitions strictly as data (size/count caps; no paths, no credentials, no execution)
- FR-8.7: Re-export an imported cluster as a lossless `mistudio.cluster-definition/v1` document
- FR-8.8: Dedicated Clusters page in the Admin UI (list, import dialog with file/paste/HF tabs, activate, intensity, narrative display)

#### Unified MCP (FR-9.x) — Increment: Cluster Runtime
- FR-9.1: A single unified MCP server (evolved from the miStudio server) exposes miLLM tool categories gated by per-product health checks
- FR-9.2: miLLM tools cover model/SAE status, profile list/activate, cluster import (file + hub), intensity control, and sensing readout
- FR-9.3: miLLM publishes the management-API contract the MCP server consumes (`docs/mcp-contract.md`) and an `active_profile` block in detailed health
- FR-9.4: A single-product deployment presents a coherent, self-describing tool set (absent product's tools return structured "unavailable")

#### OWUI Cluster Dial (FR-10.x) — Increment: Cluster Runtime
- FR-10.1: Accept a per-request `steering_intensity` extension (numeric λ or symbolic off/min/max) on `/v1/chat/completions`, resolved server-side against the active cluster's intensity range
- FR-10.2: Per-request intensity is isolated (apply/restore within the request boundary) and concurrency-safe
- FR-10.3: Ship an Open WebUI Filter Function (in-repo artifact) exposing a per-user dial valve that injects the extension field
- FR-10.4: A user can compare identical prompts at dial off/min/max within one chat session

#### Co-Activation Sensing (FR-11.x) — Increment: Cluster Runtime
- FR-11.1: Detect, per forward pass, moments when a designated cluster's members co-fire (threshold ε·max_activation per member; quorum min_k), opt-in per cluster and off by default
- FR-11.2: Each event records the alone-vs-within-larger-set distinction (best-effort v1: ambient fired count when full-width monitoring is active)
- FR-11.3: Each event captures a configurable window of token context (±K tokens, decoded off the hot path; K=0 disables text capture)
- FR-11.4: Events persist with bounded retention (per-cluster cap + age pruning) and are retrievable via API, UI, and WebSocket
- FR-11.5: Sensing overhead is observable (`sensing_overhead_ms`) and bounded; sensing-armed requests route serial (never approximated on the batching path)

#### Multi-SAE Attach & Circuit Serving (FR-12.x) — Increment: Circuit Runtime
- FR-12.1: Attach multiple SAEs simultaneously, keyed by `(sae_id, layer)`, loading only the SAEs an imported circuit references (referenced-only loading)
- FR-12.2: Serve a circuit live so every member feature is steered through ITS OWN layer's SAE decoder — a feature on layer L is never steered through another layer's basis
- FR-12.3: Apply the circuit's per-layer strength budgets under a single global intensity (λ), reusing the validated per-layer allocation (`freq-budget/sim-alloc/per-layer@1`); joint cross-layer calibration is explicitly deferred
- FR-12.4: Reject at submit/activation time (422) any member whose layer has no attached SAE (`SAE_SET_INCOMPLETE`), listing the offenders — never silently steer through a wrong-layer SAE
- FR-12.5: Attach the steering weight set in fp16 within a documented VRAM envelope (measured: ~64 MB/SAE fp16; the two-SAE case is 128 MB, within the <200 MB close-out target)
- FR-12.6: Surface cross-layer over-steering hazards (compounding/cancellation) at activation, quantified from a validated effect size where present and labeled `heuristic` otherwise — detection, not auto-correction
- FR-12.7: Report the attached-SAE set (plural attachment status) wherever attachment state is surfaced (API, MCP status, Admin UI)

#### Circuit Import, Slice-Fallback & Evidence Ladder (FR-13.x) — Increment: Circuit Runtime
- FR-13.1: Import `mistudio.circuit-definition/v1` documents from JSON with strict schema validation, rejecting unknown kinds and incompatible schema major versions
- FR-13.2: Evaluate compatibility per referenced SAE (bind / warn-bind / block / unbound) and treat a circuit as fully serveable only when all referenced SAEs bind
- FR-13.3: On an incomplete/single-SAE deployment, fall back to the circuit's per-layer `mistudio.cluster-definition/v1` slice (consumed unchanged through the existing cluster import path) rather than serving any member through a mismatched SAE
- FR-13.4: Surface each circuit's and edge's EvidenceRung verbatim from the ladder (`associated` / `suggested (attribution-supported)` / `causally validated (edge)` / `faithfulness-tested (circuit)`) wherever steering state is shown; the circuit rung is the MIN over its edges
- FR-13.5: Never describe rung-below-2 steering as "causal"; require an explicit unvalidated acknowledgement to activate a circuit whose rung is below 2
- FR-13.6: Treat imported circuit definitions strictly as data (size/count caps; no paths, no credentials, no execution) — reusing the cluster-import posture
- FR-13.7: Circuits surfaced in the Admin UI (list with rung/layers/edge count, import dialog, activation with the unvalidated-rung gate, slice-fallback disclosure)

#### Circuit-Aware OWUI Dial (FR-14.x) — Increment: Circuit Runtime
- FR-14.1: Extend the per-request `steering_intensity` extension so it dials a whole active circuit (all layers scale together under one λ) off/min/max or numeric
- FR-14.2: Per-request circuit intensity is isolated (apply/restore within the request boundary, incl. client disconnect) and concurrency-safe
- FR-14.3: The Open WebUI Filter Function surfaces the active circuit's identity and evidence rung alongside the dial (a rung<2 circuit is visibly marked unvalidated)
- FR-14.4: A user can compare identical prompts at circuit influence off/min/max within one chat session

#### Circuit Edge Sensing (FR-15.x) — Increment: Circuit Runtime
- FR-15.1: Detect, per forward pass, circuit EDGE co-activation — an upstream member firing followed by its downstream partner firing within a configurable token-lag window — opt-in per circuit and off by default
- FR-15.2: Each edge event records the alone-vs-within-larger-set distinction and the upstream/downstream member activations
- FR-15.3: Each edge event captures a configurable window of token context (±K tokens, decoded off the hot path)
- FR-15.4: Edge events persist with bounded retention and are retrievable via API, UI, and WebSocket, carrying the edge's evidence rung
- FR-15.5: New additive `/api/circuits/*` endpoints and a `millm_circuits` MCP tool category (import, activate/deactivate, status, list, edge-sensing readout), tracked in `docs/mcp-contract.md` (v1.1, additive-only)

#### Steering Epoch (FR-16.x) — Increment: Circuit Consolidation
- FR-16.1: `AttachedSAEState` SHALL carry a monotonic `steering_epoch`, bumped under the attachment lock by every authoritative writer of live steering state.
- FR-16.2: A per-request steering override SHALL capture the epoch at save time and SHALL SKIP its restore when the epoch has advanced — last authoritative writer wins.
- FR-16.3: A skipped restore SHALL be logged with both epochs, so supersession is observable rather than silent.
- FR-16.4: `PUT /api/circuits/active/intensity` SHALL NOT report `"reapplied": true` for a change an in-flight request reverted; the same guarantee applies to the Feature 10 profile path.

#### Request-Scoped Sensing Context (FR-17.x) — Increment: Circuit Consolidation
- FR-17.1: Absolute token position SHALL be owned by ONE request-scoped counter, replacing the N per-SAE counters whose divergence caused three of Feature 15's eight criticals.
- FR-17.2: Each `(request, circuit)` pair SHALL have its OWN fire ring; rings SHALL NOT be shared across circuits, since an `edge_key` present in two circuits would otherwise let one circuit's upstream fire match another's downstream and fabricate an observation of an edge that fired in neither.
- FR-17.3: The per-request event budget SHALL be attributed per circuit so one busy circuit cannot exhaust another's observation budget.
- FR-17.4: Ring lifetime (creation, pruning, release) SHALL be owned by the context, not by whichever hook happens to run last.
- FR-17.5: The edge machinery SHALL live in its own module, exercisable without constructing a `LoadedSAE`.
- FR-17.6: Characterization tests SHALL pin current matcher behaviour BEFORE any code moves, and mutation testing SHALL be applied to the result.

#### Single Circuit-Serving Derivation (FR-18.x) — Increment: Circuit Consolidation
- FR-18.1: Serving a circuit SHALL have exactly one implementation, consumed by activation, intensity changes and the per-request dial.
- FR-18.2: No caller SHALL construct a service by bypassing its constructor in order to reach steering; a half-constructed service whose failure mode is a swallowed `AttributeError` and a silently unsteered response SHALL NOT be reachable.
- FR-18.3: A circuit's claim set (the layers its serving members reach) SHALL be computed by that same derivation, so activation and contention agree by construction.

#### Concurrent Circuit Serving (FR-19.x) — Increment: Circuit Consolidation
- FR-19.1: A layer SHALL be claimed by at most one active circuit; circuits with disjoint claim sets SHALL serve concurrently.
- FR-19.2: Activation whose claim set overlaps an incumbent's SHALL be refused with `CIRCUIT_LAYER_CONTENTION` (200 + `success:false`), naming the incumbent circuit and the contended layers.
- FR-19.3: An explicit `allow_layer_overlap` acknowledgement SHALL permit additive composition; while any layer is composed, `X-miLLM-Circuit-Rung` SHALL be OMITTED, because no single circuit's evidence describes the response. The refusal preceding any override SHALL carry the MEASURED hazard (close-out: two steered layers at individually-harmless strength destroyed generation) together with its "one model, one fixture" caveat, and every override SHALL be echoed, logged and surfaced — an override taken blind is a footgun; one taken informed is a research decision.
- FR-19.4: Two active circuits naming the same `(layer, feature_idx)` SHALL be refused unconditionally, with no override, since the merge would serve a strength belonging to neither author.
- FR-19.5: Deactivation SHALL release only that circuit's own claims and steering keys, never a co-tenant's.
- FR-19.6: The capability SHALL ship behind `CIRCUIT_ALLOW_CONCURRENT` (default false for exactly ONE release, with the flip to true recorded as a DATED commitment) and a tested downgrade, since the first concurrent activation is a one-way door in deployed data. While the flag is false a second activation SHALL be refused LOUDLY, naming configuration as the reason; it SHALL NOT fall back to the silent single-active disarm this feature replaces.

#### MCP Circuit Surface & Reachability (FR-20.x) — Increment: Circuit Consolidation
- FR-20.1: A `millm_circuits` MCP category SHALL expose every circuit capability reachable by REST — list, import, activate, deactivate, export, set intensity, status, and edge-sensing status/events/enable/disable.
- FR-20.2: Every circuit- and edge-bearing MCP response SHALL carry `rung` and server-rendered `rung_language` verbatim; the build-failing copy audit SHALL extend to the MCP modules and their tool descriptions.
- FR-20.3: No capability SHALL be accepted as shipped without a test that FAILS when its user- or agent-facing wiring is removed; a test asserting only that an entry point exists SHALL NOT satisfy this.
- FR-20.4: Documentation status marks SHALL distinguish "endpoint exists" from "reachable by a user or agent".
- FR-20.5: `docs/mcp-contract.md` SHALL move to v1.2, additive-only.

#### Chat Scoring, Structured Output, Seed & Request Validation (FR-25.x) — Increment: Dataworks Support

- **FR-25.1:** Every `/v1` endpoint SHALL report the request fields it did not use, top-level and inside `messages`, in an `X-miLLM-Ignored-Fields` response header and a logged warning; a field SHALL never be dropped without a trace. (R-04.1)
- **FR-25.2:** A request sending `X-miLLM-Strict: true` SHALL get `400` instead of a warning when any field would be ignored, naming every such field. (R-04.2)
- **FR-25.3:** A single named list of output-changing fields — at least `response_format`, `seed`, `logprobs`, `top_logprobs`, `allowed_token_ids`, `n`, `dimensions`, `tools`, `tool_choice`, `logit_bias` and `steering` — SHALL never be ignored, strict or not; each SHALL be honoured or refused with `400`, and a test SHALL assert every entry on every endpoint. (R-04.3)
- **FR-25.4:** `n` on `/v1/completions` SHALL either produce `n` choices per prompt, indexed as OpenAI indexes them, or return `400` for `n > 1`. (R-04.4)
- **FR-25.5:** `/v1/chat/completions` SHALL accept `logprobs`, `top_logprobs` (0–20) and `allowed_token_ids`, render the chat template with the generation prompt, and score through the same next-token scoring path `/v1/completions` uses, without adding the rendered prompt's special tokens a second time. (R-04.6)
- **FR-25.6:** Chat scoring SHALL carry the completion-scoring limits (`max_tokens` 1, `n` 1, no streaming, the temperature floor) and SHALL refuse a GGUF model before any auto-load. (R-04.7)
- **FR-25.7:** Chat scoring SHALL be unsteered — every attached sparse autoencoder (SAE) suppressed — and probes and sensing SHALL record nothing for it. (R-04.8)
- **FR-25.8:** The chat scoring response SHALL use OpenAI's chat logprobs shape (`choices[].logprobs.content[]` with `token`, `logprob`, `bytes`, `top_logprobs[]`), and `return_tokens_as_token_ids` SHALL behave as on completions. (R-04.9)
- **FR-25.9:** A scoring request carrying `extra_messages` SHALL score each conversation and return one choice per conversation, `index` in input order. (R-04.10)
- **FR-25.10:** `response_format` on `/v1/chat/completions` SHALL support `json_object` and `json_schema` on the transformers engine through constrained decoding, so the output parses and validates. (R-04.11)
- **FR-25.11:** Where structured output is unsupported (a GGUF model unless extended, a schema feature outside the supported subset, the continuous batching manager (CBM) path), the request SHALL return `400` naming `response_format` and the reason — before any auto-load when decidable from the model row. (R-04.12)
- **FR-25.12:** A constrained generation stopped by `max_tokens` SHALL report `finish_reason: "length"` and never return truncated JSON as complete; the response SHALL carry `X-miLLM-Constrained` naming the format applied. (R-04.13)
- **FR-25.13:** `seed` SHALL be accepted on chat and text completions and applied to sampling; on the serial transformers path the same seed, request, model and batch shape SHALL give identical output, echoed in `X-miLLM-Seed`, with a `system_fingerprint` naming model, revision, precision and engine. (R-04.14)
- **FR-25.14:** Where a seed cannot promise identical output (batched rows, deterministic per batch shape only; llama.cpp), the response SHALL say so, and on llama.cpp the seed SHALL be forwarded to the engine or the request refused. (R-04.15)

#### Batch API (FR-26.x) — Increment: Dataworks Support

- **FR-26.1:** The system SHALL provide an OpenAI-shaped batch API: `POST /v1/files` (a JSON Lines (JSONL) file, `purpose: "batch"`) and `POST /v1/batches` (`input_file_id`, `endpoint`, `completion_window`), each line holding `custom_id`, `method`, `url` and `body`, for `/v1/chat/completions`, `/v1/completions`, `/v1/embeddings` and `/api/probes/score`. (R-04.16)
- **FR-26.2:** Every line SHALL be validated against its endpoint's schema, under strict mode, before any row runs; invalid lines SHALL go to the error file with line number and reason, and a file with no valid line SHALL be refused. (R-04.17)
- **FR-26.3:** Batches SHALL be persisted in PostgreSQL with OpenAI's status values (`validating` … `expired`) and SHALL survive a pod restart, resuming from the first row without a recorded result; no recorded row SHALL run twice. (R-04.18)
- **FR-26.4:** Batch rows SHALL reach the model only through the single admission path `_admit()`, taking one slot per chunk and releasing it between chunks so interactive requests interleave, and SHALL NOT count against `MAX_PENDING_REQUESTS`. (R-04.19)
- **FR-26.5:** Scoring and embedding rows MAY be packed into one padded forward pass, on by default, with `pack: false` giving single-row semantics; the packed-versus-single difference SHALL be measured on the reference model and stated in the API reference. (R-04.20)
- **FR-26.6:** The system SHALL report request counts on `GET /v1/batches/{id}`, emit progress on a Socket.IO event, list batches, return file content, and cancel at the next row boundary keeping completed rows. (R-04.21)
- **FR-26.7:** A batch SHALL name one model, take the model lease for its whole run, refuse to start if another holder has the lease, and never load or swap a model. (R-04.22)
- **FR-26.8:** Rows per batch and bytes per file SHALL be configurable with stated defaults; a file over a limit SHALL be refused at upload, never truncated. (R-04.23)

#### Probe Scoring, Per-Request Activations & Probe-Path Fixes (FR-27.x) — Increment: Dataworks Support

- **FR-27.1:** Chat and text completions SHALL accept `return_sae_activations: {sae_id?, features?, top_k, positions}` (`positions`: `last`, `prompt`, `completion`, `all` or an index range) and return this request's activations only, keyed by position, under a `millm` extension object. (R-04.24)
- **FR-27.2:** The request SHALL be refused, naming the SAE, unless a matching SAE is attached; the response SHALL state whether activations were read before or after steering; positions × `top_k` over a configured cap SHALL return `400`. (R-04.25)
- **FR-27.3:** `return_sae_activations` SHALL work in scoring mode. (R-04.26)
- **FR-27.4:** `POST /api/probes/score` SHALL take `{probe_ids?, inputs, windows?}` (`token_ids` authoritative over `messages` or `text`) and return per input, probe and window the score, threshold, verdict, evidence rung and provisional flag, generating nothing, writing no probe event and changing no armed state. (R-04.27)
- **FR-27.5:** Stateless scoring SHALL work on any imported probe, armed or not, reusing the parity forward and the armed-probe construction parity uses, applying the same window bars and length bands as live scoring, and refusing a probe that fails the identity check (naming the mismatch) or a GGUF model. (R-04.28)
- **FR-27.6:** Stateless scoring SHALL take a request slot through `_admit()`, with probes on one layer sharing a forward pass per input; the existing parity route SHALL be brought under `_admit()` too. (R-04.29)
- **FR-27.7:** Offline scoring SHALL need no global arming and SHALL force no other traffic onto the serial path. (R-04.30)
- **FR-27.8:** `_create_batched_chat_completion` SHALL open a probe context and, until it can score each row, mark the request not scored with reason `batched_request`, so the verdict header and probe status say why no verdict exists. (R-04.46)
- **FR-27.9:** A test SHALL discover every generation entry point from the service (serial, streaming and batched chat, text completion, the CBM paths, the llama.cpp paths) and fail if any reaches generation with a probe armed and neither a probe context nor a recorded not-scored reason; a hand-kept list SHALL NOT satisfy this. (R-04.47)

#### Inline Steering & Steering-State Header (FR-28.x) — Increment: Dataworks Support

- **FR-28.1:** Chat and text completions SHALL accept `steering: {sae_id?, features: [{index, strength}]}`, applied to this request only inside the admission slot and restored afterwards on the existing per-request apply/restore lifecycle; no saved profile SHALL be created, and an unattached SAE SHALL be refused. (R-04.31)
- **FR-28.2:** `steering` and `profile` SHALL be mutually exclusive, and `steering: {"features": []}` SHALL mean explicitly unsteered, suppressing every attached SAE. (R-04.32)
- **FR-28.3:** Every generation response SHALL state its steering state in `X-miLLM-Steering` (none; profile with intensity; inline with feature count and hash; circuit) — after generation on non-streaming responses, in a final Server-Sent Events (SSE) chunk on streaming ones, and in the body of batch output lines. (R-04.33)
- **FR-28.4:** `/v1/completions` SHALL gain `profile`, `steering_intensity` and `steering`, so a text completion can choose or refuse the active profile. (R-04.34)

#### Model Lease, Backpressure & GPU Visibility (FR-29.x) — Increment: Dataworks Support

- **FR-29.1:** `POST /api/models/{id}/lease` SHALL take `{holder, ttl_seconds, reason}` and return a lease ID; one lease SHALL exist at a time, renewable and releasable by its holder, reported by `GET`, and expiring on its own at its time to live (TTL). (R-04.38)
- **FR-29.2:** While a model is leased, a load, unload or swap by anyone else — including every `/v1` auto-load and `POST /api/models/{id}/load` / `/unload` — SHALL be refused with `409 MODEL_LEASED`, naming holder and expiry. (R-04.39)
- **FR-29.3:** Requests and model operations carrying the holder's `X-miLLM-Lease: <id>` SHALL proceed. (R-04.40)
- **FR-29.4:** `X-miLLM-Load-Policy: refuse` SHALL turn an auto-load of a non-resident model into `409`; the default SHALL stay auto-load. (R-04.41)
- **FR-29.5:** Lease state SHALL appear in `/api/health/detailed` and on the Admin UI's model page. (R-04.42)
- **FR-29.6:** Every `503` (`QUEUE_FULL`, `MODEL_BUSY`, `MODEL_LOADING`, `MODEL_NOT_LOADED`, `INSUFFICIENT_MEMORY`) SHALL carry `Retry-After` in seconds, with the OpenAI error envelope and codes unchanged. (R-04.43)
- **FR-29.7:** `/api/health/detailed` SHALL add the in-flight count, the batch backlog in rows and an estimated wait beside the existing queue depth, documented as a stable contract. (R-04.44)
- **FR-29.8:** A REST endpoint SHALL return, per card: index, universally unique identifier (UUID), name, total, used and free memory in MiB, and miLLM's own allocated and reserved memory on that card. (R-04.45)

#### Embedding Options (FR-30.x) — Increment: Dataworks Support

- **FR-30.1:** `dimensions` on `/v1/embeddings` SHALL be honoured only for a model whose metadata declares support for truncated embeddings, and otherwise SHALL return `400`. (R-04.5)
- **FR-30.2:** `/v1/embeddings` SHALL accept `pooling` (`mean` default, `last`, `cls`) and `normalize` (default false). (R-04.35)
- **FR-30.3:** An input longer than the model's limit SHALL return `400` naming its index and SHALL never be truncated silently; inputs per request SHALL be capped with a stated default. (R-04.36)
- **FR-30.4:** The embeddings route's comments SHALL match its code (no GGUF guard is described that the route does not contain). (R-04.37)

### Non-Functional Requirements

#### Probe Monitor Runtime (FR-24.x) — Increment: Probe Monitors

- **FR-24.1:** The system SHALL import a `mistudio.probe-definition/v1` from a file, from HF (tag `mistudio-probe-definition`, manifest-first) or through MCP, refusing unknown kinds and payloads over 2 MB.
- **FR-24.2:** The system SHALL browse and import probe definitions from HF with the cluster hub's cache and circuit breaker.
- **FR-24.3:** The system SHALL refuse to arm a probe whose model identity (HF id, d_model, layer count, chat-template hash, revision) differs from the loaded model, naming every mismatch; an unverifiable revision SHALL be a recorded warning, not a pass.
- **FR-24.4:** The system SHALL run a definition's test vectors through the live model before arming and SHALL refuse to arm when any score differs beyond tolerance, reporting tokenization drift separately.
- **FR-24.5:** The system SHALL read probe activations through a read-only, prepended forward hook that does not depend on SAE attachment and never modifies the residual.
- **FR-24.6:** The system SHALL score each request on the probe's scope (all / prompt / response) with its combining rule, with one device-to-host copy per forward pass.
- **FR-24.7:** The system SHALL return verdicts in an `X-miLLM-Probe-Verdicts` header on non-streaming responses and in a final chunk before `[DONE]` on streaming responses, only when a probe is armed.
- **FR-24.8:** The system SHALL record a `probe_events` row per armed probe per request and emit `probe:event` without prompt or context text.
- **FR-24.9:** The system SHALL report armed probes, paused reasons and overhead; a probe SHALL never be silently quiet.
- **FR-24.10:** The system SHALL force the serial path while probes are armed, mark batched / n>1 / speculative requests not scored with a reason, refuse arming on llama.cpp, and disarm on model change.
- **FR-24.11:** The system SHALL show each probe's evidence rung with miStudio's language verbatim and SHALL require an operator acknowledgement to arm below rung 2.
- **FR-24.12:** The Admin UI SHALL provide a Probe Monitors page and SHALL rename the "Probe" page to "Feature Monitor" without changing its route.
- **FR-24.13:** The MCP contract SHALL add a `millm_probes` category (v1.6), co-released with miStudio's registry.
- **FR-24.15:** The system SHALL run k-sparse SAE probes using a private copy of the encoder columns for the probe's features, loaded from an SAE downloaded in miLLM and verified against the definition (repo, revision, weights hash, architecture, dimensions); the SAE SHALL NOT be attached and SHALL NOT steer.
- **FR-24.14:** Probe scoring SHALL add under `PROBE_MAX_OVERHEAD_MS` (5 ms) per request at 4k-token contexts with two armed probes.

#### Performance (NFR-1.x)
- NFR-1.1: SAE hook overhead <15% vs base model latency
- NFR-1.2: Graceful request queuing for 5+ pending requests
- NFR-1.3: Time to first token <500ms after model loaded
- NFR-1.4: Sensing (armed) adds no user-perceivable latency — overhead observable and warned above 5 ms/request (Increment: Cluster Runtime)
- NFR-1.5: Multi-SAE attach + edge sensing keep the OpenAI-compatible path within the CBM latency budget; attached-SAE VRAM scales linearly (~64 MB/SAE fp16) and only referenced SAEs are loaded (Increment: Circuit Runtime)

#### Reliability (NFR-2.x)
- NFR-2.1: Configuration errors fail fast with clear messages
- NFR-2.2: Runtime errors (OOM) degrade gracefully when possible
- NFR-2.3: Structured logging with sufficient debug context

#### Deployability (NFR-3.x)
- NFR-3.1: Single `docker-compose up` deployment
- NFR-3.2: `pip install` + `python run` for development
- NFR-3.3: NVIDIA GPU passthrough in Docker
- NFR-3.4: Environment variable configuration

#### Security (NFR-4.x)
- NFR-4.1: Assumes trusted local network (no auth in v1)
- NFR-4.2: Architecture supports future API key authentication
- NFR-4.3: UI abstracts system paths and sensitive details

### Integration Requirements

#### Hugging Face Integration
- Download models via huggingface_hub library
- Support private model access via HF_TOKEN environment variable
- Configurable local cache directory

#### OpenAI API Client Compatibility
- Configurable as backend for any OpenAI API client
- Standard chat functionality works without client modification
- Tested with Open WebUI and LibreChat

#### miStudio Integration
- Profile export format: JSON schema (model, SAE, layer, features)
- Profile import with validation
- Management API designed for miStudio direct integration
- Increment (Cluster Runtime): `mistudio.cluster-definition/v1` + bundle as the sole cluster interchange (kind-keyed, frozen v1 schema; vendored copy + sync test); unified MCP server contract; Hugging Face tag convention (consume-only)
- Increment (Circuit Runtime): `mistudio.circuit-definition/v1` (new kind; per-layer SAE refs, typed edges, per-layer budgets, evidence rungs) consumed live, plus its per-layer `mistudio.cluster-definition/v1` slice projection as the single-SAE fallback; the EvidenceRung ladder vocabulary carried verbatim; `docs/mcp-contract.md` advanced to v1.1 (additive `millm_circuits` category)

---

## 6. Feature Breakdown

Features organized by UI workflow tabs, with requirements matrix:

### Core Features (MVP/Essential)

#### Feature 1: Model Management
**User Value:** Users can easily download and manage LLMs from Hugging Face with appropriate quantization for their hardware.

**UI Tab:** Models

**Requirements Covered:** FR-1.1 through FR-1.6

**Key Capabilities:**
- HuggingFace repository search/download
- Quantization selection (FP32, FP16, Q8, Q4, Q2)
- Model loading/unloading
- Memory estimation display with per-quantization breakdown
- Rich model preview with HuggingFace metadata and download-from-preview
- Local cache management

**Dependencies:** None (foundational)

---

#### Feature 2: SAE Management
**User Value:** Users can download SAEs and attach them to loaded models to enable feature steering.

**UI Tab:** SAEs

**Requirements Covered:** FR-2.1 through FR-2.6

**Key Capabilities:**
- SAE repository download
- Layer selection for attachment
- Link SAE to specific model
- Attach/detach operations
- SAE metadata display

**Dependencies:** Feature 1 (Model Management)

---

#### Feature 3: Feature Steering
**User Value:** Users can adjust feature activation strengths to influence model behavior in real-time.

**UI Tab:** Steering

**Requirements Covered:** FR-3.1 through FR-3.5

**Key Capabilities:**
- Feature selection by index
- Strength adjustment slider (-10 to +10)
- Multiple feature simultaneous adjustment
- Live activation display
- Steering enable/disable toggle

**Dependencies:** Feature 2 (SAE Management)

---

#### Feature 4: OpenAI API Compatibility
**User Value:** Users can connect any OpenAI API-compatible client to miLLM without modification.

**UI Tab:** N/A (Backend service)

**Requirements Covered:** FR-5.1 through FR-5.6

**Key Capabilities:**
- `/v1/chat/completions` endpoint
- `/v1/completions` endpoint
- `/v1/models` endpoint
- `/v1/embeddings` endpoint
- SSE streaming support

**Dependencies:** Feature 1 (Model Management)

---

#### Feature 5: Administrative UI
**User Value:** Users have a visual interface to manage all aspects of miLLM without CLI commands.

**UI Tab:** All tabs

**Requirements Covered:** FR-7.1 through FR-7.7

**Key Capabilities:**
- Unified navigation (Models, SAEs, Steering, Profiles, Monitor)
- Status bar with system metrics
- Consistent visual design
- Responsive interactions

**Dependencies:** All other features (UI layer)

---

### Secondary Features (Important)

#### Feature 6: Profile Management
**User Value:** Users can save steering configurations and quickly switch between them.

**UI Tab:** Profiles

**Requirements Covered:** FR-6.1 through FR-6.5

**Key Capabilities:**
- Create/edit/delete profiles
- Activate profile with single click
- API-based profile selection
- JSON import/export
- Profile format documentation

**Dependencies:** Feature 3 (Feature Steering)

---

#### Feature 7: Feature Monitoring
**User Value:** Users can observe feature activations in real-time to understand model behavior.

**UI Tab:** Monitor

**Requirements Covered:** FR-4.1 through FR-4.4

**Key Capabilities:**
- Real-time activation display
- Configurable feature selection
- Historical activation log
- Statistics (min/max/avg)
- Pause/resume monitoring

**Dependencies:** Feature 2 (SAE Management)

---

### Increment: Cluster Runtime (BRD-MILLM-CLUSTERS-001)

#### Feature 8: Cluster Import
**User Value:** Clusters tuned and validated in miStudio (or published to Hugging Face by the community) run in miLLM with zero manual strength entry — import, activate, steer.

**UI Tab:** Clusters (new)

**Requirements Covered:** FR-8.1 through FR-8.8

**Key Capabilities:**
- `mistudio.cluster-definition/v1` + bundle import (file/paste/HF browse)
- Compatibility matrix vs attached SAE (bind/warn/block/unbound)
- Cluster-typed profiles: members→steering, narrative, budget+λ, provenance retained losslessly
- Anonymous Hugging Face pack browse/import (consume-only)
- Dedicated Clusters Admin-UI page

**Dependencies:** Feature 6 (Profile Management), Feature 3 (Feature Steering)

---

#### Feature 9: Unified MCP
**User Value:** Agents work across authoring (miStudio) and serving (miLLM) through ONE MCP endpoint that adapts to whichever back ends are present.

**UI Tab:** None (agent surface)

**Requirements Covered:** FR-9.1 through FR-9.4

**Key Capabilities:**
- miLLM tool categories on the evolved miStudio MCP server (cross-repo)
- Per-product health gating with structured degradation
- Published miLLM management-API contract (`docs/mcp-contract.md`)
- `active_profile` block added to detailed health

**Dependencies:** Feature 8 (endpoints), Features 10/11 (tools); miStudio MCP server (cross-repo)

---

#### Feature 10: OWUI Cluster Dial
**User Value:** End users feel a cluster's influence live in real chat — off/min/max on identical prompts — without leaving Open WebUI.

**UI Tab:** None (OWUI-side Filter Function + OpenAI-API extension)

**Requirements Covered:** FR-10.1 through FR-10.4

**Key Capabilities:**
- Per-request `steering_intensity` extension (numeric or off/min/max)
- Server-side λ resolution from the cluster's intensity range; request-scoped apply/restore
- In-repo Open WebUI Filter Function with per-user dial valve

**Dependencies:** Feature 8 (imported clusters + intensity semantics)

---

#### Feature 11: Co-Activation Sensing
**User Value:** Close the authoring loop — observe when a cluster's members actually fire together in production traffic, with token context, to learn what patterns to monitor for.

**UI Tab:** Clusters (sensing panel)

**Requirements Covered:** FR-11.1 through FR-11.5

**Key Capabilities:**
- Per-forward-pass member-only detection (ε·max_activation thresholds, min_k quorum)
- Alone-vs-within side channel (best-effort v1)
- ±K token context per event (configurable, off-hot-path decode)
- Bounded persistence (per-cluster cap + age prune) + API/UI/WS readout
- Serial-only, opt-in, observable overhead

**Dependencies:** Feature 8 (cluster profiles), Feature 7 (monitoring hook path)

---

### Increment: Circuit Runtime (BRD-MILLM-CIRCUITS-001)

#### Feature 12: Multi-SAE Attach & Circuit Serving
**User Value:** A cross-layer circuit discovered and validated in miStudio runs live in miLLM — every member steered through its own layer's SAE, at its tuned per-layer budget, under one dial — instead of being trapped as a single-SAE approximation.

**UI Tab:** Circuits (attachment status shows the plural SAE set)

**Requirements Covered:** FR-12.1 through FR-12.7

**Key Capabilities:**
- Multi-SAE attachment keyed by `(sae_id, layer)`; only referenced SAEs loaded (fp16, ~64 MB/SAE; two-SAE = 128 MB, within the <200 MB envelope — measured)
- One hook per referenced SAE/layer, each bound to its own decoder — a feature on layer L is never steered through another layer's basis
- Per-layer strength budgets under a single global λ (`freq-budget/sim-alloc/per-layer@1`); joint calibration deferred
- Submit/activation-time rejection (422, `SAE_SET_INCOMPLETE`) when a member's layer has no attached SAE — never a silent wrong-basis path
- Cross-layer over-steering hazards (compounding/cancellation) surfaced at activation, quantified from validated effect size where present (else `heuristic`)

**Dependencies:** Feature 2 (SAE Management), Feature 3 (Feature Steering), Feature 13 (circuit import)

---

#### Feature 13: Circuit Import, Slice-Fallback & Evidence Ladder
**User Value:** miLLM imports the portable circuit artifact and always tells the truth about how much to trust it — a mined circuit is never presented as causal, and a single-SAE host still gets a usable per-layer projection.

**UI Tab:** Circuits (new)

**Requirements Covered:** FR-13.1 through FR-13.7

**Key Capabilities:**
- `mistudio.circuit-definition/v1` import with strict schema validation (unknown kind / major-version rejected)
- Per-referenced-SAE compatibility matrix (bind / warn-bind / block / unbound); serveable only when all bind
- Per-layer `mistudio.cluster-definition/v1` slice fallback on single-SAE/incomplete deployments (consumed unchanged through the cluster path)
- EvidenceRung surfaced verbatim (circuit rung = MIN over edges); "causal" forbidden below rung 2; unvalidated (rung<2) activation gated behind an explicit acknowledgement
- Definitions treated strictly as data (caps, no paths/credentials/execution)

**Dependencies:** Feature 8 (cluster import path — reused for slices), Feature 12 (multi-SAE serving)

---

#### Feature 14: Circuit-Aware OWUI Dial
**User Value:** End users dial a whole circuit's influence live in real chat — off/min/max on identical prompts — and see whether the circuit is validated, without leaving Open WebUI.

**UI Tab:** None (OWUI-side Filter Function + OpenAI-API extension)

**Requirements Covered:** FR-14.1 through FR-14.4

**Key Capabilities:**
- `steering_intensity` extension dials a whole active circuit (all layers scale together under one λ)
- Request-scoped apply/restore (incl. client disconnect), concurrency-safe
- OWUI Filter surfaces the active circuit's identity and evidence rung (rung<2 visibly marked unvalidated)

**Dependencies:** Feature 10 (OWUI dial filter — extended), Feature 12/13 (active circuit)

---

#### Feature 15: Circuit Edge Sensing
**User Value:** Turn a validated circuit into a live monitor — observe when its EDGES actually fire in production (upstream member firing followed by its downstream partner), closing the loop back to authoring.

**UI Tab:** Circuits (edge-sensing panel)

**Requirements Covered:** FR-15.1 through FR-15.5

**Key Capabilities:**
- Per-forward-pass edge detection (upstream→downstream within a token-lag window), opt-in, off by default
- Alone-vs-within side channel + upstream/downstream member activations
- ±K token context per event (off-hot-path decode)
- Bounded persistence + API/UI/WS readout carrying the edge's rung
- New additive `/api/circuits/*` endpoints + `millm_circuits` MCP category (`docs/mcp-contract.md` v1.1)

**Dependencies:** Feature 11 (sensing hook path — extended), Feature 13 (circuit edges)

---

### Increment: Circuit Consolidation (BRD-MILLM-CIRCUITS-002)

Structural consolidation of the shipped circuit runtime plus the agent surface it never got. Driven by an empirical result rather than an aesthetic one: across the 001 increment, **every review round found a critical regression in the previous round's fix — twelve rounds, twelve for twelve** — because correctness is maintained by convention (three code comments, three duplicate derivations) rather than enforced by construction. Also folds in three shipped-but-unreachable capabilities found by a post-close-out audit, and two hazards measured at the 2026-07-20 GPU close-out.

#### Feature 16: Steering Epoch
**User Value:** An operator's change to live steering is never silently undone by a request that was already in flight — and the API stops reporting success for a change that was reverted.

**UI Tab:** Circuits / Clusters (no new surface; the lie disappears)

**Requirements Covered:** FR-16.1 through FR-16.4

**Key Capabilities:**
- Monotonic `steering_epoch` on `AttachedSAEState`, bumped under the attachment lock by every authoritative writer
- Per-request restore compares the epoch it captured and SKIPS when superseded — last authoritative writer wins
- Covers BOTH the circuit path and the Feature 10 profile path in one change; fixing only one leaves the identical window open a file away
- `set_intensity` stops returning `"reapplied": true` for a change an in-flight request reverted

**Dependencies:** Feature 12 (attachment registry), Feature 14 (per-request dial)

---

#### Feature 17: Request-Scoped Sensing Context
**User Value:** The edge-sensing invariants that took eight criticals across three review rounds to get right become impossible to violate rather than guarded by comments.

**UI Tab:** none (internal)

**Requirements Covered:** FR-17.1 through FR-17.6

**Key Capabilities:**
- One `SensingRequestContext` per request owning the absolute position counter, the fire rings, and the event budget — replacing N independently-advanced per-SAE counters
- **One ring per (request, circuit)**: the ring is keyed by `edge_key` and two circuits can legitimately contain the same edge, so a shared ring would fabricate observations
- Edge machinery extracted from `sae_wrapper.py` (91 `_edge` references in 1373 lines) into `millm/ml/edge_sensing.py`
- Event budget attributed per circuit so one busy circuit cannot starve another's observations
- Characterization tests green BEFORE the move; mutation practice applied after

**Dependencies:** Feature 15 (edge sensing), Feature 19 (contention model shapes the N-circuit design)

---

#### Feature 18: Single Circuit-Serving Derivation
**User Value:** Changing how a circuit is served means changing one thing, not finding three copies that must agree.

**UI Tab:** none (internal)

**Requirements Covered:** FR-18.1 through FR-18.3

**Key Capabilities:**
- `CircuitSteeringEngine` as the ONE derivation, consumed by activation, `set_intensity` and the per-request dial (today: `circuit_service.py:424`, `:799`, `inference_service.py:955`)
- Retires the `SAEService.__new__` bypass at `inference_service.py:743`, whose failure mode is a swallowed `AttributeError` and a silently unsteered response
- F14's two worst defects were both consequences of these derivations drifting

**Dependencies:** Feature 17 (lands on the settled context)

---

#### Feature 19: Concurrent Circuit Serving
**User Value:** Several circuits serve at once, and the one configuration that reliably destroys generation is refused by default rather than discovered in production.

**UI Tab:** Circuits (contention state, incumbent naming)

**Requirements Covered:** FR-19.1 through FR-19.6

**Key Capabilities:**
- **Layer-exclusive claims**: a layer is claimed by at most one active circuit; non-overlapping circuits serve freely
- Overlap REFUSED with `CIRCUIT_LAYER_CONTENTION` (200 + `success:false`), naming the incumbent and the contended layers
- Explicit `allow_layer_overlap` override, mirroring `acknowledge_unvalidated` — the rung header is **omitted** when used, because no single circuit's evidence describes a composed response
- Same-`(layer, feature_idx)` collision refused unconditionally: the merge would serve a strength belonging to neither author
- Drops `uq_circuits_active` for a `circuit_layer_claims` table; tested downgrade; `CIRCUIT_ALLOW_CONCURRENT` flag (default false for one release)
- Design of record: `0xcc/docs/circuit-contention-model.md`

**Dependencies:** Feature 13 (activation), Feature 18 (claim sets from the single derivation)

---

#### Feature 20: MCP Circuit Surface & Reachability Assurance
**User Value:** An agent can do for circuits everything it can already do for clusters — and a capability can no longer ship with no way to invoke it.

**UI Tab:** none (MCP + process)

**Requirements Covered:** FR-20.1 through FR-20.5

**Key Capabilities:**
- A `millm_circuits` category on the existing unified miStudio-hosted MCP server: list, import, activate, deactivate, export, set intensity, status, plus edge-sensing status/events/enable/disable
- Every circuit- and edge-bearing response carries `rung` + server-rendered `rung_language` verbatim; the copy audit extends to the MCP modules
- Reachability rule: no capability is accepted without a test that FAILS when the wiring is removed
- `docs/mcp-contract.md` → v1.2, with status marks distinguishing "endpoint exists" from "reachable"

**Dependencies:** Features 13, 15 (the endpoints), Feature 18 (written against settled code)

---

### Future Features (Post v1.0)

#### Feature 21: Multi-User Authentication
**User Value:** Enable secure access for team environments and non-local deployments.

**Priority:** v1.1+

---

#### Feature 23: GGUF Serving

**User Value:** Run the models the local-inference ecosystem actually
distributes — including large models that only fit on one card once quantized —
through the same OpenAI-compatible API as everything else.

**Priority:** Delivered 2026-09-08 (was listed out-of-scope for v1.0; the
premise did not survive contact with the goal — see BRD-MILLM-GGUF-001)

**Requirements Covered:** FR-23.1 through FR-23.8

**Key Capabilities:**
- Serve `.gguf` files through `/v1/chat/completions`, `/v1/completions`,
  `/v1/embeddings` and `/v1/models`, with no separate client path
- Several quantizations of one repository coexist, named `repo:QUANT`; a bare
  name that has become ambiguous returns 400 naming the alternatives rather than
  picking one
- The context window is **computed** from what the file declares and what free
  VRAM can hold, then confirmed by the load — not searched for by trial
- KV-cache quantization as a first-class VRAM lever: `q8_0` roughly triples the
  usable window over `f16` at near-lossless quality (measured)
- Embeddings on by default, with a graceful fall-back to serving without them
  for architectures that refuse the required pooling mode
- An oversized prompt is a 400, and a truncated answer can be continued

**Not included:** SAE attachment and steering. llama.cpp does not expose the
hook points those need, so interpretability work remains transformers-only.

---

#### Feature 24: Probe Monitor Runtime

**User Value:** Run the detectors trained in miStudio on live traffic at almost
no cost — the cheap first stage every published monitoring cascade uses — with
proof that miLLM scores exactly as miStudio did.

**Priority:** ⏳ **IMPLEMENTED 2026-09-27** (BRD-MILLM-PROBES-001). Phases 0–10 shipped and
merged to `main`; co-released with miStudio 033 phase 7 (MCP contract v1.6), which closed 033
at 49 of 49. **NOT marked ✅**: hardware acceptance (10.3, 10.3b and the SC-4 absolute figure)
is outstanding and needs the deployment plus a loaded model.

**UI Tab:** Probe Monitors (new); the existing "Probe" tab is renamed "Feature Monitor"

**Requirements Covered:** FR-24.1 through FR-24.15

**Key Capabilities:**
- Import probe definitions from file, HF or MCP; refuse the wrong model by name
- Parity gate on the definition's own test vectors before a probe can be armed
- Pre-steering read hook that never depends on SAE attachment; several probes per layer share one hook
- Dense probes and k-sparse SAE probes (a private encoder copy of the probe's features; the SAE is downloaded, never attached)
- Verdicts in a response header or a final stream chunk; live event feed without prompt text
- Evidence rung shown everywhere; acknowledgement to arm below rung 2
- MCP `millm_probes`

**Dependencies:** miStudio Feature 33 (`mistudio.probe-definition/v1`); F11 sensing
lifecycle; F20 MCP contract discipline; F23 (GGUF refused)

#### Feature 22: Neuronpedia Integration
**User Value:** Browse and search features with human-readable labels from Neuronpedia.

**Priority:** v1.1+ (partially delivered post-v1.0: probe feature links derive from attached SAE metadata)

---

### Increment: Dataworks Support (BRD-04)

miLLM as a dependable backend for offline labelling, steered generation and detector work, driven by miDataworks (BRD-03) and the miStudio Model Context Protocol (MCP) tools in BRD-MIS-DATAWORKS-001, which call this HTTP surface. On 2026-10-04/05 miLLM labelled 25,000 rows at 19–21 rows per second through a client loop of single requests to the `/v1/completions` scoring mode. It worked, and it exposed the gaps: scoring on one endpoint only, no job API, nothing protecting a job from a model swap, GPU memory invisible from the REST API, and no way to ask a probe about stored text. **The cross-cutting hazard is the silent drop** — every `/v1` request schema sets `extra="ignore"`, so a client sending `response_format` or `seed` gets a 200 and an answer that ignored them. The increment's rule: every request field is honoured or refused, every long job survives and reports, and every answer says how it was produced. Authentication, co-residency of several models, MCP tools, raising `MAX_CONCURRENT_REQUESTS` and re-enabling continuous batching are out of scope (BRD-04 §4, §7).

#### Feature 25: Chat Scoring, Structured Output, Seed & Request Validation
**User Value:** A labelling job records exactly what produced each row — no field it sent was quietly ignored — and a chat-format classifier scores without rendering its own template.

**Priority:** ❌ Planned (BRD-04, 2026-10-06). First in the build order (BRD-04 RSK-09).

**UI Tab:** none (OpenAI-compatible API)

**Requirements Covered:** FR-25.1 through FR-25.14

**Key Capabilities:**
- `X-miLLM-Ignored-Fields` on every `/v1` response that dropped a field; `X-miLLM-Strict: true` turns the warning into `400`
- One named list of output-changing fields that are always honoured or refused, asserted on every endpoint by one test
- `n` on `/v1/completions` honoured or refused, never silently one choice
- Chat scoring (`logprobs`, `top_logprobs`, `allowed_token_ids`, `extra_messages`) through the shared completion-scoring function, unsteered, in OpenAI's chat logprobs shape
- `response_format` (`json_object`, `json_schema`) through constrained decoding on transformers; `400` with a reason elsewhere; `finish_reason: "length"` instead of truncated JSON
- `seed` applied and echoed (`X-miLLM-Seed`), `system_fingerprint`, and an explicit statement where a seed cannot promise identical output

**Dependencies:** none new. Reuses `_score_prompts` (the per-prompt loop extracted from `_score_text_completion`; Stage 3, 2026-10-06, requested by 025) / `next_token_scores` and `_unsteered`; F23 (GGUF refused before auto-load). Features 28 and 30 implement two entries on FR-25.3's list (`steering`, `dimensions`).

---

#### Feature 26: Batch API
**User Value:** A 50,000-row labelling run is one job with server-side progress, cancel and resume — not 50,000 requests that die with the pod.

**Priority:** ❌ Planned (BRD-04, 2026-10-06). After Features 25 and 29 (RSK-09).

**UI Tab:** none (OpenAI-shaped API; progress on Socket.IO)

**Requirements Covered:** FR-26.1 through FR-26.10 (FR-26.9 retention and FR-26.10 provenance trace to checkpoint decisions, not to a BRD-04 requirement; Stage 3, 2026-10-06, requested by 026)

**Key Capabilities:**
- `POST /v1/files` + `POST /v1/batches` in OpenAI's shape, for chat, completions, embeddings and stateless probe scoring
- Every line validated under strict mode before any row runs; invalid lines to the error file
- PostgreSQL persistence with OpenAI's status values; resume after a pod restart with no recorded row run twice
- Rows admitted only through `_admit()`, one slot per chunk, released between chunks so interactive chat interleaves
- Packed scoring/embedding rows by default, `pack: false` for single-row semantics, with the measured difference published
- One model per batch, held under the model lease for the whole run; row and byte limits enforced at upload

**Dependencies:** Feature 25 (strict validation per line, chat scoring), Feature 29 (lease), Feature 27 (`/api/probes/score` as a batch endpoint), Feature 28 (`X-miLLM-Steering` in batch output lines)

---

#### Feature 27: Probe Scoring, Per-Request Activations & Probe-Path Fixes
**User Value:** A detector operator can ask "what would this probe say about these 10,000 rows" without arming anything, can tag stored text with SAE features, and can trust that no generation path silently skips an armed probe.

**Priority:** ❌ Planned (BRD-04, 2026-10-06). R-04.46 closes a latent defect in the shipped Feature 24.

**UI Tab:** none (API); probe status reports the new not-scored reason

**Requirements Covered:** FR-27.1 through FR-27.10 (FR-27.10, the `>=` verdict boundary, implements decision P-03, not a BRD-04 requirement; Stage 3, 2026-10-06, requested by 027)

**Key Capabilities:**
- `return_sae_activations` on chat and text completions, including scoring mode: this request's activations only, keyed by position, with a pre- or post-steering statement and a size cap
- `POST /api/probes/score`: stateless scoring of any imported probe on `token_ids`, `messages` or `text`, through the parity forward and the armed-probe construction parity uses, same window bars and length bands as live scoring, no event written
- Stateless scoring and the existing parity route both brought under `_admit()`; no global arming, no traffic forced off continuous batching
- Batched chat opens a probe context and records `batched_request` as its not-scored reason
- A discovery-based test over every generation entry point that fails when an armed probe could be skipped silently

**Dependencies:** Feature 24 (probe import, identity check, parity forward, windows and length bands); F11/F17 sensing lifecycle; Feature 25 (scoring mode)

---

#### Feature 28: Inline Steering & Steering-State Header
**User Value:** A steered-generation job sends its feature set with the request instead of creating a saved profile per experiment, and every answer says which steering produced it.

**Priority:** ❌ Planned (BRD-04, 2026-10-06)

**UI Tab:** none (API)

**Requirements Covered:** FR-28.1 through FR-28.4

**Key Capabilities:**
- `steering: {sae_id?, features: [{index, strength}]}` on chat and text completions, applied and restored inside the admission slot; no profile created
- `steering` and `profile` mutually exclusive; `features: []` means explicitly unsteered
- `X-miLLM-Steering` on every generation response — none, profile + intensity, inline + feature count and hash, or circuit — in a header, a final SSE chunk, or the batch output line body
- `/v1/completions` gains `profile`, `steering_intensity` and `steering`

**Dependencies:** F10/F14 per-request dial, F16 steering epoch (per-request restore), Feature 25 (`steering` is on FR-25.3's list)

---

#### Feature 29: Model Lease, Backpressure & GPU Visibility
**User Value:** A long job cannot have its model swapped out from under it by an unrelated request, a busy server says when to retry, and GPU memory held by miLLM is visible without shell access to the node.

**Priority:** ❌ Planned (BRD-04, 2026-10-06). First in the build order with Feature 25 (RSK-09).

**UI Tab:** Models (lease state); Dashboard health

**Requirements Covered:** FR-29.1 through FR-29.8

**Key Capabilities:**
- A model lease with holder, reason and TTL; renew, release, automatic expiry
- `409 MODEL_LEASED` for any load, unload or swap by a non-holder, including every `/v1` auto-load; `X-miLLM-Lease` lets the holder through
- `X-miLLM-Load-Policy: refuse` opts a request out of auto-load; the default stays auto-load for Open WebUI
- `Retry-After` on every `503`; in-flight count, batch backlog and estimated wait in `/api/health/detailed` as a stable contract
- A per-card memory endpoint including miLLM's own allocated and reserved memory

**Dependencies:** F1 model management (load/unload, the existing `locked` flag); F23 (GGUF auto-load paths)

---

#### Feature 30: Embedding Options
**User Value:** Embeddings for retrieval and deduplication use the pooling the caller asked for, and a long input is refused rather than silently cut.

**Priority:** ❌ Planned (BRD-04, 2026-10-06)

**UI Tab:** none (API)

**Requirements Covered:** FR-30.1 through FR-30.4

**Key Capabilities:**
- `dimensions` honoured only where model metadata declares truncated-embedding support; `400` otherwise
- `pooling` (`mean` default, `last`, `cls`) and `normalize` (default false)
- Over-limit input returns `400` naming its index; inputs per request capped
- Route comments brought in line with the code (GGUF embeddings are served)

**Dependencies:** Feature 25 (`dimensions` is on FR-25.3's list); F23 (GGUF embeddings)

**Split note:** BRD-04 places R-04.5 (`dimensions`) in §5.1 Request validation. It is mapped here rather than to Feature 25 because honouring it is embeddings work: truncation must follow pooling, and a truncated vector must be re-normalised when `normalize` (R-04.35) is set, so the two cannot be designed apart. The *refusal* path is FR-25.3's list mechanism and is built in Feature 25; FR-30.1 owns only the honour-or-refuse decision per model.

---

#### BRD-04 Coverage

Every BRD-04 requirement maps to exactly one feature.

| BRD-04 section | Requirements | Feature | FR |
|---|---|---|---|
| 5.1 Request validation | R-04.1–R-04.4 | 25 | FR-25.1–FR-25.4 |
| 5.1 Request validation | R-04.5 | 30 | FR-30.1 |
| 5.2 Scoring on chat completions | R-04.6–R-04.10 | 25 | FR-25.5–FR-25.9 |
| 5.3 Structured output | R-04.11–R-04.13 | 25 | FR-25.10–FR-25.12 |
| 5.4 Reproducibility | R-04.14–R-04.15 | 25 | FR-25.13–FR-25.14 |
| 5.5 Batch API | R-04.16–R-04.23 | 26 | FR-26.1–FR-26.8 |
| 5.6 Per-request SAE activations | R-04.24–R-04.26 | 27 | FR-27.1–FR-27.3 |
| 5.7 Stateless probe scoring | R-04.27–R-04.30 | 27 | FR-27.4–FR-27.7 |
| 5.8 Inline steering | R-04.31–R-04.34 | 28 | FR-28.1–FR-28.4 |
| 5.9 Embeddings | R-04.35–R-04.37 | 30 | FR-30.2–FR-30.4 |
| 5.10 Model lease | R-04.38–R-04.42 | 29 | FR-29.1–FR-29.5 |
| 5.11 Backpressure | R-04.43–R-04.44 | 29 | FR-29.6–FR-29.7 |
| 5.12 GPU visibility | R-04.45 | 29 | FR-29.8 |
| 5.13 Probes on every generation path | R-04.46–R-04.47 | 27 | FR-27.8–FR-27.9 |

Totals: Feature 25 — 14; Feature 26 — 8; Feature 27 — 9; Feature 28 — 4; Feature 29 — 8; Feature 30 — 4; **47 of 47**.

**Acceptance criteria by feature (BRD-04 §6):** 25 — 1, 2 (`n`), 3, 4, 5, 6; 26 — 7, 8, 9; 27 — 10, 11, 17; 28 — 12; 29 — 14, 15, 16; 30 — 2 (`dimensions`), 13. Each wiring item is accepted only by a test that fails when its registration or call line is removed, asserting payload and call count (FR-20.3).

**Open questions the feature documents must resolve (BRD-04 §9):** whether the lease replaces `locked` or sits beside it (Feature 29); the acceptable packed-versus-single difference on JEV-9B-decision before packing defaults off (Feature 26); structured output on GGUF through llama.cpp grammars or refusal (Feature 25); batch file retention on `/data` (Feature 26); `return_sae_activations` default of pre- or post-steering (Feature 27); maximum lease TTL and whether miStudio's GPU workers take the lease (Feature 29).

**Resolved (Stage 3, 2026-10-06):** all six are closed by the checkpoint decisions of 2026-10-06 and the register's defaults. (1) The lease sits beside `locked`; `locked` is retired later (C8). (2) Packing stays on until BRD-04 acceptance 7 measures a difference; a single differing label flips `BATCH_PACK_DEFAULT` to false (T-63). (3) Structured output on GGUF is refused in v1. (4) Batch files are kept 30 days (T-68). (5) Activations default to post-steering, with a pre-steering option. (6) The maximum TTL is 2 hours, renewable; miStudio's GPU workers do not take the lease (T-84).

**Tracked debt from the feature documents (Stage 3, 2026-10-06):** saved profiles steer the first attached SAE, not their own (Feature 28; PADR §10 records it and the open question of fixing both paths together); the management load and unload routes ignore `locked`, and the lease registry assumes a single-process server (Feature 29; PADR §10).

---

### Feature-Requirements Matrix

| Feature | FR-1.x | FR-2.x | FR-3.x | FR-4.x | FR-5.x | FR-6.x | FR-7.x | FR-8.x | FR-9.x | FR-10.x | FR-11.x | FR-12.x | FR-13.x | FR-14.x | FR-15.x | FR-16.x | FR-17.x | FR-18.x | FR-19.x | FR-20.x |
|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|
| 1. Model Management | ✓ | | | | | | ✓ | | | | | | | | | | | | | |
| 2. SAE Management | | ✓ | | | | | ✓ | | | | | ✓ | | | | | | | | |
| 3. Feature Steering | | | ✓ | | | | ✓ | | | | | ✓ | | | | | | | | |
| 4. OpenAI API | ✓ | | ✓ | | ✓ | | | | | | | | | ✓ | | | | | | |
| 5. Admin UI | | | | | | | ✓ | | | | | | ✓ | | | | | | | |
| 6. Profile Management | | | ✓ | | | ✓ | ✓ | | | | | | | | | | | | | |
| 7. Feature Monitoring | | ✓ | | ✓ | | | ✓ | | | | | | | | | | | | | |
| 8. Cluster Import | | | ✓ | | | ✓ | | ✓ | | | | | ✓ | | | | | | | |
| 9. Unified MCP | | | | | | | | ✓ | ✓ | ✓ | ✓ | | | | ✓ | | | | | |
| 10. OWUI Cluster Dial | | | ✓ | | ✓ | | | ✓ | | ✓ | | | | | | | | | | |
| 11. Co-Activation Sensing | | | | ✓ | | | | ✓ | | | ✓ | | | | | | | | | |
| 12. Multi-SAE Attach & Circuit Serving | | ✓ | ✓ | | | | | | | | | ✓ | | | | | | | | |
| 13. Circuit Import, Slice-Fallback & Evidence Ladder | | | ✓ | | | ✓ | | ✓ | | | | ✓ | ✓ | | | | | | | |
| 14. Circuit-Aware OWUI Dial | | | ✓ | | ✓ | | | | | ✓ | | ✓ | | ✓ | | | | | | |
| 15. Circuit Edge Sensing | | | | ✓ | | | | | ✓ | | ✓ | | ✓ | | ✓ | | | | | |
| 16. Steering Epoch | | | | | | | | | | | | | | |   | ✅ | | | | |
| 17. Request-Scoped Sensing Context | | | | | | | | | | | | | | | |   | ✅ | | | |
| 18. Single Serving Derivation | | | | | | | | | | | | | | | | |   | ✅ | | |
| 19. Concurrent Circuit Serving | | | | | | | | | | | | | | | | | |   | ✅ | |
| 20. MCP Circuit Surface | | | | | | | | | | | | | | | | | | |   | ✅ |

---

*Features 23 (GGUF Serving) and 24 (Probe Monitor Runtime) map one-to-one to their own requirement groups, FR-23.x and FR-24.x. Features 25–30 (Dataworks Support) likewise map one-to-one to FR-25.x–FR-30.x; their BRD-04 requirement mapping is the coverage table under "Increment: Dataworks Support (BRD-04)".*


## 7. User Experience Goals

### Overall UX Principles

1. **Progressive Disclosure:** Simple by default, advanced options available
2. **Immediate Feedback:** Real-time updates for all operations
3. **Fail Gracefully:** Clear error messages with recovery guidance
4. **Consistent Patterns:** Same interaction patterns across all tabs
5. **Keyboard Accessible:** Full functionality without mouse

### Visual Design Guidelines
- Dark theme optimized for extended use (per UI mockup)
- Monospace fonts for technical values (feature indices, activations)
- Color-coded status indicators (green=active, cyan=ready, purple=attached, yellow=active profile)
- Minimal animations, focused on functional feedback

### Accessibility Requirements
- WCAG 2.1 AA compliance target
- Screen reader compatibility for core workflows
- Sufficient color contrast ratios
- Keyboard navigation support

### Performance Expectations
- UI responsive during model operations (loading indicators)
- Real-time monitoring updates without lag
- Slider adjustments reflected immediately
- Page transitions <100ms

### Error Handling UX
- Toast notifications for transient errors
- Inline validation for form inputs
- Clear error states with resolution steps
- No silent failures

---

## 8. Business Considerations

### Budget and Resource Constraints
- Open source project with community development model
- No commercial licensing constraints
- GPU hardware required for development and testing

### Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| SAE hooking latency unacceptable | Medium | High | Benchmark early; provide bypass option |
| SAE format incompatibilities | Medium | Medium | Start with SAELens, document formats |
| Memory exhaustion (model + SAE) | Medium | Medium | Require quantization for large models; show estimates |
| OpenAI API spec drift | Low | Medium | Target stable v1 endpoints; integration tests |
| Misuse for harmful manipulation | Medium | Medium | Document ethical considerations; demonstrate risk |

### Competitive Landscape
- **Ollama:** Popular but no SAE support, requires model conversion
- **vLLM:** High performance but no interpretability features
- **llama.cpp:** Lightweight but no SAE integration
- **TransformerLens:** Research-focused, not production inference

**miLLM Differentiation:** Only solution bridging interpretability research with production-style inference.

### Value Creation Model
- Open source community value
- Research enablement
- Educational resource for interpretability
- Foundation for miStudio ecosystem

---

## 9. Technical Considerations (High-Level)

### Deployment Environment
- Primary: Docker with NVIDIA Container Toolkit
- Secondary: Direct Python installation for development
- Target: Single-machine, GPU-equipped workstations

### Two API Architecture

miLLM exposes two distinct API surfaces:

#### 1. OpenAI-Compatible Inference API
- Purpose: Model inference for client applications
- Endpoints: `/v1/chat/completions`, `/v1/completions`, `/v1/models`, `/v1/embeddings`
- Consumers: Open WebUI, LibreChat, custom applications
- Protocol: REST + SSE streaming

#### 2. miLLM Management API
- Purpose: Server configuration and control
- Functions: Model management, SAE management, steering control, profile management, monitoring
- Consumers: Admin UI, future miStudio integration
- Protocol: REST + WebSocket (for real-time monitoring)

### Security and Privacy
- Local-first architecture (no data leaves user's machine)
- No authentication in v1.0 (trusted network assumption)
- Architecture supports future auth layer
- No telemetry or usage tracking

### Performance and Scalability
- Single-user focus for v1.0
- Request queuing for concurrent requests
- GPU memory optimization via quantization
- Lazy loading for models and SAEs

### Technology Preferences
- **Backend:** Python (FastAPI) - required for PyTorch/Transformers ecosystem
- **Frontend:** Modern web framework (specific choice in ADR)
- **Model Loading:** Hugging Face Transformers
- **SAE Framework:** SAELens-compatible
- **Quantization:** bitsandbytes
- **Container:** Docker with NVIDIA runtime

**Note:** Detailed technology stack decisions will be made in the Architecture Decision Record (ADR).

---

## 10. Project Constraints

### Timeline Constraints
- Standard development cycle (2-4 months)
- Quality over speed - thorough and accurate implementation
- Complete v1.0 scope before launch (no partial releases)

### Technical Constraints
- NVIDIA GPU required (CUDA dependency)
- Python ecosystem (Transformers, PyTorch)
- Hugging Face model format dependency
- Single-SAE limitation for v1.0

### Resource Constraints
- Open source development model
- Community contribution dependent
- Testing hardware availability

### Regulatory Constraints
- None identified for v1.0
- Future: Consider implications of steering for safety-critical applications

---

## 11. Success Metrics

### Quantitative Measures

| Metric | Target | Measurement |
|--------|--------|-------------|
| SAE overhead | <15% latency | Automated benchmark |
| API compatibility | 100% | Integration test suite |
| Time to first token | <500ms | Performance monitoring |
| Docker startup | <30s (excluding model load) | Automated test |
| UI responsiveness | <100ms interactions | Performance audit |

### Qualitative Indicators
- Users can complete the "yelling demo" scenario end-to-end
- Documentation enables self-service setup
- Error messages lead to successful resolution
- UI feels responsive and professional

### User Satisfaction Metrics
- GitHub stars/forks as adoption proxy
- Issue resolution time
- Community contributions
- Documentation completeness feedback

### Business Impact Measurements
- Adoption in interpretability research papers
- Integration with miStudio (when available)
- Community growth and engagement
- Reference in interpretability tooling discussions

---

## 12. Next Steps

### Immediate Actions
1. **Create Architecture Decision Record (ADR)**
   - Technology stack selection (frontend framework, etc.)
   - Development standards and patterns
   - Project structure decisions

2. **Update CLAUDE.md**
   - Copy Project Standards section from ADR
   - Update document inventory
   - Set feature priority order

### Feature Development Sequence

Based on dependencies and logical workflow:

| Priority | Feature | Rationale |
|----------|---------|-----------|
| 1 | Model Management | Foundation - everything depends on this |
| 2 | OpenAI API Compatibility | Core value proposition |
| 3 | SAE Management | Enables interpretability features |
| 4 | Feature Steering | Core differentiator |
| 5 | Feature Monitoring | Complements steering |
| 6 | Profile Management | Workflow optimization |
| 7 | Admin UI | Integrates all features (parallel development) |

### Architecture Evaluation Needs
- Frontend framework selection (React vs Vue vs Svelte)
- State management approach
- WebSocket vs polling for monitoring
- SAE hooking mechanism design
- Profile format schema definition

### Resource Planning
- Identify core contributors
- Establish development environment standards
- Set up CI/CD pipeline
- Create contribution guidelines

---

## Appendix A: Glossary

| Term | Definition |
|------|------------|
| SAE | Sparse Autoencoder - neural network that decomposes activations into interpretable features |
| Feature | Learned direction in activation space corresponding to human-interpretable concept |
| Steering | Modifying model behavior by adjusting feature activation strengths during inference |
| Gemma-Scope | Project that trained SAEs on Gemma 2 models with feature annotations |
| Neuronpedia | Platform hosting visualizations and labels for SAE features |
| Hooking | Intercepting model activations at a specific layer to read or modify them |
| miStudio | Companion application for SAE training, feature discovery, and steering experiments |
| SAELens | Library/format for working with Sparse Autoencoders |

---

## Appendix B: Reference Documents

- **BRD:** `0xcc/docs/miLLM_BRD_v1.0.md`
- **UI Mockup:** `0xcc/spec/miLLM_UI.jsx`
- **Framework Guide:** `0xcc/instruct/000_README.md`

---

## Appendix C: Example Use Case

**Scenario: Demonstrating Feature Steering (from BRD)**

1. Launch miLLM and access the admin UI
2. Download `google/gemma-2-2b` from Hugging Face
3. Download the corresponding Gemma-Scope SAE for layer 12
4. Attach the SAE to the loaded model
5. Locate feature #1234 (labeled "yelling/capitalization" in Neuronpedia)
6. Set feature #1234 strength to +5.0
7. Save this configuration as profile "yelling-demo"
8. Configure Open WebUI to use miLLM as backend
9. Send a chat message; observe responses in ALL CAPS
10. Return to admin UI; observe feature #1234 activation values during conversation

---

**Document Status:** Ready for ADR Creation
**Next Document:** `000_PADR|miLLM.md` (Architecture Decision Record)
**Instruction File:** `@0xcc/instruct/002_create-adr.md`
