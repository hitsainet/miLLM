# Project PRD: miLLM

## Mechanistic Interpretability LLM Server

**Document Version:** 1.3
**Created:** January 30, 2026
**Status:** Draft
**Reference:** BRD v1.0 (January 29, 2026) · BRD-MILLM-CLUSTERS-001 (July 16, 2026) · BRD-MILLM-CIRCUITS-001 (July 20, 2026) · BRD-MILLM-CIRCUITS-002 (July 20, 2026)

### Document Revision History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2026-01-30 | Initial project PRD (Features 1–7, from BRD v1.0) |
| 1.1 | 2026-07-16 | Cluster Runtime increment (BRD-MILLM-CLUSTERS-001): Features 8–11 (Cluster Import, Unified MCP, OWUI Cluster Dial, Co-Activation Sensing), FR-8.x–FR-11.x, NFR-1.4, matrix extension; former future stubs renumbered 12–14 |
| 1.2 | 2026-07-20 | Circuit Runtime increment (BRD-MILLM-CIRCUITS-001): Features 12–15 (Multi-SAE Attach & Circuit Serving, Circuit Import + Slice-Fallback + Evidence Ladder, Circuit-Aware OWUI Dial, Circuit Edge Sensing), FR-12.x–FR-15.x, NFR-1.5, matrix extension; retired the former "Multi-SAE Support" future stub (now specified as Feature 12); remaining future stubs renumbered 16+ |
| 1.3 | July 20, 2026 | Circuit Consolidation increment (BRD-MILLM-CIRCUITS-002): Features 16-20 (steering epoch, request-scoped sensing context, single serving derivation, concurrent circuit serving, MCP circuit surface + reachability assurance), FR-16.x-20.x, matrix columns; future stubs renumbered 21/22. |

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
| GGUF model format | Focus on Transformers ecosystem | v1.1+ |
| Kubernetes deployment | Docker sufficient for target users | v1.1+ |
| Feature discovery/analysis tools | Delegated to miStudio | N/A |
| Neuronpedia API integration | Nice-to-have, not core | v1.1+ |
| Direct miStudio push integration | miStudio developing simultaneously | v1.1+ |

### Future Roadmap Considerations
- Multi-layer SAE support with coordinated feature adjustment
- Additional model format support (GGUF, etc.)
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
- FR-19.3: An explicit `allow_layer_overlap` acknowledgement SHALL permit additive composition; while any layer is composed, `X-miLLM-Circuit-Rung` SHALL be OMITTED, because no single circuit's evidence describes the response.
- FR-19.4: Two active circuits naming the same `(layer, feature_idx)` SHALL be refused unconditionally, with no override, since the merge would serve a strength belonging to neither author.
- FR-19.5: Deactivation SHALL release only that circuit's own claims and steering keys, never a co-tenant's.
- FR-19.6: The capability SHALL ship behind `CIRCUIT_ALLOW_CONCURRENT` (default false for one release) with a tested downgrade, since the first concurrent activation is a one-way door in deployed data.

#### MCP Circuit Surface & Reachability (FR-20.x) — Increment: Circuit Consolidation
- FR-20.1: A `millm_circuits` MCP category SHALL expose every circuit capability reachable by REST — list, import, activate, deactivate, export, set intensity, status, and edge-sensing status/events/enable/disable.
- FR-20.2: Every circuit- and edge-bearing MCP response SHALL carry `rung` and server-rendered `rung_language` verbatim; the build-failing copy audit SHALL extend to the MCP modules and their tool descriptions.
- FR-20.3: No capability SHALL be accepted as shipped without a test that FAILS when its user- or agent-facing wiring is removed; a test asserting only that an entry point exists SHALL NOT satisfy this.
- FR-20.4: Documentation status marks SHALL distinguish "endpoint exists" from "reachable by a user or agent".
- FR-20.5: `docs/mcp-contract.md` SHALL move to v1.2, additive-only.

### Non-Functional Requirements

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

#### Feature 22: Neuronpedia Integration
**User Value:** Browse and search features with human-readable labels from Neuronpedia.

**Priority:** v1.1+ (partially delivered post-v1.0: probe feature links derive from attached SAE metadata)

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
| 16. Steering Epoch | | | | | | | | | | | | | | | ✅ | | | | | |
| 17. Request-Scoped Sensing Context | | | | | | | | | | | | | | | | ✅ | | | | |
| 18. Single Serving Derivation | | | | | | | | | | | | | | | | | ✅ | | | |
| 19. Concurrent Circuit Serving | | | | | | | | | | | | | | | | | | ✅ | | |
| 20. MCP Circuit Surface | | | | | | | | | | | | | | | | | | | ✅ | |

---

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
