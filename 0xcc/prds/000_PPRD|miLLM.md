# Project PRD: miLLM

## Mechanistic Interpretability LLM Server

**Document Version:** 1.0
**Created:** January 30, 2026
**Status:** Draft
**Reference:** BRD v1.0 (January 29, 2026)

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
3. **Model Setup:** Download model from HuggingFace, select quantization
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
- 4-bit and 8-bit quantization via bitsandbytes
- Local model caching
- Memory requirement estimation

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
- FR-1.3: Support 4-bit and 8-bit quantization
- FR-1.4: Cache downloaded models locally
- FR-1.5: Display memory requirements before loading
- FR-1.6: Support extensible model formats via Transformers

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

### Non-Functional Requirements

#### Performance (NFR-1.x)
- NFR-1.1: SAE hook overhead <15% vs base model latency
- NFR-1.2: Graceful request queuing for 5+ pending requests
- NFR-1.3: Time to first token <500ms after model loaded

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

#### miStudio Integration (Future-Ready)
- Profile export format: JSON schema (model, SAE, layer, features)
- Profile import with validation
- Management API designed for future miStudio direct integration

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
- Quantization selection (Q4, Q8, FP16)
- Model loading/unloading
- Memory estimation display
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

### Future Features (Post v1.0)

#### Feature 8: Multi-User Authentication
**User Value:** Enable secure access for team environments and non-local deployments.

**Priority:** v1.1+

---

#### Feature 9: Multi-SAE Support
**User Value:** Attach SAEs to multiple layers for more sophisticated steering.

**Priority:** v2.0

---

#### Feature 10: Neuronpedia Integration
**User Value:** Browse and search features with human-readable labels from Neuronpedia.

**Priority:** v1.1+

---

### Feature-Requirements Matrix

| Feature | FR-1.x | FR-2.x | FR-3.x | FR-4.x | FR-5.x | FR-6.x | FR-7.x |
|---------|--------|--------|--------|--------|--------|--------|--------|
| 1. Model Management | ✓ | | | | | | ✓ |
| 2. SAE Management | | ✓ | | | | | ✓ |
| 3. Feature Steering | | | ✓ | | | | ✓ |
| 4. OpenAI API | ✓ | | ✓ | | ✓ | | |
| 5. Admin UI | | | | | | | ✓ |
| 6. Profile Management | | | ✓ | | | ✓ | ✓ |
| 7. Feature Monitoring | | ✓ | | ✓ | | | ✓ |

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
