# Project: miLLM - Mechanistic Interpretability LLM Server

## Current Status
- **Phase:** **BRD-MILLM-CLUSTERS-001 increment CLOSED (2026-07-16)** — Features 008/010/011/009 all ACCEPTED (implementation + 3 deep review rounds each + live E2E)
- **Last Session:** July 16, 2026 — **increment closeout**: 011 Coactivation_Sensing ACCEPTED (3 rounds, 56 findings/36 fixed; live E2E: real co-firing event w/ context, warm overhead 3.9ms, arm-refusal path) and 009 Unified_MCP ACCEPTED (3 rounds, 41 findings/32 fixed; deployed unified server with 15 millm_* tools live, per-product /health, MCP tools/call verified, cross-product import→activate→dial→sensing flow E2E). Live E2E also caught+fixed profiles.sae_id VARCHAR(50) truncation (migration 009). Production fixes ridden along: SAE attach 500 (stdlib-logger kwargs, user-reported), mirror CI red (sync filter stripped the public contract), MAX_CONCURRENT_REQUESTS 2→1 (serial correctness boundary), OWUI steering-tile grid. Suites: backend 1084+/frontend 205.
- **Next Steps:** Increment complete. Backlog: sweep `manual/docs/` for model-family-specific recommendations and genericize ('latest popular offline models...'); refactor debts named in the review records (SensingRequestContext consolidation, ProfileResolver, queue-serialized steering/arm mutations, miStudio-tools gating, span-highlight context_start_pos)
- **Active Document:** none — increment closed; review records in 0xcc/reviews/
- **Backlog:**
  - Copy audit: no UI/docs text may recommend a SPECIFIC model family (dated-info risk — user directive 2026-07-16). UI pages fixed (SAE/Models tips now "latest popular offline models"; SteeringSlider Neuronpedia link now derives from the attached SAE). Remaining: sweep `manual/docs/` for model-family-specific recommendations and genericize.
- **New Increment (scoped 2026-07-16):** cluster-definition import (mistudio.cluster-definition/v1, file + HF consume-only) → steering profiles; unified MCP server (both products, health-gated categories); Open WebUI live cluster dial; cluster co-activation sensing. Locked: incremental BRD style; HF publishing stays miStudio-side; budgets frozen on import (assumed, confirm pre-PRD).
- **Baseline:** v1.0 features 001–007 all ✅; increment suites: backend 984 / frontend 198 (2026-07-16) + post-v1.0: GitOps/ArgoCD K8s deploy, CBM + speculative decoding + torch.compile, hybrid/Mamba support, Neuronpedia links, security hardening. Version 0.5.1.

## Quick Resume Commands
```
# XCC session start sequence
"Please help me resume where I left off"
# Or manual if needed: read CLAUDE.md, 0xcc/session_state.json
ls -la 0xcc/*/

# Load project context on demand (use Read tool):
#   0xcc/prds/000_PPRD|miLLM.md
#   0xcc/adrs/000_PADR|miLLM.md

# Load current work area based on phase:
#   0xcc/prds/      # For PRD work
#   0xcc/tdds/      # For TDD work
#   0xcc/tids/      # For TID work
#   0xcc/tasks/     # For task execution
```

## Housekeeping Commands
```bash
"Please create a checkpoint"        # Save complete state
"Please help me resume"            # Restore context for new session
"My context is getting too large"  # Clean context, restore essentials
"Please save the session transcript" # Save session transcript
"Please show me project status"    # Display current state
```

## Project Standards

### Technology Stack
- **Frontend:** React 18 + TypeScript, Tailwind CSS, Zustand, React Query
- **Backend:** Python 3.11+ / FastAPI, SQLAlchemy 2.0, Pydantic v2
- **Database:** PostgreSQL 14+ (primary), Redis 7+ (cache/real-time)
- **ML Stack:** PyTorch 2.0+, Transformers, SAELens, bitsandbytes
- **Real-time:** Socket.IO (WebSocket) for monitoring, SSE for inference streaming
- **Testing:** pytest (80%+ coverage), Vitest + RTL, Playwright for E2E
- **Deployment:** Docker with NVIDIA runtime, docker-compose

### Code Organization

#### Backend (Layer-Based)
```
millm/
├── api/routes/          # API endpoints (openai/, management/, system/)
├── services/            # Business logic (model_service, sae_service, etc.)
├── ml/                  # ML code (model_loader, sae_loader, hooks, steering)
├── db/                  # Database (models/, repositories/, migrations/)
├── sockets/             # WebSocket handlers
├── core/                # Config, logging, errors
└── main.py
```

#### Frontend (Feature-Based)
```
src/
├── components/          # React components (common/, layout/, models/, etc.)
├── pages/               # Route pages
├── stores/              # Zustand stores
├── services/            # API clients
├── hooks/               # Custom hooks
├── types/               # TypeScript types
└── utils/
```

### Coding Patterns

#### Backend
- Service layer pattern with dependency injection
- Pydantic schemas for all request/response models
- Async/await for all I/O operations
- Structured logging with structlog

#### Frontend
- Zustand for global state, React Query for server state
- Custom hooks for data fetching (useModels, useSAEs, etc.)
- Tailwind CSS for styling (no inline styles in production)
- TypeScript strict mode
- **IMPORTANT:** Never use computed getters in Zustand stores - they don't trigger re-renders. Always access state properties directly (e.g., use `steering` not `get steeringState()`)

### API Conventions
- OpenAI API: `/v1/chat/completions`, `/v1/completions`, `/v1/models`, `/v1/embeddings`
- Management API: `/api/models`, `/api/saes`, `/api/steering`, `/api/profiles`
- WebSocket: `/monitor`, `/system`, `/progress` namespaces
- Response format: `{ success: boolean, data: T | null, error: ErrorDetails | null }`

### Quality Requirements
- Backend coverage: 80%+ (pytest --cov-fail-under=80)
- Frontend: Unit tests for stores/hooks, component tests for UI
- E2E: Critical paths (model load → SAE attach → steering workflow)
- All PRs require review and passing CI

### Error Handling
- Custom exceptions: MiLLMError, ModelNotLoadedError, SAENotAttachedError
- Error codes: MODEL_NOT_FOUND, SAE_NOT_ATTACHED, INSUFFICIENT_MEMORY, etc.
- Graceful degradation: Disable SAE on OOM, continue with base model

### Commit Format
```
feat(models): add quantization selection
fix(steering): correct activation scaling
docs(api): add OpenAPI examples
refactor(services): extract HuggingFace logic
```

## AI Dev Tasks Framework Workflow

### Document Creation Sequence
1. **Project Foundation**
   - `000_PPRD|[project-name].md` → `0xcc/prds/` (Project PRD)
   - `000_PADR|[project-name].md` → `0xcc/adrs/` (Architecture Decision Record)
   - Update this CLAUDE.md with Project Standards from ADR

2. **Feature Development** (repeat for each feature)
   - `[###]_FPRD|[feature-name].md` → `0xcc/prds/` (Feature PRD)
   - `[###]_FTDD|[feature-name].md` → `0xcc/tdds/` (Technical Design Doc)
   - `[###]_FTID|[feature-name].md` → `0xcc/tids/` (Technical Implementation Doc)
   - `[###]_FTASKS|[feature-name].md` → `0xcc/tasks/` (Task List)

### Instruction Documents Reference
- `0xcc/instruct/001_create-project-prd.md` - Creates project vision and feature breakdown
- `0xcc/instruct/002_create-adr.md` - Establishes tech stack and standards
- `0xcc/instruct/003_create-feature-prd.md` - Details individual feature requirements
- `0xcc/instruct/004_create-tdd.md` - Creates technical architecture and design
- `0xcc/instruct/005_create-tid.md` - Provides implementation guidance and coding hints
- `0xcc/instruct/006_generate-tasks.md` - Generates actionable development tasks
- `0xcc/instruct/007_process-task-list.md` - Guides task execution and progress tracking
- `0xcc/instruct/008_housekeeping.md` - Session management and context preservation

## Document Inventory

### Project Level Documents
- ✅ 0xcc/docs/miLLM_BRD_v1.0.md (original v1.0 BRD)
- ✅ 0xcc/prds/BRD-MILLM-CLUSTERS-001.md (Incremental BRD — Cluster Runtime: import / unified MCP / OWUI dial / sensing, drafted 2026-07-16)
- ✅ 0xcc/prds/000_PPRD|miLLM.md (Project PRD v1.1 — Features 8–11 added 2026-07-16)
- ✅ 0xcc/adrs/000_PADR|miLLM.md (Architecture Decision Record v1.1 — increment decisions 2026-07-16)

**Increment: Cluster Runtime (BRD-MILLM-CLUSTERS-001) — docs complete 2026-07-16, implementation not started**

**Feature 8: Cluster Import**
- ✅ 0xcc/prds/008_FPRD|Cluster_Import.md
- ✅ 0xcc/tdds/008_FTDD|Cluster_Import.md
- ✅ 0xcc/tids/008_FTID|Cluster_Import.md
- ✅ 0xcc/tasks/008_FTASKS|Cluster_Import.md

**Feature 9: Unified MCP**
- ✅ 0xcc/prds/009_FPRD|Unified_MCP.md
- ✅ 0xcc/tdds/009_FTDD|Unified_MCP.md
- ✅ 0xcc/tids/009_FTID|Unified_MCP.md
- ✅ 0xcc/tasks/009_FTASKS|Unified_MCP.md

**Feature 10: OWUI Cluster Dial**
- ✅ 0xcc/prds/010_FPRD|OWUI_Cluster_Dial.md
- ✅ 0xcc/tdds/010_FTDD|OWUI_Cluster_Dial.md
- ✅ 0xcc/tids/010_FTID|OWUI_Cluster_Dial.md
- ✅ 0xcc/tasks/010_FTASKS|OWUI_Cluster_Dial.md

**Feature 11: Co-Activation Sensing**
- ✅ 0xcc/prds/011_FPRD|Coactivation_Sensing.md
- ✅ 0xcc/tdds/011_FTDD|Coactivation_Sensing.md
- ✅ 0xcc/tids/011_FTID|Coactivation_Sensing.md
- ✅ 0xcc/tasks/011_FTASKS|Coactivation_Sensing.md

### Feature Documents

**Feature 1: Model Management**
- ✅ 0xcc/prds/001_FPRD|Model_Management.md
- ✅ 0xcc/tdds/001_FTDD|Model_Management.md
- ✅ 0xcc/tids/001_FTID|Model_Management.md
- ✅ 0xcc/tasks/001_FTASKS|Model_Management.md

**Feature 2: OpenAI API Compatibility**
- ✅ 0xcc/prds/002_FPRD|OpenAI_API.md
- ✅ 0xcc/tdds/002_FTDD|OpenAI_API.md
- ✅ 0xcc/tids/002_FTID|OpenAI_API.md
- ✅ 0xcc/tasks/002_FTASKS|OpenAI_API.md

**Feature 3: SAE Management**
- ✅ 0xcc/prds/003_FPRD|SAE_Management.md
- ✅ 0xcc/tdds/003_FTDD|SAE_Management.md
- ✅ 0xcc/tids/003_FTID|SAE_Management.md
- ✅ 0xcc/tasks/003_FTASKS|SAE_Management.md

**Feature 4: Feature Steering**
- ✅ 0xcc/prds/004_FPRD|Feature_Steering.md
- ✅ 0xcc/tdds/004_FTDD|Feature_Steering.md
- ✅ 0xcc/tids/004_FTID|Feature_Steering.md
- ✅ 0xcc/tasks/004_FTASKS|Feature_Steering.md

**Feature 5: Feature Monitoring**
- ✅ 0xcc/prds/005_FPRD|Feature_Monitoring.md
- ✅ 0xcc/tdds/005_FTDD|Feature_Monitoring.md
- ✅ 0xcc/tids/005_FTID|Feature_Monitoring.md
- ✅ 0xcc/tasks/005_FTASKS|Feature_Monitoring.md

**Feature 6: Profile Management**
- ✅ 0xcc/prds/006_FPRD|Profile_Management.md
- ✅ 0xcc/tdds/006_FTDD|Profile_Management.md
- ✅ 0xcc/tids/006_FTID|Profile_Management.md
- ✅ 0xcc/tasks/006_FTASKS|Profile_Management.md

**Feature 7: Admin UI**
- ✅ 0xcc/prds/007_FPRD|Admin_UI.md
- ✅ 0xcc/tdds/007_FTDD|Admin_UI.md
- ✅ 0xcc/tids/007_FTID|Admin_UI.md
- ✅ 0xcc/tasks/007_FTASKS|Admin_UI.md

### Status Indicators
- ✅ **Complete:** Document finished and reviewed
- ⏳ **In Progress:** Currently being worked on
- ❌ **Pending:** Not yet started
- 🔄 **Needs Update:** Requires revision based on changes

## Housekeeping Status
- **Last Checkpoint:** May 11, 2026 - Inference optimisation + test fixes + security + code review
- **Last Transcript Save:** N/A
- **Context Health:** Excellent
- **Session Count:** Multiple sessions (documentation + implementation + bug fixes + audit + hardening)
- **Total Development Time:** All features implemented, audited, hardened, and 705 tests passing

## Task Execution Standards

### Completion Protocol
- ✅ One sub-task at a time, ask permission before next
- ✅ Mark sub-tasks complete immediately: `[ ]` → `[x]`
- ✅ When parent task complete: Run tests → Stage → Clean → Commit → Mark parent complete
- ✅ Never commit without passing tests
- ✅ Always clean up temporary files before commit

### Commit Message Format
```bash
git commit -m "feat: [brief description]" -m "- [key change 1]" -m "- [key change 2]" -m "Related to [Task#] in [PRD]"
```

### Test Commands
- **Backend:** `pytest` or `pytest --cov=millm`
- **Frontend:** `npm test` or `npm run test:coverage`
- **E2E:** `npx playwright test`
- **Lint:** `ruff check .` (backend), `npm run lint` (frontend)
- **Type Check:** `mypy millm/` (backend), `npm run typecheck` (frontend)

## Code Quality Checklist

### Before Any Commit
- [ ] All tests passing
- [ ] No console.log/print debugging statements
- [ ] No commented-out code blocks
- [ ] No temporary files (*.tmp, .cache, etc.)
- [ ] Code follows project naming conventions
- [ ] Functions/methods have docstrings if required
- [ ] Error handling implemented per ADR standards

### File Organization Rules
- **Backend:** Layer-based (api/, services/, ml/, db/, sockets/, core/)
- **Frontend:** Feature-based (components/, pages/, stores/, services/, hooks/, types/)
- **Tests:** `test_*.py` alongside Python modules, `*.test.tsx` alongside components
- **Naming:** PascalCase for React/classes, snake_case for Python, camelCase for TS functions
- **Imports:** External → internal → relative
- **Framework files:** `0xcc/` directory
- **Project files:** `millm/` (backend), `frontend/src/` (frontend)

## Context Management

### Session End Protocol
```bash
# 1. Update CLAUDE.md status section
# 2. Create session summary
"Please create a checkpoint"
# 3. Commit progress
git add .
git commit -m "docs: completed [task] - Next: [specific action]"
```

### Context Recovery (If Lost)
```
# Mild context loss - read these files:
#   CLAUDE.md
#   0xcc/session_state.json
#   ls -la 0xcc/*/
#   0xcc/instruct/[current-phase].md

# Severe context loss - read these files:
#   CLAUDE.md
#   0xcc/prds/000_PPRD|miLLM.md
#   0xcc/adrs/000_PADR|miLLM.md
#   ls -la 0xcc/*/
```

### Resume Commands for Next Session
```
# Standard resume sequence
"Please help me resume where I left off"
# Or manual if needed, read these files:
#   CLAUDE.md
#   0xcc/session_state.json
#   [specific-file-currently-working-on]
# Specific next action: [detailed action]
```

## Progress Tracking

### Task List Maintenance
- Update task list file after each sub-task completion
- Add newly discovered tasks as they emerge
- Update "Relevant Files" section with any new files created/modified
- Include one-line description for each file's purpose
- Distinguish between framework files (0xcc/) and project files (src/, tests/, etc.)

### Status Indicators for Tasks
- `[ ]` = Not started
- `[x]` = Completed
- `[~]` = In progress (use sparingly, only for current sub-task)
- `[?]` = Blocked/needs clarification

### Session Documentation
After each development session, update:
- Current task position in this CLAUDE.md
- Any blockers or questions encountered
- Next session starting point
- Files modified in this session (both 0xcc/ and project files)

## Implementation Patterns

### Error Handling
- Custom exceptions: MiLLMError (base), ModelNotLoadedError, SAENotAttachedError
- Error codes: MODEL_NOT_FOUND, SAE_NOT_ATTACHED, INSUFFICIENT_MEMORY, etc.
- Graceful degradation: Disable SAE on OOM, continue with base model
- Response format: `{ success: false, data: null, error: { code, message, details } }`
- Log errors with structlog at appropriate levels

### Testing Patterns
- Backend: pytest with fixtures, 80%+ coverage required
- Frontend: Vitest + React Testing Library for components
- E2E: Playwright for critical paths
- Test naming: `test_[function]_[scenario]` (Python), `describe/it` (TypeScript)
- Mock external dependencies (HuggingFace, GPU operations)
- Test both happy path and error cases

## Debugging Protocols

### When Tests Fail
1. Read error message carefully
2. Check recent changes for obvious issues
3. Run individual test to isolate problem
4. Use debugger/console to trace execution
5. Check dependencies and imports
6. Ask for help if stuck > 30 minutes

### When Task is Unclear
1. Review original PRD requirements
2. Check TDD for design intent
3. Look at TID for implementation hints
4. Ask clarifying questions before proceeding
5. Update task description for future clarity

## Feature Priority Order
*From Project PRD - Based on dependencies and logical workflow*

**Core Features (MVP):**
1. Model Management - Foundation, everything depends on this
2. OpenAI API Compatibility - Core value proposition
3. SAE Management - Enables interpretability features
4. Feature Steering - Core differentiator
5. Feature Monitoring - Complements steering
6. Profile Management - Workflow optimization
7. Admin UI - Integrates all features (parallel development possible)

## Session History Log

### Session 1: January 30, 2026 - Project Foundation
- **Accomplished:**
  - Reviewed BRD v1.0 and UI mockup
  - Completed strategic clarifying questions for Project PRD
  - Created comprehensive Project PRD (000_PPRD|miLLM.md)
  - Defined 7 core features organized by UI workflow
  - Established two-API architecture (OpenAI inference + Management API)
  - Completed technical evaluation questions for ADR
  - Created comprehensive ADR (000_PADR|miLLM.md)
  - Updated CLAUDE.md with Project Standards from ADR
- **Key Decisions (PRD):**
  - Full v1.0 scope (all BRD requirements are hard requirements)
  - Standard development timeline (2-4 months, quality focus)
  - All success criteria equally weighted
  - Features organized by UI tabs (Models, SAEs, Steering, Profiles, Monitor)
  - miStudio integration deferred to post-v1.0
  - SAELens format support (broad SAE format compatibility)
- **Key Decisions (ADR - aligned with miStudio):**
  - Frontend: React 18 + Tailwind CSS + Zustand
  - Backend: Python 3.11+ / FastAPI (required for PyTorch)
  - Database: PostgreSQL 14+ / Redis 7+
  - Real-time: Socket.IO for monitoring, SSE for inference
  - Testing: Balanced pyramid (80%+ backend coverage)
  - Code Organization: Layer-based
- **Files Created:**
  - 0xcc/prds/000_PPRD|miLLM.md
  - 0xcc/adrs/000_PADR|miLLM.md
- **Files Modified:** CLAUDE.md
- **Next:** Create Model Management TDD using `0xcc/instruct/004_create-tdd.md`

### Session 1 (continued): Feature 1 PRD
- **Accomplished:**
  - Completed feature-specific clarifying questions for Model Management
  - Created comprehensive Feature PRD (001_FPRD|Model_Management.md)
  - Defined 6 primary user stories with acceptance criteria
  - Specified 30+ functional requirements
  - Detailed API endpoints and WebSocket events
  - Documented error codes and edge cases
- **Key Decisions (Model Management):**
  - HuggingFace + Local path support (no URL downloads)
  - trust_remote_code explicit opt-in per download
  - Progress + Cancel for downloads (no pause/resume)
  - Quantization at download time (save quantized weights)
  - Memory estimation + verification before load
  - Graceful unload (wait for pending requests)
  - Multiple models on disk, one loaded at a time
  - Hard delete (remove files from disk)
- **Files Created:** 0xcc/prds/001_FPRD|Model_Management.md
- **Files Modified:** CLAUDE.md
- **Next:** Create Model Management TDD using `0xcc/instruct/004_create-tdd.md`

### Session: February 1, 2026 - Zustand Reactivity Bug Fix
- **Issue:** Steering page UI not updating when clicking "Enable" despite backend succeeding (toast showed success)
- **Root Cause:** Zustand computed getters (`get steeringState()`, `get monitoringConfig()`, `get systemMetrics()`) don't trigger React re-renders when underlying data changes
- **Solution:** Changed all pages to use direct store properties instead of computed getters
- **Files Modified:**
  - `admin-ui/src/pages/SteeringPage.tsx` - Changed `steeringState` → `steering`
  - `admin-ui/src/pages/MonitoringPage.tsx` - Changed `monitoringConfig` → `monitoring`
  - `admin-ui/src/pages/DashboardPage.tsx` - Changed `systemMetrics` → direct properties
  - `admin-ui/src/pages/ProfilesPage.tsx` - Changed `steeringState` → `steering`
  - `admin-ui/src/stores/serverStore.ts` - Removed unused getters and SystemMetrics interface
  - `admin-ui/src/hooks/useSAE.ts` - Fixed race condition with staleTime, removed improper null-setting
  - `admin-ui/src/hooks/useModels.ts` - Added staleTime to prevent refetch issues
  - `admin-ui/src/hooks/useSteering.ts` - Added staleTime for consistency
- **Key Learning:** Never use computed getters in Zustand stores for values that need to trigger re-renders. Always access state properties directly.
- **Related Fix:** Also fixed state persistence issue where model/SAE state was being cleared when useSAE query refetched

### Session: February 4-7, 2026 - Open WebUI Integration & Comprehensive Audit
- **Accomplished:**
  - Connected Open WebUI (K8s on 192.168.244.61) to miLLM (192.168.224.222)
  - Fixed model name returning "1" instead of "gemma-2-2b" in /v1/models
  - Implemented AUTO_LOAD_MODEL for automatic model loading on startup
  - Fixed SSE streaming double data: prefix issue (EventSourceResponse → StreamingResponse)
  - Fixed Gemma chat template fallback (system message turn markers)
  - Added proper eos_token_id to generate() calls
  - Downloaded gemma-2-2b-it (instruction-tuned) model
  - Added delete button for failed SAE downloads in UI
  - **Comprehensive OpenAI API spec audit (15 issues found, all fixed)**:
    - Error format routing (OpenAI format for /v1/*, management format for /api/*)
    - finish_reason "length" detection, stop sequence enforcement in streaming
    - frequency_penalty/presence_penalty mapped to repetition_penalty
    - Model name validation, n parameter, encoding_format for embeddings
    - Thread error propagation in streaming, SSE error events
    - Tool/function role support in ChatMessage
  - **Full 7-feature audit against PRDs (23 issues found, all fixed)**:
    - HIGH: Monitoring wired to inference, streaming stop sequences, steering WS events, HF_TOKEN for SAE downloads
    - MEDIUM: owned_by fix, context length validation, profile parameter, steering range validation, slider precision, GPU memory check, graceful detach/unload, profile export/import endpoints, local model paths
    - LOW: asyncio deprecation fix, console.error removal
  - Updated task list 007 (Admin UI) - marked all implemented tasks complete
  - Updated PRDs with undocumented features
- **Key Decisions:**
  - SAEs trained on base model (gemma-2-2b) don't work with instruction-tuned (gemma-2-2b-it)
  - Steering strength range: -200 to +200 (Neuronpedia compatible, typical: +/-50-100)
  - Direct residual stream steering (miStudio/Neuronpedia compatible)
- **Files Modified:** 20+ files across backend, frontend, and documentation
- **Commits:** 8 commits covering SSE fix, chat template, OpenAI spec compliance, audit fixes

### Session: May 11, 2026 - Inference Optimisation, Test Suite Fixes, Security & Code Review

**Accomplished:**

**Inference Performance (perf commit):**
- Re-enabled speculative decoding with SAE attached (was incorrectly disabled — the hook fires correctly on the main model's verification pass)
- CBM routing safety: `sampling_params_match()` added; requests with wrong temperature/top_p fall back to serial path
- `CBM_FORCE_SERIAL_MONITORING` setting: when enabled, monitored requests bypass CBM for accurate per-request activation attribution
- `TORCH_COMPILE=None` (auto-detect): enabled for CUDA + non-bitsandbytes models
- SAE per-batch activation storage: `_last_feature_acts_per_item` list enables per-request routing in CBM batches

**Security Hardening (security commit):**
- CORS: removed `allow_credentials=True` (no auth → unnecessary and dangerous with wildcard origin)
- Socket.IO CORS now reads `settings.CORS_ORIGINS` instead of hardcoded `"*"`
- `local_path`: blocks /proc, /sys, /dev, /etc and other system dirs
- SAE `file_path`: rejects `..` components
- Batch steering: capped at 1000 features; batch endpoint now enforces [-200, 200] value range
- SAE request schemas: `hf_token` added `Field(exclude=True)`
- Health endpoint: replaced `str(e)` with generic message in component health responses
- structlog API fix: `isEnabledFor` → `is_enabled_for`
- GatedRepoError ordering fix in `get_model_info` (subclass caught by wrong handler)
- `sae_downloader` path concatenation bug: `cache_dir / "models--" + name / "snapshots"` was a TypeError

**Test Suite Fixes (85 pre-existing failures → 0 failures):**
- Module-level `import torch` in memory_utils.py and model_loader.py for test patch targets
- LoadedModel/ModelLoadContext constructor: added `model_name` arg to all test call sites
- HuggingFace API: `HfHubHTTPError` now requires `response=` keyword
- Integration test infrastructure: conftest files redirect `/app` paths to tmp_path; mock DB session
- Multiple mock setup fixes (MagicMock name param, GPT-2 config spec, request queue overflow timing, circuit breaker isolation)

**Business Logic Review (/review multi-agent analysis → fix commit):**
- `MonitoringService` stale DB session (CRITICAL): singleton captured first request's SAE service with closed DB session; now uses `AttachedSAEState` directly
- `QueueFullError` in streaming: pre-check added before `StreamingResponse` to return proper HTTP status
- Prefix cache: flushed on profile activation/deactivation when steering changes
- Profile activation: activating empty-steering profile now clears existing steering explicitly
- Profile import: rejects invalid entries with 400 instead of silently dropping
- Auto-load failures: escalated to `error` level with operator hints
- Orphaned code: `SteeringError`, `InvalidFeatureIndexError`, `InvalidSteeringValueError` removed
- Service→API layer violation: `ModelService` and `SAEService` now accept `inference_service` param

**New Tests (+82):**
- `tests/unit/ml/test_prefix_cache.py` — LRU eviction, invalidation, steering hash
- `tests/unit/ml/test_generation_config.py` — from_request mapping, to_generate_kwargs
- `tests/unit/core/test_config.py` — defaults, env override, cors_origins_list

**Key Numbers:**
- Tests: 705 passing (was 623 — +82 new tests), 0 failures (was 85 failures)
- Commits: 4 clean commits (perf, security, test-fixes, review-fixes)
- Files changed: ~40 production + test files across all concerns

*[Add new sessions as they occur]*

## Research Integration

### MCP Research Support
When available, the framework supports research integration via:
```bash
# Use MCP ref server for contextual research
/mcp ref search "[context-specific query]"

# Research is integrated into all instruction documents as option B
# Example: "🔍 Research first: Use /mcp ref search 'MVP development timeline'"
```

### Research History Tracking
- Research queries and findings captured in session transcripts
- Key research decisions documented in session state
- Research context preserved across sessions for consistency

## Quick Reference

### 0xcc Folder Structure
```
project-root/
├── CLAUDE.md                       # This file (project memory)
├── 0xcc/                           # XCC Framework directory
│   ├── adrs/                       # Architecture Decision Records
│   ├── docs/                       # Additional documentation
│   ├── instruct/                   # Framework instruction files
│   ├── prds/                       # Product Requirements Documents
│   ├── tasks/                      # Task Lists
│   ├── tdds/                       # Technical Design Documents
│   ├── tids/                       # Technical Implementation Documents
│   ├── transcripts/                # Session transcripts
│   ├── checkpoints/                # Automated state backups
│   ├── scripts/                    # Optional automation scripts
│   ├── session_state.json          # Current session tracking
│   └── research_context.json       # Research history and context
├── src/                            # Your project code
├── tests/                          # Your project tests
└── README.md                       # Project README
```

### File Naming Convention
- **Project Level:** `000_PPRD|ProjectName.md`, `000_PADR|ProjectName.md`
- **Feature Level:** `001_FPRD|FeatureName.md`, `001_FTDD|FeatureName.md`, etc.
- **Sequential:** Use 001, 002, 003... for features in priority order
- **Framework Files:** All in `0xcc/` directory for clear organization
- **Project Files:** Standard locations (src/, tests/, package.json, etc.)

### Emergency Contacts & Resources
- **Framework Documentation:** `0xcc/instruct/000_README.md`
- **Current Project PRD:** `0xcc/prds/000_PPRD|miLLM.md`
- **Tech Standards:** `0xcc/adrs/000_PADR|miLLM.md`
- **BRD Reference:** `0xcc/docs/miLLM_BRD_v1.0.md`
- **UI Mockup:** `0xcc/spec/miLLM_UI.jsx`
- **Housekeeping Guide:** `0xcc/instruct/008_housekeeping.md`

---

**Framework Version:** 1.1
**Last Updated:** February 1, 2026
**Project Started:** January 30, 2026
**Structure:** 0xcc framework with MCP research integration