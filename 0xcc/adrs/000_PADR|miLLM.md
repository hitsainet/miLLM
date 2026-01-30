# Architecture Decision Record: miLLM

## Mechanistic Interpretability LLM Server

**Document Version:** 1.0
**Created:** January 30, 2026
**Status:** Approved
**Reference:** Project PRD (000_PPRD|miLLM.md)

---

## 1. Decision Summary

### Date and Context
- **Decision Date:** January 30, 2026
- **Project Phase:** Foundation
- **Decision Makers:** Project stakeholders
- **Alignment:** Architectural decisions aligned with miStudio for ecosystem consistency

### Key Architectural Decisions Overview

| Area | Decision | Rationale |
|------|----------|-----------|
| Frontend | React + Tailwind CSS | Matches UI mockup, large ecosystem, miStudio alignment |
| Backend | Python/FastAPI | Required for PyTorch/Transformers ecosystem |
| Database | PostgreSQL + Redis | miStudio alignment, production-ready |
| State Management | Zustand | Lightweight, miStudio pattern consistency |
| Real-time | REST + WebSocket (Socket.IO) | REST for CRUD, WebSocket for monitoring streams |
| Testing | Balanced pyramid | 80%+ backend coverage, Vitest frontend, Playwright E2E |
| Code Organization | Layer-based | Clear separation of concerns |

### Decision-Making Criteria
1. **miStudio Compatibility:** Align architecture for future integration
2. **PyTorch Ecosystem:** Backend must support Transformers, SAELens, bitsandbytes
3. **Developer Experience:** Clear patterns, good tooling, maintainability
4. **Quality Focus:** Thorough testing, comprehensive documentation
5. **Performance:** Real-time monitoring with minimal overhead

---

## 2. Technology Stack Decisions

### 2.1 Frontend Stack

#### Primary Framework: React 18+
**Rationale:**
- Matches existing UI mockup (0xcc/spec/miLLM_UI.jsx)
- Large ecosystem with mature tooling
- Consistent with miStudio architecture
- Excellent support for complex state and real-time updates

**Key Libraries:**
```json
{
  "react": "^18.2.0",
  "react-dom": "^18.2.0",
  "react-router-dom": "^6.x",
  "zustand": "^4.x",
  "@tanstack/react-query": "^5.x",
  "socket.io-client": "^4.x",
  "lucide-react": "^0.x"
}
```

#### Styling: Tailwind CSS
**Rationale:**
- Utility-first approach enables rapid UI development
- Consistent design system across components
- Easy to match the dark theme from mockup
- Excellent IDE support and documentation

**Configuration:**
- Custom color palette matching mockup (cyan accents, slate backgrounds)
- JetBrains Mono for monospace text
- Inter for UI text
- Custom animations for status indicators

#### State Management: Zustand
**Rationale:**
- Lightweight and simple API
- No boilerplate compared to Redux
- Excellent TypeScript support
- Consistent with miStudio patterns

**Store Organization:**
```
src/stores/
├── modelStore.ts      # Model loading, status, memory
├── saeStore.ts        # SAE management, attachment status
├── steeringStore.ts   # Feature adjustments, active steering
├── profileStore.ts    # Saved profiles, active profile
├── monitorStore.ts    # Real-time activations, monitoring config
└── serverStore.ts     # Server status, system metrics
```

#### Build Tools
- **Vite** for development and bundling (fast HMR, ESM-native)
- **TypeScript** for type safety
- **ESLint + Prettier** for code quality

### 2.2 Backend Stack

#### Server Framework: FastAPI
**Rationale:**
- Required for PyTorch/Transformers ecosystem (Python)
- Automatic OpenAPI documentation
- Native async support for concurrent requests
- Excellent performance for Python
- Type hints with Pydantic validation

**Key Dependencies:**
```python
# Core
fastapi>=0.109.0
uvicorn[standard]>=0.27.0
pydantic>=2.0

# ML/AI
torch>=2.0
transformers>=4.36.0
sae-lens>=0.x  # or compatible SAE library
bitsandbytes>=0.42.0
huggingface-hub>=0.20.0

# Database
sqlalchemy>=2.0
asyncpg>=0.29.0
redis>=5.0
alembic>=1.13.0

# Real-time
python-socketio>=5.10.0

# Utilities
python-dotenv>=1.0.0
structlog>=24.1.0
```

#### API Design: Two-Surface Architecture

**1. OpenAI-Compatible Inference API**
- Base path: `/v1/`
- Endpoints: `/v1/chat/completions`, `/v1/completions`, `/v1/models`, `/v1/embeddings`
- Protocol: REST + SSE (Server-Sent Events for streaming)
- Authentication: None for v1.0 (local trust model)

**2. miLLM Management API**
- Base path: `/api/`
- Protocol: REST + WebSocket (Socket.IO)
- Functions: Model management, SAE management, steering, profiles, monitoring

#### Background Processing
- **asyncio** for async operations within FastAPI
- **Threading** for model loading operations (CPU-bound)
- **Queue** for request management

### 2.3 Database & Data

#### Primary Database: PostgreSQL 14+
**Rationale:**
- miStudio alignment for ecosystem consistency
- Production-ready, ACID compliant
- Excellent JSON support for flexible configurations
- Strong ORM support (SQLAlchemy)

**Schema Overview:**
```sql
-- Core entities
models          -- Downloaded model metadata
saes            -- Downloaded SAE metadata
profiles        -- Steering configuration profiles
feature_labels  -- Cached feature labels/descriptions

-- Operational
server_config   -- Runtime configuration
request_logs    -- Optional request/response logging
```

#### Caching: Redis 7+
**Rationale:**
- miStudio alignment
- Fast in-memory storage for:
  - Session state
  - Real-time activation buffers
  - Request queue management
  - Server status caching

#### Data Patterns
- **ORM:** SQLAlchemy 2.0 with async support
- **Migrations:** Alembic for schema versioning
- **JSON Storage:** PostgreSQL JSONB for flexible profile data
- **File Storage:** Local filesystem for model/SAE caches (configurable path)

### 2.4 Infrastructure & Deployment

#### Container Strategy: Docker
**Primary Deployment:**
```yaml
# docker-compose.yml structure
services:
  millm:
    build: .
    runtime: nvidia  # GPU passthrough
    ports:
      - "8000:8000"  # API
      - "3000:3000"  # Admin UI
    volumes:
      - ./data:/app/data  # Model/SAE cache
      - ./config:/app/config
    environment:
      - DATABASE_URL
      - REDIS_URL
      - HF_TOKEN

  postgres:
    image: postgres:14

  redis:
    image: redis:7
```

**Development Mode:**
```bash
# Direct Python execution
pip install -e .
python -m millm.server
```

#### Environment Configuration
Following 12-factor app principles:
```bash
# Required
DATABASE_URL=postgresql+asyncpg://user:pass@localhost/millm
REDIS_URL=redis://localhost:6379

# Optional
HF_TOKEN=hf_xxxx           # For gated models
MODEL_CACHE_DIR=/path/to/cache
LOG_LEVEL=INFO
LOG_FORMAT=json            # or "console" for development
```

#### Monitoring & Logging
- **Structured Logging:** structlog with JSON output (production) or console (development)
- **Metrics:** Built-in `/api/health` and `/api/metrics` endpoints
- **GPU Monitoring:** nvidia-smi integration for VRAM/utilization

---

## 3. Development Standards

### 3.1 Code Organization

#### Backend Directory Structure (Layer-Based)
```
millm/
├── api/                    # API layer
│   ├── routes/
│   │   ├── openai/         # OpenAI-compatible endpoints
│   │   │   ├── chat.py
│   │   │   ├── completions.py
│   │   │   ├── models.py
│   │   │   └── embeddings.py
│   │   ├── management/     # miLLM Management API
│   │   │   ├── models.py
│   │   │   ├── saes.py
│   │   │   ├── steering.py
│   │   │   ├── profiles.py
│   │   │   └── monitor.py
│   │   └── system/         # Health, metrics
│   ├── middleware/
│   ├── dependencies.py
│   └── exceptions.py
├── services/               # Business logic
│   ├── model_service.py
│   ├── sae_service.py
│   ├── steering_service.py
│   ├── profile_service.py
│   ├── monitor_service.py
│   └── inference_service.py
├── ml/                     # ML-specific code
│   ├── model_loader.py
│   ├── sae_loader.py
│   ├── hooks.py            # SAE hooking mechanism
│   ├── steering.py         # Feature steering logic
│   └── quantization.py
├── db/                     # Database layer
│   ├── models/             # SQLAlchemy models
│   ├── repositories/       # Data access patterns
│   └── migrations/         # Alembic migrations
├── sockets/                # WebSocket handlers
│   ├── monitor.py
│   └── events.py
├── core/                   # Core utilities
│   ├── config.py
│   ├── logging.py
│   └── errors.py
└── main.py                 # Application entry point
```

#### Frontend Directory Structure
```
src/
├── components/             # React components
│   ├── common/             # Shared components
│   │   ├── Button.tsx
│   │   ├── Card.tsx
│   │   ├── Input.tsx
│   │   └── Badge.tsx
│   ├── layout/             # Layout components
│   │   ├── Header.tsx
│   │   ├── StatusBar.tsx
│   │   └── Navigation.tsx
│   ├── models/             # Model management components
│   ├── saes/               # SAE management components
│   ├── steering/           # Steering components
│   ├── profiles/           # Profile components
│   └── monitor/            # Monitoring components
├── pages/                  # Route pages
│   ├── ModelsPage.tsx
│   ├── SAEsPage.tsx
│   ├── SteeringPage.tsx
│   ├── ProfilesPage.tsx
│   └── MonitorPage.tsx
├── stores/                 # Zustand stores
├── services/               # API client services
│   ├── api.ts              # Base API client
│   ├── modelService.ts
│   ├── saeService.ts
│   └── socketService.ts
├── hooks/                  # Custom React hooks
│   ├── useModels.ts
│   ├── useSAEs.ts
│   ├── useSteering.ts
│   └── useMonitor.ts
├── types/                  # TypeScript types
│   ├── model.ts
│   ├── sae.ts
│   ├── profile.ts
│   └── api.ts
├── utils/                  # Utility functions
├── styles/                 # Global styles
│   └── globals.css         # Tailwind imports
├── App.tsx
└── main.tsx
```

#### File Naming Conventions
| Type | Pattern | Example |
|------|---------|---------|
| React Component | PascalCase.tsx | `ModelCard.tsx` |
| React Page | PascalCase.tsx | `ModelsPage.tsx` |
| Zustand Store | camelCase.ts | `modelStore.ts` |
| Service | camelCase.ts | `modelService.ts` |
| Hook | useCamelCase.ts | `useModels.ts` |
| Python Module | snake_case.py | `model_service.py` |
| Python Class | PascalCase | `ModelService` |
| Test File | *.test.ts / test_*.py | `ModelCard.test.tsx`, `test_model_service.py` |

### 3.2 Coding Patterns

#### Backend Patterns

**Service Layer Pattern:**
```python
# services/model_service.py
class ModelService:
    def __init__(self, db: AsyncSession, cache: Redis):
        self.db = db
        self.cache = cache
        self._loaded_model: Optional[LoadedModel] = None

    async def download_model(self, repo_id: str, quantization: str) -> Model:
        """Download model from HuggingFace."""
        # Implementation

    async def load_model(self, model_id: int) -> LoadedModel:
        """Load model into GPU memory."""
        # Implementation

    async def unload_model(self) -> None:
        """Unload current model from memory."""
        # Implementation
```

**Dependency Injection:**
```python
# api/dependencies.py
async def get_model_service(
    db: AsyncSession = Depends(get_db),
    cache: Redis = Depends(get_redis)
) -> ModelService:
    return ModelService(db, cache)
```

**Error Handling:**
```python
# core/errors.py
class MiLLMError(Exception):
    """Base error for miLLM."""
    def __init__(self, message: str, code: str, details: dict = None):
        self.message = message
        self.code = code
        self.details = details or {}

class ModelNotLoadedError(MiLLMError):
    """Raised when operation requires loaded model."""
    def __init__(self, message: str = "No model currently loaded"):
        super().__init__(message, "MODEL_NOT_LOADED")

class SAENotAttachedError(MiLLMError):
    """Raised when operation requires attached SAE."""
    pass
```

**Pydantic Schemas:**
```python
# api/schemas/model.py
class ModelDownloadRequest(BaseModel):
    repo_id: str = Field(..., example="google/gemma-2-2b")
    quantization: Literal["Q4", "Q8", "FP16"] = "Q4"
    trust_remote_code: bool = False
    hf_token: Optional[str] = None

class ModelResponse(BaseModel):
    id: int
    name: str
    repo_id: str
    params: str
    quantization: str
    memory_mb: int
    status: Literal["ready", "loaded", "downloading"]

    model_config = ConfigDict(from_attributes=True)
```

#### Frontend Patterns

**Zustand Store Pattern:**
```typescript
// stores/modelStore.ts
interface ModelState {
  models: Model[];
  loadedModel: Model | null;
  isLoading: boolean;
  error: string | null;

  // Actions
  fetchModels: () => Promise<void>;
  loadModel: (id: number) => Promise<void>;
  unloadModel: () => Promise<void>;
  downloadModel: (request: ModelDownloadRequest) => Promise<void>;
}

export const useModelStore = create<ModelState>((set, get) => ({
  models: [],
  loadedModel: null,
  isLoading: false,
  error: null,

  fetchModels: async () => {
    set({ isLoading: true, error: null });
    try {
      const models = await modelService.getModels();
      set({ models, isLoading: false });
    } catch (error) {
      set({ error: error.message, isLoading: false });
    }
  },
  // ... other actions
}));
```

**Custom Hook Pattern:**
```typescript
// hooks/useModels.ts
export function useModels() {
  const { models, loadedModel, isLoading, error, fetchModels, loadModel } =
    useModelStore();

  useEffect(() => {
    fetchModels();
  }, []);

  return {
    models,
    loadedModel,
    isLoading,
    error,
    loadModel,
    isModelLoaded: loadedModel !== null,
  };
}
```

**Component Pattern:**
```typescript
// components/models/ModelCard.tsx
interface ModelCardProps {
  model: Model;
  onLoad: (id: number) => void;
  onUnload: () => void;
  onDelete: (id: number) => void;
}

export function ModelCard({ model, onLoad, onUnload, onDelete }: ModelCardProps) {
  const isLoaded = model.status === 'loaded';

  return (
    <Card className="flex items-center justify-between p-4">
      <div className="flex items-center gap-4">
        <div className="w-10 h-10 bg-cyan-500/10 rounded-lg flex items-center justify-center">
          <Server className="w-5 h-5 text-cyan-400" />
        </div>
        <div>
          <h3 className="font-semibold text-slate-100">{model.name}</h3>
          <p className="text-sm text-slate-400">
            {model.params} • {model.quantization} • {model.memory_mb} MB
          </p>
        </div>
      </div>
      <div className="flex items-center gap-2">
        <Badge variant={isLoaded ? 'success' : 'default'}>
          {model.status}
        </Badge>
        {isLoaded ? (
          <Button variant="destructive" size="sm" onClick={onUnload}>
            <Square className="w-4 h-4 mr-1" /> Unload
          </Button>
        ) : (
          <Button variant="primary" size="sm" onClick={() => onLoad(model.id)}>
            <Play className="w-4 h-4 mr-1" /> Load
          </Button>
        )}
      </div>
    </Card>
  );
}
```

### 3.3 Quality Assurance

#### Testing Strategy: Balanced Pyramid

**Backend Testing (pytest):**
- **Target Coverage:** 80%+
- **Unit Tests:** Services, utilities, ML logic
- **Integration Tests:** API endpoints, database operations
- **Fixtures:** Shared test fixtures for models, SAEs, profiles

```python
# tests/conftest.py
@pytest.fixture
async def db_session():
    """Provide test database session."""
    async with async_session() as session:
        yield session
        await session.rollback()

@pytest.fixture
def mock_model():
    """Provide mock loaded model for testing."""
    return MockLoadedModel(name="test-model", params="1B")

# tests/services/test_model_service.py
class TestModelService:
    async def test_download_model_success(self, db_session, mocker):
        mocker.patch('huggingface_hub.snapshot_download')
        service = ModelService(db_session)
        model = await service.download_model("test/model", "Q4")
        assert model.status == "ready"

    async def test_load_model_insufficient_memory(self, db_session, mock_model):
        # Test graceful failure
        pass
```

**Frontend Testing (Vitest + React Testing Library):**
- **Unit Tests:** Stores, utilities, hooks
- **Component Tests:** Component rendering and interactions
- **Integration Tests:** Page-level workflows

```typescript
// src/components/models/ModelCard.test.tsx
describe('ModelCard', () => {
  it('renders model information correctly', () => {
    const model = createMockModel({ name: 'gemma-2-2b', status: 'ready' });
    render(<ModelCard model={model} onLoad={vi.fn()} onUnload={vi.fn()} />);

    expect(screen.getByText('gemma-2-2b')).toBeInTheDocument();
    expect(screen.getByText('Load')).toBeInTheDocument();
  });

  it('shows unload button when model is loaded', () => {
    const model = createMockModel({ status: 'loaded' });
    render(<ModelCard model={model} onLoad={vi.fn()} onUnload={vi.fn()} />);

    expect(screen.getByText('Unload')).toBeInTheDocument();
  });
});
```

**E2E Testing (Playwright):**
- **Critical Paths:** Model download → load → SAE attach → steering workflow
- **Smoke Tests:** All pages load, basic navigation
- **API Integration:** OpenAI client compatibility

```typescript
// e2e/steering-workflow.spec.ts
test('complete steering workflow', async ({ page }) => {
  // Navigate to models page
  await page.goto('/models');

  // Load a model
  await page.click('[data-testid="load-model-1"]');
  await expect(page.locator('[data-testid="model-status"]')).toHaveText('Loaded');

  // Navigate to SAEs and attach
  await page.click('text=SAEs');
  await page.click('[data-testid="attach-sae-1"]');

  // Navigate to steering and adjust
  await page.click('text=Steering');
  await page.fill('[data-testid="feature-1234-slider"]', '5');

  // Verify steering is active
  await expect(page.locator('[data-testid="steering-status"]')).toHaveText('Active');
});
```

#### Code Review Standards
- All PRs require at least one review
- Tests must pass before merge
- No decrease in coverage allowed
- Lint and format checks in CI

#### Continuous Integration
```yaml
# .github/workflows/ci.yml
name: CI
on: [push, pull_request]
jobs:
  backend:
    runs-on: ubuntu-latest
    services:
      postgres:
        image: postgres:14
      redis:
        image: redis:7
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      - run: pip install -e ".[dev]"
      - run: pytest --cov=millm --cov-fail-under=80
      - run: ruff check .
      - run: mypy millm/

  frontend:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-node@v4
        with:
          node-version: '20'
      - run: npm ci
      - run: npm run lint
      - run: npm run typecheck
      - run: npm run test:coverage
      - run: npm run build

  e2e:
    runs-on: ubuntu-latest
    needs: [backend, frontend]
    steps:
      - uses: actions/checkout@v4
      - run: docker-compose up -d
      - run: npx playwright test
```

### 3.4 Development Workflow

#### Version Control: Git
- **Branching Model:** GitHub Flow (feature branches → main)
- **Branch Naming:** `feature/`, `fix/`, `docs/`, `refactor/`
- **Commit Format:** Conventional Commits

```
feat(models): add quantization selection to download form
fix(steering): correct activation scaling for negative values
docs(api): add OpenAPI examples for chat endpoint
refactor(services): extract common HuggingFace download logic
```

#### Development Environment Setup
```bash
# Clone and setup
git clone https://github.com/org/millm.git
cd millm

# Backend setup
python -m venv venv
source venv/bin/activate
pip install -e ".[dev]"

# Frontend setup
cd frontend
npm install

# Database setup
docker-compose up -d postgres redis
alembic upgrade head

# Run development servers
# Terminal 1: Backend
uvicorn millm.main:app --reload

# Terminal 2: Frontend
npm run dev
```

#### Package Management
- **Backend:** pip with pyproject.toml, pin major versions
- **Frontend:** npm with package-lock.json, pin major versions
- **Updates:** Monthly dependency review, security patches immediate

---

## 4. Architectural Principles

### Core Design Principles

1. **Separation of Concerns**
   - Clear boundaries between API, service, and data layers
   - Frontend and backend independently deployable
   - ML code isolated from web framework code

2. **Fail Fast, Fail Gracefully**
   - Validate inputs at API boundaries
   - Clear error messages with recovery guidance
   - Graceful degradation (e.g., disable SAE if OOM, continue with base model)

3. **Configuration over Code**
   - Environment variables for deployment settings
   - Database-stored profiles for user configurations
   - No hardcoded paths or credentials

4. **Stateless API Design**
   - Server state in database/Redis, not in memory
   - Exception: Loaded model (single-user assumption for v1.0)
   - All configuration retrievable via API

### Scalability Considerations

**v1.0 (Single-User):**
- One model loaded at a time
- One SAE attached at a time
- Request queue for concurrent requests
- Local file caching for models/SAEs

**Future Scaling Path:**
- Multiple model instances (if multi-GPU)
- Multi-layer SAE support
- User authentication and isolation
- Horizontal scaling for API servers (separate from inference)

### Security Requirements

**v1.0 (Local Trust Model):**
- No authentication (assumes trusted local network)
- Input validation on all API endpoints
- No arbitrary code execution (trust_remote_code requires explicit flag)
- HF_TOKEN stored securely (env var, not in database)

**Architecture for Future Auth:**
- API key middleware ready to activate
- User isolation patterns in database schema
- Rate limiting infrastructure in place

### Maintainability Standards

- **Documentation:** Docstrings required for public APIs
- **Type Hints:** Required for Python (mypy strict), TypeScript strict mode
- **Logging:** Structured logs at service boundaries
- **Metrics:** Key operations instrumented for performance tracking

---

## 5. Package and Library Standards

### Approved Libraries

#### Backend (Python)
| Category | Library | Version | Purpose |
|----------|---------|---------|---------|
| Web Framework | FastAPI | ^0.109 | API server |
| ASGI Server | Uvicorn | ^0.27 | Production server |
| Validation | Pydantic | ^2.0 | Request/response schemas |
| ORM | SQLAlchemy | ^2.0 | Database access |
| Migrations | Alembic | ^1.13 | Schema migrations |
| Cache | redis-py | ^5.0 | Redis client |
| WebSocket | python-socketio | ^5.10 | Real-time communication |
| ML Framework | PyTorch | ^2.0 | Model inference |
| Transformers | transformers | ^4.36 | Model loading |
| Quantization | bitsandbytes | ^0.42 | 4-bit/8-bit quantization |
| HuggingFace | huggingface-hub | ^0.20 | Model downloads |
| SAE | sae-lens | ^0.x | SAE loading and operations |
| Logging | structlog | ^24.1 | Structured logging |
| Testing | pytest | ^8.0 | Test framework |
| Linting | ruff | ^0.1 | Fast Python linter |
| Type Checking | mypy | ^1.8 | Static type checking |

#### Frontend (TypeScript/React)
| Category | Library | Version | Purpose |
|----------|---------|---------|---------|
| UI Framework | React | ^18.2 | Component framework |
| Routing | react-router-dom | ^6.x | Client-side routing |
| State | Zustand | ^4.x | Global state management |
| Server State | @tanstack/react-query | ^5.x | API data caching |
| WebSocket | socket.io-client | ^4.x | Real-time updates |
| Icons | lucide-react | ^0.x | Icon library |
| Styling | Tailwind CSS | ^3.4 | Utility CSS |
| HTTP Client | ky | ^1.x | Fetch wrapper |
| Forms | react-hook-form | ^7.x | Form management |
| Testing | Vitest | ^1.x | Unit testing |
| Testing | @testing-library/react | ^14.x | Component testing |
| E2E | Playwright | ^1.x | End-to-end testing |
| Build | Vite | ^5.x | Build tool |
| Linting | ESLint | ^8.x | Code linting |
| Formatting | Prettier | ^3.x | Code formatting |

### Package Selection Criteria
1. **Active Maintenance:** Updated within last 6 months
2. **Community Adoption:** Significant GitHub stars/downloads
3. **TypeScript Support:** First-class types or @types package
4. **Bundle Size:** Consider bundle impact for frontend
5. **License:** MIT, Apache 2.0, or similarly permissive

### Version Management
- Pin major versions in requirements/package.json
- Monthly review for security updates
- Test thoroughly before major version upgrades
- Document breaking changes in CHANGELOG

---

## 6. Integration Guidelines

### API Design Standards

#### REST Conventions
```
GET    /api/models              # List all models
POST   /api/models              # Download new model
GET    /api/models/{id}         # Get model details
DELETE /api/models/{id}         # Delete model
POST   /api/models/{id}/load    # Load model into memory
POST   /api/models/{id}/unload  # Unload model from memory
```

#### Response Format
```json
{
  "success": true,
  "data": { ... },
  "error": null
}

// Error response
{
  "success": false,
  "data": null,
  "error": {
    "code": "MODEL_NOT_FOUND",
    "message": "Model with ID 123 not found",
    "details": {}
  }
}
```

#### WebSocket Events (Socket.IO)
```typescript
// Namespaces
/monitor     # Feature activation monitoring
/system      # System metrics (GPU, memory)
/progress    # Download/load progress

// Events
monitor:subscribe    # Subscribe to feature activations
monitor:activations  # Activation data stream
system:metrics       # GPU/memory metrics
progress:update      # Download/load progress
```

### Data Exchange Formats

#### Profile Format (miStudio Compatible)
```json
{
  "version": "1.0",
  "name": "yelling-demo",
  "description": "Demonstrates capitalization steering",
  "model": {
    "repo_id": "google/gemma-2-2b",
    "quantization": "Q4"
  },
  "sae": {
    "repo_id": "google/gemma-scope-2b-pt-res",
    "layer": 12
  },
  "features": [
    {
      "index": 1234,
      "strength": 5.0,
      "label": "Yelling/Capitalization"
    }
  ],
  "created_at": "2026-01-30T12:00:00Z",
  "updated_at": "2026-01-30T12:00:00Z"
}
```

### Error Handling Standards

#### Error Codes
| Code | HTTP Status | Description |
|------|-------------|-------------|
| MODEL_NOT_FOUND | 404 | Requested model doesn't exist |
| MODEL_NOT_LOADED | 400 | Operation requires loaded model |
| SAE_NOT_ATTACHED | 400 | Operation requires attached SAE |
| INVALID_FEATURE_INDEX | 400 | Feature index out of range |
| INSUFFICIENT_MEMORY | 507 | Not enough GPU memory |
| DOWNLOAD_FAILED | 502 | HuggingFace download failed |
| VALIDATION_ERROR | 422 | Request validation failed |

#### Logging Standards
```python
# Log levels
DEBUG   # Detailed debugging (not in production)
INFO    # Normal operations (requests, completions)
WARNING # Recoverable issues (fallback activated)
ERROR   # Failures requiring attention

# Log format (production)
{
  "timestamp": "2026-01-30T12:00:00Z",
  "level": "INFO",
  "message": "Model loaded successfully",
  "context": {
    "model_id": 1,
    "model_name": "gemma-2-2b",
    "load_time_ms": 4523
  }
}
```

---

## 7. Development Environment

### Required Tools
| Tool | Version | Purpose |
|------|---------|---------|
| Python | 3.11+ | Backend runtime |
| Node.js | 20+ | Frontend tooling |
| Docker | 24+ | Containerization |
| Docker Compose | 2.x | Local services |
| Git | 2.x | Version control |
| NVIDIA Driver | 535+ | GPU support |
| CUDA | 12.x | GPU acceleration |

### IDE Recommendations
- **VSCode** with extensions:
  - Python (ms-python)
  - Pylance (type checking)
  - Ruff (linting)
  - ES7+ React snippets
  - Tailwind CSS IntelliSense
  - Docker

### Local Development Configuration
```bash
# .env.development
DATABASE_URL=postgresql+asyncpg://postgres:postgres@localhost:5432/millm
REDIS_URL=redis://localhost:6379
LOG_LEVEL=DEBUG
LOG_FORMAT=console
MODEL_CACHE_DIR=./data/models
SAE_CACHE_DIR=./data/saes
```

### Debugging Tools
- **Backend:** debugpy, pdb, VSCode debugger
- **Frontend:** React DevTools, Zustand DevTools
- **Network:** Postman/Insomnia for API testing
- **Database:** pgAdmin, DBeaver

---

## 8. Security Standards

### Input Validation
- All API inputs validated via Pydantic
- File paths sanitized (no path traversal)
- Repository IDs validated against HuggingFace patterns
- Feature indices validated against SAE dimensions

### Secure Coding Practices
- No eval() or exec() on user input
- trust_remote_code explicit opt-in only
- Secrets via environment variables only
- No sensitive data in logs

### Vulnerability Management
- Dependabot enabled for security updates
- Weekly dependency audit (npm audit, pip-audit)
- Security patches applied within 48 hours
- CVE monitoring for core dependencies

---

## 9. Performance Guidelines

### Performance Targets
| Metric | Target | Measurement |
|--------|--------|-------------|
| SAE overhead | <15% latency increase | Benchmark vs base model |
| Time to first token | <500ms | After model loaded |
| API response (non-inference) | <100ms | Health check, list endpoints |
| UI interaction | <100ms | Button clicks, navigation |
| Monitoring update | 100ms intervals | Activation streaming |

### Optimization Strategies

**Backend:**
- Async I/O for all database and network operations
- Connection pooling for PostgreSQL and Redis
- Lazy loading for models and SAEs
- Request queuing to prevent overload

**Frontend:**
- Code splitting by route
- Lazy loading for heavy components
- Virtualized lists for large datasets
- Debounced slider inputs

**ML Operations:**
- Quantization by default (Q4 recommended)
- Mixed precision inference
- Activation caching for monitoring
- Batch processing where applicable

### Caching Policies
| Data | Cache Location | TTL | Invalidation |
|------|---------------|-----|--------------|
| Model metadata | PostgreSQL | Permanent | Manual delete |
| Server metrics | Redis | 5s | Auto-expire |
| Activation buffer | Redis | 60s | Rolling window |
| API responses | React Query | 30s | On mutation |

---

## 10. Decision Rationale

### Major Trade-offs

#### PostgreSQL vs SQLite
**Decision:** PostgreSQL
**Trade-off:** More setup complexity vs miStudio alignment and production readiness
**Rationale:** Ecosystem consistency outweighs local simplicity; Docker makes setup trivial

#### Zustand vs Redux
**Decision:** Zustand
**Trade-off:** Less tooling (no Redux DevTools time-travel) vs simplicity
**Rationale:** miLLM's state is simpler than miStudio's; Zustand's minimal API fits well

#### Socket.IO vs Native WebSocket
**Decision:** Socket.IO
**Trade-off:** Larger bundle vs reconnection handling and namespaces
**Rationale:** Built-in reconnection and room management worth the bundle cost

### Alternatives Evaluated

| Decision | Alternative | Why Rejected |
|----------|------------|--------------|
| FastAPI | Flask | Less async support, no automatic docs |
| React | Svelte | Smaller ecosystem, less familiar |
| PostgreSQL | SQLite | Doesn't align with miStudio |
| Tailwind | CSS-in-JS | More familiar to team, faster development |

### Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Socket.IO performance | Low | Medium | Fallback to SSE for monitoring |
| PostgreSQL overhead | Low | Low | SQLite option documented for simple deploys |
| React bundle size | Medium | Low | Code splitting, lazy loading |

---

## 11. Implementation Guidelines

### Applying These Decisions

1. **New Features:** Reference this ADR when starting any feature PRD/TDD
2. **Code Reviews:** Verify adherence to patterns and conventions
3. **Dependencies:** Check approved library list before adding new packages
4. **API Design:** Follow REST conventions and response format
5. **Testing:** Maintain coverage targets, write tests for new code

### Exception Process
1. Document the exception and rationale
2. Discuss in code review
3. Update ADR if exception becomes pattern
4. Track technical debt if temporary

### Documentation Requirements
- Public APIs: Docstrings with examples
- Complex logic: Inline comments explaining "why"
- Configuration: Document all environment variables
- Architecture: Update ADR for significant changes

### Team Onboarding
1. Read Project PRD for context
2. Read this ADR for technical standards
3. Set up local development environment
4. Complete "hello world" task (add simple feature)
5. Pair with experienced team member for first PR

---

## Appendix A: CLAUDE.md Project Standards Section

The following section should be copied into CLAUDE.md to replace the placeholder:

```markdown
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

### Test Commands
- **Backend:** `pytest` or `pytest --cov=millm`
- **Frontend:** `npm test` or `npm run test:coverage`
- **E2E:** `npx playwright test`
- **Lint:** `ruff check .` (backend), `npm run lint` (frontend)
- **Type Check:** `mypy millm/` (backend), `npm run typecheck` (frontend)
```

---

**Document Status:** Approved
**Next Document:** Feature PRDs starting with 001_FPRD|Model_Management.md
**Instruction File:** `@0xcc/instruct/003_create-feature-prd.md`
