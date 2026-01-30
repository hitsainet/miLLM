# Technical Design Document: Admin UI

## miLLM Feature 7

**Document Version:** 1.0
**Created:** January 30, 2026
**Status:** Draft
**References:**
- Feature PRD: `007_FPRD|Admin_UI.md`
- Project ADR: `000_PADR|miLLM.md`

---

## 1. Technical Overview

### 1.1 Summary

The Admin UI is a React-based single-page application that provides a web dashboard for managing the miLLM server. It connects to the backend via REST API and WebSocket (Socket.IO) for real-time updates.

### 1.2 Architecture Pattern

```
┌─────────────────────────────────────────────────────────────┐
│                        Admin UI                              │
├──────────────┬──────────────┬───────────────────────────────┤
│    Pages     │  Components  │           Services            │
│  (Routes)    │   (Shared)   │    (API + WebSocket)          │
├──────────────┼──────────────┼───────────────────────────────┤
│              │              │                               │
│  Dashboard   │   Card       │   apiClient (REST)            │
│  Models      │   Button     │   socketClient (WS)           │
│  SAE         │   Slider     │                               │
│  Steering    │   Modal      │                               │
│  Monitoring  │   Table      │                               │
│  Profiles    │   Toast      │                               │
│  Settings    │   Form       │                               │
│              │              │                               │
├──────────────┴──────────────┼───────────────────────────────┤
│         Zustand Stores      │     Type Definitions          │
│  (serverStore, uiStore)     │   (API types, UI types)       │
└─────────────────────────────┴───────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    miLLM Backend                             │
│  ┌─────────────────────┐    ┌─────────────────────────────┐ │
│  │   REST API          │    │   WebSocket (Socket.IO)     │ │
│  │   /api/*            │    │   Real-time events          │ │
│  └─────────────────────┘    └─────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

### 1.3 Key Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| State Management | Zustand | Simple, lightweight, TypeScript-friendly |
| Styling | Tailwind CSS | Rapid development, consistent design |
| HTTP Client | fetch + React Query | Built-in, caching, automatic refetching |
| WebSocket | socket.io-client | Matches backend, auto-reconnect |
| Routing | React Router v6 | Standard, declarative routing |
| Build Tool | Vite | Fast dev server, optimized builds |

---

## 2. Component Architecture

### 2.1 Component Hierarchy

```
App
├── Layout
│   ├── Sidebar
│   │   ├── Logo
│   │   ├── NavItem (×7)
│   │   └── CollapseToggle
│   ├── Header
│   │   ├── Breadcrumbs
│   │   ├── ServerStatus
│   │   └── ThemeToggle
│   └── MainContent
│       └── <Outlet /> (React Router)
│
├── Pages
│   ├── DashboardPage
│   │   ├── StatusCard (×4)
│   │   └── QuickActions
│   │
│   ├── ModelsPage
│   │   ├── ModelLoadForm
│   │   └── LoadedModelCard
│   │
│   ├── SAEPage
│   │   ├── SAEDownloadForm
│   │   ├── SAEList
│   │   │   └── SAEListItem (×n)
│   │   └── AttachedSAECard
│   │
│   ├── SteeringPage
│   │   ├── SteeringControls
│   │   │   ├── FeatureSearch
│   │   │   ├── SteeringSlider (×n)
│   │   │   └── BatchAddForm
│   │   └── SteeringActions
│   │
│   ├── MonitoringPage
│   │   ├── MonitoringControls
│   │   ├── ActivationChart
│   │   ├── ActivationHistory
│   │   └── StatisticsPanel
│   │
│   ├── ProfilesPage
│   │   ├── ProfileList
│   │   │   └── ProfileListItem (×n)
│   │   ├── ProfileForm
│   │   └── ImportExportButtons
│   │
│   └── SettingsPage
│       ├── ThemeSettings
│       └── ConnectionSettings
│
└── Shared Components
    ├── Button
    ├── Card
    ├── Modal
    ├── Slider
    ├── Input
    ├── Select
    ├── Toast
    ├── Spinner
    ├── Badge
    └── EmptyState
```

### 2.2 Component Design Patterns

**Container/Presentational Pattern**
- Page components handle data fetching and state
- Child components are presentational (props-driven)

**Composition Pattern**
- Cards, Modals, Forms are composable
- Slots for custom content

**Controlled Components**
- All form inputs are controlled
- State lives in parent or Zustand store

---

## 3. State Management

### 3.1 Store Architecture

```typescript
// stores/serverStore.ts - Server state from backend
interface ServerStore {
  // Connection
  isConnected: boolean;
  connectionError: string | null;

  // Model
  model: ModelInfo | null;
  modelLoading: boolean;

  // SAE
  attachedSAE: SAEInfo | null;
  downloadedSAEs: SAEInfo[];
  saeLoading: boolean;

  // Steering
  steeringEnabled: boolean;
  steeringValues: Record<number, number>;

  // Monitoring
  monitoringEnabled: boolean;
  monitoringConfig: MonitoringConfig;
  activations: ActivationRecord[];
  statistics: FeatureStatistics;

  // Profiles
  profiles: Profile[];
  activeProfile: Profile | null;

  // Actions
  setModel: (model: ModelInfo | null) => void;
  setSAE: (sae: SAEInfo | null) => void;
  updateSteering: (feature: number, value: number) => void;
  clearSteering: () => void;
  addActivation: (activation: ActivationRecord) => void;
  // ... more actions
}

// stores/uiStore.ts - UI-only state
interface UIStore {
  // Theme
  theme: 'light' | 'dark';

  // Sidebar
  sidebarCollapsed: boolean;

  // Modals
  activeModal: string | null;
  modalData: unknown;

  // Toasts
  toasts: Toast[];

  // Monitoring UI
  monitoringPaused: boolean;

  // Actions
  setTheme: (theme: 'light' | 'dark') => void;
  toggleSidebar: () => void;
  showModal: (id: string, data?: unknown) => void;
  hideModal: () => void;
  addToast: (toast: Omit<Toast, 'id'>) => void;
  removeToast: (id: string) => void;
}
```

### 3.2 State Flow

```
User Action
    │
    ▼
Component Handler
    │
    ├──► API Call (REST)
    │       │
    │       ▼
    │    Backend Response
    │       │
    │       ▼
    │    Update Zustand Store
    │       │
    │       ▼
    │    Component Re-renders
    │
    └──► WebSocket Event (real-time)
            │
            ▼
         Socket Handler
            │
            ▼
         Update Zustand Store
            │
            ▼
         Component Re-renders
```

### 3.3 Optimistic Updates

For responsive UI, use optimistic updates with rollback:

```typescript
// Example: Steering update
const updateSteering = async (feature: number, value: number) => {
  // 1. Optimistically update store
  const previousValue = store.steeringValues[feature];
  store.updateSteering(feature, value);

  try {
    // 2. Make API call
    await api.setSteering(feature, value);
  } catch (error) {
    // 3. Rollback on failure
    store.updateSteering(feature, previousValue);
    toast.error('Failed to update steering');
  }
};
```

---

## 4. API Integration

### 4.1 API Client Design

```typescript
// services/api.ts
const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000';

class ApiClient {
  private baseUrl: string;

  constructor(baseUrl: string) {
    this.baseUrl = baseUrl;
  }

  private async request<T>(
    method: string,
    path: string,
    body?: unknown
  ): Promise<T> {
    const response = await fetch(`${this.baseUrl}${path}`, {
      method,
      headers: { 'Content-Type': 'application/json' },
      body: body ? JSON.stringify(body) : undefined,
    });

    if (!response.ok) {
      const error = await response.json();
      throw new ApiError(error.detail || 'Request failed', response.status);
    }

    return response.json();
  }

  // Models
  getModel = () => this.request<ModelInfo>('GET', '/api/models');
  loadModel = (req: LoadModelRequest) =>
    this.request<ModelInfo>('POST', '/api/models', req);
  unloadModel = () => this.request<void>('DELETE', '/api/models');

  // SAE
  listSAEs = () => this.request<SAEInfo[]>('GET', '/api/sae');
  downloadSAE = (req: DownloadSAERequest) =>
    this.request<SAEInfo>('POST', '/api/sae/download', req);
  attachSAE = (id: string) =>
    this.request<void>('POST', `/api/sae/${id}/attach`);
  detachSAE = () => this.request<void>('POST', '/api/sae/detach');
  deleteSAE = (id: string) =>
    this.request<void>('DELETE', `/api/sae/${id}`);

  // Steering
  getSteering = () => this.request<SteeringState>('GET', '/api/steering');
  setSteering = (feature: number, value: number) =>
    this.request<void>('POST', '/api/steering', { feature_index: feature, value });
  batchSteering = (values: Record<number, number>) =>
    this.request<void>('POST', '/api/steering/batch', { values });
  clearSteering = () => this.request<void>('DELETE', '/api/steering');
  toggleSteering = (enabled: boolean) =>
    this.request<void>('POST', '/api/steering/toggle', { enabled });

  // Monitoring
  getMonitoring = () => this.request<MonitoringState>('GET', '/api/monitoring');
  configureMonitoring = (config: MonitoringConfig) =>
    this.request<void>('POST', '/api/monitoring/configure', config);
  enableMonitoring = (enabled: boolean) =>
    this.request<void>('POST', '/api/monitoring/enable', { enabled });
  getHistory = () => this.request<ActivationRecord[]>('GET', '/api/monitoring/history');
  clearHistory = () => this.request<void>('DELETE', '/api/monitoring/history');
  getStatistics = () => this.request<FeatureStatistics>('GET', '/api/monitoring/statistics');

  // Profiles
  listProfiles = () => this.request<Profile[]>('GET', '/api/profiles');
  getProfile = (id: string) => this.request<Profile>('GET', `/api/profiles/${id}`);
  createProfile = (req: CreateProfileRequest) =>
    this.request<Profile>('POST', '/api/profiles', req);
  updateProfile = (id: string, req: UpdateProfileRequest) =>
    this.request<Profile>('PUT', `/api/profiles/${id}`, req);
  deleteProfile = (id: string) => this.request<void>('DELETE', `/api/profiles/${id}`);
  activateProfile = (id: string) =>
    this.request<void>('POST', `/api/profiles/${id}/activate`);
  exportProfile = (id: string) =>
    this.request<ProfileExport>('GET', `/api/profiles/${id}/export`);
  importProfile = (data: ProfileExport) =>
    this.request<Profile>('POST', '/api/profiles/import', data);
}

export const api = new ApiClient(API_BASE);
```

### 4.2 React Query Integration

```typescript
// hooks/useModels.ts
export function useModel() {
  return useQuery({
    queryKey: ['model'],
    queryFn: api.getModel,
    refetchInterval: false, // WebSocket handles updates
  });
}

export function useLoadModel() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: api.loadModel,
    onSuccess: (model) => {
      queryClient.setQueryData(['model'], model);
    },
  });
}
```

---

## 5. WebSocket Integration

### 5.1 Socket Client Design

```typescript
// services/socket.ts
import { io, Socket } from 'socket.io-client';

const SOCKET_URL = import.meta.env.VITE_WS_URL || 'http://localhost:8000';

class SocketClient {
  private socket: Socket | null = null;
  private store: ServerStore;

  constructor() {
    this.store = useServerStore.getState();
  }

  connect() {
    this.socket = io(SOCKET_URL, {
      transports: ['websocket'],
      reconnection: true,
      reconnectionDelay: 1000,
      reconnectionDelayMax: 5000,
      reconnectionAttempts: Infinity,
    });

    this.socket.on('connect', () => {
      this.store.setConnected(true);
      this.joinRooms();
    });

    this.socket.on('disconnect', () => {
      this.store.setConnected(false);
    });

    this.socket.on('connect_error', (error) => {
      this.store.setConnectionError(error.message);
    });

    // Register event handlers
    this.registerEventHandlers();
  }

  private joinRooms() {
    // Join monitoring room to receive activation updates
    this.socket?.emit('join', { room: 'monitoring' });
    this.socket?.emit('join', { room: 'status' });
  }

  private registerEventHandlers() {
    // Model events
    this.socket?.on('model:status', (data: ModelStatusEvent) => {
      if (data.status === 'loaded') {
        this.store.setModel(data.model);
      } else if (data.status === 'unloaded') {
        this.store.setModel(null);
      }
      this.store.setModelLoading(data.status === 'loading');
    });

    // SAE events
    this.socket?.on('sae:status', (data: SAEStatusEvent) => {
      if (data.status === 'attached') {
        this.store.setSAE(data.sae);
      } else if (data.status === 'detached') {
        this.store.setSAE(null);
      }
    });

    // Steering events
    this.socket?.on('steering:update', (data: SteeringUpdateEvent) => {
      this.store.setSteeringValues(data.values);
      this.store.setSteeringEnabled(data.enabled);
    });

    // Monitoring events
    this.socket?.on('monitoring:activation', (data: ActivationEvent) => {
      if (!useUIStore.getState().monitoringPaused) {
        this.store.addActivation(data);
      }
    });

    // Server status
    this.socket?.on('server:status', (data: ServerStatusEvent) => {
      this.store.setServerStatus(data);
    });
  }

  disconnect() {
    this.socket?.disconnect();
    this.socket = null;
  }
}

export const socketClient = new SocketClient();
```

### 5.2 WebSocket Events

| Event | Direction | Payload | Description |
|-------|-----------|---------|-------------|
| `model:status` | Server→Client | `{status, model?}` | Model load/unload |
| `sae:status` | Server→Client | `{status, sae?}` | SAE attach/detach |
| `steering:update` | Server→Client | `{values, enabled}` | Steering changes |
| `monitoring:activation` | Server→Client | `{features, timestamp}` | Live activations |
| `server:status` | Server→Client | `{health, uptime}` | Server health |
| `join` | Client→Server | `{room}` | Join event room |
| `leave` | Client→Server | `{room}` | Leave event room |

---

## 6. Page Designs

### 6.1 Dashboard Page

**Purpose:** Overview of system state with quick actions

**Layout:**
```
┌──────────────────────────────────────────────────────────┐
│                    Dashboard                              │
├──────────────────────────────────────────────────────────┤
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────┐ │
│  │   Model     │ │    SAE      │ │  Steering   │ │ Mon │ │
│  │  GPT-2     │ │ gpt2-res-jb │ │  3 features │ │ ON  │ │
│  │  ✓ Loaded   │ │  ✓ Attached │ │  ✓ Active   │ │     │ │
│  └─────────────┘ └─────────────┘ └─────────────┘ └─────┘ │
├──────────────────────────────────────────────────────────┤
│  Quick Actions                                            │
│  [Load Model] [Attach SAE] [Clear Steering] [View Logs]  │
└──────────────────────────────────────────────────────────┘
```

**Data Flow:**
1. On mount, fetch initial state from API
2. Subscribe to WebSocket events for real-time updates
3. Status cards update reactively from Zustand store

### 6.2 Steering Page

**Purpose:** Configure feature steering values

**Layout:**
```
┌──────────────────────────────────────────────────────────┐
│  Steering Controls              [Enabled ●] [Clear All] │
├──────────────────────────────────────────────────────────┤
│  Add Feature: [_________] [Add] │ Batch: [___,___] [+]  │
├──────────────────────────────────────────────────────────┤
│  Active Steering Values                                   │
│  ┌────────────────────────────────────────────────────┐  │
│  │ Feature 1234    [-10 ════●════════════ +10] 2.5 [×]│  │
│  │ Feature 5678    [-10 ══════════●══════ +10] 5.0 [×]│  │
│  │ Feature 9012    [-10 ●════════════════ +10]-8.0 [×]│  │
│  └────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────┘
```

**Interactions:**
- Slider drag: Debounced API call (300ms)
- Numeric input: API call on blur or Enter
- Clear button: Optimistic removal + API call
- Enable toggle: Immediate API call

### 6.3 Monitoring Page

**Purpose:** Real-time activation observation

**Layout:**
```
┌──────────────────────────────────────────────────────────┐
│  Monitoring                     [Enabled ●] [Pause ⏸]   │
├──────────────────────────────────────────────────────────┤
│  Config: Top-K [10 ▼]                    [Clear History] │
├──────────────────────────────────────────────────────────┤
│  Live Activations              │ Statistics              │
│  ┌─────────────────────────┐   │ ┌─────────────────────┐ │
│  │ ▓▓▓▓▓▓▓▓░░ 1234: 0.85   │   │ │ Top Features:       │ │
│  │ ▓▓▓▓▓▓░░░░ 5678: 0.65   │   │ │ 1234: μ=0.72 σ=0.1 │ │
│  │ ▓▓▓▓░░░░░░ 9012: 0.42   │   │ │ 5678: μ=0.58 σ=0.2 │ │
│  │ ▓▓▓░░░░░░░ 3456: 0.31   │   │ │ Total: 1,234        │ │
│  └─────────────────────────┘   │ └─────────────────────┘ │
├──────────────────────────────────────────────────────────┤
│  History (last 100)                                       │
│  ┌────────────────────────────────────────────────────┐  │
│  │ 12:34:56 │ 1234(0.85), 5678(0.65), 9012(0.42)     │  │
│  │ 12:34:55 │ 5678(0.91), 1234(0.72), 3456(0.33)     │  │
│  └────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────┘
```

**Performance Considerations:**
- Virtual scrolling for history list
- Throttled updates (max 10/sec visual update)
- Canvas-based chart for smooth rendering

---

## 7. Routing

### 7.1 Route Structure

```typescript
// App.tsx
const router = createBrowserRouter([
  {
    path: '/',
    element: <Layout />,
    children: [
      { index: true, element: <DashboardPage /> },
      { path: 'models', element: <ModelsPage /> },
      { path: 'sae', element: <SAEPage /> },
      { path: 'steering', element: <SteeringPage /> },
      { path: 'monitoring', element: <MonitoringPage /> },
      { path: 'profiles', element: <ProfilesPage /> },
      { path: 'settings', element: <SettingsPage /> },
    ],
  },
]);
```

### 7.2 Navigation Items

```typescript
const navItems = [
  { path: '/', icon: HomeIcon, label: 'Dashboard' },
  { path: '/models', icon: CpuChipIcon, label: 'Models' },
  { path: '/sae', icon: CubeIcon, label: 'SAE' },
  { path: '/steering', icon: AdjustmentsIcon, label: 'Steering' },
  { path: '/monitoring', icon: ChartBarIcon, label: 'Monitoring' },
  { path: '/profiles', icon: UserCircleIcon, label: 'Profiles' },
  { path: '/settings', icon: CogIcon, label: 'Settings' },
];
```

---

## 8. Styling System

### 8.1 Tailwind Configuration

```javascript
// tailwind.config.js
module.exports = {
  darkMode: 'class',
  content: ['./src/**/*.{js,ts,jsx,tsx}'],
  theme: {
    extend: {
      colors: {
        surface: {
          DEFAULT: 'var(--color-surface)',
          hover: 'var(--color-surface-hover)',
        },
      },
    },
  },
  plugins: [],
};
```

### 8.2 Theme Variables

```css
/* index.css */
:root {
  --color-surface: #1e293b;
  --color-surface-hover: #334155;
  --color-border: #475569;
  --color-text: #f8fafc;
  --color-text-muted: #94a3b8;
}

:root.light {
  --color-surface: #ffffff;
  --color-surface-hover: #f1f5f9;
  --color-border: #e2e8f0;
  --color-text: #0f172a;
  --color-text-muted: #64748b;
}
```

### 8.3 Component Classes

```typescript
// Reusable class patterns
const buttonVariants = {
  primary: 'bg-blue-600 hover:bg-blue-700 text-white',
  secondary: 'bg-slate-700 hover:bg-slate-600 text-slate-200',
  danger: 'bg-red-600 hover:bg-red-700 text-white',
  ghost: 'hover:bg-slate-700 text-slate-300',
};

const cardClass = 'bg-surface border border-slate-700 rounded-lg p-4';
const inputClass = 'bg-slate-800 border border-slate-600 rounded px-3 py-2 text-slate-200';
```

---

## 9. Error Handling

### 9.1 Error Boundary

```typescript
// components/ErrorBoundary.tsx
class ErrorBoundary extends React.Component<Props, State> {
  static getDerivedStateFromError(error: Error) {
    return { hasError: true, error };
  }

  render() {
    if (this.state.hasError) {
      return (
        <div className="flex items-center justify-center h-screen">
          <div className="text-center">
            <h1>Something went wrong</h1>
            <p>{this.state.error.message}</p>
            <button onClick={() => window.location.reload()}>
              Reload Page
            </button>
          </div>
        </div>
      );
    }
    return this.props.children;
  }
}
```

### 9.2 Toast Notifications

```typescript
// hooks/useToast.ts
export function useToast() {
  const { addToast, removeToast } = useUIStore();

  return {
    success: (message: string) => addToast({ type: 'success', message }),
    error: (message: string) => addToast({ type: 'error', message }),
    info: (message: string) => addToast({ type: 'info', message }),
    warning: (message: string) => addToast({ type: 'warning', message }),
  };
}
```

### 9.3 API Error Handling

```typescript
// All API calls wrapped with error handling
async function handleApiCall<T>(
  apiCall: () => Promise<T>,
  toast: ReturnType<typeof useToast>
): Promise<T | null> {
  try {
    return await apiCall();
  } catch (error) {
    if (error instanceof ApiError) {
      toast.error(error.message);
    } else {
      toast.error('An unexpected error occurred');
    }
    return null;
  }
}
```

---

## 10. Testing Strategy

### 10.1 Test Structure

```
admin-ui/
├── src/
│   ├── components/
│   │   └── Button/
│   │       ├── Button.tsx
│   │       └── Button.test.tsx
│   ├── hooks/
│   │   └── useModels.test.ts
│   └── stores/
│       └── serverStore.test.ts
├── tests/
│   ├── integration/
│   │   └── pages/
│   │       └── Dashboard.test.tsx
│   └── e2e/
│       └── steering.spec.ts
```

### 10.2 Test Categories

| Type | Tools | Coverage Target |
|------|-------|-----------------|
| Unit | Vitest + RTL | Components, hooks, utils |
| Integration | Vitest + RTL | Page-level, API mocking |
| E2E | Playwright | Critical user flows |

### 10.3 Mocking Strategy

```typescript
// Mock API client for tests
vi.mock('../services/api', () => ({
  api: {
    getModel: vi.fn().mockResolvedValue({ id: 'gpt2', name: 'GPT-2' }),
    loadModel: vi.fn().mockResolvedValue({ id: 'gpt2', name: 'GPT-2' }),
    // ... other mocks
  },
}));

// Mock WebSocket for tests
vi.mock('../services/socket', () => ({
  socketClient: {
    connect: vi.fn(),
    disconnect: vi.fn(),
  },
}));
```

---

## 11. Build & Deployment

### 11.1 Build Configuration

```typescript
// vite.config.ts
export default defineConfig({
  plugins: [react()],
  build: {
    outDir: 'dist',
    sourcemap: true,
    rollupOptions: {
      output: {
        manualChunks: {
          vendor: ['react', 'react-dom', 'react-router-dom'],
          charts: ['chart.js', 'react-chartjs-2'],
        },
      },
    },
  },
  server: {
    proxy: {
      '/api': 'http://localhost:8000',
      '/socket.io': {
        target: 'http://localhost:8000',
        ws: true,
      },
    },
  },
});
```

### 11.2 Deployment Options

**Option A: Static Files (Recommended)**
- Build to `dist/`
- Serve via FastAPI's static file serving
- Single deployment unit

**Option B: Separate Deployment**
- Build to `dist/`
- Deploy to CDN or static host
- Configure CORS on backend

---

## 12. File Structure (Complete)

```
admin-ui/
├── public/
│   └── favicon.ico
├── src/
│   ├── components/
│   │   ├── common/
│   │   │   ├── Button.tsx
│   │   │   ├── Card.tsx
│   │   │   ├── Input.tsx
│   │   │   ├── Select.tsx
│   │   │   ├── Slider.tsx
│   │   │   ├── Modal.tsx
│   │   │   ├── Toast.tsx
│   │   │   ├── Spinner.tsx
│   │   │   ├── Badge.tsx
│   │   │   └── EmptyState.tsx
│   │   ├── layout/
│   │   │   ├── Layout.tsx
│   │   │   ├── Sidebar.tsx
│   │   │   ├── Header.tsx
│   │   │   └── NavItem.tsx
│   │   ├── dashboard/
│   │   │   ├── StatusCard.tsx
│   │   │   └── QuickActions.tsx
│   │   ├── models/
│   │   │   ├── ModelLoadForm.tsx
│   │   │   └── LoadedModelCard.tsx
│   │   ├── sae/
│   │   │   ├── SAEDownloadForm.tsx
│   │   │   ├── SAEList.tsx
│   │   │   └── AttachedSAECard.tsx
│   │   ├── steering/
│   │   │   ├── SteeringControls.tsx
│   │   │   ├── SteeringSlider.tsx
│   │   │   └── BatchAddForm.tsx
│   │   ├── monitoring/
│   │   │   ├── MonitoringControls.tsx
│   │   │   ├── ActivationChart.tsx
│   │   │   ├── ActivationHistory.tsx
│   │   │   └── StatisticsPanel.tsx
│   │   └── profiles/
│   │       ├── ProfileList.tsx
│   │       ├── ProfileForm.tsx
│   │       └── ImportExportButtons.tsx
│   ├── hooks/
│   │   ├── useModels.ts
│   │   ├── useSAE.ts
│   │   ├── useSteering.ts
│   │   ├── useMonitoring.ts
│   │   ├── useProfiles.ts
│   │   └── useToast.ts
│   ├── pages/
│   │   ├── DashboardPage.tsx
│   │   ├── ModelsPage.tsx
│   │   ├── SAEPage.tsx
│   │   ├── SteeringPage.tsx
│   │   ├── MonitoringPage.tsx
│   │   ├── ProfilesPage.tsx
│   │   └── SettingsPage.tsx
│   ├── services/
│   │   ├── api.ts
│   │   └── socket.ts
│   ├── stores/
│   │   ├── serverStore.ts
│   │   └── uiStore.ts
│   ├── types/
│   │   ├── api.ts
│   │   └── ui.ts
│   ├── utils/
│   │   ├── formatters.ts
│   │   └── validators.ts
│   ├── App.tsx
│   ├── main.tsx
│   └── index.css
├── tests/
│   ├── integration/
│   └── e2e/
├── package.json
├── tsconfig.json
├── vite.config.ts
├── tailwind.config.js
└── postcss.config.js
```

---

**Document Status:** Complete
**Next Document:** 007_FTID|Admin_UI.md (Technical Implementation Document)
