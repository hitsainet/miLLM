# Technical Implementation Document: Admin UI

## miLLM Feature 7

**Document Version:** 1.0
**Created:** January 30, 2026
**Status:** Draft
**References:**
- Feature PRD: `007_FPRD|Admin_UI.md`
- Feature TDD: `007_FTDD|Admin_UI.md`

---

## 1. Implementation Overview

This document provides specific implementation guidance for the miLLM Admin UI - a React-based dashboard for server management.

### 1.1 Technology Stack

| Technology | Version | Purpose |
|------------|---------|---------|
| React | 18.x | UI framework |
| TypeScript | 5.x | Type safety |
| Vite | 5.x | Build tool |
| Tailwind CSS | 3.x | Styling |
| Zustand | 4.x | State management |
| React Query | 5.x | Server state |
| React Router | 6.x | Routing |
| socket.io-client | 4.x | WebSocket |
| Heroicons | 2.x | Icons |

### 1.2 Project Setup

```bash
# Create Vite project
npm create vite@latest admin-ui -- --template react-ts
cd admin-ui

# Install dependencies
npm install zustand @tanstack/react-query react-router-dom socket.io-client
npm install -D tailwindcss postcss autoprefixer @heroicons/react

# Initialize Tailwind
npx tailwindcss init -p
```

---

## 2. File Structure Implementation

### 2.1 Directory Setup

```bash
mkdir -p src/{components/{common,layout,dashboard,models,sae,steering,monitoring,profiles},hooks,pages,services,stores,types,utils}
mkdir -p tests/{integration,e2e}
```

### 2.2 Base Configuration Files

**tailwind.config.js**
```javascript
/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,ts,jsx,tsx}'],
  darkMode: 'class',
  theme: {
    extend: {
      colors: {
        surface: {
          DEFAULT: '#1e293b',
          hover: '#334155',
          light: '#ffffff',
          'light-hover': '#f1f5f9',
        },
      },
    },
  },
  plugins: [],
};
```

**tsconfig.json** (paths)
```json
{
  "compilerOptions": {
    "baseUrl": ".",
    "paths": {
      "@/*": ["./src/*"],
      "@components/*": ["./src/components/*"],
      "@hooks/*": ["./src/hooks/*"],
      "@services/*": ["./src/services/*"],
      "@stores/*": ["./src/stores/*"],
      "@types/*": ["./src/types/*"]
    }
  }
}
```

**vite.config.ts**
```typescript
import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import path from 'path';

export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
      '@components': path.resolve(__dirname, './src/components'),
      '@hooks': path.resolve(__dirname, './src/hooks'),
      '@services': path.resolve(__dirname, './src/services'),
      '@stores': path.resolve(__dirname, './src/stores'),
      '@types': path.resolve(__dirname, './src/types'),
    },
  },
  server: {
    port: 3000,
    proxy: {
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
      '/socket.io': {
        target: 'http://localhost:8000',
        ws: true,
      },
    },
  },
});
```

---

## 3. Type Definitions

### 3.1 API Types

**src/types/api.ts**
```typescript
// Model Types
export interface ModelInfo {
  id: string;
  name: string;
  device: string;
  dtype: string;
  loaded_at: string;
}

export interface LoadModelRequest {
  model_id: string;
  device?: 'auto' | 'cuda' | 'cpu';
  dtype?: 'auto' | 'float16' | 'bfloat16' | 'float32';
}

// SAE Types
export interface SAEInfo {
  id: string;
  repo_id: string;
  filename: string;
  layer: number;
  num_features: number;
  downloaded_at: string;
  attached: boolean;
}

export interface DownloadSAERequest {
  repo_id: string;
  filename?: string;
}

// Steering Types
export interface SteeringState {
  enabled: boolean;
  values: Record<number, number>;
  feature_count: number;
}

export interface SetSteeringRequest {
  feature_index: number;
  value: number;
}

export interface BatchSteeringRequest {
  values: Record<number, number>;
}

// Monitoring Types
export interface MonitoringConfig {
  enabled: boolean;
  top_k: number;
  throttle_ms: number;
}

export interface ActivationRecord {
  timestamp: string;
  request_id: string;
  features: Array<{ index: number; value: number }>;
}

export interface FeatureStatistics {
  total_activations: number;
  feature_stats: Record<number, {
    count: number;
    mean: number;
    max: number;
  }>;
}

// Profile Types
export interface Profile {
  id: string;
  name: string;
  description: string;
  steering: Record<number, number>;
  is_active: boolean;
  created_at: string;
  updated_at: string;
}

export interface CreateProfileRequest {
  name: string;
  description?: string;
  steering?: Record<number, number>;
}

export interface UpdateProfileRequest {
  name?: string;
  description?: string;
  steering?: Record<number, number>;
}

export interface ProfileExport {
  version: string;
  type: 'miLLM-profile';
  profile: {
    name: string;
    description: string;
    steering: Record<number, number>;
  };
  exported_at: string;
}
```

### 3.2 UI Types

**src/types/ui.ts**
```typescript
export type Theme = 'light' | 'dark';

export interface Toast {
  id: string;
  type: 'success' | 'error' | 'warning' | 'info';
  message: string;
  duration?: number;
}

export interface NavItem {
  path: string;
  label: string;
  icon: React.ComponentType<{ className?: string }>;
}

export type ConnectionStatus = 'connected' | 'disconnected' | 'connecting';
```

---

## 4. Zustand Stores

### 4.1 Server Store

**src/stores/serverStore.ts**
```typescript
import { create } from 'zustand';
import { subscribeWithSelector } from 'zustand/middleware';
import type {
  ModelInfo,
  SAEInfo,
  SteeringState,
  MonitoringConfig,
  ActivationRecord,
  FeatureStatistics,
  Profile,
} from '@/types/api';
import type { ConnectionStatus } from '@/types/ui';

interface ServerState {
  // Connection
  connectionStatus: ConnectionStatus;
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
  statistics: FeatureStatistics | null;

  // Profiles
  profiles: Profile[];
  activeProfile: Profile | null;
}

interface ServerActions {
  // Connection
  setConnectionStatus: (status: ConnectionStatus) => void;
  setConnectionError: (error: string | null) => void;

  // Model
  setModel: (model: ModelInfo | null) => void;
  setModelLoading: (loading: boolean) => void;

  // SAE
  setAttachedSAE: (sae: SAEInfo | null) => void;
  setDownloadedSAEs: (saes: SAEInfo[]) => void;
  addDownloadedSAE: (sae: SAEInfo) => void;
  removeDownloadedSAE: (id: string) => void;
  setSAELoading: (loading: boolean) => void;

  // Steering
  setSteeringEnabled: (enabled: boolean) => void;
  setSteeringValues: (values: Record<number, number>) => void;
  updateSteeringValue: (feature: number, value: number) => void;
  removeSteeringValue: (feature: number) => void;
  clearSteeringValues: () => void;

  // Monitoring
  setMonitoringEnabled: (enabled: boolean) => void;
  setMonitoringConfig: (config: MonitoringConfig) => void;
  addActivation: (activation: ActivationRecord) => void;
  clearActivations: () => void;
  setStatistics: (stats: FeatureStatistics | null) => void;

  // Profiles
  setProfiles: (profiles: Profile[]) => void;
  addProfile: (profile: Profile) => void;
  updateProfile: (id: string, profile: Partial<Profile>) => void;
  removeProfile: (id: string) => void;
  setActiveProfile: (profile: Profile | null) => void;

  // Reset
  reset: () => void;
}

const initialState: ServerState = {
  connectionStatus: 'disconnected',
  connectionError: null,
  model: null,
  modelLoading: false,
  attachedSAE: null,
  downloadedSAEs: [],
  saeLoading: false,
  steeringEnabled: false,
  steeringValues: {},
  monitoringEnabled: false,
  monitoringConfig: { enabled: false, top_k: 10, throttle_ms: 100 },
  activations: [],
  statistics: null,
  profiles: [],
  activeProfile: null,
};

const MAX_ACTIVATIONS = 100;

export const useServerStore = create<ServerState & ServerActions>()(
  subscribeWithSelector((set) => ({
    ...initialState,

    // Connection
    setConnectionStatus: (status) => set({ connectionStatus: status }),
    setConnectionError: (error) => set({ connectionError: error }),

    // Model
    setModel: (model) => set({ model }),
    setModelLoading: (loading) => set({ modelLoading: loading }),

    // SAE
    setAttachedSAE: (sae) => set({ attachedSAE: sae }),
    setDownloadedSAEs: (saes) => set({ downloadedSAEs: saes }),
    addDownloadedSAE: (sae) =>
      set((state) => ({ downloadedSAEs: [...state.downloadedSAEs, sae] })),
    removeDownloadedSAE: (id) =>
      set((state) => ({
        downloadedSAEs: state.downloadedSAEs.filter((s) => s.id !== id),
      })),
    setSAELoading: (loading) => set({ saeLoading: loading }),

    // Steering
    setSteeringEnabled: (enabled) => set({ steeringEnabled: enabled }),
    setSteeringValues: (values) => set({ steeringValues: values }),
    updateSteeringValue: (feature, value) =>
      set((state) => ({
        steeringValues: { ...state.steeringValues, [feature]: value },
      })),
    removeSteeringValue: (feature) =>
      set((state) => {
        const { [feature]: _, ...rest } = state.steeringValues;
        return { steeringValues: rest };
      }),
    clearSteeringValues: () => set({ steeringValues: {} }),

    // Monitoring
    setMonitoringEnabled: (enabled) => set({ monitoringEnabled: enabled }),
    setMonitoringConfig: (config) => set({ monitoringConfig: config }),
    addActivation: (activation) =>
      set((state) => ({
        activations: [activation, ...state.activations].slice(0, MAX_ACTIVATIONS),
      })),
    clearActivations: () => set({ activations: [] }),
    setStatistics: (stats) => set({ statistics: stats }),

    // Profiles
    setProfiles: (profiles) => set({ profiles }),
    addProfile: (profile) =>
      set((state) => ({ profiles: [...state.profiles, profile] })),
    updateProfile: (id, updates) =>
      set((state) => ({
        profiles: state.profiles.map((p) =>
          p.id === id ? { ...p, ...updates } : p
        ),
      })),
    removeProfile: (id) =>
      set((state) => ({
        profiles: state.profiles.filter((p) => p.id !== id),
      })),
    setActiveProfile: (profile) => set({ activeProfile: profile }),

    // Reset
    reset: () => set(initialState),
  }))
);
```

### 4.2 UI Store

**src/stores/uiStore.ts**
```typescript
import { create } from 'zustand';
import { persist } from 'zustand/middleware';
import type { Theme, Toast } from '@/types/ui';

interface UIState {
  theme: Theme;
  sidebarCollapsed: boolean;
  activeModal: string | null;
  modalData: unknown;
  toasts: Toast[];
  monitoringPaused: boolean;
}

interface UIActions {
  setTheme: (theme: Theme) => void;
  toggleTheme: () => void;
  setSidebarCollapsed: (collapsed: boolean) => void;
  toggleSidebar: () => void;
  showModal: (id: string, data?: unknown) => void;
  hideModal: () => void;
  addToast: (toast: Omit<Toast, 'id'>) => void;
  removeToast: (id: string) => void;
  setMonitoringPaused: (paused: boolean) => void;
  toggleMonitoringPaused: () => void;
}

export const useUIStore = create<UIState & UIActions>()(
  persist(
    (set) => ({
      theme: 'dark',
      sidebarCollapsed: false,
      activeModal: null,
      modalData: null,
      toasts: [],
      monitoringPaused: false,

      setTheme: (theme) => {
        document.documentElement.classList.toggle('dark', theme === 'dark');
        set({ theme });
      },
      toggleTheme: () =>
        set((state) => {
          const newTheme = state.theme === 'dark' ? 'light' : 'dark';
          document.documentElement.classList.toggle('dark', newTheme === 'dark');
          return { theme: newTheme };
        }),

      setSidebarCollapsed: (collapsed) => set({ sidebarCollapsed: collapsed }),
      toggleSidebar: () =>
        set((state) => ({ sidebarCollapsed: !state.sidebarCollapsed })),

      showModal: (id, data) => set({ activeModal: id, modalData: data }),
      hideModal: () => set({ activeModal: null, modalData: null }),

      addToast: (toast) =>
        set((state) => ({
          toasts: [...state.toasts, { ...toast, id: crypto.randomUUID() }],
        })),
      removeToast: (id) =>
        set((state) => ({
          toasts: state.toasts.filter((t) => t.id !== id),
        })),

      setMonitoringPaused: (paused) => set({ monitoringPaused: paused }),
      toggleMonitoringPaused: () =>
        set((state) => ({ monitoringPaused: !state.monitoringPaused })),
    }),
    {
      name: 'millm-ui-settings',
      partialize: (state) => ({
        theme: state.theme,
        sidebarCollapsed: state.sidebarCollapsed,
      }),
    }
  )
);
```

---

## 5. Services

### 5.1 API Client

**src/services/api.ts**
```typescript
import type {
  ModelInfo,
  LoadModelRequest,
  SAEInfo,
  DownloadSAERequest,
  SteeringState,
  SetSteeringRequest,
  BatchSteeringRequest,
  MonitoringConfig,
  ActivationRecord,
  FeatureStatistics,
  Profile,
  CreateProfileRequest,
  UpdateProfileRequest,
  ProfileExport,
} from '@/types/api';

const API_BASE = import.meta.env.VITE_API_URL || '';

export class ApiError extends Error {
  constructor(
    message: string,
    public status: number,
    public code?: string
  ) {
    super(message);
    this.name = 'ApiError';
  }
}

async function request<T>(
  method: string,
  path: string,
  body?: unknown
): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, {
    method,
    headers: {
      'Content-Type': 'application/json',
    },
    body: body ? JSON.stringify(body) : undefined,
  });

  if (!response.ok) {
    let message = 'Request failed';
    let code: string | undefined;
    try {
      const error = await response.json();
      message = error.detail || error.message || message;
      code = error.code;
    } catch {
      // Ignore JSON parse error
    }
    throw new ApiError(message, response.status, code);
  }

  // Handle 204 No Content
  if (response.status === 204) {
    return undefined as T;
  }

  return response.json();
}

export const api = {
  // Models
  getModel: () => request<ModelInfo | null>('GET', '/api/models'),
  loadModel: (req: LoadModelRequest) =>
    request<ModelInfo>('POST', '/api/models', req),
  unloadModel: () => request<void>('DELETE', '/api/models'),

  // SAE
  listSAEs: () => request<SAEInfo[]>('GET', '/api/sae'),
  downloadSAE: (req: DownloadSAERequest) =>
    request<SAEInfo>('POST', '/api/sae/download', req),
  attachSAE: (id: string) => request<void>('POST', `/api/sae/${id}/attach`),
  detachSAE: () => request<void>('POST', '/api/sae/detach'),
  deleteSAE: (id: string) => request<void>('DELETE', `/api/sae/${id}`),

  // Steering
  getSteering: () => request<SteeringState>('GET', '/api/steering'),
  setSteering: (req: SetSteeringRequest) =>
    request<void>('POST', '/api/steering', req),
  batchSteering: (req: BatchSteeringRequest) =>
    request<void>('POST', '/api/steering/batch', req),
  clearSteering: () => request<void>('DELETE', '/api/steering'),
  toggleSteering: (enabled: boolean) =>
    request<void>('POST', '/api/steering/toggle', { enabled }),

  // Monitoring
  getMonitoring: () => request<MonitoringConfig>('GET', '/api/monitoring'),
  configureMonitoring: (config: Partial<MonitoringConfig>) =>
    request<void>('POST', '/api/monitoring/configure', config),
  enableMonitoring: (enabled: boolean) =>
    request<void>('POST', '/api/monitoring/enable', { enabled }),
  getHistory: () => request<ActivationRecord[]>('GET', '/api/monitoring/history'),
  clearHistory: () => request<void>('DELETE', '/api/monitoring/history'),
  getStatistics: () =>
    request<FeatureStatistics>('GET', '/api/monitoring/statistics'),
  resetStatistics: () => request<void>('DELETE', '/api/monitoring/statistics'),

  // Profiles
  listProfiles: () => request<Profile[]>('GET', '/api/profiles'),
  getProfile: (id: string) => request<Profile>('GET', `/api/profiles/${id}`),
  createProfile: (req: CreateProfileRequest) =>
    request<Profile>('POST', '/api/profiles', req),
  updateProfile: (id: string, req: UpdateProfileRequest) =>
    request<Profile>('PUT', `/api/profiles/${id}`, req),
  deleteProfile: (id: string) => request<void>('DELETE', `/api/profiles/${id}`),
  activateProfile: (id: string) =>
    request<void>('POST', `/api/profiles/${id}/activate`),
  deactivateProfile: (id: string) =>
    request<void>('POST', `/api/profiles/${id}/deactivate`),
  exportProfile: (id: string) =>
    request<ProfileExport>('GET', `/api/profiles/${id}/export`),
  importProfile: (data: ProfileExport) =>
    request<Profile>('POST', '/api/profiles/import', data),
};
```

### 5.2 WebSocket Client

**src/services/socket.ts**
```typescript
import { io, Socket } from 'socket.io-client';
import { useServerStore } from '@/stores/serverStore';
import { useUIStore } from '@/stores/uiStore';
import type { ModelInfo, SAEInfo, ActivationRecord } from '@/types/api';

const SOCKET_URL = import.meta.env.VITE_WS_URL || '';

interface ModelStatusEvent {
  status: 'loading' | 'loaded' | 'unloading' | 'unloaded' | 'error';
  model?: ModelInfo;
  error?: string;
}

interface SAEStatusEvent {
  status: 'downloading' | 'attaching' | 'attached' | 'detaching' | 'detached' | 'error';
  sae?: SAEInfo;
  progress?: number;
  error?: string;
}

interface SteeringUpdateEvent {
  enabled: boolean;
  values: Record<number, number>;
}

interface ActivationEvent {
  timestamp: string;
  request_id: string;
  features: Array<{ index: number; value: number }>;
}

class SocketClient {
  private socket: Socket | null = null;
  private reconnectAttempts = 0;
  private maxReconnectAttempts = 10;

  connect() {
    const store = useServerStore.getState();

    store.setConnectionStatus('connecting');

    this.socket = io(SOCKET_URL, {
      transports: ['websocket'],
      reconnection: true,
      reconnectionDelay: 1000,
      reconnectionDelayMax: 5000,
      reconnectionAttempts: this.maxReconnectAttempts,
    });

    this.socket.on('connect', () => {
      const store = useServerStore.getState();
      store.setConnectionStatus('connected');
      store.setConnectionError(null);
      this.reconnectAttempts = 0;
      this.joinRooms();
    });

    this.socket.on('disconnect', (reason) => {
      const store = useServerStore.getState();
      store.setConnectionStatus('disconnected');
      if (reason === 'io server disconnect') {
        // Server initiated disconnect, try to reconnect
        this.socket?.connect();
      }
    });

    this.socket.on('connect_error', (error) => {
      const store = useServerStore.getState();
      this.reconnectAttempts++;
      store.setConnectionError(
        `Connection failed (attempt ${this.reconnectAttempts}): ${error.message}`
      );
      if (this.reconnectAttempts >= this.maxReconnectAttempts) {
        store.setConnectionStatus('disconnected');
      }
    });

    this.registerEventHandlers();
  }

  private joinRooms() {
    this.socket?.emit('join', { room: 'status' });
    this.socket?.emit('join', { room: 'monitoring' });
  }

  private registerEventHandlers() {
    if (!this.socket) return;

    // Model events
    this.socket.on('model:status', (data: ModelStatusEvent) => {
      const store = useServerStore.getState();

      switch (data.status) {
        case 'loading':
          store.setModelLoading(true);
          break;
        case 'loaded':
          store.setModel(data.model || null);
          store.setModelLoading(false);
          break;
        case 'unloading':
          store.setModelLoading(true);
          break;
        case 'unloaded':
          store.setModel(null);
          store.setModelLoading(false);
          break;
        case 'error':
          store.setModelLoading(false);
          useUIStore.getState().addToast({
            type: 'error',
            message: data.error || 'Model operation failed',
          });
          break;
      }
    });

    // SAE events
    this.socket.on('sae:status', (data: SAEStatusEvent) => {
      const store = useServerStore.getState();

      switch (data.status) {
        case 'downloading':
        case 'attaching':
          store.setSAELoading(true);
          break;
        case 'attached':
          store.setAttachedSAE(data.sae || null);
          store.setSAELoading(false);
          break;
        case 'detaching':
          store.setSAELoading(true);
          break;
        case 'detached':
          store.setAttachedSAE(null);
          store.setSAELoading(false);
          break;
        case 'error':
          store.setSAELoading(false);
          useUIStore.getState().addToast({
            type: 'error',
            message: data.error || 'SAE operation failed',
          });
          break;
      }
    });

    // Steering events
    this.socket.on('steering:update', (data: SteeringUpdateEvent) => {
      const store = useServerStore.getState();
      store.setSteeringEnabled(data.enabled);
      store.setSteeringValues(data.values);
    });

    // Monitoring events
    this.socket.on('monitoring:activation', (data: ActivationEvent) => {
      const uiStore = useUIStore.getState();
      if (!uiStore.monitoringPaused) {
        const store = useServerStore.getState();
        store.addActivation(data);
      }
    });
  }

  disconnect() {
    this.socket?.disconnect();
    this.socket = null;
  }

  reconnect() {
    this.disconnect();
    this.reconnectAttempts = 0;
    this.connect();
  }
}

export const socketClient = new SocketClient();
```

---

## 6. Common Components

### 6.1 Button

**src/components/common/Button.tsx**
```typescript
import { forwardRef } from 'react';
import { Spinner } from './Spinner';

type ButtonVariant = 'primary' | 'secondary' | 'danger' | 'ghost';
type ButtonSize = 'sm' | 'md' | 'lg';

interface ButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: ButtonVariant;
  size?: ButtonSize;
  loading?: boolean;
  leftIcon?: React.ReactNode;
  rightIcon?: React.ReactNode;
}

const variantStyles: Record<ButtonVariant, string> = {
  primary: 'bg-blue-600 hover:bg-blue-700 text-white disabled:bg-blue-800',
  secondary:
    'bg-slate-700 hover:bg-slate-600 text-slate-200 disabled:bg-slate-800',
  danger: 'bg-red-600 hover:bg-red-700 text-white disabled:bg-red-800',
  ghost: 'hover:bg-slate-700 text-slate-300 disabled:text-slate-600',
};

const sizeStyles: Record<ButtonSize, string> = {
  sm: 'px-2 py-1 text-sm',
  md: 'px-4 py-2',
  lg: 'px-6 py-3 text-lg',
};

export const Button = forwardRef<HTMLButtonElement, ButtonProps>(
  (
    {
      variant = 'primary',
      size = 'md',
      loading = false,
      leftIcon,
      rightIcon,
      disabled,
      children,
      className = '',
      ...props
    },
    ref
  ) => {
    return (
      <button
        ref={ref}
        disabled={disabled || loading}
        className={`
          inline-flex items-center justify-center gap-2
          font-medium rounded-lg transition-colors
          disabled:cursor-not-allowed disabled:opacity-60
          ${variantStyles[variant]}
          ${sizeStyles[size]}
          ${className}
        `}
        {...props}
      >
        {loading ? <Spinner size="sm" /> : leftIcon}
        {children}
        {!loading && rightIcon}
      </button>
    );
  }
);

Button.displayName = 'Button';
```

### 6.2 Card

**src/components/common/Card.tsx**
```typescript
interface CardProps {
  children: React.ReactNode;
  className?: string;
  padding?: 'none' | 'sm' | 'md' | 'lg';
}

const paddingStyles = {
  none: '',
  sm: 'p-3',
  md: 'p-4',
  lg: 'p-6',
};

export function Card({ children, className = '', padding = 'md' }: CardProps) {
  return (
    <div
      className={`
        bg-slate-800 border border-slate-700 rounded-lg
        ${paddingStyles[padding]}
        ${className}
      `}
    >
      {children}
    </div>
  );
}

interface CardHeaderProps {
  title: string;
  subtitle?: string;
  action?: React.ReactNode;
}

export function CardHeader({ title, subtitle, action }: CardHeaderProps) {
  return (
    <div className="flex items-center justify-between mb-4">
      <div>
        <h3 className="text-lg font-semibold text-slate-100">{title}</h3>
        {subtitle && <p className="text-sm text-slate-400">{subtitle}</p>}
      </div>
      {action}
    </div>
  );
}
```

### 6.3 Slider

**src/components/common/Slider.tsx**
```typescript
import { useCallback, useState } from 'react';

interface SliderProps {
  value: number;
  onChange: (value: number) => void;
  min?: number;
  max?: number;
  step?: number;
  label?: string;
  showValue?: boolean;
  disabled?: boolean;
}

export function Slider({
  value,
  onChange,
  min = -10,
  max = 10,
  step = 0.1,
  label,
  showValue = true,
  disabled = false,
}: SliderProps) {
  const [localValue, setLocalValue] = useState(value);
  const [isDragging, setIsDragging] = useState(false);

  const handleChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const newValue = parseFloat(e.target.value);
      setLocalValue(newValue);
    },
    []
  );

  const handleMouseUp = useCallback(() => {
    setIsDragging(false);
    if (localValue !== value) {
      onChange(localValue);
    }
  }, [localValue, value, onChange]);

  const handleMouseDown = useCallback(() => {
    setIsDragging(true);
  }, []);

  // Calculate percentage for gradient
  const percentage = ((localValue - min) / (max - min)) * 100;

  return (
    <div className="w-full">
      {(label || showValue) && (
        <div className="flex items-center justify-between mb-1">
          {label && (
            <label className="text-sm font-medium text-slate-300">{label}</label>
          )}
          {showValue && (
            <span className="text-sm font-mono text-slate-400">
              {localValue.toFixed(1)}
            </span>
          )}
        </div>
      )}
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={isDragging ? localValue : value}
        onChange={handleChange}
        onMouseDown={handleMouseDown}
        onMouseUp={handleMouseUp}
        onTouchStart={handleMouseDown}
        onTouchEnd={handleMouseUp}
        disabled={disabled}
        className="w-full h-2 bg-slate-700 rounded-lg appearance-none cursor-pointer
          disabled:cursor-not-allowed disabled:opacity-50
          [&::-webkit-slider-thumb]:appearance-none
          [&::-webkit-slider-thumb]:w-4
          [&::-webkit-slider-thumb]:h-4
          [&::-webkit-slider-thumb]:rounded-full
          [&::-webkit-slider-thumb]:bg-blue-500
          [&::-webkit-slider-thumb]:cursor-pointer
          [&::-webkit-slider-thumb]:hover:bg-blue-400"
        style={{
          background: `linear-gradient(to right, #3b82f6 0%, #3b82f6 ${percentage}%, #334155 ${percentage}%, #334155 100%)`,
        }}
      />
    </div>
  );
}
```

### 6.4 Toast

**src/components/common/Toast.tsx**
```typescript
import { useEffect } from 'react';
import {
  CheckCircleIcon,
  ExclamationCircleIcon,
  ExclamationTriangleIcon,
  InformationCircleIcon,
  XMarkIcon,
} from '@heroicons/react/24/outline';
import { useUIStore } from '@/stores/uiStore';
import type { Toast as ToastType } from '@/types/ui';

const icons = {
  success: CheckCircleIcon,
  error: ExclamationCircleIcon,
  warning: ExclamationTriangleIcon,
  info: InformationCircleIcon,
};

const styles = {
  success: 'bg-green-900/80 border-green-700 text-green-100',
  error: 'bg-red-900/80 border-red-700 text-red-100',
  warning: 'bg-amber-900/80 border-amber-700 text-amber-100',
  info: 'bg-blue-900/80 border-blue-700 text-blue-100',
};

interface ToastItemProps {
  toast: ToastType;
}

function ToastItem({ toast }: ToastItemProps) {
  const removeToast = useUIStore((s) => s.removeToast);
  const Icon = icons[toast.type];

  useEffect(() => {
    const duration = toast.duration ?? 5000;
    const timer = setTimeout(() => {
      removeToast(toast.id);
    }, duration);

    return () => clearTimeout(timer);
  }, [toast.id, toast.duration, removeToast]);

  return (
    <div
      className={`
        flex items-center gap-3 px-4 py-3 rounded-lg border
        shadow-lg backdrop-blur-sm animate-slide-in
        ${styles[toast.type]}
      `}
    >
      <Icon className="w-5 h-5 flex-shrink-0" />
      <p className="flex-1 text-sm">{toast.message}</p>
      <button
        onClick={() => removeToast(toast.id)}
        className="p-1 hover:bg-white/10 rounded"
      >
        <XMarkIcon className="w-4 h-4" />
      </button>
    </div>
  );
}

export function ToastContainer() {
  const toasts = useUIStore((s) => s.toasts);

  return (
    <div className="fixed bottom-4 right-4 z-50 flex flex-col gap-2 max-w-sm">
      {toasts.map((toast) => (
        <ToastItem key={toast.id} toast={toast} />
      ))}
    </div>
  );
}
```

### 6.5 Modal

**src/components/common/Modal.tsx**
```typescript
import { useEffect, useRef } from 'react';
import { XMarkIcon } from '@heroicons/react/24/outline';
import { useUIStore } from '@/stores/uiStore';

interface ModalProps {
  id: string;
  title: string;
  children: React.ReactNode;
  footer?: React.ReactNode;
  size?: 'sm' | 'md' | 'lg';
}

const sizeStyles = {
  sm: 'max-w-md',
  md: 'max-w-lg',
  lg: 'max-w-2xl',
};

export function Modal({ id, title, children, footer, size = 'md' }: ModalProps) {
  const { activeModal, hideModal } = useUIStore();
  const overlayRef = useRef<HTMLDivElement>(null);

  const isOpen = activeModal === id;

  useEffect(() => {
    const handleEscape = (e: KeyboardEvent) => {
      if (e.key === 'Escape' && isOpen) {
        hideModal();
      }
    };

    document.addEventListener('keydown', handleEscape);
    return () => document.removeEventListener('keydown', handleEscape);
  }, [isOpen, hideModal]);

  useEffect(() => {
    if (isOpen) {
      document.body.style.overflow = 'hidden';
    } else {
      document.body.style.overflow = '';
    }
    return () => {
      document.body.style.overflow = '';
    };
  }, [isOpen]);

  if (!isOpen) return null;

  const handleOverlayClick = (e: React.MouseEvent) => {
    if (e.target === overlayRef.current) {
      hideModal();
    }
  };

  return (
    <div
      ref={overlayRef}
      onClick={handleOverlayClick}
      className="fixed inset-0 z-50 flex items-center justify-center
        bg-black/50 backdrop-blur-sm animate-fade-in"
    >
      <div
        className={`
          w-full mx-4 bg-slate-800 border border-slate-700 rounded-xl
          shadow-2xl animate-scale-in
          ${sizeStyles[size]}
        `}
      >
        {/* Header */}
        <div className="flex items-center justify-between px-6 py-4 border-b border-slate-700">
          <h2 className="text-lg font-semibold text-slate-100">{title}</h2>
          <button
            onClick={hideModal}
            className="p-1 hover:bg-slate-700 rounded-lg transition-colors"
          >
            <XMarkIcon className="w-5 h-5 text-slate-400" />
          </button>
        </div>

        {/* Body */}
        <div className="px-6 py-4">{children}</div>

        {/* Footer */}
        {footer && (
          <div className="flex items-center justify-end gap-3 px-6 py-4 border-t border-slate-700">
            {footer}
          </div>
        )}
      </div>
    </div>
  );
}
```

---

## 7. Layout Components

### 7.1 Layout

**src/components/layout/Layout.tsx**
```typescript
import { Outlet } from 'react-router-dom';
import { Sidebar } from './Sidebar';
import { Header } from './Header';
import { ToastContainer } from '@/components/common/Toast';
import { useUIStore } from '@/stores/uiStore';

export function Layout() {
  const sidebarCollapsed = useUIStore((s) => s.sidebarCollapsed);

  return (
    <div className="min-h-screen bg-slate-900 text-slate-100">
      <Sidebar />
      <div
        className={`
          transition-all duration-300
          ${sidebarCollapsed ? 'ml-16' : 'ml-60'}
        `}
      >
        <Header />
        <main className="p-6">
          <div className="max-w-7xl mx-auto">
            <Outlet />
          </div>
        </main>
      </div>
      <ToastContainer />
    </div>
  );
}
```

### 7.2 Sidebar

**src/components/layout/Sidebar.tsx**
```typescript
import { NavLink } from 'react-router-dom';
import {
  HomeIcon,
  CpuChipIcon,
  CubeIcon,
  AdjustmentsHorizontalIcon,
  ChartBarIcon,
  UserCircleIcon,
  Cog6ToothIcon,
  ChevronLeftIcon,
  ChevronRightIcon,
} from '@heroicons/react/24/outline';
import { useUIStore } from '@/stores/uiStore';

const navItems = [
  { path: '/', icon: HomeIcon, label: 'Dashboard' },
  { path: '/models', icon: CpuChipIcon, label: 'Models' },
  { path: '/sae', icon: CubeIcon, label: 'SAE' },
  { path: '/steering', icon: AdjustmentsHorizontalIcon, label: 'Steering' },
  { path: '/monitoring', icon: ChartBarIcon, label: 'Monitoring' },
  { path: '/profiles', icon: UserCircleIcon, label: 'Profiles' },
  { path: '/settings', icon: Cog6ToothIcon, label: 'Settings' },
];

export function Sidebar() {
  const { sidebarCollapsed, toggleSidebar } = useUIStore();

  return (
    <aside
      className={`
        fixed top-0 left-0 h-screen bg-slate-800 border-r border-slate-700
        transition-all duration-300 z-40
        ${sidebarCollapsed ? 'w-16' : 'w-60'}
      `}
    >
      {/* Logo */}
      <div className="h-16 flex items-center justify-center border-b border-slate-700">
        {sidebarCollapsed ? (
          <span className="text-xl font-bold text-blue-500">m</span>
        ) : (
          <span className="text-xl font-bold">
            <span className="text-blue-500">mi</span>LLM
          </span>
        )}
      </div>

      {/* Navigation */}
      <nav className="p-2 space-y-1">
        {navItems.map((item) => (
          <NavLink
            key={item.path}
            to={item.path}
            end={item.path === '/'}
            className={({ isActive }) => `
              flex items-center gap-3 px-3 py-2.5 rounded-lg
              transition-colors
              ${
                isActive
                  ? 'bg-blue-600/20 text-blue-400'
                  : 'text-slate-400 hover:bg-slate-700 hover:text-slate-200'
              }
            `}
          >
            <item.icon className="w-5 h-5 flex-shrink-0" />
            {!sidebarCollapsed && (
              <span className="font-medium">{item.label}</span>
            )}
          </NavLink>
        ))}
      </nav>

      {/* Collapse Toggle */}
      <button
        onClick={toggleSidebar}
        className="absolute bottom-4 right-0 translate-x-1/2
          p-1.5 bg-slate-700 border border-slate-600 rounded-full
          hover:bg-slate-600 transition-colors"
      >
        {sidebarCollapsed ? (
          <ChevronRightIcon className="w-4 h-4 text-slate-300" />
        ) : (
          <ChevronLeftIcon className="w-4 h-4 text-slate-300" />
        )}
      </button>
    </aside>
  );
}
```

### 7.3 Header

**src/components/layout/Header.tsx**
```typescript
import { useLocation } from 'react-router-dom';
import { SunIcon, MoonIcon } from '@heroicons/react/24/outline';
import { useUIStore } from '@/stores/uiStore';
import { useServerStore } from '@/stores/serverStore';
import { Badge } from '@/components/common/Badge';

const pageTitles: Record<string, string> = {
  '/': 'Dashboard',
  '/models': 'Model Management',
  '/sae': 'SAE Management',
  '/steering': 'Feature Steering',
  '/monitoring': 'Activation Monitoring',
  '/profiles': 'Profile Management',
  '/settings': 'Settings',
};

export function Header() {
  const location = useLocation();
  const { theme, toggleTheme } = useUIStore();
  const connectionStatus = useServerStore((s) => s.connectionStatus);

  const title = pageTitles[location.pathname] || 'miLLM';

  return (
    <header className="h-16 bg-slate-800/50 border-b border-slate-700 backdrop-blur-sm sticky top-0 z-30">
      <div className="h-full px-6 flex items-center justify-between">
        {/* Page Title */}
        <h1 className="text-xl font-semibold text-slate-100">{title}</h1>

        {/* Right Side */}
        <div className="flex items-center gap-4">
          {/* Connection Status */}
          <Badge
            variant={
              connectionStatus === 'connected'
                ? 'success'
                : connectionStatus === 'connecting'
                ? 'warning'
                : 'error'
            }
          >
            {connectionStatus === 'connected'
              ? 'Connected'
              : connectionStatus === 'connecting'
              ? 'Connecting...'
              : 'Disconnected'}
          </Badge>

          {/* Theme Toggle */}
          <button
            onClick={toggleTheme}
            className="p-2 hover:bg-slate-700 rounded-lg transition-colors"
          >
            {theme === 'dark' ? (
              <SunIcon className="w-5 h-5 text-slate-400" />
            ) : (
              <MoonIcon className="w-5 h-5 text-slate-400" />
            )}
          </button>
        </div>
      </div>
    </header>
  );
}
```

---

## 8. Custom Hooks

### 8.1 useToast

**src/hooks/useToast.ts**
```typescript
import { useCallback } from 'react';
import { useUIStore } from '@/stores/uiStore';

export function useToast() {
  const addToast = useUIStore((s) => s.addToast);

  return {
    success: useCallback(
      (message: string) => addToast({ type: 'success', message }),
      [addToast]
    ),
    error: useCallback(
      (message: string) => addToast({ type: 'error', message }),
      [addToast]
    ),
    warning: useCallback(
      (message: string) => addToast({ type: 'warning', message }),
      [addToast]
    ),
    info: useCallback(
      (message: string) => addToast({ type: 'info', message }),
      [addToast]
    ),
  };
}
```

### 8.2 useSteering

**src/hooks/useSteering.ts**
```typescript
import { useCallback } from 'react';
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { api } from '@/services/api';
import { useServerStore } from '@/stores/serverStore';
import { useToast } from './useToast';

export function useSteering() {
  const queryClient = useQueryClient();
  const toast = useToast();
  const store = useServerStore();

  // Initial fetch
  const query = useQuery({
    queryKey: ['steering'],
    queryFn: api.getSteering,
    refetchInterval: false, // WebSocket handles updates
  });

  // Set single value
  const setMutation = useMutation({
    mutationFn: api.setSteering,
    onMutate: async ({ feature_index, value }) => {
      // Optimistic update
      store.updateSteeringValue(feature_index, value);
    },
    onError: (_, { feature_index }) => {
      // Revert on error
      queryClient.invalidateQueries({ queryKey: ['steering'] });
      toast.error(`Failed to set steering for feature ${feature_index}`);
    },
  });

  // Batch update
  const batchMutation = useMutation({
    mutationFn: api.batchSteering,
    onSuccess: () => {
      toast.success('Steering values updated');
    },
    onError: () => {
      queryClient.invalidateQueries({ queryKey: ['steering'] });
      toast.error('Failed to update steering values');
    },
  });

  // Clear all
  const clearMutation = useMutation({
    mutationFn: api.clearSteering,
    onMutate: () => {
      store.clearSteeringValues();
    },
    onSuccess: () => {
      toast.success('Steering cleared');
    },
    onError: () => {
      queryClient.invalidateQueries({ queryKey: ['steering'] });
      toast.error('Failed to clear steering');
    },
  });

  // Toggle enabled
  const toggleMutation = useMutation({
    mutationFn: api.toggleSteering,
    onMutate: (enabled) => {
      store.setSteeringEnabled(enabled);
    },
    onError: () => {
      queryClient.invalidateQueries({ queryKey: ['steering'] });
      toast.error('Failed to toggle steering');
    },
  });

  return {
    isLoading: query.isLoading,
    error: query.error,
    enabled: store.steeringEnabled,
    values: store.steeringValues,
    setValue: useCallback(
      (feature: number, value: number) => {
        setMutation.mutate({ feature_index: feature, value });
      },
      [setMutation]
    ),
    setBatch: useCallback(
      (values: Record<number, number>) => {
        batchMutation.mutate({ values });
      },
      [batchMutation]
    ),
    clear: useCallback(() => {
      clearMutation.mutate();
    }, [clearMutation]),
    toggle: useCallback(
      (enabled: boolean) => {
        toggleMutation.mutate(enabled);
      },
      [toggleMutation]
    ),
    remove: useCallback(
      (feature: number) => {
        store.removeSteeringValue(feature);
        // Also clear on server
        api.setSteering({ feature_index: feature, value: 0 });
      },
      [store]
    ),
  };
}
```

---

## 9. Page Components

### 9.1 Dashboard Page

**src/pages/DashboardPage.tsx**
```typescript
import { useServerStore } from '@/stores/serverStore';
import { Card, CardHeader } from '@/components/common/Card';
import { Badge } from '@/components/common/Badge';
import {
  CpuChipIcon,
  CubeIcon,
  AdjustmentsHorizontalIcon,
  ChartBarIcon,
} from '@heroicons/react/24/outline';

interface StatusCardProps {
  title: string;
  icon: React.ComponentType<{ className?: string }>;
  status: 'active' | 'inactive' | 'loading';
  value: string;
  subtitle?: string;
}

function StatusCard({ title, icon: Icon, status, value, subtitle }: StatusCardProps) {
  const statusColors = {
    active: 'border-green-500/50',
    inactive: 'border-slate-600',
    loading: 'border-yellow-500/50',
  };

  return (
    <Card className={`border-l-4 ${statusColors[status]}`}>
      <div className="flex items-start gap-4">
        <div className="p-2 bg-slate-700 rounded-lg">
          <Icon className="w-6 h-6 text-slate-300" />
        </div>
        <div className="flex-1">
          <p className="text-sm text-slate-400">{title}</p>
          <p className="text-lg font-semibold text-slate-100 truncate">{value}</p>
          {subtitle && <p className="text-xs text-slate-500">{subtitle}</p>}
        </div>
        <Badge variant={status === 'active' ? 'success' : status === 'loading' ? 'warning' : 'default'}>
          {status === 'active' ? 'Active' : status === 'loading' ? 'Loading' : 'Inactive'}
        </Badge>
      </div>
    </Card>
  );
}

export function DashboardPage() {
  const {
    model,
    modelLoading,
    attachedSAE,
    saeLoading,
    steeringEnabled,
    steeringValues,
    monitoringEnabled,
  } = useServerStore();

  const steeringCount = Object.keys(steeringValues).length;

  return (
    <div className="space-y-6">
      {/* Status Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <StatusCard
          title="Model"
          icon={CpuChipIcon}
          status={modelLoading ? 'loading' : model ? 'active' : 'inactive'}
          value={model?.name || 'No model loaded'}
          subtitle={model?.device}
        />
        <StatusCard
          title="SAE"
          icon={CubeIcon}
          status={saeLoading ? 'loading' : attachedSAE ? 'active' : 'inactive'}
          value={attachedSAE?.repo_id.split('/').pop() || 'Not attached'}
          subtitle={attachedSAE ? `Layer ${attachedSAE.layer}` : undefined}
        />
        <StatusCard
          title="Steering"
          icon={AdjustmentsHorizontalIcon}
          status={steeringEnabled && steeringCount > 0 ? 'active' : 'inactive'}
          value={steeringCount > 0 ? `${steeringCount} features` : 'No steering'}
          subtitle={steeringEnabled ? 'Enabled' : 'Disabled'}
        />
        <StatusCard
          title="Monitoring"
          icon={ChartBarIcon}
          status={monitoringEnabled ? 'active' : 'inactive'}
          value={monitoringEnabled ? 'Active' : 'Inactive'}
        />
      </div>

      {/* Quick Actions or additional content */}
      <Card>
        <CardHeader title="Quick Start" />
        <div className="text-slate-400 text-sm">
          {!model && <p>1. Load a model from the Models page</p>}
          {model && !attachedSAE && <p>2. Attach an SAE from the SAE page</p>}
          {model && attachedSAE && steeringCount === 0 && (
            <p>3. Add steering values from the Steering page</p>
          )}
          {model && attachedSAE && steeringCount > 0 && (
            <p>Ready for inference with {steeringCount} steered features!</p>
          )}
        </div>
      </Card>
    </div>
  );
}
```

---

## 10. App Entry Point

### 10.1 Main Entry

**src/main.tsx**
```typescript
import React from 'react';
import ReactDOM from 'react-dom/client';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { App } from './App';
import { socketClient } from '@/services/socket';
import { useUIStore } from '@/stores/uiStore';
import './index.css';

// Initialize theme
const theme = useUIStore.getState().theme;
document.documentElement.classList.toggle('dark', theme === 'dark');

// Connect WebSocket
socketClient.connect();

// Create query client
const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      refetchOnWindowFocus: false,
      retry: 1,
    },
  },
});

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <QueryClientProvider client={queryClient}>
      <App />
    </QueryClientProvider>
  </React.StrictMode>
);
```

### 10.2 App Component

**src/App.tsx**
```typescript
import { createBrowserRouter, RouterProvider } from 'react-router-dom';
import { Layout } from '@/components/layout/Layout';
import { DashboardPage } from '@/pages/DashboardPage';
import { ModelsPage } from '@/pages/ModelsPage';
import { SAEPage } from '@/pages/SAEPage';
import { SteeringPage } from '@/pages/SteeringPage';
import { MonitoringPage } from '@/pages/MonitoringPage';
import { ProfilesPage } from '@/pages/ProfilesPage';
import { SettingsPage } from '@/pages/SettingsPage';

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

export function App() {
  return <RouterProvider router={router} />;
}
```

### 10.3 Global Styles

**src/index.css**
```css
@tailwind base;
@tailwind components;
@tailwind utilities;

@layer base {
  body {
    @apply bg-slate-900 text-slate-100 antialiased;
  }
}

@layer utilities {
  .animate-fade-in {
    animation: fadeIn 0.2s ease-out;
  }

  .animate-scale-in {
    animation: scaleIn 0.2s ease-out;
  }

  .animate-slide-in {
    animation: slideIn 0.3s ease-out;
  }

  @keyframes fadeIn {
    from { opacity: 0; }
    to { opacity: 1; }
  }

  @keyframes scaleIn {
    from { transform: scale(0.95); opacity: 0; }
    to { transform: scale(1); opacity: 1; }
  }

  @keyframes slideIn {
    from { transform: translateX(100%); opacity: 0; }
    to { transform: translateX(0); opacity: 1; }
  }
}
```

---

## 11. Testing Guidance

### 11.1 Test Setup

**vitest.config.ts**
```typescript
import { defineConfig } from 'vitest/config';
import react from '@vitejs/plugin-react';
import path from 'path';

export default defineConfig({
  plugins: [react()],
  test: {
    environment: 'jsdom',
    setupFiles: ['./tests/setup.ts'],
    globals: true,
  },
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
    },
  },
});
```

**tests/setup.ts**
```typescript
import '@testing-library/jest-dom';
import { vi } from 'vitest';

// Mock WebSocket
vi.mock('@/services/socket', () => ({
  socketClient: {
    connect: vi.fn(),
    disconnect: vi.fn(),
    reconnect: vi.fn(),
  },
}));
```

### 11.2 Component Test Example

**src/components/common/Button.test.tsx**
```typescript
import { render, screen, fireEvent } from '@testing-library/react';
import { Button } from './Button';

describe('Button', () => {
  it('renders children', () => {
    render(<Button>Click me</Button>);
    expect(screen.getByText('Click me')).toBeInTheDocument();
  });

  it('handles click events', () => {
    const handleClick = vi.fn();
    render(<Button onClick={handleClick}>Click</Button>);
    fireEvent.click(screen.getByText('Click'));
    expect(handleClick).toHaveBeenCalledTimes(1);
  });

  it('disables when loading', () => {
    render(<Button loading>Loading</Button>);
    expect(screen.getByRole('button')).toBeDisabled();
  });

  it('applies variant styles', () => {
    render(<Button variant="danger">Delete</Button>);
    expect(screen.getByRole('button')).toHaveClass('bg-red-600');
  });
});
```

---

**Document Status:** Complete
**Next Document:** 007_FTASKS|Admin_UI.md (Task List)
