import { io, Socket } from 'socket.io-client';
import { useServerStore } from '@/stores/serverStore';
import { useUIStore } from '@/stores/uiStore';
import type {
  ActivationEvent,
  SystemMetricsEvent,
} from '@/types';

type EventCallback<T> = (data: T) => void;

interface SocketEventHandlers {
  // Model events (backend uses camelCase keys)
  'model:download:progress': EventCallback<{ modelId: number; progress: number }>;
  'model:download:complete': EventCallback<{ modelId: number }>;
  'model:download:error': EventCallback<{ modelId: number; error: unknown }>;
  'model:load:progress': EventCallback<{ modelId: number; stage: string; progress: number }>;
  'model:load:complete': EventCallback<{ modelId: number; memoryUsedMb: number }>;
  'model:load:error': EventCallback<{ modelId: number; error: unknown }>;
  'model:unload:complete': EventCallback<{ modelId: number }>;

  // SAE events (backend uses camelCase keys)
  'sae:download:progress': EventCallback<{ saeId: string; percent: number }>;
  'sae:download:complete': EventCallback<{ saeId: string }>;
  'sae:download:error': EventCallback<{ saeId: string; error: string }>;
  'sae:attached': EventCallback<{ saeId: string; layer: number; memoryMb: number }>;
  'sae:detached': EventCallback<{ saeId: string }>;

  // Steering events
  'steering:update': EventCallback<{ enabled: boolean; values: Record<string, number>; activeCount?: number }>;

  // Monitoring events
  'monitoring:activation': EventCallback<ActivationEvent>;

  // System events
  'system:metrics': EventCallback<SystemMetricsEvent>;
}

class SocketClient {
  private socket: Socket | null = null;
  private reconnectAttempts = 0;
  private maxReconnectAttempts = 5;
  private reconnectDelay = 1000;
  private disconnectTimeout: ReturnType<typeof setTimeout> | null = null;

  connect(url: string = ''): void {
    // Cancel any pending disconnect (handles React StrictMode double-invoke)
    if (this.disconnectTimeout) {
      clearTimeout(this.disconnectTimeout);
      this.disconnectTimeout = null;
    }

    // If already connected or connecting, don't create new socket
    if (this.socket?.connected || this.socket?.active) {
      return;
    }

    const serverStore = useServerStore.getState();
    serverStore.setConnectionStatus('connecting');

    this.socket = io(url, {
      transports: ['websocket', 'polling'],
      reconnection: true,
      reconnectionAttempts: this.maxReconnectAttempts,
      reconnectionDelay: this.reconnectDelay,
    });

    this.setupEventHandlers();
    this.setupConnectionHandlers();
  }

  private setupConnectionHandlers(): void {
    if (!this.socket) return;

    const serverStore = useServerStore.getState();
    const uiStore = useUIStore.getState();

    this.socket.on('connect', () => {
      serverStore.setConnectionStatus('connected');
      this.reconnectAttempts = 0;
      this.joinRooms();
    });

    this.socket.on('disconnect', (reason) => {
      serverStore.setConnectionStatus('disconnected');
      if (reason === 'io server disconnect') {
        // Server disconnected, try to reconnect
        this.socket?.connect();
      }
    });

    this.socket.on('connect_error', () => {
      this.reconnectAttempts++;
      if (this.reconnectAttempts >= this.maxReconnectAttempts) {
        serverStore.setConnectionStatus('error');
        uiStore.addToast({
          type: 'error',
          message: 'Failed to connect to server',
        });
      }
    });
  }

  private setupEventHandlers(): void {
    if (!this.socket) return;

    const serverStore = useServerStore.getState();
    const uiStore = useUIStore.getState();

    // Helper to extract error message from backend error payloads
    // Backend sends error as { code, message, details } object or plain string
    const getErrorMessage = (error: unknown): string => {
      if (typeof error === 'string') return error;
      if (error && typeof error === 'object' && 'message' in error) {
        return (error as { message: string }).message;
      }
      return 'Unknown error';
    };

    // Model download events
    // Backend sends: { modelId, progress, downloadedBytes, totalBytes, speedBps? }
    this.socket.on('model:download:progress', (data: { modelId: number; progress: number }) => {
      if (data.modelId) {
        serverStore.setDownloadProgress(data.modelId, data.progress);
      }
    });

    // Backend sends: { modelId, localPath }
    this.socket.on('model:download:complete', (data: { modelId: number }) => {
      serverStore.updateModel(data.modelId, { status: 'ready' });
      serverStore.clearDownloadProgress(data.modelId);
      uiStore.addToast({
        type: 'success',
        message: 'Model downloaded successfully',
      });
    });

    // Backend sends: { modelId, error: { code, message, details } }
    this.socket.on(
      'model:download:error',
      (data: { modelId: number; error: unknown }) => {
        serverStore.clearDownloadProgress(data.modelId);
        serverStore.updateModel(data.modelId, { status: 'error' });
        uiStore.addToast({
          type: 'error',
          message: `Download failed: ${getErrorMessage(data.error)}`,
        });
      }
    );

    // Model load events
    // Backend sends: { modelId, stage, progress }
    this.socket.on(
      'model:load:progress',
      // eslint-disable-next-line @typescript-eslint/no-unused-vars
      (_data: { modelId: number; stage: string; progress: number }) => {
        serverStore.setModelLoading(true);
      }
    );

    // Backend sends: { modelId, memoryUsedMb }
    this.socket.on('model:load:complete', (data: { modelId: number; memoryUsedMb: number }) => {
      serverStore.updateModel(data.modelId, { status: 'loaded' });
      serverStore.setModelLoading(false);
      uiStore.addToast({
        type: 'success',
        message: 'Model loaded successfully',
      });
    });

    // Backend sends: { modelId, error: { code, message } }
    this.socket.on(
      'model:load:error',
      (data: { modelId: number; error: unknown }) => {
        serverStore.setModelLoading(false);
        serverStore.updateModel(data.modelId, { status: 'error' });
        uiStore.addToast({
          type: 'error',
          message: `Failed to load model: ${getErrorMessage(data.error)}`,
        });
      }
    );

    // Backend sends: { modelId }
    this.socket.on('model:unload:complete', (data: { modelId: number }) => {
      serverStore.setLoadedModel(null);
      serverStore.updateModel(data.modelId, { status: 'ready' });
      uiStore.addToast({
        type: 'info',
        message: 'Model unloaded',
      });
    });

    // SAE events
    // Backend sends: { saeId, percent }
    this.socket.on('sae:download:progress', (data: { saeId: string; percent: number }) => {
      if (data.saeId) {
        serverStore.setSAEDownloadProgress(data.saeId, data.percent);
      }
    });

    // Backend sends: { saeId }
    this.socket.on('sae:download:complete', (data: { saeId: string }) => {
      serverStore.updateSAE(data.saeId, { status: 'cached' });
      serverStore.clearSAEDownloadProgress(data.saeId);
      uiStore.addToast({
        type: 'success',
        message: 'SAE downloaded successfully',
      });
    });

    // Backend sends: { saeId, error } (error is a plain string)
    this.socket.on(
      'sae:download:error',
      (data: { saeId: string; error: string }) => {
        serverStore.clearSAEDownloadProgress(data.saeId);
        serverStore.updateSAE(data.saeId, { status: 'error' });
        uiStore.addToast({
          type: 'error',
          message: `SAE download failed: ${data.error}`,
        });
      }
    );

    // Backend emits 'sae:attached' with { saeId, layer, memoryMb }
    this.socket.on('sae:attached', (data: { saeId: string; layer: number; memoryMb: number }) => {
      serverStore.updateSAE(data.saeId, { status: 'attached' });
      uiStore.addToast({
        type: 'success',
        message: 'SAE attached',
      });
    });

    // Backend emits 'sae:detached' with { saeId }
    this.socket.on('sae:detached', () => {
      const attachedSAE = useServerStore.getState().attachedSAE;
      if (attachedSAE) {
        serverStore.updateSAE(attachedSAE.id, { status: 'cached' });
      }
      serverStore.setAttachedSAE(null);
      uiStore.addToast({
        type: 'info',
        message: 'SAE detached',
      });
    });

    // Steering events
    // Backend sends { enabled, values: {idx: strength}, activeCount }
    // Frontend expects { enabled, sae_id, features: [{index, strength}] }
    this.socket.on('steering:update', (data: { enabled: boolean; values: Record<string, number>; activeCount?: number }) => {
      const features = Object.entries(data.values || {}).map(([index, strength]) => ({
        index: parseInt(index, 10),
        strength,
      }));
      serverStore.setSteering({
        enabled: data.enabled,
        sae_id: null,
        features,
      });
    });

    // Monitoring events
    this.socket.on('monitoring:activation', (data: ActivationEvent) => {
      const uiState = useUIStore.getState();
      if (!uiState.monitoringPaused) {
        serverStore.addActivationRecord(data);
      }
    });

    // System events
    this.socket.on('system:metrics', (data: SystemMetricsEvent) => {
      serverStore.setSystemMetrics({
        gpuMemoryUsed: data.gpu_memory_used_mb,
        gpuMemoryTotal: data.gpu_memory_total_mb,
        gpuUtilization: data.gpu_utilization,
        gpuTemperature: data.gpu_temperature,
      });
    });
  }

  private joinRooms(): void {
    if (!this.socket?.connected) return;

    // Join monitoring room if enabled
    const serverStore = useServerStore.getState();
    if (serverStore.monitoring.enabled) {
      this.socket.emit('monitoring:join');
    }

    // Join system metrics room
    this.socket.emit('system:join');
  }

  disconnect(): void {
    // Use delayed disconnect to handle React StrictMode double-invoke
    // If connect() is called within 100ms, the disconnect is cancelled
    if (this.disconnectTimeout) {
      clearTimeout(this.disconnectTimeout);
    }

    this.disconnectTimeout = setTimeout(() => {
      if (this.socket) {
        this.socket.disconnect();
        this.socket = null;
      }
      useServerStore.getState().setConnectionStatus('disconnected');
      this.disconnectTimeout = null;
    }, 100);
  }

  disconnectImmediate(): void {
    // For cases where immediate disconnect is needed
    if (this.disconnectTimeout) {
      clearTimeout(this.disconnectTimeout);
      this.disconnectTimeout = null;
    }
    if (this.socket) {
      this.socket.disconnect();
      this.socket = null;
    }
    useServerStore.getState().setConnectionStatus('disconnected');
  }

  reconnect(): void {
    this.disconnectImmediate();
    this.connect();
  }

  emit<T>(event: string, data?: T): void {
    this.socket?.emit(event, data);
  }

  on<K extends keyof SocketEventHandlers>(
    event: K,
    handler: SocketEventHandlers[K]
  ): void {
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    this.socket?.on(event, handler as any);
  }

  off<K extends keyof SocketEventHandlers>(event: K): void {
    this.socket?.off(event);
  }

  get isConnected(): boolean {
    return this.socket?.connected ?? false;
  }
}

export const socketClient = new SocketClient();
export default socketClient;
