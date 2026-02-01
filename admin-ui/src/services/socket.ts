import { io, Socket } from 'socket.io-client';
import { useServerStore } from '@/stores/serverStore';
import { useUIStore } from '@/stores/uiStore';
import type {
  DownloadProgressEvent,
  LoadProgressEvent,
  ActivationEvent,
  SystemMetricsEvent,
  ModelInfo,
  SAEInfo,
  SteeringState,
} from '@/types';

type EventCallback<T> = (data: T) => void;

interface SocketEventHandlers {
  // Model events
  'model:download:progress': EventCallback<DownloadProgressEvent>;
  'model:download:complete': EventCallback<ModelInfo>;
  'model:download:error': EventCallback<{ model_id: number; error: string }>;
  'model:load:progress': EventCallback<LoadProgressEvent>;
  'model:load:complete': EventCallback<ModelInfo>;
  'model:load:error': EventCallback<{ model_id: number; error: string }>;
  'model:unload:complete': EventCallback<{ model_id: number }>;

  // SAE events
  'sae:download:progress': EventCallback<DownloadProgressEvent>;
  'sae:download:complete': EventCallback<SAEInfo>;
  'sae:download:error': EventCallback<{ sae_id: number; error: string }>;
  'sae:attach:complete': EventCallback<SAEInfo>;
  'sae:detach:complete': EventCallback<void>;

  // Steering events
  'steering:update': EventCallback<SteeringState>;

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

    // Model download events
    this.socket.on('model:download:progress', (data: DownloadProgressEvent) => {
      if (data.model_id) {
        serverStore.setDownloadProgress(data.model_id, data.progress);
      }
    });

    this.socket.on('model:download:complete', (data: ModelInfo) => {
      serverStore.updateModel(data.id, data);
      serverStore.clearDownloadProgress(data.id);
      uiStore.addToast({
        type: 'success',
        message: `Model "${data.name}" downloaded successfully`,
      });
    });

    this.socket.on(
      'model:download:error',
      (data: { model_id: number; error: string }) => {
        serverStore.clearDownloadProgress(data.model_id);
        serverStore.updateModel(data.model_id, { status: 'error' });
        uiStore.addToast({
          type: 'error',
          message: `Download failed: ${data.error}`,
        });
      }
    );

    // Model load events
    this.socket.on(
      'model:load:progress',
      // eslint-disable-next-line @typescript-eslint/no-unused-vars
      (_data: LoadProgressEvent) => {
        // Could update a loading progress indicator
        serverStore.setModelLoading(true);
      }
    );

    this.socket.on('model:load:complete', (data: ModelInfo) => {
      serverStore.setLoadedModel(data);
      serverStore.updateModel(data.id, { status: 'loaded' });
      serverStore.setModelLoading(false);
      uiStore.addToast({
        type: 'success',
        message: `Model "${data.name}" loaded successfully`,
      });
    });

    this.socket.on(
      'model:load:error',
      (data: { model_id: number; error: string }) => {
        serverStore.setModelLoading(false);
        serverStore.updateModel(data.model_id, { status: 'error' });
        uiStore.addToast({
          type: 'error',
          message: `Failed to load model: ${data.error}`,
        });
      }
    );

    this.socket.on('model:unload:complete', (data: { model_id: number }) => {
      serverStore.setLoadedModel(null);
      serverStore.updateModel(data.model_id, { status: 'ready' });
      uiStore.addToast({
        type: 'info',
        message: 'Model unloaded',
      });
    });

    // SAE events
    this.socket.on('sae:download:progress', (data: DownloadProgressEvent) => {
      if (data.sae_id) {
        serverStore.setSAEDownloadProgress(data.sae_id, data.progress);
      }
    });

    this.socket.on('sae:download:complete', (data: SAEInfo) => {
      serverStore.updateSAE(data.id, data);
      serverStore.clearSAEDownloadProgress(data.id);
      uiStore.addToast({
        type: 'success',
        message: `SAE "${data.name}" downloaded successfully`,
      });
    });

    this.socket.on(
      'sae:download:error',
      (data: { sae_id: string; error: string }) => {
        serverStore.clearSAEDownloadProgress(data.sae_id);
        serverStore.updateSAE(data.sae_id, { status: 'error' });
        uiStore.addToast({
          type: 'error',
          message: `SAE download failed: ${data.error}`,
        });
      }
    );

    this.socket.on('sae:attach:complete', (data: SAEInfo) => {
      serverStore.setAttachedSAE(data);
      serverStore.updateSAE(data.id, { status: 'attached' });
      uiStore.addToast({
        type: 'success',
        message: `SAE "${data.name}" attached`,
      });
    });

    this.socket.on('sae:detach:complete', () => {
      const attachedSAE = serverStore.attachedSAE;
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
    this.socket.on('steering:update', (data: SteeringState) => {
      serverStore.setSteering(data);
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
