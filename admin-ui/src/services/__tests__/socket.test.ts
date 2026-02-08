/**
 * WebSocket resilience tests for Socket.IO client.
 *
 * Tests connection handling, reconnection logic, and error scenarios.
 */

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';

// Use vi.hoisted so mock variables are available in vi.mock factories
const { mockSocket, mockIo, mockServerStore, mockUIStore } = vi.hoisted(() => {
  const mockSocket = {
    connected: false,
    active: false,
    on: vi.fn(),
    off: vi.fn(),
    emit: vi.fn(),
    connect: vi.fn(),
    disconnect: vi.fn(),
  };

  const mockIo = vi.fn(() => mockSocket);

  const mockServerStore = {
    setConnectionStatus: vi.fn(),
    setDownloadProgress: vi.fn(),
    clearDownloadProgress: vi.fn(),
    updateModel: vi.fn(),
    setModelLoading: vi.fn(),
    setLoadedModel: vi.fn(),
    setSAEDownloadProgress: vi.fn(),
    clearSAEDownloadProgress: vi.fn(),
    updateSAE: vi.fn(),
    setAttachedSAE: vi.fn(),
    setSteering: vi.fn(),
    addActivationRecord: vi.fn(),
    setSystemMetrics: vi.fn(),
    attachedSAE: null as { id: string; name: string } | null,
    monitoring: { enabled: false },
  };

  const mockUIStore = {
    addToast: vi.fn(),
    monitoringPaused: false,
  };

  return { mockSocket, mockIo, mockServerStore, mockUIStore };
});

vi.mock('socket.io-client', () => ({
  io: mockIo,
}));

vi.mock('@/stores/serverStore', () => ({
  useServerStore: {
    getState: () => mockServerStore,
  },
}));

vi.mock('@/stores/uiStore', () => ({
  useUIStore: {
    getState: () => mockUIStore,
  },
}));

// Import after mocks are set up
import { socketClient } from '../socket';

describe('SocketClient', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mockSocket.connected = false;
    mockSocket.on.mockClear();
    mockSocket.emit.mockClear();
    mockSocket.disconnect.mockClear();
    mockServerStore.setConnectionStatus.mockClear();
    mockUIStore.addToast.mockClear();
  });

  afterEach(() => {
    socketClient.disconnectImmediate();
  });

  describe('connect', () => {
    it('creates socket with correct configuration', () => {
      socketClient.connect('http://localhost:8000');

      expect(mockIo).toHaveBeenCalledWith('http://localhost:8000', {
        transports: ['websocket', 'polling'],
        reconnection: true,
        reconnectionAttempts: 5,
        reconnectionDelay: 1000,
      });
    });

    it('sets connection status to connecting', () => {
      socketClient.connect();

      expect(mockServerStore.setConnectionStatus).toHaveBeenCalledWith(
        'connecting'
      );
    });

    it('does not create new socket if already connected', () => {
      // First connect creates the socket
      socketClient.connect();
      const callCountAfterFirst = mockIo.mock.calls.length;

      // Mark as connected
      mockSocket.connected = true;

      // Second connect should be a no-op
      socketClient.connect();
      expect(mockIo.mock.calls.length).toBe(callCountAfterFirst);
    });
  });

  describe('connection event handlers', () => {
    beforeEach(() => {
      socketClient.connect();
    });

    it('sets connected status on connect event', () => {
      const connectHandler = mockSocket.on.mock.calls.find(
        (call) => call[0] === 'connect'
      )?.[1];

      connectHandler?.();

      expect(mockServerStore.setConnectionStatus).toHaveBeenCalledWith(
        'connected'
      );
    });

    it('sets disconnected status on disconnect event', () => {
      const disconnectHandler = mockSocket.on.mock.calls.find(
        (call) => call[0] === 'disconnect'
      )?.[1];

      disconnectHandler?.('transport close');

      expect(mockServerStore.setConnectionStatus).toHaveBeenCalledWith(
        'disconnected'
      );
    });

    it('attempts reconnect when server disconnects', () => {
      const disconnectHandler = mockSocket.on.mock.calls.find(
        (call) => call[0] === 'disconnect'
      )?.[1];

      disconnectHandler?.('io server disconnect');

      expect(mockSocket.connect).toHaveBeenCalled();
    });

    it('shows error toast after max reconnect attempts', () => {
      const connectErrorHandler = mockSocket.on.mock.calls.find(
        (call) => call[0] === 'connect_error'
      )?.[1];

      // Simulate max reconnect attempts
      for (let i = 0; i < 5; i++) {
        connectErrorHandler?.();
      }

      expect(mockServerStore.setConnectionStatus).toHaveBeenCalledWith('error');
      expect(mockUIStore.addToast).toHaveBeenCalledWith({
        type: 'error',
        message: 'Failed to connect to server',
      });
    });
  });

  describe('model event handlers', () => {
    beforeEach(() => {
      socketClient.connect();
    });

    it('handles model download progress', () => {
      const progressHandler = mockSocket.on.mock.calls.find(
        (call) => call[0] === 'model:download:progress'
      )?.[1];

      progressHandler?.({ modelId: 1, progress: 50 });

      expect(mockServerStore.setDownloadProgress).toHaveBeenCalledWith(1, 50);
    });

    it('handles model download complete', () => {
      const completeHandler = mockSocket.on.mock.calls.find(
        (call) => call[0] === 'model:download:complete'
      )?.[1];

      completeHandler?.({ modelId: 1 });

      expect(mockServerStore.updateModel).toHaveBeenCalledWith(1, {
        status: 'ready',
      });
      expect(mockServerStore.clearDownloadProgress).toHaveBeenCalledWith(1);
      expect(mockUIStore.addToast).toHaveBeenCalledWith({
        type: 'success',
        message: 'Model downloaded successfully',
      });
    });

    it('handles model download error with object error', () => {
      const errorHandler = mockSocket.on.mock.calls.find(
        (call) => call[0] === 'model:download:error'
      )?.[1];

      errorHandler?.({ modelId: 1, error: { code: 'GATED_MODEL_NO_TOKEN', message: 'Model is gated' } });

      expect(mockServerStore.clearDownloadProgress).toHaveBeenCalledWith(1);
      expect(mockServerStore.updateModel).toHaveBeenCalledWith(1, {
        status: 'error',
      });
      expect(mockUIStore.addToast).toHaveBeenCalledWith({
        type: 'error',
        message: 'Download failed: Model is gated',
      });
    });

    it('handles model load complete', () => {
      const loadHandler = mockSocket.on.mock.calls.find(
        (call) => call[0] === 'model:load:complete'
      )?.[1];

      loadHandler?.({ modelId: 1, memoryUsedMb: 4096 });

      expect(mockServerStore.updateModel).toHaveBeenCalledWith(1, {
        status: 'loaded',
      });
      expect(mockServerStore.setModelLoading).toHaveBeenCalledWith(false);
    });

    it('handles model load error with object error', () => {
      const errorHandler = mockSocket.on.mock.calls.find(
        (call) => call[0] === 'model:load:error'
      )?.[1];

      errorHandler?.({ modelId: 1, error: { code: 'MODEL_LOAD_FAILED', message: 'Out of memory' } });

      expect(mockServerStore.setModelLoading).toHaveBeenCalledWith(false);
      expect(mockServerStore.updateModel).toHaveBeenCalledWith(1, {
        status: 'error',
      });
      expect(mockUIStore.addToast).toHaveBeenCalledWith({
        type: 'error',
        message: 'Failed to load model: Out of memory',
      });
    });

    it('handles model unload complete', () => {
      const unloadHandler = mockSocket.on.mock.calls.find(
        (call) => call[0] === 'model:unload:complete'
      )?.[1];

      unloadHandler?.({ modelId: 1 });

      expect(mockServerStore.setLoadedModel).toHaveBeenCalledWith(null);
      expect(mockServerStore.updateModel).toHaveBeenCalledWith(1, {
        status: 'ready',
      });
    });
  });

  describe('SAE event handlers', () => {
    beforeEach(() => {
      socketClient.connect();
    });

    it('handles SAE download progress', () => {
      const progressHandler = mockSocket.on.mock.calls.find(
        (call) => call[0] === 'sae:download:progress'
      )?.[1];

      progressHandler?.({ saeId: 'sae-1', percent: 75 });

      expect(mockServerStore.setSAEDownloadProgress).toHaveBeenCalledWith(
        'sae-1',
        75
      );
    });

    it('handles SAE download complete', () => {
      const completeHandler = mockSocket.on.mock.calls.find(
        (call) => call[0] === 'sae:download:complete'
      )?.[1];

      completeHandler?.({ saeId: 'sae-1' });

      expect(mockServerStore.updateSAE).toHaveBeenCalledWith('sae-1', {
        status: 'cached',
      });
      expect(mockServerStore.clearSAEDownloadProgress).toHaveBeenCalledWith('sae-1');
    });

    it('handles SAE attached', () => {
      const attachHandler = mockSocket.on.mock.calls.find(
        (call) => call[0] === 'sae:attached'
      )?.[1];

      attachHandler?.({ saeId: 'sae-1', layer: 12, memoryMb: 256 });

      expect(mockServerStore.updateSAE).toHaveBeenCalledWith('sae-1', {
        status: 'attached',
      });
    });

    it('handles SAE detached', () => {
      mockServerStore.attachedSAE = { id: 'sae-1', name: 'test-sae' };

      const detachHandler = mockSocket.on.mock.calls.find(
        (call) => call[0] === 'sae:detached'
      )?.[1];

      detachHandler?.();

      expect(mockServerStore.setAttachedSAE).toHaveBeenCalledWith(null);
      expect(mockServerStore.updateSAE).toHaveBeenCalledWith('sae-1', {
        status: 'cached',
      });
    });
  });

  describe('monitoring event handlers', () => {
    beforeEach(() => {
      socketClient.connect();
    });

    it('handles activation events when not paused', () => {
      mockUIStore.monitoringPaused = false;

      const activationHandler = mockSocket.on.mock.calls.find(
        (call) => call[0] === 'monitoring:activation'
      )?.[1];

      const activationData = {
        timestamp: '2024-01-15T12:00:00Z',
        features: [{ idx: 1234, value: 0.85 }],
      };

      activationHandler?.(activationData);

      expect(mockServerStore.addActivationRecord).toHaveBeenCalledWith(
        activationData
      );
    });

    it('ignores activation events when paused', () => {
      mockUIStore.monitoringPaused = true;

      const activationHandler = mockSocket.on.mock.calls.find(
        (call) => call[0] === 'monitoring:activation'
      )?.[1];

      activationHandler?.({
        timestamp: '2024-01-15T12:00:00Z',
        features: [],
      });

      expect(mockServerStore.addActivationRecord).not.toHaveBeenCalled();
    });
  });

  describe('system metrics event handlers', () => {
    beforeEach(() => {
      socketClient.connect();
    });

    it('handles system metrics events', () => {
      const metricsHandler = mockSocket.on.mock.calls.find(
        (call) => call[0] === 'system:metrics'
      )?.[1];

      metricsHandler?.({
        gpu_memory_used_mb: 4096,
        gpu_memory_total_mb: 24576,
        gpu_utilization: 45,
        gpu_temperature: 52,
      });

      expect(mockServerStore.setSystemMetrics).toHaveBeenCalledWith({
        gpuMemoryUsed: 4096,
        gpuMemoryTotal: 24576,
        gpuUtilization: 45,
        gpuTemperature: 52,
      });
    });
  });

  describe('disconnect', () => {
    it('disconnects socket and sets status', () => {
      socketClient.connect();
      socketClient.disconnectImmediate();

      expect(mockSocket.disconnect).toHaveBeenCalled();
      expect(mockServerStore.setConnectionStatus).toHaveBeenCalledWith(
        'disconnected'
      );
    });
  });

  describe('reconnect', () => {
    it('disconnects and reconnects', () => {
      socketClient.connect();
      const initialCallCount = mockIo.mock.calls.length;

      socketClient.reconnect();

      expect(mockSocket.disconnect).toHaveBeenCalled();
      // Should create a new socket
      expect(mockIo.mock.calls.length).toBeGreaterThan(initialCallCount);
    });
  });

  describe('emit', () => {
    it('emits events through socket', () => {
      socketClient.connect();

      socketClient.emit('test:event', { data: 'test' });

      expect(mockSocket.emit).toHaveBeenCalledWith('test:event', {
        data: 'test',
      });
    });
  });

  describe('isConnected', () => {
    it('returns socket connection status', () => {
      socketClient.connect();
      mockSocket.connected = false;

      expect(socketClient.isConnected).toBe(false);

      mockSocket.connected = true;
      expect(socketClient.isConnected).toBe(true);
    });
  });
});

describe('SocketClient resilience', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mockSocket.connected = false;
  });

  afterEach(() => {
    socketClient.disconnectImmediate();
  });

  it('recovers from temporary disconnection', () => {
    socketClient.connect();

    // Simulate connection
    const connectHandler = mockSocket.on.mock.calls.find(
      (call) => call[0] === 'connect'
    )?.[1];
    connectHandler?.();

    expect(mockServerStore.setConnectionStatus).toHaveBeenCalledWith(
      'connected'
    );

    // Simulate disconnect
    const disconnectHandler = mockSocket.on.mock.calls.find(
      (call) => call[0] === 'disconnect'
    )?.[1];
    disconnectHandler?.('transport close');

    expect(mockServerStore.setConnectionStatus).toHaveBeenCalledWith(
      'disconnected'
    );

    // Simulate reconnection
    connectHandler?.();

    expect(mockServerStore.setConnectionStatus).toHaveBeenLastCalledWith(
      'connected'
    );
  });

  it('handles multiple rapid connect errors gracefully', () => {
    socketClient.connect();

    const connectErrorHandler = mockSocket.on.mock.calls.find(
      (call) => call[0] === 'connect_error'
    )?.[1];

    // Rapid fire errors
    for (let i = 0; i < 3; i++) {
      connectErrorHandler?.();
    }

    // Should not show error yet (under max attempts)
    expect(mockServerStore.setConnectionStatus).not.toHaveBeenCalledWith(
      'error'
    );

    // Two more to exceed max
    connectErrorHandler?.();
    connectErrorHandler?.();

    // Now should show error
    expect(mockServerStore.setConnectionStatus).toHaveBeenCalledWith('error');
  });

  it('resets reconnect attempts on successful connection', () => {
    socketClient.connect();

    const connectHandler = mockSocket.on.mock.calls.find(
      (call) => call[0] === 'connect'
    )?.[1];
    const connectErrorHandler = mockSocket.on.mock.calls.find(
      (call) => call[0] === 'connect_error'
    )?.[1];

    // First, reset counter by simulating a successful connection
    connectHandler?.();
    vi.clearAllMocks();

    // Now simulate some errors (should start from 0)
    connectErrorHandler?.();
    connectErrorHandler?.();
    connectErrorHandler?.();

    // Successful connection should reset counter
    connectHandler?.();
    vi.clearAllMocks();

    // Two more errors after reset shouldn't trigger max attempts (2 < 5)
    connectErrorHandler?.();
    connectErrorHandler?.();

    const errorCalls = mockServerStore.setConnectionStatus.mock.calls.filter(
      (call) => call[0] === 'error'
    );
    expect(errorCalls.length).toBe(0);
  });
});
