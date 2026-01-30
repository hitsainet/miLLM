/**
 * Tests for Model Zustand store.
 */

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { act } from '@testing-library/react';
import { useModelStore } from './modelStore';
import { modelService, socketService } from '../services';
import { Model, ModelStatus, QuantizationType } from '../types';

// Mock services
vi.mock('../services', () => ({
  modelService: {
    listModels: vi.fn(),
    downloadModel: vi.fn(),
    loadModel: vi.fn(),
    unloadModel: vi.fn(),
    deleteModel: vi.fn(),
    cancelDownload: vi.fn(),
  },
  socketService: {
    connect: vi.fn(),
    disconnect: vi.fn(),
    onDownloadProgress: vi.fn(() => vi.fn()),
    onDownloadComplete: vi.fn(() => vi.fn()),
    onDownloadError: vi.fn(() => vi.fn()),
    onLoadComplete: vi.fn(() => vi.fn()),
    onUnloadComplete: vi.fn(() => vi.fn()),
  },
}));

const mockModel: Model = {
  id: 1,
  name: 'test-model',
  repoId: 'test/model',
  source: 'huggingface',
  status: 'ready' as ModelStatus,
  quantization: 'fp16' as QuantizationType,
  createdAt: '2024-01-01T00:00:00Z',
};

describe('useModelStore', () => {
  beforeEach(() => {
    // Reset store state before each test
    useModelStore.setState({
      models: [],
      loadedModelId: null,
      isLoading: false,
      downloadProgress: {},
      error: null,
    });
    vi.clearAllMocks();
  });

  afterEach(() => {
    vi.resetAllMocks();
  });

  describe('initial state', () => {
    it('should have correct initial state', () => {
      const state = useModelStore.getState();

      expect(state.models).toEqual([]);
      expect(state.loadedModelId).toBeNull();
      expect(state.isLoading).toBe(false);
      expect(state.downloadProgress).toEqual({});
      expect(state.error).toBeNull();
    });
  });

  describe('fetchModels', () => {
    it('should fetch models and update state', async () => {
      const mockModels = [mockModel];
      vi.mocked(modelService.listModels).mockResolvedValue(mockModels);

      await act(async () => {
        await useModelStore.getState().fetchModels();
      });

      const state = useModelStore.getState();
      expect(state.models).toEqual(mockModels);
      expect(state.isLoading).toBe(false);
      expect(state.error).toBeNull();
    });

    it('should set loadedModelId when a model is loaded', async () => {
      const loadedModel = { ...mockModel, status: 'loaded' as ModelStatus };
      vi.mocked(modelService.listModels).mockResolvedValue([loadedModel]);

      await act(async () => {
        await useModelStore.getState().fetchModels();
      });

      const state = useModelStore.getState();
      expect(state.loadedModelId).toBe(1);
    });

    it('should handle fetch error', async () => {
      vi.mocked(modelService.listModels).mockRejectedValue(new Error('Network error'));

      await act(async () => {
        await useModelStore.getState().fetchModels();
      });

      const state = useModelStore.getState();
      expect(state.error).toBe('Failed to fetch models');
      expect(state.isLoading).toBe(false);
    });
  });

  describe('downloadModel', () => {
    it('should add model to list on successful download start', async () => {
      const downloadingModel = { ...mockModel, status: 'downloading' as ModelStatus };
      vi.mocked(modelService.downloadModel).mockResolvedValue(downloadingModel);

      await act(async () => {
        await useModelStore.getState().downloadModel('test/model', 'fp16');
      });

      const state = useModelStore.getState();
      expect(state.models).toContainEqual(downloadingModel);
      expect(modelService.downloadModel).toHaveBeenCalledWith({
        source: 'huggingface',
        repoId: 'test/model',
        quantization: 'fp16',
      });
    });

    it('should handle download error', async () => {
      vi.mocked(modelService.downloadModel).mockRejectedValue(new Error('Download failed'));

      await expect(
        act(async () => {
          await useModelStore.getState().downloadModel('test/model', 'fp16');
        })
      ).rejects.toThrow();

      const state = useModelStore.getState();
      expect(state.error).toBe('Failed to start download');
    });
  });

  describe('loadModel', () => {
    it('should update model status on load', async () => {
      const loadedModel = { ...mockModel, status: 'loaded' as ModelStatus };
      useModelStore.setState({ models: [mockModel] });
      vi.mocked(modelService.loadModel).mockResolvedValue(loadedModel);

      await act(async () => {
        await useModelStore.getState().loadModel(1);
      });

      const state = useModelStore.getState();
      expect(state.models[0].status).toBe('loaded');
    });

    it('should handle load error', async () => {
      useModelStore.setState({ models: [mockModel] });
      vi.mocked(modelService.loadModel).mockRejectedValue(new Error('Load failed'));

      await expect(
        act(async () => {
          await useModelStore.getState().loadModel(1);
        })
      ).rejects.toThrow();

      const state = useModelStore.getState();
      expect(state.error).toBe('Failed to load model');
    });
  });

  describe('unloadModel', () => {
    it('should update model status and clear loadedModelId on unload', async () => {
      const loadedModel = { ...mockModel, status: 'loaded' as ModelStatus };
      const readyModel = { ...mockModel, status: 'ready' as ModelStatus };
      useModelStore.setState({ models: [loadedModel], loadedModelId: 1 });
      vi.mocked(modelService.unloadModel).mockResolvedValue(readyModel);

      await act(async () => {
        await useModelStore.getState().unloadModel(1);
      });

      const state = useModelStore.getState();
      expect(state.models[0].status).toBe('ready');
      expect(state.loadedModelId).toBeNull();
    });
  });

  describe('deleteModel', () => {
    it('should remove model from list on delete', async () => {
      useModelStore.setState({ models: [mockModel] });
      vi.mocked(modelService.deleteModel).mockResolvedValue(undefined);

      await act(async () => {
        await useModelStore.getState().deleteModel(1);
      });

      const state = useModelStore.getState();
      expect(state.models).toHaveLength(0);
    });
  });

  describe('cancelDownload', () => {
    it('should update model and clear download progress', async () => {
      const downloadingModel = { ...mockModel, status: 'downloading' as ModelStatus };
      const errorModel = { ...mockModel, status: 'error' as ModelStatus };
      useModelStore.setState({
        models: [downloadingModel],
        downloadProgress: { 1: { modelId: 1, progress: 50, downloadedBytes: 500, totalBytes: 1000 } },
      });
      vi.mocked(modelService.cancelDownload).mockResolvedValue(errorModel);

      await act(async () => {
        await useModelStore.getState().cancelDownload(1);
      });

      const state = useModelStore.getState();
      expect(state.models[0].status).toBe('error');
      expect(state.downloadProgress[1]).toBeUndefined();
    });
  });

  describe('clearError', () => {
    it('should clear error state', () => {
      useModelStore.setState({ error: 'Some error' });

      act(() => {
        useModelStore.getState().clearError();
      });

      const state = useModelStore.getState();
      expect(state.error).toBeNull();
    });
  });

  describe('socket event handlers', () => {
    describe('handleDownloadProgress', () => {
      it('should update download progress for model', () => {
        const progress = { modelId: 1, progress: 50, downloadedBytes: 500, totalBytes: 1000 };

        act(() => {
          useModelStore.getState().handleDownloadProgress(progress);
        });

        const state = useModelStore.getState();
        expect(state.downloadProgress[1]).toEqual(progress);
      });
    });

    describe('handleDownloadComplete', () => {
      it('should update model status and clear progress', () => {
        const downloadingModel = { ...mockModel, status: 'downloading' as ModelStatus };
        useModelStore.setState({
          models: [downloadingModel],
          downloadProgress: { 1: { modelId: 1, progress: 100, downloadedBytes: 1000, totalBytes: 1000 } },
        });
        vi.mocked(modelService.listModels).mockResolvedValue([mockModel]);

        act(() => {
          useModelStore.getState().handleDownloadComplete({ modelId: 1 });
        });

        const state = useModelStore.getState();
        expect(state.models[0].status).toBe('ready');
        expect(state.downloadProgress[1]).toBeUndefined();
      });
    });

    describe('handleDownloadError', () => {
      it('should update model status and set error', () => {
        const downloadingModel = { ...mockModel, status: 'downloading' as ModelStatus };
        useModelStore.setState({
          models: [downloadingModel],
          downloadProgress: { 1: { modelId: 1, progress: 50, downloadedBytes: 500, totalBytes: 1000 } },
        });

        act(() => {
          useModelStore.getState().handleDownloadError({
            modelId: 1,
            error: { code: 'DOWNLOAD_FAILED', message: 'Network error' },
          });
        });

        const state = useModelStore.getState();
        expect(state.models[0].status).toBe('error');
        expect(state.models[0].errorMessage).toBe('Network error');
        expect(state.error).toBe('Network error');
        expect(state.downloadProgress[1]).toBeUndefined();
      });
    });

    describe('handleLoadComplete', () => {
      it('should update model status and set loadedModelId', () => {
        useModelStore.setState({ models: [mockModel] });
        vi.mocked(modelService.listModels).mockResolvedValue([{ ...mockModel, status: 'loaded' }]);

        act(() => {
          useModelStore.getState().handleLoadComplete({ modelId: 1 });
        });

        const state = useModelStore.getState();
        expect(state.models[0].status).toBe('loaded');
        expect(state.loadedModelId).toBe(1);
      });
    });

    describe('handleUnloadComplete', () => {
      it('should update model status and clear loadedModelId', () => {
        const loadedModel = { ...mockModel, status: 'loaded' as ModelStatus };
        useModelStore.setState({ models: [loadedModel], loadedModelId: 1 });

        act(() => {
          useModelStore.getState().handleUnloadComplete({ modelId: 1 });
        });

        const state = useModelStore.getState();
        expect(state.models[0].status).toBe('ready');
        expect(state.loadedModelId).toBeNull();
      });
    });
  });

  describe('initSocketListeners', () => {
    it('should connect to socket and register listeners', () => {
      const cleanup = useModelStore.getState().initSocketListeners();

      expect(socketService.connect).toHaveBeenCalled();
      expect(socketService.onDownloadProgress).toHaveBeenCalled();
      expect(socketService.onDownloadComplete).toHaveBeenCalled();
      expect(socketService.onDownloadError).toHaveBeenCalled();
      expect(socketService.onLoadComplete).toHaveBeenCalled();
      expect(socketService.onUnloadComplete).toHaveBeenCalled();

      // Cleanup
      cleanup();
      expect(socketService.disconnect).toHaveBeenCalled();
    });
  });
});
