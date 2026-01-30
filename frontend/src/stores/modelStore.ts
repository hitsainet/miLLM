/**
 * Zustand store for model state management.
 */

import { create } from 'zustand';
import { Model, DownloadProgress, ApiError, isApiError } from '../types';
import { modelService, socketService } from '../services';

interface ModelState {
  // State
  models: Model[];
  loadedModelId: number | null;
  isLoading: boolean;
  downloadProgress: Record<number, DownloadProgress>;
  error: string | null;

  // Actions
  fetchModels: () => Promise<void>;
  downloadModel: (repoId: string, quantization: string) => Promise<void>;
  loadModel: (modelId: number) => Promise<void>;
  unloadModel: (modelId: number) => Promise<void>;
  deleteModel: (modelId: number) => Promise<void>;
  cancelDownload: (modelId: number) => Promise<void>;
  clearError: () => void;

  // Socket event handlers
  handleDownloadProgress: (data: DownloadProgress) => void;
  handleDownloadComplete: (data: { modelId: number }) => void;
  handleDownloadError: (data: { modelId: number; error: { code: string; message: string } }) => void;
  handleLoadComplete: (data: { modelId: number }) => void;
  handleUnloadComplete: (data: { modelId: number }) => void;

  // Initialize socket listeners
  initSocketListeners: () => () => void;
}

export const useModelStore = create<ModelState>((set, get) => ({
  // Initial state
  models: [],
  loadedModelId: null,
  isLoading: false,
  downloadProgress: {},
  error: null,

  // Actions
  fetchModels: async () => {
    set({ isLoading: true, error: null });
    try {
      const models = await modelService.listModels();
      const loadedModel = models.find((m) => m.status === 'loaded');
      set({
        models,
        loadedModelId: loadedModel?.id || null,
        isLoading: false,
      });
    } catch (error) {
      const message = isApiError(error) ? error.message : 'Failed to fetch models';
      set({ error: message, isLoading: false });
    }
  },

  downloadModel: async (repoId: string, quantization: string) => {
    set({ error: null });
    try {
      const model = await modelService.downloadModel({
        source: 'huggingface',
        repoId,
        quantization: quantization as 'Q4' | 'Q8' | 'FP16',
      });
      set((state) => ({
        models: [model, ...state.models],
      }));
    } catch (error) {
      const message = isApiError(error) ? error.message : 'Failed to start download';
      set({ error: message });
      throw error;
    }
  },

  loadModel: async (modelId: number) => {
    set({ error: null });
    try {
      const model = await modelService.loadModel(modelId);
      set((state) => ({
        models: state.models.map((m) => (m.id === modelId ? model : m)),
      }));
    } catch (error) {
      const message = isApiError(error) ? error.message : 'Failed to load model';
      set({ error: message });
      throw error;
    }
  },

  unloadModel: async (modelId: number) => {
    set({ error: null });
    try {
      const model = await modelService.unloadModel(modelId);
      set((state) => ({
        models: state.models.map((m) => (m.id === modelId ? model : m)),
        loadedModelId: null,
      }));
    } catch (error) {
      const message = isApiError(error) ? error.message : 'Failed to unload model';
      set({ error: message });
      throw error;
    }
  },

  deleteModel: async (modelId: number) => {
    set({ error: null });
    try {
      await modelService.deleteModel(modelId);
      set((state) => ({
        models: state.models.filter((m) => m.id !== modelId),
      }));
    } catch (error) {
      const message = isApiError(error) ? error.message : 'Failed to delete model';
      set({ error: message });
      throw error;
    }
  },

  cancelDownload: async (modelId: number) => {
    set({ error: null });
    try {
      const model = await modelService.cancelDownload(modelId);
      set((state) => ({
        models: state.models.map((m) => (m.id === modelId ? model : m)),
        downloadProgress: {
          ...state.downloadProgress,
          [modelId]: undefined as unknown as DownloadProgress,
        },
      }));
    } catch (error) {
      const message = isApiError(error) ? error.message : 'Failed to cancel download';
      set({ error: message });
      throw error;
    }
  },

  clearError: () => set({ error: null }),

  // Socket event handlers
  handleDownloadProgress: (data: DownloadProgress) => {
    set((state) => ({
      downloadProgress: {
        ...state.downloadProgress,
        [data.modelId]: data,
      },
    }));
  },

  handleDownloadComplete: (data: { modelId: number }) => {
    set((state) => {
      const { [data.modelId]: _, ...rest } = state.downloadProgress;
      return {
        downloadProgress: rest,
        models: state.models.map((m) =>
          m.id === data.modelId ? { ...m, status: 'ready' as const } : m
        ),
      };
    });
    // Refresh to get updated model data
    get().fetchModels();
  },

  handleDownloadError: (data: { modelId: number; error: { code: string; message: string } }) => {
    set((state) => {
      const { [data.modelId]: _, ...rest } = state.downloadProgress;
      return {
        downloadProgress: rest,
        models: state.models.map((m) =>
          m.id === data.modelId
            ? { ...m, status: 'error' as const, errorMessage: data.error.message }
            : m
        ),
        error: data.error.message,
      };
    });
  },

  handleLoadComplete: (data: { modelId: number }) => {
    set((state) => ({
      models: state.models.map((m) =>
        m.id === data.modelId ? { ...m, status: 'loaded' as const } : m
      ),
      loadedModelId: data.modelId,
    }));
    // Refresh to get updated model data
    get().fetchModels();
  },

  handleUnloadComplete: (data: { modelId: number }) => {
    set((state) => ({
      models: state.models.map((m) =>
        m.id === data.modelId ? { ...m, status: 'ready' as const } : m
      ),
      loadedModelId: null,
    }));
  },

  // Initialize socket listeners
  initSocketListeners: () => {
    socketService.connect();

    const unsubProgress = socketService.onDownloadProgress(get().handleDownloadProgress);
    const unsubComplete = socketService.onDownloadComplete(get().handleDownloadComplete);
    const unsubError = socketService.onDownloadError(get().handleDownloadError);
    const unsubLoadComplete = socketService.onLoadComplete(get().handleLoadComplete);
    const unsubUnloadComplete = socketService.onUnloadComplete(get().handleUnloadComplete);

    return () => {
      unsubProgress();
      unsubComplete();
      unsubError();
      unsubLoadComplete();
      unsubUnloadComplete();
      socketService.disconnect();
    };
  },
}));
