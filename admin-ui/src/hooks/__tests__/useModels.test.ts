import { describe, it, expect, vi, beforeEach } from 'vitest';
import { renderHook, waitFor, act } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { createElement } from 'react';
import type { ModelInfo, ModelDownloadRequest } from '@/types';

// Mock the api module
const mockModelApi = {
  list: vi.fn(),
  download: vi.fn(),
  load: vi.fn(),
  unload: vi.fn(),
  delete: vi.fn(),
  cancelDownload: vi.fn(),
  preview: vi.fn(),
};

vi.mock('@/services/api', () => ({
  modelApi: mockModelApi,
}));

// Mock the server store
const mockSetModels = vi.fn();
const mockSetLoadedModel = vi.fn();
const mockSetModelLoading = vi.fn();

vi.mock('@/stores/serverStore', () => ({
  useServerStore: Object.assign(
    (selector?: (state: unknown) => unknown) => {
      const state = {
        setModels: mockSetModels,
        setLoadedModel: mockSetLoadedModel,
        setModelLoading: mockSetModelLoading,
      };
      return selector ? selector(state) : state;
    },
    {
      getState: () => ({
        setModels: mockSetModels,
        setLoadedModel: mockSetLoadedModel,
        setModelLoading: mockSetModelLoading,
      }),
    }
  ),
}));

// Mock the toast hook
const mockToast = {
  success: vi.fn(),
  error: vi.fn(),
  warning: vi.fn(),
  info: vi.fn(),
};

vi.mock('../useToast', () => ({
  useToast: () => mockToast,
}));

// Helper to create a fresh QueryClient for each test
function createTestQueryClient() {
  return new QueryClient({
    defaultOptions: {
      queries: {
        retry: false,
        gcTime: 0,
      },
      mutations: {
        retry: false,
      },
    },
  });
}

// Wrapper component for renderHook
function createWrapper() {
  const queryClient = createTestQueryClient();
  return function Wrapper({ children }: { children: React.ReactNode }) {
    return createElement(QueryClientProvider, { client: queryClient }, children);
  };
}

// Helper factory for mock models
function createMockModel(overrides: Partial<ModelInfo> = {}): ModelInfo {
  return {
    id: 1,
    name: 'gemma-2-2b',
    repo_id: 'google/gemma-2-2b',
    source: 'huggingface',
    quantization: 'Q4',
    params: '2.5B',
    memory_mb: 1800,
    local_path: '/data/models/gemma-2-2b',
    status: 'ready',
    created_at: '2026-01-30T12:00:00Z',
    updated_at: '2026-01-30T12:00:00Z',
    ...overrides,
  };
}

// Import after mocks
import { useModels } from '../useModels';

describe('useModels', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('returns models list from React Query', async () => {
    const models = [
      createMockModel({ id: 1, name: 'model-a' }),
      createMockModel({ id: 2, name: 'model-b' }),
    ];
    mockModelApi.list.mockResolvedValue(models);

    const { result } = renderHook(() => useModels(), {
      wrapper: createWrapper(),
    });

    await waitFor(() => {
      expect(result.current.isLoading).toBe(false);
    });

    expect(result.current.models).toEqual(models);
    expect(mockSetModels).toHaveBeenCalledWith(models);
  });

  it('sets loaded model from query when a model has loaded status', async () => {
    const loadedModel = createMockModel({ id: 1, name: 'loaded-model', status: 'loaded' });
    const models = [loadedModel, createMockModel({ id: 2, name: 'other', status: 'ready' })];
    mockModelApi.list.mockResolvedValue(models);

    const { result } = renderHook(() => useModels(), {
      wrapper: createWrapper(),
    });

    await waitFor(() => {
      expect(result.current.isLoading).toBe(false);
    });

    expect(mockSetLoadedModel).toHaveBeenCalledWith(loadedModel);
  });

  it('loadModel calls modelApi.load and updates store on success', async () => {
    const loadedModel = createMockModel({ id: 1, status: 'loaded' });
    mockModelApi.list.mockResolvedValue([]);
    mockModelApi.load.mockResolvedValue(loadedModel);

    const { result } = renderHook(() => useModels(), {
      wrapper: createWrapper(),
    });

    await waitFor(() => {
      expect(result.current.isLoading).toBe(false);
    });

    await act(async () => {
      result.current.load(1);
    });

    await waitFor(() => {
      expect(mockModelApi.load).toHaveBeenCalledWith(1);
    });

    expect(mockSetModelLoading).toHaveBeenCalledWith(true);
  });

  it('unloadModel calls modelApi.unload and clears loaded model', async () => {
    mockModelApi.list.mockResolvedValue([]);
    mockModelApi.unload.mockResolvedValue(undefined);

    const { result } = renderHook(() => useModels(), {
      wrapper: createWrapper(),
    });

    await waitFor(() => {
      expect(result.current.isLoading).toBe(false);
    });

    await act(async () => {
      result.current.unload(1);
    });

    await waitFor(() => {
      expect(mockModelApi.unload).toHaveBeenCalledWith(1);
    });

    expect(mockSetLoadedModel).toHaveBeenCalledWith(null);
    expect(mockToast.info).toHaveBeenCalledWith('Model unloaded');
  });

  it('downloadModel calls modelApi.download with correct request', async () => {
    const downloadedModel = createMockModel({ id: 3, status: 'downloading' });
    mockModelApi.list.mockResolvedValue([]);
    mockModelApi.download.mockResolvedValue(downloadedModel);

    const { result } = renderHook(() => useModels(), {
      wrapper: createWrapper(),
    });

    await waitFor(() => {
      expect(result.current.isLoading).toBe(false);
    });

    const downloadReq: ModelDownloadRequest = {
      source: 'huggingface',
      repo_id: 'google/gemma-2-2b',
      quantization: 'Q4',
    };

    await act(async () => {
      result.current.download(downloadReq);
    });

    await waitFor(() => {
      expect(mockModelApi.download).toHaveBeenCalledWith(downloadReq);
    });
  });

  it('deleteModel calls modelApi.delete with correct id', async () => {
    mockModelApi.list.mockResolvedValue([]);
    mockModelApi.delete.mockResolvedValue(undefined);

    const { result } = renderHook(() => useModels(), {
      wrapper: createWrapper(),
    });

    await waitFor(() => {
      expect(result.current.isLoading).toBe(false);
    });

    await act(async () => {
      result.current.delete(1);
    });

    await waitFor(() => {
      expect(mockModelApi.delete).toHaveBeenCalledWith(1);
    });

    expect(mockToast.success).toHaveBeenCalledWith('Model deleted');
  });

  it('reports loading state while fetching models', async () => {
    let resolveList: (value: ModelInfo[]) => void;
    mockModelApi.list.mockImplementation(
      () => new Promise<ModelInfo[]>((resolve) => { resolveList = resolve; })
    );

    const { result } = renderHook(() => useModels(), {
      wrapper: createWrapper(),
    });

    expect(result.current.isLoading).toBe(true);

    await act(async () => {
      resolveList!([]);
    });

    await waitFor(() => {
      expect(result.current.isLoading).toBe(false);
    });
  });

  it('handles query error and exposes error message', async () => {
    mockModelApi.list.mockRejectedValue(new Error('Network error'));

    const { result } = renderHook(() => useModels(), {
      wrapper: createWrapper(),
    });

    await waitFor(() => {
      expect(result.current.isLoading).toBe(false);
    });

    expect(result.current.error).toBe('Network error');
  });
});
