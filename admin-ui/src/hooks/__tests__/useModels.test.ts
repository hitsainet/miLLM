import { describe, it, expect, vi, beforeEach } from 'vitest';
import { renderHook, waitFor, act } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { createElement } from 'react';
import type { ModelInfo, ModelDownloadRequest } from '@/types';

// Use vi.hoisted so these variables are available when vi.mock factories run
const mockModelApi = vi.hoisted(() => ({
  list: vi.fn(),
  download: vi.fn(),
  load: vi.fn(),
  unload: vi.fn(),
  delete: vi.fn(),
  cancelDownload: vi.fn(),
  preview: vi.fn(),
}));

vi.mock('@/services/api', () => ({
  modelApi: mockModelApi,
}));

const mockSetModels = vi.hoisted(() => vi.fn());
const mockSetLoadedModel = vi.hoisted(() => vi.fn());
const mockSetModelLoading = vi.hoisted(() => vi.fn());
const mockLoadGpu = vi.hoisted(() => ({ value: 'auto' }));

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
        loadGpu: mockLoadGpu.value,
      }),
    }
  ),
}));

const mockToast = vi.hoisted(() => ({
  success: vi.fn(),
  error: vi.fn(),
  warning: vi.fn(),
  info: vi.fn(),
}));

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
      expect(mockModelApi.load).toHaveBeenCalledWith(1, 'auto');
    });

    expect(mockSetModelLoading).toHaveBeenCalledWith(true);
  });

  it('load sends the card chosen in the GPU selector', async () => {
    // MUTATION CONTROL: call modelApi.load(id) without the store's loadGpu -> fails.
    mockLoadGpu.value = 'GPU-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee';
    try {
      mockModelApi.list.mockResolvedValue([]);
      mockModelApi.load.mockResolvedValue(createMockModel({ id: 2, status: 'loaded' }));
      const { result } = renderHook(() => useModels(), { wrapper: createWrapper() });
      await waitFor(() => expect(result.current.isLoading).toBe(false));

      await act(async () => {
        result.current.load(2);
      });

      await waitFor(() => {
        expect(mockModelApi.load).toHaveBeenCalledWith(
          2,
          'GPU-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee',
        );
      });
    } finally {
      mockLoadGpu.value = 'auto';
    }
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

describe('the loaded model is RECONCILED, not only ever set', () => {
  // ⚠ REPORTED FROM THE RUNNING UI 2026-09-30, where the header read "No Model" while
  // Llama-3.1-8B was loaded, holding 15.7 GB, and scoring live traffic through an armed probe.
  //
  // Two defects met in one line, `if (loaded) setLoadedModel(loaded)`:
  //
  //   1. It only ever SET. A model unloaded on the server left the old name in the store until
  //      a socket event happened to arrive — the header could show a model that was gone.
  //   2. It is the ONLY seeder, and `useModels()` was called by just three pages. Landing
  //      anywhere else showed "No Model" forever. The badge described which pages you had
  //      visited, not the server.
  //
  // A read that can only add is not a reconciliation. This estate has shipped that exact shape
  // before: a stale in-memory flag outlived restarts and hid thirteen models from /v1/models for
  // three months.

  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('sets the loaded model when the server reports one', async () => {
    mockModelApi.list.mockResolvedValue([
      { id: 1, name: 'Other', status: 'ready' },
      { id: 9, name: 'Llama-3.1-8B-Instruct', status: 'loaded' },
    ] as unknown as ModelInfo[]);
    renderHook(() => useModels(), { wrapper: createWrapper() });
    await waitFor(() => expect(mockSetLoadedModel).toHaveBeenCalled());
    expect(mockSetLoadedModel).toHaveBeenCalledWith(
      expect.objectContaining({ id: 9, name: 'Llama-3.1-8B-Instruct' })
    );
  });

  it('⚠ CLEARS it when the server reports none', async () => {
    // The half that was missing. Against `if (loaded) …` this never fires and the header keeps
    // displaying a model the server has unloaded.
    mockModelApi.list.mockResolvedValue([
      { id: 1, name: 'Other', status: 'ready' },
      { id: 9, name: 'Llama-3.1-8B-Instruct', status: 'ready' },
    ] as unknown as ModelInfo[]);
    renderHook(() => useModels(), { wrapper: createWrapper() });
    await waitFor(() => expect(mockSetLoadedModel).toHaveBeenCalled());
    expect(mockSetLoadedModel).toHaveBeenCalledWith(null);
  });

  it('an empty model list clears it too', async () => {
    mockModelApi.list.mockResolvedValue([] as unknown as ModelInfo[]);
    renderHook(() => useModels(), { wrapper: createWrapper() });
    await waitFor(() => expect(mockSetLoadedModel).toHaveBeenCalledWith(null));
  });
});
