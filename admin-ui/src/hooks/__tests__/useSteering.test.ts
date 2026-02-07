import { describe, it, expect, vi, beforeEach } from 'vitest';
import { renderHook, waitFor, act } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { createElement } from 'react';
import type { SteeringState, FeatureSteering } from '@/types';

// Mock the steering API
const mockSteeringApi = {
  getState: vi.fn(),
  set: vi.fn(),
  batch: vi.fn(),
  remove: vi.fn(),
  clear: vi.fn(),
  enable: vi.fn(),
  disable: vi.fn(),
};

vi.mock('@/services/api', () => ({
  steeringApi: mockSteeringApi,
}));

// Mock the server store
const mockSetSteering = vi.fn();

vi.mock('@/stores/serverStore', () => ({
  useServerStore: Object.assign(
    (selector?: (state: unknown) => unknown) => {
      const state = { setSteering: mockSetSteering };
      return selector ? selector(state) : state;
    },
    {
      getState: () => ({ setSteering: mockSetSteering }),
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

// Helper factory for mock steering state
function createMockSteeringState(overrides: Partial<SteeringState> = {}): SteeringState {
  return {
    enabled: false,
    sae_id: null,
    features: [],
    ...overrides,
  };
}

// Import after mocks
import { useSteering } from '../useSteering';

describe('useSteering', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('returns steering state from React Query', async () => {
    const steeringState = createMockSteeringState({
      enabled: true,
      features: [{ index: 1234, strength: 5.0 }],
    });
    mockSteeringApi.getState.mockResolvedValue(steeringState);

    const { result } = renderHook(() => useSteering(), {
      wrapper: createWrapper(),
    });

    await waitFor(() => {
      expect(result.current.isLoading).toBe(false);
    });

    expect(result.current.steering).toEqual(steeringState);
    expect(mockSetSteering).toHaveBeenCalledWith(steeringState);
  });

  it('setFeature calls steeringApi.set with correct params', async () => {
    const initialState = createMockSteeringState();
    const updatedState = createMockSteeringState({
      features: [{ index: 1234, strength: 5.0 }],
    });
    mockSteeringApi.getState.mockResolvedValue(initialState);
    mockSteeringApi.set.mockResolvedValue(updatedState);

    const { result } = renderHook(() => useSteering(), {
      wrapper: createWrapper(),
    });

    await waitFor(() => {
      expect(result.current.isLoading).toBe(false);
    });

    await act(async () => {
      result.current.setFeature({ index: 1234, strength: 5.0 });
    });

    await waitFor(() => {
      expect(mockSteeringApi.set).toHaveBeenCalledWith({
        feature_index: 1234,
        strength: 5.0,
      });
    });

    expect(mockSetSteering).toHaveBeenCalledWith(updatedState);
  });

  it('removeFeature calls steeringApi.remove with the feature index', async () => {
    const initialState = createMockSteeringState({
      features: [{ index: 1234, strength: 5.0 }],
    });
    const updatedState = createMockSteeringState({ features: [] });
    mockSteeringApi.getState.mockResolvedValue(initialState);
    mockSteeringApi.remove.mockResolvedValue(updatedState);

    const { result } = renderHook(() => useSteering(), {
      wrapper: createWrapper(),
    });

    await waitFor(() => {
      expect(result.current.isLoading).toBe(false);
    });

    await act(async () => {
      result.current.removeFeature(1234);
    });

    await waitFor(() => {
      expect(mockSteeringApi.remove).toHaveBeenCalledWith(1234);
    });

    expect(mockSetSteering).toHaveBeenCalledWith(updatedState);
  });

  it('enable calls steeringApi.enable and shows success toast', async () => {
    const initialState = createMockSteeringState({ enabled: false });
    const enabledState = createMockSteeringState({ enabled: true });
    mockSteeringApi.getState.mockResolvedValue(initialState);
    mockSteeringApi.enable.mockResolvedValue(enabledState);

    const { result } = renderHook(() => useSteering(), {
      wrapper: createWrapper(),
    });

    await waitFor(() => {
      expect(result.current.isLoading).toBe(false);
    });

    await act(async () => {
      result.current.enable();
    });

    await waitFor(() => {
      expect(mockSteeringApi.enable).toHaveBeenCalled();
    });

    expect(mockSetSteering).toHaveBeenCalledWith(enabledState);
    expect(mockToast.success).toHaveBeenCalledWith('Steering enabled');
  });

  it('disable calls steeringApi.disable and shows info toast', async () => {
    const initialState = createMockSteeringState({ enabled: true });
    const disabledState = createMockSteeringState({ enabled: false });
    mockSteeringApi.getState.mockResolvedValue(initialState);
    mockSteeringApi.disable.mockResolvedValue(disabledState);

    const { result } = renderHook(() => useSteering(), {
      wrapper: createWrapper(),
    });

    await waitFor(() => {
      expect(result.current.isLoading).toBe(false);
    });

    await act(async () => {
      result.current.disable();
    });

    await waitFor(() => {
      expect(mockSteeringApi.disable).toHaveBeenCalled();
    });

    expect(mockSetSteering).toHaveBeenCalledWith(disabledState);
    expect(mockToast.info).toHaveBeenCalledWith('Steering disabled');
  });

  it('clear calls steeringApi.clear and shows info toast', async () => {
    const initialState = createMockSteeringState({
      features: [
        { index: 100, strength: 2.0 },
        { index: 200, strength: -1.0 },
      ],
    });
    const clearedState = createMockSteeringState({ features: [] });
    mockSteeringApi.getState.mockResolvedValue(initialState);
    mockSteeringApi.clear.mockResolvedValue(clearedState);

    const { result } = renderHook(() => useSteering(), {
      wrapper: createWrapper(),
    });

    await waitFor(() => {
      expect(result.current.isLoading).toBe(false);
    });

    await act(async () => {
      result.current.clear();
    });

    await waitFor(() => {
      expect(mockSteeringApi.clear).toHaveBeenCalled();
    });

    expect(mockSetSteering).toHaveBeenCalledWith(clearedState);
    expect(mockToast.info).toHaveBeenCalledWith('Steering cleared');
  });

  it('shows error toast when setFeature mutation fails', async () => {
    const initialState = createMockSteeringState();
    mockSteeringApi.getState.mockResolvedValue(initialState);
    mockSteeringApi.set.mockRejectedValue(new Error('SAE not attached'));

    const { result } = renderHook(() => useSteering(), {
      wrapper: createWrapper(),
    });

    await waitFor(() => {
      expect(result.current.isLoading).toBe(false);
    });

    await act(async () => {
      result.current.setFeature({ index: 1234, strength: 5.0 });
    });

    await waitFor(() => {
      expect(mockToast.error).toHaveBeenCalledWith(
        'Failed to set feature: SAE not attached'
      );
    });
  });

  it('batchUpdate calls steeringApi.batch with features array', async () => {
    const initialState = createMockSteeringState();
    const batchedState = createMockSteeringState({
      features: [
        { index: 100, strength: 2.0 },
        { index: 200, strength: -1.0 },
      ],
    });
    mockSteeringApi.getState.mockResolvedValue(initialState);
    mockSteeringApi.batch.mockResolvedValue(batchedState);

    const { result } = renderHook(() => useSteering(), {
      wrapper: createWrapper(),
    });

    await waitFor(() => {
      expect(result.current.isLoading).toBe(false);
    });

    const features: FeatureSteering[] = [
      { index: 100, strength: 2.0 },
      { index: 200, strength: -1.0 },
    ];

    await act(async () => {
      result.current.batchUpdate(features);
    });

    await waitFor(() => {
      expect(mockSteeringApi.batch).toHaveBeenCalledWith({ features });
    });

    expect(mockSetSteering).toHaveBeenCalledWith(batchedState);
    expect(mockToast.success).toHaveBeenCalledWith('Steering updated');
  });
});
