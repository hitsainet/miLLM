/**
 * F19 R3-04 — the compose success toast must NOT assert a rung.
 *
 * Composition is exactly the state where the runtime SUPPRESSES
 * `X-miLLM-Circuit-Rung`, because no single circuit's evidence describes a
 * summed response. A success toast reading `"X" is now steering (causally
 * validated (edge))` makes that claim at the one moment the server has stopped
 * making it.
 *
 * R2-14 fixed this same contradiction on the circuit card and missed the toast.
 */

import { describe, expect, it, vi, beforeEach } from 'vitest';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { renderHook, act, waitFor } from '@testing-library/react';
import type { ReactNode } from 'react';

const activateMock = vi.fn();
const toastSuccess = vi.fn();
const toastWarning = vi.fn();

vi.mock('@/services/circuits', () => ({
  circuitApi: {
    list: vi.fn().mockResolvedValue({ circuits: [], total: 0 }),
    activate: (...a: unknown[]) => activateMock(...a),
    deactivate: vi.fn(),
    setIntensity: vi.fn(),
    delete: vi.fn(),
    export: vi.fn(),
    claims: vi.fn().mockResolvedValue([]),
  },
}));

vi.mock('../useToast', () => ({
  useToast: () => ({
    success: toastSuccess,
    warning: toastWarning,
    error: vi.fn(),
  }),
}));

import { useCircuits } from '../useCircuits';

function wrapper({ children }: { children: ReactNode }) {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false }, mutations: { retry: false } },
  });
  return <QueryClientProvider client={client}>{children}</QueryClientProvider>;
}

describe('useCircuits — compose success toast', () => {
  beforeEach(() => {
    activateMock.mockReset();
    toastSuccess.mockReset();
    toastWarning.mockReset();
  });

  it('does NOT report a rung when layers were composed', async () => {
    activateMock.mockResolvedValue({
      name: 'hedging',
      serving_mode: 'full',
      rung_language: 'causally validated (edge)',
      composed_layers: [10, 13],
      allowed_layer_overlap: true,
    });
    const { result } = renderHook(() => useCircuits(), { wrapper });

    await act(async () => {
      await result.current.activateCircuit({
        circuitId: 'circ_1',
        allowLayerOverlap: true,
      });
    });

    await waitFor(() => expect(toastWarning).toHaveBeenCalled());
    const message = toastWarning.mock.calls[0][0] as string;
    expect(message).not.toContain('causally validated');
    expect(message).toContain('COMPOSED');
    expect(message).toContain('circuit-rung header is omitted');
    expect(toastSuccess).not.toHaveBeenCalled();
  });

  it('DOES report the rung on an ordinary activation', async () => {
    // Specificity: suppressing the rung everywhere would delete a disclosure
    // the feature exists to provide.
    activateMock.mockResolvedValue({
      name: 'hedging',
      serving_mode: 'full',
      rung_language: 'causally validated (edge)',
      composed_layers: [],
    });
    const { result } = renderHook(() => useCircuits(), { wrapper });

    await act(async () => {
      await result.current.activateCircuit({ circuitId: 'circ_1' });
    });

    await waitFor(() => expect(toastSuccess).toHaveBeenCalled());
    expect(toastSuccess.mock.calls[0][0]).toContain('causally validated (edge)');
  });
});
