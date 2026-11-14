/**
 * Tests for EdgeSensingPanel / EdgeSensingToggle (Feature 15): status strip
 * states, event list, live WS prepend, unsensable-edge disclosure, the
 * up→down detail table, and the verbatim rung language.
 */

import { beforeEach, describe, expect, it, vi } from 'vitest';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { act, fireEvent, render, screen, waitFor } from '@testing-library/react';
import type {
  CircuitSensingEvent,
  CircuitSensingStatus,
} from '@/types/circuitSensing';

const handlers: Record<string, (data: unknown) => void> = {};

vi.mock('@/services/socket', () => ({
  socketClient: {
    on: vi.fn((event: string, handler: (data: unknown) => void) => {
      handlers[event] = handler;
    }),
    off: vi.fn((event: string) => {
      delete handlers[event];
    }),
  },
}));

const statusMock = vi.fn();
const eventsMock = vi.fn();
const eventDetailMock = vi.fn();
const clearEventsMock = vi.fn();
const setEnabledMock = vi.fn();

vi.mock('@/services/api', () => ({
  circuitSensingApi: {
    status: (...args: unknown[]) => statusMock(...args),
    events: (...args: unknown[]) => eventsMock(...args),
    eventDetail: (...args: unknown[]) => eventDetailMock(...args),
    clearEvents: (...args: unknown[]) => clearEventsMock(...args),
    setEnabled: (...args: unknown[]) => setEnabledMock(...args),
  },
}));

import { EdgeSensingPanel } from '../EdgeSensingPanel';
import { EdgeSensingToggle } from '../EdgeSensingToggle';

function makeStatus(
  overrides: Partial<CircuitSensingStatus> = {}
): CircuitSensingStatus {
  return {
    armed: true,
    circuit_id: 'circ_1',
    circuit_name: 'IOI circuit',
    layers: [6, 9],
    sensable_edges: 2,
    unsensable_edges: [],
    max_token_lag: 4,
    last_request_overhead_ms: 0.37,
    events_recorded: 3,
    ws_dropped: 0,
    enabled_circuits: [],
    ...overrides,
  };
}

function makeEvent(
  overrides: Partial<CircuitSensingEvent> = {}
): CircuitSensingEvent {
  return {
    id: 1,
    circuit_id: 'circ_1',
    request_id: 'req-1',
    phase: 'decode',
    edge_key: 'L6:F12→L9:F44',
    up: { layer: 6, feature_idx: 12, pos: 5, act: 8.4 },
    down: { layer: 9, feature_idx: 44, pos: 7, act: 3.1 },
    token_lag: 2,
    edge_rung: 1,
    edge_rung_language: 'attribution-supported',
    edge_type: null,
    ambient_fired_count: null,
    summary: 'IOI: L6:F12 → L9:F44 fired (lag 2) during decode',
    truncated: false,
    created_at: '2026-07-19T12:00:00Z',
    ...overrides,
  };
}

function renderPanel() {
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  render(
    <QueryClientProvider client={queryClient}>
      <EdgeSensingPanel />
    </QueryClientProvider>
  );
  return queryClient;
}

beforeEach(() => {
  vi.clearAllMocks();
  Object.keys(handlers).forEach((key) => delete handlers[key]);
});

describe('EdgeSensingPanel', () => {
  it('renders the armed status strip with layers, edges and lag', async () => {
    statusMock.mockResolvedValue(makeStatus());
    eventsMock.mockResolvedValue({ events: [], total: 0 });
    renderPanel();
    expect(await screen.findByText('armed')).toBeInTheDocument();
    expect(screen.getByText('IOI circuit')).toBeInTheDocument();
    expect(screen.getByText('L6 → L9')).toBeInTheDocument();
    expect(screen.getByText('2 edges watched')).toBeInTheDocument();
    expect(screen.getByText(/lag ≤ 4/)).toBeInTheDocument();
  });

  it('renders not-armed hint and enabled-but-unarmed circuits', async () => {
    statusMock.mockResolvedValue(
      makeStatus({
        armed: false,
        circuit_id: null,
        ws_dropped: 5,
        enabled_circuits: [{ id: 'circ_1', name: 'IOI circuit', is_active: false }],
      })
    );
    eventsMock.mockResolvedValue({ events: [], total: 0 });
    renderPanel();
    expect(
      await screen.findByText(/edge sensing is enabled for IOI circuit/)
    ).toBeInTheDocument();
    expect(screen.getByText(/5 live updates throttled/)).toBeInTheDocument();
  });

  it('surfaces unsensable edges with their reason', async () => {
    statusMock.mockResolvedValue(
      makeStatus({
        sensable_edges: 1,
        unsensable_edges: [
          {
            edge_key: 'L6:F12→L20:F3',
            reason: 'layer_not_attached',
            detail: 'no SAE at L20',
          },
        ],
      })
    );
    eventsMock.mockResolvedValue({ events: [], total: 0 });
    renderPanel();
    expect(await screen.findByTestId('unsensable-edges')).toBeInTheDocument();
    expect(screen.getByText('L6:F12→L20:F3')).toBeInTheDocument();
    expect(screen.getByText('no SAE attached at that layer')).toBeInTheDocument();
    expect(screen.getByText(/no SAE at L20/)).toBeInTheDocument();
    expect(screen.getByText(/1 edge is not being watched/)).toBeInTheDocument();
  });

  it('warns that an empty list is not evidence the edge did not fire', async () => {
    statusMock.mockResolvedValue(makeStatus());
    eventsMock.mockResolvedValue({ events: [], total: 0 });
    renderPanel();
    expect(
      await screen.findByText(/not that the edges did not fire/)
    ).toBeInTheDocument();
  });

  it('lists events and expands the up→down detail on click', async () => {
    statusMock.mockResolvedValue(makeStatus());
    eventsMock.mockResolvedValue({ events: [makeEvent()], total: 1 });
    eventDetailMock.mockResolvedValue(
      makeEvent({ context_text: 'the deep ocean current' })
    );
    renderPanel();
    fireEvent.click(await screen.findByText(/L6:F12 → L9:F44 fired/));
    expect(await screen.findByText('the deep ocean current')).toBeInTheDocument();
    expect(eventDetailMock).toHaveBeenCalledWith(1);
    // endpoint table: both ends, with layer / feature / pos / act
    expect(screen.getByText('#12')).toBeInTheDocument();
    expect(screen.getByText('#44')).toBeInTheDocument();
    expect(screen.getByText('@5')).toBeInTheDocument();
    expect(screen.getByText('8.400')).toBeInTheDocument();
    expect(screen.getByText('lag 2')).toBeInTheDocument();
  });

  it('renders edge_rung_language verbatim and marks rung<2 unvalidated', async () => {
    statusMock.mockResolvedValue(makeStatus());
    eventsMock.mockResolvedValue({ events: [makeEvent()], total: 1 });
    eventDetailMock.mockResolvedValue(makeEvent());
    renderPanel();
    fireEvent.click(await screen.findByText(/L6:F12 → L9:F44 fired/));
    const badge = await screen.findByTestId('edge-rung-badge');
    expect(badge).toHaveTextContent('attribution-supported');
    expect(screen.getByTestId('edge-unvalidated-badge')).toBeInTheDocument();
  });

  it('does not mark a rung-2 edge unvalidated', async () => {
    statusMock.mockResolvedValue(makeStatus());
    const validated = makeEvent({
      edge_rung: 2,
      edge_rung_language: 'causally validated',
    });
    eventsMock.mockResolvedValue({ events: [validated], total: 1 });
    eventDetailMock.mockResolvedValue(validated);
    renderPanel();
    fireEvent.click(await screen.findByText(/L6:F12 → L9:F44 fired/));
    // Asserted against the fixture's own value, never a re-typed phrase: the
    // badge must echo whatever the server sent, verbatim.
    expect(await screen.findByTestId('edge-rung-badge')).toHaveTextContent(
      validated.edge_rung_language
    );
    expect(screen.queryByTestId('edge-unvalidated-badge')).not.toBeInTheDocument();
  });

  it('highlights the fired span in the detail context', async () => {
    statusMock.mockResolvedValue(makeStatus());
    eventsMock.mockResolvedValue({ events: [makeEvent()], total: 1 });
    eventDetailMock.mockResolvedValue(
      makeEvent({
        context_parts: { before: 'the deep ', span: 'ocean', after: ' current' },
      })
    );
    renderPanel();
    fireEvent.click(await screen.findByText(/L6:F12 → L9:F44 fired/));
    const mark = await screen.findByText('ocean');
    expect(mark.tagName).toBe('MARK');
    expect(screen.getByText(/the deep/)).toBeInTheDocument();
  });

  it('prepends live WS events to the list', async () => {
    statusMock.mockResolvedValue(makeStatus());
    eventsMock.mockResolvedValue({ events: [makeEvent()], total: 1 });
    renderPanel();
    await screen.findByText(/L6:F12 → L9:F44 fired/);
    expect(handlers['circuit:sensing:event']).toBeDefined();

    act(() => {
      handlers['circuit:sensing:event'](
        makeEvent({ id: 2, summary: 'IOI: L6:F12 → L9:F44 fired (LIVE)' })
      );
    });
    expect(await screen.findByText(/LIVE/)).toBeInTheDocument();
    const items = screen.getAllByRole('listitem');
    expect(items[0].textContent).toContain('LIVE');
  });

  it('does not prepend another circuit’s event into a scoped list', async () => {
    statusMock.mockResolvedValue(makeStatus());
    eventsMock.mockResolvedValue({ events: [makeEvent()], total: 1 });
    const queryClient = new QueryClient({
      defaultOptions: { queries: { retry: false } },
    });
    render(
      <QueryClientProvider client={queryClient}>
        <EdgeSensingPanel circuitId="circ_1" />
      </QueryClientProvider>
    );
    await screen.findByText(/L6:F12 → L9:F44 fired/);
    act(() => {
      handlers['circuit:sensing:event'](
        makeEvent({ id: 3, circuit_id: 'circ_other', summary: 'OTHER circuit' })
      );
    });
    expect(screen.queryByText(/OTHER circuit/)).not.toBeInTheDocument();
  });

  it('clears ALL events after confirmation', async () => {
    statusMock.mockResolvedValue(makeStatus());
    eventsMock.mockResolvedValue({ events: [makeEvent()], total: 1 });
    clearEventsMock.mockResolvedValue({ deleted: 1 });
    const confirmSpy = vi.spyOn(window, 'confirm').mockReturnValue(true);
    renderPanel();
    await screen.findByText(/L6:F12 → L9:F44 fired/);
    fireEvent.click(screen.getByText('Clear all'));
    await waitFor(() => expect(clearEventsMock).toHaveBeenCalledWith(undefined));
    confirmSpy.mockRestore();
  });

  it('declining the confirm does not clear', async () => {
    statusMock.mockResolvedValue(makeStatus());
    eventsMock.mockResolvedValue({ events: [makeEvent()], total: 1 });
    const confirmSpy = vi.spyOn(window, 'confirm').mockReturnValue(false);
    renderPanel();
    await screen.findByText(/L6:F12 → L9:F44 fired/);
    fireEvent.click(screen.getByText('Clear all'));
    expect(clearEventsMock).not.toHaveBeenCalled();
    confirmSpy.mockRestore();
  });
});

describe('EdgeSensingToggle', () => {
  function renderToggle(circuitId = 'circ_1') {
    const queryClient = new QueryClient({
      defaultOptions: { queries: { retry: false } },
    });
    render(
      <QueryClientProvider client={queryClient}>
        <EdgeSensingToggle circuitId={circuitId} />
      </QueryClientProvider>
    );
  }

  it('enables sensing for a circuit that is not yet enabled', async () => {
    statusMock.mockResolvedValue(makeStatus({ armed: false, circuit_id: null }));
    setEnabledMock.mockResolvedValue({
      circuit_id: 'circ_1',
      enabled: true,
      armed: false,
      unsensable_edges: [],
      message: 'Enabled; the circuit will arm when it is activated.',
    });
    renderToggle();
    const button = await screen.findByTestId('edge-sensing-toggle');
    await waitFor(() => expect(button).toHaveAttribute('aria-pressed', 'false'));
    fireEvent.click(button);
    await waitFor(() =>
      expect(setEnabledMock).toHaveBeenCalledWith('circ_1', true)
    );
  });

  it('reflects enabled state from the status endpoint and disables on click', async () => {
    statusMock.mockResolvedValue(
      makeStatus({
        enabled_circuits: [{ id: 'circ_1', name: 'IOI circuit', is_active: true }],
      })
    );
    setEnabledMock.mockResolvedValue({
      circuit_id: 'circ_1',
      enabled: false,
      armed: false,
      unsensable_edges: [],
      message: '',
    });
    renderToggle();
    const button = await screen.findByTestId('edge-sensing-toggle');
    await waitFor(() => expect(button).toHaveAttribute('aria-pressed', 'true'));
    fireEvent.click(button);
    await waitFor(() =>
      expect(setEnabledMock).toHaveBeenCalledWith('circ_1', false)
    );
  });

  it('does not report another circuit’s enablement as its own', async () => {
    statusMock.mockResolvedValue(
      makeStatus({
        enabled_circuits: [{ id: 'circ_other', name: 'other', is_active: true }],
      })
    );
    renderToggle('circ_1');
    const button = await screen.findByTestId('edge-sensing-toggle');
    await waitFor(() => expect(button).toHaveAttribute('aria-pressed', 'false'));
  });
});
