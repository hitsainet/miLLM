/**
 * Tests for SensingPanel (Feature 11): status strip states, event list,
 * live WS prepend, clear action, and detail expansion.
 */

import { beforeEach, describe, expect, it, vi } from 'vitest';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { act, fireEvent, render, screen, waitFor } from '@testing-library/react';
import type { SensingEvent, SensingStatus } from '@/types/sensing';

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
const setConfigMock = vi.fn();

vi.mock('@/services/api', () => ({
  sensingApi: {
    status: (...args: unknown[]) => statusMock(...args),
    events: (...args: unknown[]) => eventsMock(...args),
    eventDetail: (...args: unknown[]) => eventDetailMock(...args),
    clearEvents: (...args: unknown[]) => clearEventsMock(...args),
    setEnabled: (...args: unknown[]) => setEnabledMock(...args),
    setConfig: (...args: unknown[]) => setConfigMock(...args),
  },
}));

import { SensingPanel } from '../SensingPanel';

function makeStatus(overrides: Partial<SensingStatus> = {}): SensingStatus {
  return {
    armed: true,
    profile_id: 'prof_s1',
    profile_name: 'fear cluster',
    member_count: 3,
    min_k: 2,
    threshold_mode: 'epsilon_max',
    context_tokens: 16,
    last_request_overhead_ms: 0.42,
    overhead_warn_threshold_ms: 5,
    events_recorded_since_start: 2,
    ws_events_dropped: 0,
    sensable_count: 3,
    enabled_clusters: [],
    retention: { max_events_per_cluster: 1000, max_age_days: 7 },
    ...overrides,
  };
}

function makeEvent(overrides: Partial<SensingEvent> = {}): SensingEvent {
  return {
    id: 1,
    profile_id: 'prof_s1',
    request_id: 'req-1',
    phase: 'decode',
    pos_start: 5,
    pos_end: 6,
    fired_members: [
      [7, 8.4],
      [9, 2.2],
    ],
    fired_count: 2,
    score: 2.1,
    ambient_fired_count: null,
    summary: 'fear: 2/3 members fired (peak F7 2.1×θ) during decode @ 5–6',
    truncated: false,
    created_at: '2026-07-16T12:00:00Z',
    ...overrides,
  };
}

function renderPanel() {
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  render(
    <QueryClientProvider client={queryClient}>
      <SensingPanel />
    </QueryClientProvider>
  );
  return queryClient;
}

beforeEach(() => {
  vi.clearAllMocks();
  Object.keys(handlers).forEach((key) => delete handlers[key]);
});

describe('SensingPanel', () => {
  it('renders armed status strip with threshold mode warning', async () => {
    statusMock.mockResolvedValue(makeStatus({ threshold_mode: 'floor_only' }));
    eventsMock.mockResolvedValue({ events: [], total: 0 });
    renderPanel();
    expect(await screen.findByText('armed')).toBeInTheDocument();
    expect(screen.getByText('fear cluster')).toBeInTheDocument();
    expect(screen.getByText('floor-only thresholds')).toBeInTheDocument();
  });

  it('renders not-armed hint when idle', async () => {
    statusMock.mockResolvedValue(makeStatus({ armed: false, profile_id: null }));
    eventsMock.mockResolvedValue({ events: [], total: 0 });
    renderPanel();
    expect(await screen.findByText(/Not armed/)).toBeInTheDocument();
    expect(await screen.findByText(/No events yet/)).toBeInTheDocument();
  });

  it('lists events newest-first and expands detail on click', async () => {
    statusMock.mockResolvedValue(makeStatus());
    eventsMock.mockResolvedValue({ events: [makeEvent()], total: 1 });
    eventDetailMock.mockResolvedValue(
      makeEvent({ context_text: 'the deep ocean current' })
    );
    renderPanel();
    const row = await screen.findByText(/fear: 2\/3 members fired/);
    fireEvent.click(row);
    expect(await screen.findByText('the deep ocean current')).toBeInTheDocument();
    expect(eventDetailMock).toHaveBeenCalledWith(1);
  });

  it('prepends live WS events to the list', async () => {
    statusMock.mockResolvedValue(makeStatus());
    eventsMock.mockResolvedValue({ events: [makeEvent()], total: 1 });
    renderPanel();
    await screen.findByText(/fear: 2\/3 members fired/);
    expect(handlers['sensing:event']).toBeDefined();

    act(() => {
      handlers['sensing:event'](
        makeEvent({ id: 2, summary: 'fear: 3/3 members fired (LIVE)' })
      );
    });
    expect(await screen.findByText(/LIVE/)).toBeInTheDocument();
    // both events visible; the live one is first in the DOM
    const items = screen.getAllByRole('listitem');
    expect(items[0].textContent).toContain('LIVE');
  });

  it('clears ALL events after confirmation (R3: scope matches the list)', async () => {
    statusMock.mockResolvedValue(makeStatus());
    eventsMock.mockResolvedValue({ events: [makeEvent()], total: 1 });
    clearEventsMock.mockResolvedValue({ deleted: 1 });
    const confirmSpy = vi.spyOn(window, 'confirm').mockReturnValue(true);
    renderPanel();
    await screen.findByText(/fear: 2\/3 members fired/);
    fireEvent.click(screen.getByText('Clear all'));
    await waitFor(() => expect(clearEventsMock).toHaveBeenCalledWith(undefined));
    expect(confirmSpy).toHaveBeenCalled();
    confirmSpy.mockRestore();
  });

  it('declining the confirm does not clear', async () => {
    statusMock.mockResolvedValue(makeStatus());
    eventsMock.mockResolvedValue({ events: [makeEvent()], total: 1 });
    const confirmSpy = vi.spyOn(window, 'confirm').mockReturnValue(false);
    renderPanel();
    await screen.findByText(/fear: 2\/3 members fired/);
    fireEvent.click(screen.getByText('Clear all'));
    expect(clearEventsMock).not.toHaveBeenCalled();
    confirmSpy.mockRestore();
  });

  it('shows enabled-but-unarmed clusters and throttled-drop count', async () => {
    statusMock.mockResolvedValue(
      makeStatus({
        armed: false,
        profile_id: null,
        ws_events_dropped: 7,
        enabled_clusters: [
          { id: 'prof_s1', name: 'fear cluster', is_active: false },
        ],
      })
    );
    eventsMock.mockResolvedValue({ events: [], total: 0 });
    renderPanel();
    expect(
      await screen.findByText(/sensing is enabled for fear cluster/)
    ).toBeInTheDocument();
    expect(screen.getByText(/7 live updates throttled/)).toBeInTheDocument();
  });
});


describe('SensingPanel quorum control', () => {
  it('commits a typed quorum on Enter', async () => {
    statusMock.mockResolvedValue(makeStatus());
    eventsMock.mockResolvedValue({ events: [], total: 0 });
    setConfigMock.mockResolvedValue({
      profile_id: 'prof_s1', min_k: 2, effective_min_k: 2, armed: true,
    });
    renderPanel();
    const input = await screen.findByLabelText(
      'Quorum (members that must co-fire)'
    );
    fireEvent.change(input, { target: { value: '2' } });
    fireEvent.keyDown(input, { key: 'Enter' });
    fireEvent.blur(input);
    await waitFor(() => expect(setConfigMock).toHaveBeenCalledWith('prof_s1', 2));
    expect(setConfigMock).toHaveBeenCalledTimes(1); // Enter->blur = ONE commit
  });

  it('reset button clears the override (null)', async () => {
    statusMock.mockResolvedValue(makeStatus());
    eventsMock.mockResolvedValue({ events: [], total: 0 });
    setConfigMock.mockResolvedValue({
      profile_id: 'prof_s1', min_k: null, effective_min_k: 3, armed: true,
    });
    renderPanel();
    fireEvent.click(await screen.findByLabelText('Reset quorum to default'));
    await waitFor(() =>
      expect(setConfigMock).toHaveBeenCalledWith('prof_s1', null)
    );
  });

  it('highlights the fired span in event detail', async () => {
    statusMock.mockResolvedValue(makeStatus());
    eventsMock.mockResolvedValue({ events: [makeEvent()], total: 1 });
    eventDetailMock.mockResolvedValue(
      makeEvent({
        context_parts: {
          before: 'the deep ',
          span: 'ocean',
          after: ' current',
        },
      })
    );
    renderPanel();
    fireEvent.click(await screen.findByText(/fear: 2\/3 members fired/));
    const mark = await screen.findByText('ocean');
    expect(mark.tagName).toBe('MARK');
    expect(screen.getByText(/the deep/)).toBeInTheDocument();
  });

  it('falls back to plain context_text without parts', async () => {
    statusMock.mockResolvedValue(makeStatus());
    eventsMock.mockResolvedValue({ events: [makeEvent()], total: 1 });
    eventDetailMock.mockResolvedValue(
      makeEvent({ context_text: 'plain old context', context_parts: null })
    );
    renderPanel();
    fireEvent.click(await screen.findByText(/fear: 2\/3 members fired/));
    expect(await screen.findByText('plain old context')).toBeInTheDocument();
  });
});
