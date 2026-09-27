/**
 * Probe Monitors page.
 *
 * The tests worth keeping are the ones about what the page must NOT say. A probe with no
 * threshold, and a request that was not scored, both have to render as something other than
 * "did not fire" — they are the absence of a decision, not a negative one. Rendering either as a
 * quiet nothing is how a monitor comes to be trusted for a claim it never made.
 */

import { beforeEach, describe, expect, it, vi } from 'vitest';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { MemoryRouter } from 'react-router-dom';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import type { Probe, ProbeEvent, ProbeStatus } from '@/types/probe';

const state = {
  probes: [] as Probe[],
  status: undefined as ProbeStatus | undefined,
  events: [] as ProbeEvent[],
  pendingAck: undefined as
    | { id: string; details: { rung?: number; rung_language?: string; next_step?: string } }
    | undefined,
};

const calls = {
  arm: vi.fn(),
  dismissAck: vi.fn(),
  checkParity: vi.fn(),
  disarm: vi.fn(),
};

vi.mock('@/hooks/useProbes', () => ({
  useProbes: () => ({
    probes: state.probes,
    probesLoading: false,
    status: state.status,
    statusLoading: false,
    events: state.events,
    eventsLoading: false,
    importProbe: vi.fn(),
    importing: false,
    arm: calls.arm,
    arming: false,
    pendingAck: state.pendingAck,
    dismissAck: calls.dismissAck,
    checkParity: calls.checkParity,
    checkingParity: false,
    disarm: calls.disarm,
    remove: vi.fn(),
    clearEvents: vi.fn(),
  }),
  useProbeEventDetail: () => ({ data: undefined, isLoading: false }),
}));

// Imported after the mock so the page picks it up.
const { ProbeMonitorsPage } = await import('../ProbeMonitorsPage');

function probe(over: Partial<Probe> = {}): Probe {
  return {
    id: 'pr_1',
    name: 'high-stakes',
    hf_id: 'LiquidAI/LFM2.5-1.2B-Instruct',
    layer: 11,
    rule: 'mean',
    scope: 'all',
    basis: 'residual',
    streamable: true,
    threshold: 1.07,
    target_fpr: 0.01,
    rung: 2,
    rung_language: 'detects on unseen tasks',
    next_step: 'run the judge baseline on the same out-of-distribution sets',
    armed: false,
    paused_reason: null,
    parity: null,
    created_at: '2026-09-27T00:00:00Z',
    ...over,
  };
}

function event(over: Partial<ProbeEvent> = {}): ProbeEvent {
  return {
    id: 1,
    probe_id: 'pr_1',
    request_id: 'chatcmpl-aaa',
    scored: true,
    not_scored_reason: null,
    score: 2.31,
    threshold: 1.07,
    verdict: true,
    rung: 2,
    top_positions: [3],
    n_scored_tokens: 12,
    summary: 'score 2.3100 above threshold',
    created_at: '2026-09-27T00:00:00Z',
    ...over,
  };
}

function renderPage() {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(
    <QueryClientProvider client={client}>
      <MemoryRouter>
        <ProbeMonitorsPage />
      </MemoryRouter>
    </QueryClientProvider>
  );
}

describe('ProbeMonitorsPage', () => {
  beforeEach(() => {
    state.pendingAck = undefined;
    state.status = undefined;
    state.events = [];
    state.probes = [];
    calls.arm.mockClear();
    calls.checkParity.mockClear();
    calls.dismissAck.mockClear();
  });

  it('renders an empty state that says where probes come from', () => {
    state.probes = [];
    state.events = [];
    state.status = undefined;
    renderPage();
    expect(screen.getByTestId('probe-monitors-page')).toBeInTheDocument();
    expect(screen.getByText(/Export one from miStudio/i)).toBeInTheDocument();
  });

  it('says plainly that it records rather than intervenes', () => {
    state.probes = [];
    renderPage();
    expect(
      screen.getByText(/they do not stop or alter a generation/i)
    ).toBeInTheDocument();
  });

  it('shows a probe with its SERVER-rendered rung language', () => {
    state.probes = [probe()];
    renderPage();
    expect(screen.getByTestId('rung-badge')).toHaveTextContent('detects on unseen tasks');
  });

  it('shows the next step for a probe below rung 2', () => {
    state.probes = [probe({ rung: 1, rung_language: 'detects on held-out data' })];
    renderPage();
    expect(screen.getByText(/Next: run the judge baseline/i)).toBeInTheDocument();
  });

  it('renders a null threshold as "ranks, does not decide" — not as zero', () => {
    state.probes = [probe({ threshold: null })];
    renderPage();
    expect(screen.getByText(/ranks, does not decide/i)).toBeInTheDocument();
    expect(screen.queryByText(/threshold 0\.0000/)).not.toBeInTheDocument();
  });

  it('⚠ an armed probe that is not scoring SAYS WHY', () => {
    // Silence here reads as "nothing detected", which is a claim the probe never made.
    state.probes = [probe({ armed: true, paused_reason: 'speculative_decoding' })];
    renderPage();
    expect(screen.getByTestId('paused-reason')).toHaveTextContent('speculative_decoding');
  });

  it('shows a failed parity check rather than hiding it', () => {
    state.probes = [
      probe({
        parity: {
          passed: false,
          tolerance: 0.001,
          max_abs_diff: 3.2,
          vector_index: 0,
          vectors: [],
          tokenization_drift: {},
          error: null,
        },
      }),
    ];
    renderPage();
    expect(screen.getByTestId('parity')).toHaveTextContent('FAILED');
  });

  it('⚠ an unscored verdict shows its REASON, not a blank row', () => {
    state.probes = [probe()];
    state.events = [
      event({ scored: false, not_scored_reason: 'continuous_batching', score: null, verdict: null }),
    ];
    renderPage();
    expect(screen.getByTestId('not-scored')).toHaveTextContent('continuous_batching');
  });

  it('a scored verdict with a null decision does not render as "fires"', () => {
    state.probes = [probe({ threshold: null })];
    state.events = [event({ verdict: null, threshold: null })];
    renderPage();
    expect(screen.getByTestId('event-row')).toHaveTextContent('(no threshold)');
    expect(screen.getByTestId('event-row')).not.toHaveTextContent('· fires');
  });

  it('reports overhead as em-dash when nothing has been scored yet', () => {
    // null means "no request scored", which is not the same as 0.00 ms.
    state.probes = [];
    state.events = [];
    state.status = {
      armed: [],
      armed_count: 0,
      max_armed: 8,
      imported_count: 0,
      paused_reasons: [],
      last_request_overhead_ms: null,
      overhead_warn_threshold_ms: 5,
      events_recorded: 0,
      socket_events_dropped: 0,
      force_serial: true,
    };
    renderPage();
    const status = screen.getByTestId('probe-status');
    expect(status).toHaveTextContent('0 / 8');
    expect(status).toHaveTextContent('—');
  });

  it('summarises paused reasons at the top when any probe is quiet', () => {
    state.probes = [];
    state.status = {
      armed: [],
      armed_count: 1,
      max_armed: 8,
      imported_count: 1,
      paused_reasons: ['model_changed'],
      last_request_overhead_ms: 1.2,
      overhead_warn_threshold_ms: 5,
      events_recorded: 3,
      socket_events_dropped: 0,
      force_serial: true,
    };
    renderPage();
    expect(screen.getByTestId('paused-summary')).toHaveTextContent('model_changed');
  });

  it('an idle probe offers Arm, an armed one offers Disarm — never both', async () => {
    state.probes = [probe()];
    renderPage();
    expect(screen.getByRole('button', { name: 'Arm' })).toBeInTheDocument();
    expect(screen.queryByRole('button', { name: 'Disarm' })).not.toBeInTheDocument();
    await userEvent.click(screen.getByRole('button', { name: 'Arm' }));
    // ⚠ The PAYLOAD: arming from the row must NOT pre-acknowledge. Passing
    // acknowledgeBelowRung2: true here would make the gate unreachable by clicking Arm.
    expect(calls.arm).toHaveBeenCalledTimes(1);
    expect(calls.arm).toHaveBeenCalledWith({ id: 'pr_1' });
  });

  it('parity can be re-checked without arming', async () => {
    state.probes = [probe()];
    renderPage();
    await userEvent.click(screen.getByRole('button', { name: 'Check parity' }));
    expect(calls.checkParity).toHaveBeenCalledWith('pr_1');
    expect(calls.arm).not.toHaveBeenCalled();
  });

  it('⚠ the ack dialog shows the SERVER\'s rung words and its next step', () => {
    state.probes = [probe({ rung: 1 })];
    state.pendingAck = {
      id: 'pr_1',
      details: {
        rung: 1,
        rung_language: 'detects on held-out data',
        next_step: 'evaluate on out-of-distribution sets',
      },
    };
    renderPage();
    const dialog = screen.getByTestId('ack-dialog');
    expect(dialog).toHaveTextContent('rung 1');
    expect(dialog).toHaveTextContent('detects on held-out data');
    expect(dialog).toHaveTextContent('evaluate on out-of-distribution sets');
  });

  it('⚠ confirming the ack sends acknowledge_below_rung2 AND the typed reason', async () => {
    state.probes = [probe({ rung: 1 })];
    state.pendingAck = { id: 'pr_1', details: { rung: 1, rung_language: 'held-out' } };
    renderPage();
    await userEvent.type(screen.getByLabelText(/Why are you arming it/i), 'triage only');
    await userEvent.click(screen.getByTestId('ack-confirm'));
    expect(calls.arm).toHaveBeenCalledWith({
      id: 'pr_1',
      acknowledgeBelowRung2: true,
      reason: 'triage only',
    });
  });

  it('cancelling the ack arms NOTHING', async () => {
    state.probes = [probe({ rung: 1 })];
    state.pendingAck = { id: 'pr_1', details: { rung: 1 } };
    renderPage();
    await userEvent.click(screen.getByTestId('ack-cancel'));
    expect(calls.dismissAck).toHaveBeenCalled();
    expect(calls.arm).not.toHaveBeenCalled();
  });

  it('the ack dialog is absent until the server asks for one', () => {
    state.probes = [probe({ rung: 1 })];
    renderPage();
    // A probe being weak is not itself a prompt: the SERVER decides when consent is needed, so a
    // dialog rendered from `rung < 2` alone would appear for probes nobody tried to arm.
    expect(screen.queryByTestId('ack-dialog')).not.toBeInTheDocument();
  });

  it('the Hub browser is opt-in, not always mounted', async () => {
    state.probes = [];
    renderPage();
    expect(screen.queryByTestId('probe-hub')).not.toBeInTheDocument();
    await userEvent.click(screen.getByRole('button', { name: 'Browse Hub' }));
    expect(screen.getByTestId('probe-hub')).toBeInTheDocument();
  });

  it('reports that continuous batching is off while probes are armed', () => {
    state.probes = [];
    state.status = {
      armed: [],
      armed_count: 2,
      max_armed: 8,
      imported_count: 2,
      paused_reasons: [],
      last_request_overhead_ms: 0.8,
      overhead_warn_threshold_ms: 5,
      events_recorded: 10,
      socket_events_dropped: 0,
      force_serial: true,
    };
    renderPage();
    expect(screen.getByTestId('probe-status')).toHaveTextContent('off (serial)');
  });
});
