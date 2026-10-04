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
import { fireEvent, render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import type { Probe, ProbeEvent, ProbeStatus } from '@/types/probe';

const state = {
  probes: [] as Probe[],
  status: undefined as ProbeStatus | undefined,
  events: [] as ProbeEvent[],
  eventDetails: {} as Record<number, { summary?: string | null; context_text?: string | null }>,
  eventDetailError: false,
  armingId: null as string | null,
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
    arming: state.armingId !== null,
    armingId: state.armingId,
    checkingParityId: null,
    pendingAck: state.pendingAck,
    dismissAck: calls.dismissAck,
    checkParity: calls.checkParity,
    checkingParity: false,
    disarm: calls.disarm,
    remove: vi.fn(),
    clearEvents: vi.fn(),
  }),
  // Keyed by event id, so a test can assert that EACH expanded row fetched ITS OWN event —
  // a single shared stub would pass against a component that ignored the id.
  useProbeEventDetail: (id: number | null) => ({
    data: id === null ? undefined : state.eventDetails[id],
    isLoading: false,
    isError: Boolean(state.eventDetailError),
  }),
}));

// Imported after the mock so the page picks it up.
const { ProbeMonitorsPage } = await import('../ProbeMonitorsPage');
import { groupByRequest } from '@/utils/probeEvents';
import { labelSeparation, trainingLabels } from '@/utils/probeLabels';

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
    state.armingId = null;
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
    // ⚠ The PAYLOAD, still exact. `windows` joined it on 2026-09-30 and defaults to all three —
    // the capability is useless if it only works for someone who remembers it exists. Asserting
    // the default explicitly is what would catch it silently becoming one window, or none.
    // 2026-10-04: `last_user` joined the default, first — it is the window that reads the
    // newest message alone on a client that resends the conversation.
    expect(calls.arm).toHaveBeenCalledWith({
      id: 'pr_1',
      windows: ['last_user', 'prompt', 'response', 'all'],
    });
  });

  it('the window picker changes what Arm sends', async () => {
    state.probes = [probe()];
    renderPage();
    await userEvent.click(screen.getByRole('button', { name: 'response' }));
    await userEvent.click(screen.getByRole('button', { name: 'Arm' }));
    expect(calls.arm).toHaveBeenCalledWith({ id: 'pr_1', windows: ['last_user', 'prompt', 'all'] });
  });

  it('Arm is refused when every window is deselected', async () => {
    // Not an empty list to the server — an empty list MEANS something there ("the probe's own
    // scope alone"), so a UI that sent one would arm a probe the operator did not ask for.
    state.probes = [probe()];
    renderPage();
    for (const name of ['last user turn', 'all', 'prompt', 'response']) {
      await userEvent.click(screen.getByRole('button', { name }));
    }
    expect(screen.getByRole('button', { name: 'Arm' })).toBeDisabled();
    expect(calls.arm).not.toHaveBeenCalled();
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

  it('⚠ only the probe being armed shows Arming — not every row', async () => {
    // A shared `isPending` flag was passed to every row, so clicking Arm on one probe made ALL
    // of them read "Arming…" and disabled every button, then appear to fail together when the
    // single real request errored. Reported by the operator 2026-09-28.
    state.probes = [
      probe({ id: 'pr_1', name: 'first' }),
      probe({ id: 'pr_2', name: 'second' }),
      probe({ id: 'pr_3', name: 'third' }),
    ];
    state.armingId = 'pr_2';
    renderPage();

    const rows = screen.getAllByTestId('probe-row');
    expect(rows).toHaveLength(3);
    const arming = rows.filter((r) => r.textContent?.includes('Arming'));
    expect(arming).toHaveLength(1);
    expect(arming[0]).toHaveTextContent('second');
  });

  it('the other rows stay clickable while one arms', async () => {
    // Disabling every button is the same defect wearing a different hat: an operator cannot
    // start a second probe, or disarm a running one, while any request is in flight.
    state.probes = [probe({ id: 'pr_1', name: 'first' }), probe({ id: 'pr_2', name: 'second' })];
    state.armingId = 'pr_1';
    renderPage();

    const buttons = screen.getAllByRole('button', { name: /^Arm$/ });
    expect(buttons).toHaveLength(1);
    expect(buttons[0]).not.toBeDisabled();
  });

  it('nothing shows Arming when nothing is in flight', async () => {
    state.probes = [probe({ id: 'pr_1' }), probe({ id: 'pr_2' })];
    state.armingId = null;
    renderPage();
    expect(screen.queryByText(/Arming/)).not.toBeInTheDocument();
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

describe('a verdict row says which probe and which prompt', () => {
  beforeEach(() => {
    state.probes = [];
    state.events = [];
  });

  it('⚠ names the PROBE, because two probes on one layer produce two verdicts per prompt', () => {
    // The reported case: one chat, two verdicts, one firing and one not. Without a name the pair
    // is indistinguishable and the operator cannot tell which detector said what.
    state.probes = [
      probe({ id: 'pr_dense', name: 'high-stakes-dense', armed: true }),
      probe({ id: 'pr_sae', name: 'high-stakes-ksparse', basis: 'sae_features', armed: true }),
    ];
    state.events = [
      event({ id: 2, probe_id: 'pr_sae', request_id: 'chatcmpl-x', verdict: false, score: -1.2 }),
      event({ id: 1, probe_id: 'pr_dense', request_id: 'chatcmpl-x', verdict: true, score: 6.3 }),
    ];
    renderPage();

    const names = screen.getAllByTestId('event-probe-name').map((n) => n.textContent);
    expect(names).toContain('high-stakes-ksparse');
    expect(names).toContain('high-stakes-dense');
  });

  it('falls back to the probe id when the probe is gone — an orphan verdict is still evidence', () => {
    state.probes = [];
    state.events = [event({ probe_id: 'pr_deleted' })];
    renderPage();
    expect(screen.getByTestId('event-probe-name')).toHaveTextContent('pr_deleted');
  });

  it('groups the verdicts from one request together, under that request', () => {
    state.probes = [probe({ id: 'pr_a', name: 'A' }), probe({ id: 'pr_b', name: 'B' })];
    state.events = [
      event({ id: 3, probe_id: 'pr_a', request_id: 'chatcmpl-second' }),
      event({ id: 2, probe_id: 'pr_b', request_id: 'chatcmpl-first' }),
      event({ id: 1, probe_id: 'pr_a', request_id: 'chatcmpl-first' }),
    ];
    renderPage();

    const groups = screen.getAllByTestId('request-group');
    expect(groups).toHaveLength(2);
    const ids = screen.getAllByTestId('group-request-id').map((n) => n.textContent);
    // Arrival order preserved: the newest request stays first.
    expect(ids).toEqual(['chatcmpl-second', 'chatcmpl-first']);
    expect(groups[1]).toHaveTextContent('2 verdicts');
    expect(groups[0]).toHaveTextContent('1 verdict');
  });

  it('shows a timestamp, so two verdicts on the same prompt can be placed in time', () => {
    state.probes = [probe()];
    state.events = [event({ created_at: '2026-09-28T09:24:38Z' })];
    renderPage();
    // Rendered via toLocaleTimeString, so assert it is non-empty rather than a fixed locale string.
    expect(screen.getByTestId('event-row').textContent).toMatch(/\d/);
  });
});

describe('groupByRequest', () => {
  it('keeps arrival order and does not merge different requests', () => {
    const rows = [
      event({ id: 3, request_id: 'r2' }),
      event({ id: 2, request_id: 'r1' }),
      event({ id: 1, request_id: 'r1' }),
    ];
    expect(groupByRequest(rows).map(([key, group]) => [key, group.length])).toEqual([
      ['r2', 1],
      ['r1', 2],
    ]);
  });

  it('a null request_id does not swallow every other group', () => {
    const rows = [event({ id: 2, request_id: null }), event({ id: 1, request_id: 'r1' })];
    const groups = groupByRequest(rows);
    expect(groups).toHaveLength(2);
    expect(groups[0][0]).toBe('—');
  });
});

describe('the prompt window opens inline, per event', () => {
  beforeEach(() => {
    state.probes = [];
    state.events = [];
    state.eventDetails = {};
    state.eventDetailError = false;
  });

  it('is closed until the row is clicked, and then sits under that row', async () => {
    state.probes = [probe({ id: 'pr_1', name: 'A' })];
    state.events = [event({ id: 7 })];
    state.eventDetails = { 7: { summary: 'score 2.31', context_text: 'the user asked about X' } };
    renderPage();

    expect(screen.queryByTestId('event-context')).toBeNull();

    const row = screen.getByTestId('event-row');
    expect(row).toHaveAttribute('aria-expanded', 'false');
    fireEvent.click(row);

    const panel = await screen.findByTestId('event-context');
    expect(panel).toHaveTextContent('the user asked about X');
    expect(screen.getByTestId('event-row')).toHaveAttribute('aria-expanded', 'true');
    // ⚠ Under THAT row, not in a detached panel: the row and its window share a parent.
    expect(screen.getByTestId('event-row').parentElement).toContainElement(panel);
  });

  it('clicking again closes it', async () => {
    state.probes = [probe()];
    state.events = [event({ id: 7 })];
    state.eventDetails = { 7: { context_text: 'words' } };
    renderPage();

    fireEvent.click(screen.getByTestId('event-row'));
    expect(await screen.findByTestId('event-context')).toBeTruthy();
    fireEvent.click(screen.getByTestId('event-row'));
    expect(screen.queryByTestId('event-context')).toBeNull();
  });

  it('⚠ TWO events can be open at once, each showing ITS OWN prompt', async () => {
    // The reported ask. A single `openEventId` could only ever show one window, and showed it
    // detached from the row. Two probes on one request is the normal case, so comparing their
    // two windows side by side is the thing this has to support.
    state.probes = [probe({ id: 'pr_a', name: 'A' }), probe({ id: 'pr_b', name: 'B' })];
    state.events = [
      event({ id: 11, probe_id: 'pr_a', request_id: 'chatcmpl-x' }),
      event({ id: 12, probe_id: 'pr_b', request_id: 'chatcmpl-x' }),
    ];
    state.eventDetails = {
      11: { context_text: 'window for eleven' },
      12: { context_text: 'window for twelve' },
    };
    renderPage();

    const rows = screen.getAllByTestId('event-row');
    fireEvent.click(rows[0]);
    fireEvent.click(rows[1]);

    const panels = await screen.findAllByTestId('event-context');
    expect(panels).toHaveLength(2);
    const texts = screen.getAllByTestId('event-context-text').map((n) => n.textContent);
    expect(texts).toContain('window for eleven');
    expect(texts).toContain('window for twelve');
  });

  it('opening one row does not open the others', async () => {
    state.probes = [probe({ id: 'pr_a', name: 'A' }), probe({ id: 'pr_b', name: 'B' })];
    state.events = [
      event({ id: 11, probe_id: 'pr_a', request_id: 'r1' }),
      event({ id: 12, probe_id: 'pr_b', request_id: 'r1' }),
    ];
    state.eventDetails = { 11: { context_text: 'eleven' }, 12: { context_text: 'twelve' } };
    renderPage();

    fireEvent.click(screen.getAllByTestId('event-row')[0]);
    await screen.findByTestId('event-context');

    expect(screen.getAllByTestId('event-context')).toHaveLength(1);
    expect(screen.getAllByTestId('event-row')[1]).toHaveAttribute('aria-expanded', 'false');
  });

  it('⚠ an event with no recorded window SAYS SO rather than rendering blank', async () => {
    // A blank panel is indistinguishable from the defect where nothing captured the window at
    // all — which is exactly what was reported. Events from before the capture shipped have no
    // text and never will, so the absence has to be stated.
    state.probes = [probe()];
    state.events = [event({ id: 3 })];
    state.eventDetails = { 3: { summary: 'score 1.0', context_text: null } };
    renderPage();

    fireEvent.click(screen.getByTestId('event-row'));
    expect(await screen.findByTestId('event-context-absent')).toHaveTextContent(
      'No prompt window was recorded',
    );
    expect(screen.queryByTestId('event-context-text')).toBeNull();
  });

  it('a failed fetch is reported, not silently empty', async () => {
    state.probes = [probe()];
    state.events = [event({ id: 3 })];
    state.eventDetailError = true;
    renderPage();

    fireEvent.click(screen.getByTestId('event-row'));
    expect(await screen.findByText(/Could not load/)).toBeTruthy();
  });
});

describe('the overhead tile judges the RATE, and survives a backend that does not send it', () => {
  // ⚠ SELF-INFLICTED, CAUGHT BY THE SUITE 2026-09-30. The per-pass fields were added to the
  // status payload and the tile guarded them with `!== null` — which is TRUE for `undefined`,
  // so `.toFixed(3)` threw and took the WHOLE PAGE down, not just the tile. Thirteen tests went
  // red at once. During a rolling deploy this bundle can meet a backend from before the change,
  // which is exactly the fixture below.
  //
  // The old fixtures cover this only by accident, because they predate the fields. An accident
  // stops covering the moment someone updates a fixture, so it is asserted here on purpose.

  it('renders when the backend sends none of the per-pass fields', () => {
    state.probes = [];
    state.events = [];
    state.status = {
      armed: [],
      armed_count: 1,
      max_armed: 8,
      imported_count: 1,
      paused_reasons: [],
      last_request_overhead_ms: 11.36,
      overhead_warn_threshold_ms: 500,
      events_recorded: 2,
      socket_events_dropped: 0,
      force_serial: true,
      // last_request_n_passes, last_overhead_ms_per_pass and
      // overhead_warn_threshold_ms_per_pass deliberately ABSENT.
    } as never;
    renderPage();
    const status = screen.getByTestId('probe-status');
    expect(status).toHaveTextContent('11.36 ms');
    expect(status).not.toHaveTextContent('per pass');
  });

  it('shows the rate against its budget when the backend does send it', () => {
    state.probes = [];
    state.events = [];
    state.status = {
      armed: [],
      armed_count: 1,
      max_armed: 8,
      imported_count: 1,
      paused_reasons: [],
      last_request_overhead_ms: 11.36,
      overhead_warn_threshold_ms: 500,
      last_request_n_passes: 121,
      last_overhead_ms_per_pass: 0.0939,
      overhead_warn_threshold_ms_per_pass: 0.25,
      events_recorded: 2,
      socket_events_dropped: 0,
      force_serial: true,
    } as never;
    renderPage();
    const status = screen.getByTestId('probe-status');
    // ⚠ The total is shown against the PASS COUNT, never against a budget — pairing it with a
    // threshold is what made a long healthy answer look like a fault.
    expect(status).toHaveTextContent('over 121 passes');
    expect(status).toHaveTextContent('0.094 /0.25 ms per pass');
  });
});

describe('the parity badge shows the numbers that DECIDED the verdict', () => {
  // ⚠ REPORTED FROM THE RUNNING UI 2026-09-30. The badge read
  //     parity passed · max Δ 9.37e+0 of 0.05
  // — a figure 187x its stated tolerance, printed beside the word "passed". Both numbers were
  // real and neither decided anything: they are the PER-TOKEN pair, which the parity engine
  // marks `per_token_is_informational: true`. `passed` came from the aggregate score difference,
  // 0.0789 against a `score_tolerance` of 0.1.
  //
  // The UI type did not even carry the deciding fields, which is why the wrong pair was shown.
  //
  // This matters more than a cosmetic mislabel: a reader either distrusts a correct pass, or
  // believes a 9.37 drift is meaningful. A parity check that reports the wrong thing is believed
  // the first time — this estate shipped one that told a correct consumer it was wrong on every
  // vector, against a tolerance it was never measured against.

  const parity = {
    passed: true,
    tolerance: 0.05,
    max_abs_diff: 9.374565124511719,
    score_tolerance: 0.1,
    max_gated_diff: 0.07888734340667725,
    max_combined_diff: 0.07888734340667725,
    per_token_is_informational: true,
    at_risk_tokens: 0,
    scored_tokens: 6301,
    vector_index: 12,
    vectors: [],
    tokenization_drift: {},
    error: null,
  };

  it('shows the gated difference against the tolerance that gated it', () => {
    state.probes = [probe({ parity } as never)];
    renderPage();
    expect(screen.getByTestId('parity')).toHaveTextContent('0.079 / 0.100');
  });

  it('rounds a relative tolerance instead of printing float noise', () => {
    // The matched floor is 0.6% of the bar: 0.006 x 46.78 is 0.28068 in floating point.
    state.probes = [probe({ parity: { ...parity, score_tolerance: 0.006 * 46.78 } } as never)];
    renderPage();
    const text = screen.getByTestId('parity').textContent ?? '';
    expect(text).toContain('0.079 / 0.281');
    expect(text).not.toMatch(/0\.2806\d{5,}/);
  });

  it('⚠ never pairs the per-token maximum with a tolerance', () => {
    // The exact string that started this. "9.37e+0 of 0.05" reads as a failed check.
    state.probes = [probe({ parity } as never)];
    renderPage();
    const text = screen.getByTestId('parity').textContent ?? '';
    expect(text).not.toMatch(/9\.37e\+0 of 0\.05/);
    expect(text).not.toMatch(/of 0\.05/);
  });

  it('still shows the per-token figure, LABELLED as informational', () => {
    // Not hidden: one token drifting far is worth knowing. It just must not read as the verdict.
    state.probes = [probe({ parity } as never)];
    renderPage();
    const text = screen.getByTestId('parity').textContent ?? '';
    expect(text).toContain('per-token max Δ 9.37');
    expect(text).toContain('informational');
  });

  it('a FAILED check still reads as failed', () => {
    state.probes = [probe({ parity: { ...parity, passed: false, max_gated_diff: 0.4 } } as never)];
    renderPage();
    const text = screen.getByTestId('parity').textContent ?? '';
    expect(text).toContain('FAILED');
    expect(text).toContain('0.400 / 0.100');
  });

  it('omits the pair entirely when the backend does not send it', () => {
    // ⚠ A report stored before these fields existed, or an older backend mid-deploy. Falling
    // back to the per-token pair would reinstate the exact defect; showing nothing is honest.
    const older = { ...parity };
    delete (older as Record<string, unknown>).max_gated_diff;
    delete (older as Record<string, unknown>).score_tolerance;
    state.probes = [probe({ parity: older } as never)];
    renderPage();
    const text = screen.getByTestId('parity').textContent ?? '';
    expect(text).toContain('parity passed');
    expect(text).not.toMatch(/of 0\.05/);
    expect(text).toContain('informational');
  });
});

describe('one probe, several windows, told apart', () => {
  /* ⚠ THE PROBE NAME STOPPED BEING A DISCRIMINATOR ON 2026-09-30.
   *
   * A probe reports up to three windows per request — `all`, `prompt`, `response` — so a request
   * group can hold three rows that share a probe, a name and a request id, differing only in a
   * number. The page's own note used to say the pair was told apart BY NAME, which is exactly
   * what stops working.
   *
   * `provisional` matters more than it looks: the `response` window is not merely uncalibrated,
   * it is UNTRAINED — miStudio's corpus is prose wrapped as a single user turn, so those weights
   * never saw a model reply. The operator chose to let it fire anyway. The marker is the
   * condition that choice was made under.
   */

  it('shows the window on every row', () => {
    state.probes = [probe()];
    state.events = [
      event({ id: 1, window: 'all', score: 2.31 }),
      event({ id: 2, window: 'prompt', score: 9.86 }),
      event({ id: 3, window: 'response', score: -4.2, verdict: false }),
    ];
    renderPage();
    const windows = screen.getAllByTestId('event-window').map((e) => e.textContent);
    expect(windows).toEqual(['all', 'prompt', 'response']);
  });

  it('marks a provisional verdict, and only a provisional one', () => {
    state.probes = [probe()];
    state.events = [
      event({ id: 1, window: 'all', provisional: false }),
      event({ id: 2, window: 'response', provisional: true }),
    ];
    renderPage();
    // Specificity: a marker on every row says nothing.
    expect(screen.getAllByTestId('event-provisional')).toHaveLength(1);
    const rows = screen.getAllByTestId('event-row');
    expect(rows[1]).toHaveTextContent('provisional');
    expect(rows[0]).not.toHaveTextContent('provisional');
  });

  it('three windows of one probe still group under one request', () => {
    state.probes = [probe()];
    state.events = [
      event({ id: 1, window: 'all' }),
      event({ id: 2, window: 'prompt' }),
      event({ id: 3, window: 'response' }),
    ];
    renderPage();
    expect(screen.getByTestId('request-group')).toHaveTextContent('3 verdicts');
  });

  it('⚠ renders when the backend sends no window at all', () => {
    // A backend from before this shipped omits both fields, and a rolling deploy puts this
    // bundle in front of one. A strict check on an absent field has already taken this page
    // down once this week.
    state.probes = [probe()];
    state.events = [event({ id: 1 })];
    renderPage();
    expect(screen.getByTestId('event-row')).toBeInTheDocument();
    expect(screen.queryByTestId('event-window')).not.toBeInTheDocument();
    expect(screen.queryByTestId('event-provisional')).not.toBeInTheDocument();
  });
});

/*
 * ⚠ THE TILE SAID WHERE A PROBE READS AND NEVER WHAT IT DETECTS. The row carried
 * `LiquidAI/LFM2.5-1.2B-Instruct · L11 · mean · dense residual · scope all` — the read point,
 * identical across probes that differ only in the corpus they were fitted on. The operator
 * asked for the training labels on 2026-10-01.
 */
describe('a probe row states the separation it was fitted to make', () => {
  const MAPPING = { 'low-stakes': 'negative', 'high-stakes': 'positive' };

  it('names both sides, positive first', () => {
    /*
     * ⚠ THIS EXACT PAIR IS PINNED IN miStudio TOO (`probeTile.test.tsx`). The two repos do not
     * share a frontend, so the format string is what can drift; the same real mapping must
     * produce the same caption on both sides or one of these fails.
     */
    expect(labelSeparation(MAPPING)).toBe('high-stakes vs low-stakes');
  });

  it('joins several labels on either side rather than showing the first', () => {
    expect(
      labelSeparation({ critical: 'positive', 'high-stakes': 'positive', calm: 'negative', idle: 'negative' })
    ).toBe('critical or high-stakes vs calm or idle');
  });

  it('refuses to print half a contrast', () => {
    expect(labelSeparation({ 'high-stakes': 'positive' })).toBeNull();
    expect(labelSeparation({ 'low-stakes': 'negative' })).toBeNull();
    expect(labelSeparation({})).toBeNull();
    expect(labelSeparation(null)).toBeNull();
    expect(labelSeparation(undefined)).toBeNull();
  });

  it('does not count an excluded label as either side', () => {
    expect(labelSeparation({ 'high-stakes': 'positive', ambiguous: 'excluded' })).toBeNull();
    expect(trainingLabels({ 'high-stakes': 'positive', ambiguous: 'excluded' })).toEqual({
      positive: ['high-stakes'],
      negative: [],
      excluded: ['ambiguous'],
    });
  });

  it('renders on the row', () => {
    state.probes = [probe({ label_mapping: MAPPING })];
    render(<ProbeMonitorsPage />);
    expect(screen.getByTestId('probe-labels')).toHaveTextContent(
      'trained on high-stakes vs low-stakes'
    );
  });

  it('puts the whole mapping in the tooltip, including a label on neither side', () => {
    // The caption shows the boundary; the title must hide nothing the operator decided.
    state.probes = [probe({ label_mapping: { ...MAPPING, ambiguous: 'excluded' } })];
    render(<ProbeMonitorsPage />);
    const title = screen.getByTestId('probe-labels').getAttribute('title') ?? '';
    expect(title).toContain('ambiguous → excluded');
    expect(title).toContain('high-stakes → positive');
    expect(title).toContain('low-stakes → negative');
  });

  it('renders nothing at all when the backend sends no mapping', () => {
    // An older backend mid-deploy omits the field. A blank is honest; a placeholder word is not.
    state.probes = [probe({ label_mapping: undefined })];
    render(<ProbeMonitorsPage />);
    expect(screen.queryByTestId('probe-labels')).toBeNull();
  });
});

describe('a bar that can move, and verdicts judged under different ones', () => {
  /* ⚠ A THRESHOLD IS NO LONGER A PROPERTY OF THE RUN THAT PRODUCED IT.
   *
   * miStudio can re-cut a probe's bar in milliseconds — a threshold is the (1 − target_fpr)
   * quantile of negatives it already has on disk — and that cut can now reach an ALREADY
   * IMPORTED, possibly ARMED probe in place, because the alternative (disarm → delete → import)
   * assigns a new probe id and cascade-deletes the probe's entire verdict history.
   *
   * Two consequences land on this page, and both are honesty problems rather than features:
   *
   *   1. An event list can hold verdicts judged against DIFFERENT bars. The row showed a score
   *      and `· fires` and never the threshold, so two rows with the same score and opposite
   *      verdicts were indistinguishable — an omission while a bar was fixed, a lie once it can
   *      move.
   *   2. `GET /api/probes` serialises the ROW. If a re-cut writes the database and the in-memory
   *      `ArmedProbe` is not replaced, this tile shows the new number while every verdict keeps
   *      using the old one. `status()` reads the live registry and REPORTS that disagreement
   *      rather than reconciling it; the tile renders what it says.
   */

  it('shows the bar each verdict was judged against, beside the score', () => {
    state.probes = [probe()];
    state.events = [event({ id: 1, score: 12.5, threshold: 11.9144, verdict: true })];
    renderPage();
    expect(screen.getByTestId('event-row')).toHaveTextContent('12.5000 / 11.9144');
  });

  it('marks a verdict judged under an EARLIER revision, and only that one', () => {
    state.probes = [probe({ threshold_revision: 3 })];
    state.events = [
      event({ id: 1, threshold_revision: 1 }),
      event({ id: 2, threshold_revision: 3 }),
    ];
    renderPage();
    // Specificity again: a chip on every row says nothing.
    expect(screen.getAllByTestId('event-threshold-revision')).toHaveLength(1);
    const rows = screen.getAllByTestId('event-row');
    expect(rows[0]).toHaveTextContent('rev 1');
    expect(rows[1]).not.toHaveTextContent('rev');
  });

  it('OMITS the chip for an orphaned verdict rather than defaulting it', () => {
    /* A probe that has since been deleted has no current revision, so there is nothing to
       compare against. A chip there would warn about a bar nobody can look up. */
    state.probes = [];
    state.events = [event({ id: 1, probe_id: 'pr_gone', threshold_revision: 1 })];
    renderPage();
    expect(screen.getByTestId('event-row')).toBeInTheDocument();
    expect(screen.queryByTestId('event-threshold-revision')).not.toBeInTheDocument();
  });

  it('says nothing about revisions on a probe whose bar has never moved', () => {
    state.probes = [probe({ threshold_revision: 1 })];
    state.events = [event({ id: 1, threshold_revision: 1 })];
    renderPage();
    expect(screen.queryByTestId('event-threshold-revision')).not.toBeInTheDocument();
    expect(screen.getByTestId('probe-threshold')).not.toHaveTextContent('rev');
  });

  it('shows the precision the probe was fitted at', () => {
    state.probes = [probe({ load_dtype: 'bfloat16' })];
    renderPage();
    expect(screen.getByTestId('probe-load-dtype')).toHaveTextContent('fitted at bfloat16');
  });

  it('says a precision was not recorded rather than supplying one', () => {
    // A definition from before 2026-10-03 states no precision. It WAS float16, but the tile must
    // not supply a fact the document did not record.
    state.probes = [probe({ load_dtype: null })];
    renderPage();
    expect(screen.getByTestId('probe-load-dtype')).toHaveTextContent('unrecorded precision');
  });

  it('shows the cut on the tile once a bar HAS moved', () => {
    state.probes = [probe({ threshold_revision: 4 })];
    renderPage();
    expect(screen.getByTestId('probe-threshold')).toHaveTextContent('rev 4');
  });

  it("⚠ renders a backend that sends no revision at all", () => {
    // A rolling deploy puts this bundle in front of an older backend. A strict check on an
    // absent field has taken this page down before.
    state.probes = [probe()];
    state.events = [event({ id: 1 })];
    renderPage();
    expect(screen.getByTestId('event-row')).toBeInTheDocument();
    expect(screen.queryByTestId('event-threshold-revision')).not.toBeInTheDocument();
  });
});

describe('the bar in force, when it is not the one stored', () => {
  /* ⚠ THE DEFECT THIS RENDERS IS THE ONE THE WHOLE CROSS-REPO DESIGN EXISTS TO PREVENT.
   *
   * `armed_probe_from_row` resolves a row into the runtime shape ONCE, at arm time, and the
   * request path never touches the database. So a recalibration that writes the row and fails to
   * replace the in-memory `ArmedProbe` leaves this tile — which serialises the ROW — showing the
   * new threshold while every verdict is still judged against the old one. Invisible-but-visible,
   * which is worse than a plain failure.
   *
   * `status()` is the only surface that reads the live registry, and it reports the disagreement
   * rather than reconciling it: a read path that quietly writes would hide how often it happens.
   */

  const statusWith = (armed: Record<string, unknown>[]): ProbeStatus =>
    ({
      armed,
      armed_count: armed.length,
      max_armed: 8,
      imported_count: armed.length,
      paused_reasons: [],
      last_request_overhead_ms: 0.3,
      overhead_warn_threshold_ms: 5,
      events_recorded: 0,
      socket_events_dropped: 0,
      force_serial: true,
    }) as unknown as ProbeStatus;

  const armedEntry = (over: Record<string, unknown> = {}) => ({
    id: 'pr_1',
    name: 'high-stakes',
    layer: 11,
    rule: 'mean',
    rung: 2,
    rung_language: 'detects on unseen tasks',
    next_step: '',
    streamable: true,
    basis: 'residual',
    paused_reason: null,
    hook_installed: true,
    windows: ['all'],
    ...over,
  });

  it("shows the server's sentence when the live bar is not the stored one", () => {
    const warning =
      'the stored bar is revision 2 and this probe is judging against revision 1 (11.9144) — ' +
      'the registry was not refreshed when the threshold moved; re-arm it';
    state.probes = [probe({ armed: true, threshold: 14.5201, threshold_revision: 2 })];
    state.status = statusWith([
      armedEntry({ threshold: 11.9144, threshold_revision: 1, threshold_disagreement: warning }),
    ]);
    renderPage();
    expect(screen.getByTestId('threshold-disagreement')).toHaveTextContent(warning);
  });

  it('says nothing when the live bar and the row agree', () => {
    state.probes = [probe({ armed: true, threshold: 14.5201, threshold_revision: 2 })];
    state.status = statusWith([
      armedEntry({ threshold: 14.5201, threshold_revision: 2, threshold_disagreement: null }),
    ]);
    renderPage();
    expect(screen.queryByTestId('threshold-disagreement')).not.toBeInTheDocument();
  });

  it('keeps the disagreement OUT of paused_reason', () => {
    /* A probe judging against a previous bar IS still scoring. Folding the two together would
       dilute the one field whose whole job is "a probe never goes silently quiet". */
    state.probes = [probe({ armed: true, threshold: 14.5201, threshold_revision: 2 })];
    state.status = statusWith([
      armedEntry({ threshold: 11.9144, threshold_revision: 1, threshold_disagreement: 'moved' }),
    ]);
    renderPage();
    expect(screen.getByTestId('threshold-disagreement')).toBeInTheDocument();
    expect(screen.queryByTestId('paused-reason')).not.toBeInTheDocument();
  });

  it('⚠ renders against a backend that reports no live bar at all', () => {
    state.probes = [probe({ armed: true, threshold: 14.5201 })];
    state.status = statusWith([armedEntry()]);
    renderPage();
    expect(screen.getByTestId('probe-threshold')).toBeInTheDocument();
    expect(screen.queryByTestId('threshold-disagreement')).not.toBeInTheDocument();
  });
});

describe('the tile says which tokens a probe reads and names its rolling window (2026-10-04)', () => {
  const bars = { all: 46.78, prompt: 28.8, response: 46.57 };

  it('names the rolling window, the only thing telling w=32 from w=64 apart', () => {
    state.probes = [
      probe({ id: 'pr_32', rule: 'rolling_mean_max', rule_params: { window: 32 } }),
      probe({ id: 'pr_64', rule: 'rolling_mean_max', rule_params: { window: 64 } }),
    ];
    renderPage();
    const readouts = screen.getAllByTestId('probe-readout').map((el) => el.textContent);
    expect(readouts[0]).toContain('rolling_mean_max w=32');
    expect(readouts[1]).toContain('rolling_mean_max w=64');
  });

  it('adds nothing for a rule with no window', () => {
    state.probes = [probe({ rule: 'mean', rule_params: {} })];
    renderPage();
    expect(screen.getByTestId('probe-readout').textContent).not.toContain('w=');
  });

  it("shows what it was fitted on and each window's own bar, response provisional", () => {
    state.probes = [probe({ window_thresholds: bars, length_band_count: 4 })];
    renderPage();
    expect(screen.getByTestId('probe-scope')).toHaveTextContent('fitted on every token (prompt and reply)');
    expect(screen.getByTestId('window-prompt')).toHaveTextContent('prompt ≥ 28.80');
    expect(screen.getByTestId('window-response')).toHaveTextContent('response ≥ 46.57 · provisional');
    expect(screen.getByTestId('window-all')).toHaveTextContent('all ≥ 46.78');
    expect(screen.getByTestId('window-prompt')).not.toHaveTextContent('provisional');
    expect(screen.getByTestId('window-length-bands')).toHaveTextContent('+ 4 length bands on all');
  });

  it('says so when one bar serves every window', () => {
    state.probes = [probe({ window_thresholds: {} })];
    renderPage();
    expect(screen.getByTestId('window-single-bar')).toBeInTheDocument();
  });
});

describe('the last user turn window (2026-10-04)', () => {
  it('is offered first in the window picker, labelled for a person', () => {
    state.probes = [probe({ armed: false })];
    renderPage();
    const labels = screen.getAllByRole('button', { pressed: undefined })
      .map((b) => b.textContent)
      .filter((t) => ['last user turn', 'prompt', 'response', 'all'].includes(t ?? ''));
    expect(labels[0]).toBe('last user turn');
  });

  it('shows its own bar on the tile, not provisional', () => {
    state.probes = [probe({ window_thresholds: { last_user: 19.5, prompt: 28.8, response: 46.57, all: 46.78 } })];
    renderPage();
    expect(screen.getByTestId('window-last_user')).toHaveTextContent('last user turn ≥ 19.50');
    expect(screen.getByTestId('window-last_user')).not.toHaveTextContent('provisional');
  });
});
