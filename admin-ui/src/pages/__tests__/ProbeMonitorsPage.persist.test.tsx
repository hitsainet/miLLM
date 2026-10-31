/**
 * The page's filters and open results panels survive a refresh (operator, 2026-10-29: "Refreshing
 * the page clears the filter setting and closes all the results pages that have been opened").
 *
 * A refresh is modelled as unmount + a fresh mount with a NEW QueryClient, so nothing survives
 * through React state or the query cache — only through `localStorage`. The real hooks and API
 * client run; `fetch`, the socket and storage are fakes. The global setup replaces
 * `window.localStorage` with `vi.fn()` stubs, so each test gives them a backing map (or makes them
 * throw).
 */

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { MemoryRouter } from 'react-router-dom';
import { cleanup, fireEvent, render, screen, waitFor, within } from '@testing-library/react';
import type { Probe, ProbeEvent } from '@/types/probe';

vi.mock('@/services/socket', () => ({
  socketClient: { on: () => undefined, off: () => undefined },
}));
vi.mock('@/components/probes/BehaviourSets', () => ({ BehaviourSets: () => null }));
vi.mock('@/components/probes/RuleSets', () => ({ RuleSets: () => null }));
vi.mock('@/components/probes/ProbeHubBrowser', () => ({ ProbeHubBrowser: () => null }));

const { ProbeMonitorsPage } = await import('../ProbeMonitorsPage');

function probe(id: string, armed: boolean): Probe {
  return {
    id, name: `probe ${id}`, hf_id: 'm', layer: 11, rule: 'mean', scope: 'all',
    basis: 'residual', streamable: true, threshold: 1, target_fpr: 0.01, rung: 2,
    rung_language: 'detects on unseen tasks', next_step: 'x', armed, paused_reason: null,
    parity: null, created_at: '2026-10-29T00:00:00Z',
  } as Probe;
}

function result(id: number, probeId: string, window: string, verdict: boolean): ProbeEvent {
  return {
    id, probe_id: probeId, request_id: `chat-${id}`, scored: true, not_scored_reason: null,
    score: 2, threshold: 1, verdict, rung: 2, top_positions: [1], n_scored_tokens: 4,
    summary: 's', created_at: '2026-10-29T12:00:00Z', window,
  } as ProbeEvent;
}

let probes: Probe[];
let history: Record<string, ProbeEvent[]>;
let store: Map<string, string>;
const storage = window.localStorage as unknown as {
  getItem: ReturnType<typeof vi.fn>;
  setItem: ReturnType<typeof vi.fn>;
};

function respond(body: unknown) {
  return {
    status: 200, ok: true, statusText: 'OK',
    headers: { get: () => 'application/json' },
    json: async () => body,
  } as unknown as Response;
}

function fakeFetch(input: RequestInfo | URL) {
  const url = new URL(String(input), 'http://test');
  if (url.pathname === '/api/probes') return Promise.resolve(respond({ success: true, data: probes }));
  if (url.pathname === '/api/probes/events') {
    const id = url.searchParams.get('probe_id');
    const data = id ? (history[id] ?? []) : [];
    return Promise.resolve(respond({ success: true, data, meta: { next: null, limit: 100 } }));
  }
  return Promise.resolve(respond({ success: false, error: { code: 'NOT_FOUND', message: 'no' } }));
}

/** A fresh page with a fresh cache: what a browser refresh leaves behind is storage alone. */
function mountPage() {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(
    <QueryClientProvider client={client}>
      <MemoryRouter>
        <ProbeMonitorsPage />
      </MemoryRouter>
    </QueryClientProvider>,
  );
}

async function refresh() {
  cleanup();
  mountPage();
  await waitFor(() => expect(screen.getAllByTestId('probe-row').length).toBeGreaterThan(0));
}

const tileNamed = (id: string) =>
  screen
    .getAllByTestId('probe-row')
    .find((row) => within(row).getByTestId('probe-id').textContent === id)!;
const isOpen = (id: string) =>
  within(tileNamed(id)).getByTestId('probe-results-toggle').getAttribute('aria-expanded') === 'true';

beforeEach(() => {
  probes = [probe('pa', true), probe('pb', true), probe('pc', false)];
  history = {};
  store = new Map();
  storage.getItem.mockImplementation((key: string) => store.get(key) ?? null);
  storage.setItem.mockImplementation((key: string, value: string) => {
    store.set(key, String(value));
  });
  vi.stubGlobal('fetch', vi.fn(fakeFetch));
});

afterEach(() => {
  storage.getItem.mockReset();
  storage.setItem.mockReset();
  vi.unstubAllGlobals();
});

describe('page settings survive a refresh (operator 2026-10-29)', () => {
  it('keeps the Armed filter and the open results panels across unmount + remount', async () => {
    await refresh();
    fireEvent.click(screen.getByTestId('probe-filter-armed'));
    fireEvent.click(within(tileNamed('pb')).getByTestId('probe-results-toggle'));
    fireEvent.click(within(tileNamed('pa')).getByTestId('probe-results-toggle'));
    fireEvent.click(within(tileNamed('pa')).getByTestId('probe-results-toggle')); // closed again
    expect(store.get('millm.probes.filter')).toBe('"armed"');
    expect(JSON.parse(store.get('millm.probes.openResults')!)).toEqual(['pb']);

    await refresh();
    expect(screen.getByTestId('probe-filter-armed')).toHaveAttribute('aria-pressed', 'true');
    expect(screen.getAllByTestId('probe-row')).toHaveLength(2); // pc is not armed: hidden
    expect(isOpen('pb')).toBe(true);
    expect(isOpen('pa')).toBe(false);
    // An open panel after a refresh fetches its results again: it is a panel, not a stale copy.
    await waitFor(() =>
      expect(within(tileNamed('pb')).queryByText(/Loading this probe/)).toBeNull(),
    );
  });

  it('drops a remembered id whose probe no longer exists, once the list has arrived', async () => {
    store.set('millm.probes.openResults', JSON.stringify(['pa', 'gone']));
    await refresh();
    expect(isOpen('pa')).toBe(true);
    await waitFor(() =>
      expect(JSON.parse(store.get('millm.probes.openResults')!)).toEqual(['pa']),
    );
  });

  it('does not drop remembered ids while the probe list failed to load', async () => {
    store.set('millm.probes.openResults', JSON.stringify(['pa']));
    vi.stubGlobal('fetch', vi.fn(() => Promise.reject(new Error('offline'))));
    cleanup();
    mountPage();
    await new Promise((resolve) => setTimeout(resolve, 20));
    expect(JSON.parse(store.get('millm.probes.openResults')!)).toEqual(['pa']);
  });

  it('keeps a results panel\'s "fires only" and window filters, per probe', async () => {
    history.pa = [result(1, 'pa', 'prompt', true), result(2, 'pa', 'response', false)];
    await refresh();
    const toggle = within(tileNamed('pa')).getByTestId('probe-results-toggle');
    fireEvent.click(toggle);
    await waitFor(() => expect(within(tileNamed('pa')).getAllByTestId('filter-window').length).toBe(3));
    fireEvent.click(within(tileNamed('pa')).getByTestId('filter-fires-only'));
    fireEvent.click(
      within(tileNamed('pa')).getAllByTestId('filter-window').find((b) => b.textContent === 'prompt')!,
    );

    await refresh();
    await waitFor(() => expect(within(tileNamed('pa')).getAllByTestId('filter-window').length).toBe(3));
    const tile = tileNamed('pa');
    expect(within(tile).getByTestId('filter-fires-only')).toHaveAttribute('aria-pressed', 'true');
    expect(
      within(tile).getAllByTestId('filter-window').find((b) => b.textContent === 'prompt'),
    ).toHaveAttribute('aria-pressed', 'true');
    // Another probe's panel is not narrowed by pa's filters.
    expect(JSON.parse(store.get('millm.probes.resultFilters')!)).toEqual({
      pa: { firesOnly: true, window: 'prompt' },
    });
  });

  it('ignores a remembered window the loaded results do not contain', async () => {
    store.set('millm.probes.openResults', JSON.stringify(['pa']));
    store.set(
      'millm.probes.resultFilters',
      JSON.stringify({ pa: { firesOnly: false, window: 'last_user' } }),
    );
    history.pa = [result(1, 'pa', 'all', true)];
    await refresh();
    await waitFor(() => expect(within(tileNamed('pa')).getAllByTestId('request-group')).toHaveLength(1));
    expect(within(tileNamed('pa')).queryByText('No results match the filter.')).toBeNull();
  });

  it('renders normally when storage throws on every read and write', async () => {
    storage.getItem.mockImplementation(() => {
      throw new DOMException('denied', 'SecurityError');
    });
    storage.setItem.mockImplementation(() => {
      throw new DOMException('full', 'QuotaExceededError');
    });
    await refresh();
    expect(screen.getAllByTestId('probe-row')).toHaveLength(3);
    expect(screen.getByTestId('probe-filter-all')).toHaveAttribute('aria-pressed', 'true');
    fireEvent.click(screen.getByTestId('probe-filter-not_armed'));
    expect(screen.getAllByTestId('probe-row')).toHaveLength(1);
    fireEvent.click(within(tileNamed('pc')).getByTestId('probe-results-toggle'));
    expect(isOpen('pc')).toBe(true);
  });

  it('treats unrecognised stored values as nothing stored', async () => {
    store.set('millm.probes.filter', '"everything"');
    store.set('millm.probes.openResults', '{not json');
    store.set('millm.probes.resultFilters', '[1,2]');
    await refresh();
    expect(screen.getByTestId('probe-filter-all')).toHaveAttribute('aria-pressed', 'true');
    expect(['pa', 'pb', 'pc'].some(isOpen)).toBe(false);
  });
});
