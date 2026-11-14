/**
 * Each probe's results are ITS OWN list (2026-10-29).
 *
 * Operator: "why does it reach a max of 22 verdicts and seemingly overwrite the oldest as each new
 * one comes in". Every probe's panel was a slice of ONE shared list of the newest results across
 * all probes (100 fetched, live appends trimmed to 200). With nine probes armed each chat adds
 * nine results, so about 22 chats fit and each new chat pushed the oldest out of every panel.
 *
 * Unlike `ProbeMonitorsPage.test.tsx`, nothing here mocks `useProbes`: the real hooks, the real
 * API client and the real live-feed reducer run, and only `fetch` and the socket are fakes. A test
 * that stubbed the hook would pass against a page still reading the shared list.
 */

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { MemoryRouter } from 'react-router-dom';
import { act, fireEvent, render, screen, waitFor, within } from '@testing-library/react';
import type { Probe, ProbeEvent, ProbeResultsMessage } from '@/types/probe';

// --- the socket: capture the page's `probe:results` handler so a test can deliver messages ---
const socketHandlers = new Map<string, Set<(payload: unknown) => void>>();
vi.mock('@/services/socket', () => ({
  socketClient: {
    on: (event: string, handler: (payload: unknown) => void) => {
      if (!socketHandlers.has(event)) socketHandlers.set(event, new Set());
      socketHandlers.get(event)!.add(handler);
    },
    off: (event: string, handler: (payload: unknown) => void) => {
      socketHandlers.get(event)?.delete(handler);
    },
  },
}));

// Panels with their own data and their own tests; here they are markers.
vi.mock('@/components/probes/BehaviourSets', () => ({ BehaviourSets: () => null }));
vi.mock('@/components/probes/RuleSets', () => ({ RuleSets: () => null }));
vi.mock('@/components/probes/ProbeHubBrowser', () => ({ ProbeHubBrowser: () => null }));

const { ProbeMonitorsPage } = await import('../ProbeMonitorsPage');
const { MAX_LIVE_PER_PROBE } = await import('@/hooks/useProbes');

/** Hands a message to every `probe:results` subscriber, then lets React Query's batched
 *  notifications reach the page (they are scheduled, not synchronous). */
async function deliver(message: ProbeResultsMessage) {
  const handlers = socketHandlers.get('probe:results') ?? new Set();
  expect(handlers.size).toBeGreaterThan(0);
  await act(async () => {
    for (const handler of handlers) handler(message);
    await new Promise((resolve) => setTimeout(resolve, 0));
  });
}

function probe(id: string): Probe {
  return {
    id, name: `probe ${id}`, hf_id: 'm', layer: 11, rule: 'mean', scope: 'all',
    basis: 'residual', streamable: true, threshold: 1, target_fpr: 0.01, rung: 2,
    rung_language: 'detects on unseen tasks', next_step: 'x', armed: true, paused_reason: null,
    parity: null, created_at: '2026-10-29T00:00:00Z',
  } as Probe;
}

function result(id: number, probeId: string, requestId: string): ProbeEvent {
  return {
    id, probe_id: probeId, request_id: requestId, scored: true, not_scored_reason: null,
    score: 2, threshold: 1, verdict: true, rung: 2, top_positions: [1], n_scored_tokens: 4,
    summary: 's', created_at: '2026-10-29T12:00:00Z', window: 'all',
  } as ProbeEvent;
}

// --- the server: a routed `fetch` that records every URL it was asked for ---
interface Server {
  probes: Probe[];
  /** What `GET /probes/events` (no probe_id) answers: the shared newest-first list. */
  shared: ProbeEvent[];
  /** What `GET /probes/events?probe_id=X` answers, by cursor (`''` = first page). */
  pages: Record<string, Record<string, { data: ProbeEvent[]; next: string | null }>>;
}
let server: Server;
const urls: string[] = [];
const deletes: string[] = [];

function respond(body: unknown) {
  return {
    status: 200,
    ok: true,
    statusText: 'OK',
    headers: { get: () => 'application/json' },
    json: async () => body,
  } as unknown as Response;
}

function fakeFetch(input: RequestInfo | URL, init?: RequestInit) {
  const raw = String(input);
  urls.push(raw);
  const url = new URL(raw, 'http://test');
  if (init?.method === 'DELETE') {
    deletes.push(raw);
    return Promise.resolve(respond({ success: true, data: { removed: 1 } }));
  }
  if (url.pathname === '/api/probes') return Promise.resolve(respond({ success: true, data: server.probes }));
  if (url.pathname === '/api/probes/events') {
    const probeId = url.searchParams.get('probe_id');
    if (!probeId) return Promise.resolve(respond({ success: true, data: server.shared, meta: { next: null } }));
    const page = server.pages[probeId]?.[url.searchParams.get('cursor') ?? ''] ?? { data: [], next: null };
    return Promise.resolve(respond({ success: true, data: page.data, meta: { next: page.next, limit: 100 } }));
  }
  return Promise.resolve(respond({ success: false, error: { code: 'NOT_FOUND', message: 'not here' } }));
}

function renderPage(queries: { refetchOnWindowFocus?: boolean } = {}) {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false, ...queries } } });
  render(
    <QueryClientProvider client={client}>
      <MemoryRouter>
        <ProbeMonitorsPage />
      </MemoryRouter>
    </QueryClientProvider>,
  );
}

async function tileOf(probeId: string) {
  await waitFor(() => expect(screen.getAllByTestId('probe-row').length).toBe(server.probes.length));
  const index = server.probes.findIndex((p) => p.id === probeId);
  return screen.getAllByTestId('probe-row')[index];
}

async function openResults(probeId: string) {
  const tile = await tileOf(probeId);
  fireEvent.click(within(tile).getByTestId('probe-results-toggle'));
  await waitFor(() => expect(within(tile).queryByText(/Loading this probe/)).toBeNull());
  return tile;
}

const requestIds = (tile: HTMLElement) =>
  within(tile).queryAllByTestId('group-request-id').map((el) => el.textContent);

const NINE = Array.from({ length: 9 }, (_, i) => `pr_${i}`);

beforeEach(() => {
  urls.length = 0;
  deletes.length = 0;
  socketHandlers.clear();
  server = { probes: NINE.map(probe), shared: [], pages: {} };
  vi.stubGlobal('fetch', vi.fn(fakeFetch));
});

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("a probe's results are its own list (2026-10-29)", () => {
  it('fetches with THAT probe id, only once opened', async () => {
    server.pages.pr_3 = { '': { data: [result(7, 'pr_3', 'chat-7')], next: null } };
    renderPage();
    await tileOf('pr_3');
    expect(urls.some((u) => u.includes('probe_id='))).toBe(false);

    const tile = await openResults('pr_3');
    const scoped = urls.filter((u) => u.includes('probe_id='));
    expect(scoped).toHaveLength(1);
    const asked = new URL(scoped[0], 'http://test').searchParams;
    expect(asked.get('probe_id')).toBe('pr_3');
    expect(asked.get('limit')).toBe('100');
    expect(requestIds(tile)).toEqual(['chat-7']);
    expect(within(tile).getByTestId('probe-results-count')).toHaveTextContent('Showing 1 result');
  });

  it("⚠ nine armed probes' live results do NOT push another probe's history out", async () => {
    // pr_0 has five older chats on record. Under the old shared list (newest 100 across every
    // probe, live appends trimmed to 200) thirty chats x nine probes = 270 results push them out.
    const old = [5, 4, 3, 2, 1].map((n) => result(n, 'pr_0', `old-${n}`));
    server.shared = old;
    server.pages.pr_0 = { '': { data: old, next: null } };
    renderPage();
    const tile = await openResults('pr_0');
    expect(requestIds(tile)).toEqual(['old-5', 'old-4', 'old-3', 'old-2', 'old-1']);

    let id = 1000;
    for (let chat = 1; chat <= 30; chat += 1) {
      await deliver({
        request_id: `chat-${chat}`,
        results: NINE.map((p) => result(id++, p, `chat-${chat}`)),
      });
    }

    const shown = requestIds(tile);
    expect(shown).toHaveLength(35);
    expect(shown.slice(0, 2)).toEqual(['chat-30', 'chat-29']);
    expect(shown.slice(-5)).toEqual(['old-5', 'old-4', 'old-3', 'old-2', 'old-1']);
    expect(within(tile).getByTestId('probe-results-count')).toHaveTextContent('Showing 35 results');
  });

  it('"Show older" sends the cursor from meta.next and appends the older page', async () => {
    server.pages.pr_1 = {
      '': { data: [result(20, 'pr_1', 'new-20')], next: 'CURSOR-ONE' },
      'CURSOR-ONE': { data: [result(10, 'pr_1', 'mid-10')], next: 'CURSOR-TWO' },
      'CURSOR-TWO': { data: [result(5, 'pr_1', 'old-5')], next: null },
    };
    renderPage();
    const tile = await openResults('pr_1');
    expect(requestIds(tile)).toEqual(['new-20']);

    fireEvent.click(within(tile).getByTestId('probe-results-older'));
    await waitFor(() => expect(requestIds(tile)).toEqual(['new-20', 'mid-10']));
    const second = new URL(urls.filter((u) => u.includes('probe_id=pr_1'))[1], 'http://test');
    expect(second.searchParams.get('cursor')).toBe('CURSOR-ONE');
    expect(second.searchParams.get('probe_id')).toBe('pr_1');

    fireEvent.click(within(tile).getByTestId('probe-results-older'));
    await waitFor(() => expect(requestIds(tile)).toEqual(['new-20', 'mid-10', 'old-5']));
    const third = new URL(urls.filter((u) => u.includes('probe_id=pr_1'))[2], 'http://test');
    expect(third.searchParams.get('cursor')).toBe('CURSOR-TWO');

    // The last page said there is no next: the button goes, and the panel says that is all.
    expect(within(tile).queryByTestId('probe-results-older')).toBeNull();
    expect(within(tile).getByTestId('probe-results-count')).toHaveTextContent(
      'That is every result recorded for this probe',
    );
  });

  it('has no "Show older" button when the first page has no next page', async () => {
    server.pages.pr_2 = { '': { data: [result(3, 'pr_2', 'only')], next: null } };
    renderPage();
    const tile = await openResults('pr_2');
    expect(requestIds(tile)).toEqual(['only']);
    expect(within(tile).queryByTestId('probe-results-older')).toBeNull();
  });

  it("live results for probe A do not appear under probe B", async () => {
    server.pages.pr_0 = { '': { data: [result(1, 'pr_0', 'a-old')], next: null } };
    server.pages.pr_1 = { '': { data: [result(2, 'pr_1', 'b-old')], next: null } };
    renderPage();
    const tileA = await openResults('pr_0');
    const tileB = await openResults('pr_1');

    await deliver({ request_id: 'a-only', results: [result(50, 'pr_0', 'a-only')] });

    expect(requestIds(tileA)).toEqual(['a-only', 'a-old']);
    expect(requestIds(tileB)).toEqual(['b-old']);
  });

  it('a live result already on the list is not listed twice', async () => {
    server.pages.pr_0 = { '': { data: [result(9, 'pr_0', 'chat-9')], next: null } };
    renderPage();
    const tile = await openResults('pr_0');
    await deliver({ request_id: 'chat-9', results: [result(9, 'pr_0', 'chat-9')] });
    expect(within(tile).getAllByTestId('event-row')).toHaveLength(1);
  });

  it('says "Results" with no count before the first fetch, not a count from another list', async () => {
    // The shared list knows of a result for pr_4; the toggle must not present that as pr_4's count.
    server.shared = [result(1, 'pr_4', 'x')];
    renderPage();
    const tile = await tileOf('pr_4');
    expect(within(tile).getByTestId('probe-results-toggle')).toHaveTextContent(/^Results$/);
  });

  it("Clear empties THIS probe's open list (its own list is refetched, not left stale)", async () => {
    server.pages.pr_0 = { '': { data: [result(1, 'pr_0', 'gone')], next: null } };
    renderPage();
    const tile = await openResults('pr_0');
    expect(requestIds(tile)).toEqual(['gone']);

    server.pages.pr_0 = { '': { data: [], next: null } };
    fireEvent.click(within(tile).getByTestId('probe-results-clear'));
    await waitFor(() =>
      expect(within(tile).getByTestId('probe-results')).toHaveTextContent('No results recorded'),
    );
    expect(deletes).toEqual(['/api/probes/events?probe_id=pr_0']);
  });

  it('says so, and offers the newest, once the live cap holds results back', async () => {
    server.pages.pr_0 = { '': { data: [result(1, 'pr_0', 'first')], next: null } };
    renderPage();
    const tile = await openResults('pr_0');
    // One delivery past the cap, then one more that is held back rather than added.
    await deliver({
      request_id: 'flood',
      results: Array.from({ length: MAX_LIVE_PER_PROBE }, (_, i) => result(10_000 + i, 'pr_0', 'flood')),
    });
    expect(within(tile).queryByTestId('probe-results-held')).toBeNull();
    await deliver({ request_id: 'late', results: [result(99_999, 'pr_0', 'late')] });
    expect(within(tile).getByTestId('probe-results-held')).toHaveTextContent('1 more has come in');
    // Nothing already on screen was dropped to make room.
    expect(requestIds(tile)).toEqual(['flood', 'first']);
    expect(within(tile).getAllByTestId('event-row')).toHaveLength(MAX_LIVE_PER_PROBE + 1);
  });
});

describe("the tile's record line reads THIS probe's own loaded results (2026-10-29)", () => {
  // "Fired X of the last Y results" sits in the tile header. Its only honest source is the list
  // the probe's own panel loaded; before that list exists the line is absent, never a count taken
  // from the shared newest-first list or made up.
  it('is absent until the probe has loaded its own results, even when the shared list holds some', async () => {
    server.shared = [result(1, 'pr_4', 'x'), result(2, 'pr_4', 'y')];
    renderPage();
    const tile = await tileOf('pr_4');
    expect(within(tile).queryByTestId('probe-record')).toBeNull();
    expect(tile.textContent).not.toMatch(/Fired \d+ of/);
  });

  it("counts fired results from that probe's own list, not another probe's", async () => {
    const quiet = { ...result(3, 'pr_0', 'c-3'), verdict: false } as ProbeEvent;
    server.pages.pr_0 = {
      '': { data: [result(1, 'pr_0', 'c-1'), result(2, 'pr_0', 'c-2'), quiet], next: null },
    };
    server.pages.pr_1 = {
      '': { data: [{ ...result(9, 'pr_1', 'd-9'), verdict: false } as ProbeEvent], next: null },
    };
    renderPage();
    const tileA = await openResults('pr_0');
    // The panel's list holds every result (no older page), so it is exact: not "the last".
    expect(within(tileA).getByTestId('probe-record')).toHaveTextContent(
      /^Fired 2 of 3 results \(last at .+\)\.$/,
    );
    // pr_1 not opened yet: no line. Opened: its own single, quiet result.
    const tileB = await tileOf('pr_1');
    expect(within(tileB).queryByTestId('probe-record')).toBeNull();
    await openResults('pr_1');
    expect(within(tileB).getByTestId('probe-record')).toHaveTextContent(
      /^Fired 0 of 1 result \(last at .+\)\.$/,
    );
  });
});

describe("a closed tile shows the server's totals (operator 2026-10-29, contract v1.24)", () => {
  // "The number of results is no longer displayed on the tile until I open it." The list route now
  // sends each probe's own `results_total`, `results_fired` and `last_result_at`.
  const withTotals = (id: string, total: number, fired: number, last: string | null) =>
    ({ ...probe(id), results_total: total, results_fired: fired, last_result_at: last }) as Probe;

  it('shows "Fired X of Y results (last at …)" on a closed tile, from that probe\'s own totals', async () => {
    server.probes = NINE.map((id, i) =>
      withTotals(id, 100 + i, i, `2026-10-29T1${i}:00:00Z`),
    );
    renderPage();
    const tile = await tileOf('pr_3');
    const line = within(tile).getByTestId('probe-record');
    expect(line).toHaveTextContent(/^Fired 3 of 103 results \(last at .+\)\.$/);
    expect(within(line).getByTestId('probe-record-last')).toHaveAttribute(
      'dateTime',
      '2026-10-29T13:00:00.000Z',
    );
    // The toggle says the same numbers, and nothing was fetched for the panel.
    const toggle = within(tile).getByTestId('probe-results-toggle');
    expect(toggle).toHaveTextContent('103 results');
    expect(toggle).toHaveTextContent('3 fired');
    expect(urls.some((u) => u.includes('probe_id='))).toBe(false);
  });

  it('a probe with no results reads "No results recorded", with no time', async () => {
    server.probes = NINE.map((id) => withTotals(id, 0, 0, null));
    renderPage();
    const tile = await tileOf('pr_0');
    expect(within(tile).getByTestId('probe-record')).toHaveTextContent(
      'No results recorded for this probe yet.',
    );
    expect(within(tile).queryByTestId('probe-record-last')).toBeNull();
  });

  it('⚠ an open panel holding only the newest page never shows a second, smaller count', async () => {
    server.probes = NINE.map((id) => withTotals(id, 250, 40, '2026-10-29T12:00:00Z'));
    server.pages.pr_0 = { '': { data: [result(1, 'pr_0', 'c-1')], next: 'MORE' } };
    renderPage();
    const tile = await openResults('pr_0');
    expect(within(tile).getByTestId('probe-record')).toHaveTextContent('Fired 40 of 250 results');
    expect(within(tile).getByTestId('probe-results-toggle')).toHaveTextContent('250 results');
    expect(within(tile).getByTestId('probe-results-count')).toHaveTextContent(
      'Showing 1 result of 250',
    );
    expect(tile.textContent).not.toMatch(/Fired \d+ of (the last )?1 result/);
  });

  it('a live result for a closed probe adds to ITS totals only', async () => {
    server.probes = NINE.map((id) => withTotals(id, 10, 2, '2026-10-29T11:00:00Z'));
    renderPage();
    await tileOf('pr_0');
    await deliver({ request_id: 'chat-x', results: [result(500, 'pr_0', 'chat-x')] });
    expect(within(await tileOf('pr_0')).getByTestId('probe-record')).toHaveTextContent(
      'Fired 3 of 11 results',
    );
    expect(within(await tileOf('pr_1')).getByTestId('probe-record')).toHaveTextContent(
      'Fired 2 of 10 results',
    );
  });

  it('⚠ a CLOSED tile counts and colours live results without being expanded (operator 2026-10-29)', async () => {
    // "the number and color of responses showing on the tile is not subscribed to that counter
    // until I expand the results first."
    server.probes = NINE.map((id) => withTotals(id, 4, 0, '2026-10-29T11:00:00Z'));
    renderPage();
    const tile = await tileOf('pr_2');
    const fetchesBefore = urls.length;
    expect(within(tile).getByTestId('probe-record')).toHaveAttribute('data-fired', 'false');
    expect(within(tile).getByTestId('probe-record')).toHaveClass('text-slate-400');
    expect(within(tile).getByTestId('probe-results-toggle')).not.toHaveTextContent('fired');

    const quiet = { ...result(600, 'pr_2', 'c-a'), verdict: false } as ProbeEvent;
    await deliver({ request_id: 'c-a', results: [quiet] });
    expect(within(tile).getByTestId('probe-record')).toHaveTextContent('Fired 0 of 5 results');
    expect(within(tile).getByTestId('probe-record')).toHaveAttribute('data-fired', 'false');

    await deliver({ request_id: 'c-b', results: [result(601, 'pr_2', 'c-b')] });
    const line = within(tile).getByTestId('probe-record');
    expect(line).toHaveTextContent('Fired 1 of 6 results');
    expect(line).toHaveAttribute('data-fired', 'true');
    expect(line).toHaveClass('text-violet-300');
    expect(within(tile).getByTestId('probe-results-toggle')).toHaveTextContent('6 results');
    expect(within(tile).getByTestId('probe-results-toggle')).toHaveTextContent('1 fired');
    // Still closed, and nothing was fetched to get there.
    expect(within(tile).getByTestId('probe-results-toggle')).toHaveAttribute('aria-expanded', 'false');
    // (The status poll is the page's own; neither the list nor any results were fetched.)
    expect(
      urls.slice(fetchesBefore).filter((u) => new URL(u, 'http://test').pathname !== '/api/probes/status'),
    ).toEqual([]);
  });

  it('the header and an open panel agree, before and after a live result', async () => {
    server.probes = NINE.map((id) => withTotals(id, 2, 1, '2026-10-29T11:00:00Z'));
    server.pages.pr_0 = {
      '': { data: [result(2, 'pr_0', 'c-2'), { ...result(1, 'pr_0', 'c-1'), verdict: false } as ProbeEvent], next: null },
    };
    renderPage();
    const tile = await openResults('pr_0');
    expect(within(tile).getByTestId('probe-record')).toHaveTextContent('Fired 1 of 2 results');
    expect(within(tile).getByTestId('probe-results-count')).toHaveTextContent('Showing 2 results,');
    await deliver({ request_id: 'c-3', results: [result(3, 'pr_0', 'c-3')] });
    expect(within(tile).getByTestId('probe-record')).toHaveTextContent('Fired 2 of 3 results');
    expect(within(tile).getByTestId('probe-results-toggle')).toHaveTextContent('3 results');
    expect(within(tile).getByTestId('probe-results-count')).toHaveTextContent('Showing 3 results,');
  });

  it('returning to the tab re-syncs the totals from the server', async () => {
    const { focusManager } = await import('@tanstack/react-query');
    server.probes = NINE.map((id) => withTotals(id, 4, 0, '2026-10-29T11:00:00Z'));
    // The app's QueryClient turns focus refetch OFF by default (App.tsx); the list opts back in.
    renderPage({ refetchOnWindowFocus: false });
    const tile = await tileOf('pr_5');
    server.probes = NINE.map((id) => withTotals(id, 9, 3, '2026-10-29T12:00:00Z'));
    act(() => {
      focusManager.setFocused(false);
      focusManager.setFocused(true);
    });
    await waitFor(() =>
      expect(within(tile).getByTestId('probe-record')).toHaveTextContent('Fired 3 of 9 results'),
    );
    focusManager.setFocused(undefined);
  });

  it('a live result already in the counted totals, or one not stored, is not added again', async () => {
    // The list was fetched at 13:00, after the 12:00 result was stored: it is in the count.
    server.probes = NINE.map((id) => withTotals(id, 10, 2, '2026-10-29T13:00:00Z'));
    renderPage();
    await tileOf('pr_0');
    const unstored = { ...result(501, 'pr_0', 'c-2'), created_at: '2026-10-29T14:00:00Z',
      persistence: 'not_persisted' } as ProbeEvent;
    await deliver({ request_id: 'c-1', results: [result(500, 'pr_0', 'c-1'), unstored] });
    expect(within(await tileOf('pr_0')).getByTestId('probe-record')).toHaveTextContent(
      'Fired 2 of 10 results',
    );
  });

  it('"Clear all results" refetches the totals, so a closed tile does not keep the old count', async () => {
    server.probes = NINE.map((id) => withTotals(id, 10, 2, '2026-10-29T11:00:00Z'));
    server.shared = [result(1, 'pr_1', 'x')];
    renderPage();
    const tile = await tileOf('pr_1');
    expect(within(tile).getByTestId('probe-record')).toHaveTextContent('Fired 2 of 10 results');
    server.probes = NINE.map((id) => withTotals(id, 0, 0, null));
    server.shared = [];
    fireEvent.click(screen.getByText('Clear all results'));
    await waitFor(() =>
      expect(within(tile).getByTestId('probe-record')).toHaveTextContent('No results recorded'),
    );
  });

  it('never takes a count from the shared newest-results list', async () => {
    server.probes = NINE.map((id) => withTotals(id, 7, 1, '2026-10-29T11:00:00Z'));
    server.shared = [result(1, 'pr_4', 'x'), result(2, 'pr_4', 'y')];
    renderPage();
    const tile = await tileOf('pr_4');
    expect(within(tile).getByTestId('probe-record')).toHaveTextContent('Fired 1 of 7 results');
  });
});
