/**
 * The live feed appends a WHOLE request (Feature 39, TD39-28; FTASKS 7.4).
 *
 * `probe:results` carries every result of one request; the reducer puts them at the head of each
 * cached list, together and in the server's order, and a list scoped to one probe keeps only that
 * probe's results.
 */

import { describe, expect, it } from 'vitest';
import { QueryClient } from '@tanstack/react-query';
import { appendRequest, MAX_LIVE_PER_PROBE } from '../useProbes';
import type { ProbeEvent } from '@/types/probe';

function event(id: number, probeId: string, window = 'all'): ProbeEvent {
  return {
    id, probe_id: probeId, request_id: 'chatcmpl-new', scored: true, not_scored_reason: null,
    score: 1, threshold: 0.5, verdict: true, rung: 2, top_positions: [], n_scored_tokens: 3,
    summary: 's', created_at: '2026-10-24T00:00:00Z', window,
  } as ProbeEvent;
}

describe('appendRequest', () => {
  it('puts the whole request at the head of every list, scoped lists keeping their probe', () => {
    const client = new QueryClient();
    const old = event(1, 'pr_a');
    client.setQueryData(['probes', 'events', 'all'], [old]);
    client.setQueryData(['probes', 'events', 'pr_b'], []);
    const results = [event(10, 'pr_a'), event(11, 'pr_a', 'prompt'), event(12, 'pr_b')];
    appendRequest(client, { request_id: 'chatcmpl-new', results });
    expect(client.getQueryData<ProbeEvent[]>(['probes', 'events', 'all'])?.map((e) => e.id))
      .toEqual([10, 11, 12, 1]);
    expect(client.getQueryData<ProbeEvent[]>(['probes', 'events', 'pr_b'])?.map((e) => e.id))
      .toEqual([12]);
  });

  it("adds a probe's own results to its open history, and stops (never trims) at the live cap", () => {
    const client = new QueryClient();
    const key = ['probes', 'history', 'pr_a'];
    client.setQueryData(key, {
      pages: [{ data: [event(1, 'pr_a')], next: 'C' }, { data: [event(0, 'pr_a')], next: null }],
      pageParams: [null, 'C'],
    });
    appendRequest(client, { request_id: 'r', results: [event(10, 'pr_a'), event(11, 'pr_b')] });
    type H = { pages: { data: ProbeEvent[]; next: string | null; live?: number; held?: number }[] };
    let data = client.getQueryData<H>(key)!;
    expect(data.pages.map((p) => p.data.map((e) => e.id))).toEqual([[10, 1], [0]]);
    expect(data.pages[0].live).toBe(1);

    // At the cap: nothing is added and nothing is dropped; what was held back is counted.
    client.setQueryData(key, { ...data, pages: [{ ...data.pages[0], live: MAX_LIVE_PER_PROBE }, data.pages[1]] });
    appendRequest(client, { request_id: 'r2', results: [event(12, 'pr_a'), event(13, 'pr_a')] });
    data = client.getQueryData<H>(key)!;
    expect(data.pages.map((p) => p.data.map((e) => e.id))).toEqual([[10, 1], [0]]);
    expect(data.pages[0].held).toBe(2);
  });

  it('does not store a live result the open history already holds (nor count it as live)', () => {
    const client = new QueryClient();
    const key = ['probes', 'history', 'pr_a'];
    client.setQueryData(key, { pages: [{ data: [event(1, 'pr_a')], next: null }], pageParams: [null] });
    appendRequest(client, { request_id: 'r', results: [event(1, 'pr_a')] });
    const data = client.getQueryData<{ pages: { data: ProbeEvent[]; live?: number }[] }>(key)!;
    expect(data.pages[0].data.map((e) => e.id)).toEqual([1]);
    expect(data.pages[0].live ?? 0).toBe(0);
  });
});
