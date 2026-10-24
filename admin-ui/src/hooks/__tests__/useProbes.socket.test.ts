/**
 * The live feed appends a WHOLE request (Feature 39, TD39-28; FTASKS 7.4).
 *
 * `probe:results` carries every result of one request; the reducer puts them at the head of each
 * cached list, together and in the server's order, and a list scoped to one probe keeps only that
 * probe's results.
 */

import { describe, expect, it } from 'vitest';
import { QueryClient } from '@tanstack/react-query';
import { appendRequest } from '../useProbes';
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
});
