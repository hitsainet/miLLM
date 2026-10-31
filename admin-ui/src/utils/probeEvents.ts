import type { ProbeEvent } from '@/types/probe';

/** Group in arrival order, preserving it. A Map keeps insertion order, so the newest request
 *  stays first without sorting by a timestamp the socket payload once did not carry. */
export function groupByRequest(events: ProbeEvent[]): [string, ProbeEvent[]][] {
  const groups = new Map<string, ProbeEvent[]>();
  for (const event of events) {
    const key = event.request_id ?? '—';
    const existing = groups.get(key);
    if (existing) existing.push(event);
    else groups.set(key, [event]);
  }
  return [...groups.entries()];
}
