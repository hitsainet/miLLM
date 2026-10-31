/**
 * What a probe tile says about its results (operator 2026-10-29: the count had gone from the tile
 * until it was opened).
 */

import type { Probe, ProbeEvent } from '@/types/probe';

/** What a tile says about a probe's results: how many, how many fired, when the newest came. */
export interface ProbeRecord {
  total: number;
  fired: number;
  lastAt: string | null;
  /** True when `total` is every stored result; false for "the last N" of an older server. */
  complete: boolean;
}

/**
 * ONE source for the tile header and the results toggle (operator 2026-10-29: the count had gone
 * from the tile until it was opened).
 *
 * 1. The server's totals (`results_total`, contract v1.24), on a closed tile and an open one
 *    alike, kept current between fetches by the live feed (`bumpTotals`). The open panel does
 *    not count for itself; it says "Showing N of Y" against these.
 * 2. Otherwise (an older server, no totals) the panel's loaded list: exact when it holds every
 *    result, else labelled "the last N".
 * 3. Otherwise nothing: no count before there is one to give.
 *
 * ⚠ Never the shared newest-results list. That is where the old, wrong count came from — with
 * nine probes armed it held about 22 chats.
 */
export function probeRecord(
  probe: Probe,
  history: { events: ProbeEvent[]; loaded: boolean; hasOlder: boolean; held: number },
): ProbeRecord | null {
  const fromList = () => ({
    total: history.events.length,
    fired: history.events.filter((e) => e.verdict === true).length,
    lastAt: history.events[0]?.created_at ?? null,
  });
  if (typeof probe.results_total === 'number') {
    return {
      total: probe.results_total,
      fired: probe.results_fired ?? 0,
      lastAt: probe.last_result_at ?? null,
      complete: true,
    };
  }
  if (history.loaded) {
    return { ...fromList(), complete: !history.hasOlder && history.held === 0 };
  }
  return null;
}
