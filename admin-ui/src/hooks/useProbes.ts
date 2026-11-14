/**
 * React Query hooks + live socket subscription for probe monitors (Feature 24).
 *
 * Follows `useSensing`, including the two things that file learned the hard way:
 *
 * * **a real `useRef` for the debounce**, not an object literal — the literal version only
 *   worked because the effect deps never changed;
 * * **cleanup removes only THIS handler**, because `socketClient` subscriptions are shared and
 *   an unscoped `off` silences every other component listening to the same event.
 */

import { useEffect, useMemo, useRef, useState } from 'react';
import {
  type InfiniteData,
  useInfiniteQuery,
  useMutation,
  useQuery,
  useQueryClient,
} from '@tanstack/react-query';
import { ApiError, probesApi } from '@/services/api';
import { socketClient } from '@/services/socket';
import type { Page, Probe, ProbeEvent, ProbeResultsMessage } from '@/types/probe';
import { useToast } from './useToast';

const PROBES_KEY = ['probes'];
const EVENTS_KEY = ['probes', 'events'];
const STATUS_KEY = ['probes', 'status'];
/** How often the probe list (and its result totals) re-syncs from the server. */
export const PROBES_RESYNC_MS = 60_000;
const MAX_LIVE_EVENTS = 200;
/** One probe's own history (2026-10-29), paged from the server by cursor. */
export const HISTORY_KEY = ['probes', 'history'];
/** How many results one "page" of a probe's history holds. */
export const HISTORY_PAGE_SIZE = 100;
/**
 * How many LIVE results one probe's open list takes in before it stops adding them.
 *
 * ⚠ Past this the list does NOT drop its oldest rows (that was the defect: a list that quietly
 * forgot history). It stops adding, counts what it held back, and the panel says so and offers a
 * reload, which fetches the newest page from the server.
 */
export const MAX_LIVE_PER_PROBE = 1000;

/** A page of a probe's history. The first page also counts what the live feed added and held. */
export interface ProbeHistoryPage extends Page<ProbeEvent> {
  /** Live results put at the head of this list since it was fetched. */
  live?: number;
  /** Live results NOT added because `live` reached `MAX_LIVE_PER_PROBE`. */
  held?: number;
}
type ProbeHistory = InfiniteData<ProbeHistoryPage, string | null>;

/**
 * What the server's arm refusals mean, in the operator's words.
 *
 * ⚠ Keyed on the server's codes, NOT re-derived. The gate a probe failed is the single most useful
 * thing about a refusal: "wrong model" and "needs your acknowledgement" call for completely
 * different actions, and a generic "arm failed" tells an operator to retry something that will
 * never work.
 */
const ARM_REFUSALS: Record<string, string> = {
  PROBE_MODEL_MISMATCH: 'This probe was fitted on a different model',
  PROBE_PARITY_FAILED: 'This build does not reproduce the recorded scores',
  PROBE_LIMIT: 'Too many probes armed',
  PROBE_BUDGET: 'Arming this probe would go over the probe cost budget',
  PROBE_SAE_MISSING: "This probe's SAE is not downloaded here",
  PROBE_SAE_MISMATCH: 'The downloaded SAE is not the one this probe was fitted against',
  PROBE_NO_MODEL_LOADED: 'No model is loaded',
  PROBE_HOOK_UNSUPPORTED: 'This runtime exposes no layer to hook',
};

/** `details` on an `UNVALIDATED_PROBE` refusal — the server's own words for the rung. */
export interface ProbeAckDetails {
  rung?: number;
  rung_language?: string;
  next_step?: string;
}

/**
 * Append one request's results to every cached event list, each against ITS OWN scope key (a
 * hook scoped to probe X must not drop probe Y's events from the 'all' cache). The request's
 * results stay together and in the order the server sent them; the list is newest request first.
 */
export function appendRequest(
  queryClient: ReturnType<typeof useQueryClient>,
  message: ProbeResultsMessage,
): void {
  const results = message.results ?? [];
  const queries = queryClient.getQueriesData<ProbeEvent[]>({ queryKey: EVENTS_KEY });
  for (const [key, data] of queries) {
    if (!data) continue;
    const scope = key[2];
    const mine = scope === 'all' ? results : results.filter((r) => r.probe_id === scope);
    if (mine.length === 0) continue;
    queryClient.setQueryData<ProbeEvent[]>(key, [...mine, ...data].slice(0, MAX_LIVE_EVENTS));
  }

  bumpTotals(queryClient, results);

  // Each open probe history takes THAT probe's results only, at its head, skipping any it already
  // has. Nothing is trimmed from the tail: another probe's traffic can never push it out.
  const histories = queryClient.getQueriesData<ProbeHistory>({ queryKey: HISTORY_KEY });
  for (const [key, data] of histories) {
    if (!data || data.pages.length === 0) continue;
    const probeId = key[2];
    const mine = results.filter((r) => r.kind !== 'rule' && r.probe_id === probeId);
    if (mine.length === 0) continue;
    const seen = new Set(data.pages.flatMap((page) => page.data.map((e) => e.id)));
    const fresh = mine.filter((r) => !seen.has(r.id));
    if (fresh.length === 0) continue;
    const [first, ...older] = data.pages;
    const live = first.live ?? 0;
    const head: ProbeHistoryPage =
      live >= MAX_LIVE_PER_PROBE
        ? { ...first, held: (first.held ?? 0) + fresh.length }
        : { ...first, data: [...fresh, ...first.data], live: live + fresh.length };
    queryClient.setQueryData<ProbeHistory>(key, { ...data, pages: [head, ...older] });
  }
}

/**
 * Keeps each tile's "Fired X of Y" current between list fetches (contract v1.24, 2026-10-29).
 *
 * The list's `results_total` / `results_fired` / `last_result_at` were counted when it was
 * fetched; each live result recorded for a probe since then adds to that probe's numbers, with no
 * fetch, on a closed tile as on an open one. A result OLDER than the probe's `last_result_at` is
 * already in the count (the list was fetched after it was stored) and is skipped; one at the same
 * instant is counted, since two requests can share one. A rare double count there is corrected by
 * the list's re-sync (`PROBES_RESYNC_MS`, and on return to the tab). Rule results name no probe and are not counted; a result the
 * server says it did not store is not in the stored total either. A probe without totals (an
 * older server) is left alone.
 */
function bumpTotals(queryClient: ReturnType<typeof useQueryClient>, results: ProbeEvent[]): void {
  const list = queryClient.getQueryData<Probe[]>(PROBES_KEY);
  if (!list) return;
  let changed = false;
  const next = list.map((probe) => {
    if (typeof probe.results_total !== 'number') return probe;
    const since = probe.last_result_at ? Date.parse(probe.last_result_at) : Number.NEGATIVE_INFINITY;
    const mine = results.filter((r) => {
      if (r.kind === 'rule' || r.probe_id !== probe.id) return false;
      if (r.persistence === 'not_persisted') return false;
      const at = Date.parse(r.created_at);
      // Strictly older only: two requests can share an instant, and each is a result.
      return Number.isNaN(at) || Number.isNaN(since) || at >= since;
    });
    if (mine.length === 0) return probe;
    changed = true;
    const newest = mine
      .map((r) => r.created_at)
      .reduce((a, b) => (Date.parse(b) > Date.parse(a) ? b : a), probe.last_result_at ?? mine[0].created_at);
    return {
      ...probe,
      results_total: probe.results_total + mine.length,
      results_fired: (probe.results_fired ?? 0) + mine.filter((r) => r.verdict === true).length,
      last_result_at: newest,
    };
  });
  if (changed) queryClient.setQueryData<Probe[]>(PROBES_KEY, next);
}

/**
 * ONE probe's results, fetched for that probe alone, newest first, older pages on request.
 *
 * Operator, 2026-10-29: "why does it reach a max of 22 verdicts and seemingly overwrite the oldest
 * as each new one comes in". Each probe's panel was a slice of one shared list of the newest
 * results across EVERY probe; with nine armed, each chat added nine and pushed the oldest chat
 * out of every panel. This list is fetched with `probe_id`, grows only with this probe's live
 * results (see `appendRequest`), and reaches back with the server's `meta.next` cursor.
 *
 * Only fetched while `enabled` (the panel is open); the cache keeps it after the panel closes.
 */
export function useProbeHistory(probeId: string, enabled: boolean) {
  const queryClient = useQueryClient();
  const query = useInfiniteQuery({
    queryKey: [...HISTORY_KEY, probeId],
    queryFn: ({ pageParam }): Promise<ProbeHistoryPage> =>
      probesApi.eventsPage({ probeId, limit: HISTORY_PAGE_SIZE, cursor: pageParam }),
    initialPageParam: null as string | null,
    getNextPageParam: (last: ProbeHistoryPage) => last.next,
    enabled,
  });
  const pages = query.data?.pages;
  const events = useMemo(() => {
    const seen = new Set<number>();
    const out: ProbeEvent[] = [];
    for (const page of pages ?? []) {
      for (const event of page.data) {
        if (seen.has(event.id)) continue;
        seen.add(event.id);
        out.push(event);
      }
    }
    return out;
  }, [pages]);
  return {
    events,
    /** False until the first page has arrived: a count before then would be a guess. */
    loaded: pages !== undefined,
    loading: query.isLoading,
    failed: query.isError,
    hasOlder: query.hasNextPage,
    loadingOlder: query.isFetchingNextPage,
    loadOlder: () => {
      void query.fetchNextPage();
    },
    /** Live results not added because the live cap was reached. */
    held: pages?.[0]?.held ?? 0,
    /** Start again from the newest page (what the "held back" notice offers). */
    reload: () => {
      void queryClient.resetQueries({ queryKey: [...HISTORY_KEY, probeId] });
    },
  };
}

export function useProbes(probeId?: string) {
  const queryClient = useQueryClient();
  const toast = useToast();

  const probesQuery = useQuery({
    queryKey: PROBES_KEY,
    queryFn: () => probesApi.list(),
    // The tiles' result totals move with live traffic (`bumpTotals`); these re-sync them from
    // the server — on return to the tab, and once a minute — so a missed socket message cannot
    // leave a count wrong for long (2026-10-29). The app's default turns focus refetch off.
    refetchOnWindowFocus: true,
    refetchInterval: PROBES_RESYNC_MS,
  });

  const statusQuery = useQuery({
    queryKey: STATUS_KEY,
    queryFn: () => probesApi.status(),
    refetchInterval: 15_000,
  });

  const eventsQuery = useQuery({
    queryKey: [...EVENTS_KEY, probeId ?? 'all'],
    queryFn: () => probesApi.events({ probeId, limit: 100 }),
  });

  // One status GET per burst, not one per event. A real ref: an object literal here would be
  // re-created every render and the debounce would never fire.
  const lastStatusInvalidate = useRef(0);

  useEffect(() => {
    // Feature 39: ONE `probe:results` message per request, every result in contract order.
    const handler = (message: ProbeResultsMessage) => {
      appendRequest(queryClient, message);
      const now = Date.now();
      if (now - lastStatusInvalidate.current > 5000) {
        lastStatusInvalidate.current = now;
        queryClient.invalidateQueries({ queryKey: STATUS_KEY });
      }
    };
    socketClient.on('probe:results', handler);
    return () => {
      // Only this handler — an unscoped off() silences every other subscriber.
      socketClient.off('probe:results', handler);
    };
  }, [queryClient]);

  const importMutation = useMutation({
    mutationFn: ({
      definition,
      onConflict,
    }: {
      definition: Record<string, unknown>;
      onConflict?: 'rename' | 'fail';
    }) => probesApi.import(definition, onConflict ?? 'rename'),
    onSuccess: (probe: Probe) => {
      queryClient.invalidateQueries({ queryKey: PROBES_KEY });
      toast.success(`Imported "${probe.name}" — ${probe.rung_language}`);
    },
    onError: (error: Error) => toast.error(`Import failed: ${error.message}`),
  });

  /**
   * ⚠ ARMING IS THE ONLY PATH THAT RUNS THE FOUR GATES, and a refusal is the interesting outcome.
   *
   * `UNVALIDATED_PROBE` is not a failure — it is the server asking the operator to say, explicitly,
   * that they are choosing to monitor live traffic with a probe below rung 2. It is surfaced to the
   * caller through `pendingAck` rather than a toast, because a toast is dismissed and the decision
   * is not made. Every other code is a real refusal and names its gate.
   */
  // ⚠ `windows` IS CARRIED THROUGH THE REFUSAL. The acknowledge dialog re-arms, and without this
  // the operator's window choice is silently discarded on exactly the probes that needed a
  // deliberate decision to arm at all.
  const [pendingAck, setPendingAck] = useState<{
    id: string;
    details: ProbeAckDetails;
    windows?: string[];
  } | null>(
    null
  );

  const armMutation = useMutation({
    mutationFn: ({
      id,
      acknowledgeBelowRung2,
      reason,
      windows,
    }: {
      id: string;
      acknowledgeBelowRung2?: boolean;
      reason?: string;
      /** Which slices to report. Omitted means the server's default (all three). */
      windows?: string[];
    }) => probesApi.arm(id, { acknowledgeBelowRung2, reason, windows }),
    onSuccess: (probe: Probe) => {
      setPendingAck(null);
      queryClient.invalidateQueries({ queryKey: PROBES_KEY });
      queryClient.invalidateQueries({ queryKey: STATUS_KEY });
      toast.success(`Armed "${probe.name}" — ${probe.rung_language}`);
    },
    onError: (error: Error, variables) => {
      if (error instanceof ApiError && error.code === 'UNVALIDATED_PROBE') {
        // Not an error to dismiss — a decision to put in front of the operator.
        setPendingAck({
          id: variables.id,
          details: (error.details ?? {}) as ProbeAckDetails,
          windows: armMutation.variables?.windows,
        });
        return;
      }
      // Every other refusal names its gate, so say which one rather than "arm failed".
      //
      // ⚠ AND WHICH PROBE. With several imported, a toast naming only the gate leaves the
      // operator unable to tell which row it came from — which is how one failing probe read as
      // all of them failing.
      const gate =
        error instanceof ApiError ? (ARM_REFUSALS[error.code] ?? error.code) : 'Arming';
      const which =
        queryClient
          .getQueryData<Probe[]>(PROBES_KEY)
          ?.find((p) => p.id === variables.id)?.name ?? variables.id;
      toast.error(`${which} — ${gate}: ${error.message}`);
    },
  });

  const parityMutation = useMutation({
    mutationFn: (id: string) => probesApi.checkParity(id),
    onSuccess: (report) => {
      queryClient.invalidateQueries({ queryKey: PROBES_KEY });
      if (report.passed) {
        toast.success('Parity passed — this build reproduces the recorded scores');
      } else {
        // ⚠ `max_abs_diff` may be null: no vector could be compared at all, which is not "0 off".
        toast.error(
          report.max_abs_diff === null
            ? `Parity could not be checked: ${report.error ?? 'no comparable vectors'}`
            : `Parity FAILED — max Δ ${report.max_abs_diff.toExponential(2)} of ${report.tolerance}`
        );
      }
    },
    onError: (error: Error) => toast.error(`Parity check failed: ${error.message}`),
  });

  const disarmMutation = useMutation({
    mutationFn: (id: string) => probesApi.disarm(id),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: PROBES_KEY });
      queryClient.invalidateQueries({ queryKey: STATUS_KEY });
      toast.success('Probe disarmed');
    },
    onError: (error: Error) => toast.error(`Disarm failed: ${error.message}`),
  });

  const deleteMutation = useMutation({
    mutationFn: (id: string) => probesApi.remove(id),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: PROBES_KEY });
      toast.success('Probe deleted');
    },
    // The server refuses while armed; surface its reason rather than a generic failure.
    onError: (error: Error) => toast.error(`Delete failed: ${error.message}`),
  });

  const clearEventsMutation = useMutation({
    mutationFn: (id?: string) => probesApi.clearEvents(id),
    onSuccess: (result) => {
      queryClient.invalidateQueries({ queryKey: EVENTS_KEY });
      queryClient.invalidateQueries({ queryKey: HISTORY_KEY });
      // The tiles' totals: the list alone (`exact`: its key is a prefix of every other here).
      queryClient.invalidateQueries({ queryKey: PROBES_KEY, exact: true });
      toast.success(`Cleared ${result.removed} event${result.removed === 1 ? '' : 's'}`);
    },
    onError: (error: Error) => toast.error(`Clear failed: ${error.message}`),
  });

  return {
    probes: probesQuery.data ?? [],
    probesLoading: probesQuery.isLoading,
    /** True once the list has arrived. An errored or pending list is not "no probes exist". */
    probesLoaded: probesQuery.isSuccess,
    status: statusQuery.data,
    statusLoading: statusQuery.isLoading,
    events: eventsQuery.data ?? [],
    eventsLoading: eventsQuery.isLoading,
    importProbe: importMutation.mutate,
    importing: importMutation.isPending,
    arm: armMutation.mutate,
    arming: armMutation.isPending,
    /**
     * ⚠ WHICH probe is arming, not merely THAT one is.
     *
     * `isPending` is one flag for the whole mutation. The page passed it to every row, so
     * clicking Arm on one probe made EVERY row read "Arming…" and disabled every button — it
     * looked like all of them were being armed at once, and then all failing together when the
     * single real request errored. Reported by the operator on 2026-09-28.
     *
     * `variables` carries the arguments of the request in flight, so the id is already here.
     */
    armingId: armMutation.isPending ? (armMutation.variables?.id ?? null) : null,
    /** Set when the server asked for an acknowledgement. `null` once given or cancelled. */
    pendingAck,
    dismissAck: () => setPendingAck(null),
    checkParity: parityMutation.mutate,
    checkingParity: parityMutation.isPending,
    /** Same, for the parity check — it shares the row's disabled state. */
    checkingParityId: parityMutation.isPending ? (parityMutation.variables ?? null) : null,
    disarm: disarmMutation.mutate,
    remove: deleteMutation.mutate,
    clearEvents: clearEventsMutation.mutate,
  };
}

/**
 * One event WITH its decoded context window.
 *
 * ⚠ A separate, explicit fetch. The list and the socket carry no context text, so a reviewer
 * asks for one event rather than a dashboard receiving a feed of prompts.
 */
export function useProbeEventDetail(eventId: number | null) {
  return useQuery({
    queryKey: ['probes', 'event', eventId],
    queryFn: () => probesApi.eventDetail(eventId as number),
    enabled: eventId !== null,
  });
}
