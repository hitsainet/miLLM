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

import { useEffect, useRef, useState } from 'react';
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { ApiError, probesApi } from '@/services/api';
import { socketClient } from '@/services/socket';
import type { Probe, ProbeEvent } from '@/types/probe';
import { useToast } from './useToast';

const PROBES_KEY = ['probes'];
const EVENTS_KEY = ['probes', 'events'];
const STATUS_KEY = ['probes', 'status'];
const MAX_LIVE_EVENTS = 200;

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

export function useProbes(probeId?: string) {
  const queryClient = useQueryClient();
  const toast = useToast();

  const probesQuery = useQuery({
    queryKey: PROBES_KEY,
    queryFn: () => probesApi.list(),
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
    const handler = (event: ProbeEvent) => {
      // Update each cached list against ITS OWN scope key. A hook scoped to probe X must not
      // drop probe Y's events from the 'all' cache.
      const queries = queryClient.getQueriesData<ProbeEvent[]>({ queryKey: EVENTS_KEY });
      for (const [key, data] of queries) {
        if (!data) continue;
        const scope = key[2];
        if (scope !== 'all' && event.probe_id !== scope) continue;
        queryClient.setQueryData<ProbeEvent[]>(key, [event, ...data].slice(0, MAX_LIVE_EVENTS));
      }
      const now = Date.now();
      if (now - lastStatusInvalidate.current > 5000) {
        lastStatusInvalidate.current = now;
        queryClient.invalidateQueries({ queryKey: STATUS_KEY });
      }
    };
    socketClient.on('probe:event', handler);
    return () => {
      // Only this handler — an unscoped off() silences every other subscriber.
      socketClient.off('probe:event', handler);
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
  const [pendingAck, setPendingAck] = useState<{ id: string; details: ProbeAckDetails } | null>(
    null
  );

  const armMutation = useMutation({
    mutationFn: ({
      id,
      acknowledgeBelowRung2,
      reason,
    }: {
      id: string;
      acknowledgeBelowRung2?: boolean;
      reason?: string;
    }) => probesApi.arm(id, { acknowledgeBelowRung2, reason }),
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
      toast.success(`Cleared ${result.removed} event${result.removed === 1 ? '' : 's'}`);
    },
    onError: (error: Error) => toast.error(`Clear failed: ${error.message}`),
  });

  return {
    probes: probesQuery.data ?? [],
    probesLoading: probesQuery.isLoading,
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
