/**
 * React Query hooks + live WS subscription for circuit edge sensing
 * (Feature 15). Mirrors useSensing.ts (Feature 11), including the deliberate
 * three-way export split.
 */

import { useEffect, useRef } from 'react';
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { circuitSensingApi } from '@/services/api';
import { socketClient } from '@/services/socket';
import type {
  CircuitSensingEvent,
  CircuitSensingEventList,
} from '@/types/circuitSensing';
import { useToast } from './useToast';

const EVENTS_KEY = ['circuitSensing', 'events'];
const STATUS_KEY = ['circuitSensing', 'status'];
const MAX_LIVE_EVENTS = 200;

export function useCircuitSensing(circuitId?: string) {
  const queryClient = useQueryClient();
  const toast = useToast();

  const statusQuery = useQuery({
    queryKey: STATUS_KEY,
    queryFn: () => circuitSensingApi.status(),
    refetchInterval: 15_000,
  });

  const eventsQuery = useQuery({
    queryKey: [...EVENTS_KEY, circuitId ?? 'all'],
    queryFn: () => circuitSensingApi.events({ circuitId, limit: 100 }),
  });

  // Status invalidation debounced via a real ref: one GET per burst, not one
  // per event (011 R3 / enh R2 #9 — the object-literal version only worked
  // because the effect deps never changed).
  const lastStatusInvalidate = useRef(0);

  // Live prepend: WS events land at the top of the list without a refetch.
  // Each cached list is updated against ITS OWN scope key — a hook scoped to
  // circuit X must not drop circuit Y's events from the 'all' cache (011 R1);
  // cleanup removes only THIS handler.
  useEffect(() => {
    const handler = (event: CircuitSensingEvent) => {
      const queries = queryClient.getQueriesData<CircuitSensingEventList>({
        queryKey: EVENTS_KEY,
      });
      for (const [key, data] of queries) {
        if (!data) continue;
        const scope = key[2];
        if (scope !== 'all' && event.circuit_id !== scope) continue;
        queryClient.setQueryData<CircuitSensingEventList>(key, {
          total: data.total + 1,
          events: [event, ...data.events].slice(0, MAX_LIVE_EVENTS),
        });
      }
      const now = Date.now();
      if (now - lastStatusInvalidate.current > 5000) {
        lastStatusInvalidate.current = now;
        queryClient.invalidateQueries({ queryKey: STATUS_KEY });
      }
    };
    socketClient.on('circuit:sensing:event', handler);
    return () => {
      socketClient.off('circuit:sensing:event', handler);
    };
  }, [queryClient]);

  const toggleMutation = useMutation({
    mutationFn: ({ id, enabled }: { id: string; enabled: boolean }) =>
      circuitSensingApi.setEnabled(id, enabled),
    onSuccess: (result) => {
      queryClient.invalidateQueries({ queryKey: STATUS_KEY });
      queryClient.invalidateQueries({ queryKey: ['circuits'] });
      // The server's own message explains a non-arm (inactive circuit, or no
      // sensable edge) — surface it rather than composing our own account.
      toast.success(
        `Edge sensing ${result.enabled ? 'enabled' : 'disabled'}` +
          (result.message ? ` — ${result.message}` : '')
      );
    },
    onError: (error: Error) =>
      toast.error(`Edge sensing toggle failed: ${error.message}`),
  });

  const clearMutation = useMutation({
    mutationFn: (id?: string) => circuitSensingApi.clearEvents(id),
    onSuccess: (result) => {
      queryClient.invalidateQueries({ queryKey: EVENTS_KEY });
      toast.success(
        `Cleared ${result.deleted} event${result.deleted === 1 ? '' : 's'}`
      );
    },
    onError: (error: Error) => toast.error(`Clear failed: ${error.message}`),
  });

  return {
    status: statusQuery.data,
    statusLoading: statusQuery.isLoading,
    events: eventsQuery.data?.events ?? [],
    totalEvents: eventsQuery.data?.total ?? 0,
    eventsLoading: eventsQuery.isLoading,
    setEnabled: (id: string, enabled: boolean) =>
      toggleMutation.mutate({ id, enabled }),
    isToggling: toggleMutation.isPending,
    clearEvents: (id?: string) => clearMutation.mutate(id),
  };
}

/**
 * Toggle-only variant: no WS subscription, no event queries. CircuitCard uses
 * this for the per-circuit toggle so mounting it alongside EdgeSensingPanel
 * doesn't double-subscribe (011 R1: every live event was prepended twice).
 */
export function useCircuitSensingToggle() {
  const queryClient = useQueryClient();
  const toast = useToast();

  const toggleMutation = useMutation({
    mutationFn: ({ id, enabled }: { id: string; enabled: boolean }) =>
      circuitSensingApi.setEnabled(id, enabled),
    onSuccess: (result) => {
      queryClient.invalidateQueries({ queryKey: STATUS_KEY });
      queryClient.invalidateQueries({ queryKey: ['circuits'] });
      toast.success(
        `Edge sensing ${result.enabled ? 'enabled' : 'disabled'}` +
          (result.message ? ` — ${result.message}` : '')
      );
    },
    onError: (error: Error) =>
      toast.error(`Edge sensing toggle failed: ${error.message}`),
  });

  return {
    setEnabled: (id: string, enabled: boolean) =>
      toggleMutation.mutate({ id, enabled }),
    isToggling: toggleMutation.isPending,
  };
}

export function useCircuitSensingEventDetail(eventId: number | null) {
  return useQuery({
    queryKey: ['circuitSensing', 'event', eventId],
    queryFn: () => circuitSensingApi.eventDetail(eventId as number),
    enabled: eventId != null,
  });
}
