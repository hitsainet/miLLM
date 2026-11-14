/**
 * React Query hooks + live WS subscription for cluster co-activation
 * sensing (Feature 11).
 */

import { useEffect, useRef } from 'react';
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { sensingApi } from '@/services/api';
import { socketClient } from '@/services/socket';
import type { SensingEvent, SensingEventList } from '@/types/sensing';
import { useToast } from './useToast';

const EVENTS_KEY = ['sensing', 'events'];
const STATUS_KEY = ['sensing', 'status'];
const MAX_LIVE_EVENTS = 200;

export function useSensing(profileId?: string) {
  const queryClient = useQueryClient();
  const toast = useToast();

  const statusQuery = useQuery({
    queryKey: STATUS_KEY,
    queryFn: () => sensingApi.status(),
    refetchInterval: 15_000,
  });

  const eventsQuery = useQuery({
    queryKey: [...EVENTS_KEY, profileId ?? 'all'],
    queryFn: () => sensingApi.events({ profileId, limit: 100 }),
  });

  // Status invalidation debounced: one GET per burst, not per event (011
  // R3). A real ref — the object-literal version only worked because the
  // effect deps never changed (enh R2 #9).
  const lastStatusInvalidate = useRef(0);

  // Live prepend: WS events land at the top of the list without a refetch.
  // Each cached list is updated against ITS OWN scope key — a hook scoped
  // to cluster X must not drop cluster Y's events from the 'all' cache
  // (011 R1); cleanup removes only THIS handler.
  useEffect(() => {
    const handler = (event: SensingEvent) => {
      const queries = queryClient.getQueriesData<SensingEventList>({
        queryKey: EVENTS_KEY,
      });
      for (const [key, data] of queries) {
        if (!data) continue;
        const scope = key[2];
        if (scope !== 'all' && event.profile_id !== scope) continue;
        queryClient.setQueryData<SensingEventList>(key, {
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
    socketClient.on('sensing:event', handler);
    return () => {
      socketClient.off('sensing:event', handler);
    };
  }, [queryClient]);

  const toggleMutation = useMutation({
    mutationFn: ({ id, enabled }: { id: string; enabled: boolean }) =>
      sensingApi.setEnabled(id, enabled),
    onSuccess: (result) => {
      queryClient.invalidateQueries({ queryKey: STATUS_KEY });
      queryClient.invalidateQueries({ queryKey: ['clusters'] });
      toast.success(
        `Sensing ${result.sensing_enabled ? 'enabled' : 'disabled'}` +
          (result.sensing_enabled && !result.armed
            ? ' (arms when this cluster is active with an SAE attached)'
            : '')
      );
    },
    onError: (error: Error) => toast.error(`Sensing toggle failed: ${error.message}`),
  });

  const configMutation = useMutation({
    mutationFn: ({ id, minK }: { id: string; minK: number | null }) =>
      sensingApi.setConfig(id, minK),
    onSuccess: (result) => {
      queryClient.invalidateQueries({ queryKey: STATUS_KEY });
      toast.success(
        result.min_k == null
          ? `Quorum reset to default (${result.effective_min_k ?? 'all'})`
          : `Quorum set to ${result.effective_min_k}`
      );
    },
    onError: (error: Error) => toast.error(`Quorum update failed: ${error.message}`),
  });

  const clearMutation = useMutation({
    mutationFn: (id?: string) => sensingApi.clearEvents(id),
    onSuccess: (result) => {
      queryClient.invalidateQueries({ queryKey: EVENTS_KEY });
      toast.success(`Cleared ${result.deleted} event${result.deleted === 1 ? '' : 's'}`);
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
    setMinK: (id: string, minK: number | null) =>
      configMutation.mutate({ id, minK }),
    isToggling: toggleMutation.isPending,
    clearEvents: (id?: string) => clearMutation.mutate(id),
  };
}

/**
 * Toggle-only variant: no WS subscription, no event queries. ClustersPage
 * uses this for the per-cluster toggle so mounting it alongside
 * SensingPanel doesn't double-subscribe (011 R1: every live event was
 * prepended twice).
 */
export function useSensingToggle() {
  const queryClient = useQueryClient();
  const toast = useToast();

  const toggleMutation = useMutation({
    mutationFn: ({ id, enabled }: { id: string; enabled: boolean }) =>
      sensingApi.setEnabled(id, enabled),
    onSuccess: (result) => {
      queryClient.invalidateQueries({ queryKey: STATUS_KEY });
      queryClient.invalidateQueries({ queryKey: ['clusters'] });
      toast.success(
        `Sensing ${result.sensing_enabled ? 'enabled' : 'disabled'}` +
          (result.sensing_enabled && !result.armed
            ? ' (arms when this cluster is active with an SAE attached)'
            : '')
      );
    },
    onError: (error: Error) => toast.error(`Sensing toggle failed: ${error.message}`),
  });

  return {
    setEnabled: (id: string, enabled: boolean) =>
      toggleMutation.mutate({ id, enabled }),
    isToggling: toggleMutation.isPending,
  };
}

export function useSensingEventDetail(eventId: number | null) {
  return useQuery({
    queryKey: ['sensing', 'event', eventId],
    queryFn: () => sensingApi.eventDetail(eventId as number),
    enabled: eventId != null,
  });
}
