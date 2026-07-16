/**
 * React Query hooks + live WS subscription for cluster co-activation
 * sensing (Feature 11).
 */

import { useEffect } from 'react';
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

  // Live prepend: WS events land at the top of the list without a refetch.
  useEffect(() => {
    const handler = (event: SensingEvent) => {
      queryClient.setQueriesData<SensingEventList>(
        { queryKey: EVENTS_KEY },
        (current) => {
          if (!current) return current;
          if (profileId && event.profile_id !== profileId) return current;
          return {
            total: current.total + 1,
            events: [event, ...current.events].slice(0, MAX_LIVE_EVENTS),
          };
        }
      );
      queryClient.invalidateQueries({ queryKey: STATUS_KEY });
    };
    socketClient.on('sensing:event', handler);
    return () => {
      socketClient.off('sensing:event');
    };
  }, [queryClient, profileId]);

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
    isToggling: toggleMutation.isPending,
    clearEvents: (id?: string) => clearMutation.mutate(id),
  };
}

export function useSensingEventDetail(eventId: number | null) {
  return useQuery({
    queryKey: ['sensing', 'event', eventId],
    queryFn: () => sensingApi.eventDetail(eventId as number),
    enabled: eventId != null,
  });
}
