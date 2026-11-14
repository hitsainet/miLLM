/**
 * Probe sessions (Feature 36): the list, one session's timeline, and live refresh.
 *
 * ⚠ The transcript is NOT a query here. It is user text, fetched only when the operator clicks
 * "Show text" (`SessionsPage`), so no render, refetch or socket event ever pulls it.
 *
 * Live updates follow `useProbes`: `probe:session` and `probe:results` (Feature 39: one message
 * per request) invalidate the queries
 * (throttled by React Query's own de-duplication); a refetch reconciles anything the socket drops.
 * The cleanup removes only THIS handler — socket subscriptions are shared.
 */

import { useEffect } from 'react';
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { probesApi } from '@/services/api';
import { socketClient } from '@/services/socket';
import type { SessionListFilters } from '@/types/probeSession';

const SESSIONS_KEY = ['probes', 'sessions'];

export function useProbeSessions(filters: SessionListFilters) {
  const queryClient = useQueryClient();
  const list = useQuery({
    queryKey: [...SESSIONS_KEY, filters],
    queryFn: () => probesApi.sessions(filters),
  });

  useEffect(() => {
    const handler = () => {
      void queryClient.invalidateQueries({ queryKey: SESSIONS_KEY });
    };
    socketClient.on('probe:session', handler);
    socketClient.on('probe:results', handler);
    return () => {
      socketClient.off('probe:session', handler);
      socketClient.off('probe:results', handler);
    };
  }, [queryClient]);

  return list;
}

export function useProbeSession(id: string | null) {
  return useQuery({
    queryKey: [...SESSIONS_KEY, 'detail', id],
    queryFn: () => probesApi.session(id as string),
    enabled: Boolean(id),
  });
}

export function useDeleteProbeSession() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: ({ id, events }: { id: string; events: 'keep' | 'delete' }) =>
      probesApi.deleteSession(id, events),
    onSuccess: () => {
      void queryClient.invalidateQueries({ queryKey: SESSIONS_KEY });
      void queryClient.invalidateQueries({ queryKey: ['probes', 'events'] });
    },
  });
}

export function useRecordingStatus() {
  return useQuery({
    queryKey: ['probes', 'status'],
    queryFn: () => probesApi.status(),
  });
}
