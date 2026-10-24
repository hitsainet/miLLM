/**
 * Polls the attachable-weights status for the Dashboard "Weights" card (Feature 31 FTID §6,
 * Feature 33 status block).
 *
 * Two endpoints, polled every 5 s: `GET /api/weights/status` (031) and `GET /api/weights/adapter`
 * (033, D-28). Either may be absent on a server that predates the feature, and a 404 there is
 * not a failure — it means the capability is not deployed. Such a read resolves to
 * `{ available: false }` rather than an error, so the card says "not available" instead of
 * alarming red, and the poll keeps running so the card fills in once the route ships.
 */

import { useQuery } from '@tanstack/react-query';
import { ApiError, weightsApi } from '@/services/api';
import type { AdapterStatus, WeightsStatus } from '@/types';

export const WEIGHTS_POLL_MS = 5000;

/** A read that distinguishes "the route is not served here" from a real answer. */
export type Availability<T> = { available: true; data: T } | { available: false };

/** Resolve a 404 to `{ available: false }`; every other failure stays an error. */
export async function readOptional<T>(fetcher: () => Promise<T>): Promise<Availability<T>> {
  try {
    return { available: true, data: await fetcher() };
  } catch (error) {
    if (error instanceof ApiError && error.status === 404) return { available: false };
    throw error;
  }
}

export interface WeightsStatusView {
  status: Availability<WeightsStatus> | undefined;
  adapter: Availability<AdapterStatus> | undefined;
  statusLoading: boolean;
  adapterLoading: boolean;
  statusError: Error | null;
  adapterError: Error | null;
}

export function useWeightsStatus(): WeightsStatusView {
  const statusQuery = useQuery({
    queryKey: ['weights', 'status'],
    queryFn: () => readOptional(weightsApi.status),
    refetchInterval: WEIGHTS_POLL_MS,
  });

  const adapterQuery = useQuery({
    queryKey: ['weights', 'adapter'],
    queryFn: () => readOptional(weightsApi.adapter),
    refetchInterval: WEIGHTS_POLL_MS,
  });

  return {
    status: statusQuery.data,
    adapter: adapterQuery.data,
    statusLoading: statusQuery.isLoading,
    adapterLoading: adapterQuery.isLoading,
    statusError: statusQuery.error,
    adapterError: adapterQuery.error,
  };
}
