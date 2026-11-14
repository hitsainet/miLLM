/**
 * Polls `GET /api/health/detailed` every 15 s and returns only its `telemetry` block, for the
 * Dashboard "Telemetry" card (038 FR-38.13, FTASKS 9.3; one new browser request to an existing
 * route — OI-2, confirmed by the operator in 0.2).
 *
 * The detailed health route answers its model directly (no `{success, data}` envelope), so this
 * hook reads it with `fetch`, not the envelope-unwrapping `request`. A server that predates 038
 * answers without the block: that is `{ available: false }` ("not available"), not an error.
 */

import { useQuery } from '@tanstack/react-query';
import type { TelemetryHealth } from '@/types';
import type { Availability } from './useWeightsStatus';

export const TELEMETRY_POLL_MS = 15000;

export async function readTelemetryHealth(): Promise<Availability<TelemetryHealth>> {
  const response = await fetch('/api/health/detailed');
  if (!response.ok) throw new Error(`health/detailed answered ${response.status}`);
  const body = (await response.json()) as { telemetry?: TelemetryHealth | null };
  return body.telemetry ? { available: true, data: body.telemetry } : { available: false };
}

export interface TelemetryHealthView {
  health: Availability<TelemetryHealth> | undefined;
  isLoading: boolean;
  error: Error | null;
}

export function useTelemetryHealth(): TelemetryHealthView {
  const query = useQuery({
    queryKey: ['health', 'telemetry'],
    queryFn: readTelemetryHealth,
    refetchInterval: TELEMETRY_POLL_MS,
  });
  return { health: query.data, isLoading: query.isLoading, error: query.error };
}
