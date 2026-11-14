/**
 * The old addresses this server still answers on, from the server (040 FR-040.31; TID §6).
 *
 * Reads `GET /api/health/detailed` once per page load and returns only `legacy_names_meta`'s
 * `transition_hosts` (the old hosts), `new_host_for` (each one's new host) and `retire_on` (the
 * date they stop answering; null until it is set). The admin UI keeps no copy of the list: the
 * server builds all three from one table (`miserver/core/legacy_names.py`).
 *
 * The route answers its model directly (no `{success, data}` envelope), so this reads it with
 * `fetch`, as `useTelemetryHealth` does. A server without the fields, or a failed read, gives an
 * empty list: the notice is then not drawn.
 */

import { useQuery } from '@tanstack/react-query';

export interface LegacyHosts {
  hosts: string[];
  retireOn: string | null;
  newHostFor: (host: string) => string | null;
}

interface LegacyNamesMeta {
  transition_hosts?: string[] | null;
  new_host_for?: Record<string, string> | null;
  retire_on?: string | null;
}

const NONE: LegacyNamesMeta = {};

export async function readLegacyNamesMeta(): Promise<LegacyNamesMeta> {
  const response = await fetch('/api/health/detailed');
  if (!response.ok) throw new Error(`health/detailed answered ${response.status}`);
  const body = (await response.json()) as { legacy_names_meta?: LegacyNamesMeta | null };
  return body.legacy_names_meta ?? NONE;
}

export function useLegacyHost(): LegacyHosts {
  const query = useQuery({
    queryKey: ['health', 'legacy-hosts'],
    queryFn: readLegacyNamesMeta,
    staleTime: Infinity,
    retry: false,
  });
  const meta = query.data ?? NONE;
  const pairs = meta.new_host_for ?? {};
  return {
    hosts: meta.transition_hosts ?? [],
    retireOn: meta.retire_on ?? null,
    newHostFor: (host) => (Object.prototype.hasOwnProperty.call(pairs, host) ? pairs[host] : null),
  };
}
