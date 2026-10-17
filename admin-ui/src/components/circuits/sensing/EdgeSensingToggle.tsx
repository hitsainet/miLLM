/**
 * EdgeSensingToggle (Feature 15) — per-circuit edge sensing intent, shown on
 * the circuit card.
 *
 * Uses the toggle-ONLY hook: mounting this alongside EdgeSensingPanel must not
 * double-subscribe the socket (011 R1 — every live event was prepended twice).
 *
 * The circuit list carries no `sensing_enabled` field, so the enabled state is
 * read from the sensing status endpoint's `enabled_circuits` (persistent
 * operator intent) rather than being tracked locally — a local guess would
 * disagree with the server after any other client toggled it.
 */

import { useQuery } from '@tanstack/react-query';
import { Eye, EyeOff } from 'lucide-react';

import { circuitSensingApi } from '@/services/api';
import { useCircuitSensingToggle } from '@hooks/useCircuitSensing';

interface EdgeSensingToggleProps {
  circuitId: string;
}

export function EdgeSensingToggle({ circuitId }: EdgeSensingToggleProps) {
  const { setEnabled, isToggling } = useCircuitSensingToggle();

  // Shares the status cache key with useCircuitSensing, so a toggle anywhere
  // refreshes every card at once.
  const statusQuery = useQuery({
    queryKey: ['circuitSensing', 'status'],
    queryFn: () => circuitSensingApi.status(),
    refetchInterval: 15_000,
  });

  const enabled = Boolean(
    statusQuery.data?.enabled_circuits?.some((c) => c.id === circuitId)
  );
  const armed = Boolean(
    statusQuery.data?.armed && statusQuery.data.circuit_id === circuitId
  );

  return (
    <button
      type="button"
      data-testid="edge-sensing-toggle"
      disabled={isToggling}
      onClick={() => setEnabled(circuitId, !enabled)}
      aria-pressed={enabled}
      aria-label={`${enabled ? 'Disable' : 'Enable'} edge sensing`}
      title={
        enabled
          ? armed
            ? 'Edge sensing is armed and observing this circuit'
            : 'Edge sensing enabled — arms when this circuit is active with its SAEs attached'
          : 'Observe this circuit’s edges firing during generation'
      }
      className={`inline-flex items-center gap-1 rounded px-2 py-1 text-xs transition-colors disabled:opacity-50 ${
        enabled
          ? 'bg-cyan-500/15 text-cyan-300 hover:bg-cyan-500/25'
          : 'text-slate-400 hover:bg-slate-700/50 hover:text-slate-200'
      }`}
    >
      {enabled ? <Eye className="h-3.5 w-3.5" /> : <EyeOff className="h-3.5 w-3.5" />}
      Sensing
      {enabled && armed && (
        <span className="ml-0.5 h-1.5 w-1.5 rounded-full bg-emerald-400" />
      )}
    </button>
  );
}
