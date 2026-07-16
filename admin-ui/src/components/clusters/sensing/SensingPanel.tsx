/**
 * SensingPanel (Feature 11) — status strip (armed cluster, threshold mode,
 * overhead) + newest-first event list with WS live prepend.
 */

import { useState } from 'react';
import { Activity, Trash2 } from 'lucide-react';
import { Badge, Button } from '@components/common';
import { useSensing } from '@hooks/useSensing';
import type { SensingEvent } from '@/types/sensing';
import { SensingEventDetail } from './SensingEventDetail';

export function SensingPanel() {
  const { status, events, totalEvents, eventsLoading, clearEvents } = useSensing();
  const [selected, setSelected] = useState<SensingEvent | null>(null);

  return (
    <section className="space-y-3" data-testid="sensing-panel">
      <div className="flex items-center justify-between">
        <h2 className="flex items-center gap-2 text-lg font-semibold text-slate-200">
          <Activity className="h-5 w-5 text-emerald-400" />
          Co-Activation Sensing
        </h2>
        {events.length > 0 && (
          <Button
            variant="ghost"
            size="sm"
            onClick={() => {
              // Clears what the list shows: ALL clusters' events. A scoped
              // clear silently left other clusters' rows (011 R3) — and a
              // destructive action gets a confirm.
              if (window.confirm('Delete ALL stored sensing events?')) {
                clearEvents(undefined);
              }
            }}
          >
            <Trash2 className="mr-1 h-4 w-4" />
            Clear all
          </Button>
        )}
      </div>

      {/* Status strip */}
      <div className="flex flex-wrap items-center gap-3 rounded-lg border border-slate-700 bg-slate-800/60 px-3 py-2 text-xs text-slate-400">
        {status?.armed ? (
          <>
            <Badge variant="success">armed</Badge>
            <span className="text-slate-300">{status.profile_name}</span>
            <span className="font-mono">
              {status.member_count} members · quorum {status.min_k}
            </span>
            {status.threshold_mode === 'floor_only' && (
              <span title="No max_activation data in the definition — thresholds degraded to the floor">
                <Badge variant="warning">floor-only thresholds</Badge>
              </span>
            )}
            <span
              className={`font-mono ${
                status.last_request_overhead_ms > status.overhead_warn_threshold_ms
                  ? 'text-amber-400'
                  : ''
              }`}
            >
              {status.last_request_overhead_ms.toFixed(2)} ms/req
            </span>
          </>
        ) : (
          <span>
            {status?.enabled_clusters?.length
              ? `Not armed — sensing is enabled for ${status.enabled_clusters
                  .map((c) => c.name)
                  .join(', ')}; it arms when that cluster is active with an SAE attached.`
              : 'Not armed — activate a cluster with sensing enabled (and an SAE attached) to start observing co-activations.'}
          </span>
        )}
        {(status?.ws_events_dropped ?? 0) > 0 && (
          <span
            className="font-mono text-amber-400"
            title="Live updates throttled under burst — the stored list is complete; refresh to reconcile"
          >
            {status?.ws_events_dropped} live updates throttled
          </span>
        )}
        <span className="ml-auto font-mono">{totalEvents} stored</span>
      </div>

      {/* Event list */}
      {eventsLoading ? (
        <div className="text-sm text-slate-500">Loading events…</div>
      ) : events.length === 0 ? (
        <div className="rounded-lg border border-dashed border-slate-700 p-4 text-sm text-slate-500">
          No events yet. Events appear here live when the armed cluster's
          members co-fire during generation.
        </div>
      ) : (
        <ul className="space-y-1">
          {events.map((event) => (
            <li key={event.id}>
              <button
                type="button"
                onClick={() =>
                  setSelected(selected?.id === event.id ? null : event)
                }
                className={`w-full rounded px-3 py-2 text-left text-sm transition-colors ${
                  selected?.id === event.id
                    ? 'bg-slate-700/70 text-slate-200'
                    : 'bg-slate-800/40 text-slate-300 hover:bg-slate-800'
                }`}
              >
                <div className="flex items-center gap-2">
                  <span className="flex-1 truncate">{event.summary}</span>
                  <span className="shrink-0 font-mono text-xs text-slate-500">
                    {event.created_at
                      ? new Date(event.created_at).toLocaleTimeString()
                      : ''}
                  </span>
                </div>
              </button>
              {selected?.id === event.id && (
                <div className="mt-1">
                  <SensingEventDetail
                    event={selected}
                    onClose={() => setSelected(null)}
                  />
                </div>
              )}
            </li>
          ))}
        </ul>
      )}
    </section>
  );
}
