/**
 * EdgeSensingPanel (Feature 15) — status strip (armed circuit, layers,
 * sensable edges, token lag, overhead) + newest-first observed-edge list with
 * WS live prepend.
 *
 * ABSENCE OF OBSERVATION IS NOT EVIDENCE OF ABSENCE: unsensable edges are
 * surfaced prominently, and above the event list, so an empty list can never
 * be read as "the edge never fired". The list's own empty state says the same
 * thing in words.
 */

import { useState } from 'react';
import { Activity, EyeOff, Trash2 } from 'lucide-react';
import { Badge, Button } from '@components/common';
import { useCircuitSensing } from '@hooks/useCircuitSensing';
import type {
  CircuitSensingEvent,
  UnsensableEdgeInfo,
} from '@/types/circuitSensing';
import { EdgeSensingEventDetail } from './EdgeSensingEventDetail';

/** Reason code → plain description of what was NOT watched. These describe
 *  instrumentation coverage only — never evidence about the edge itself. */
const REASON_TEXT: Record<string, string> = {
  layer_not_attached: 'no SAE attached at that layer',
  no_activation_threshold: 'no activation threshold available',
  endpoint_not_a_feature: 'an endpoint is not an SAE feature',
};

function describeReason(info: UnsensableEdgeInfo): string {
  return REASON_TEXT[info.reason] ?? info.reason;
}

export function EdgeSensingPanel({ circuitId }: { circuitId?: string }) {
  const { status, events, totalEvents, eventsLoading, clearEvents } =
    useCircuitSensing(circuitId);
  const [selected, setSelected] = useState<CircuitSensingEvent | null>(null);

  const unsensable = status?.unsensable_edges ?? [];

  return (
    <section className="space-y-3" data-testid="edge-sensing-panel">
      <div className="flex items-center justify-between">
        <h2 className="flex items-center gap-2 text-lg font-semibold text-slate-200">
          <Activity className="h-5 w-5 text-cyan-400" />
          Edge Sensing
        </h2>
        {events.length > 0 && (
          <Button
            variant="ghost"
            size="sm"
            onClick={() => {
              // Clears what the list shows: ALL circuits' events. A scoped
              // clear silently left other circuits' rows (011 R3) — and a
              // destructive action gets a confirm.
              if (window.confirm('Delete ALL stored edge sensing events?')) {
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
            <span className="text-slate-300">{status.circuit_name}</span>
            <span className="font-mono">
              {status.layers.map((l) => `L${l}`).join(' → ')}
            </span>
            <span className="font-mono">
              {status.sensable_edges} edge
              {status.sensable_edges === 1 ? '' : 's'} watched
            </span>
            <span
              className="font-mono"
              title="An up→down pair counts as one firing only within this many tokens"
            >
              lag ≤ {status.max_token_lag}
            </span>
            <span className="font-mono">
              {status.last_request_overhead_ms.toFixed(2)} ms/req
            </span>
          </>
        ) : (
          <span>
            {status?.enabled_circuits?.length
              ? `Not armed — edge sensing is enabled for ${status.enabled_circuits
                  .map((c) => c.name)
                  .join(', ')}; it arms when that circuit is active with its SAEs attached.`
              : 'Not armed — activate a circuit with edge sensing enabled (and its SAEs attached) to start observing edge firings.'}
          </span>
        )}
        {(status?.ws_dropped ?? 0) > 0 && (
          <span
            className="font-mono text-amber-400"
            title="Live updates throttled under burst — the stored list is complete; refresh to reconcile"
          >
            {status?.ws_dropped} live updates throttled
          </span>
        )}
        <span className="ml-auto font-mono">{totalEvents} stored</span>
      </div>

      {/* Unsensable edges — coverage gaps, surfaced ABOVE the list so an empty
          list is never read as "the edge never fired". */}
      {unsensable.length > 0 && (
        <div
          data-testid="unsensable-edges"
          className="rounded-lg border border-amber-500/30 bg-amber-500/5 p-3 space-y-2"
        >
          <div className="flex items-center gap-2 text-xs font-medium text-amber-200">
            <EyeOff className="h-4 w-4 shrink-0" />
            {unsensable.length} edge{unsensable.length === 1 ? ' is' : 's are'}{' '}
            not being watched
          </div>
          <p className="text-xs text-amber-200/80">
            These edges cannot be observed, so no event will ever appear for
            them. An empty list below says nothing about whether they fired.
          </p>
          <ul className="space-y-1">
            {unsensable.map((info) => (
              <li
                key={info.edge_key}
                className="flex flex-wrap items-baseline gap-2 text-xs"
              >
                <span className="font-mono text-slate-300">{info.edge_key}</span>
                <span className="text-amber-200/90">{describeReason(info)}</span>
                {info.detail && (
                  <span className="text-slate-400">— {info.detail}</span>
                )}
              </li>
            ))}
          </ul>
        </div>
      )}

      {/* Event list */}
      {eventsLoading ? (
        <div className="text-sm text-slate-500">Loading events…</div>
      ) : events.length === 0 ? (
        <div className="rounded-lg border border-dashed border-slate-700 p-4 text-sm text-slate-500">
          No edge firings observed yet. Events appear here live when a watched
          edge fires during generation. An empty list means nothing was
          observed — not that the edges did not fire.
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
                  {event.edge_rung < 2 && (
                    <Badge variant="warning" size="sm">
                      unvalidated
                    </Badge>
                  )}
                  <span className="shrink-0 font-mono text-xs text-slate-500">
                    {event.created_at
                      ? new Date(event.created_at).toLocaleTimeString()
                      : ''}
                  </span>
                </div>
              </button>
              {selected?.id === event.id && (
                <div className="mt-1">
                  <EdgeSensingEventDetail
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
