/**
 * EdgeSensingEventDetail (Feature 15) — the up→down endpoint pair, token lag,
 * the evidence rung, and the context window with the fired span highlighted.
 * Context is fetched via REST (WS payloads carry no prompt text).
 *
 * EVIDENCE HONESTY: `edge_rung_language` is rendered VERBATIM. This component
 * never derives, maps, or re-phrases an evidence claim, and an edge below
 * rung 2 is visibly marked "unvalidated".
 */

import { X } from 'lucide-react';
import { Badge } from '@components/common';
import { useCircuitSensingEventDetail } from '@hooks/useCircuitSensing';
import type { CircuitSensingEvent, EdgeEndpoint } from '@/types/circuitSensing';

interface EdgeSensingEventDetailProps {
  event: CircuitSensingEvent;
  onClose: () => void;
}

/** Rung → badge colour. Deliberately NOT a language map: the phrase itself
 *  always comes from the server so the two can never disagree. */
const RUNG_VARIANT: Record<number, 'default' | 'warning' | 'success' | 'purple'> = {
  0: 'default',
  1: 'warning',
  2: 'success',
  3: 'purple',
};

function EndpointRow({ role, endpoint }: { role: string; endpoint: EdgeEndpoint }) {
  return (
    <tr className="border-t border-slate-700/60">
      <td className="py-1 pr-4 text-slate-400">{role}</td>
      <td className="py-1 pr-4 font-mono text-slate-300">L{endpoint.layer}</td>
      <td className="py-1 pr-4 font-mono text-slate-300">#{endpoint.feature_idx}</td>
      <td className="py-1 pr-4 font-mono text-slate-300">@{endpoint.pos}</td>
      <td className="py-1 font-mono text-slate-300">{endpoint.act.toFixed(3)}</td>
    </tr>
  );
}

export function EdgeSensingEventDetail({
  event,
  onClose,
}: EdgeSensingEventDetailProps) {
  const detailQuery = useCircuitSensingEventDetail(event.id);
  const detail = detailQuery.data ?? event;

  return (
    <div className="rounded-lg border border-slate-700 bg-slate-800/80 p-4 space-y-3">
      <div className="flex items-start justify-between">
        <div>
          <div className="text-sm font-medium text-slate-200">{detail.summary}</div>
          <div className="mt-1 flex items-center gap-2 flex-wrap text-xs text-slate-400">
            <Badge variant={detail.phase === 'prefill' ? 'primary' : 'success'}>
              {detail.phase}
            </Badge>
            <span className="font-mono">{detail.edge_key}</span>
            <span className="font-mono">lag {detail.token_lag}</span>
            {detail.edge_type && <span className="font-mono">{detail.edge_type}</span>}
            {detail.ambient_fired_count != null && (
              <span className="font-mono">ambient {detail.ambient_fired_count}</span>
            )}
            {detail.truncated && <Badge variant="warning">truncated</Badge>}
          </div>

          {/* Evidence rung — server-rendered phrase, verbatim. */}
          <div className="mt-2 flex items-center gap-2 flex-wrap">
            <span data-testid="edge-rung-badge">
              <Badge variant={RUNG_VARIANT[detail.edge_rung] ?? 'default'} size="sm">
                {detail.edge_rung_language}
              </Badge>
            </span>
            {detail.edge_rung < 2 && (
              <span data-testid="edge-unvalidated-badge">
                <Badge variant="warning" size="sm">
                  unvalidated
                </Badge>
              </span>
            )}
          </div>
        </div>
        <button
          type="button"
          onClick={onClose}
          aria-label="Close detail"
          className="text-slate-500 hover:text-slate-300"
        >
          <X className="h-4 w-4" />
        </button>
      </div>

      <table className="w-full text-xs">
        <thead>
          <tr className="text-left text-slate-500">
            <th className="py-1 pr-4 font-normal">End</th>
            <th className="py-1 pr-4 font-normal">Layer</th>
            <th className="py-1 pr-4 font-normal">Feature</th>
            <th className="py-1 pr-4 font-normal">Position</th>
            <th className="py-1 font-normal">Activation</th>
          </tr>
        </thead>
        <tbody>
          <EndpointRow role="up" endpoint={detail.up} />
          <EndpointRow role="down" endpoint={detail.down} />
        </tbody>
      </table>

      {detailQuery.isLoading && (
        <div className="text-xs text-slate-500">Loading context…</div>
      )}
      {(detail.context_parts != null ||
        (detail.context_text != null && detail.context_text !== '')) && (
        <div>
          <div className="mb-1 text-xs text-slate-500">
            Context (±K tokens around the span)
          </div>
          <div className="whitespace-pre-wrap rounded bg-slate-900/70 p-2 font-mono text-xs leading-relaxed text-slate-300">
            {detail.context_parts ? (
              <>
                {detail.context_parts.before}
                <mark className="rounded-sm bg-cyan-500/25 px-0.5 font-semibold text-cyan-200">
                  {detail.context_parts.span}
                </mark>
                {detail.context_parts.after}
              </>
            ) : (
              detail.context_text
            )}
          </div>
          {!detail.context_parts && (
            <div className="mt-1 text-[10px] text-slate-600">
              Recorded before span highlighting — plain context only.
            </div>
          )}
        </div>
      )}
      {detailQuery.isSuccess && !detail.context_text && (
        <div className="text-xs text-slate-500">
          No context captured for this event (context capture disabled for the
          circuit, or the window could not be decoded).
        </div>
      )}
    </div>
  );
}
