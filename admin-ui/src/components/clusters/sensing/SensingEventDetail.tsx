/**
 * SensingEventDetail (Feature 11) — member table + context window with the
 * event span highlighted. Context is fetched via REST (WS payloads exclude
 * user content).
 */

import { X } from 'lucide-react';
import { Badge } from '@components/common';
import { useSensingEventDetail } from '@hooks/useSensing';
import type { SensingEvent } from '@/types/sensing';

interface SensingEventDetailProps {
  event: SensingEvent;
  onClose: () => void;
}

export function SensingEventDetail({ event, onClose }: SensingEventDetailProps) {
  const detailQuery = useSensingEventDetail(event.id);
  const detail = detailQuery.data ?? event;

  return (
    <div className="rounded-lg border border-slate-700 bg-slate-800/80 p-4 space-y-3">
      <div className="flex items-start justify-between">
        <div>
          <div className="text-sm font-medium text-slate-200">{detail.summary}</div>
          <div className="mt-1 flex items-center gap-2 text-xs text-slate-400">
            <Badge variant={detail.phase === 'prefill' ? 'primary' : 'success'}>
              {detail.phase}
            </Badge>
            <span className="font-mono">
              span {detail.pos_start}
              {detail.pos_end !== detail.pos_start && `–${detail.pos_end}`}
            </span>
            <span className="font-mono">score {detail.score.toFixed(2)}×θ</span>
            {detail.ambient_fired_count != null && (
              <span className="font-mono">ambient {detail.ambient_fired_count}</span>
            )}
            {detail.truncated && <Badge variant="warning">truncated</Badge>}
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
            <th className="py-1 pr-4 font-normal">Member</th>
            <th className="py-1 font-normal">Peak activation</th>
          </tr>
        </thead>
        <tbody>
          {detail.fired_members.map(([idx, act]) => (
            <tr key={idx} className="border-t border-slate-700/60">
              <td className="py-1 pr-4 font-mono text-slate-300">#{idx}</td>
              <td className="py-1 font-mono text-slate-300">{act.toFixed(3)}</td>
            </tr>
          ))}
        </tbody>
      </table>

      {detailQuery.isLoading && (
        <div className="text-xs text-slate-500">Loading context…</div>
      )}
      {detail.context_text != null && detail.context_text !== '' && (
        <div>
          <div className="mb-1 text-xs text-slate-500">
            Context (±K tokens around the span)
          </div>
          <div className="rounded bg-slate-900/70 p-2 font-mono text-xs leading-relaxed text-slate-300">
            {detail.context_text}
          </div>
        </div>
      )}
      {detailQuery.isSuccess && !detail.context_text && (
        <div className="text-xs text-slate-500">
          No context captured for this event (context capture disabled for
          the cluster, or the window could not be decoded).
        </div>
      )}
    </div>
  );
}
