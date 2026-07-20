/**
 * AttachmentPanel (Feature 12) — the multi-SAE attachment set for circuit
 * serving: one chip per attached (sae_id, layer), the summed VRAM readout, and
 * a VRAM-over-envelope warning badge.
 */
import { Layers, AlertTriangle } from 'lucide-react';

import { Badge, Spinner, EmptyState } from '@components/common';
import { useAttachments } from '@/hooks/useAttachments';

interface AttachmentPanelProps {
  /**
   * Only render once MORE than this many SAEs are attached. Defaults to 0
   * (always render). The SAE page passes 1 so the single-SAE case is covered
   * by the existing AttachedSAECard and is not shown twice.
   */
  minCount?: number;
}

export function AttachmentPanel({ minCount = 0 }: AttachmentPanelProps = {}) {
  const { attachments, isLoading, error } = useAttachments();

  // Below the render threshold (e.g. the single-SAE case on the SAE page,
  // already covered by AttachedSAECard) — render nothing at all, including
  // while loading/erroring, rather than a duplicate or empty panel.
  if (minCount > 0 && (attachments?.count ?? 0) <= minCount) {
    return null;
  }

  if (isLoading) {
    return (
      <div className="flex items-center gap-2 text-slate-400 text-sm p-4">
        <Spinner /> Loading attachments…
      </div>
    );
  }

  if (error) {
    return (
      <div className="text-sm text-red-400 p-4">
        Could not load attachment status. It will retry automatically.
      </div>
    );
  }

  if (!attachments || !attachments.is_attached || attachments.count === 0) {
    return (
      <EmptyState
        icon={<Layers className="w-6 h-6" />}
        title="No SAEs attached"
        description="Attach a set of per-layer SAEs to serve a cross-layer circuit."
      />
    );
  }

  const total = attachments.total_memory_usage_mb;
  const envelope = attachments.vram_envelope_mb;

  return (
    <div className="space-y-3">
      <div className="flex items-center justify-between gap-3 flex-wrap">
        <div className="flex items-center gap-2">
          <Layers className="w-4 h-4 text-slate-400" />
          <span className="text-sm font-medium text-slate-100">
            Attached SAEs
          </span>
          <span className="text-xs text-slate-400">({attachments.count})</span>
        </div>
        <div className="flex items-center gap-2">
          <span className="text-xs font-mono text-slate-400">
            {total != null ? `${total} MB` : '—'}
            {envelope != null && (
              <span className="text-slate-500"> / {envelope} MB</span>
            )}
          </span>
          {attachments.vram_warning && (
            <span data-testid="vram-warning">
              <Badge variant="warning">
                <AlertTriangle className="w-3 h-3 mr-1" />
                VRAM over envelope
              </Badge>
            </span>
          )}
        </div>
      </div>

      <div className="flex flex-wrap gap-2">
        {attachments.entries.map((entry) => (
          <div
            key={`${entry.sae_id}:${entry.layer}`}
            data-testid="attachment-chip"
            className="flex items-center gap-2 rounded-md border border-slate-700 bg-slate-800/60 px-2.5 py-1.5"
          >
            <span className="text-xs font-medium text-emerald-300 font-mono">
              L{entry.layer}
            </span>
            <span className="text-xs text-slate-300 font-mono truncate max-w-[10rem]">
              {entry.sae_id}
            </span>
            {entry.memory_usage_mb != null && (
              <span className="text-[11px] text-slate-500 font-mono">
                {entry.memory_usage_mb} MB
              </span>
            )}
            {entry.steering_enabled && (
              <span
                className="w-1.5 h-1.5 rounded-full bg-emerald-400"
                title="steering active"
              />
            )}
          </div>
        ))}
      </div>
    </div>
  );
}
