/**
 * AttachmentPanel (Feature 12) — the multi-SAE attachment set for circuit
 * serving: one chip per attached (sae_id, layer), the summed VRAM readout, a
 * VRAM-over-envelope warning badge, and the "Attach set" control that opens
 * the multi-select picker (AttachSetDialog).
 */
import { useMemo, useState } from 'react';
import { Layers, AlertTriangle, Plus } from 'lucide-react';

import { Badge, Button, Spinner, EmptyState } from '@components/common';
import { useAttachments } from '@/hooks/useAttachments';
import { useSAE } from '@/hooks/useSAE';
import { useServerStore } from '@/stores/serverStore';
import type { AttachSetItem, SAEInfo } from '@/types';

import { AttachSetDialog, type AttachCandidate } from './AttachSetDialog';

interface AttachmentPanelProps {
  /**
   * Only render once MORE than this many SAEs are attached. Defaults to 0
   * (always render). The SAE page passes 1 so the single-SAE case is covered
   * by the existing AttachedSAECard and is not shown twice.
   */
  minCount?: number;
}

/**
 * The layer an SAE would attach to. The server checks `trained_layer` against
 * the requested layer and WARNS on a mismatch, so defaulting to trained_layer
 * is the only choice that does not manufacture a warning. SAEs with no
 * trained_layer cannot be placed without guessing and are excluded.
 */
function toCandidates(saes: SAEInfo[]): AttachCandidate[] {
  return saes
    .filter((s) => s.trained_layer != null && (s.status === 'cached' || s.status === 'attached'))
    .map((s) => ({ sae: s, layer: s.trained_layer as number }))
    .sort((a, b) => a.layer - b.layer || a.sae.id.localeCompare(b.sae.id));
}

export function AttachmentPanel({ minCount = 0 }: AttachmentPanelProps = {}) {
  const { attachments, isLoading, error, attachSet, isAttaching } = useAttachments();
  const { saes } = useSAE();
  const loadedModel = useServerStore((s) => s.loadedModel);
  const [pickerOpen, setPickerOpen] = useState(false);

  const candidates = useMemo(() => toCandidates(saes ?? []), [saes]);

  const attachedItems: AttachSetItem[] = useMemo(
    () =>
      (attachments?.entries ?? []).map((e) => ({
        sae_id: e.sae_id,
        layer: e.layer,
      })),
    [attachments],
  );

  const dialog = (
    <AttachSetDialog
      open={pickerOpen}
      onClose={() => setPickerOpen(false)}
      candidates={candidates}
      attached={attachedItems}
      loadedModelName={loadedModel?.name ?? null}
      onSubmit={attachSet}
      isAttaching={isAttaching}
    />
  );

  const attachButton = (
    <Button
      variant="secondary"
      size="sm"
      data-testid="open-attach-set"
      onClick={() => setPickerOpen(true)}
      disabled={isAttaching}
    >
      <Plus className="mr-1 h-3.5 w-3.5" />
      Attach set
    </Button>
  );

  // Below the render threshold (e.g. the single-SAE case on the SAE page,
  // already covered by AttachedSAECard) — render nothing at all, including
  // while loading/erroring, rather than a duplicate or empty panel.
  if (minCount > 0 && (attachments?.count ?? 0) <= minCount) {
    return null;
  }

  if (isLoading) {
    return (
      <div className="flex items-center gap-2 p-4 text-sm text-slate-400">
        <Spinner /> Loading attachments…
      </div>
    );
  }

  if (error) {
    return (
      <div className="p-4 text-sm text-red-400">
        Could not load attachment status. It will retry automatically.
      </div>
    );
  }

  if (!attachments || !attachments.is_attached || attachments.count === 0) {
    return (
      <>
        <EmptyState
          icon={<Layers className="h-6 w-6" />}
          title="No SAEs attached"
          description="Attach a set of per-layer SAEs to serve a cross-layer circuit."
          action={attachButton}
        />
        {dialog}
      </>
    );
  }

  const total = attachments.total_memory_usage_mb;
  const envelope = attachments.vram_envelope_mb;

  return (
    <div className="space-y-3">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div className="flex items-center gap-2">
          <Layers className="h-4 w-4 text-slate-400" />
          <span className="text-sm font-medium text-slate-100">
            Attached SAEs
          </span>
          <span className="text-xs text-slate-400">({attachments.count})</span>
        </div>
        <div className="flex items-center gap-2">
          <span className="font-mono text-xs text-slate-400">
            {total != null ? `${total} MB` : '—'}
            {envelope != null && (
              <span className="text-slate-500"> / {envelope} MB</span>
            )}
          </span>
          {attachments.vram_warning && (
            <span data-testid="vram-warning">
              <Badge variant="warning">
                <AlertTriangle className="mr-1 h-3 w-3" />
                VRAM over envelope
              </Badge>
            </span>
          )}
          {attachButton}
        </div>
      </div>

      <div className="flex flex-wrap gap-2">
        {attachments.entries.map((entry) => (
          <div
            key={`${entry.sae_id}:${entry.layer}`}
            data-testid="attachment-chip"
            className="flex items-center gap-2 rounded-md border border-slate-700 bg-slate-800/60 px-2.5 py-1.5"
          >
            <span className="font-mono text-xs font-medium text-emerald-300">
              L{entry.layer}
            </span>
            <span className="max-w-[10rem] truncate font-mono text-xs text-slate-300">
              {entry.sae_id}
            </span>
            {entry.memory_usage_mb != null && (
              <span className="font-mono text-[11px] text-slate-500">
                {entry.memory_usage_mb} MB
              </span>
            )}
            {entry.steering_enabled && (
              <span
                className="h-1.5 w-1.5 rounded-full bg-emerald-400"
                title="steering active"
              />
            )}
          </div>
        ))}
      </div>

      {dialog}
    </div>
  );
}
