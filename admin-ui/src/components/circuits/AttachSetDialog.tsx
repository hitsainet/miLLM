/**
 * AttachSetDialog (Feature 12) — the multi-SAE attach-set control.
 *
 * The server's `/saes/attach-set` is the ONLY path that puts more than one SAE
 * in the registry; the single-attach path on the SAE page silently REPLACES the
 * previous SAE. This dialog is that path's UI.
 *
 * Honesty rules (the server can partially fail):
 *  - `attach_set` PRE-VALIDATES every new key, then rolls back everything it
 *    attached in THIS call if a later load/hook throws. A rejection therefore
 *    means "nothing changed", and we say so instead of showing a bare toast.
 *  - The response carries per-entry `warnings` (compatibility notes: wrong
 *    trained layer, different trained model) plus the summed VRAM and the
 *    envelope. All are surfaced verbatim rather than collapsed into "attached".
 */

import { useEffect, useMemo, useState } from 'react';
import { AlertTriangle, Info, Layers } from 'lucide-react';

import { Button, Modal, Spinner } from '@components/common';
import { ApiError } from '@/services/api';
import type { AttachSetItem, AttachSetResponse, SAEInfo } from '@/types';

/** One row in the picker: an SAE plus the layer it would attach to. */
export interface AttachCandidate {
  sae: SAEInfo;
  layer: number;
}

interface AttachSetDialogProps {
  open: boolean;
  onClose: () => void;
  /** Downloaded SAEs eligible for the currently loaded model. */
  candidates: AttachCandidate[];
  /** Keys already attached — used to pre-select ("edit the set"). */
  attached: AttachSetItem[];
  /** Name of the loaded model, or null when nothing is loaded. */
  loadedModelName: string | null;
  onSubmit: (saes: AttachSetItem[]) => Promise<AttachSetResponse>;
  isAttaching?: boolean;
}

const keyOf = (sae_id: string, layer: number) => `${sae_id}:${layer}`;

/** Per-entry warnings the server returned, flattened for display. */
function extractWarnings(res: AttachSetResponse): string[] {
  const out: string[] = [];
  for (const entry of res.entries ?? []) {
    const saeId = typeof entry.sae_id === 'string' ? entry.sae_id : '?';
    const layer = typeof entry.layer === 'number' ? entry.layer : '?';
    const warnings = entry.warnings;
    if (Array.isArray(warnings)) {
      for (const w of warnings) {
        if (typeof w === 'string' && w.trim()) {
          out.push(`${saeId} @ L${layer}: ${w}`);
        }
      }
    }
  }
  return out;
}

export function AttachSetDialog({
  open,
  onClose,
  candidates,
  attached,
  loadedModelName,
  onSubmit,
  isAttaching = false,
}: AttachSetDialogProps) {
  const attachedKeys = useMemo(
    () => attached.map((a) => keyOf(a.sae_id, a.layer)),
    [attached],
  );

  const [selected, setSelected] = useState<string[]>(attachedKeys);
  const [result, setResult] = useState<AttachSetResponse | null>(null);
  const [rejection, setRejection] = useState<{ message: string; code: string } | null>(
    null,
  );

  // Re-seed the selection from what is actually attached each time the dialog
  // opens (and whenever the attached set changes underneath it) so the control
  // always reads as "edit the current set", never a stale one.
  useEffect(() => {
    if (open) {
      setSelected(attachedKeys);
      setResult(null);
      setRejection(null);
    }
  }, [open, attachedKeys]);

  const toggle = (k: string) => {
    setSelected((prev) =>
      prev.includes(k) ? prev.filter((x) => x !== k) : [...prev, k],
    );
  };

  const selectedItems: AttachSetItem[] = useMemo(
    () =>
      candidates
        .filter((c) => selected.includes(keyOf(c.sae.id, c.layer)))
        .map((c) => ({ sae_id: c.sae.id, layer: c.layer })),
    [candidates, selected],
  );

  const noModel = !loadedModelName;
  const noCandidates = candidates.length === 0;

  const handleSubmit = async () => {
    setResult(null);
    setRejection(null);
    try {
      const res = await onSubmit(selectedItems);
      setResult(res);
    } catch (e) {
      // Show the server's reason VERBATIM. attach_set rolls back everything it
      // attached in this call, so the previous set is intact — say that too.
      if (e instanceof ApiError) {
        setRejection({ message: e.message, code: e.code });
      } else {
        setRejection({
          message: e instanceof Error ? e.message : 'Attach failed for an unknown reason.',
          code: 'UNKNOWN_ERROR',
        });
      }
    }
  };

  const warnings = result ? extractWarnings(result) : [];

  return (
    <Modal
      id="attach-set"
      isOpen={open}
      onClose={onClose}
      title="Attach SAE set"
      size="2xl"
    >
      <div className="space-y-4">
        {noModel ? (
          <p data-testid="no-model" className="text-sm text-slate-400">
            No model is loaded. Load a model before attaching SAEs — the server
            resolves each SAE&apos;s layer against the loaded model.
          </p>
        ) : (
          <>
            <p className="text-xs text-slate-500">
              SAEs available for{' '}
              <span className="font-mono text-slate-300">{loadedModelName}</span>.
              Attaching is <span className="text-slate-300">additive</span>:
              checked SAEs are attached alongside anything already attached.
              Unchecking a row does <span className="text-slate-300">not</span>{' '}
              detach it — use Detach on the SAE page for that.
            </p>

            {noCandidates ? (
              <p data-testid="no-candidates" className="text-sm text-slate-400">
                No downloaded SAEs are available for this model. Download an SAE
                first from the SAE page.
              </p>
            ) : (
              <div
                data-testid="candidate-list"
                className="max-h-72 space-y-1 overflow-y-auto rounded border border-slate-700 bg-slate-900/60 p-2"
              >
                {candidates.map((c) => {
                  const k = keyOf(c.sae.id, c.layer);
                  const isSelected = selected.includes(k);
                  const isAttached = attachedKeys.includes(k);
                  return (
                    <label
                      key={k}
                      data-testid="candidate-row"
                      className={`flex cursor-pointer items-center gap-3 rounded px-2 py-1.5 ${
                        isSelected ? 'bg-slate-800' : 'hover:bg-slate-800/50'
                      }`}
                    >
                      <input
                        type="checkbox"
                        checked={isSelected}
                        disabled={isAttaching}
                        onChange={() => toggle(k)}
                        aria-label={`${c.sae.id} at layer ${c.layer}`}
                        className="h-4 w-4 rounded border-slate-600 bg-slate-800 accent-cyan-500"
                      />
                      <span className="font-mono text-xs font-medium text-emerald-300">
                        L{c.layer}
                      </span>
                      <span className="flex-1 truncate font-mono text-xs text-slate-200">
                        {c.sae.id}
                      </span>
                      {c.sae.width && (
                        <span className="font-mono text-[11px] text-slate-500">
                          {c.sae.width}
                        </span>
                      )}
                      {isAttached && (
                        <span className="rounded bg-slate-700 px-1.5 py-0.5 text-[10px] text-slate-300">
                          attached
                        </span>
                      )}
                    </label>
                  );
                })}
              </div>
            )}

            {candidates.length === 1 && (
              <p data-testid="single-candidate-note" className="text-xs text-slate-500">
                Only one SAE is available, so this set cannot span layers. A
                cross-layer circuit needs one SAE per referenced layer.
              </p>
            )}
          </>
        )}

        {rejection && (
          <div
            data-testid="attach-rejection"
            className="rounded border border-red-500/40 bg-red-500/10 p-3"
          >
            <p className="flex items-center gap-1.5 text-xs font-semibold text-red-300">
              <AlertTriangle className="h-3.5 w-3.5" />
              Attach refused ({rejection.code}) — nothing was attached
            </p>
            <p className="mt-1 whitespace-pre-wrap text-xs text-red-200">
              {rejection.message}
            </p>
            <p className="mt-1 text-[11px] text-red-300/70">
              The server rolled back this call; the previously attached set is
              unchanged.
            </p>
          </div>
        )}

        {result && (
          <div
            data-testid="attach-result"
            className="space-y-2 rounded border border-slate-700 bg-slate-900/60 p-3"
          >
            <div className="flex flex-wrap items-center justify-between gap-2">
              <span className="flex items-center gap-1.5 text-xs text-slate-300">
                <Layers className="h-3.5 w-3.5 text-slate-400" />
                {result.attached_count} SAE(s) attached
              </span>
              <span
                data-testid="result-vram"
                className="font-mono text-xs text-slate-400"
              >
                {result.total_memory_usage_mb} MB
                <span className="text-slate-500">
                  {' '}
                  / {result.vram_envelope_mb} MB advisory
                </span>
              </span>
            </div>

            {result.vram_warning && (
              <p
                data-testid="result-vram-warning"
                className="flex items-center gap-1.5 text-xs text-slate-400"
              >
                <Info className="h-3.5 w-3.5" />
                Above the advisory budget — the attach succeeded. Real capacity
                is checked against free GPU memory, not this figure.
              </p>
            )}

            {warnings.length > 0 && (
              <ul data-testid="result-warnings" className="space-y-1">
                {warnings.map((w) => (
                  <li
                    key={w}
                    className="flex items-start gap-1.5 text-xs text-amber-300/90"
                  >
                    <Info className="mt-0.5 h-3 w-3 shrink-0" />
                    <span className="whitespace-pre-wrap">{w}</span>
                  </li>
                ))}
              </ul>
            )}
          </div>
        )}

        <div className="flex items-center justify-between gap-3">
          <span className="text-xs text-slate-500">
            {selectedItems.length} selected
          </span>
          <div className="flex gap-2">
            <Button variant="secondary" size="sm" onClick={onClose} data-testid="attach-set-cancel">
              {result ? 'Done' : 'Cancel'}
            </Button>
            <Button
              variant="primary"
              size="sm"
              data-testid="attach-set-submit"
              disabled={noModel || noCandidates || isAttaching}
              onClick={() => void handleSubmit()}
            >
              {isAttaching ? (
                <>
                  <Spinner /> Attaching…
                </>
              ) : (
                'Attach set'
              )}
            </Button>
          </div>
        </div>
      </div>
    </Modal>
  );
}
