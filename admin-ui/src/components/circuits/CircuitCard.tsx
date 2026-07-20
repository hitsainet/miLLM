/**
 * CircuitCard (Feature 13) — one imported circuit: identity, evidence rung,
 * layers/edges, per-SAE compatibility, and the activate/deactivate control.
 *
 * EVIDENCE HONESTY: the rung badge renders the SERVER's `rung_language`
 * verbatim. This component never derives or re-phrases an evidence claim, and
 * a circuit below rung 2 is visibly marked "unvalidated".
 */

import { useState } from 'react';
import { AlertTriangle, Download, Share2, Square, Trash2 } from 'lucide-react';

import { Badge, Button } from '@components/common';
import type { CircuitSummary, PerSAEVerdictKind } from '@/types/circuits';
import { CircuitActivateControl } from './CircuitActivateControl';
import { EdgeSensingToggle } from './sensing/EdgeSensingToggle';

interface CircuitCardProps {
  circuit: CircuitSummary;
  onActivate: (acknowledgeUnvalidated: boolean) => void;
  onDeactivate: () => void;
  onDelete: () => void;
  onExport: () => void;
  isActivating?: boolean;
  isDeactivating?: boolean;
}

/** Rung → badge colour. Deliberately NOT a language map: the phrase itself
 *  always comes from the server so the two can never disagree. */
const RUNG_VARIANT: Record<number, 'default' | 'warning' | 'success' | 'purple'> = {
  0: 'default',
  1: 'warning',
  2: 'success',
  3: 'purple',
};

const VERDICT_VARIANT: Record<PerSAEVerdictKind, 'success' | 'warning' | 'danger' | 'default'> = {
  bind: 'success',
  warn: 'warning',
  block: 'danger',
  unbound: 'default',
};

export function CircuitCard({
  circuit,
  onActivate,
  onDeactivate,
  onDelete,
  onExport,
  isActivating = false,
  isDeactivating = false,
}: CircuitCardProps) {
  const [showDetail, setShowDetail] = useState(false);

  return (
    <div
      data-testid="circuit-card"
      className="rounded-lg border border-slate-700 bg-slate-800/40 p-4 space-y-3"
    >
      <div className="flex items-start justify-between gap-3">
        <div className="min-w-0">
          <div className="flex items-center gap-2 flex-wrap">
            <Share2 className="w-4 h-4 text-cyan-400 shrink-0" />
            <span className="font-medium text-slate-100 truncate">{circuit.name}</span>

            {/* Evidence rung — server-rendered phrase, verbatim. */}
            <span data-testid="rung-badge">
              <Badge variant={RUNG_VARIANT[circuit.rung] ?? 'default'} size="sm">
                {circuit.rung_language}
              </Badge>
            </span>

            {!circuit.validated && (
              <span data-testid="unvalidated-badge">
                <Badge variant="warning" size="sm">
                  unvalidated
                </Badge>
              </span>
            )}

            {circuit.is_active && (
              <Badge variant="success" size="sm">
                {circuit.serving_mode === 'slice_fallback' ? 'serving slice' : 'serving'}
              </Badge>
            )}
          </div>

          <div className="text-xs text-slate-400 mt-1 flex items-center gap-2 flex-wrap">
            <span className="font-mono">
              {circuit.layers.map((l) => `L${l}`).join(' → ')}
            </span>
            <span className="text-slate-500">·</span>
            <span>{circuit.edge_count} edge{circuit.edge_count === 1 ? '' : 's'}</span>
            {!circuit.serveable && (
              <>
                <span className="text-slate-500">·</span>
                <span className="text-yellow-400/90">SAE set incomplete</span>
              </>
            )}
          </div>

          {circuit.description && (
            <p className="text-xs text-slate-400 mt-2 line-clamp-2">{circuit.description}</p>
          )}
        </div>

        <div className="flex items-center gap-1 shrink-0">
          <EdgeSensingToggle circuitId={circuit.id} />
          <Button variant="ghost" size="sm" onClick={onExport} title="Export definition">
            <Download className="w-4 h-4" />
          </Button>
          <Button variant="ghost" size="sm" onClick={onDelete} title="Delete circuit">
            <Trash2 className="w-4 h-4 text-red-400/80" />
          </Button>
        </div>
      </div>

      {/* Per-SAE compatibility verdicts */}
      {circuit.per_sae_warnings.length > 0 && (
        <div className="flex flex-wrap gap-1.5" data-testid="per-sae-verdicts">
          {circuit.per_sae_warnings.map((v) => (
            <span
              key={`${v.layer}:${v.sae_id ?? 'none'}`}
              title={v.reason ?? undefined}
              className="inline-flex"
            >
              <Badge variant={VERDICT_VARIANT[v.verdict]} size="sm">
                L{v.layer} {v.verdict}
              </Badge>
            </span>
          ))}
        </div>
      )}

      {/* Slice-fallback disclosure — a slice is NOT the circuit. */}
      {circuit.is_active && circuit.serving_mode === 'slice_fallback' && (
        <div
          data-testid="slice-disclosure"
          className="flex items-start gap-2 rounded border border-yellow-500/30 bg-yellow-500/5 p-2"
        >
          <AlertTriangle className="w-4 h-4 text-yellow-400 shrink-0 mt-0.5" />
          <p className="text-xs text-yellow-200/90">
            Serving a <strong>per-layer slice</strong> of this circuit, not the whole
            circuit — some referenced SAEs are not attached.
          </p>
        </div>
      )}

      <div className="flex items-center justify-between gap-2">
        <button
          type="button"
          onClick={() => setShowDetail((s) => !s)}
          className="text-xs text-slate-400 hover:text-slate-200"
        >
          {showDetail ? 'Hide' : 'What would raise this rung?'}
        </button>

        {circuit.is_active ? (
          <Button variant="secondary" size="sm" onClick={onDeactivate} disabled={isDeactivating}>
            <Square className="w-4 h-4 mr-1" />
            Deactivate
          </Button>
        ) : (
          <CircuitActivateControl
            circuit={circuit}
            onActivate={onActivate}
            isActivating={isActivating}
          />
        )}
      </div>

      {showDetail && (
        <p data-testid="next-step" className="text-xs text-slate-400 border-t border-slate-700 pt-2">
          {circuit.rung_next_step}
        </p>
      )}
    </div>
  );
}
