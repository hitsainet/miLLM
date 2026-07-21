/**
 * ClaimsStrip (Feature 19) — layer → claimant.
 *
 * The unit of contention is the LAYER, not the circuit, so "what is serving"
 * is not answerable from the circuit list alone: two circuits can both be
 * active while contending for nothing at all. This strip is what makes a
 * refusal intelligible BEFORE it happens — an operator can see that L13 is
 * taken before trying to activate onto it.
 *
 * A composed layer is badged, because it is the one case where the rung header
 * disappears and the response carries a summed effect no single circuit
 * describes.
 */

import { Layers } from 'lucide-react';

export interface LayerClaim {
  layer: number;
  circuit_id: string;
  circuit_name: string | null;
  composed: boolean;
  steering_keys?: number[];
}

interface ClaimsStripProps {
  claims: LayerClaim[];
}

export function ClaimsStrip({ claims }: ClaimsStripProps) {
  if (claims.length === 0) {
    return (
      <div
        className="flex items-center gap-2 text-xs text-slate-500"
        data-testid="claims-strip-empty"
      >
        <Layers className="h-3.5 w-3.5" />
        No layers are currently claimed
      </div>
    );
  }

  // Group by layer: a composed layer legitimately has several claimants, and
  // showing them separately would read as two unrelated claims rather than as
  // the composition it is.
  const byLayer = new Map<number, LayerClaim[]>();
  for (const claim of claims) {
    const existing = byLayer.get(claim.layer);
    if (existing) existing.push(claim);
    else byLayer.set(claim.layer, [claim]);
  }

  return (
    <div className="flex flex-wrap items-center gap-1.5" data-testid="claims-strip">
      {[...byLayer.entries()]
        .sort(([a], [b]) => a - b)
        .map(([layer, holders]) => {
          const composed = holders.some((h) => h.composed) || holders.length > 1;
          return (
            <span
              key={layer}
              data-testid={`claim-L${layer}`}
              className={[
                'inline-flex items-center gap-1 rounded px-2 py-0.5 text-xs',
                composed
                  ? 'bg-amber-950/40 text-amber-200 border border-amber-800/60'
                  : 'bg-slate-800 text-slate-300 border border-slate-700',
              ].join(' ')}
              title={holders
                .map((h) => h.circuit_name ?? h.circuit_id)
                .join(' + ')}
            >
              <span className="font-mono">L{layer}</span>
              <span className="text-slate-400">·</span>
              <span className="truncate max-w-[10rem]">
                {holders.map((h) => h.circuit_name ?? h.circuit_id).join(' + ')}
              </span>
              {composed && (
                <span
                  className="ml-0.5 rounded bg-amber-900/60 px-1 text-[10px] uppercase tracking-wide"
                  data-testid={`composed-badge-L${layer}`}
                >
                  composed
                </span>
              )}
            </span>
          );
        })}
    </div>
  );
}
