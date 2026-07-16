/**
 * ClusterCard (Feature 8) — one imported cluster: identity, members, bound
 * state, warnings, narrative (collapsible markdown-ish text), the lambda
 * intensity slider, and activate/export actions.
 */

import { useState } from 'react';
import {
  Activity,
  AlertTriangle,
  ChevronDown,
  ChevronRight,
  Download,
  Play,
  Square,
} from 'lucide-react';
import { Badge, Button } from '@components/common';
import type { ClusterSummary } from '@/types/clusters';

interface ClusterCardProps {
  cluster: ClusterSummary;
  onActivate: () => void;
  onDeactivate: () => void;
  onSetIntensity: (intensity: number) => void;
  onExport: () => void;
  onToggleSensing?: (enabled: boolean) => void;
  isActivating?: boolean;
  isDeactivating?: boolean;
}

export function ClusterCard({
  cluster,
  onActivate,
  onDeactivate,
  onSetIntensity,
  onExport,
  onToggleSensing,
  isActivating,
  isDeactivating,
}: ClusterCardProps) {
  const [showNarrative, setShowNarrative] = useState(false);
  const [pendingIntensity, setPendingIntensity] = useState<number | null>(null);

  const intensity = pendingIntensity ?? cluster.intensity;
  // Authored safe envelope (server-enforced); dial-off (0) is always allowed.
  const [rangeLo, rangeHi] =
    cluster.intensity_range?.length === 2 ? cluster.intensity_range : [0, 2];
  const sliderMin = Math.min(0, rangeLo);
  const sliderMax = Math.max(rangeHi, 0);

  return (
    <div
      className={`p-4 rounded-lg border transition-colors ${
        cluster.is_active
          ? 'bg-green-500/5 border-green-500/30'
          : 'bg-slate-800/30 border-slate-700/50 hover:border-slate-600/50'
      }`}
    >
      <div className="flex items-start justify-between gap-3">
        <div className="min-w-0">
          <div className="flex items-center gap-2 flex-wrap">
            <span className="font-medium text-slate-100 truncate">{cluster.name}</span>
            {cluster.display_token && (
              <Badge variant="primary">{cluster.display_token}</Badge>
            )}
            {cluster.is_active && <Badge variant="success">active</Badge>}
            {!cluster.bound && <Badge variant="warning">unbound</Badge>}
            {cluster.hub_ref && <Badge variant="default">hub</Badge>}
          </div>
          <div className="text-xs text-slate-400 mt-1">
            {cluster.member_count} member{cluster.member_count === 1 ? '' : 's'}
            {cluster.layer != null && ` · L${cluster.layer}`}
            {cluster.model_id && ` · ${cluster.model_id}`}
            {cluster.hub_ref && ` · ${cluster.hub_ref.repo_id}@${cluster.hub_ref.revision}`}
            {cluster.budget_b != null && (
              <span className="font-mono"> · B {cluster.budget_b.toFixed(2)}</span>
            )}
            {cluster.formula_id && (
              <span className="font-mono text-slate-500"> · {cluster.formula_id}</span>
            )}
          </div>
        </div>

        <div className="flex items-center gap-2 shrink-0">
          {onToggleSensing && (
            <Button
              variant="ghost"
              size="sm"
              onClick={() => onToggleSensing(!cluster.sensing_enabled)}
              title={
                cluster.sensing_enabled
                  ? 'Sensing enabled — co-activation events are recorded while this cluster is active. Click to disable.'
                  : 'Enable co-activation sensing (arms while this cluster is active with an SAE attached)'
              }
              data-testid="sensing-toggle"
            >
              <Activity
                className={`w-4 h-4 ${
                  cluster.sensing_enabled ? 'text-emerald-400' : 'text-slate-500'
                }`}
              />
            </Button>
          )}
          <Button
            variant="ghost"
            size="sm"
            onClick={onExport}
            title="Export portable definition (.cluster.json)"
          >
            <Download className="w-4 h-4" />
          </Button>
          {cluster.is_active ? (
            <Button
              variant="secondary"
              size="sm"
              onClick={onDeactivate}
              disabled={isDeactivating}
            >
              <Square className="w-4 h-4 mr-1" />
              Deactivate
            </Button>
          ) : (
            <Button
              variant="primary"
              size="sm"
              onClick={onActivate}
              disabled={isActivating}
              title={
                cluster.bound
                  ? 'Apply all members at their tuned strengths'
                  : 'Unbound — activating binds this cluster against the attached SAE (blocked server-side if incompatible)'
              }
            >
              <Play className="w-4 h-4 mr-1" />
              Activate
            </Button>
          )}
        </div>
      </div>

      {cluster.warnings.length > 0 && (
        <div className="mt-2 space-y-1">
          {cluster.warnings.map((w) => (
            <div key={w} className="flex items-start gap-1.5 text-xs text-amber-400/90">
              <AlertTriangle className="w-3.5 h-3.5 shrink-0 mt-0.5" />
              <span>{w}</span>
            </div>
          ))}
        </div>
      )}

      {/* Intensity dial — values are stored at λ=1; the server scales+clamps */}
      <div className="mt-3 flex items-center gap-3">
        <span className="text-xs text-slate-400 font-mono w-16">λ {intensity.toFixed(2)}</span>
        <input
          type="range"
          min={sliderMin}
          max={sliderMax}
          step={0.05}
          value={intensity}
          aria-label={`Intensity for ${cluster.name}`}
          className="flex-1 accent-cyan-500"
          onChange={(e) => setPendingIntensity(parseFloat(e.target.value))}
          onMouseUp={() => {
            if (pendingIntensity !== null && pendingIntensity !== cluster.intensity) {
              onSetIntensity(pendingIntensity);
            }
            setPendingIntensity(null);
          }}
          onTouchEnd={() => {
            if (pendingIntensity !== null && pendingIntensity !== cluster.intensity) {
              onSetIntensity(pendingIntensity);
            }
            setPendingIntensity(null);
          }}
          onBlur={() => {
            // Keyboard users adjust with arrows then Tab away — commit on blur.
            if (pendingIntensity !== null && pendingIntensity !== cluster.intensity) {
              onSetIntensity(pendingIntensity);
            }
            setPendingIntensity(null);
          }}
        />
      </div>

      {cluster.description && (
        <div className="mt-2">
          <button
            className="flex items-center gap-1 text-xs text-slate-400 hover:text-slate-200"
            onClick={() => setShowNarrative(!showNarrative)}
          >
            {showNarrative ? (
              <ChevronDown className="w-3 h-3" />
            ) : (
              <ChevronRight className="w-3 h-3" />
            )}
            Narrative
          </button>
          {showNarrative && (
            <p className="mt-1 text-xs text-slate-300 whitespace-pre-wrap max-h-48 overflow-y-auto border-l-2 border-slate-700 pl-3">
              {cluster.description}
            </p>
          )}
        </div>
      )}
    </div>
  );
}
