import { ExternalLink, Hash } from 'lucide-react';
import { featureColor } from '@/utils/featureColors';
import { Card, CardHeader } from '@components/common';
import type { FeatureActivation } from '@/types';

interface ActivationChartProps {
  activations: FeatureActivation[];
  maxValue?: number;
  title?: string;
  subtitle?: string;
  /** Base URL for Neuronpedia feature links, e.g. "https://neuronpedia.hitsai.net/" */
  neuronpediaBaseUrl?: string;
}

export function ActivationChart({
  activations,
  maxValue,
  title = 'Top Feature Activations',
  subtitle,
  neuronpediaBaseUrl = 'https://neuronpedia.hitsai.net/',
}: ActivationChartProps) {
  const maxActivation = maxValue || Math.max(...activations.map((a) => Math.abs(a.activation)), 1);

  return (
    <Card>
      <CardHeader
        title={title}
        subtitle={subtitle || `${activations.length} features`}
      />

      {activations.length > 0 ? (
        <div className="space-y-2 max-h-80 overflow-y-auto pr-1">
          {activations.map((item, index) => {
            const widthPercent = (Math.abs(item.activation) / maxActivation) * 100;
            // Key color to the FEATURE, not its rank — top-K reorders every
            // emission and rank-keyed colors flicker/misidentify (review find).
            const c = featureColor(item.feature_index);

            return (
              <div
                key={item.feature_index}
                className={`group flex items-center gap-3 px-2 py-1.5 rounded-lg border ${c.border} ${c.bg}`}
              >
                {/* Rank */}
                <span className="w-5 text-xs text-slate-500 text-right font-mono">
                  {index + 1}.
                </span>

                {/* Feature Index */}
                <div className="w-24 flex items-center gap-1.5">
                  <span className={`w-2 h-2 rounded-full ${c.dot} shrink-0`} />
                  <span className={`font-mono text-sm font-semibold ${c.text}`}>
                    #{item.feature_index}
                  </span>
                  <a
                    href={`${neuronpediaBaseUrl.replace(/\/$/, '')}/${item.feature_index}`}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="opacity-0 group-hover:opacity-100 transition-opacity"
                    title="View on Neuronpedia"
                  >
                    <ExternalLink className="w-3 h-3 text-slate-500 hover:text-primary-400" />
                  </a>
                </div>

                {/* Bar — feature-color fill (miStudio tile language) */}
                <div className="flex-1 h-4 bg-slate-800/50 rounded overflow-hidden">
                  <div
                    className="h-full transition-all duration-300"
                    style={{ width: `${widthPercent}%`, backgroundColor: c.accent }}
                  />
                </div>

                {/* Value */}
                <span className="w-16 text-right font-mono text-sm text-slate-300">
                  {item.activation.toFixed(3)}
                </span>
              </div>
            );
          })}
        </div>
      ) : (
        <div className="text-center py-8 text-slate-500">
          No activation data available
        </div>
      )}
    </Card>
  );
}

interface ActivationBarProps {
  featureIndex: number;
  activation: number;
  maxValue: number;
  rank?: number;
  label?: string;
}

export function ActivationBar({
  featureIndex,
  activation,
  maxValue,
  rank,
  label,
}: ActivationBarProps) {
  const widthPercent = (Math.abs(activation) / maxValue) * 100;

  const getBarColor = () => {
    if (activation > 0.7) return 'bg-green-500';
    if (activation > 0.4) return 'bg-primary-500';
    if (activation > 0.2) return 'bg-yellow-500';
    return 'bg-slate-500';
  };

  return (
    <div className="flex items-center gap-3 py-1">
      {rank !== undefined && (
        <span className="w-6 text-xs text-slate-500 text-right font-mono">
          {rank}.
        </span>
      )}

      <div className="w-20">
        <div className="flex items-center gap-1">
          <Hash className="w-3 h-3 text-primary-400" />
          <span className="font-mono text-sm text-primary-400">{featureIndex}</span>
        </div>
        {label && (
          <p className="text-xs text-slate-500 truncate" title={label}>
            {label}
          </p>
        )}
      </div>

      <div className="flex-1 h-4 bg-slate-800/50 rounded overflow-hidden">
        <div
          className={`h-full ${getBarColor()} transition-all duration-300`}
          style={{ width: `${widthPercent}%` }}
        />
      </div>

      <span className="w-14 text-right font-mono text-sm text-slate-300">
        {activation.toFixed(3)}
      </span>
    </div>
  );
}
