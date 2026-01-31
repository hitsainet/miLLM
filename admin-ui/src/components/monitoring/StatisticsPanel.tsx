import { TrendingUp, TrendingDown, Activity, Hash } from 'lucide-react';
import { Card, CardHeader, EmptyState } from '@components/common';
import type { FeatureStatistics } from '@/types';

interface StatisticsPanelProps {
  statistics: FeatureStatistics[];
  title?: string;
}

export function StatisticsPanel({
  statistics,
  title = 'Feature Statistics',
}: StatisticsPanelProps) {
  if (statistics.length === 0) {
    return (
      <Card>
        <CardHeader
          title={title}
          icon={<Activity className="w-5 h-5 text-slate-400" />}
        />
        <EmptyState
          icon={<Activity className="w-8 h-8" />}
          title="No statistics available"
          description="Statistics will appear after processing some requests"
        />
      </Card>
    );
  }

  return (
    <Card>
      <CardHeader
        title={title}
        subtitle={`${statistics.length} feature${statistics.length !== 1 ? 's' : ''} tracked`}
        icon={<Activity className="w-5 h-5 text-slate-400" />}
      />

      <div className="space-y-3">
        {statistics.map((stat) => (
          <div
            key={stat.feature_index}
            className="p-3 bg-slate-800/30 rounded-lg border border-slate-700/50"
          >
            <div className="flex items-center justify-between mb-3">
              <div className="flex items-center gap-2">
                <Hash className="w-4 h-4 text-primary-400" />
                <span className="font-mono text-sm font-semibold text-primary-400">
                  {stat.feature_index}
                </span>
              </div>
              <span className="text-xs text-slate-500">
                {stat.count} samples
              </span>
            </div>

            <div className="grid grid-cols-4 gap-3">
              <StatValue
                label="Mean"
                value={stat.mean}
                icon={<Activity className="w-3 h-3" />}
              />
              <StatValue
                label="Std Dev"
                value={stat.std}
                icon={<Activity className="w-3 h-3" />}
              />
              <StatValue
                label="Min"
                value={stat.min}
                icon={<TrendingDown className="w-3 h-3" />}
                colorClass="text-red-400"
              />
              <StatValue
                label="Max"
                value={stat.max}
                icon={<TrendingUp className="w-3 h-3" />}
                colorClass="text-green-400"
              />
            </div>
          </div>
        ))}
      </div>
    </Card>
  );
}

interface StatValueProps {
  label: string;
  value: number;
  icon?: React.ReactNode;
  colorClass?: string;
}

function StatValue({ label, value, icon, colorClass = 'text-slate-300' }: StatValueProps) {
  return (
    <div className="text-center">
      <div className="flex items-center justify-center gap-1 text-slate-500 mb-1">
        {icon}
        <span className="text-xs">{label}</span>
      </div>
      <span className={`font-mono text-sm font-semibold ${colorClass}`}>
        {value.toFixed(3)}
      </span>
    </div>
  );
}

interface CompactStatsProps {
  mean: number;
  std: number;
  min: number;
  max: number;
}

export function CompactStats({ mean, std, min, max }: CompactStatsProps) {
  return (
    <div className="flex items-center gap-4 text-xs">
      <span className="text-slate-500">
        Mean: <span className="text-slate-300 font-mono">{mean.toFixed(3)}</span>
      </span>
      <span className="text-slate-500">
        Std: <span className="text-slate-300 font-mono">{std.toFixed(3)}</span>
      </span>
      <span className="text-slate-500">
        Range: <span className="text-red-400 font-mono">{min.toFixed(3)}</span>
        {' - '}
        <span className="text-green-400 font-mono">{max.toFixed(3)}</span>
      </span>
    </div>
  );
}
