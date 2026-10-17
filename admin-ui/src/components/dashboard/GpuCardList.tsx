import { Cpu, HardDrive, Thermometer } from 'lucide-react';
import { Card } from '@components/common';
import type { GpuMetrics } from '@/types';

export interface GpuCardListProps {
  gpus: GpuMetrics[];
}

function tempClass(temperature: number): string {
  if (temperature >= 80) return 'text-red-400';
  if (temperature >= 65) return 'text-amber-400';
  return 'text-emerald-400';
}

/**
 * One tile per GPU. The aggregate cards above sum memory and average
 * utilization, which hides a full 12 GB card behind a half-empty 24 GB one.
 */
export function GpuCardList({ gpus }: GpuCardListProps) {
  if (!gpus.length) {
    return <p className="text-sm text-slate-500">No GPU reported.</p>;
  }

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
      {gpus.map((gpu) => {
        const percent = gpu.memory_total_mb
          ? Math.round((gpu.memory_used_mb / gpu.memory_total_mb) * 100)
          : 0;
        return (
          <Card key={gpu.uuid || gpu.index} padding="sm">
            <div data-testid={`gpu-card-${gpu.index}`} className="space-y-2">
              <div className="flex items-center justify-between">
                <span className="text-sm font-medium text-slate-200">{gpu.name}</span>
                <span className="text-xs text-slate-500 font-mono">GPU {gpu.index}</span>
              </div>
              <div className="flex items-center gap-4 text-xs">
                <span className="flex items-center gap-1 text-cyan-400" title="Utilization">
                  <Cpu className="w-3.5 h-3.5" />
                  {gpu.utilization}%
                </span>
                <span className="flex items-center gap-1 text-violet-400" title="Memory">
                  <HardDrive className="w-3.5 h-3.5" />
                  {(gpu.memory_used_mb / 1024).toFixed(1)}/{(gpu.memory_total_mb / 1024).toFixed(1)} GB
                </span>
                <span className={`flex items-center gap-1 ${tempClass(gpu.temperature)}`} title="Temperature">
                  <Thermometer className="w-3.5 h-3.5" />
                  {gpu.temperature}°C
                </span>
              </div>
              <div className="w-full bg-slate-700/50 rounded-full h-1.5 overflow-hidden">
                <div
                  className="bg-violet-500 h-full rounded-full"
                  style={{ width: `${percent}%` }}
                />
              </div>
            </div>
          </Card>
        );
      })}
    </div>
  );
}
