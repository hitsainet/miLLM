import { Trash2, Clock, Hash } from 'lucide-react';
import { Card, CardHeader, Button, EmptyState } from '@components/common';
import type { ActivationRecord } from '@/types';

interface ActivationHistoryProps {
  history: ActivationRecord[];
  onClear: () => void;
  isClearing?: boolean;
  maxItems?: number;
}

export function ActivationHistory({
  history,
  onClear,
  isClearing,
  maxItems = 50,
}: ActivationHistoryProps) {
  const displayHistory = history.slice(0, maxItems);

  const formatTime = (timestamp: string) => {
    const date = new Date(timestamp);
    return date.toLocaleTimeString('en-US', {
      hour: '2-digit',
      minute: '2-digit',
      second: '2-digit',
      hour12: false,
    });
  };

  return (
    <Card>
      <CardHeader
        title="Activation History"
        subtitle={`${history.length} record${history.length !== 1 ? 's' : ''}`}
        icon={<Clock className="w-5 h-5 text-slate-400" />}
        action={
          history.length > 0 && (
            <Button
              variant="ghost"
              size="sm"
              onClick={onClear}
              loading={isClearing}
              className="text-slate-400 hover:text-red-400"
              leftIcon={<Trash2 className="w-4 h-4" />}
            >
              Clear
            </Button>
          )
        }
      />

      {displayHistory.length > 0 ? (
        <div className="space-y-2 max-h-96 overflow-y-auto">
          {displayHistory.map((record, index) => (
            <div
              key={`${record.timestamp}-${index}`}
              className="p-3 bg-slate-800/30 rounded-lg border border-slate-700/50"
            >
              <div className="flex items-center justify-between mb-2">
                <span className="text-xs text-slate-500 font-mono">
                  {formatTime(record.timestamp)}
                </span>
                <span className="text-xs text-slate-500">
                  {record.activations.length} features
                </span>
              </div>

              <div className="flex flex-wrap gap-2">
                {record.activations.slice(0, 5).map((activation) => (
                  <div
                    key={activation.feature_index}
                    className="flex items-center gap-1 px-2 py-1 bg-slate-700/50 rounded text-xs"
                  >
                    <Hash className="w-3 h-3 text-primary-400" />
                    <span className="font-mono text-primary-400">
                      {activation.feature_index}
                    </span>
                    <span className="text-slate-400">:</span>
                    <span className="font-mono text-slate-300">
                      {activation.activation.toFixed(2)}
                    </span>
                  </div>
                ))}
                {record.activations.length > 5 && (
                  <span className="text-xs text-slate-500 self-center">
                    +{record.activations.length - 5} more
                  </span>
                )}
              </div>
            </div>
          ))}

          {history.length > maxItems && (
            <p className="text-xs text-slate-500 text-center pt-2">
              Showing {maxItems} of {history.length} records
            </p>
          )}
        </div>
      ) : (
        <EmptyState
          icon={<Clock className="w-8 h-8" />}
          title="No history yet"
          description="Activation records will appear here during inference"
        />
      )}
    </Card>
  );
}
