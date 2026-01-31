import { Power, Pause, Play, Settings } from 'lucide-react';
import { Card, Button, Select, Badge } from '@components/common';
import { useUIStore } from '@stores/uiStore';

interface MonitoringControlsProps {
  isEnabled: boolean;
  topK: number;
  onToggle: () => void;
  onTopKChange: (topK: number) => void;
  isToggling?: boolean;
  disabled?: boolean;
}

const topKOptions = [
  { value: '5', label: 'Top 5' },
  { value: '10', label: 'Top 10' },
  { value: '20', label: 'Top 20' },
  { value: '50', label: 'Top 50' },
  { value: '100', label: 'Top 100' },
];

export function MonitoringControls({
  isEnabled,
  topK,
  onToggle,
  onTopKChange,
  isToggling,
  disabled,
}: MonitoringControlsProps) {
  const { isMonitoringPaused, toggleMonitoringPause } = useUIStore();

  return (
    <Card>
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-4">
          <div className="flex items-center gap-3">
            <div
              className={`
                w-3 h-3 rounded-full
                ${isEnabled && !isMonitoringPaused
                  ? 'bg-green-500 animate-pulse'
                  : isEnabled
                    ? 'bg-yellow-500'
                    : 'bg-slate-600'
                }
              `}
            />
            <div>
              <h3 className="text-sm font-medium text-slate-200">
                Feature Monitoring
              </h3>
              <p className="text-xs text-slate-500">
                {isEnabled
                  ? isMonitoringPaused
                    ? 'Paused'
                    : 'Capturing activations'
                  : 'Disabled'
                }
              </p>
            </div>
          </div>

          <Badge variant={isEnabled ? (isMonitoringPaused ? 'warning' : 'success') : 'neutral'}>
            {isEnabled ? (isMonitoringPaused ? 'PAUSED' : 'LIVE') : 'OFF'}
          </Badge>
        </div>

        <div className="flex items-center gap-3">
          <div className="flex items-center gap-2">
            <Settings className="w-4 h-4 text-slate-500" />
            <Select
              value={topK.toString()}
              onChange={(e) => onTopKChange(parseInt(e.target.value, 10))}
              options={topKOptions}
              className="w-28"
              disabled={disabled}
            />
          </div>

          {isEnabled && (
            <Button
              variant="secondary"
              size="sm"
              onClick={toggleMonitoringPause}
              leftIcon={isMonitoringPaused ? <Play className="w-4 h-4" /> : <Pause className="w-4 h-4" />}
            >
              {isMonitoringPaused ? 'Resume' : 'Pause'}
            </Button>
          )}

          <Button
            variant={isEnabled ? 'danger' : 'primary'}
            size="sm"
            onClick={onToggle}
            loading={isToggling}
            disabled={disabled}
            leftIcon={<Power className="w-4 h-4" />}
          >
            {isEnabled ? 'Disable' : 'Enable'}
          </Button>
        </div>
      </div>
    </Card>
  );
}
