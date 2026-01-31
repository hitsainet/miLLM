import { Power, Trash2, Save, RotateCcw } from 'lucide-react';
import { Card, Button, Badge } from '@components/common';

interface SteeringControlsProps {
  isEnabled: boolean;
  featureCount: number;
  onToggle: () => void;
  onClear: () => void;
  onSaveProfile?: () => void;
  onReset?: () => void;
  isToggling?: boolean;
  isClearing?: boolean;
  disabled?: boolean;
}

export function SteeringControls({
  isEnabled,
  featureCount,
  onToggle,
  onClear,
  onSaveProfile,
  onReset,
  isToggling,
  isClearing,
  disabled,
}: SteeringControlsProps) {
  return (
    <Card>
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-4">
          <div className="flex items-center gap-3">
            <div
              className={`
                w-3 h-3 rounded-full
                ${isEnabled && featureCount > 0
                  ? 'bg-green-500 animate-pulse'
                  : isEnabled
                    ? 'bg-yellow-500'
                    : 'bg-slate-600'
                }
              `}
            />
            <div>
              <h3 className="text-sm font-medium text-slate-200">
                Feature Steering
              </h3>
              <p className="text-xs text-slate-500">
                {isEnabled
                  ? featureCount > 0
                    ? `Active with ${featureCount} feature${featureCount !== 1 ? 's' : ''}`
                    : 'Enabled, no features configured'
                  : 'Disabled'
                }
              </p>
            </div>
          </div>

          <Badge variant={isEnabled ? 'success' : 'neutral'}>
            {isEnabled ? 'ON' : 'OFF'}
          </Badge>
        </div>

        <div className="flex items-center gap-2">
          {onSaveProfile && featureCount > 0 && (
            <Button
              variant="secondary"
              size="sm"
              onClick={onSaveProfile}
              disabled={disabled}
              leftIcon={<Save className="w-4 h-4" />}
            >
              Save as Profile
            </Button>
          )}

          {onReset && featureCount > 0 && (
            <Button
              variant="ghost"
              size="sm"
              onClick={onReset}
              disabled={disabled}
              leftIcon={<RotateCcw className="w-4 h-4" />}
            >
              Reset
            </Button>
          )}

          {featureCount > 0 && (
            <Button
              variant="ghost"
              size="sm"
              onClick={onClear}
              loading={isClearing}
              disabled={disabled}
              className="text-slate-400 hover:text-red-400"
              leftIcon={<Trash2 className="w-4 h-4" />}
            >
              Clear All
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
