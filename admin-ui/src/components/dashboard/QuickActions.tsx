import { useNavigate } from 'react-router-dom';
import {
  Server,
  Layers,
  Sliders,
  Activity,
  Play,
} from 'lucide-react';
import { Card, CardHeader, Button } from '@components/common';

export interface QuickAction {
  id: string;
  label: string;
  description: string;
  icon: React.ReactNode;
  onClick: () => void;
  variant?: 'primary' | 'secondary';
  disabled?: boolean;
}

interface QuickActionsProps {
  hasModel: boolean;
  hasSAE: boolean;
  hasSteering: boolean;
}

export function QuickActions({
  hasModel,
  hasSAE,
  hasSteering,
}: QuickActionsProps) {
  const navigate = useNavigate();

  const getQuickStartStep = () => {
    if (!hasModel) return 1;
    if (!hasSAE) return 2;
    if (!hasSteering) return 3;
    return 4;
  };

  const currentStep = getQuickStartStep();

  const steps = [
    {
      step: 1,
      title: 'Load a Model',
      description: 'Start by loading a language model from Hugging Face',
      action: () => navigate('/models'),
      actionLabel: 'Go to Models',
      icon: <Server className="w-5 h-5" />,
    },
    {
      step: 2,
      title: 'Download & Attach SAE',
      description: 'Download a Sparse Autoencoder and attach it to your model',
      action: () => navigate('/sae'),
      actionLabel: 'Go to SAE',
      icon: <Layers className="w-5 h-5" />,
    },
    {
      step: 3,
      title: 'Configure Steering',
      description: 'Adjust feature strengths to steer model behavior',
      action: () => navigate('/steering'),
      actionLabel: 'Go to Steering',
      icon: <Sliders className="w-5 h-5" />,
    },
    {
      step: 4,
      title: 'Monitor Activations',
      description: 'View real-time feature activations during inference',
      action: () => navigate('/monitoring'),
      actionLabel: 'Go to Monitoring',
      icon: <Activity className="w-5 h-5" />,
    },
  ];

  return (
    <Card>
      <CardHeader
        title="Quick Start"
        subtitle="Follow these steps to get started with miLLM"
      />
      <div className="space-y-3">
        {steps.map((step) => {
          const isCompleted = step.step < currentStep;
          const isCurrent = step.step === currentStep;
          const isDisabled = step.step > currentStep;

          return (
            <div
              key={step.step}
              className={`
                flex items-center gap-4 p-3 rounded-lg border transition-colors
                ${isCompleted
                  ? 'bg-green-500/5 border-green-500/20'
                  : isCurrent
                    ? 'bg-primary-500/5 border-primary-500/30'
                    : 'bg-slate-800/30 border-slate-700/50 opacity-50'
                }
              `}
            >
              <div
                className={`
                  flex-shrink-0 w-8 h-8 rounded-full flex items-center justify-center text-sm font-semibold
                  ${isCompleted
                    ? 'bg-green-500 text-white'
                    : isCurrent
                      ? 'bg-primary-500 text-white'
                      : 'bg-slate-700 text-slate-400'
                  }
                `}
              >
                {isCompleted ? '✓' : step.step}
              </div>
              <div className="flex-1 min-w-0">
                <h4 className={`text-sm font-medium ${isDisabled ? 'text-slate-500' : 'text-slate-200'}`}>
                  {step.title}
                </h4>
                <p className={`text-xs ${isDisabled ? 'text-slate-600' : 'text-slate-400'}`}>
                  {step.description}
                </p>
              </div>
              <div className="flex-shrink-0">
                {isCurrent && (
                  <Button
                    size="sm"
                    variant="primary"
                    onClick={step.action}
                    rightIcon={<Play className="w-3 h-3" />}
                  >
                    {step.actionLabel}
                  </Button>
                )}
                {isCompleted && (
                  <span className="text-xs text-green-400 font-medium">Complete</span>
                )}
              </div>
            </div>
          );
        })}
      </div>
    </Card>
  );
}

interface ActionButtonsProps {
  hasModel: boolean;
  hasSAE: boolean;
}

export function ActionButtons({ hasModel, hasSAE }: ActionButtonsProps) {
  const navigate = useNavigate();

  return (
    <Card>
      <CardHeader title="Quick Actions" />
      <div className="grid grid-cols-2 gap-3">
        <Button
          variant="secondary"
          size="sm"
          leftIcon={<Server className="w-4 h-4" />}
          onClick={() => navigate('/models')}
        >
          {hasModel ? 'Manage Model' : 'Load Model'}
        </Button>
        <Button
          variant="secondary"
          size="sm"
          leftIcon={<Layers className="w-4 h-4" />}
          onClick={() => navigate('/sae')}
          disabled={!hasModel}
        >
          {hasSAE ? 'Manage SAE' : 'Attach SAE'}
        </Button>
        <Button
          variant="secondary"
          size="sm"
          leftIcon={<Sliders className="w-4 h-4" />}
          onClick={() => navigate('/steering')}
          disabled={!hasSAE}
        >
          Configure Steering
        </Button>
        <Button
          variant="secondary"
          size="sm"
          leftIcon={<Activity className="w-4 h-4" />}
          onClick={() => navigate('/monitoring')}
          disabled={!hasSAE}
        >
          View Monitoring
        </Button>
      </div>
    </Card>
  );
}
