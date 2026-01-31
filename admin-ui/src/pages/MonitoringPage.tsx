import { useNavigate } from 'react-router-dom';
import { AlertCircle, Info } from 'lucide-react';
import { useMonitoring } from '@hooks/useMonitoring';
import { useServerStore } from '@stores/serverStore';
import { useUIStore } from '@stores/uiStore';
import {
  MonitoringControls,
  ActivationChart,
  ActivationHistory,
  StatisticsPanel,
} from '@components/monitoring';
import { Card, Spinner, Button } from '@components/common';

export function MonitoringPage() {
  const navigate = useNavigate();
  const { loadedModel, attachedSAE, monitoringConfig, latestActivations } = useServerStore();
  const { isMonitoringPaused } = useUIStore();
  const {
    history,
    statistics,
    isLoadingHistory,
    isLoadingStats,
    configureMonitoring,
    enableMonitoring,
    disableMonitoring,
    isEnabling,
    isDisabling,
    clearHistory,
    isClearing,
  } = useMonitoring();

  const isEnabled = monitoringConfig?.enabled || false;
  const topK = monitoringConfig?.top_k || 10;

  const handleToggle = async () => {
    if (isEnabled) {
      await disableMonitoring();
    } else {
      await enableMonitoring();
    }
  };

  const handleTopKChange = async (newTopK: number) => {
    await configureMonitoring({ top_k: newTopK });
  };

  const handleClearHistory = async () => {
    await clearHistory();
  };

  // No model loaded
  if (!loadedModel) {
    return (
      <Card className="border-yellow-500/30 bg-yellow-500/5">
        <div className="flex items-start gap-3">
          <AlertCircle className="w-5 h-5 text-yellow-400 flex-shrink-0 mt-0.5" />
          <div>
            <h3 className="text-sm font-medium text-yellow-400">No Model Loaded</h3>
            <p className="text-sm text-slate-400 mt-1">
              You need to load a model and attach an SAE before you can monitor activations.
            </p>
            <Button
              variant="primary"
              size="sm"
              className="mt-3"
              onClick={() => navigate('/models')}
            >
              Go to Models
            </Button>
          </div>
        </div>
      </Card>
    );
  }

  // No SAE attached
  if (!attachedSAE) {
    return (
      <Card className="border-yellow-500/30 bg-yellow-500/5">
        <div className="flex items-start gap-3">
          <AlertCircle className="w-5 h-5 text-yellow-400 flex-shrink-0 mt-0.5" />
          <div>
            <h3 className="text-sm font-medium text-yellow-400">No SAE Attached</h3>
            <p className="text-sm text-slate-400 mt-1">
              You need to attach an SAE to the loaded model before you can monitor activations.
            </p>
            <Button
              variant="primary"
              size="sm"
              className="mt-3"
              onClick={() => navigate('/sae')}
            >
              Go to SAE
            </Button>
          </div>
        </div>
      </Card>
    );
  }

  return (
    <div className="space-y-6">
      {/* Monitoring Controls */}
      <MonitoringControls
        isEnabled={isEnabled}
        topK={topK}
        onToggle={handleToggle}
        onTopKChange={handleTopKChange}
        isToggling={isEnabling || isDisabling}
      />

      {/* Live Activations */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <ActivationChart
          activations={latestActivations || []}
          title="Latest Activations"
          subtitle={isMonitoringPaused ? 'Paused' : 'Real-time'}
        />

        {isLoadingStats ? (
          <Card>
            <div className="flex items-center justify-center h-48">
              <Spinner size="lg" />
            </div>
          </Card>
        ) : (
          <StatisticsPanel statistics={statistics || []} />
        )}
      </div>

      {/* Activation History */}
      {isLoadingHistory ? (
        <Card>
          <div className="flex items-center justify-center h-48">
            <Spinner size="lg" />
          </div>
        </Card>
      ) : (
        <ActivationHistory
          history={history || []}
          onClear={handleClearHistory}
          isClearing={isClearing}
        />
      )}

      {/* Info Section */}
      <Card padding="sm">
        <div className="flex items-start gap-3">
          <Info className="w-5 h-5 text-primary-400 flex-shrink-0 mt-0.5" />
          <div className="text-sm text-slate-400">
            <p className="mb-2">
              <strong className="text-slate-300">Monitoring:</strong> Captures the top-K most activated features for each inference request.
            </p>
            <p>
              <strong className="text-slate-300">Tip:</strong> Use monitoring to understand which features are active and their typical activation ranges.
              This can help you decide which features to steer.
            </p>
          </div>
        </div>
      </Card>
    </div>
  );
}

export default MonitoringPage;
