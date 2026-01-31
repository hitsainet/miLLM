import {
  Server,
  Layers,
  Sliders,
  Activity,
  Cpu,
  HardDrive,
  Thermometer,
  Zap,
} from 'lucide-react';
import { useServerStore } from '@stores/serverStore';
import {
  StatusCard,
  SystemMetricCard,
  QuickActions,
  ActionButtons,
} from '@components/dashboard';
import type { StatusType } from '@components/dashboard/StatusCard';

export function DashboardPage() {
  const {
    loadedModel,
    attachedSAE,
    steeringState,
    monitoringConfig,
    systemMetrics,
    connectionStatus,
  } = useServerStore();

  const hasModel = loadedModel !== null;
  const hasSAE = attachedSAE !== null;
  const hasSteering = steeringState !== null && steeringState.enabled && (steeringState.features?.length || 0) > 0;

  const getModelStatus = (): { status: StatusType; text: string; details?: string } => {
    if (!hasModel) {
      return { status: 'neutral', text: 'No Model' };
    }
    return {
      status: 'success',
      text: 'Loaded',
      details: loadedModel?.name,
    };
  };

  const getSAEStatus = (): { status: StatusType; text: string; details?: string } => {
    if (!hasModel) {
      return { status: 'neutral', text: 'Waiting', details: 'Load a model first' };
    }
    if (!hasSAE) {
      return { status: 'warning', text: 'Not Attached' };
    }
    return {
      status: 'success',
      text: 'Attached',
      details: `Layer ${attachedSAE?.layer}`,
    };
  };

  const getSteeringStatus = (): { status: StatusType; text: string; details?: string } => {
    if (!hasSAE) {
      return { status: 'neutral', text: 'Waiting', details: 'Attach SAE first' };
    }
    if (!steeringState?.enabled) {
      return { status: 'warning', text: 'Disabled' };
    }
    const featureCount = steeringState?.features?.length || 0;
    if (featureCount === 0) {
      return { status: 'info', text: 'Enabled', details: 'No features configured' };
    }
    return {
      status: 'success',
      text: 'Active',
      details: `${featureCount} feature${featureCount !== 1 ? 's' : ''}`,
    };
  };

  const getMonitoringStatus = (): { status: StatusType; text: string; details?: string } => {
    if (!hasSAE) {
      return { status: 'neutral', text: 'Waiting', details: 'Attach SAE first' };
    }
    if (!monitoringConfig?.enabled) {
      return { status: 'warning', text: 'Disabled' };
    }
    return {
      status: 'success',
      text: 'Active',
      details: `Top ${monitoringConfig?.top_k || 10} features`,
    };
  };

  const modelStatus = getModelStatus();
  const saeStatus = getSAEStatus();
  const steeringStatus = getSteeringStatus();
  const monitoringStatus = getMonitoringStatus();

  const getGpuTempStatus = (): StatusType => {
    const temp = systemMetrics?.gpuTemperature || 0;
    if (temp >= 85) return 'error';
    if (temp >= 70) return 'warning';
    return 'success';
  };

  const getGpuUsageStatus = (): StatusType => {
    const usage = systemMetrics?.gpuUtilization || 0;
    if (usage >= 95) return 'warning';
    return 'success';
  };

  return (
    <div className="space-y-6">
      {/* System Status Cards */}
      <section>
        <h2 className="text-lg font-semibold text-slate-200 mb-4">System Status</h2>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
          <StatusCard
            title="Model"
            icon={<Server className="w-5 h-5" />}
            status={modelStatus.status}
            statusText={modelStatus.text}
            details={modelStatus.details}
          />
          <StatusCard
            title="SAE"
            icon={<Layers className="w-5 h-5" />}
            status={saeStatus.status}
            statusText={saeStatus.text}
            details={saeStatus.details}
          />
          <StatusCard
            title="Steering"
            icon={<Sliders className="w-5 h-5" />}
            status={steeringStatus.status}
            statusText={steeringStatus.text}
            details={steeringStatus.details}
          />
          <StatusCard
            title="Monitoring"
            icon={<Activity className="w-5 h-5" />}
            status={monitoringStatus.status}
            statusText={monitoringStatus.text}
            details={monitoringStatus.details}
          />
        </div>
      </section>

      {/* System Metrics */}
      <section>
        <h2 className="text-lg font-semibold text-slate-200 mb-4">System Metrics</h2>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <SystemMetricCard
            label="GPU Utilization"
            value={systemMetrics?.gpuUtilization ?? '--'}
            unit="%"
            icon={<Zap className="w-5 h-5" />}
            status={getGpuUsageStatus()}
          />
          <SystemMetricCard
            label="GPU Memory"
            value={systemMetrics?.gpuMemoryUsed
              ? `${(systemMetrics.gpuMemoryUsed / 1024).toFixed(1)}/${(systemMetrics.gpuMemoryTotal / 1024).toFixed(1)}`
              : '--'}
            unit="GB"
            icon={<HardDrive className="w-5 h-5" />}
            status="neutral"
          />
          <SystemMetricCard
            label="GPU Temp"
            value={systemMetrics?.gpuTemperature ?? '--'}
            unit="°C"
            icon={<Thermometer className="w-5 h-5" />}
            status={getGpuTempStatus()}
          />
          <SystemMetricCard
            label="CPU Usage"
            value={'--'}
            unit="%"
            icon={<Cpu className="w-5 h-5" />}
            status="neutral"
          />
        </div>
      </section>

      {/* Quick Start / Actions */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <QuickActions
          hasModel={hasModel}
          hasSAE={hasSAE}
          hasSteering={hasSteering}
        />
        <ActionButtons
          hasModel={hasModel}
          hasSAE={hasSAE}
        />
      </div>

      {/* Connection Status Banner */}
      {connectionStatus !== 'connected' && (
        <div className={`
          p-4 rounded-lg border
          ${connectionStatus === 'connecting'
            ? 'bg-yellow-500/10 border-yellow-500/30 text-yellow-400'
            : 'bg-red-500/10 border-red-500/30 text-red-400'
          }
        `}>
          <div className="flex items-center gap-3">
            <div className={`
              w-3 h-3 rounded-full
              ${connectionStatus === 'connecting' ? 'bg-yellow-500 animate-pulse' : 'bg-red-500'}
            `} />
            <span className="font-medium">
              {connectionStatus === 'connecting'
                ? 'Connecting to server...'
                : 'Disconnected from server'
              }
            </span>
          </div>
        </div>
      )}
    </div>
  );
}

export default DashboardPage;
