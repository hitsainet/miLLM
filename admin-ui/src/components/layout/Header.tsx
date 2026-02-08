import { useLocation } from 'react-router-dom';
import { Wifi, WifiOff, Moon, Sun, Cpu, HardDrive, Thermometer, Server, Layers, Zap } from 'lucide-react';
import { useUIStore } from '@/stores/uiStore';
import { useServerStore } from '@/stores/serverStore';

const pageTitles: Record<string, string> = {
  '/': 'Dashboard',
  '/dashboard': 'Dashboard',
  '/models': 'Models',
  '/sae': 'Sparse Autoencoders',
  '/steering': 'Feature Steering',
  '/monitoring': 'Feature Monitoring',
  '/profiles': 'Steering Profiles',
  '/settings': 'Settings',
};

export function Header() {
  const location = useLocation();
  const { theme, toggleTheme } = useUIStore();
  const {
    connectionStatus,
    gpuMemoryUsed,
    gpuMemoryTotal,
    gpuUtilization,
    gpuTemperature,
    loadedModel,
    attachedSAE,
    steering,
  } = useServerStore();

  const pageTitle = pageTitles[location.pathname] || 'miLLM';
  const isConnected = connectionStatus === 'connected';
  const hasGPU = gpuMemoryTotal > 0;

  return (
    <header className="h-16 bg-slate-900/80 border-b border-slate-700/50 backdrop-blur-sm sticky top-0 z-30">
      <div className="h-full px-6 flex items-center justify-between">
        {/* Page Title */}
        <h1 className="text-xl font-semibold text-slate-100">{pageTitle}</h1>

        {/* Right Section */}
        <div className="flex items-center gap-3">
          {/* Model Status */}
          <div
            className={`
              hidden lg:flex items-center gap-2 px-2.5 py-1.5 rounded-lg text-xs font-medium
              ${loadedModel
                ? 'bg-green-500/10 text-green-400 border border-green-500/30'
                : 'bg-slate-700/50 text-slate-500 border border-slate-600/30'
              }
            `}
            title={loadedModel ? `Model: ${loadedModel.name}` : 'No model loaded'}
          >
            <Server className="w-3.5 h-3.5" />
            <span className="max-w-24 truncate">
              {loadedModel ? loadedModel.name : 'No Model'}
            </span>
          </div>

          {/* SAE Status */}
          <div
            className={`
              hidden lg:flex items-center gap-2 px-2.5 py-1.5 rounded-lg text-xs font-medium
              ${attachedSAE
                ? 'bg-purple-500/10 text-purple-400 border border-purple-500/30'
                : 'bg-slate-700/50 text-slate-500 border border-slate-600/30'
              }
            `}
            title={attachedSAE ? `SAE: ${attachedSAE.name}` : 'No SAE attached'}
          >
            <Layers className="w-3.5 h-3.5" />
            <span className="max-w-24 truncate">
              {attachedSAE ? attachedSAE.name : 'No SAE'}
            </span>
          </div>

          {/* Steering Status */}
          {steering.enabled && (steering.features?.length ?? 0) > 0 && (
            <div
              className="hidden lg:flex items-center gap-2 px-2.5 py-1.5 rounded-lg text-xs font-medium bg-amber-500/10 text-amber-400 border border-amber-500/30"
              title={`${steering.features.length} features being steered`}
            >
              <Zap className="w-3.5 h-3.5" />
              <span>{steering.features.length} Steering</span>
            </div>
          )}

          {/* Separator */}
          <div className="hidden md:block h-6 w-px bg-slate-700" />

          {/* System Metrics */}
          <div className="hidden md:flex items-center gap-3 text-xs">
            {/* GPU Utilization - Cyan */}
            <div
              className="flex items-center gap-1.5 text-cyan-400"
              title="GPU Utilization"
            >
              <Cpu className="w-3.5 h-3.5" />
              <span className="font-mono font-medium">
                {hasGPU ? `${gpuUtilization}%` : 'N/A'}
              </span>
            </div>
            {/* GPU Memory - Purple/Violet */}
            <div
              className="flex items-center gap-1.5 text-violet-400"
              title="GPU Memory"
            >
              <HardDrive className="w-3.5 h-3.5" />
              <span className="font-mono font-medium">
                {hasGPU
                  ? `${(gpuMemoryUsed / 1024).toFixed(1)}/${(gpuMemoryTotal / 1024).toFixed(1)} GB`
                  : 'No GPU'
                }
              </span>
            </div>
            {/* GPU Temperature - Dynamic color based on temp */}
            <div
              className={`flex items-center gap-1.5 ${
                gpuTemperature >= 80
                  ? 'text-red-400'
                  : gpuTemperature >= 65
                    ? 'text-amber-400'
                    : 'text-emerald-400'
              }`}
              title="GPU Temperature"
            >
              <Thermometer className="w-3.5 h-3.5" />
              <span className="font-mono font-medium">
                {hasGPU ? `${gpuTemperature}°C` : 'N/A'}
              </span>
            </div>
          </div>

          {/* Separator */}
          <div className="hidden md:block h-6 w-px bg-slate-700" />

          {/* Connection Status */}
          <div
            className={`
              flex items-center gap-2 px-3 py-1.5 rounded-lg text-xs font-medium
              ${isConnected
                ? 'bg-green-500/10 text-green-400 border border-green-500/30'
                : 'bg-red-500/10 text-red-400 border border-red-500/30'
              }
            `}
          >
            {isConnected ? (
              <>
                <Wifi className="w-3.5 h-3.5" />
                <span>Connected</span>
              </>
            ) : (
              <>
                <WifiOff className="w-3.5 h-3.5" />
                <span>Disconnected</span>
              </>
            )}
          </div>

          {/* Theme Toggle */}
          <button
            onClick={toggleTheme}
            className="p-2 text-slate-400 hover:text-slate-200 hover:bg-slate-800 rounded-lg transition-colors"
            title={`Switch to ${theme === 'dark' ? 'light' : 'dark'} mode`}
          >
            {theme === 'dark' ? (
              <Sun className="w-5 h-5" />
            ) : (
              <Moon className="w-5 h-5" />
            )}
          </button>
        </div>
      </div>
    </header>
  );
}

export default Header;
