import { useLocation } from 'react-router-dom';
import { Wifi, WifiOff, Moon, Sun, Cpu, HardDrive, Thermometer } from 'lucide-react';
import { useUIStore } from '@/stores/uiStore';
import { useServerStore } from '@/stores/serverStore';
import { Badge } from '@/components/common';

const pageTitles: Record<string, string> = {
  '/': 'Dashboard',
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
  const { connectionStatus, gpuMemoryUsed, gpuMemoryTotal, gpuUtilization, gpuTemperature } =
    useServerStore();

  const pageTitle = pageTitles[location.pathname] || 'miLLM';
  const isConnected = connectionStatus === 'connected';

  return (
    <header className="h-16 bg-slate-900/80 border-b border-slate-700/50 backdrop-blur-sm sticky top-0 z-30">
      <div className="h-full px-6 flex items-center justify-between">
        {/* Page Title */}
        <h1 className="text-xl font-semibold text-slate-100">{pageTitle}</h1>

        {/* Right Section */}
        <div className="flex items-center gap-4">
          {/* System Metrics */}
          <div className="hidden md:flex items-center gap-4 text-xs text-slate-400">
            <div className="flex items-center gap-1.5">
              <Cpu className="w-3.5 h-3.5" />
              <span className="font-mono">{gpuUtilization}%</span>
            </div>
            <div className="flex items-center gap-1.5">
              <HardDrive className="w-3.5 h-3.5" />
              <span className="font-mono">
                {(gpuMemoryUsed / 1024).toFixed(1)}/{(gpuMemoryTotal / 1024).toFixed(1)} GB
              </span>
            </div>
            <div className="flex items-center gap-1.5">
              <Thermometer className="w-3.5 h-3.5" />
              <span className="font-mono">{gpuTemperature}°C</span>
            </div>
          </div>

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
