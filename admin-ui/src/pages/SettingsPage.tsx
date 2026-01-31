import { Sun, Moon, Wifi, WifiOff, RefreshCw, Server, Info } from 'lucide-react';
import { useUIStore } from '@stores/uiStore';
import { useServerStore } from '@stores/serverStore';
import { socketClient } from '@services/socket';
import { Card, CardHeader, Button, Badge } from '@components/common';

export function SettingsPage() {
  const { theme, setTheme } = useUIStore();
  const { connectionStatus, serverUrl } = useServerStore();

  const handleReconnect = () => {
    socketClient.reconnect();
  };

  return (
    <div className="space-y-6 max-w-2xl">
      {/* Appearance */}
      <Card>
        <CardHeader
          title="Appearance"
          subtitle="Customize the look and feel"
          icon={theme === 'dark' ? <Moon className="w-5 h-5 text-slate-400" /> : <Sun className="w-5 h-5 text-yellow-400" />}
        />
        <div className="flex items-center justify-between">
          <div>
            <p className="text-sm text-slate-300">Theme</p>
            <p className="text-xs text-slate-500">Choose between light and dark mode</p>
          </div>
          <div className="flex items-center gap-2">
            <Button
              variant={theme === 'light' ? 'primary' : 'secondary'}
              size="sm"
              onClick={() => setTheme('light')}
              leftIcon={<Sun className="w-4 h-4" />}
            >
              Light
            </Button>
            <Button
              variant={theme === 'dark' ? 'primary' : 'secondary'}
              size="sm"
              onClick={() => setTheme('dark')}
              leftIcon={<Moon className="w-4 h-4" />}
            >
              Dark
            </Button>
          </div>
        </div>
      </Card>

      {/* Connection */}
      <Card>
        <CardHeader
          title="Connection"
          subtitle="WebSocket connection to the server"
          icon={connectionStatus === 'connected'
            ? <Wifi className="w-5 h-5 text-green-400" />
            : <WifiOff className="w-5 h-5 text-red-400" />
          }
        />
        <div className="space-y-4">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-slate-300">Status</p>
              <p className="text-xs text-slate-500">Current connection state</p>
            </div>
            <Badge
              variant={
                connectionStatus === 'connected' ? 'success' :
                connectionStatus === 'connecting' ? 'warning' : 'danger'
              }
            >
              {connectionStatus === 'connected' ? 'Connected' :
               connectionStatus === 'connecting' ? 'Connecting...' : 'Disconnected'}
            </Badge>
          </div>

          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-slate-300">Server URL</p>
              <p className="text-xs text-slate-500 font-mono">{serverUrl || 'Auto (relative)'}</p>
            </div>
            <Button
              variant="secondary"
              size="sm"
              onClick={handleReconnect}
              disabled={connectionStatus === 'connecting'}
              leftIcon={<RefreshCw className="w-4 h-4" />}
            >
              Reconnect
            </Button>
          </div>
        </div>
      </Card>

      {/* Server Info */}
      <Card>
        <CardHeader
          title="Server Information"
          subtitle="About the miLLM server"
          icon={<Server className="w-5 h-5 text-slate-400" />}
        />
        <div className="space-y-3 text-sm">
          <div className="flex justify-between">
            <span className="text-slate-400">Version</span>
            <span className="text-slate-200 font-mono">1.0.0</span>
          </div>
          <div className="flex justify-between">
            <span className="text-slate-400">API Base</span>
            <span className="text-slate-200 font-mono">/api</span>
          </div>
          <div className="flex justify-between">
            <span className="text-slate-400">OpenAI API</span>
            <span className="text-slate-200 font-mono">/v1</span>
          </div>
          <div className="flex justify-between">
            <span className="text-slate-400">WebSocket</span>
            <span className="text-slate-200 font-mono">/socket.io</span>
          </div>
        </div>
      </Card>

      {/* About */}
      <Card>
        <CardHeader
          title="About miLLM"
          icon={<Info className="w-5 h-5 text-primary-400" />}
        />
        <div className="text-sm text-slate-400 space-y-3">
          <p>
            <strong className="text-slate-300">miLLM</strong> is a Mechanistic Interpretability LLM Server
            that enables feature steering through Sparse Autoencoders (SAEs).
          </p>
          <p>
            It provides an OpenAI-compatible API for inference while allowing real-time
            manipulation of model behavior through feature activation adjustments.
          </p>
          <div className="pt-2">
            <a
              href="https://github.com/your-org/millm"
              target="_blank"
              rel="noopener noreferrer"
              className="text-primary-400 hover:text-primary-300"
            >
              View on GitHub →
            </a>
          </div>
        </div>
      </Card>

      {/* Keyboard Shortcuts */}
      <Card>
        <CardHeader
          title="Keyboard Shortcuts"
          subtitle="Quick navigation"
        />
        <div className="space-y-2 text-sm">
          <div className="flex justify-between py-1">
            <span className="text-slate-400">Go to Dashboard</span>
            <kbd className="px-2 py-1 bg-slate-800 rounded text-xs font-mono text-slate-300">G D</kbd>
          </div>
          <div className="flex justify-between py-1">
            <span className="text-slate-400">Go to Models</span>
            <kbd className="px-2 py-1 bg-slate-800 rounded text-xs font-mono text-slate-300">G M</kbd>
          </div>
          <div className="flex justify-between py-1">
            <span className="text-slate-400">Go to SAE</span>
            <kbd className="px-2 py-1 bg-slate-800 rounded text-xs font-mono text-slate-300">G S</kbd>
          </div>
          <div className="flex justify-between py-1">
            <span className="text-slate-400">Go to Steering</span>
            <kbd className="px-2 py-1 bg-slate-800 rounded text-xs font-mono text-slate-300">G T</kbd>
          </div>
          <div className="flex justify-between py-1">
            <span className="text-slate-400">Toggle Theme</span>
            <kbd className="px-2 py-1 bg-slate-800 rounded text-xs font-mono text-slate-300">Ctrl+Shift+T</kbd>
          </div>
        </div>
      </Card>
    </div>
  );
}

export default SettingsPage;
