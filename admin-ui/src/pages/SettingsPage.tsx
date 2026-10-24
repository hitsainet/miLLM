import { useEffect, useState } from 'react';
import { Sun, Moon, Wifi, WifiOff, RefreshCw, Server, Info } from 'lucide-react';
import { useUIStore } from '@stores/uiStore';
import { useServerStore } from '@stores/serverStore';
import { socketClient } from '@services/socket';
import { Card, CardHeader, Button, Badge } from '@components/common';
import { EnvironmentSettings, HfTokenSettings } from '@components/settings';
import { useHfTokenStatus } from '@/hooks/useHfTokenStatus';
import { useEnvironmentSettings } from '@/hooks/useEnvironmentSettings';

export function SettingsPage() {
  const { theme, setTheme } = useUIStore();
  const { connectionStatus, serverUrl } = useServerStore();
  const hfToken = useHfTokenStatus();
  const environment = useEnvironmentSettings();

  // ⚠ THIS USED TO BE THE LITERAL `1.0.0` IN THE MARKUP, under a card headed "About the miLLM
  // server" — so it described nothing, matched no other version in the repo, and could never go
  // stale because it was never right. The server has served its own version at
  // /api/health/version all along. `null` renders as an em dash: "not reported" is a different
  // fact from a version number, and inventing one here is what caused this.
  const [serverVersion, setServerVersion] = useState<string | null>(null);
  useEffect(() => {
    let cancelled = false;
    fetch('/api/health/version')
      .then((r) => (r.ok ? r.json() : null))
      .then((d) => {
        if (!cancelled && d?.version) setServerVersion(d.version);
      })
      .catch(() => {
        /* leave it unreported rather than guess */
      });
    return () => {
      cancelled = true;
    };
  }, []);

  const handleReconnect = () => {
    socketClient.reconnect();
  };

  return (
    <div className="stack-y-6 max-w-3xl">
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
        <div className="stack-y-4">
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

      {/* Hugging Face token (Feature 34) */}
      <HfTokenSettings
        status={hfToken.status}
        onStatus={hfToken.setStatus}
        onRefresh={hfToken.refresh}
      />

      {/* Environment settings (Feature 35): the 30 that can change while miLLM runs */}
      <EnvironmentSettings state={environment} />

      {/* Server Info */}
      <Card>
        <CardHeader
          title="Server Information"
          subtitle="About the miLLM server"
          icon={<Server className="w-5 h-5 text-slate-400" />}
        />
        <div className="stack-y-3 text-sm">
          <div className="flex justify-between">
            <span className="text-slate-400">Version</span>
            <span className="text-slate-200 font-mono">{serverVersion ?? '—'}</span>
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
        <div className="text-sm text-slate-400 stack-y-3">
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
        <div className="stack-y-2 text-sm">
          <div className="flex justify-between py-1">
            <span className="text-slate-400">Go to Dashboard</span>
            <kbd className="px-2 py-1 bg-slate-800 rounded-sm text-xs font-mono text-slate-300">G D</kbd>
          </div>
          <div className="flex justify-between py-1">
            <span className="text-slate-400">Go to Models</span>
            <kbd className="px-2 py-1 bg-slate-800 rounded-sm text-xs font-mono text-slate-300">G M</kbd>
          </div>
          <div className="flex justify-between py-1">
            <span className="text-slate-400">Go to SAE</span>
            <kbd className="px-2 py-1 bg-slate-800 rounded-sm text-xs font-mono text-slate-300">G S</kbd>
          </div>
          <div className="flex justify-between py-1">
            <span className="text-slate-400">Go to Steering</span>
            <kbd className="px-2 py-1 bg-slate-800 rounded-sm text-xs font-mono text-slate-300">G T</kbd>
          </div>
          <div className="flex justify-between py-1">
            <span className="text-slate-400">Toggle Theme</span>
            <kbd className="px-2 py-1 bg-slate-800 rounded-sm text-xs font-mono text-slate-300">Ctrl+Shift+T</kbd>
          </div>
        </div>
      </Card>
    </div>
  );
}

export default SettingsPage;
