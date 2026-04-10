import { NavLink } from 'react-router-dom';
import {
  Zap,
  LayoutDashboard,
  Server,
  Layers,
  Sliders,
  Activity,
  FileJson,
  Settings,
  ChevronLeft,
  ChevronRight,
  BookOpen,
} from 'lucide-react';
import { useUIStore } from '@/stores/uiStore';

const APP_VERSION = '0.5.0';

const navItems = [
  { id: 'dashboard', label: 'Dashboard', path: '/dashboard', icon: LayoutDashboard },
  { id: 'models', label: 'Models', path: '/models', icon: Server },
  { id: 'sae', label: 'SAEs', path: '/sae', icon: Layers },
  { id: 'steering', label: 'Steering', path: '/steering', icon: Sliders },
  { id: 'monitoring', label: 'Probe', path: '/monitoring', icon: Activity },
  { id: 'profiles', label: 'Profiles', path: '/profiles', icon: FileJson },
];

const bottomNavItems = [
  { id: 'settings', label: 'Settings', path: '/settings', icon: Settings },
];

export function Sidebar() {
  const { sidebar, toggleSidebar } = useUIStore();
  const { collapsed } = sidebar;

  return (
    <aside
      className={`
        fixed left-0 top-0 h-screen
        bg-slate-900/95 border-r border-slate-700/50
        backdrop-blur-sm z-40
        transition-all duration-300 ease-in-out
        ${collapsed ? 'w-16' : 'w-64'}
      `}
    >
      {/* Logo */}
      <div className="h-16 flex items-center px-4 border-b border-slate-700/50">
        <div className="flex items-center gap-3">
          <div className="w-9 h-9 bg-gradient-to-br from-primary-500 to-primary-600 rounded-lg flex items-center justify-center flex-shrink-0">
            <Zap className="w-5 h-5 text-white" />
          </div>
          {!collapsed && (
            <div className="overflow-hidden">
              <div className="text-lg font-bold text-slate-100">miLLM</div>
              <div className="text-[10px] text-slate-500 leading-tight">
                Interpretability Server
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Navigation - flex column to push settings to bottom */}
      <div className="flex flex-col h-[calc(100vh-4rem)]">
        <nav className="p-3 space-y-1 flex-1">
          {navItems.map((item) => (
            <NavLink
              key={item.id}
              to={item.path}
              className={({ isActive }) => `
                flex items-center gap-3 px-3 py-2.5 rounded-lg
                transition-all duration-200
                ${isActive
                  ? 'bg-primary-500/10 text-primary-400'
                  : 'text-slate-400 hover:bg-slate-800 hover:text-slate-200'
                }
                ${collapsed ? 'justify-center' : ''}
              `}
              title={collapsed ? item.label : undefined}
            >
              <item.icon className="w-5 h-5 flex-shrink-0" />
              {!collapsed && (
                <span className="text-sm font-medium">{item.label}</span>
              )}
            </NavLink>
          ))}
        </nav>

        {/* Bottom nav (Settings + Manual link) + version badge */}
        <div className="p-3 border-t border-slate-700/50">
          {bottomNavItems.map((item) => (
            <div key={item.id} className="flex items-center">
              <NavLink
                to={item.path}
                className={({ isActive }) => `
                  flex-1 flex items-center gap-3 px-3 py-2.5 rounded-lg
                  transition-all duration-200
                  ${isActive
                    ? 'bg-primary-500/10 text-primary-400'
                    : 'text-slate-400 hover:bg-slate-800 hover:text-slate-200'
                  }
                  ${collapsed ? 'justify-center' : ''}
                `}
                title={collapsed ? item.label : undefined}
              >
                <item.icon className="w-5 h-5 flex-shrink-0" />
                {!collapsed && (
                  <span className="text-sm font-medium">{item.label}</span>
                )}
              </NavLink>
              {!collapsed && (
                <a
                  href="https://hitsainet.github.io/miLLM/"
                  target="_blank"
                  rel="noopener noreferrer"
                  title="User Manual"
                  className="flex-shrink-0 p-2 rounded hover:bg-slate-700 transition-colors"
                >
                  <BookOpen className="w-4 h-4 text-slate-400 hover:text-primary-400" />
                </a>
              )}
            </div>
          ))}
          {!collapsed && (
            <div className="px-3 pt-1">
              <a
                href="https://github.com/hitsainet/millm/releases"
                target="_blank"
                rel="noopener noreferrer"
                className="text-[10px] text-slate-500 hover:text-emerald-400 transition-colors font-mono"
              >
                v{APP_VERSION}
              </a>
            </div>
          )}
        </div>
      </div>

      {/* Collapse Toggle */}
      <button
        onClick={toggleSidebar}
        className="absolute -right-3 top-20 w-6 h-6 bg-slate-800 border border-slate-700 rounded-full flex items-center justify-center text-slate-400 hover:text-slate-200 hover:bg-slate-700 transition-colors"
      >
        {collapsed ? (
          <ChevronRight className="w-3 h-3" />
        ) : (
          <ChevronLeft className="w-3 h-3" />
        )}
      </button>
    </aside>
  );
}

export default Sidebar;
