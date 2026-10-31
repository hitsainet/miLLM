import { NavLink } from 'react-router-dom';
import { BookOpen, ChevronLeft, ChevronRight, Zap } from 'lucide-react';
import { useUIStore } from '@/stores/uiStore';
import { useApprovalsStore } from '@/stores/approvalsStore';
import { MANUAL_BASE_URL } from '@/config/manual';
import { APP_VERSION } from '@/config/version';

import { bottomNavItems, navItems } from './navItems';

export function Sidebar() {
  const { sidebar, toggleSidebar } = useUIStore();
  const { collapsed } = sidebar;
  const waitingCount = useApprovalsStore((s) => s.waiting.length);

  return (
    <aside
      className={`
        fixed left-0 top-0 h-screen
        bg-slate-900/95 border-r border-slate-700/50
        backdrop-blur-xs z-40
        transition-all duration-300 ease-in-out
        ${collapsed ? 'w-16' : 'w-64'}
      `}
    >
      {/* Logo */}
      <div className="h-16 flex items-center px-4 border-b border-slate-700/50">
        <div className="flex items-center gap-3">
          <div className="w-9 h-9 bg-linear-to-br/srgb from-primary-500 to-primary-600 rounded-control flex items-center justify-center shrink-0">
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
        <nav className="p-3 stack-y-1 flex-1">
          {navItems.map((item) => (
            <NavLink
              key={item.id}
              to={item.path}
              className={({ isActive }) => `
                flex items-center gap-3 px-3 py-2.5 rounded-control
                transition-all duration-200
                ${isActive
                  ? 'bg-primary-500/10 text-primary-400'
                  : 'text-slate-400 hover:bg-slate-800 hover:text-slate-200'
                }
                ${collapsed ? 'justify-center' : ''}
              `}
              title={collapsed ? item.label : undefined}
            >
              <item.icon className="w-5 h-5 shrink-0" />
              {!collapsed && (
                <span className="text-sm font-medium">{item.label}</span>
              )}
              {item.id === 'approvals' && waitingCount > 0 && (
                // ENH-008 D7: the waiting count, hidden at 0. A solid amber pill, the same as
                // miDataworks and miStudio (operator 2026-10-28: a translucent tint read faint).
                <span
                  data-testid="nav-badge-approvals"
                  aria-label={`${waitingCount} waiting`}
                  title={`${waitingCount} waiting`}
                  className="ml-auto min-w-5 h-5 px-1.5 rounded-full bg-amber-500 text-white text-[11px] font-semibold leading-5 text-center"
                >
                  {waitingCount}
                </span>
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
                  flex-1 flex items-center gap-3 px-3 py-2.5 rounded-control
                  transition-all duration-200
                  ${isActive
                    ? 'bg-primary-500/10 text-primary-400'
                    : 'text-slate-400 hover:bg-slate-800 hover:text-slate-200'
                  }
                  ${collapsed ? 'justify-center' : ''}
                `}
                title={collapsed ? item.label : undefined}
              >
                <item.icon className="w-5 h-5 shrink-0" />
                {!collapsed && (
                  <span className="text-sm font-medium">{item.label}</span>
                )}
              </NavLink>
              {!collapsed && (
                // The address comes from config/manual.ts, which a test pins to the manual's own
                // docusaurus.config.ts. This was the literal https://hitsainet.github.io/miLLM/,
                // which only reached the manual through GitHub's redirect (ENH-004, D5).
                <a
                  href={MANUAL_BASE_URL}
                  target="_blank"
                  rel="noopener noreferrer"
                  title="User Manual"
                  aria-label="User Manual (opens in a new tab)"
                  data-testid="sidebar-manual"
                  className="shrink-0 p-2 rounded-control hover:bg-slate-700 transition-colors"
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
