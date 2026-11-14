import { create } from 'zustand';
import { persist } from 'zustand/middleware';
import type { Theme, Toast, ModalState, SidebarState } from '@/types/ui';

interface UIState {
  // Theme
  theme: Theme;

  // Sidebar
  sidebar: SidebarState;

  // Modal
  modal: ModalState;

  // Toasts
  toasts: Toast[];

  // Monitoring pause state
  monitoringPaused: boolean;
  isMonitoringPaused: boolean; // Alias
}

interface UIActions {
  // Theme actions
  setTheme: (theme: Theme) => void;
  toggleTheme: () => void;

  // Sidebar actions
  setSidebarCollapsed: (collapsed: boolean) => void;
  toggleSidebar: () => void;
  setMobileMenuOpen: (open: boolean) => void;
  toggleMobileMenu: () => void;

  // Modal actions
  openModal: (id: string, props?: Record<string, unknown>) => void;
  closeModal: () => void;

  // Toast actions
  addToast: (toast: Omit<Toast, 'id'>) => void;
  removeToast: (id: string) => void;
  clearToasts: () => void;

  // Monitoring pause actions
  setMonitoringPaused: (paused: boolean) => void;
  toggleMonitoringPaused: () => void;
  toggleMonitoringPause: () => void; // Alias
}

const generateId = () => Math.random().toString(36).substring(2, 9);

export const useUIStore = create<UIState & UIActions>()(
  persist(
    (set) => ({
      // Initial state
      theme: 'dark',
      sidebar: {
        collapsed: false,
        mobileOpen: false,
      },
      modal: {
        id: null,
        props: undefined,
      },
      toasts: [],
      monitoringPaused: false,
      isMonitoringPaused: false,

      // Theme actions
      setTheme: (theme) => set({ theme }),
      toggleTheme: () =>
        set((state) => ({
          theme: state.theme === 'dark' ? 'light' : 'dark',
        })),

      // Sidebar actions
      setSidebarCollapsed: (collapsed) =>
        set((state) => ({
          sidebar: { ...state.sidebar, collapsed },
        })),
      toggleSidebar: () =>
        set((state) => ({
          sidebar: { ...state.sidebar, collapsed: !state.sidebar.collapsed },
        })),
      setMobileMenuOpen: (mobileOpen) =>
        set((state) => ({
          sidebar: { ...state.sidebar, mobileOpen },
        })),
      toggleMobileMenu: () =>
        set((state) => ({
          sidebar: { ...state.sidebar, mobileOpen: !state.sidebar.mobileOpen },
        })),

      // Modal actions
      openModal: (id, props) => set({ modal: { id, props } }),
      closeModal: () => set({ modal: { id: null, props: undefined } }),

      // Toast actions
      addToast: (toast) =>
        set((state) => ({
          toasts: [...state.toasts, { ...toast, id: generateId() }],
        })),
      removeToast: (id) =>
        set((state) => ({
          toasts: state.toasts.filter((t) => t.id !== id),
        })),
      clearToasts: () => set({ toasts: [] }),

      // Monitoring pause actions
      setMonitoringPaused: (paused) =>
        set({ monitoringPaused: paused, isMonitoringPaused: paused }),
      toggleMonitoringPaused: () =>
        set((state) => ({
          monitoringPaused: !state.monitoringPaused,
          isMonitoringPaused: !state.isMonitoringPaused,
        })),
      toggleMonitoringPause: () =>
        set((state) => ({
          monitoringPaused: !state.monitoringPaused,
          isMonitoringPaused: !state.isMonitoringPaused,
        })),
    }),
    {
      name: 'millm-ui-preferences',
      partialize: (state) => ({
        theme: state.theme,
        sidebar: { collapsed: state.sidebar.collapsed, mobileOpen: false },
      }),
    }
  )
);

// Convenience hooks for common toast patterns
export const useToast = () => {
  const addToast = useUIStore((state) => state.addToast);

  return {
    success: (message: string, duration?: number) =>
      addToast({ type: 'success', message, duration }),
    error: (message: string, duration?: number) =>
      addToast({ type: 'error', message, duration }),
    warning: (message: string, duration?: number) =>
      addToast({ type: 'warning', message, duration }),
    info: (message: string, duration?: number) =>
      addToast({ type: 'info', message, duration }),
  };
};
