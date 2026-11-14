import { describe, it, expect, beforeEach } from 'vitest';
import { useUIStore } from '../uiStore';

describe('uiStore', () => {
  beforeEach(() => {
    // Reset store state before each test
    useUIStore.setState({
      theme: 'dark',
      sidebar: { collapsed: false, mobileOpen: false },
      modal: { id: null, props: undefined },
      toasts: [],
      monitoringPaused: false,
      isMonitoringPaused: false,
    });
  });

  describe('theme', () => {
    it('has dark theme by default', () => {
      const { theme } = useUIStore.getState();
      expect(theme).toBe('dark');
    });

    it('setTheme changes the theme', () => {
      const { setTheme } = useUIStore.getState();
      setTheme('light');
      expect(useUIStore.getState().theme).toBe('light');
    });

    it('toggleTheme toggles between dark and light', () => {
      const { toggleTheme } = useUIStore.getState();

      toggleTheme();
      expect(useUIStore.getState().theme).toBe('light');

      toggleTheme();
      expect(useUIStore.getState().theme).toBe('dark');
    });
  });

  describe('sidebar', () => {
    it('starts with sidebar expanded', () => {
      const { sidebar } = useUIStore.getState();
      expect(sidebar.collapsed).toBe(false);
    });

    it('setSidebarCollapsed changes collapsed state', () => {
      const { setSidebarCollapsed } = useUIStore.getState();
      setSidebarCollapsed(true);
      expect(useUIStore.getState().sidebar.collapsed).toBe(true);
    });

    it('toggleSidebar toggles collapsed state', () => {
      const { toggleSidebar } = useUIStore.getState();

      toggleSidebar();
      expect(useUIStore.getState().sidebar.collapsed).toBe(true);

      toggleSidebar();
      expect(useUIStore.getState().sidebar.collapsed).toBe(false);
    });

    it('setMobileMenuOpen changes mobileOpen state', () => {
      const { setMobileMenuOpen } = useUIStore.getState();
      setMobileMenuOpen(true);
      expect(useUIStore.getState().sidebar.mobileOpen).toBe(true);
    });

    it('toggleMobileMenu toggles mobileOpen state', () => {
      const { toggleMobileMenu } = useUIStore.getState();

      toggleMobileMenu();
      expect(useUIStore.getState().sidebar.mobileOpen).toBe(true);

      toggleMobileMenu();
      expect(useUIStore.getState().sidebar.mobileOpen).toBe(false);
    });
  });

  describe('modal', () => {
    it('starts with no modal open', () => {
      const { modal } = useUIStore.getState();
      expect(modal.id).toBeNull();
    });

    it('openModal opens modal with id', () => {
      const { openModal } = useUIStore.getState();
      openModal('test-modal');
      expect(useUIStore.getState().modal.id).toBe('test-modal');
    });

    it('openModal opens modal with id and props', () => {
      const { openModal } = useUIStore.getState();
      openModal('test-modal', { foo: 'bar' });
      const { modal } = useUIStore.getState();
      expect(modal.id).toBe('test-modal');
      expect(modal.props).toEqual({ foo: 'bar' });
    });

    it('closeModal closes the modal', () => {
      const { openModal, closeModal } = useUIStore.getState();
      openModal('test-modal');
      closeModal();
      expect(useUIStore.getState().modal.id).toBeNull();
    });
  });

  describe('toasts', () => {
    it('starts with no toasts', () => {
      const { toasts } = useUIStore.getState();
      expect(toasts).toHaveLength(0);
    });

    it('addToast adds a toast with generated id', () => {
      const { addToast } = useUIStore.getState();
      addToast({ type: 'success', message: 'Success!' });
      const { toasts } = useUIStore.getState();
      expect(toasts).toHaveLength(1);
      expect(toasts[0].message).toBe('Success!');
      expect(toasts[0].type).toBe('success');
      expect(toasts[0].id).toBeDefined();
    });

    it('removeToast removes a specific toast', () => {
      const { addToast, removeToast } = useUIStore.getState();
      addToast({ type: 'success', message: 'First' });
      addToast({ type: 'error', message: 'Second' });

      const { toasts: beforeToasts } = useUIStore.getState();
      expect(beforeToasts).toHaveLength(2);

      removeToast(beforeToasts[0].id);
      const { toasts: afterToasts } = useUIStore.getState();
      expect(afterToasts).toHaveLength(1);
      expect(afterToasts[0].message).toBe('Second');
    });

    it('clearToasts removes all toasts', () => {
      const { addToast, clearToasts } = useUIStore.getState();
      addToast({ type: 'success', message: 'First' });
      addToast({ type: 'error', message: 'Second' });

      clearToasts();
      const { toasts } = useUIStore.getState();
      expect(toasts).toHaveLength(0);
    });
  });

  describe('monitoring', () => {
    it('starts with monitoring not paused', () => {
      const { monitoringPaused, isMonitoringPaused } = useUIStore.getState();
      expect(monitoringPaused).toBe(false);
      expect(isMonitoringPaused).toBe(false);
    });

    it('setMonitoringPaused sets both flags', () => {
      const { setMonitoringPaused } = useUIStore.getState();
      setMonitoringPaused(true);
      const { monitoringPaused, isMonitoringPaused } = useUIStore.getState();
      expect(monitoringPaused).toBe(true);
      expect(isMonitoringPaused).toBe(true);
    });

    it('toggleMonitoringPaused toggles both flags', () => {
      const { toggleMonitoringPaused } = useUIStore.getState();

      toggleMonitoringPaused();
      let state = useUIStore.getState();
      expect(state.monitoringPaused).toBe(true);
      expect(state.isMonitoringPaused).toBe(true);

      toggleMonitoringPaused();
      state = useUIStore.getState();
      expect(state.monitoringPaused).toBe(false);
      expect(state.isMonitoringPaused).toBe(false);
    });

    it('toggleMonitoringPause is an alias for toggleMonitoringPaused', () => {
      const { toggleMonitoringPause } = useUIStore.getState();

      toggleMonitoringPause();
      expect(useUIStore.getState().monitoringPaused).toBe(true);
    });
  });
});
