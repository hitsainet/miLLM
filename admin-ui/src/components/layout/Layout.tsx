import { Outlet, useLocation } from 'react-router-dom';
import { Sidebar } from './Sidebar';
import { Header } from './Header';
import { ToastContainer } from '@/components/common';
import { useUIStore } from '@/stores/uiStore';
import { ApprovalsBanner } from '@/components/approvals/ApprovalsBanner';
import { useApprovalsPolling } from '@/stores/approvalsStore';

export function Layout() {
  const { sidebar } = useUIStore();
  const { collapsed } = sidebar;
  const { pathname } = useLocation();
  // ENH-008: the approval queue's ONE poll, for the banner, the badge and the page.
  useApprovalsPolling();

  return (
    <div className="min-h-screen">
      <Sidebar />

      <div
        className={`
          transition-all duration-300 ease-in-out
          ${collapsed ? 'ml-16' : 'ml-64'}
        `}
      >
        <Header />

        {/* ENH-008 D5: agent requests waiting for approval, at the top of every page. */}
        <ApprovalsBanner onApprovalsPage={pathname.replace(/\/+$/, '') === '/approvals'} />

        <main className="p-6">
          <Outlet />
        </main>
      </div>

      <ToastContainer />
    </div>
  );
}

export default Layout;
