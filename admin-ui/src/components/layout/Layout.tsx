import { Outlet } from 'react-router-dom';
import { Sidebar } from './Sidebar';
import { Header } from './Header';
import { ToastContainer } from '@/components/common';
import { useUIStore } from '@/stores/uiStore';

export function Layout() {
  const { sidebar } = useUIStore();
  const { collapsed } = sidebar;

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

        <main className="p-6">
          <Outlet />
        </main>
      </div>

      <ToastContainer />
    </div>
  );
}

export default Layout;
