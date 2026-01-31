import { useEffect } from 'react';
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { Layout } from '@components/layout';
import ErrorBoundary from '@components/common/ErrorBoundary';
import {
  DashboardPage,
  ModelsPage,
  SAEPage,
  SteeringPage,
  MonitoringPage,
  ProfilesPage,
  SettingsPage,
} from '@pages/index';
import { useUIStore } from '@stores/uiStore';
import { socketClient } from '@services/socket';

// Create React Query client with defaults
const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      staleTime: 10 * 1000, // 10 seconds
      retry: 2,
      refetchOnWindowFocus: false,
    },
    mutations: {
      retry: 1,
    },
  },
});

function AppContent() {
  const { theme } = useUIStore();

  // Apply theme to document
  useEffect(() => {
    document.documentElement.classList.remove('light', 'dark');
    document.documentElement.classList.add(theme);
  }, [theme]);

  // Connect WebSocket on mount
  useEffect(() => {
    socketClient.connect();

    return () => {
      socketClient.disconnect();
    };
  }, []);

  return (
    <BrowserRouter>
      <Routes>
        <Route element={<Layout />}>
          <Route index element={<Navigate to="/dashboard" replace />} />
          <Route path="/dashboard" element={<DashboardPage />} />
          <Route path="/models" element={<ModelsPage />} />
          <Route path="/sae" element={<SAEPage />} />
          <Route path="/steering" element={<SteeringPage />} />
          <Route path="/monitoring" element={<MonitoringPage />} />
          <Route path="/profiles" element={<ProfilesPage />} />
          <Route path="/settings" element={<SettingsPage />} />
          <Route path="*" element={<Navigate to="/dashboard" replace />} />
        </Route>
      </Routes>
    </BrowserRouter>
  );
}

/**
 * Root application component.
 *
 * Wraps the application with:
 * - ErrorBoundary for catching and displaying errors
 * - QueryClientProvider for React Query data fetching
 *
 * @returns The root application component
 */
function App() {
  return (
    <ErrorBoundary
      onError={(error, errorInfo) => {
        // Log errors to console in production
        // In a real app, you might send this to an error tracking service
        console.error('Application error:', error);
        console.error('Error info:', errorInfo);
      }}
    >
      <QueryClientProvider client={queryClient}>
        <AppContent />
      </QueryClientProvider>
    </ErrorBoundary>
  );
}

export default App;
