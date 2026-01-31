// Theme types
export type Theme = 'dark' | 'light';

// Toast types
export type ToastType = 'success' | 'error' | 'warning' | 'info';

export interface Toast {
  id: string;
  type: ToastType;
  message: string;
  duration?: number;
}

// Navigation types
export interface NavItem {
  id: string;
  label: string;
  path: string;
  icon: string;
}

// Connection status
export type ConnectionStatus = 'connected' | 'disconnected' | 'connecting' | 'error';

// Modal types
export interface ModalState {
  id: string | null;
  props?: Record<string, unknown>;
}

// Form states
export interface FormField<T = string> {
  value: T;
  error?: string;
  touched: boolean;
}

// Component variant types
export type ButtonVariant = 'primary' | 'secondary' | 'danger' | 'ghost';
export type ButtonSize = 'sm' | 'md' | 'lg';

export type BadgeVariant = 'success' | 'warning' | 'danger' | 'primary' | 'purple' | 'default';

export type InputSize = 'sm' | 'md' | 'lg';

// Table types
export interface Column<T> {
  key: keyof T | string;
  header: string;
  width?: string;
  render?: (item: T) => React.ReactNode;
}

// Pagination
export interface PaginationState {
  page: number;
  pageSize: number;
  total: number;
}

// Sort
export interface SortState {
  field: string;
  direction: 'asc' | 'desc';
}

// Filter
export interface FilterState {
  [key: string]: string | number | boolean | null;
}

// Loading states
export interface AsyncState<T> {
  data: T | null;
  isLoading: boolean;
  error: string | null;
}

// Sidebar state
export interface SidebarState {
  collapsed: boolean;
  mobileOpen: boolean;
}
