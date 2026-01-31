import type { ReactNode } from 'react';
import type { BadgeVariant } from '@/types/ui';

export type BadgeSize = 'sm' | 'md';

export interface BadgeProps {
  variant?: BadgeVariant | 'neutral';
  size?: BadgeSize;
  children: ReactNode;
  className?: string;
}

const variantStyles: Record<BadgeVariant | 'neutral', string> = {
  success: 'bg-green-500/15 text-green-400',
  warning: 'bg-yellow-500/15 text-yellow-400',
  danger: 'bg-red-500/15 text-red-400',
  primary: 'bg-primary-400/15 text-primary-400',
  purple: 'bg-purple-500/15 text-purple-400',
  default: 'bg-slate-600/30 text-slate-400',
  neutral: 'bg-slate-600/30 text-slate-400',
};

const sizeStyles: Record<BadgeSize, string> = {
  sm: 'px-2 py-0.5 text-[10px]',
  md: 'px-2.5 py-1 text-xs',
};

export function Badge({
  variant = 'default',
  size = 'md',
  children,
  className = '',
}: BadgeProps) {
  return (
    <span
      className={`
        inline-flex items-center
        rounded-full font-semibold uppercase tracking-wide
        ${variantStyles[variant]}
        ${sizeStyles[size]}
        ${className}
      `}
    >
      {children}
    </span>
  );
}

// Specialized status badges
export function StatusBadge({
  status,
  children,
}: {
  status: string;
  children?: ReactNode;
}) {
  const statusMap: Record<string, BadgeVariant | 'neutral'> = {
    loaded: 'success',
    ready: 'primary',
    attached: 'purple',
    downloading: 'warning',
    loading: 'warning',
    error: 'danger',
    active: 'success',
    connected: 'success',
    connecting: 'warning',
    disconnected: 'danger',
    inactive: 'default',
    warning: 'warning',
  };

  return (
    <Badge variant={statusMap[status] || 'default'}>
      {children || status}
    </Badge>
  );
}

export default Badge;
