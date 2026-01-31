import type { ReactNode } from 'react';
import type { BadgeVariant } from '@/types/ui';

interface BadgeProps {
  variant?: BadgeVariant;
  children: ReactNode;
  className?: string;
}

const variantStyles: Record<BadgeVariant, string> = {
  success: 'bg-green-500/15 text-green-400',
  warning: 'bg-yellow-500/15 text-yellow-400',
  danger: 'bg-red-500/15 text-red-400',
  primary: 'bg-primary-400/15 text-primary-400',
  purple: 'bg-purple-500/15 text-purple-400',
  default: 'bg-slate-600/30 text-slate-400',
};

export function Badge({
  variant = 'default',
  children,
  className = '',
}: BadgeProps) {
  return (
    <span
      className={`
        inline-flex items-center px-2.5 py-1
        rounded-full text-xs font-semibold uppercase tracking-wide
        ${variantStyles[variant]}
        ${className}
      `}
    >
      {children}
    </span>
  );
}

// Specialized status badges
export function StatusBadge({ status }: { status: string }) {
  const statusMap: Record<string, BadgeVariant> = {
    loaded: 'success',
    ready: 'primary',
    attached: 'purple',
    downloading: 'warning',
    loading: 'warning',
    error: 'danger',
    active: 'success',
  };

  return <Badge variant={statusMap[status] || 'default'}>{status}</Badge>;
}

export default Badge;
