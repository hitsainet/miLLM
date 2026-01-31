import { type ReactNode } from 'react';
import { Card } from '@components/common';
import { StatusBadge } from '@components/common/Badge';
import { Spinner } from '@components/common/Spinner';
import type { ConnectionStatus } from '@/types';

export type StatusType = 'success' | 'warning' | 'error' | 'info' | 'neutral';

export interface StatusCardProps {
  title: string;
  icon: ReactNode;
  status: StatusType;
  statusText: string;
  details?: string;
  isLoading?: boolean;
  children?: ReactNode;
  action?: ReactNode;
}

const statusColors: Record<StatusType, string> = {
  success: 'border-l-green-500',
  warning: 'border-l-yellow-500',
  error: 'border-l-red-500',
  info: 'border-l-primary-500',
  neutral: 'border-l-slate-500',
};

const statusBadgeMap: Record<StatusType, ConnectionStatus | 'ready' | 'active' | 'inactive' | 'warning'> = {
  success: 'connected',
  warning: 'warning',
  error: 'disconnected',
  info: 'ready',
  neutral: 'inactive',
};

export function StatusCard({
  title,
  icon,
  status,
  statusText,
  details,
  isLoading,
  children,
  action,
}: StatusCardProps) {
  return (
    <Card className={`border-l-4 ${statusColors[status]}`}>
      <div className="flex items-start justify-between">
        <div className="flex items-start gap-3">
          <div className="p-2 rounded-lg bg-slate-800/50 text-slate-400">
            {icon}
          </div>
          <div className="flex-1">
            <h3 className="text-sm font-medium text-slate-300">{title}</h3>
            <div className="mt-1 flex items-center gap-2">
              {isLoading ? (
                <Spinner size="sm" />
              ) : (
                <>
                  <StatusBadge status={statusBadgeMap[status] as ConnectionStatus}>
                    {statusText}
                  </StatusBadge>
                  {details && (
                    <span className="text-xs text-slate-500">{details}</span>
                  )}
                </>
              )}
            </div>
            {children && <div className="mt-3">{children}</div>}
          </div>
        </div>
        {action && <div>{action}</div>}
      </div>
    </Card>
  );
}

export interface SystemMetricCardProps {
  label: string;
  value: string | number;
  unit?: string;
  icon: ReactNode;
  status?: StatusType;
}

export function SystemMetricCard({
  label,
  value,
  unit,
  icon,
  status = 'neutral',
}: SystemMetricCardProps) {
  return (
    <Card padding="sm">
      <div className="flex items-center gap-3">
        <div className={`p-2 rounded-lg ${
          status === 'success' ? 'bg-green-500/10 text-green-400' :
          status === 'warning' ? 'bg-yellow-500/10 text-yellow-400' :
          status === 'error' ? 'bg-red-500/10 text-red-400' :
          'bg-slate-800/50 text-slate-400'
        }`}>
          {icon}
        </div>
        <div>
          <p className="text-xs text-slate-500 uppercase tracking-wider">{label}</p>
          <p className="text-lg font-semibold text-slate-200">
            {value}
            {unit && <span className="text-sm text-slate-400 ml-1">{unit}</span>}
          </p>
        </div>
      </div>
    </Card>
  );
}
