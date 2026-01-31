import type { ReactNode, HTMLAttributes } from 'react';

interface CardProps extends HTMLAttributes<HTMLDivElement> {
  children: ReactNode;
  padding?: 'none' | 'sm' | 'md' | 'lg';
}

const paddingStyles = {
  none: '',
  sm: 'p-3',
  md: 'p-5',
  lg: 'p-6',
};

export function Card({
  children,
  padding = 'md',
  className = '',
  ...props
}: CardProps) {
  return (
    <div
      className={`
        bg-slate-900/60 border border-primary-400/10 rounded-xl
        ${paddingStyles[padding]}
        ${className}
      `}
      {...props}
    >
      {children}
    </div>
  );
}

interface CardHeaderProps {
  children: ReactNode;
  icon?: ReactNode;
  action?: ReactNode;
  className?: string;
}

export function CardHeader({
  children,
  icon,
  action,
  className = '',
}: CardHeaderProps) {
  return (
    <div
      className={`flex items-center justify-between mb-4 ${className}`}
    >
      <div className="flex items-center gap-2 text-sm font-semibold text-slate-100">
        {icon && <span className="text-primary-400">{icon}</span>}
        {children}
      </div>
      {action}
    </div>
  );
}

export default Card;
