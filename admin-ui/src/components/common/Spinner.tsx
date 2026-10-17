import { Loader2 } from 'lucide-react';

interface SpinnerProps {
  size?: 'sm' | 'md' | 'lg';
  className?: string;
}

const sizeStyles = {
  sm: 'w-4 h-4',
  md: 'w-6 h-6',
  lg: 'w-8 h-8',
};

export function Spinner({ size = 'md', className = '' }: SpinnerProps) {
  return (
    <Loader2
      className={`animate-spin text-primary-400 ${sizeStyles[size]} ${className}`}
    />
  );
}

interface LoadingOverlayProps {
  message?: string;
}

export function LoadingOverlay({ message }: LoadingOverlayProps) {
  return (
    <div className="absolute inset-0 flex flex-col items-center justify-center bg-slate-900/80 backdrop-blur-sm rounded-xl z-10">
      <Spinner size="lg" />
      {message && <p className="mt-3 text-sm text-slate-400">{message}</p>}
    </div>
  );
}

export default Spinner;
