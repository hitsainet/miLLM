import { forwardRef, useId } from 'react';
import type { InputHTMLAttributes, ReactNode } from 'react';
import type { InputSize } from '@/types/ui';

export interface InputProps extends InputHTMLAttributes<HTMLInputElement> {
  label?: string;
  error?: string;
  helper?: string;
  helperText?: string; // Alias for helper
  leftAddon?: ReactNode;
  rightAddon?: ReactNode;
  inputSize?: InputSize;
}

const sizeStyles: Record<InputSize, string> = {
  sm: 'px-2.5 py-1.5 text-xs',
  md: 'px-3.5 py-2.5 text-sm',
  lg: 'px-4 py-3 text-base',
};

export const Input = forwardRef<HTMLInputElement, InputProps>(
  (
    {
      label,
      error,
      helper,
      helperText,
      leftAddon,
      rightAddon,
      inputSize = 'md',
      className = '',
      id,
      ...props
    },
    ref
  ) => {
    const generatedId = useId();
    const inputId = id || generatedId;

    return (
      <div className="w-full">
        {label && (
          <label
            htmlFor={inputId}
            className="block text-xs font-medium text-slate-400 mb-1.5"
          >
            {label}
          </label>
        )}
        <div className="relative flex items-center">
          {leftAddon && (
            <div className="absolute left-3 text-slate-500">{leftAddon}</div>
          )}
          <input
            ref={ref}
            id={inputId}
            className={`
              w-full bg-slate-800/50 border rounded-lg
              text-slate-200 placeholder-slate-500
              transition-colors duration-200
              focus:outline-none focus:border-primary-400/50
              ${error ? 'border-red-500/50' : 'border-slate-600/50'}
              ${leftAddon ? 'pl-10' : ''}
              ${rightAddon ? 'pr-10' : ''}
              ${sizeStyles[inputSize]}
              ${className}
            `}
            {...props}
          />
          {rightAddon && (
            <div className="absolute right-3 text-slate-500">{rightAddon}</div>
          )}
        </div>
        {error && <p className="mt-1 text-xs text-red-400">{error}</p>}
        {(helper || helperText) && !error && (
          <p className="mt-1 text-xs text-slate-500">{helper || helperText}</p>
        )}
      </div>
    );
  }
);

Input.displayName = 'Input';

export default Input;
