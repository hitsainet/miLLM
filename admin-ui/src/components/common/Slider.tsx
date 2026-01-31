import { useState, useCallback } from 'react';
import type { ChangeEvent } from 'react';

interface SliderProps {
  value: number;
  onChange: (value: number) => void;
  min?: number;
  max?: number;
  step?: number;
  label?: string;
  showValue?: boolean;
  formatValue?: (value: number) => string;
  disabled?: boolean;
  className?: string;
}

export function Slider({
  value,
  onChange,
  min = 0,
  max = 100,
  step = 1,
  label,
  showValue = true,
  formatValue = (v) => v.toFixed(1),
  disabled = false,
  className = '',
}: SliderProps) {
  const [localValue, setLocalValue] = useState(value);
  const [isDragging, setIsDragging] = useState(false);

  const handleChange = useCallback(
    (e: ChangeEvent<HTMLInputElement>) => {
      const newValue = parseFloat(e.target.value);
      setLocalValue(newValue);
    },
    []
  );

  const handleMouseDown = useCallback(() => {
    setIsDragging(true);
  }, []);

  const handleMouseUp = useCallback(() => {
    setIsDragging(false);
    onChange(localValue);
  }, [localValue, onChange]);

  // Sync local value with prop when not dragging
  const displayValue = isDragging ? localValue : value;

  // Calculate fill percentage
  const percentage = ((displayValue - min) / (max - min)) * 100;

  return (
    <div className={`w-full ${className}`}>
      {(label || showValue) && (
        <div className="flex items-center justify-between mb-2">
          {label && (
            <span className="text-xs font-medium text-slate-400">{label}</span>
          )}
          {showValue && (
            <span className="text-sm font-semibold text-slate-100 font-mono min-w-[50px] text-right">
              {displayValue > 0 ? '+' : ''}
              {formatValue(displayValue)}
            </span>
          )}
        </div>
      )}
      <div className="relative">
        <input
          type="range"
          min={min}
          max={max}
          step={step}
          value={displayValue}
          onChange={handleChange}
          onMouseDown={handleMouseDown}
          onMouseUp={handleMouseUp}
          onTouchStart={handleMouseDown}
          onTouchEnd={handleMouseUp}
          disabled={disabled}
          className={`
            w-full h-1.5 bg-slate-700/50 rounded-full appearance-none cursor-pointer
            disabled:opacity-50 disabled:cursor-not-allowed
          `}
          style={{
            background: `linear-gradient(to right,
              ${displayValue >= 0 ? '#06b6d4' : '#f87171'} ${Math.min(percentage, 50)}%,
              ${displayValue >= 0 ? '#06b6d4' : '#f87171'} ${Math.max(percentage, 50)}%,
              rgba(71, 85, 105, 0.5) ${Math.max(percentage, 50)}%,
              rgba(71, 85, 105, 0.5) 100%)`,
          }}
        />
      </div>
    </div>
  );
}

export default Slider;
