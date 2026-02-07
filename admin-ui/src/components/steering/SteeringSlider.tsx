import { useState, useEffect } from 'react';
import { X, Hash, ExternalLink } from 'lucide-react';
import { Button, Slider } from '@components/common';

interface SteeringSliderProps {
  featureIndex: number;
  strength: number;
  onStrengthChange: (strength: number) => void;
  onRemove: () => void;
  disabled?: boolean;
  label?: string;
  min?: number;
  max?: number;
  step?: number;
}

export function SteeringSlider({
  featureIndex,
  strength,
  onStrengthChange,
  onRemove,
  disabled,
  label,
  min = -200,
  max = 200,
  step = 0.1,
}: SteeringSliderProps) {
  const [localStrength, setLocalStrength] = useState(strength);
  const [inputValue, setInputValue] = useState(String(strength));

  useEffect(() => {
    setLocalStrength(strength);
    setInputValue(String(strength));
  }, [strength]);

  const roundToStep = (value: number) => Math.round(value * 10) / 10;

  const handleSliderChange = (value: number) => {
    // Only update local state during drag - no API call
    const rounded = roundToStep(value);
    setLocalStrength(rounded);
    setInputValue(String(rounded));
  };

  const handleSliderCommit = (value: number) => {
    // Call API only when drag ends
    const rounded = roundToStep(value);
    onStrengthChange(rounded);
  };

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setInputValue(e.target.value);
  };

  const handleInputBlur = () => {
    const value = parseFloat(inputValue);
    if (!isNaN(value)) {
      const clamped = Math.round(Math.max(min, Math.min(max, value)));
      setLocalStrength(clamped);
      setInputValue(String(clamped));
      onStrengthChange(clamped);
    } else {
      setInputValue(String(Math.round(localStrength)));
    }
  };

  const handleInputKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter') {
      (e.target as HTMLInputElement).blur();
    }
  };

  const getStrengthColor = () => {
    if (localStrength > 0) return 'text-green-400';
    if (localStrength < 0) return 'text-red-400';
    return 'text-slate-400';
  };

  return (
    <div className="group flex items-center gap-4 p-3 bg-slate-800/30 rounded-lg border border-slate-700/50 hover:border-slate-600/50 transition-colors">
      {/* Feature Info */}
      <div className="flex-shrink-0 min-w-[120px]">
        <div className="flex items-center gap-1.5">
          <Hash className="w-3.5 h-3.5 text-primary-400" />
          <span className="font-mono text-sm font-semibold text-primary-400">
            {featureIndex}
          </span>
          <a
            href={`https://www.neuronpedia.org/gemma-2-2b/${featureIndex}`}
            target="_blank"
            rel="noopener noreferrer"
            className="opacity-0 group-hover:opacity-100 transition-opacity"
            title="View on Neuronpedia"
          >
            <ExternalLink className="w-3 h-3 text-slate-500 hover:text-primary-400" />
          </a>
        </div>
        {label && (
          <p className="text-xs text-slate-500 truncate mt-0.5" title={label}>
            {label}
          </p>
        )}
      </div>

      {/* Slider */}
      <div className="flex-1">
        <Slider
          min={min}
          max={max}
          step={step}
          value={localStrength}
          onChange={handleSliderChange}
          onChangeEnd={handleSliderCommit}
          disabled={disabled}
          showValue={false}
        />
      </div>

      {/* Value Input */}
      <input
        type="number"
        step={step}
        min={min}
        max={max}
        value={inputValue}
        onChange={handleInputChange}
        onBlur={handleInputBlur}
        onKeyDown={handleInputKeyDown}
        disabled={disabled}
        className={`
          w-16 px-2 py-1 bg-slate-900/50 border border-slate-700 rounded text-center text-sm font-mono
          focus:outline-none focus:ring-1 focus:ring-primary-500 focus:border-primary-500
          disabled:opacity-50 disabled:cursor-not-allowed
          ${getStrengthColor()}
        `}
      />

      {/* Remove Button */}
      <Button
        variant="ghost"
        size="sm"
        onClick={onRemove}
        disabled={disabled}
        className="flex-shrink-0 opacity-0 group-hover:opacity-100 transition-opacity text-slate-400 hover:text-red-400"
      >
        <X className="w-4 h-4" />
      </Button>
    </div>
  );
}
