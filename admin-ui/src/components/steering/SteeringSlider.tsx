/**
 * SteeringSlider — a miStudio-style feature tile: colored card border/tint,
 * mono #index header with layer chip, boxed strength input with coef readout,
 * and a warning-zone gradient range slider.
 */

import { useEffect, useState } from 'react';
import { ExternalLink, Layers, X } from 'lucide-react';
import { featureColor, steeringTrackGradient } from '@/utils/featureColors';

interface SteeringSliderProps {
  featureIndex: number;
  /** Position in the list — drives the palette color (miStudio parity). */
  colorOrder?: number;
  /** Layer the attached SAE steers (shown as a chip). */
  layer?: number | null;
  /** Neuronpedia base derived from the attached SAE (utils/neuronpedia). */
  neuronpediaBase?: string;
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
  colorOrder = 0,
  layer,
  neuronpediaBase,
  strength,
  onStrengthChange,
  onRemove,
  disabled,
  label,
  min = -300,
  max = 300,
  step = 0.1,
}: SteeringSliderProps) {
  const [localStrength, setLocalStrength] = useState(strength);
  const [inputValue, setInputValue] = useState(String(strength));
  const c = featureColor(colorOrder);

  useEffect(() => {
    setLocalStrength(strength);
    setInputValue(String(strength));
  }, [strength]);

  const roundToStep = (value: number) => Math.round(value * 10) / 10;

  const commit = (value: number) => {
    const clamped = roundToStep(Math.max(min, Math.min(max, value)));
    setLocalStrength(clamped);
    setInputValue(String(clamped));
    onStrengthChange(clamped);
  };

  const handleInputBlur = () => {
    const value = parseFloat(inputValue);
    if (!isNaN(value)) {
      commit(value);
    } else {
      setInputValue(String(localStrength));
    }
  };

  return (
    <div className={`p-2 rounded-lg border ${c.border} ${c.bg} transition-colors`}>
      {/* Header: dot · #index · layer chip · label — link/remove right */}
      <div className="flex items-center gap-1.5 min-w-0">
        <span className={`w-2 h-2 rounded-full ${c.dot} shrink-0`} />
        <span className={`font-mono text-sm font-semibold ${c.text} shrink-0`}>
          #{featureIndex}
        </span>
        {layer != null && (
          <span className="flex items-center gap-0.5 text-[10px] text-slate-500 shrink-0">
            <Layers className="w-3 h-3" />
            L{layer}
          </span>
        )}
        {label && (
          <span className="text-xs text-slate-500 truncate" title={label}>
            {label}
          </span>
        )}
        <span className="flex-1" />
        <a
          href={`${(neuronpediaBase ?? 'https://neuronpedia.hitsai.net').replace(/\/$/, '')}/${featureIndex}`}
          target="_blank"
          rel="noopener noreferrer"
          title="View on Neuronpedia"
          className="shrink-0"
        >
          <ExternalLink className="w-3 h-3 text-slate-500 hover:text-slate-300" />
        </a>
        <button
          onClick={onRemove}
          disabled={disabled}
          title="Remove feature"
          aria-label={`Remove feature ${featureIndex}`}
          className="shrink-0 text-slate-500 hover:text-red-400 disabled:opacity-50"
        >
          <X className="w-3.5 h-3.5" />
        </button>
      </div>

      {/* Strength input + coef readout */}
      <div className="mt-1.5 flex items-center gap-2">
        <input
          type="number"
          step={step}
          min={min}
          max={max}
          value={inputValue}
          onChange={(e) => setInputValue(e.target.value)}
          onBlur={handleInputBlur}
          onKeyDown={(e) => e.key === 'Enter' && (e.target as HTMLInputElement).blur()}
          disabled={disabled}
          aria-label={`Strength for feature ${featureIndex}`}
          className="w-16 px-2 py-0.5 bg-slate-900/70 border border-slate-700 rounded text-center text-sm font-mono text-slate-100 focus:outline-none focus:border-slate-500 disabled:opacity-50 disabled:cursor-not-allowed"
        />
        <span className="text-xs text-slate-500">strength</span>
        <span className="flex-1" />
        <span className="text-xs font-mono text-slate-300">
          coef: <span className={c.text}>{localStrength}</span>
        </span>
      </div>

      {/* Warning-zone gradient slider (red extremes, feature color mid-band).
          Drag updates only local state; the API commit happens on release —
          committing per change-tick would flood PUT /saes/steering. */}
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={localStrength}
        onChange={(e) => {
          const v = roundToStep(parseFloat(e.target.value));
          setLocalStrength(v);
          setInputValue(String(v));
        }}
        onMouseUp={() => commit(localStrength)}
        onTouchEnd={() => commit(localStrength)}
        onBlur={() => commit(localStrength)}
        disabled={disabled}
        aria-label={`Strength slider for feature ${featureIndex}`}
        className="mt-1.5 w-full h-1.5 rounded-full appearance-none cursor-pointer disabled:cursor-not-allowed"
        style={{
          background: steeringTrackGradient(c.accent),
          accentColor: c.accent,
        }}
      />
    </div>
  );
}
