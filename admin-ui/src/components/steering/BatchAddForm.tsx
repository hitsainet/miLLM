import { useState } from 'react';
import { Plus, FileText } from 'lucide-react';
import { Button, Input, Modal } from '@components/common';

interface BatchAddFormProps {
  onBatchAdd: (features: Array<{ index: number; strength: number }>) => void;
  disabled?: boolean;
  maxFeatureIndex?: number;
}

export function BatchAddForm({
  onBatchAdd,
  disabled,
  maxFeatureIndex,
}: BatchAddFormProps) {
  const [isOpen, setIsOpen] = useState(false);
  const [input, setInput] = useState('');
  const [defaultStrength, setDefaultStrength] = useState('1.0');
  const [error, setError] = useState<string | undefined>();

  const parseInput = (): Array<{ index: number; strength: number }> | null => {
    const lines = input.trim().split(/[\n,]+/).filter(Boolean);
    const features: Array<{ index: number; strength: number }> = [];
    const strength = parseFloat(defaultStrength) || 1.0;

    for (const line of lines) {
      const trimmed = line.trim();
      if (!trimmed) continue;

      // Support formats: "1234", "1234:2.5", "1234=2.5"
      const match = trimmed.match(/^(\d+)(?:[:=](-?\d+(?:\.\d+)?))?$/);
      if (!match) {
        setError(`Invalid format: "${trimmed}". Use: 1234 or 1234:2.5`);
        return null;
      }

      const featureIndex = parseInt(match[1], 10);
      const featureStrength = match[2] ? parseFloat(match[2]) : strength;

      if (maxFeatureIndex !== undefined && featureIndex >= maxFeatureIndex) {
        setError(`Feature ${featureIndex} exceeds max index ${maxFeatureIndex - 1}`);
        return null;
      }

      features.push({ index: featureIndex, strength: featureStrength });
    }

    if (features.length === 0) {
      setError('Please enter at least one feature index');
      return null;
    }

    return features;
  };

  const handleSubmit = () => {
    const features = parseInput();
    if (features) {
      onBatchAdd(features);
      setInput('');
      setError(undefined);
      setIsOpen(false);
    }
  };

  return (
    <>
      <Button
        variant="secondary"
        size="sm"
        onClick={() => setIsOpen(true)}
        disabled={disabled}
        leftIcon={<FileText className="w-4 h-4" />}
      >
        Batch Add
      </Button>

      <Modal
        id="batch-add-features"
        title="Batch Add Features"
        isOpen={isOpen}
        onClose={() => {
          setIsOpen(false);
          setError(undefined);
        }}
        footer={
          <>
            <Button variant="secondary" onClick={() => setIsOpen(false)}>
              Cancel
            </Button>
            <Button
              variant="primary"
              onClick={handleSubmit}
              leftIcon={<Plus className="w-4 h-4" />}
            >
              Add Features
            </Button>
          </>
        }
      >
        <div className="space-y-4">
          <div>
            <label className="block text-sm font-medium text-slate-300 mb-2">
              Feature Indices
            </label>
            <textarea
              className="w-full h-32 px-3 py-2 bg-slate-800/50 border border-slate-700 rounded-lg text-slate-200 text-sm font-mono resize-none focus:outline-none focus:ring-2 focus:ring-primary-500/50 focus:border-primary-500"
              placeholder="Enter feature indices (one per line or comma-separated):&#10;1234&#10;5678:2.5&#10;9012=-1.0"
              value={input}
              onChange={(e) => {
                setInput(e.target.value);
                setError(undefined);
              }}
            />
            <p className="text-xs text-slate-500 mt-1">
              Optionally specify strength with : or = (e.g., 1234:2.5)
            </p>
          </div>

          <Input
            label="Default Strength"
            type="number"
            step="0.1"
            value={defaultStrength}
            onChange={(e) => setDefaultStrength(e.target.value)}
            helperText="Used for features without explicit strength"
          />

          {error && (
            <p className="text-sm text-red-400">{error}</p>
          )}
        </div>
      </Modal>
    </>
  );
}
