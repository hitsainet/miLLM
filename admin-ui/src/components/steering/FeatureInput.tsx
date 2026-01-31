import { useState } from 'react';
import { Plus, Hash } from 'lucide-react';
import { Button, Input } from '@components/common';

interface FeatureInputProps {
  onAdd: (featureIndex: number, strength?: number) => void;
  disabled?: boolean;
  maxFeatureIndex?: number;
}

export function FeatureInput({
  onAdd,
  disabled,
  maxFeatureIndex,
}: FeatureInputProps) {
  const [featureIndex, setFeatureIndex] = useState('');
  const [error, setError] = useState<string | undefined>();

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();

    const index = parseInt(featureIndex, 10);

    if (isNaN(index) || index < 0) {
      setError('Please enter a valid positive number');
      return;
    }

    if (maxFeatureIndex !== undefined && index >= maxFeatureIndex) {
      setError(`Feature index must be less than ${maxFeatureIndex}`);
      return;
    }

    setError(undefined);
    onAdd(index);
    setFeatureIndex('');
  };

  return (
    <form onSubmit={handleSubmit} className="flex items-end gap-2">
      <div className="flex-1">
        <Input
          label="Add Feature by Index"
          type="number"
          min={0}
          max={maxFeatureIndex ? maxFeatureIndex - 1 : undefined}
          placeholder="e.g., 1234"
          value={featureIndex}
          onChange={(e) => {
            setFeatureIndex(e.target.value);
            setError(undefined);
          }}
          error={error}
          disabled={disabled}
          leftAddon={<Hash className="w-4 h-4 text-slate-500" />}
        />
      </div>
      <Button
        type="submit"
        variant="primary"
        disabled={disabled || !featureIndex}
        leftIcon={<Plus className="w-4 h-4" />}
      >
        Add
      </Button>
    </form>
  );
}
