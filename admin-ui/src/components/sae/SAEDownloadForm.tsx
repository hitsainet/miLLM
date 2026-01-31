import { useState } from 'react';
import { Download, Layers } from 'lucide-react';
import { Card, CardHeader, Button, Input, Select } from '@components/common';
import type { DownloadSAERequest } from '@/types';

interface SAEDownloadFormProps {
  onSubmit: (data: DownloadSAERequest) => void;
  isLoading?: boolean;
  availableLayers?: number[];
}

export function SAEDownloadForm({
  onSubmit,
  isLoading,
  availableLayers = [0, 6, 12, 18, 24],
}: SAEDownloadFormProps) {
  const [formData, setFormData] = useState<DownloadSAERequest>({
    repo_id: '',
    filename: '',
    layer: availableLayers[Math.floor(availableLayers.length / 2)] || 12,
  });
  const [errors, setErrors] = useState<Record<string, string>>({});

  const validateForm = (): boolean => {
    const newErrors: Record<string, string> = {};

    if (!formData.repo_id.trim()) {
      newErrors.repo_id = 'Repository ID is required';
    } else if (!formData.repo_id.includes('/')) {
      newErrors.repo_id = 'Invalid format. Use: owner/repo-name';
    }

    setErrors(newErrors);
    return Object.keys(newErrors).length === 0;
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (validateForm()) {
      onSubmit({
        ...formData,
        filename: formData.filename || undefined,
      });
    }
  };

  const layerOptions = availableLayers.map((layer) => ({
    value: layer.toString(),
    label: `Layer ${layer}`,
  }));

  return (
    <Card>
      <CardHeader
        title="Download SAE"
        subtitle="Download a Sparse Autoencoder from Hugging Face"
        icon={<Download className="w-5 h-5 text-primary-400" />}
      />
      <form onSubmit={handleSubmit} className="space-y-4">
        <Input
          label="Hugging Face Repository ID"
          placeholder="e.g., google/gemma-scope-2b-pt-res"
          value={formData.repo_id}
          onChange={(e) => setFormData({ ...formData, repo_id: e.target.value })}
          error={errors.repo_id}
          helperText="SAELens format repositories from Hugging Face"
        />

        <div className="grid grid-cols-2 gap-4">
          <Select
            label="Target Layer"
            value={(formData.layer ?? 12).toString()}
            onChange={(e) => setFormData({ ...formData, layer: parseInt(e.target.value, 10) })}
            options={layerOptions}
          />
          <Input
            label="Filename (Optional)"
            placeholder="Auto-detect from repo"
            value={formData.filename || ''}
            onChange={(e) => setFormData({ ...formData, filename: e.target.value })}
            helperText="Leave empty for auto-detection"
          />
        </div>

        <Button
          type="submit"
          variant="primary"
          loading={isLoading}
          leftIcon={<Layers className="w-4 h-4" />}
          className="w-full"
        >
          Download SAE
        </Button>
      </form>
    </Card>
  );
}
