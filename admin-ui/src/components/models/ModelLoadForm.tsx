import { useState } from 'react';
import { Download, Play, HelpCircle } from 'lucide-react';
import { Card, CardHeader, Button, Input, Select } from '@components/common';

export interface ModelLoadFormData {
  repo_id: string;
  quantization: 'FP32' | 'FP16' | 'Q8' | 'Q4' | 'Q2';
  device: 'auto' | 'cuda' | 'cpu';
  trust_remote_code: boolean;
  hf_token?: string;
}

interface ModelLoadFormProps {
  onSubmit: (data: ModelLoadFormData) => void;
  onPreview?: (repo_id: string) => void;
  isLoading?: boolean;
  isPreviewLoading?: boolean;
}

const quantizationOptions = [
  { value: 'Q4', label: 'Q4 - 4-bit (Recommended)' },
  { value: 'Q8', label: 'Q8 - 8-bit' },
  { value: 'FP16', label: 'FP16 - Half Precision' },
  { value: 'FP32', label: 'FP32 - Full Precision' },
  { value: 'Q2', label: 'Q2 - 2-bit' },
];

const deviceOptions = [
  { value: 'auto', label: 'Auto' },
  { value: 'cuda', label: 'CUDA (GPU)' },
  { value: 'cpu', label: 'CPU' },
];

export function ModelLoadForm({
  onSubmit,
  onPreview,
  isLoading,
  isPreviewLoading,
}: ModelLoadFormProps) {
  const [formData, setFormData] = useState<ModelLoadFormData>({
    repo_id: '',
    quantization: 'Q4',
    device: 'auto',
    trust_remote_code: false,
    hf_token: '',
  });
  const [errors, setErrors] = useState<Record<string, string>>({});

  const validateForm = (): boolean => {
    const newErrors: Record<string, string> = {};

    if (!formData.repo_id.trim()) {
      newErrors.repo_id = 'Repository ID is required';
    } else if (!formData.repo_id.includes('/')) {
      newErrors.repo_id = 'Invalid format. Use: owner/model-name';
    }

    setErrors(newErrors);
    return Object.keys(newErrors).length === 0;
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (validateForm()) {
      onSubmit({
        ...formData,
        hf_token: formData.hf_token || undefined,
      });
    }
  };

  const handlePreview = () => {
    if (formData.repo_id.trim() && formData.repo_id.includes('/')) {
      onPreview?.(formData.repo_id);
    }
  };

  return (
    <Card>
      <CardHeader
        title="Load Model"
        subtitle="Download and load a model from Hugging Face"
        icon={<Download className="w-5 h-5 text-primary-400" />}
      />
      <form onSubmit={handleSubmit} className="space-y-4">
        <Input
          label="Hugging Face Repository ID"
          placeholder="e.g., google/gemma-2-2b"
          value={formData.repo_id}
          onChange={(e) => setFormData({ ...formData, repo_id: e.target.value })}
          error={errors.repo_id}
          helperText="Enter the model repository in format: owner/model-name"
        />

        <div className="grid grid-cols-2 gap-4">
          <Select
            label="Quantization"
            value={formData.quantization}
            onChange={(e) => setFormData({ ...formData, quantization: e.target.value as ModelLoadFormData['quantization'] })}
            options={quantizationOptions}
          />
          <Select
            label="Device"
            value={formData.device}
            onChange={(e) => setFormData({ ...formData, device: e.target.value as ModelLoadFormData['device'] })}
            options={deviceOptions}
          />
        </div>

        <Input
          label="Hugging Face Token"
          type="password"
          placeholder="hf_xxxx... (optional)"
          value={formData.hf_token}
          onChange={(e) => setFormData({ ...formData, hf_token: e.target.value })}
          helperText="Required for gated models like Llama"
        />

        <div className="flex items-center gap-2">
          <input
            type="checkbox"
            id="trust_remote_code"
            checked={formData.trust_remote_code}
            onChange={(e) => setFormData({ ...formData, trust_remote_code: e.target.checked })}
            className="w-4 h-4 rounded border-slate-600 bg-slate-800 text-primary-500 focus:ring-primary-500 focus:ring-offset-0"
          />
          <label htmlFor="trust_remote_code" className="text-sm text-slate-300 flex items-center gap-1">
            Trust remote code
            <span className="text-yellow-500 text-xs">(Required for some models)</span>
          </label>
        </div>

        <div className="flex gap-3 pt-2">
          {onPreview && (
            <Button
              type="button"
              variant="secondary"
              onClick={handlePreview}
              loading={isPreviewLoading}
              disabled={!formData.repo_id.includes('/')}
              leftIcon={<HelpCircle className="w-4 h-4" />}
            >
              Preview
            </Button>
          )}
          <Button
            type="submit"
            variant="primary"
            loading={isLoading}
            leftIcon={<Play className="w-4 h-4" />}
            className="flex-1"
          >
            Download & Load Model
          </Button>
        </div>
      </form>
    </Card>
  );
}
