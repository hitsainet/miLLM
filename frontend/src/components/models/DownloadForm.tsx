/**
 * DownloadForm component for downloading models from HuggingFace.
 */

import React, { useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle, Input, Select, Button } from '../common';
import { QuantizationType } from '../../types';

interface DownloadFormProps {
  onDownload: (repoId: string, quantization: QuantizationType) => Promise<void>;
  isLoading?: boolean;
}

const quantizationOptions = [
  { value: 'Q4', label: 'Q4 (4-bit) - Smallest, fastest' },
  { value: 'Q8', label: 'Q8 (8-bit) - Balanced' },
  { value: 'FP16', label: 'FP16 (16-bit) - Full precision' },
];

export function DownloadForm({ onDownload, isLoading = false }: DownloadFormProps) {
  const [repoId, setRepoId] = useState('');
  const [quantization, setQuantization] = useState<QuantizationType>('Q4');
  const [error, setError] = useState('');

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError('');

    if (!repoId.trim()) {
      setError('Please enter a repository ID');
      return;
    }

    // Validate repo ID format
    if (!repoId.includes('/')) {
      setError('Repository ID should be in format: owner/model-name');
      return;
    }

    try {
      await onDownload(repoId.trim(), quantization);
      setRepoId('');
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to start download');
    }
  };

  return (
    <Card>
      <CardHeader>
        <CardTitle>Download Model</CardTitle>
      </CardHeader>
      <CardContent>
        <form onSubmit={handleSubmit} className="space-y-4">
          <Input
            label="HuggingFace Repository"
            placeholder="google/gemma-2-2b"
            value={repoId}
            onChange={(e) => setRepoId(e.target.value)}
            error={error}
            helperText="Enter a HuggingFace model repository ID"
            disabled={isLoading}
          />

          <Select
            label="Quantization"
            options={quantizationOptions}
            value={quantization}
            onChange={(e) => setQuantization(e.target.value as QuantizationType)}
            disabled={isLoading}
          />

          <Button
            type="submit"
            variant="primary"
            isLoading={isLoading}
            className="w-full"
          >
            Download Model
          </Button>
        </form>
      </CardContent>
    </Card>
  );
}
