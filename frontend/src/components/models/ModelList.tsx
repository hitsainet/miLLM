/**
 * ModelList component for displaying a grid of models.
 */

import React from 'react';
import { Model, DownloadProgress } from '../../types';
import { ModelCard } from './ModelCard';

interface ModelListProps {
  models: Model[];
  downloadProgress: Record<number, DownloadProgress>;
  loadedModelId: number | null;
  onLoad: (modelId: number) => void;
  onUnload: (modelId: number) => void;
  onDelete: (modelId: number) => void;
  onCancelDownload: (modelId: number) => void;
}

export function ModelList({
  models,
  downloadProgress,
  loadedModelId,
  onLoad,
  onUnload,
  onDelete,
  onCancelDownload,
}: ModelListProps) {
  if (models.length === 0) {
    return (
      <div className="text-center py-12">
        <svg
          className="mx-auto h-12 w-12 text-gray-400"
          fill="none"
          viewBox="0 0 24 24"
          stroke="currentColor"
        >
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            strokeWidth={2}
            d="M9 3v2m6-2v2M9 19v2m6-2v2M5 9H3m2 6H3m18-6h-2m2 6h-2M7 19h10a2 2 0 002-2V7a2 2 0 00-2-2H7a2 2 0 00-2 2v10a2 2 0 002 2zM9 9h6v6H9V9z"
          />
        </svg>
        <h3 className="mt-2 text-sm font-medium text-gray-900">No models</h3>
        <p className="mt-1 text-sm text-gray-500">
          Download a model from HuggingFace to get started.
        </p>
      </div>
    );
  }

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
      {models.map((model) => (
        <ModelCard
          key={model.id}
          model={model}
          downloadProgress={downloadProgress[model.id]}
          isLoaded={model.id === loadedModelId}
          onLoad={() => onLoad(model.id)}
          onUnload={() => onUnload(model.id)}
          onDelete={() => onDelete(model.id)}
          onCancelDownload={() => onCancelDownload(model.id)}
        />
      ))}
    </div>
  );
}
