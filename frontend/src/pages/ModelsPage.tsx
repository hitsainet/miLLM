/**
 * ModelsPage - Main page for model management.
 */

import React, { useEffect } from 'react';
import { useModelStore } from '../stores/modelStore';
import { ModelList, DownloadForm } from '../components/models';
import { QuantizationType } from '../types';

export function ModelsPage() {
  const {
    models,
    loadedModelId,
    isLoading,
    downloadProgress,
    error,
    fetchModels,
    downloadModel,
    loadModel,
    unloadModel,
    deleteModel,
    cancelDownload,
    clearError,
    initSocketListeners,
  } = useModelStore();

  // Initialize socket listeners and fetch models on mount
  useEffect(() => {
    const cleanup = initSocketListeners();
    fetchModels();

    return cleanup;
  }, [initSocketListeners, fetchModels]);

  const handleDownload = async (repoId: string, quantization: QuantizationType) => {
    await downloadModel(repoId, quantization);
  };

  return (
    <div className="container mx-auto px-4 py-8">
      {/* Header */}
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-gray-900">Models</h1>
        <p className="mt-2 text-gray-600">
          Download and manage LLM models for mechanistic interpretability.
        </p>
      </div>

      {/* Error Banner */}
      {error && (
        <div className="mb-6 p-4 bg-red-50 border border-red-200 rounded-lg flex items-center justify-between">
          <p className="text-red-800">{error}</p>
          <button
            onClick={clearError}
            className="text-red-600 hover:text-red-800"
          >
            <svg className="h-5 w-5" fill="currentColor" viewBox="0 0 20 20">
              <path
                fillRule="evenodd"
                d="M4.293 4.293a1 1 0 011.414 0L10 8.586l4.293-4.293a1 1 0 111.414 1.414L11.414 10l4.293 4.293a1 1 0 01-1.414 1.414L10 11.414l-4.293 4.293a1 1 0 01-1.414-1.414L8.586 10 4.293 5.707a1 1 0 010-1.414z"
                clipRule="evenodd"
              />
            </svg>
          </button>
        </div>
      )}

      {/* Main Content */}
      <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
        {/* Download Form */}
        <div className="lg:col-span-1">
          <DownloadForm onDownload={handleDownload} isLoading={isLoading} />
        </div>

        {/* Model List */}
        <div className="lg:col-span-3">
          {isLoading && models.length === 0 ? (
            <div className="flex items-center justify-center py-12">
              <svg
                className="animate-spin h-8 w-8 text-blue-600"
                fill="none"
                viewBox="0 0 24 24"
              >
                <circle
                  className="opacity-25"
                  cx="12"
                  cy="12"
                  r="10"
                  stroke="currentColor"
                  strokeWidth="4"
                />
                <path
                  className="opacity-75"
                  fill="currentColor"
                  d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
                />
              </svg>
            </div>
          ) : (
            <ModelList
              models={models}
              downloadProgress={downloadProgress}
              loadedModelId={loadedModelId}
              onLoad={loadModel}
              onUnload={unloadModel}
              onDelete={deleteModel}
              onCancelDownload={cancelDownload}
            />
          )}
        </div>
      </div>
    </div>
  );
}
