import { useState } from 'react';
import { Server, Info, Play } from 'lucide-react';
import { useModels } from '@hooks/useModels';
import { useServerStore } from '@stores/serverStore';
import { ModelLoadForm, LoadedModelCard, ModelDetailsModal } from '@components/models';
import type { ModelLoadFormData } from '@components/models';
import { Card, CardHeader, Spinner, EmptyState, Badge } from '@components/common';
import type { ModelInfo } from '@/types';

export function ModelsPage() {
  const { loadedModel } = useServerStore();
  const {
    models,
    isLoading,
    downloadModel,
    isDownloading,
    load,
    isLoadingModel,
    unloadModel,
    isUnloading,
    delete: deleteModel,
    isDeleting,
    previewModel,
    isPreviewingModel,
    previewData,
    clearPreview,
  } = useModels();

  // State for selected model in modal
  const [selectedModel, setSelectedModel] = useState<ModelInfo | null>(null);
  const [isModalOpen, setIsModalOpen] = useState(false);

  const handleLoadModel = async (data: ModelLoadFormData) => {
    await downloadModel({
      source: 'huggingface',
      repo_id: data.repo_id,
      quantization: data.quantization,
      device: data.device,
      trust_remote_code: data.trust_remote_code,
      hf_token: data.hf_token,
    });
  };

  const handlePreview = async (repo_id: string) => {
    await previewModel(repo_id);
    setSelectedModel(null); // Clear any selected model
    setIsModalOpen(true);
  };

  const handleUnload = async () => {
    if (loadedModel && loadedModel.id !== undefined) {
      await unloadModel(loadedModel.id);
    }
  };

  const handleModelClick = (model: ModelInfo) => {
    setSelectedModel(model);
    clearPreview(); // Clear any preview data
    setIsModalOpen(true);
  };

  const handleCloseModal = () => {
    setIsModalOpen(false);
    setSelectedModel(null);
    clearPreview();
  };

  const handleLoadFromModal = (id: number) => {
    load(id);
  };

  const handleUnloadFromModal = (id: number) => {
    if (id === undefined || id === null) {
      console.error('Cannot unload: model ID is undefined');
      return;
    }
    unloadModel(id).then(() => {
      // Refresh selected model state
      setSelectedModel(null);
    });
  };

  const handleDeleteFromModal = (id: number) => {
    deleteModel(id);
    handleCloseModal();
  };

  const getStatusBadge = (status: ModelInfo['status']) => {
    switch (status) {
      case 'loaded':
        return <Badge variant="success">Loaded</Badge>;
      case 'ready':
        return <Badge variant="primary">Ready</Badge>;
      case 'downloading':
        return <Badge variant="warning">Downloading</Badge>;
      case 'loading':
        return <Badge variant="warning">Loading</Badge>;
      case 'error':
        return <Badge variant="danger">Error</Badge>;
      default:
        return <Badge>{status}</Badge>;
    }
  };

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-64">
        <Spinner size="lg" />
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Loaded Model Section */}
      {loadedModel ? (
        <LoadedModelCard
          model={loadedModel}
          onUnload={handleUnload}
          isUnloading={isUnloading}
        />
      ) : (
        <ModelLoadForm
          onSubmit={handleLoadModel}
          onPreview={handlePreview}
          isLoading={isDownloading || isLoadingModel}
          isPreviewLoading={isPreviewingModel}
        />
      )}

      {/* Downloaded Models List */}
      <Card>
        <CardHeader
          title="Downloaded Models"
          subtitle="Click on a model to view details and actions"
          icon={<Server className="w-5 h-5 text-slate-400" />}
        />

        {models && models.length > 0 ? (
          <div className="space-y-3">
            {models.map((model) => (
              <div
                key={model.id}
                onClick={() => handleModelClick(model)}
                className={`
                  flex items-center justify-between p-3 rounded-lg border cursor-pointer
                  transition-all duration-150
                  ${model.status === 'loaded'
                    ? 'bg-green-500/5 border-green-500/20 hover:bg-green-500/10'
                    : model.status === 'error'
                    ? 'bg-red-500/5 border-red-500/20 hover:bg-red-500/10'
                    : 'bg-slate-800/30 border-slate-700/50 hover:bg-slate-800/50 hover:border-slate-600/50'
                  }
                `}
              >
                <div className="flex items-center gap-3">
                  <div className={`
                    p-2 rounded-lg
                    ${model.status === 'loaded'
                      ? 'bg-green-500/10 text-green-400'
                      : model.status === 'error'
                      ? 'bg-red-500/10 text-red-400'
                      : 'bg-slate-700/50 text-slate-400'
                    }
                  `}>
                    <Server className="w-4 h-4" />
                  </div>
                  <div>
                    <h4 className="text-sm font-medium text-slate-200">{model.name}</h4>
                    <p className="text-xs text-slate-500">
                      {model.repo_id}
                      {model.params && ` • ${model.params}`}
                      {model.quantization && ` • ${model.quantization}`}
                    </p>
                  </div>
                </div>
                <div className="flex items-center gap-3">
                  {/* Download Progress */}
                  {model.status === 'downloading' && (
                    <div className="flex items-center gap-3">
                      <div className="w-24 bg-slate-700/50 rounded-full h-2 overflow-hidden">
                        <div
                          className="bg-primary-500 h-full rounded-full transition-all duration-300"
                          style={{ width: `${model.download_progress || 0}%` }}
                        />
                      </div>
                      <span className="text-xs text-primary-400 font-medium min-w-[3rem]">
                        {model.download_progress ? `${model.download_progress}%` : '0%'}
                      </span>
                      <Spinner size="sm" />
                    </div>
                  )}

                  {/* Loading indicator */}
                  {model.status === 'loading' && (
                    <div className="flex items-center gap-2">
                      <Spinner size="sm" />
                      <span className="text-xs text-yellow-400">Loading...</span>
                    </div>
                  )}

                  {/* Status Badge */}
                  {getStatusBadge(model.status)}

                  {/* Quick actions */}
                  {(model.status === 'ready' || model.status === 'error') && (
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        load(model.id);
                      }}
                      disabled={isLoadingModel || loadedModel !== null}
                      className={`
                        flex items-center gap-1 px-2 py-1 rounded text-xs font-medium
                        transition-colors
                        ${loadedModel !== null
                          ? 'text-slate-500 cursor-not-allowed'
                          : 'text-primary-400 hover:text-primary-300 hover:bg-primary-500/10'
                        }
                      `}
                      title={loadedModel !== null ? 'Unload current model first' : 'Load model'}
                    >
                      <Play className="w-3 h-3" />
                      Load
                    </button>
                  )}
                </div>
              </div>
            ))}
          </div>
        ) : (
          <EmptyState
            icon={<Server className="w-8 h-8" />}
            title="No models downloaded"
            description="Download a model from Hugging Face to get started"
          />
        )}
      </Card>

      {/* Info Section */}
      <Card padding="sm">
        <div className="flex items-start gap-3">
          <Info className="w-5 h-5 text-primary-400 flex-shrink-0 mt-0.5" />
          <div className="text-sm text-slate-400">
            <p className="mb-2">
              <strong className="text-slate-300">Supported Models:</strong> Any model compatible with Hugging Face Transformers library.
            </p>
            <p>
              <strong className="text-slate-300">Recommended:</strong> Google Gemma 2 models work well with Gemma-Scope SAEs for feature steering.
            </p>
          </div>
        </div>
      </Card>

      {/* Model Details Modal */}
      <ModelDetailsModal
        model={selectedModel}
        previewData={previewData}
        isOpen={isModalOpen}
        onClose={handleCloseModal}
        onLoad={handleLoadFromModal}
        onUnload={handleUnloadFromModal}
        onDelete={handleDeleteFromModal}
        isLoadingModel={isLoadingModel}
        isUnloading={isUnloading}
        isDeleting={isDeleting}
        loadedModel={loadedModel}
      />
    </div>
  );
}

export default ModelsPage;
