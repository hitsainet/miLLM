import { Server, Info, X, AlertTriangle, CheckCircle, HardDrive, Cpu } from 'lucide-react';
import { useModels } from '@hooks/useModels';
import { useServerStore } from '@stores/serverStore';
import { ModelLoadForm, LoadedModelCard } from '@components/models';
import type { ModelLoadFormData } from '@components/models';
import { Card, CardHeader, Spinner, EmptyState, Button } from '@components/common';

export function ModelsPage() {
  const { loadedModel } = useServerStore();
  const {
    models,
    isLoading,
    downloadModel,
    isDownloading,
    loadModel,
    isLoadingModel,
    unloadModel,
    isUnloading,
    previewModel,
    isPreviewingModel,
    previewData,
    clearPreview,
  } = useModels();

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
  };

  const handleUnload = async () => {
    if (loadedModel) {
      await unloadModel(loadedModel.id);
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

      {/* Model Preview Card */}
      {previewData && (
        <Card>
          <div className="flex items-start justify-between">
            <CardHeader
              title={`Preview: ${previewData.name}`}
              subtitle="Model information from Hugging Face"
              icon={<Info className="w-5 h-5 text-primary-400" />}
            />
            <button
              onClick={clearPreview}
              className="p-2 text-slate-400 hover:text-slate-200 hover:bg-slate-700/50 rounded-lg transition-colors"
            >
              <X className="w-4 h-4" />
            </button>
          </div>

          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-4">
            <div className="bg-slate-800/50 rounded-lg p-3">
              <p className="text-xs text-slate-500 mb-1">Parameters</p>
              <p className="text-lg font-semibold text-slate-200">{previewData.params || 'Unknown'}</p>
            </div>
            <div className="bg-slate-800/50 rounded-lg p-3">
              <p className="text-xs text-slate-500 mb-1">Architecture</p>
              <p className="text-lg font-semibold text-slate-200">{previewData.architecture || 'Unknown'}</p>
            </div>
            <div className="bg-slate-800/50 rounded-lg p-3">
              <p className="text-xs text-slate-500 mb-1">Gated</p>
              <p className={`text-lg font-semibold ${previewData.is_gated ? 'text-yellow-400' : 'text-green-400'}`}>
                {previewData.is_gated ? 'Yes (Token Required)' : 'No'}
              </p>
            </div>
            <div className="bg-slate-800/50 rounded-lg p-3">
              <p className="text-xs text-slate-500 mb-1">Trust Remote Code</p>
              <p className={`text-lg font-semibold ${previewData.requires_trust_remote_code ? 'text-yellow-400' : 'text-green-400'}`}>
                {previewData.requires_trust_remote_code ? 'Required' : 'Not Required'}
              </p>
            </div>
          </div>

          {previewData.requires_trust_remote_code && (
            <div className="flex items-center gap-2 p-3 bg-yellow-500/10 border border-yellow-500/30 rounded-lg mb-4">
              <AlertTriangle className="w-4 h-4 text-yellow-400 flex-shrink-0" />
              <p className="text-sm text-yellow-300">
                This model requires "Trust Remote Code" to be enabled. Make sure you trust the model source.
              </p>
            </div>
          )}

          {previewData.estimated_sizes && (
            <div>
              <h4 className="text-sm font-medium text-slate-300 mb-3">Estimated Requirements by Quantization</h4>
              <div className="grid grid-cols-3 gap-3">
                {(['Q4', 'Q8', 'FP16'] as const).map((quant) => {
                  const size = previewData.estimated_sizes?.[quant];
                  if (!size) return null;
                  return (
                    <div key={quant} className="bg-slate-800/30 border border-slate-700/50 rounded-lg p-3">
                      <div className="flex items-center justify-between mb-2">
                        <span className="text-sm font-medium text-slate-200">{quant}</span>
                        {quant === 'Q4' && (
                          <span className="text-xs bg-primary-500/20 text-primary-300 px-2 py-0.5 rounded">
                            Recommended
                          </span>
                        )}
                      </div>
                      <div className="space-y-1">
                        <div className="flex items-center gap-2 text-xs text-slate-400">
                          <HardDrive className="w-3 h-3" />
                          <span>Disk: {(size.disk_mb / 1024).toFixed(1)} GB</span>
                        </div>
                        <div className="flex items-center gap-2 text-xs text-slate-400">
                          <Cpu className="w-3 h-3" />
                          <span>VRAM: {(size.memory_mb / 1024).toFixed(1)} GB</span>
                        </div>
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          )}
        </Card>
      )}

      {/* Downloaded Models List */}
      <Card>
        <CardHeader
          title="Downloaded Models"
          subtitle="Models available locally"
          icon={<Server className="w-5 h-5 text-slate-400" />}
        />

        {models && models.length > 0 ? (
          <div className="space-y-3">
            {models.map((model) => (
              <div
                key={model.id}
                className={`
                  flex items-center justify-between p-3 rounded-lg border
                  ${model.status === 'loaded'
                    ? 'bg-green-500/5 border-green-500/20'
                    : 'bg-slate-800/30 border-slate-700/50'
                  }
                `}
              >
                <div className="flex items-center gap-3">
                  <div className={`
                    p-2 rounded-lg
                    ${model.status === 'loaded'
                      ? 'bg-green-500/10 text-green-400'
                      : 'bg-slate-700/50 text-slate-400'
                    }
                  `}>
                    <Server className="w-4 h-4" />
                  </div>
                  <div>
                    <h4 className="text-sm font-medium text-slate-200">{model.name}</h4>
                    <p className="text-xs text-slate-500">{model.repo_id}</p>
                  </div>
                </div>
                <div className="flex items-center gap-3">
                  {model.status === 'downloading' && (
                    <div className="flex items-center gap-3">
                      <div className="w-32 bg-slate-700/50 rounded-full h-2 overflow-hidden">
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
                  {model.status === 'loaded' && (
                    <span className="text-xs font-medium text-green-400 bg-green-500/10 px-2 py-1 rounded">
                      Loaded
                    </span>
                  )}
                  {model.status === 'ready' && (
                    <button
                      onClick={() => loadModel(model.id)}
                      disabled={isLoadingModel || loadedModel !== null}
                      className="text-xs text-primary-400 hover:text-primary-300 disabled:text-slate-500 disabled:cursor-not-allowed"
                    >
                      {loadedModel !== null ? 'Unload current first' : 'Load'}
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
    </div>
  );
}

export default ModelsPage;
