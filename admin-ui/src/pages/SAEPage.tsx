import { useState } from 'react';
import { AlertCircle, Info } from 'lucide-react';
import { useSAE } from '@hooks/useSAE';
import { useServerStore } from '@stores/serverStore';
import {
  SAEDownloadForm,
  SAEList,
  AttachedSAECard,
} from '@components/sae';
import { Card, Spinner } from '@components/common';
import type { DownloadSAERequest } from '@/types';

export function SAEPage() {
  const { loadedModel, attachedSAE } = useServerStore();
  const {
    saes,
    isLoading,
    downloadSAE,
    isDownloading,
    attachSAE,
    detachSAE,
    isDetaching,
    deleteSAE,
  } = useSAE();

  const [attachingId, setAttachingId] = useState<number | undefined>();
  const [deletingId, setDeletingId] = useState<number | undefined>();

  const handleDownload = async (data: DownloadSAERequest) => {
    await downloadSAE(data);
  };

  const handleAttach = async (saeId: number) => {
    setAttachingId(saeId);
    try {
      await attachSAE(saeId);
    } finally {
      setAttachingId(undefined);
    }
  };

  const handleDetach = async () => {
    await detachSAE();
  };

  const handleDelete = async (saeId: number) => {
    setDeletingId(saeId);
    try {
      await deleteSAE(saeId);
    } finally {
      setDeletingId(undefined);
    }
  };

  // Model not loaded warning
  if (!loadedModel) {
    return (
      <div className="space-y-6">
        <Card className="border-yellow-500/30 bg-yellow-500/5">
          <div className="flex items-start gap-3">
            <AlertCircle className="w-5 h-5 text-yellow-400 flex-shrink-0 mt-0.5" />
            <div>
              <h3 className="text-sm font-medium text-yellow-400">No Model Loaded</h3>
              <p className="text-sm text-slate-400 mt-1">
                You need to load a model before you can attach an SAE.
                Go to the Models page to load a model first.
              </p>
            </div>
          </div>
        </Card>

        <SAEDownloadForm
          onSubmit={handleDownload}
          isLoading={isDownloading}
        />

        {isLoading ? (
          <div className="flex items-center justify-center h-32">
            <Spinner size="lg" />
          </div>
        ) : (
          <SAEList
            saes={saes || []}
            attachedSAEId={attachedSAE?.id}
            onAttach={handleAttach}
            onDetach={handleDetach}
            onDelete={handleDelete}
            attachingId={attachingId}
            isDetaching={isDetaching}
            deletingId={deletingId}
            canAttach={false}
          />
        )}
      </div>
    );
  }

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-64">
        <Spinner size="lg" />
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Attached SAE Card */}
      {attachedSAE && (
        <AttachedSAECard
          sae={attachedSAE}
          onDetach={handleDetach}
          isDetaching={isDetaching}
        />
      )}

      {/* Download Form */}
      <SAEDownloadForm
        onSubmit={handleDownload}
        isLoading={isDownloading}
      />

      {/* SAE List */}
      <SAEList
        saes={saes || []}
        attachedSAEId={attachedSAE?.id}
        onAttach={handleAttach}
        onDetach={handleDetach}
        onDelete={handleDelete}
        attachingId={attachingId}
        isDetaching={isDetaching}
        deletingId={deletingId}
        canAttach={!attachedSAE}
      />

      {/* Info Section */}
      <Card padding="sm">
        <div className="flex items-start gap-3">
          <Info className="w-5 h-5 text-primary-400 flex-shrink-0 mt-0.5" />
          <div className="text-sm text-slate-400">
            <p className="mb-2">
              <strong className="text-slate-300">Supported Formats:</strong> SAELens format from Hugging Face.
            </p>
            <p>
              <strong className="text-slate-300">Recommended:</strong> Use Gemma-Scope SAEs with Gemma 2 models for best results.
              Match the SAE layer to the model architecture.
            </p>
          </div>
        </div>
      </Card>
    </div>
  );
}

export default SAEPage;
