import { useState } from 'react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { modelApi } from '@/services/api';
import { useServerStore } from '@/stores/serverStore';
import { useToast } from './useToast';
import type { ModelDownloadRequest, ModelPreviewResponse } from '@/types';

export function useModels() {
  const queryClient = useQueryClient();
  const toast = useToast();
  const { setModels, setLoadedModel, setModelLoading } = useServerStore();

  const modelsQuery = useQuery({
    queryKey: ['models'],
    queryFn: async () => {
      const models = await modelApi.list();
      setModels(models);
      const loaded = models.find((m) => m.status === 'loaded');
      if (loaded) setLoadedModel(loaded);
      return models;
    },
    // Poll every 2 seconds when there's a downloading or loading model to show progress
    refetchInterval: (query) => {
      const models = query.state.data;
      const hasActiveOperation = models?.some(
        (m) => m.status === 'downloading' || m.status === 'loading'
      );
      return hasActiveOperation ? 2000 : false;
    },
  });

  const downloadMutation = useMutation({
    mutationFn: (req: ModelDownloadRequest) => modelApi.download(req),
    onSuccess: (model) => {
      queryClient.invalidateQueries({ queryKey: ['models'] });
      toast.info(`Downloading ${model.name}...`);
    },
    onError: (error: Error) => {
      toast.error(`Download failed: ${error.message}`);
    },
  });

  const loadMutation = useMutation({
    mutationFn: (id: number) => {
      setModelLoading(true);
      return modelApi.load(id);
    },
    onSuccess: (model) => {
      setLoadedModel(model);
      setModelLoading(false);
      queryClient.invalidateQueries({ queryKey: ['models'] });
      toast.success(`Model "${model.name}" loaded`);
    },
    onError: (error: Error) => {
      setModelLoading(false);
      toast.error(`Load failed: ${error.message}`);
    },
  });

  const unloadMutation = useMutation({
    mutationFn: (id: number) => modelApi.unload(id),
    onSuccess: () => {
      setLoadedModel(null);
      queryClient.invalidateQueries({ queryKey: ['models'] });
      toast.info('Model unloaded');
    },
    onError: (error: Error) => {
      toast.error(`Unload failed: ${error.message}`);
    },
  });

  const deleteMutation = useMutation({
    mutationFn: (id: number) => modelApi.delete(id),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['models'] });
      toast.success('Model deleted');
    },
    onError: (error: Error) => {
      toast.error(`Delete failed: ${error.message}`);
    },
  });

  const cancelMutation = useMutation({
    mutationFn: (id: number) => modelApi.cancelDownload(id),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['models'] });
      toast.info('Download cancelled');
    },
    onError: (error: Error) => {
      toast.error(`Cancel failed: ${error.message}`);
    },
  });

  const [previewData, setPreviewData] = useState<ModelPreviewResponse | null>(null);

  const previewMutation = useMutation({
    mutationFn: (repoId: string) => modelApi.preview(repoId),
    onSuccess: (data) => {
      setPreviewData(data);
    },
    onError: (error: Error) => {
      toast.error(`Preview failed: ${error.message}`);
      setPreviewData(null);
    },
  });

  return {
    models: modelsQuery.data ?? [],
    isLoading: modelsQuery.isLoading,
    error: modelsQuery.error?.message,
    refetch: modelsQuery.refetch,
    download: downloadMutation.mutate,
    downloadModel: downloadMutation.mutateAsync,
    load: loadMutation.mutate,
    loadModel: loadMutation.mutateAsync,
    unload: unloadMutation.mutate,
    unloadModel: unloadMutation.mutateAsync,
    delete: deleteMutation.mutate,
    cancel: cancelMutation.mutate,
    isDownloading: downloadMutation.isPending,
    isLoadingModel: loadMutation.isPending,
    isUnloading: unloadMutation.isPending,
    isDeleting: deleteMutation.isPending,
    previewModel: previewMutation.mutateAsync,
    isPreviewingModel: previewMutation.isPending,
    previewData,
    clearPreview: () => setPreviewData(null),
  };
}

export default useModels;
