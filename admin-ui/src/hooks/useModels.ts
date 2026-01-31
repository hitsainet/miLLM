import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { modelApi } from '@/services/api';
import { useServerStore } from '@/stores/serverStore';
import { useToast } from './useToast';
import type { ModelDownloadRequest } from '@/types';

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

  return {
    models: modelsQuery.data ?? [],
    isLoading: modelsQuery.isLoading,
    error: modelsQuery.error?.message,
    refetch: modelsQuery.refetch,
    download: downloadMutation.mutate,
    load: loadMutation.mutate,
    unload: unloadMutation.mutate,
    delete: deleteMutation.mutate,
    cancel: cancelMutation.mutate,
    isDownloading: downloadMutation.isPending,
    isLoadingModel: loadMutation.isPending,
  };
}

export default useModels;
