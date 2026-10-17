import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { saeApi } from '@/services/api';
import { useServerStore } from '@/stores/serverStore';
import { useToast } from './useToast';
import type { DownloadSAERequest, AttachSAERequest } from '@/types';

export function useSAE() {
  const queryClient = useQueryClient();
  const toast = useToast();
  const { setSAEs, setAttachedSAE, setSAELoading } = useServerStore();

  const saesQuery = useQuery({
    queryKey: ['saes'],
    queryFn: async () => {
      const response = await saeApi.listWithAttachment();
      setSAEs(response.saes);
      // Find attached SAE from the attachment status
      // Only update attachedSAE when we have valid attachment info
      // Don't clear it if the API returns is_attached=false to avoid race conditions
      if (response.attachment.is_attached && response.attachment.sae_id) {
        const attached = response.saes.find((s) => s.id === response.attachment.sae_id);
        if (attached) setAttachedSAE(attached);
      }
      // Note: We intentionally don't call setAttachedSAE(null) here
      // Clearing happens explicitly via detach mutation or WebSocket events
      return response.saes;
    },
    // Prevent unnecessary refetches that could cause state issues
    staleTime: 5000,
  });

  const downloadMutation = useMutation({
    mutationFn: (req: DownloadSAERequest) => saeApi.download(req),
    onSuccess: (response) => {
      queryClient.invalidateQueries({ queryKey: ['saes'] });
      // Show different toast based on actual status
      switch (response.status) {
        case 'downloading':
          toast.info('Downloading SAE...');
          break;
        case 'cached':
          toast.info('SAE is already downloaded');
          break;
        case 'attached':
          toast.info('SAE is already attached to the model');
          break;
        case 'already_downloading':
          toast.info('SAE download is already in progress');
          break;
        default:
          toast.info(response.message || 'SAE operation completed');
      }
    },
    onError: (error: Error) => {
      toast.error(`SAE download failed: ${error.message}`);
    },
  });

  const attachMutation = useMutation({
    mutationFn: (req: AttachSAERequest) => {
      setSAELoading(true);
      return saeApi.attach(req);
    },
    onSuccess: () => {
      setSAELoading(false);
      queryClient.invalidateQueries({ queryKey: ['saes'] });
      toast.success('SAE attached');
    },
    onError: (error: Error) => {
      setSAELoading(false);
      toast.error(`Attach failed: ${error.message}`);
    },
  });

  const detachMutation = useMutation({
    mutationFn: (saeId: string) => saeApi.detach(saeId),
    onSuccess: () => {
      setAttachedSAE(null);
      queryClient.invalidateQueries({ queryKey: ['saes'] });
      toast.info('SAE detached');
    },
    onError: (error: Error) => {
      toast.error(`Detach failed: ${error.message}`);
    },
  });

  const deleteMutation = useMutation({
    mutationFn: (id: string) => saeApi.delete(id),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['saes'] });
      toast.success('SAE deleted');
    },
    onError: (error: Error) => {
      toast.error(`Delete failed: ${error.message}`);
    },
  });

  const cancelMutation = useMutation({
    mutationFn: (id: string) => saeApi.cancelDownload(id),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['saes'] });
      toast.info('SAE download cancelled');
    },
    onError: (error: Error) => {
      toast.error(`Cancel failed: ${error.message}`);
    },
  });

  return {
    saes: saesQuery.data ?? [],
    isLoading: saesQuery.isLoading,
    error: saesQuery.error?.message,
    refetch: saesQuery.refetch,
    download: downloadMutation.mutate,
    downloadSAE: downloadMutation.mutateAsync,
    attach: attachMutation.mutate,
    attachSAE: attachMutation.mutateAsync,
    detach: detachMutation.mutate,
    detachSAE: detachMutation.mutateAsync,
    delete: deleteMutation.mutate,
    deleteSAE: deleteMutation.mutateAsync,
    cancel: cancelMutation.mutate,
    cancelSAE: cancelMutation.mutateAsync,
    isDownloading: downloadMutation.isPending,
    isAttaching: attachMutation.isPending,
    isDetaching: detachMutation.isPending,
    isDeleting: deleteMutation.isPending,
    isCancelling: cancelMutation.isPending,
  };
}

export default useSAE;
