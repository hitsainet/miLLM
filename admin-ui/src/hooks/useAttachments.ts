/**
 * useAttachments — plural (multi-SAE) attachment status for circuit serving
 * (Feature 12). Polls the attach-set status and exposes an attach-set mutation.
 */
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';

import { saeApi } from '@/services/api';
import type { AttachSetItem } from '@/types';

export const ATTACHMENTS_KEY = ['sae', 'attachments'] as const;

export function useAttachments() {
  const queryClient = useQueryClient();

  const attachmentsQuery = useQuery({
    queryKey: ATTACHMENTS_KEY,
    queryFn: () => saeApi.attachments(),
    refetchInterval: 15_000,
  });

  const attachSetMutation = useMutation({
    mutationFn: (saes: AttachSetItem[]) => saeApi.attachSet(saes),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ATTACHMENTS_KEY });
    },
  });

  return {
    attachments: attachmentsQuery.data,
    isLoading: attachmentsQuery.isLoading,
    error: attachmentsQuery.error,
    refetch: attachmentsQuery.refetch,
    attachSet: attachSetMutation.mutateAsync,
    isAttaching: attachSetMutation.isPending,
  };
}
