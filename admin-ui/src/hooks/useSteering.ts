import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { steeringApi } from '@/services/api';
import { useServerStore } from '@/stores/serverStore';
import { useToast } from './useToast';
import type { FeatureSteering } from '@/types';

export function useSteering() {
  const queryClient = useQueryClient();
  const toast = useToast();
  const { setSteering } = useServerStore();

  const steeringQuery = useQuery({
    queryKey: ['steering'],
    queryFn: async () => {
      const state = await steeringApi.getState();
      setSteering(state);
      return state;
    },
    // Prevent rapid refetches that could overwrite mutation results
    staleTime: 5000,
  });

  const setFeatureMutation = useMutation({
    mutationFn: ({ index, strength }: { index: number; strength: number }) =>
      steeringApi.set({ feature_index: index, strength }),
    onSuccess: (state) => {
      setSteering(state);
      queryClient.invalidateQueries({ queryKey: ['steering'] });
    },
    onError: (error: Error) => {
      toast.error(`Failed to set feature: ${error.message}`);
    },
  });

  const batchMutation = useMutation({
    mutationFn: (features: FeatureSteering[]) =>
      steeringApi.batch({ features }),
    onSuccess: (state) => {
      setSteering(state);
      queryClient.invalidateQueries({ queryKey: ['steering'] });
      toast.success('Steering updated');
    },
    onError: (error: Error) => {
      toast.error(`Batch update failed: ${error.message}`);
    },
  });

  const removeFeatureMutation = useMutation({
    mutationFn: (index: number) => steeringApi.remove(index),
    onSuccess: (state) => {
      setSteering(state);
      queryClient.invalidateQueries({ queryKey: ['steering'] });
    },
    onError: (error: Error) => {
      toast.error(`Failed to remove feature: ${error.message}`);
    },
  });

  const clearMutation = useMutation({
    mutationFn: () => steeringApi.clear(),
    onSuccess: (state) => {
      setSteering(state);
      queryClient.invalidateQueries({ queryKey: ['steering'] });
      toast.info('Steering cleared');
    },
    onError: (error: Error) => {
      toast.error(`Clear failed: ${error.message}`);
    },
  });

  const enableMutation = useMutation({
    mutationFn: () => steeringApi.enable(),
    onSuccess: (state) => {
      setSteering(state);
      queryClient.invalidateQueries({ queryKey: ['steering'] });
      toast.success('Steering enabled');
    },
    onError: (error: Error) => {
      toast.error(`Enable failed: ${error.message}`);
    },
  });

  const disableMutation = useMutation({
    mutationFn: () => steeringApi.disable(),
    onSuccess: (state) => {
      setSteering(state);
      queryClient.invalidateQueries({ queryKey: ['steering'] });
      toast.info('Steering disabled');
    },
    onError: (error: Error) => {
      toast.error(`Disable failed: ${error.message}`);
    },
  });

  return {
    steering: steeringQuery.data,
    isLoading: steeringQuery.isLoading,
    error: steeringQuery.error?.message,
    refetch: steeringQuery.refetch,
    setFeature: setFeatureMutation.mutate,
    setFeatureStrength: setFeatureMutation.mutateAsync,
    batchUpdate: batchMutation.mutate,
    batchSetFeatures: batchMutation.mutateAsync,
    removeFeature: removeFeatureMutation.mutate,
    clear: clearMutation.mutate,
    clearFeatures: clearMutation.mutateAsync,
    enable: enableMutation.mutate,
    enableSteering: enableMutation.mutateAsync,
    disable: disableMutation.mutate,
    disableSteering: disableMutation.mutateAsync,
    isSetting: setFeatureMutation.isPending,
    isBatchSetting: batchMutation.isPending,
    isRemoving: removeFeatureMutation.isPending,
    isClearing: clearMutation.isPending,
    isEnabling: enableMutation.isPending,
    isDisabling: disableMutation.isPending,
    isUpdating:
      setFeatureMutation.isPending ||
      batchMutation.isPending ||
      removeFeatureMutation.isPending,
  };
}

export default useSteering;
