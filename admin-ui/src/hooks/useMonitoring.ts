import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { monitoringApi } from '@/services/api';
import { useServerStore } from '@/stores/serverStore';
import { useToast } from './useToast';
import type { ConfigureMonitoringRequest } from '@/types';

export function useMonitoring() {
  const queryClient = useQueryClient();
  const toast = useToast();
  const { setMonitoring, setActivationHistory, setFeatureStatistics } = useServerStore();

  const configQuery = useQuery({
    queryKey: ['monitoring', 'config'],
    queryFn: async () => {
      const config = await monitoringApi.getConfig();
      setMonitoring(config);
      return config;
    },
  });

  const historyQuery = useQuery({
    queryKey: ['monitoring', 'history'],
    queryFn: async () => {
      const history = await monitoringApi.getHistory(100);
      setActivationHistory(history.records);
      setFeatureStatistics(history.statistics);
      return history;
    },
    enabled: configQuery.data?.enabled ?? false,
    refetchInterval: 5000, // Refresh every 5 seconds
  });

  const configureMutation = useMutation({
    mutationFn: (req: ConfigureMonitoringRequest) =>
      monitoringApi.configure(req),
    onSuccess: (config) => {
      setMonitoring(config);
      queryClient.invalidateQueries({ queryKey: ['monitoring'] });
      toast.success('Monitoring configured');
    },
    onError: (error: Error) => {
      toast.error(`Configuration failed: ${error.message}`);
    },
  });

  const enableMutation = useMutation({
    mutationFn: () => monitoringApi.enable(),
    onSuccess: (config) => {
      setMonitoring(config);
      queryClient.invalidateQueries({ queryKey: ['monitoring'] });
      toast.success('Monitoring enabled');
    },
    onError: (error: Error) => {
      toast.error(`Enable failed: ${error.message}`);
    },
  });

  const disableMutation = useMutation({
    mutationFn: () => monitoringApi.disable(),
    onSuccess: (config) => {
      setMonitoring(config);
      queryClient.invalidateQueries({ queryKey: ['monitoring'] });
      toast.info('Monitoring disabled');
    },
    onError: (error: Error) => {
      toast.error(`Disable failed: ${error.message}`);
    },
  });

  const clearHistoryMutation = useMutation({
    mutationFn: () => monitoringApi.clearHistory(),
    onSuccess: () => {
      setActivationHistory([]);
      setFeatureStatistics([]);
      queryClient.invalidateQueries({ queryKey: ['monitoring', 'history'] });
      toast.info('History cleared');
    },
    onError: (error: Error) => {
      toast.error(`Clear history failed: ${error.message}`);
    },
  });

  return {
    config: configQuery.data,
    history: historyQuery.data?.records ?? [],
    statistics: historyQuery.data?.statistics ?? [],
    isLoading: configQuery.isLoading,
    isLoadingHistory: historyQuery.isLoading,
    isLoadingStats: historyQuery.isLoading,
    error: configQuery.error?.message,
    refetch: configQuery.refetch,
    configure: configureMutation.mutate,
    configureMonitoring: configureMutation.mutateAsync,
    enable: enableMutation.mutate,
    enableMonitoring: enableMutation.mutateAsync,
    disable: disableMutation.mutate,
    disableMonitoring: disableMutation.mutateAsync,
    clearHistory: clearHistoryMutation.mutateAsync,
    isConfiguring: configureMutation.isPending,
    isEnabling: enableMutation.isPending,
    isDisabling: disableMutation.isPending,
    isClearing: clearHistoryMutation.isPending,
  };
}

export default useMonitoring;
