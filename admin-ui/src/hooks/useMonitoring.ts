import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { monitoringApi } from '@/services/api';
import { useServerStore } from '@/stores/serverStore';
import { useToast } from './useToast';
import type { ConfigureMonitoringRequest } from '@/types';

export function useMonitoring() {
  const queryClient = useQueryClient();
  const toast = useToast();
  const {
    setMonitoring,
    setActivationHistory,
    setFeatureStatistics,
    setLatestActivations,
  } = useServerStore();

  // ── Config / state ─────────────────────────────────────────────────────────
  const configQuery = useQuery({
    queryKey: ['monitoring', 'config'],
    queryFn: async () => {
      const config = await monitoringApi.getConfig();
      setMonitoring(config);
      return config;
    },
  });

  const isEnabled = configQuery.data?.enabled ?? false;

  // ── Activation history (REST poll every 5 s while enabled) ─────────────────
  const historyQuery = useQuery({
    queryKey: ['monitoring', 'history'],
    queryFn: async () => {
      const history = await monitoringApi.getHistory(100);
      setActivationHistory(history.records);
      // Seed Latest Activations from the most recent record so the panel
      // isn't blank after a page refresh (socket events only go forward).
      if (history.records.length > 0) {
        const latest = history.records[0]; // newest first
        const features =
          latest.top_k && latest.top_k.length > 0
            ? latest.top_k
            : latest.activations.slice(0, currentTopK);
        setLatestActivations(features);
      }
      return history;
    },
    enabled: isEnabled,
    refetchInterval: 5000,
  });

  // ── Feature statistics (REST poll every 10 s while enabled) ────────────────
  // Uses POST /api/monitoring/statistics/top so the panel always shows the
  // top-K features by mean activation, matching the top_k setting.
  // The query key includes top_k so it re-fetches immediately on change.
  const currentTopK = configQuery.data?.top_k ?? 10;
  const statisticsQuery = useQuery({
    queryKey: ['monitoring', 'statistics', currentTopK],
    queryFn: async () => {
      const resp = await monitoringApi.getTopFeatures(currentTopK, 'mean');
      setFeatureStatistics(resp.features);
      return resp;
    },
    enabled: isEnabled,
    refetchInterval: 10000,
  });

  // ── Mutations ──────────────────────────────────────────────────────────────

  const configureMutation = useMutation({
    mutationFn: (req: ConfigureMonitoringRequest) => monitoringApi.configure(req),
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
      queryClient.invalidateQueries({ queryKey: ['monitoring', 'history'] });
      toast.info('History cleared');
    },
    onError: (error: Error) => {
      toast.error(`Clear history failed: ${error.message}`);
    },
  });

  const resetStatisticsMutation = useMutation({
    mutationFn: () => monitoringApi.resetStatistics(),
    onSuccess: () => {
      setFeatureStatistics([]);
      queryClient.invalidateQueries({ queryKey: ['monitoring', 'statistics'] });
      toast.info('Statistics reset');
    },
    onError: (error: Error) => {
      toast.error(`Reset statistics failed: ${error.message}`);
    },
  });

  const topFeaturesMutation = useMutation({
    mutationFn: ({
      k,
      metric,
    }: {
      k?: number;
      metric?: 'mean' | 'max' | 'active_ratio' | 'count';
    }) => monitoringApi.getTopFeatures(k, metric),
  });

  return {
    // State
    config: configQuery.data,
    topK: currentTopK,
    history: historyQuery.data?.records ?? [],
    statistics: statisticsQuery.data?.features ?? [],
    totalActivations: (statisticsQuery.data as { total_activations?: number })?.total_activations ?? 0,
    statisticsSince: (statisticsQuery.data as { since?: string | null })?.since ?? null,

    // Loading states
    isLoading: configQuery.isLoading,
    isLoadingHistory: historyQuery.isLoading,
    isLoadingStats: statisticsQuery.isLoading,
    error: configQuery.error?.message,

    // Actions
    refetch: configQuery.refetch,
    configure: configureMutation.mutate,
    configureMonitoring: configureMutation.mutateAsync,
    enable: enableMutation.mutate,
    enableMonitoring: enableMutation.mutateAsync,
    disable: disableMutation.mutate,
    disableMonitoring: disableMutation.mutateAsync,
    clearHistory: clearHistoryMutation.mutateAsync,
    resetStatistics: resetStatisticsMutation.mutateAsync,
    getTopFeatures: topFeaturesMutation.mutateAsync,

    // Pending states
    isConfiguring: configureMutation.isPending,
    isEnabling: enableMutation.isPending,
    isDisabling: disableMutation.isPending,
    isClearing: clearHistoryMutation.isPending,
    isResettingStats: resetStatisticsMutation.isPending,
  };
}

export default useMonitoring;
