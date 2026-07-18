/**
 * React Query hooks for imported clusters (Feature 8).
 */

import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { clusterApi } from '@/services/api';
import { useToast } from './useToast';
import type { ClusterDefinitionV1 } from '@/types/clusters';

export function useClusters() {
  const queryClient = useQueryClient();
  const toast = useToast();

  const clustersQuery = useQuery({
    queryKey: ['clusters'],
    queryFn: () => clusterApi.list(),
  });

  const invalidate = () => {
    queryClient.invalidateQueries({ queryKey: ['clusters'] });
    // Cluster rows are profiles — keep the Profiles page consistent too.
    queryClient.invalidateQueries({ queryKey: ['profiles'] });
  };

  const importMutation = useMutation({
    mutationFn: (payload: unknown) => clusterApi.import(payload),
    onSuccess: (result) => {
      invalidate();
      if (result.errors > 0 || result.blocked > 0) {
        toast.warning(
          `Imported ${result.imported}, blocked ${result.blocked}, errors ${result.errors}`
        );
      } else {
        toast.success(`Imported ${result.imported} cluster${result.imported === 1 ? '' : 's'}`);
      }
    },
    onError: (error: Error) => toast.error(`Import failed: ${error.message}`),
  });

  const hubImportMutation = useMutation({
    mutationFn: (req: { repo_id: string; filename: string }) => clusterApi.hubImport(req),
    onSuccess: (item) => {
      invalidate();
      if (item.status === 'error') {
        toast.error(`Import failed: ${item.error ?? 'unknown error'}`);
      } else {
        toast.success(`Imported "${item.name}" from the Hub`);
      }
    },
    onError: (error: Error) => toast.error(`Hub import failed: ${error.message}`),
  });

  const activateMutation = useMutation({
    mutationFn: (id: string) => clusterApi.activate(id),
    onSuccess: (result) => {
      invalidate();
      toast.success(`Cluster active — ${result.feature_count} members steering`);
    },
    onError: (error: Error) => toast.error(`Activation blocked: ${error.message}`),
  });

  const deactivateMutation = useMutation({
    mutationFn: (id: string) => clusterApi.deactivate(id),
    onSuccess: () => {
      invalidate();
      toast.success('Cluster deactivated');
    },
    onError: (error: Error) => toast.error(`Deactivate failed: ${error.message}`),
  });

  const deleteMutation = useMutation({
    mutationFn: (id: string) => clusterApi.delete(id),
    onSuccess: (result) => {
      invalidate();
      toast.success(
        result.was_active
          ? 'Cluster deleted — steering cleared'
          : 'Cluster deleted'
      );
    },
    onError: (error: Error) => toast.error(`Delete failed: ${error.message}`),
  });

  const intensityMutation = useMutation({
    mutationFn: ({ id, intensity }: { id: string; intensity: number }) =>
      clusterApi.setIntensity(id, intensity),
    onSuccess: (result) => {
      invalidate();
      toast.success(
        `λ = ${result.intensity.toFixed(2)}${result.reapplied ? ' (re-applied)' : ''}`
      );
    },
    onError: (error: Error) => toast.error(`Intensity update failed: ${error.message}`),
  });

  const exportCluster = async (id: string, name: string) => {
    try {
      const definition: ClusterDefinitionV1 = await clusterApi.export(id);
      const blob = new Blob([JSON.stringify(definition, null, 2)], {
        type: 'application/json',
      });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `${name.toLowerCase().replace(/[^a-z0-9-_]+/g, '-')}.cluster.json`;
      a.click();
      URL.revokeObjectURL(url);
    } catch (error) {
      toast.error(`Export failed: ${(error as Error).message}`);
    }
  };

  return {
    clusters: clustersQuery.data?.clusters ?? [],
    activeClusterId: clustersQuery.data?.active_cluster_id ?? null,
    isLoading: clustersQuery.isLoading,
    importClusters: importMutation.mutateAsync,
    isImporting: importMutation.isPending,
    hubImport: hubImportMutation.mutateAsync,
    isHubImporting: hubImportMutation.isPending,
    activateCluster: activateMutation.mutateAsync,
    isActivating: activateMutation.isPending,
    deactivateCluster: deactivateMutation.mutateAsync,
    isDeactivating: deactivateMutation.isPending,
    deleteCluster: deleteMutation.mutateAsync,
    isDeleting: deleteMutation.isPending,
    setIntensity: intensityMutation.mutateAsync,
    exportCluster,
  };
}
