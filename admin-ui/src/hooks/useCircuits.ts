/**
 * React Query hooks for imported circuits (Feature 13).
 *
 * The activation mutation surfaces the UNVALIDATED_CIRCUIT refusal as a
 * first-class outcome: a circuit below rung 2 is not causally validated, so
 * the UI must ask for an explicit acknowledgement rather than retrying blindly.
 */

import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';

import { circuitApi } from '@/services/circuits';
import { ApiError } from '@/services/api';
import { useToast } from './useToast';

export const CIRCUITS_KEY = ['circuits'] as const;

export function useCircuits() {
  const queryClient = useQueryClient();
  const toast = useToast();

  const circuitsQuery = useQuery({
    queryKey: CIRCUITS_KEY,
    queryFn: () => circuitApi.list(),
  });

  const invalidate = () => {
    queryClient.invalidateQueries({ queryKey: CIRCUITS_KEY });
    // A slice-fallback activation materialises a cluster profile — keep the
    // Clusters/Profiles views consistent.
    queryClient.invalidateQueries({ queryKey: ['clusters'] });
    queryClient.invalidateQueries({ queryKey: ['profiles'] });
  };

  const importMutation = useMutation({
    mutationFn: (payload: unknown) => circuitApi.import(payload),
    onSuccess: (circuit) => {
      invalidate();
      toast.success(`Imported "${circuit.name}" (${circuit.rung_language})`);
    },
    onError: (error: Error) => toast.error(`Import failed: ${error.message}`),
  });

  const activateMutation = useMutation({
    mutationFn: ({
      circuitId,
      acknowledgeUnvalidated,
    }: {
      circuitId: string;
      acknowledgeUnvalidated?: boolean;
    }) => circuitApi.activate(circuitId, acknowledgeUnvalidated ?? false),
    onSuccess: (result) => {
      invalidate();
      if (result.serving_mode === 'slice_fallback') {
        toast.warning(
          `Serving the L${result.slice_layer} slice — a partial rendering, not the whole circuit`,
        );
      } else {
        toast.success(`"${result.name}" is now steering (${result.rung_language})`);
      }
    },
    onError: (error: Error) => {
      // UNVALIDATED_CIRCUIT is a deliberate gate, not a failure. It is normally
      // pre-empted by the card's acknowledgement checkbox — but a STALE cached
      // row (rung lowered server-side since the last fetch) renders the plain
      // Activate button, so the refusal can still arrive. Refetch so the card
      // learns the new rung and shows the checkbox, and say why the click did
      // nothing — silently returning left the button dead with no explanation.
      if (error instanceof ApiError && error.code === 'UNVALIDATED_CIRCUIT') {
        invalidate();
        const phrase =
          (error.details?.rung_language as string | undefined) ??
          'not causally validated';
        toast.warning(
          `This circuit is ${phrase} — tick the acknowledgement to steer with it anyway`,
        );
        return;
      }
      toast.error(`Activation failed: ${error.message}`);
    },
  });

  const deactivateMutation = useMutation({
    mutationFn: (circuitId: string) => circuitApi.deactivate(circuitId),
    onSuccess: () => {
      invalidate();
      toast.success('Circuit deactivated');
    },
    onError: (error: Error) => toast.error(`Deactivate failed: ${error.message}`),
  });

  const intensityMutation = useMutation({
    mutationFn: (intensity: number) => circuitApi.setActiveIntensity(intensity),
    onSuccess: () => invalidate(),
    onError: (error: Error) => toast.error(`Intensity failed: ${error.message}`),
  });

  const deleteMutation = useMutation({
    mutationFn: (circuitId: string) => circuitApi.remove(circuitId),
    onSuccess: () => {
      invalidate();
      toast.success('Circuit deleted');
    },
    onError: (error: Error) => toast.error(`Delete failed: ${error.message}`),
  });

  return {
    circuits: circuitsQuery.data?.circuits ?? [],
    activeCircuitId: circuitsQuery.data?.active_circuit_id ?? null,
    total: circuitsQuery.data?.total ?? 0,
    isLoading: circuitsQuery.isLoading,
    error: circuitsQuery.error,

    importCircuit: importMutation.mutateAsync,
    isImporting: importMutation.isPending,
    activateCircuit: activateMutation.mutateAsync,
    isActivating: activateMutation.isPending,
    deactivateCircuit: deactivateMutation.mutateAsync,
    isDeactivating: deactivateMutation.isPending,
    setIntensity: intensityMutation.mutateAsync,
    deleteCircuit: deleteMutation.mutateAsync,
    isDeleting: deleteMutation.isPending,
    exportCircuit: circuitApi.export,
  };
}
