/**
 * React Query hooks for imported circuits (Feature 13).
 *
 * The activation mutation surfaces the UNVALIDATED_CIRCUIT refusal as a
 * first-class outcome: a circuit below rung 2 is not causally validated, so
 * the UI must ask for an explicit acknowledgement rather than retrying blindly.
 */

import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { useEffect, useRef } from 'react';

import { circuitApi } from '@/services/circuits';
import { ApiError } from '@/services/api';
import { useToast } from './useToast';
import type { ContentionDetails } from '@components/circuits';

export const CIRCUITS_KEY = ['circuits'] as const;

export function useCircuits(
  onContention?: (details: ContentionDetails, message: string) => void,
) {
  // Held in a ref so the mutation's onError closure always calls the CURRENT
  // handler. Capturing it directly would pin the first render's callback and
  // silently drop later ones — a stale-closure bug that presents as "the
  // dialog stopped opening" and is miserable to trace.
  const onContentionRef = useRef(onContention);
  useEffect(() => {
    onContentionRef.current = onContention;
  }, [onContention]);

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
      allowLayerOverlap,
    }: {
      circuitId: string;
      acknowledgeUnvalidated?: boolean;
      allowLayerOverlap?: boolean;
    }) =>
      circuitApi.activate(
        circuitId,
        acknowledgeUnvalidated ?? false,
        allowLayerOverlap ?? false,
      ),
    onSuccess: (result) => {
      invalidate();
      if (result.serving_mode === 'slice_fallback') {
        toast.warning(
          `Serving the L${result.slice_layer} slice — a partial rendering, not the whole circuit`,
        );
      } else if (result.composed_layers?.length) {
        // F19 R3-04: do NOT report a rung here. Composition is exactly the
        // state where the runtime SUPPRESSES `X-miLLM-Circuit-Rung`, because
        // no single circuit's evidence describes a summed response — so
        // asserting "causally validated (edge)" in the success toast makes the
        // claim at the one moment the server has stopped making it.
        //
        // R2-14 fixed this same contradiction on the circuit card and missed
        // the toast on the compose path.
        toast.warning(
          `"${result.name}" is now steering, COMPOSED on L${result.composed_layers.join(
            ', L',
          )}. Those layers carry the summed effect of more than one circuit, ` +
            'so the circuit-rung header is omitted — no single circuit\'s ' +
            'evidence describes the response.',
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
      // F19: a layer-contention refusal is a DECISION POINT, not a failure.
      // The payload carries the incumbent, the measurement behind the refusal,
      // and (for ordinary contention) the override route — a toast would throw
      // all of it away, leaving the operator with a sentence and no actions.
      // Surfaced through `onContention` so the page can render the dialog.
      //
      // Before this, the entire contention UI was DEAD CODE: both components
      // were written, exported and unit-tested with no consumer anywhere, so
      // BR-011 §6.2's binding condition — every override is surfaced in the UI
      // — was unmet in production while the suite stayed green.
      if (error instanceof ApiError && error.code === 'CIRCUIT_LAYER_CONTENTION') {
        invalidate();
        onContentionRef.current?.(
          error.details as unknown as ContentionDetails,
          error.message,
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
    //: Fire-and-forget form. `mutateAsync` rethrows after `onError`, so a
    //: caller that only cares about the side effect (the contention dialog
    //: opening) would have to swallow a rejection it never wanted.
    activateCircuitQuiet: activateMutation.mutate,
    isActivating: activateMutation.isPending,
    deactivateCircuit: deactivateMutation.mutateAsync,
    isDeactivating: deactivateMutation.isPending,
    setIntensity: intensityMutation.mutateAsync,
    deleteCircuit: deleteMutation.mutateAsync,
    isDeleting: deleteMutation.isPending,
    exportCircuit: circuitApi.export,
  };
}
