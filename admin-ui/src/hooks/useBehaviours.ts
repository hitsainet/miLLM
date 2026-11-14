/**
 * Behaviour sets (Feature 39): the paginated behaviour list, and arming / dry-running / disarming
 * a whole behaviour.
 *
 * ⚠ The list is PAGED BY CURSOR (`GET /api/probes/behaviours`). Nothing here lists behaviours from a
 * constant: every key comes from the server.
 */

import { useState } from 'react';
import { useInfiniteQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { ApiError, probesApi } from '@/services/api';
import type { BehaviourArmResult } from '@/types/probe';
import { useToast } from './useToast';

export const BEHAVIOURS_KEY = ['probes', 'behaviours'];

export function useBehaviours(keyFilter: string) {
  return useInfiniteQuery({
    queryKey: [...BEHAVIOURS_KEY, keyFilter],
    queryFn: ({ pageParam }) =>
      probesApi.behaviours({ key: keyFilter || undefined, cursor: pageParam }),
    initialPageParam: null as string | null,
    getNextPageParam: (last) => last.next,
  });
}

/** What a set refusal said: its code and every probe's outcome. */
export interface BehaviourRefusal {
  key: string;
  code: string;
  message: string;
  details: Record<string, unknown>;
}

export function useBehaviourActions() {
  const queryClient = useQueryClient();
  const toast = useToast();
  const [dryRun, setDryRun] = useState<BehaviourArmResult | null>(null);
  const [refusal, setRefusal] = useState<BehaviourRefusal | null>(null);

  const refresh = () => {
    void queryClient.invalidateQueries({ queryKey: BEHAVIOURS_KEY });
    void queryClient.invalidateQueries({ queryKey: ['probes'] });
  };

  const arm = useMutation({
    mutationFn: ({ key, dryRun: dry }: { key: string; dryRun?: boolean }) =>
      probesApi.armBehaviour(key, { dryRun: dry ?? false }),
    onSuccess: (result) => {
      setRefusal(null);
      if (result.dry_run) {
        setDryRun(result);
        return;
      }
      setDryRun(null);
      refresh();
      toast.success(`Armed ${result.armed.length} probe(s) of ${result.behaviour.key}`);
    },
    onError: (error: Error, variables) => {
      setRefusal({
        key: variables.key,
        code: error instanceof ApiError ? error.code : 'ERROR',
        message: error.message,
        details: (error instanceof ApiError ? error.details : undefined) ?? {},
      });
    },
  });

  const disarm = useMutation({
    mutationFn: (key: string) => probesApi.disarmBehaviour(key),
    onSuccess: (result, key) => {
      refresh();
      toast.success(`Disarmed ${result.disarmed.length} probe(s) of ${key}`);
    },
    onError: (error: Error) => toast.error(`Disarm failed: ${error.message}`),
  });

  return {
    arm: (key: string) => arm.mutate({ key }),
    dryRun: (key: string) => arm.mutate({ key, dryRun: true }),
    disarm: disarm.mutate,
    busyKey: arm.isPending ? (arm.variables?.key ?? null) : null,
    dryRunResult: dryRun,
    refusal,
    clear: () => {
      setDryRun(null);
      setRefusal(null);
    },
  };
}
