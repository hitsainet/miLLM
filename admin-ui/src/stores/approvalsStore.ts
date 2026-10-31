/**
 * The approval queue (ENH-008 D7): the ONE store that owns it, and its single poll.
 *
 * The banner on every page, the Approvals page and the menu badge all read this store; none of
 * them fetches on its own. `useApprovalsPolling` (mounted once, in `Layout`) refreshes every
 * 15 seconds, and the socket's `approval:created` / `approval:resolved` events refresh at once
 * (`services/socket.ts`). A failed load is kept as `error` and shown, never swallowed.
 *
 * ⚠ Never a computed getter here (they do not re-render): components read `waiting` and derive.
 */
import { useEffect } from 'react';
import { create } from 'zustand';

import { approvalsApi } from '@/services/api';
import type { Approval, ApprovalBatchKind, ApprovalBatchResult } from '@/types/approvals';

export const APPROVALS_POLL_MS = 15_000;
export const HISTORY_PAGE = 200;

interface ApprovalsState {
  /** Waiting requests, the soonest to expire first. */
  waiting: Approval[];
  /** Decided requests, the most recent decision first. */
  history: Approval[];
  /** True when the last History page was full, so older rows may exist. */
  historyHasMore: boolean;
  /** True once one load has succeeded. */
  loaded: boolean;
  /** The last load's failure, or null after a success. */
  error: string | null;
  refresh: () => Promise<void>;
  loadOlder: () => Promise<void>;
  approve: (id: string) => Promise<Approval>;
  /** ENH-006 D79/D81: approve the given waiting requests of one kind with ONE request. */
  approveBatch: (ids: string[], kind: ApprovalBatchKind) => Promise<ApprovalBatchResult>;
  reject: (id: string, reason: string) => Promise<Approval>;
}

/** Soonest to expire first; ties by asked time. */
export function bySoonestExpiry(a: Approval, b: Approval): number {
  return (
    Date.parse(a.expires_at) - Date.parse(b.expires_at) ||
    Date.parse(a.created_at) - Date.parse(b.created_at)
  );
}

function reasonOf(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

export const useApprovalsStore = create<ApprovalsState>((set, get) => ({
  waiting: [],
  history: [],
  historyHasMore: false,
  loaded: false,
  error: null,

  refresh: async () => {
    try {
      const shown = Math.max(HISTORY_PAGE, get().history.length);
      const [waiting, history] = await Promise.all([
        approvalsApi.waiting(),
        approvalsApi.history(Math.min(shown, 500)),
      ]);
      set({
        waiting: [...waiting.approvals].sort(bySoonestExpiry),
        history: history.approvals,
        historyHasMore: history.approvals.length >= Math.min(shown, 500),
        loaded: true,
        error: null,
      });
    } catch (error) {
      set({ error: reasonOf(error) });
    }
  },

  loadOlder: async () => {
    const { history } = get();
    const older = await approvalsApi.history(HISTORY_PAGE, history.length);
    const seen = new Set(history.map((a) => a.approval_id));
    set({
      history: [...history, ...older.approvals.filter((a) => !seen.has(a.approval_id))],
      historyHasMore: older.approvals.length >= HISTORY_PAGE,
    });
  },

  approve: async (id) => {
    try {
      return await approvalsApi.approve(id);
    } finally {
      await get().refresh();
    }
  },

  approveBatch: async (ids, kind) => {
    try {
      return await approvalsApi.approveBatch(ids, kind);
    } finally {
      await get().refresh();
    }
  },

  reject: async (id, reason) => {
    try {
      return await approvalsApi.reject(id, reason);
    } finally {
      await get().refresh();
    }
  },
}));

/** The single poll: one refresh now, then every 15 seconds. Mount ONCE (in `Layout`). */
export function useApprovalsPolling(intervalMs: number = APPROVALS_POLL_MS): void {
  useEffect(() => {
    const refresh = () => void useApprovalsStore.getState().refresh();
    refresh();
    const timer = setInterval(refresh, intervalMs);
    return () => clearInterval(timer);
  }, [intervalMs]);
}
