/**
 * Feature 35: the environment settings listing and its change history, for the Settings page.
 *
 * Plain state, like `useHfTokenStatus`: the Settings page renders in tests without a
 * QueryClientProvider. A failed read is reported (`error`), never replaced by guessed values.
 */

import { useCallback, useEffect, useState } from 'react';
import { ApiError, environmentApi } from '@/services/api';
import type { EnvironmentChange, EnvironmentListing } from '@/types/environment';

export interface EnvironmentSettingsState {
  listing: EnvironmentListing | null;
  history: EnvironmentChange[];
  error: string | null;
  reload: () => Promise<void>;
}

function message(error: unknown): string {
  return error instanceof ApiError || error instanceof Error ? error.message : 'Request failed';
}

function read() {
  return Promise.all([environmentApi.list(), environmentApi.history(50)]);
}

export function useEnvironmentSettings(): EnvironmentSettingsState {
  const [listing, setListing] = useState<EnvironmentListing | null>(null);
  const [history, setHistory] = useState<EnvironmentChange[]>([]);
  const [error, setError] = useState<string | null>(null);

  const reload = useCallback(async () => {
    try {
      const [next, changes] = await read();
      setListing(next);
      setHistory(changes.changes);
      setError(null);
    } catch (e) {
      setError(message(e));
    }
  }, []);

  useEffect(() => {
    let cancelled = false;
    read()
      .then(([next, changes]) => {
        if (cancelled) return;
        setListing(next);
        setHistory(changes.changes);
        setError(null);
      })
      .catch((e: unknown) => {
        if (!cancelled) setError(message(e));
      });
    return () => {
      cancelled = true;
    };
  }, []);

  return { listing, history, error, reload };
}
