/**
 * Feature 34: the server's Hugging Face token status, for the download forms' hint and the
 * Settings card.
 *
 * Plain state, not React Query: the download forms' pages render in tests without a
 * QueryClientProvider, and the hint is informational — a failed read leaves `status` null and the
 * forms fall back to their original wording, never to a claim that a token is saved.
 */

import { useCallback, useEffect, useState } from 'react';
import { settingsApi } from '@/services/api';
import type { HfTokenStatus } from '@/types/settings';

export function useHfTokenStatus(): {
  status: HfTokenStatus | null;
  setStatus: (status: HfTokenStatus) => void;
  refresh: () => Promise<void>;
} {
  const [status, setStatus] = useState<HfTokenStatus | null>(null);

  const refresh = useCallback(async () => {
    try {
      setStatus(await settingsApi.getHfToken());
    } catch {
      /* leave it unknown rather than guess */
    }
  }, []);

  useEffect(() => {
    let cancelled = false;
    settingsApi
      .getHfToken()
      .then((s) => {
        if (!cancelled) setStatus(s);
      })
      .catch(() => {
        /* unknown: the forms keep their original wording */
      });
    return () => {
      cancelled = true;
    };
  }, []);

  return { status, setStatus, refresh };
}

/**
 * The download forms' token-field helper text. `fallback` is the form's original wording, used
 * when the server has no token or its status is unknown.
 */
export function tokenFieldHint(status: HfTokenStatus | null | undefined, fallback: string): string {
  if (status?.source === 'settings') {
    return `Leave empty to use the saved token (${status.masked ?? 'set on the Settings page'}). A token entered here overrides it for this request.`;
  }
  if (status?.source === 'env') {
    return "Leave empty to use the server's HF_TOKEN. A token entered here overrides it for this request.";
  }
  return fallback;
}
