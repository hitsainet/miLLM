/**
 * Feature 34: the saved Hugging Face token. The server never returns the token itself — only
 * whether one is configured, which tier is active, and a masked form (`hf_…abcd`).
 */

/** Which tier a Hub call uses when the request carries no token of its own. */
export type HfTokenSource = 'settings' | 'env' | 'none';

export interface HfTokenStatus {
  configured: boolean;
  source: HfTokenSource;
  masked: string | null;
  last_verified_at: string | null;
  user: string | null;
  /** A token is saved in the database (it may still be unreadable — see `stored_error`). */
  stored: boolean;
  /** Why a saved token is not used: `key_missing` / `key_mismatch` / `corrupt`. */
  stored_error: string | null;
  /** The last failed check of the saved token: `invalid` / `network`. */
  last_check_error: string | null;
  /** The server has SETTINGS_ENCRYPTION_KEY, so a save can succeed. */
  encryption_available: boolean;
  env_configured: boolean;
}

export interface HfTokenTestResult {
  ok: boolean;
  user: string | null;
  source: 'request' | HfTokenSource;
  verified_at: string;
}
