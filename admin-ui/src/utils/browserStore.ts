/**
 * Per-browser page settings kept across a refresh (operator, 2026-10-29: "Refreshing the page
 * clears the filter setting and closes all the results pages that have been opened").
 *
 * ⚠ STORAGE IS OPTIONAL. `localStorage` can be missing, can throw on access (a private window,
 * blocked site data) or on write (quota), and can hold anything — an older shape, a hand-edited
 * value. Every read here falls back to "nothing stored" and every write is best-effort, so the
 * page renders exactly as it did before persistence when storage does not cooperate.
 *
 * Only cheap settings are kept (filters, which panels are open). Loaded data — older pages, opened
 * prompt windows — is never stored: it is fetched again on demand.
 */

export const STORE_KEYS = {
  /** The All / Armed / Not armed filter on the probe list. */
  probeFilter: 'millm.probes.filter',
  /** Ids of the probes whose results panels are open. */
  openResults: 'millm.probes.openResults',
  /** Per probe: the "fires only" and window filters inside its results panel. */
  resultFilters: 'millm.probes.resultFilters',
} as const;

/** The stored value parsed as JSON, or `null` when absent, unreadable or not JSON. */
export function readStored(key: string): unknown {
  try {
    const raw = window.localStorage.getItem(key);
    if (typeof raw !== 'string') return null;
    return JSON.parse(raw) as unknown;
  } catch {
    return null;
  }
}

/** Stores `value` as JSON. Never throws: a failed write just means it is not remembered. */
export function writeStored(key: string, value: unknown): void {
  try {
    window.localStorage.setItem(key, JSON.stringify(value));
  } catch {
    // Not remembered; the page itself is unaffected.
  }
}

/** A stored list of strings, keeping only the string entries. Anything else reads as empty. */
export function readStoredStrings(key: string): string[] {
  const value = readStored(key);
  return Array.isArray(value) ? value.filter((v): v is string => typeof v === 'string') : [];
}
