/**
 * Operator-facing groups for the Environment section (2026-10-28).
 *
 * Each section of the page (when a change takes effect: the server's `group`) listed its settings
 * in the server's order, so the ~185 rows read as one long unordered list. Within a section the
 * rows are now grouped by the SUBSYSTEM they configure, in a fixed order, and sorted
 * alphabetically by label inside each group.
 *
 * The server sends no subsystem, so it is read off the key by prefix (first match wins; the more
 * specific prefixes come first). A key no rule matches is never dropped: it lands in "Other
 * settings", last.
 */
import type { EnvironmentRow } from '@/types/environment';

export interface SettingGroup {
  id: string;
  title: string;
  rows: EnvironmentRow[];
}

/** Groups in display order, each with the key prefixes (or exact keys) that belong to it. */
export const SETTING_GROUPS: ReadonlyArray<{ id: string; title: string; match: readonly string[] }> = [
  {
    id: 'serving',
    title: 'Serving and requests',
    match: [
      'MAX_PENDING_REQUESTS', 'MAX_CONCURRENT_REQUESTS', 'EMBEDDINGS_', 'IGNORED_FIELDS_',
      'RETRY_AFTER_', 'CBM_', 'ENABLE_CONTINUOUS_BATCHING', 'QUEUE_', 'SPECULATIVE_',
      'KV_CACHE_', 'STRUCTURED_OUTPUT_', 'SCORING_',
    ],
  },
  {
    id: 'models',
    title: 'Models and memory',
    match: [
      'GGUF_', 'TRANSFORMERS_', 'TORCH_COMPILE', 'AUTO_LOAD_MODEL', 'GRACEFUL_UNLOAD_',
      'MODEL_CACHE_DIR', 'HF_TOKEN', 'LEASE_',
    ],
  },
  { id: 'weights', title: 'Attached weights', match: ['WEIGHTS_'] },
  {
    id: 'steering',
    title: 'Steering and SAEs',
    // `CIRCUIT_` is every circuit setting but circuit SENSING, which the longer
    // `CIRCUIT_SENSING_` (Sensing, below) takes.
    match: ['CLUSTER_', 'CIRCUIT_', 'MULTISAE_', 'SAE_'],
  },
  { id: 'sensing', title: 'Sensing', match: ['SENSING_', 'CIRCUIT_SENSING_'] },
  { id: 'probes', title: 'Probe monitors', match: ['PROBE_'] },
  { id: 'rules', title: 'Rule signals and operator code', match: ['RULE_'] },
  { id: 'batch', title: 'Batch API', match: ['BATCH_'] },
  // ENH-008: how long an agent's request waits on the Approvals page.
  { id: 'approvals', title: 'Agent approvals', match: ['APPROVAL_'] },
  {
    id: 'telemetry',
    title: 'Telemetry and logging',
    match: ['TELEMETRY_', 'OTEL_', 'LOG_', 'DEBUG'],
  },
  {
    id: 'server',
    title: 'Server, storage and security',
    match: ['HOST', 'PORT', 'CORS_', 'DATABASE_', 'SETTINGS_ENCRYPTION_KEY', 'DATA_DIR'],
  },
];

export const OTHER_GROUP = { id: 'other', title: 'Other settings' } as const;

/** The group a key belongs to: the LONGEST matching prefix across all groups, so
 *  `CIRCUIT_SENSING_EPSILON` is sensing, not steering, whatever the table's order. */
export function groupIdOf(key: string): string {
  let best: { id: string; length: number } | null = null;
  for (const group of SETTING_GROUPS) {
    for (const prefix of group.match) {
      if ((key === prefix || key.startsWith(prefix)) && (!best || prefix.length > best.length)) {
        best = { id: group.id, length: prefix.length };
      }
    }
  }
  return best?.id ?? OTHER_GROUP.id;
}

/** Rows grouped in the fixed order, each group sorted by label; empty groups are omitted. */
export function groupSettings(rows: readonly EnvironmentRow[]): SettingGroup[] {
  const order = [...SETTING_GROUPS, OTHER_GROUP];
  const buckets = new Map<string, EnvironmentRow[]>(order.map((g) => [g.id, []]));
  for (const row of rows) buckets.get(groupIdOf(row.key))!.push(row);
  return order
    .map((g) => ({
      id: g.id,
      title: g.title,
      rows: [...buckets.get(g.id)!].sort(
        (a, b) =>
          a.label.localeCompare(b.label, undefined, { sensitivity: 'base', numeric: true }) ||
          a.key.localeCompare(b.key),
      ),
    }))
    .filter((g) => g.rows.length > 0);
}
