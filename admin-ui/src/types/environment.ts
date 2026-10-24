/**
 * Feature 35 (ENH-003): the server's environment settings, and the 30 that can change while miLLM
 * runs. Shapes of `GET /api/environment`, `PUT|DELETE /api/environment/{key}` and
 * `GET /api/environment/history`. A secret's row never carries its value: only `set`.
 */

/** Which section of the page a setting belongs to. */
export type EnvironmentGroup = 'runtime' | 'model_load' | 'restart' | 'environment';

export interface EnvironmentRow {
  key: string;
  /** Plain-language name (the key itself for read-only rows). */
  label: string;
  description: string | null;
  group: EnvironmentGroup;
  /** When a change takes effect ("now", "next request", "next model load", …). */
  effect: string;
  /** Writable rows only: how the value is edited. */
  type: 'int' | 'float' | 'choice' | 'bool' | null;
  secret: boolean;
  /** Secrets only: whether a value is configured. */
  set: boolean | null;
  /** The value in force (null for a secret). */
  value: unknown;
  /** The environment's value (null for a secret). */
  env_value: unknown;
  source: 'env' | 'override';
  runtime_writable: boolean;
  minimum: number | null;
  maximum: number | null;
  choices: string[] | null;
  updated_at: string | null;
  updated_by: string | null;
  reason: string | null;
}

export interface EnvironmentListing {
  settings: EnvironmentRow[];
  counts: Record<EnvironmentGroup, number>;
  runtime_keys: string[];
}

export interface EnvironmentChange {
  id: number;
  key: string;
  action: 'set' | 'reset';
  previous_value: unknown;
  new_value: unknown;
  actor: string;
  approval_id: string | null;
  reason: string | null;
  changed_at: string | null;
}

export interface EnvironmentHistory {
  changes: EnvironmentChange[];
}
