/** Feature 39 S2: rule sets as data, their attachments, and rule results (`kind: "rule"`). */

export interface RuleSetSummary {
  id: string;
  name: string;
  version: number;
  behaviour_key: string;
  origin: 'authored' | 'derived';
  digest: string;
  vector_status: 'passed' | 'vectors_failed';
  signals: number;
  expressions: string[];
  imported_at: string | null;
  checked_at: string | null;
  attached_to?: string[];
  created?: boolean;
}

export interface RuleSetDetail extends RuleSetSummary {
  description: string | null;
  forms: Record<string, string>;
  provenance: Record<string, unknown> | null;
  vector_report: Record<string, unknown> | null;
  not_used_at_serving: Record<string, unknown>;
  document: Record<string, unknown>;
}

export interface RuleSignalAttachment {
  behaviour: string;
  rule_set_ids: string[];
  signals: Record<string, string[]> | null;
  rule_sets: RuleSetSummary[];
  history: { changed_at: string | null; action: string; rule_set_ids: string[];
             by: string | null; reason: string | null }[];
}

/** One rule result on a probe surface: a count, never a value, and no score. */
export interface RuleResultEntry {
  kind: 'rule';
  rule_set_id: string;
  signal: string;
  form: string;
  window: string;
  fires: boolean | null;
  count: number;
  outcome: string;
  reason: string | null;
}
