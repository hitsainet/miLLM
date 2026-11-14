/** miLLM's approval queue (ENH-008; MCP contract v1.22 §2 "Pending approval"). */

export type ApprovalStatus = 'pending' | 'executing' | 'executed' | 'failed' | 'rejected' | 'expired';

export interface ApprovalError {
  code: string;
  message: string;
  details?: Record<string, unknown>;
}

export interface CodeModuleCard {
  code_sha256: string;
  functions: string[];
  runner_contract: string;
  source: string;
}

export interface Approval {
  approval_id: string;
  action: string;
  action_label: string;
  status: ApprovalStatus;
  /** The card's facts: keys are words, in display order. `code_modules` is the code-rule card. */
  summary: Record<string, unknown>;
  /** Every value the agent's request carried (never the lease id). */
  payload: Record<string, unknown>;
  request_digest: string;
  requested_by: string;
  decided_by: string | null;
  reason: string | null;
  result: unknown;
  error: ApprovalError | null;
  created_at: string;
  decided_at: string | null;
  expires_at: string;
  /** ENH-006 D79: true for a request that only changes a stored setting. */
  settings_change?: boolean;
  /** ENH-006 D79/D81: which "Approve all" batch may approve it; null: one at a time only. */
  batch_kind?: ApprovalBatchKind | null;
  /** ENH-006 D79/D81: shared by every request one "Approve all" decided. */
  batch_id?: string | null;
}

/** The "Approve all" kinds; a batch never mixes them. */
export type ApprovalBatchKind = 'settings' | 'code_rules';

/** One request's outcome in `POST /api/approvals/approve-batch`. */
export interface ApprovalBatchItem {
  approval_id: string;
  /** `executed`, `failed`, `executing` (still running), or `not_run` (decided meanwhile). */
  status: ApprovalStatus | 'not_run';
  error: ApprovalError | null;
}

export interface ApprovalBatchResult {
  batch_id: string;
  kind: ApprovalBatchKind;
  results: ApprovalBatchItem[];
}

export interface ApprovalListing {
  approvals: Approval[];
  count: number;
  pending: number;
}
