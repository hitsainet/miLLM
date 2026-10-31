/**
 * The shared approvals wording (ENH-008 BUILD_SPEC §2.2, §2.5): identical in the suite's four
 * apps. Every function returns plain text; components render it as React text, never as markup.
 */
import type { Approval, ApprovalBatchKind, ApprovalStatus } from '@/types/approvals';

export const WORDS = {
  menu: 'Approvals',
  title: 'Approvals',
  subtitle: 'Requests from AI agents that wait for your decision, and the ones already decided.',
  waitingHeading: (n: number) => `Waiting for you (${n})`,
  waitingIntro:
    'An AI agent asked to do each of these. Nothing happens until you approve it; a request you do not decide expires.',
  waitingEmpty: 'No agent requests are waiting.',
  historyHeading: 'History',
  historyIntro: 'Requests you decided, and the ones that expired.',
  historyEmpty: 'No decided requests yet.',
  older: 'Show older',
  bannerLink: 'Open Approvals',
  reasonPlaceholder: 'Reason (optional)',
  defaultReason: 'Rejected by the operator.',
  expired: 'This request expired. Ask the agent to try again.',
  bannerRegion: 'Agent requests waiting for your approval',
  bannerHeading: (n: number) => `${n} agent request${n === 1 ? '' : 's'} waiting for your approval`,
  bannerMore: (n: number) => `and ${n} more waiting — Open Approvals`,
  badge: (n: number) => `${n} waiting`,
} as const;

/** ENH-006 D79 / D81: the words of each "Approve all" batch. */
export const BATCH_WORDS: Record<
  ApprovalBatchKind,
  {
    button: (n: number) => string;
    title: (n: number) => string;
    intro: string;
    confirm: (n: number) => string;
    failed: (n: number) => string;
    history: string;
  }
> = {
  settings: {
    button: (n) => `Approve all settings changes (${n})`,
    title: (n) => `Approve ${n} settings changes`,
    intro:
      'Each of these requests changes a setting. Approving runs every one of them, once each, and records each in History.',
    confirm: (n) => `Approve these ${n} settings changes`,
    failed: (n) => `${n} of the approved changes failed; see History.`,
    history: 'Approve all settings changes',
  },
  code_rules: {
    button: (n) => `Approve all code rules (${n})`,
    title: (n) => `Approve ${n} code rules`,
    intro:
      'Each of these requests imports a rule set that names operator code, shown in full below. Approving imports every one of them, once each, and records each in History.',
    confirm: (n) => `Approve these ${n} code rules`,
    failed: (n) => `${n} of the approved code-rule imports failed; see History.`,
    history: 'Approve all code rules',
  },
};

/** A batch button appears only when at least this many requests of its kind wait (one is the card's own button). */
export const APPROVE_ALL_MIN = 2;

/** The waiting requests one "Approve all" of `kind` would approve. */
export function batchOf(waiting: Approval[], kind: ApprovalBatchKind): Approval[] {
  return waiting.filter((a) => a.status === 'pending' && a.batch_kind === kind);
}

export const OUTCOME: Record<ApprovalStatus, string> = {
  pending: 'Waiting',
  executing: 'Running',
  executed: 'Approved and run',
  failed: 'Approved, then failed',
  rejected: 'Rejected',
  expired: 'Expired',
};

/** "asked by agent {name}", or "asked by an agent" when no name was recorded. */
export function askedBy(requestedBy: string | null | undefined): string {
  const name = (requestedBy ?? '').replace(/^agent:?/, '');
  return name ? `asked by agent ${name}` : 'asked by an agent';
}

/** The History "Asked by" cell: the agent's name, or "an agent". */
export function agentName(requestedBy: string | null | undefined): string {
  const name = (requestedBy ?? '').replace(/^agent:?/, '');
  return name || 'an agent';
}

/** "operator" for a decision, "nobody (expired)" for an expiry. */
export function decidedBy(a: Approval): string {
  if (a.status === 'expired') return 'nobody (expired)';
  const who = a.decided_by ?? '—';
  if (!a.batch_id) return who;
  const name = a.batch_kind ? BATCH_WORDS[a.batch_kind].history : 'Approve all';
  return `${who} (with "${name}", batch ${a.batch_id.slice(-8)})`;
}

/** "Approve load a model in miLLM": the action, its first letter in lower case. */
export function approveLabel(actionLabel: string): string {
  return `Approve ${actionLabel.charAt(0).toLowerCase()}${actionLabel.slice(1)}`;
}

/** Any value as text: strings as they are, everything else through JSON.stringify. */
export function asText(value: unknown): string {
  if (typeof value === 'string') return value;
  if (value === null || value === undefined) return '—';
  try {
    return JSON.stringify(value);
  } catch {
    return String(value);
  }
}

/** The History "Result" cell. */
export function resultText(a: Approval): string {
  if (a.status === 'failed' && a.error) return `${a.error.code}: ${a.error.message}`;
  if (a.status === 'expired') return 'expired without a decision';
  if (a.status === 'executing') return 'running';
  if (a.status !== 'executed') return '—';
  const r = a.result as Record<string, unknown> | null;
  if (r && typeof r === 'object') {
    if (r.name !== undefined && r.id !== undefined) return `${asText(r.name)} (id ${asText(r.id)})`;
    if (r.id !== undefined) return `id ${asText(r.id)}`;
  }
  return 'done';
}
