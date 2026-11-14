/**
 * One waiting request (ENH-008 BUILD_SPEC §2.1), the same card in the banner and on the page.
 *
 * Fields, in order: action + Waiting pill; asked by; asked / expires (tooltips: the exact
 * instant); the facts; the code-rule card when there is one; every argument the agent sent
 * (collapsed); the request id with a copy button; Approve / Reject.
 *
 * ⚠ Every piece of request text is a React text node. Objects go through JSON.stringify and are
 * still text. No `dangerouslySetInnerHTML` anywhere in this folder (`ApprovalCard.hostile.test`).
 */
import { useId, useRef, useState } from 'react';

import { Button } from '@/components/common';
import { ApiError } from '@/services/api';
import { useApprovalsStore } from '@/stores/approvalsStore';
import type { Approval } from '@/types/approvals';
import { formatExpiresIn, formatLocal } from '@/utils/eventTime';

import { approveLabel, asText, askedBy, WORDS } from './approvalWords';
import { CodeRuleCard } from './CodeRuleCard';

export function ApprovalCard({ approval }: { approval: Approval }) {
  const approve = useApprovalsStore((s) => s.approve);
  const reject = useApprovalsStore((s) => s.reject);
  const [busy, setBusy] = useState(false);
  const [rejecting, setRejecting] = useState(false);
  const [reason, setReason] = useState('');
  const [error, setError] = useState<string | null>(null);
  const cardRef = useRef<HTMLElement>(null);
  const headingId = useId();

  const asked = formatLocal(approval.created_at);
  const expires = formatLocal(approval.expires_at);
  const { code_modules: codeModules, ...facts } = approval.summary ?? {};

  /** After a success focus moves to the next card; after a failure it stays on this one. */
  async function decide(call: () => Promise<unknown>) {
    const cards = Array.from(document.querySelectorAll<HTMLElement>('[data-approval-card]'));
    const next = cards[cards.indexOf(cardRef.current as HTMLElement) + 1] ?? null;
    setBusy(true);
    setError(null);
    try {
      await call();
      next?.focus();
    } catch (e) {
      if (e instanceof ApiError && e.code === 'approval_expired') setError(WORDS.expired);
      else if (e instanceof ApiError && e.status === 409) setError(`${e.message} The list has been refreshed.`);
      else setError(e instanceof Error ? e.message : String(e));
      cardRef.current?.focus();
    } finally {
      setBusy(false);
    }
  }

  return (
    <article
      ref={cardRef}
      tabIndex={-1}
      data-approval-card
      data-testid={`approval-card-${approval.approval_id}`}
      aria-labelledby={headingId}
      className="rounded-card border border-amber-500/30 bg-slate-900/80 p-4 text-sm text-slate-200 focus:outline-hidden focus:ring-2 focus:ring-amber-400/60"
    >
      <div className="flex flex-wrap items-center gap-2">
        <h3 id={headingId} className="font-semibold text-slate-100">{approval.action_label}</h3>
        <span className="rounded-full border border-amber-500/40 bg-amber-500/10 px-2 py-0.5 text-xs text-amber-300">
          Waiting
        </span>
      </div>
      <p className="mt-1 text-slate-400">{askedBy(approval.requested_by)}</p>
      <p className="mt-1 text-slate-400">
        <span title={asked.title}>asked {asked.text}</span>
        {' · '}
        <span title={expires.title}>{formatExpiresIn(approval.expires_at)}</span>
      </p>

      {Object.keys(facts).length > 0 && (
        <dl className="mt-3 grid grid-cols-1 gap-x-4 gap-y-1 sm:grid-cols-[minmax(8rem,auto)_1fr]">
          {Object.entries(facts).map(([key, value]) => (
            <div key={key} className="contents">
              <dt className="text-slate-500">{key}</dt>
              <dd className="whitespace-pre-wrap break-words text-slate-200">{asText(value)}</dd>
            </div>
          ))}
        </dl>
      )}

      {Array.isArray(codeModules) && <CodeRuleCard modules={codeModules} />}

      <details className="mt-3">
        <summary className="cursor-pointer text-slate-400">All arguments the agent sent</summary>
        <dl className="mt-2 grid grid-cols-1 gap-x-4 gap-y-1 sm:grid-cols-[minmax(8rem,auto)_1fr]">
          {Object.entries(approval.payload ?? {}).map(([key, value]) => (
            <div key={key} className="contents">
              <dt className="font-mono text-xs text-slate-500">{key}</dt>
              <dd className="whitespace-pre-wrap break-all font-mono text-xs text-slate-300">{asText(value)}</dd>
            </div>
          ))}
        </dl>
      </details>

      <p className="mt-3 flex items-center gap-2 text-xs text-slate-500">
        <span className="font-mono">{approval.approval_id}</span>
        <button
          type="button"
          className="rounded-control border border-slate-600/50 px-1.5 py-0.5 hover:text-slate-300"
          aria-label={`Copy request id ${approval.approval_id}`}
          onClick={() => {
            try {
              void navigator.clipboard?.writeText(approval.approval_id);
            } catch {
              /* copying is a convenience; the id is on screen */
            }
          }}
        >
          Copy
        </button>
      </p>

      {error && (
        <p role="alert" className="mt-3 rounded-control border border-red-500/30 bg-red-500/10 p-2 text-red-300">
          {error}
        </p>
      )}

      {!rejecting ? (
        <div className="mt-3 flex flex-wrap gap-2">
          <Button size="sm" disabled={busy} onClick={() => void decide(() => approve(approval.approval_id))}>
            {approveLabel(approval.action_label)}
          </Button>
          <Button size="sm" variant="secondary" disabled={busy} onClick={() => setRejecting(true)}>
            Reject
          </Button>
        </div>
      ) : (
        <div className="mt-3 flex flex-wrap items-center gap-2">
          <input
            type="text"
            maxLength={1000}
            value={reason}
            onChange={(e) => setReason(e.target.value)}
            placeholder={WORDS.reasonPlaceholder}
            aria-label={WORDS.reasonPlaceholder}
            className="min-w-[16rem] flex-1 rounded-control border border-slate-600/50 bg-slate-800 px-2 py-1 text-slate-100"
          />
          <Button
            size="sm"
            variant="danger"
            disabled={busy}
            onClick={() =>
              void decide(() => reject(approval.approval_id, reason.trim() || WORDS.defaultReason))
            }
          >
            Confirm reject
          </Button>
          <Button size="sm" variant="ghost" disabled={busy} onClick={() => setRejecting(false)}>
            Cancel
          </Button>
        </div>
      )}
    </article>
  );
}

export default ApprovalCard;
