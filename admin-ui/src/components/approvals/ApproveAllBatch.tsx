/**
 * "Approve all" for one batch kind (ENH-006, operator 2026-10-29):
 *
 * * `settings` (D79) — "Approve all settings changes (N)": probe limits and telemetry settings;
 * * `code_rules` (D81) — "Approve all code rules (N)": imports of rule sets that name operator code.
 *
 * Each kind has its OWN button and its own batch; they never mix. A button appears only when two or
 * more waiting requests are of its kind (`batch_kind`); with one, the card's own button does it. It
 * opens a confirm that lists every request it will approve: the action, what changes, who asked,
 * and for a code rule the rule set, each code document's first 12 hex of its sha256 and the full
 * code card. Confirming sends ONE `POST /api/approvals/approve-batch` with every id and the kind.
 *
 * Reject stays one at a time, on each card.
 *
 * ⚠ Every piece of request text is a React text node (see `ApprovalCard`).
 */
import { useState } from 'react';

import { Button, Modal } from '@/components/common';
import { ApiError } from '@/services/api';
import { useApprovalsStore } from '@/stores/approvalsStore';
import type { Approval, ApprovalBatchKind, CodeModuleCard } from '@/types/approvals';

import { APPROVE_ALL_MIN, asText, askedBy, BATCH_WORDS, batchOf } from './approvalWords';
import { CodeRuleCard } from './CodeRuleCard';

function CodeHashes({ modules }: { modules: Partial<CodeModuleCard>[] }) {
  return (
    <p className="mt-2 text-slate-300" data-testid="approve-all-code-hashes">
      Code:{' '}
      {modules
        .map((m) => {
          const head = asText(m.code_sha256).slice(0, 12);
          const fns = Array.isArray(m.functions) ? m.functions.map(asText).join(', ') : '';
          return fns ? `${head} (${fns})` : head;
        })
        .join('; ')}
    </p>
  );
}

export function ApproveAllBatch({
  waiting,
  kind,
  where,
}: {
  waiting: Approval[];
  kind: ApprovalBatchKind;
  where: 'banner' | 'page';
}) {
  const approveBatch = useApprovalsStore((s) => s.approveBatch);
  const [open, setOpen] = useState(false);
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState<string | null>(null);
  const [listed, setListed] = useState<Approval[]>([]);
  const words = BATCH_WORDS[kind];

  const matching = batchOf(waiting, kind);
  if (matching.length < APPROVE_ALL_MIN && !message) return null;

  async function confirm() {
    setBusy(true);
    setMessage(null);
    try {
      const outcome = await approveBatch(listed.map((a) => a.approval_id), kind);
      const failed = outcome.results.filter((r) => r.status === 'failed' || r.status === 'not_run');
      setMessage(failed.length > 0 ? words.failed(failed.length) : null);
    } catch (e) {
      if (e instanceof ApiError && e.status === 409) setMessage(`${e.message} The list has been refreshed.`);
      else setMessage(e instanceof Error ? e.message : String(e));
    } finally {
      setOpen(false);
      setBusy(false);
    }
  }

  const id = `approve-all-${kind}-${where}`;
  return (
    <div data-testid={id} className="flex flex-wrap items-center gap-2">
      {matching.length >= APPROVE_ALL_MIN && (
        <Button
          size="sm"
          onClick={() => {
            setListed(matching);
            setMessage(null);
            setOpen(true);
          }}
        >
          {words.button(matching.length)}
        </Button>
      )}
      {message && (
        <p role="alert" className="text-sm text-amber-300">
          {message}
        </p>
      )}
      <Modal
        id={id}
        isOpen={open}
        onClose={() => (busy ? undefined : setOpen(false))}
        title={words.title(listed.length)}
        size="2xl"
        footer={
          <>
            <Button size="sm" variant="ghost" disabled={busy} onClick={() => setOpen(false)}>
              Cancel
            </Button>
            <Button size="sm" loading={busy} disabled={busy} onClick={() => void confirm()}>
              {words.confirm(listed.length)}
            </Button>
          </>
        }
      >
        <div role="dialog" aria-label={words.title(listed.length)}>
          <p className="text-sm text-slate-400">{words.intro}</p>
          <ul className="mt-3 stack-y-3" data-testid="approve-all-list">
            {listed.map((a) => {
              const { code_modules: codeModules, ...facts } = a.summary ?? {};
              const modules = Array.isArray(codeModules) ? (codeModules as Partial<CodeModuleCard>[]) : null;
              return (
                <li
                  key={a.approval_id}
                  data-testid={`approve-all-item-${a.approval_id}`}
                  className="rounded-card border border-slate-700/50 p-3 text-sm"
                >
                  <p className="font-semibold text-slate-100">{a.action_label}</p>
                  <p className="text-slate-400">{askedBy(a.requested_by)}</p>
                  {modules && <CodeHashes modules={modules} />}
                  {Object.keys(facts).length > 0 && (
                    <dl className="mt-2 grid grid-cols-1 gap-x-4 gap-y-1 sm:grid-cols-[minmax(8rem,auto)_1fr]">
                      {Object.entries(facts).map(([key, value]) => (
                        <div key={key} className="contents">
                          <dt className="text-slate-500">{key}</dt>
                          <dd className="whitespace-pre-wrap break-words text-slate-200">{asText(value)}</dd>
                        </div>
                      ))}
                    </dl>
                  )}
                  {modules && <CodeRuleCard modules={modules} />}
                  <p className="mt-1 font-mono text-xs text-slate-500">{a.approval_id}</p>
                </li>
              );
            })}
          </ul>
        </div>
      </Modal>
    </div>
  );
}

/** Both "Approve all" buttons, each with its own batch (they never mix). */
export function ApproveAllButtons({ waiting, where }: { waiting: Approval[]; where: 'banner' | 'page' }) {
  return (
    <div className="flex flex-wrap items-center gap-3">
      <ApproveAllBatch waiting={waiting} kind="settings" where={where} />
      <ApproveAllBatch waiting={waiting} kind="code_rules" where={where} />
    </div>
  );
}

export default ApproveAllBatch;
