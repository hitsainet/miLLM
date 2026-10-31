/**
 * The Approvals page (ENH-008 D6): "Waiting for you (n)" and History, from the one store
 * (`stores/approvalsStore.ts`). The banner is not drawn here (`Layout`).
 */
import { useState } from 'react';

import { ApprovalCard, ApprovalHistory, ApproveAllButtons } from '@/components/approvals';
import { WORDS } from '@/components/approvals/approvalWords';
import { Button } from '@/components/common';
import { useApprovalsStore } from '@/stores/approvalsStore';

export function ApprovalsPage() {
  const waiting = useApprovalsStore((s) => s.waiting);
  const history = useApprovalsStore((s) => s.history);
  const hasMore = useApprovalsStore((s) => s.historyHasMore);
  const error = useApprovalsStore((s) => s.error);
  const loadOlder = useApprovalsStore((s) => s.loadOlder);
  const [olderError, setOlderError] = useState<string | null>(null);
  const [loadingOlder, setLoadingOlder] = useState(false);

  return (
    <div className="stack-y-8" data-testid="approvals-page">
      <div>
        {/* The header's <h1> titles this page "Approvals" (ENH-004: one <h1> per screen). */}
        <p className="text-sm text-slate-400">{WORDS.subtitle}</p>
        {error && (
          <p role="alert" className="mt-2 text-sm text-amber-300">
            Could not read the approval queue: {error}
          </p>
        )}
      </div>

      <section aria-labelledby="approvals-waiting">
        <h2 id="approvals-waiting" className="text-lg font-semibold text-slate-100">
          {WORDS.waitingHeading(waiting.length)}
        </h2>
        <p className="mt-1 text-sm text-slate-400">{WORDS.waitingIntro}</p>
        <div className="mt-3">
          <ApproveAllButtons waiting={waiting} where="page" />
        </div>
        <div className="mt-3 stack-y-3">
          {waiting.length === 0 ? (
            <p className="text-sm text-slate-500">{WORDS.waitingEmpty}</p>
          ) : (
            waiting.map((a) => <ApprovalCard key={a.approval_id} approval={a} />)
          )}
        </div>
      </section>

      <section aria-labelledby="approvals-history">
        <h2 id="approvals-history" className="text-lg font-semibold text-slate-100">
          {WORDS.historyHeading}
        </h2>
        <p className="mt-1 text-sm text-slate-400">{WORDS.historyIntro}</p>
        <div className="mt-3">
          {history.length === 0 ? (
            <p className="text-sm text-slate-500">{WORDS.historyEmpty}</p>
          ) : (
            <ApprovalHistory rows={history} />
          )}
        </div>
        {hasMore && (
          <div className="mt-3">
            <Button
              size="sm"
              variant="secondary"
              loading={loadingOlder}
              onClick={async () => {
                setLoadingOlder(true);
                setOlderError(null);
                try {
                  await loadOlder();
                } catch (e) {
                  setOlderError(e instanceof Error ? e.message : String(e));
                } finally {
                  setLoadingOlder(false);
                }
              }}
            >
              {WORDS.older}
            </Button>
            {olderError && <p role="alert" className="mt-1 text-sm text-red-300">{olderError}</p>}
          </div>
        )}
      </section>
    </div>
  );
}

export default ApprovalsPage;
