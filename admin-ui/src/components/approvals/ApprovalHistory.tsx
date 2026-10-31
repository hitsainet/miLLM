/**
 * History (ENH-008 BUILD_SPEC §2.2): every decided request, the most recent decision first.
 * Columns: Outcome · Action · Asked by · Asked · Decided by · Decided · Reason · Result · Id.
 * On a narrow screen each row stacks as a card in the same order (the cell labels show).
 */
import type { Approval } from '@/types/approvals';
import { formatLocal } from '@/utils/eventTime';

import { agentName, decidedBy, OUTCOME, resultText } from './approvalWords';

const COLUMNS = ['Outcome', 'Action', 'Asked by', 'Asked', 'Decided by', 'Decided', 'Reason', 'Result', 'Id'];

function Cell({ label, children, title, mono }: { label: string; children: string; title?: string; mono?: boolean }) {
  return (
    <td className="block px-2 py-1 align-top md:table-cell">
      <span className="text-slate-500 md:hidden">{label}: </span>
      <span title={title || undefined} className={`break-words ${mono ? 'font-mono text-xs' : ''}`}>
        {children}
      </span>
    </td>
  );
}

export function ApprovalHistory({ rows }: { rows: Approval[] }) {
  return (
    <table className="w-full text-left text-sm text-slate-300" data-testid="approvals-history">
      <thead className="hidden text-xs text-slate-500 md:table-header-group">
        <tr>
          {COLUMNS.map((c) => (
            <th key={c} scope="col" className="px-2 py-1 font-medium">{c}</th>
          ))}
        </tr>
      </thead>
      <tbody>
        {rows.map((a) => {
          const asked = formatLocal(a.created_at);
          const decided = formatLocal(a.decided_at);
          return (
            <tr
              key={a.approval_id}
              data-testid={`history-row-${a.approval_id}`}
              className="mb-2 block rounded-card border border-slate-700/50 md:mb-0 md:table-row md:rounded-none md:border-0 md:border-t"
            >
              <Cell label="Outcome">{OUTCOME[a.status] ?? a.status}</Cell>
              <Cell label="Action">{a.action_label}</Cell>
              <Cell label="Asked by">{agentName(a.requested_by)}</Cell>
              <Cell label="Asked" title={asked.title}>{asked.text}</Cell>
              <Cell label="Decided by">{decidedBy(a)}</Cell>
              <Cell label="Decided" title={decided.title}>{a.decided_at ? decided.text : '—'}</Cell>
              <Cell label="Reason">{a.reason ?? '—'}</Cell>
              <Cell label="Result">{resultText(a)}</Cell>
              <Cell label="Id" mono>{a.approval_id}</Cell>
            </tr>
          );
        })}
      </tbody>
    </table>
  );
}

export default ApprovalHistory;
