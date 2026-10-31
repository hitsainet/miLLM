/**
 * A fake miLLM approval queue behind `fetch`, for the ENH-008 UI tests. It answers the four
 * approvals calls from an in-memory list and records every call (method, path, headers, body), so
 * a test can assert the payload and the call count. Any other request never answers (the page
 * under test stays in its loading state, as in `helpWiring.test.tsx`).
 */
import { vi } from 'vitest';

import type { Approval } from '@/types/approvals';

export interface RecordedCall {
  method: string;
  path: string;
  headers: Record<string, string>;
  body: unknown;
}

const ISO = '2026-10-28T18:00:00.000Z';

export function approval(over: Partial<Approval> = {}): Approval {
  return {
    approval_id: `apr_${'a'.repeat(32)}`,
    action: 'model_load',
    action_label: 'Load a model in miLLM',
    status: 'pending',
    summary: { Model: 'LFM2.5-1.2B-Instruct (id 1)', 'Unloads first': 'nothing (no model is loaded)' },
    payload: { model_id: 1, request: { gpu: 'auto' } },
    request_digest: `sha256:${'0'.repeat(64)}`,
    requested_by: 'agent:mcp',
    decided_by: null,
    reason: null,
    result: null,
    error: null,
    created_at: ISO,
    decided_at: null,
    expires_at: '2099-10-29T18:00:00.000Z',
    ...over,
  };
}

/** `n` waiting requests with distinct ids; the LAST expires soonest. */
export function many(n: number): Approval[] {
  return Array.from({ length: n }, (_, i) =>
    approval({
      approval_id: `apr_${String(i).padStart(32, '0')}`,
      action_label: `Action number ${i}`,
      expires_at: new Date(Date.UTC(2099, 0, 1) - i * 3_600_000).toISOString(),
    }),
  );
}

function json(data: unknown, status = 200): Response {
  return new Response(JSON.stringify(data), {
    status,
    headers: { 'content-type': 'application/json' },
  });
}

export function installApprovalsFake(initial: Approval[] = []) {
  const state: { rows: Approval[]; failIds?: Set<string> } = { rows: [...initial] };
  const calls: RecordedCall[] = [];
  const fetchMock = vi.fn(async (input: RequestInfo | URL, init: RequestInit = {}) => {
    const url = new URL(String(input), 'http://localhost');
    const method = (init.method ?? 'GET').toUpperCase();
    const headers: Record<string, string> = {};
    new Headers(init.headers).forEach((v, k) => {
      headers[k] = v;
    });
    const body = typeof init.body === 'string' ? JSON.parse(init.body) : init.body;
    if (!url.pathname.startsWith('/api/approvals')) return new Promise<Response>(() => {});
    calls.push({ method, path: url.pathname + url.search, headers, body });
    if (url.pathname === '/api/approvals/approve-batch' && method === 'POST') {
      // ENH-006 D79/D81: all or nothing on the ids (one kind), then each approved once.
      const { approval_ids: ids, kind } = body as { approval_ids: string[]; kind: string };
      const bad = ids.filter((id) => {
        const row = state.rows.find((r) => r.approval_id === id);
        return !row || row.status !== 'pending' || row.batch_kind !== kind;
      });
      if (bad.length > 0) {
        return json({ success: false, data: null, error: { code: 'approval_batch_refused', message: `Nothing was approved: ${bad.join(', ')}.`, details: { refused: bad } } }, 409);
      }
      const batchId = `bat_${'c'.repeat(32)}`;
      const results = ids.map((id) => {
        const row = state.rows.find((r) => r.approval_id === id)!;
        const failed = state.failIds?.has(id) ?? false;
        row.status = failed ? 'failed' : 'executed';
        row.error = failed ? { code: 'execution_failed', message: 'boom' } : null;
        row.decided_by = 'operator';
        row.decided_at = '2026-10-29T19:30:00.000Z';
        row.batch_id = batchId;
        return { approval_id: id, status: row.status, error: row.error };
      });
      return json({ success: true, data: { batch_id: batchId, kind, results }, error: null }, 202);
    }
    const decide = url.pathname.match(/^\/api\/approvals\/([^/]+)\/(approve|reject)$/);
    if (decide && method === 'POST') {
      const row = state.rows.find((r) => r.approval_id === decide[1]);
      if (!row || row.status !== 'pending') {
        return json({ success: false, data: null, error: { code: 'approval_not_pending', message: 'Already decided.' } }, 409);
      }
      row.status = decide[2] === 'approve' ? 'executed' : 'rejected';
      row.decided_by = 'operator';
      row.decided_at = '2026-10-28T19:30:00.000Z';
      if (decide[2] === 'reject') row.reason = (body as { reason?: string } | undefined)?.reason ?? null;
      else row.result = { id: 1, name: 'LFM2.5-1.2B-Instruct' };
      return json({ success: true, data: row, error: null }, decide[2] === 'approve' ? 202 : 200);
    }
    const status = url.searchParams.get('status');
    const rows =
      status === 'pending'
        ? state.rows.filter((r) => r.status === 'pending')
        : status === 'decided'
          ? state.rows.filter((r) => r.status !== 'pending')
          : state.rows;
    return json({
      success: true,
      data: { approvals: rows, count: rows.length, pending: state.rows.filter((r) => r.status === 'pending').length },
      error: null,
    });
  });
  vi.stubGlobal('fetch', fetchMock);
  return { state, calls, fetchMock };
}

export function resetApprovalsStore(store: { setState: (s: object) => void }) {
  store.setState({ waiting: [], history: [], historyHasMore: false, loaded: false, error: null });
}
