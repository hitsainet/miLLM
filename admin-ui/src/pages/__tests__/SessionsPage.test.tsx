/**
 * Sessions page (Feature 36, FR-36.16).
 *
 * ⚠ The transcript is user text: these tests assert it is fetched ONLY on a click (call count 0
 * before, 1 after), and that a filter change sends exactly the query it names (payload and count).
 */

import { beforeEach, describe, expect, it, vi } from 'vitest';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { MemoryRouter } from 'react-router-dom';
import { render, screen, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import type { ProbeSessionDetail, ProbeSessionRow } from '@/types/probeSession';

const api = {
  sessions: vi.fn(),
  session: vi.fn(),
  sessionTranscript: vi.fn(),
  deleteSession: vi.fn(),
  status: vi.fn(),
};

vi.mock('@/services/api', async (importOriginal) => {
  const real = await importOriginal<typeof import('@/services/api')>();
  return { ...real, probesApi: { ...real.probesApi, ...api } };
});

const { SessionsPage } = await import('../SessionsPage');

const row: ProbeSessionRow = {
  id: 'ps_abc', client_session_id: 'chat-1', source: 'header', predecessor_id: null,
  link_ambiguous: false, served_model: 'tiny', opened_at: '2026-10-21T12:00:00Z',
  last_seen_at: '2026-10-21T12:05:00Z', open: true, n_turns: 3, n_transcribed: 1,
  behaviours: { pii: { n_fired: 1, n_scored: 3, max_score: 2.5 } },
};

function cell(turn: number, column: string, fired: boolean, score: number) {
  return {
    column, event_id: turn * 10 + (column === 'pii' ? 1 : 2), probe_id: `pr_${column}`,
    probe_name: column, window: 'all', scored: true, not_scored_reason: null,
    verdict: fired, score, threshold: 1.0, provisional: false,
  };
}

const detail: ProbeSessionDetail = {
  ...row,
  columns: [
    { column: 'pii', behaviour: { key: 'pii', name: 'PII', version: 1 }, probe_ids: ['pr_pii'] },
    { column: 'stakes', behaviour: null, probe_ids: ['pr_stakes'] },
  ],
  turns: [1, 2, 3].map((t) => ({
    turn_index: t, request_id: `r${t}`, created_at: null, has_transcript: t === 3,
    transcript_incomplete: false,
    verdicts: [cell(t, 'pii', t === 2, t === 2 ? 2.5 : 0.1 * t), cell(t, 'stakes', false, 0.4)],
  })),
};

function renderPage() {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(
    <QueryClientProvider client={client}>
      <MemoryRouter>
        <SessionsPage />
      </MemoryRouter>
    </QueryClientProvider>,
  );
}

beforeEach(() => {
  Object.values(api).forEach((f) => f.mockReset());
  api.sessions.mockResolvedValue([row]);
  api.session.mockResolvedValue(detail);
  api.status.mockResolvedValue({ transcripts: 'on' });
  api.sessionTranscript.mockResolvedValue([
    { turn_index: 3, request_id: 'r3', user_text: 'SENTINEL user', reply_text: 'SENTINEL reply',
      n_choices: 1, incomplete: false, created_at: null },
  ]);
});

describe('SessionsPage', () => {
  it('renders the timeline with the fired cell at its coordinates, in words', async () => {
    renderPage();
    await userEvent.click(await screen.findByTestId('session-ps_abc'));
    const table = await screen.findByTestId('session-timeline');
    expect(within(table).getAllByRole('columnheader').map((h) => h.textContent)).toEqual([
      'Turn', 'PII', 'stakes',
    ]);
    expect(screen.getByTestId('cell-2-pii')).toHaveTextContent('fired');
    expect(screen.getByTestId('cell-2-pii')).not.toHaveTextContent('not fired');
    for (const [t, c] of [[1, 'pii'], [3, 'pii'], [1, 'stakes'], [2, 'stakes'], [3, 'stakes']]) {
      expect(screen.getByTestId(`cell-${t}-${c}`)).toHaveTextContent('not fired');
    }
    expect(screen.getByTestId('recording-banner')).toHaveTextContent('ON');
  });

  it('a rule entry and a probe entry in one column are labelled apart; the rule has no score (039 S2)',
     async () => {
    const withRule: ProbeSessionDetail = {
      ...detail,
      turns: detail.turns.map((turn) => turn.turn_index !== 2 ? turn : {
        ...turn,
        verdicts: [...turn.verdicts, {
          column: 'pii', kind: 'rule' as const, rule_result_id: 7, rule_set_id: 'rs_000000000001',
          signal: 'valid', form: 'validator', window: 'prompt', fires: true, count: 2,
          outcome: 'completed', reason: null, event_id: 0, probe_id: '', probe_name: '',
          scored: false, not_scored_reason: null, verdict: null, score: null, threshold: null,
          provisional: false,
        }],
      }),
    };
    api.session.mockResolvedValue(withRule);
    renderPage();
    await userEvent.click(await screen.findByTestId('session-ps_abc'));
    const cellEl = await screen.findByTestId('cell-2-pii');
    const rule = within(cellEl).getByTestId('rule-cell');
    expect(rule).toHaveTextContent('rule');
    expect(rule).toHaveTextContent('valid');
    expect(rule).toHaveTextContent('fires ×2');
    expect(rule).not.toHaveTextContent('2.500');
    expect(within(cellEl).getAllByText(/fired/).length).toBe(1);  // the probe entry, still shown
  });

  it('fetches the text only when asked, once', async () => {
    renderPage();
    await userEvent.click(await screen.findByTestId('session-ps_abc'));
    await screen.findByTestId('session-timeline');
    expect(api.sessionTranscript).not.toHaveBeenCalled();
    expect(screen.queryByText(/SENTINEL/)).toBeNull();
    await userEvent.click(screen.getByTestId('show-text'));
    expect(await screen.findByText(/SENTINEL user/)).toBeInTheDocument();
    expect(api.sessionTranscript).toHaveBeenCalledTimes(1);
    expect(api.sessionTranscript).toHaveBeenCalledWith('ps_abc');
  });

  it('a filter change sends that filter, once', async () => {
    renderPage();
    await screen.findByTestId('session-ps_abc');
    expect(api.sessions).toHaveBeenCalledTimes(1);
    await userEvent.click(screen.getByLabelText('Fired only'));
    await vi.waitFor(() => expect(api.sessions).toHaveBeenCalledTimes(2));
    expect(api.sessions.mock.calls[1][0]).toMatchObject({ firedOnly: true });
    expect(api.sessions.mock.calls[0][0]).toMatchObject({ firedOnly: false });
  });

  it('delete asks what happens to the verdicts and sends that choice', async () => {
    api.deleteSession.mockResolvedValue({ turns: 3, transcripts: 1, events_deleted: 6, events_kept: 0 });
    renderPage();
    await userEvent.click(await screen.findByTestId('session-ps_abc'));
    await userEvent.click(await screen.findByTestId('delete-session'));
    await userEvent.click(screen.getByLabelText(/Delete them too/));
    await userEvent.click(screen.getByTestId('confirm-delete'));
    await vi.waitFor(() => expect(api.deleteSession).toHaveBeenCalledTimes(1));
    expect(api.deleteSession).toHaveBeenCalledWith('ps_abc', 'delete');
  });
});
