/**
 * Sessions (Feature 36): each monitored conversation as a timeline of behaviours.
 *
 * One row per turn, one column per behaviour (a probe with no behaviour is its own column). A cell
 * says, in WORDS as well as colour, whether the behaviour fired, did not fire, or was not scored —
 * `null` is not `false` here either: a probe that ranks without a bar decided nothing.
 *
 * ⚠ USER TEXT IS FETCHED ONLY ON A CLICK. The list and the timeline carry `has_transcript` and no
 * text; "Show text" calls the one route that serves it, and only for the session in view.
 */

import { useMemo, useState } from 'react';
import { AlertTriangle, MessagesSquare, Trash2 } from 'lucide-react';
import { Link } from 'react-router-dom';
import { HelpLink } from '@/components/help';
import { probesApi } from '@/services/api';
import {
  useDeleteProbeSession,
  useProbeSession,
  useProbeSessions,
  useRecordingStatus,
} from '@/hooks/useProbeSessions';
import type {
  ProbeSessionDetail,
  ProbeTranscriptTurn,
  SessionListFilters,
  SessionVerdictCell,
} from '@/types/probeSession';

const SINCE_CHOICES: Record<string, number | null> = {
  'Last hour': 1,
  'Last day': 24,
  'Last week': 24 * 7,
  'Any time': null,
};

function cellText(cell: SessionVerdictCell): string {
  if (!cell.scored) return `not scored (${cell.not_scored_reason ?? 'unknown'})`;
  if (cell.verdict === true) return 'fired';
  if (cell.verdict === false) return 'not fired';
  return 'no bar';
}

function cellClass(cell: SessionVerdictCell): string {
  if (!cell.scored) return 'text-slate-400';
  if (cell.verdict === true) return 'text-red-300 font-semibold';
  return 'text-slate-200';
}

function RecordingBanner() {
  const status = useRecordingStatus();
  const on = status.data?.transcripts === 'on';
  return (
    <div
      data-testid="recording-banner"
      className={`rounded-sm border px-3 py-2 text-sm ${
        on ? 'border-amber-700 bg-amber-900/30 text-amber-200' : 'border-slate-700 text-slate-300'
      }`}
    >
      {on
        ? 'Text recording is ON: the newest user message and the reply of each monitored turn are kept for 7 days.'
        : 'Text recording is off: sessions keep verdicts only, no text.'}{' '}
      <Link to="/settings" className="underline">
        Change in Settings
      </Link>
      . Do not put personal data in session ids.
    </div>
  );
}

function Timeline({ detail }: { detail: ProbeSessionDetail }) {
  const [text, setText] = useState<ProbeTranscriptTurn[] | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const del = useDeleteProbeSession();
  const [confirming, setConfirming] = useState(false);
  const [events, setEvents] = useState<'keep' | 'delete'>('keep');

  const showText = async () => {
    setLoading(true);
    setError(null);
    try {
      setText(await probesApi.sessionTranscript(detail.id));
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  };
  const textByTurn = useMemo(() => {
    const out = new Map<number, ProbeTranscriptTurn>();
    for (const t of text ?? []) out.set(t.turn_index, t);
    return out;
  }, [text]);

  return (
    <section data-testid="session-detail" className="stack-y-3">
      <div className="flex flex-wrap items-center gap-3 text-sm text-slate-300">
        <span className="font-mono">{detail.id}</span>
        {detail.client_session_id && <span>client id {detail.client_session_id}</span>}
        <span>found by {detail.source}</span>
        {detail.link_ambiguous && (
          <span className="inline-flex items-center gap-1 text-amber-300" data-testid="ambiguous-note">
            <AlertTriangle className="w-4 h-4" aria-hidden="true" />
            Matched more than one conversation, so it is shown on its own.
          </span>
        )}
        {detail.predecessor_id && <span>continues {detail.predecessor_id}</span>}
        {detail.n_transcribed > 0 && (
          <button
            type="button"
            onClick={showText}
            disabled={loading}
            className="rounded-sm bg-slate-700 px-2 py-1 text-xs hover:bg-slate-600"
            data-testid="show-text"
          >
            {loading ? 'Loading…' : 'Show text'}
          </button>
        )}
        <button
          type="button"
          onClick={() => setConfirming(true)}
          className="inline-flex items-center gap-1 rounded-sm bg-red-900/40 px-2 py-1 text-xs text-red-200"
          data-testid="delete-session"
        >
          <Trash2 className="w-3.5 h-3.5" aria-hidden="true" /> Delete session
        </button>
      </div>
      {error && <p className="text-sm text-red-300">{error}</p>}
      {confirming && (
        <div role="dialog" aria-label="Delete session" className="rounded-sm border border-red-800 p-3 text-sm">
          <p>Delete this session, its turns and any recorded text?</p>
          <fieldset className="mt-2 stack-y-1">
            <legend className="text-slate-300">Its verdicts:</legend>
            <label className="block">
              <input type="radio" name="events" checked={events === 'keep'} onChange={() => setEvents('keep')} />{' '}
              Keep them (detached from the session)
            </label>
            <label className="block">
              <input type="radio" name="events" checked={events === 'delete'} onChange={() => setEvents('delete')} />{' '}
              Delete them too
            </label>
          </fieldset>
          <div className="mt-2 flex gap-2">
            <button
              type="button"
              className="rounded-sm bg-red-800 px-2 py-1"
              data-testid="confirm-delete"
              onClick={() => {
                del.mutate({ id: detail.id, events });
                setConfirming(false);
              }}
            >
              Delete
            </button>
            <button type="button" className="rounded-sm bg-slate-700 px-2 py-1" onClick={() => setConfirming(false)}>
              Cancel
            </button>
          </div>
        </div>
      )}
      <table className="w-full text-sm" data-testid="session-timeline">
        <caption className="sr-only">Verdicts per turn and behaviour</caption>
        <thead>
          <tr>
            <th scope="col" className="text-left">Turn</th>
            {detail.columns.map((c) => (
              <th scope="col" key={c.column} className="text-left">
                {c.behaviour?.name ?? c.column}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {detail.turns.map((turn) => (
            <tr key={turn.turn_index} data-testid={`turn-${turn.turn_index}`}>
              <th scope="row" className="text-left align-top">
                {turn.turn_index}
                {turn.has_transcript && <span className="ml-1 text-xs text-slate-400">(text)</span>}
                {textByTurn.get(turn.turn_index) && (
                  <div className="mt-1 max-w-md whitespace-pre-wrap text-xs font-normal text-slate-300">
                    <p><strong>User:</strong> {textByTurn.get(turn.turn_index)?.user_text}</p>
                    <p><strong>Reply:</strong> {textByTurn.get(turn.turn_index)?.reply_text}</p>
                  </div>
                )}
              </th>
              {detail.columns.map((c) => {
                const cells = turn.verdicts.filter((v) => v.column === c.column);
                return (
                  <td key={c.column} className="align-top" data-testid={`cell-${turn.turn_index}-${c.column}`}>
                    {cells.length === 0 && <span className="text-slate-500">—</span>}
                    {cells.map((cell) =>
                      cell.kind === 'rule' ? (
                        // 039 S2: a rule signal's entry — labelled as a rule, a count, never a score.
                        <div key={`rule-${cell.rule_result_id}`} data-testid="rule-cell"
                             className={cell.fires ? 'text-sky-300' : 'text-slate-400'}>
                          <span className="rounded-sm border border-sky-800 px-1 text-xs">rule</span>
                          <span className="ml-1 font-mono text-xs">{cell.signal}</span>
                          <span className="ml-1 text-xs">
                            {cell.outcome === 'completed'
                              ? (cell.fires ? `fires ×${cell.count ?? 0}` : 'silent')
                              : `not run — ${cell.reason ?? 'unknown'}`}
                          </span>
                          <span className="ml-1 rounded-sm bg-slate-800 px-1 text-xs">{cell.window}</span>
                        </div>
                      ) : (
                        <div key={`probe-${cell.event_id}`} className={cellClass(cell)}>
                          {cellText(cell)}
                          {cell.score !== null && <span className="ml-1 text-xs">{cell.score.toFixed(3)}</span>}
                          <span className="ml-1 rounded-sm bg-slate-800 px-1 text-xs">{cell.window}</span>
                        </div>
                      ))}
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </section>
  );
}

export function SessionsPage() {
  const [sinceLabel, setSinceLabel] = useState('Last day');
  const [behaviour, setBehaviour] = useState('');
  const [model, setModel] = useState('');
  const [firedOnly, setFiredOnly] = useState(false);
  const [source, setSource] = useState('');
  const [selected, setSelected] = useState<string | null>(null);

  // The cut-off is computed once per choice, so a re-render does not change the query key.
  const since = useMemo(() => {
    const hours = SINCE_CHOICES[sinceLabel];
    return hours === null ? undefined : new Date(Date.now() - hours * 3600_000).toISOString();
  }, [sinceLabel]);

  const filters: SessionListFilters = {
    since,
    behaviour: behaviour || undefined,
    servedModel: model || undefined,
    firedOnly,
    source: source || undefined,
  };
  const sessions = useProbeSessions(filters);
  const detail = useProbeSession(selected);

  return (
    <div className="stack-y-4" data-testid="sessions-page">
      <div className="flex items-center gap-2">
        <MessagesSquare className="w-5 h-5" aria-hidden="true" />
        <p className="text-sm text-slate-300">
          Each conversation the armed probes saw, turn by turn. Sessions exist only while a probe is
          armed.
        </p>
        <HelpLink topic="sessions" label="How sessions are found" />
      </div>
      <RecordingBanner />
      <div className="flex flex-wrap gap-3 text-sm">
        <label>
          Time{' '}
          <select aria-label="Time" value={sinceLabel} onChange={(e) => setSinceLabel(e.target.value)}>
            {Object.keys(SINCE_CHOICES).map((k) => (
              <option key={k}>{k}</option>
            ))}
          </select>
        </label>
        <label>
          Behaviour{' '}
          <input aria-label="Behaviour" value={behaviour} onChange={(e) => setBehaviour(e.target.value)} />
        </label>
        <label>
          Model <input aria-label="Model" value={model} onChange={(e) => setModel(e.target.value)} />
        </label>
        <label>
          Source{' '}
          <select aria-label="Source" value={source} onChange={(e) => setSource(e.target.value)}>
            <option value="">Any</option>
            <option value="header">Session header</option>
            <option value="fingerprint">Continuation</option>
            <option value="single">Single request</option>
          </select>
        </label>
        <label>
          <input type="checkbox" aria-label="Fired only" checked={firedOnly} onChange={(e) => setFiredOnly(e.target.checked)} />{' '}
          Fired only
        </label>
      </div>
      {sessions.isError && <p className="text-red-300">Could not load sessions.</p>}
      <ul className="divide-y divide-slate-800" data-testid="session-list">
        {(sessions.data ?? []).map((s) => (
          <li key={s.id}>
            <button
              type="button"
              className="w-full py-2 text-left text-sm hover:bg-slate-800"
              onClick={() => setSelected(s.id)}
              data-testid={`session-${s.id}`}
            >
              <span className="font-mono">{s.id}</span> · {s.n_turns} turns · {s.served_model ?? 'model not recorded'}
              {s.open ? ' · open' : ''}
              {Object.entries(s.behaviours)
                .filter(([, b]) => b.n_fired > 0)
                .map(([k, b]) => (
                  <span key={k} className="ml-2 text-red-300">
                    {k} fired {b.n_fired}×
                  </span>
                ))}
            </button>
          </li>
        ))}
        {sessions.data && sessions.data.length === 0 && (
          <li className="py-2 text-sm text-slate-400">No sessions match.</li>
        )}
      </ul>
      {detail.data && <Timeline key={detail.data.id} detail={detail.data} />}
    </div>
  );
}

export default SessionsPage;
