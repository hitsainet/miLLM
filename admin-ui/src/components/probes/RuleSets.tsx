/**
 * Rule sets (Feature 39 S2): import a miDataworks rule-set document, see what it declares and
 * whether it still proves itself, attach it to its behaviour, delete it. There is no editor — a
 * rule set is the producer's document, imported as given.
 *
 * A set attaches only to the behaviour its own document names; the attachment adds this set to the
 * behaviour's list (the list already there is kept). Rule signals run while one of that
 * behaviour's probes is armed, and report a count beside the probes — never the matched text.
 */

import { useCallback, useEffect, useState } from 'react';
import { ruleSetsApi } from '@/services/api';
import type { RuleSetDetail, RuleSetSummary } from '@/types/ruleSet';

function errorText(e: unknown): string {
  if (e && typeof e === 'object' && 'message' in e) return String((e as Error).message);
  return 'The request failed';
}

function Detail({ detail }: { detail: RuleSetDetail }) {
  const unused = Object.keys(detail.not_used_at_serving);
  return (
    <div className="text-xs stack-y-1 pl-2 border-l border-slate-700" data-testid="rule-set-detail">
      {detail.description && <p className="text-slate-300">{detail.description}</p>}
      <table className="w-full">
        <thead>
          <tr className="text-slate-400 text-left"><th>Signal</th><th>Form</th></tr>
        </thead>
        <tbody>
          {Object.entries(detail.forms).map(([name, form]) => (
            <tr key={name} className="text-slate-200">
              <td className="font-mono">{name}</td>
              <td>{form === 'code_rule' ? 'operator code (scored after the response)' : form}</td>
            </tr>
          ))}
        </tbody>
      </table>
      <p className="text-slate-400">
        Origin: {detail.origin}
        {detail.provenance ? ` — derivation ${String(detail.provenance.derivation_id ?? '')}` : ''}
      </p>
      {detail.vector_status === 'vectors_failed' && (
        <p className="text-amber-300" data-testid="vectors-failed">
          This set disagrees with its own cases under this build and is not evaluated
          {detail.vector_report ? ` (case ${String(detail.vector_report.case_id ?? '?')})` : ''}.
        </p>
      )}
      {unused.length > 0 && (
        <p className="text-slate-500" data-testid="not-used-at-serving">
          Not used at serving: {unused.join(', ')}
        </p>
      )}
    </div>
  );
}

export function RuleSets() {
  const [items, setItems] = useState<RuleSetSummary[]>([]);
  const [next, setNext] = useState<string | null>(null);
  const [open, setOpen] = useState<RuleSetDetail | null>(null);
  const [text, setText] = useState('');
  const [problem, setProblem] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  const load = useCallback(async (cursor: string | null = null) => {
    try {
      const page = await ruleSetsApi.list({ cursor });
      setItems((prev) => (cursor ? [...prev, ...page.data] : page.data));
      setNext(page.next);
    } catch (e) {
      setProblem(errorText(e));
    }
  }, []);

  useEffect(() => {
    void load();
  }, [load]);

  const run = async (action: () => Promise<unknown>) => {
    setBusy(true);
    setProblem(null);
    try {
      await action();
      await load();
    } catch (e) {
      setProblem(errorText(e));
    } finally {
      setBusy(false);
    }
  };

  const importSet = () =>
    run(async () => {
      await ruleSetsApi.importDocument(text);
      setText('');
    });

  const attach = (set: RuleSetSummary) =>
    run(async () => {
      const current = await ruleSetsApi.attachment(set.behaviour_key);
      const ids = current.rule_set_ids.includes(set.id)
        ? current.rule_set_ids
        : [...current.rule_set_ids, set.id];
      await ruleSetsApi.setAttachment(set.behaviour_key, ids);
    });

  const detach = (set: RuleSetSummary) =>
    run(async () => {
      const current = await ruleSetsApi.attachment(set.behaviour_key);
      const ids = current.rule_set_ids.filter((id) => id !== set.id);
      if (ids.length) await ruleSetsApi.setAttachment(set.behaviour_key, ids);
      else await ruleSetsApi.clearAttachment(set.behaviour_key);
    });

  const view = async (set: RuleSetSummary) => {
    if (open?.id === set.id) {
      setOpen(null);
      return;
    }
    try {
      setOpen(await ruleSetsApi.get(set.id));
    } catch (e) {
      setProblem(errorText(e));
    }
  };

  return (
    <section className="stack-y-2" data-testid="rule-sets">
      <h2 className="text-slate-200 font-medium">Rule sets</h2>
      <div className="stack-y-1">
        <textarea
          aria-label="Rule-set document"
          data-testid="rule-set-document"
          className="w-full h-24 bg-slate-900 border border-slate-700 rounded-control text-xs font-mono p-2"
          placeholder="Paste a midataworks.rule-set/v1 document"
          value={text}
          onChange={(e) => setText(e.target.value)}
        />
        <button type="button" disabled={busy || !text.trim()} onClick={importSet}
                data-testid="rule-set-import"
                className="text-xs px-2 py-1 rounded-control bg-emerald-800 text-emerald-100">
          Import
        </button>
      </div>
      {problem && <p className="text-xs text-amber-300" data-testid="rule-set-problem">{problem}</p>}
      <ul className="stack-y-1">
        {items.map((set) => (
          <li key={set.id} className="border border-slate-700 rounded-card p-2 stack-y-1"
              data-testid={`rule-set-${set.id}`}>
            <div className="flex items-center justify-between gap-2 text-xs">
              <div>
                <span className="font-mono text-slate-100">{set.name}</span>
                <span className="ml-2 text-slate-400">v{set.version} · {set.behaviour_key}</span>
                <span className={`ml-2 ${set.vector_status === 'passed' ? 'text-emerald-300'
                                                                         : 'text-amber-300'}`}>
                  {set.vector_status === 'passed' ? 'cases pass' : 'cases fail'}
                </span>
              </div>
              <div className="flex gap-2">
                <button type="button" onClick={() => void view(set)}
                        className="px-2 py-1 rounded-control border border-slate-600">View</button>
                <button type="button" disabled={busy} onClick={() => void attach(set)}
                        data-testid={`rule-set-attach-${set.id}`}
                        className="px-2 py-1 rounded-control bg-slate-700">Attach</button>
                <button type="button" disabled={busy} onClick={() => void detach(set)}
                        data-testid={`rule-set-detach-${set.id}`}
                        className="px-2 py-1 rounded-control bg-slate-700">Detach</button>
                <button type="button" disabled={busy}
                        onClick={() => void run(() => ruleSetsApi.remove(set.id))}
                        data-testid={`rule-set-delete-${set.id}`}
                        className="px-2 py-1 rounded-control bg-red-900 text-red-100">Delete</button>
              </div>
            </div>
            {open?.id === set.id && <Detail detail={open} />}
          </li>
        ))}
      </ul>
      {next && (
        <button type="button" onClick={() => void load(next)} data-testid="rule-sets-more"
                className="text-xs px-2 py-1 rounded-control border border-slate-600">
          More
        </button>
      )}
    </section>
  );
}
