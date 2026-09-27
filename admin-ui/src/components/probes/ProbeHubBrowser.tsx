/**
 * Browse HuggingFace for probe definitions (FR-24.2).
 *
 * Anonymous and read-only: repos tagged `mistudio-probe-definition`, then that repo's
 * `.probe.json` files, then one import. Kept deliberately plain — the interesting decisions all
 * happen after import, at the arming gates.
 *
 * ⚠ **A row shows the rung when the manifest carries one, and says so when it does not.** Rendering
 * a missing rung as 0 would describe an unknown as "trained only", which is a claim about the
 * probe's evidence that nothing here measured.
 */

import { useState } from 'react';
import { useMutation, useQuery } from '@tanstack/react-query';
import { Download, Search } from 'lucide-react';
import { probesApi } from '@/services/api';
import type { ProbeHubDefinition } from '@/types/probe';

export function ProbeHubBrowser({
  onImported,
  onError,
}: {
  onImported: (name: string) => void;
  onError: (message: string) => void;
}) {
  const [query, setQuery] = useState('');
  const [submitted, setSubmitted] = useState('');
  const [openRepo, setOpenRepo] = useState<string | null>(null);

  const repos = useQuery({
    queryKey: ['probes', 'hub', 'search', submitted],
    queryFn: () => probesApi.hubSearch({ q: submitted || undefined }),
    enabled: submitted !== '',
  });

  const definitions = useQuery({
    queryKey: ['probes', 'hub', 'definitions', openRepo],
    queryFn: () => probesApi.hubDefinitions(openRepo as string),
    enabled: openRepo !== null,
  });

  const importOne = useMutation({
    mutationFn: (filename: string) =>
      probesApi.hubImport({ repo_id: openRepo as string, filename }),
    onSuccess: (probe) => onImported(probe.name),
    onError: (error: Error) => onError(error.message),
  });

  return (
    <section data-testid="probe-hub" className="border border-slate-700 rounded p-3 space-y-3">
      <form
        className="flex items-center gap-2"
        onSubmit={(e) => {
          e.preventDefault();
          setOpenRepo(null);
          setSubmitted(query.trim());
        }}
      >
        <Search className="w-4 h-4 text-slate-400" aria-hidden />
        <input
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Search Hugging Face for probe definitions"
          aria-label="Search Hugging Face for probe definitions"
          className="flex-1 bg-slate-900 border border-slate-700 rounded px-2 py-1 text-sm text-slate-100"
        />
        <button
          type="submit"
          className="text-sm px-3 py-1 rounded bg-slate-700 text-slate-100 hover:bg-slate-600"
        >
          Search
        </button>
      </form>

      {repos.isLoading && <p className="text-slate-500 text-sm">Searching…</p>}
      {repos.isError && (
        <p data-testid="hub-error" className="text-rose-400 text-sm">
          {(repos.error as Error).message}
        </p>
      )}
      {repos.data?.length === 0 && (
        <p className="text-slate-500 text-sm">
          No repos tagged <code className="text-slate-400">mistudio-probe-definition</code> matched.
        </p>
      )}

      {repos.data && repos.data.length > 0 && (
        <ul className="space-y-1">
          {repos.data.map((repo) => (
            <li key={repo.repo_id}>
              <button
                data-testid="hub-repo"
                onClick={() => setOpenRepo(repo.repo_id === openRepo ? null : repo.repo_id)}
                className="w-full text-left font-mono text-xs text-slate-300 hover:text-slate-100 py-1"
              >
                {repo.repo_id}
              </button>
              {openRepo === repo.repo_id && (
                <div className="pl-4 space-y-1">
                  {definitions.isLoading && (
                    <p className="text-slate-500 text-xs">Reading the manifest…</p>
                  )}
                  {definitions.data?.map((def: ProbeHubDefinition) => (
                    <div
                      key={def.filename}
                      data-testid="hub-definition"
                      className="flex items-center justify-between gap-2"
                    >
                      <span className="font-mono text-xs text-slate-400 truncate">
                        {def.filename}
                        {/* ⚠ An absent rung is stated, never rendered as 0. */}
                        <span className="ml-2 text-slate-500">
                          {def.rung === null || def.rung === undefined
                            ? 'rung not stated'
                            : `rung ${def.rung}`}
                        </span>
                      </span>
                      <button
                        onClick={() => importOne.mutate(def.filename)}
                        disabled={importOne.isPending}
                        aria-label={`Import ${def.filename}`}
                        className="text-xs px-2 py-1 rounded border border-slate-600 text-slate-300 hover:bg-slate-700 disabled:opacity-50 flex items-center gap-1"
                      >
                        <Download className="w-3 h-3" /> Import
                      </button>
                    </div>
                  ))}
                  {definitions.data?.length === 0 && (
                    <p className="text-slate-500 text-xs">
                      No <code>.probe.json</code> files in this repo.
                    </p>
                  )}
                </div>
              )}
            </li>
          ))}
        </ul>
      )}
    </section>
  );
}
