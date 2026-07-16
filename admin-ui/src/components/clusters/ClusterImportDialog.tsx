/**
 * ClusterImportDialog (Feature 8) — three ways in: paste JSON, upload a
 * .cluster.json / bundle file, or browse public Hugging Face cluster packs.
 */

import { useRef, useState } from 'react';
import { Search, Upload } from 'lucide-react';
import { Badge, Button, Input, Modal, Spinner } from '@components/common';
import { clusterApi } from '@/services/api';
import type { HubDefinitionRef, HubRepoInfo } from '@/types/clusters';

type Tab = 'paste' | 'file' | 'hub';

interface ClusterImportDialogProps {
  open: boolean;
  onClose: () => void;
  onImport: (payload: unknown) => Promise<unknown>;
  onHubImport: (req: { repo_id: string; filename: string }) => Promise<unknown>;
  isImporting: boolean;
}

export function ClusterImportDialog({
  open,
  onClose,
  onImport,
  onHubImport,
  isImporting,
}: ClusterImportDialogProps) {
  const [tab, setTab] = useState<Tab>('paste');
  const [pasted, setPasted] = useState('');
  const [parseError, setParseError] = useState<string | null>(null);
  const fileRef = useRef<HTMLInputElement>(null);

  // Hub browse state
  const [query, setQuery] = useState('');
  const [repos, setRepos] = useState<HubRepoInfo[] | null>(null);
  const [selectedRepo, setSelectedRepo] = useState<string | null>(null);
  const [definitions, setDefinitions] = useState<HubDefinitionRef[] | null>(null);
  const [hubBusy, setHubBusy] = useState(false);
  const [hubError, setHubError] = useState<string | null>(null);

  const submitJson = async (text: string) => {
    setParseError(null);
    let payload: unknown;
    try {
      payload = JSON.parse(text);
    } catch {
      setParseError('Not valid JSON');
      return;
    }
    await onImport(payload);
    onClose();
  };

  const handleFile = async (file: File) => {
    await submitJson(await file.text());
  };

  const runSearch = async () => {
    setHubBusy(true);
    setHubError(null);
    setSelectedRepo(null);
    setDefinitions(null);
    try {
      setRepos(await clusterApi.hubSearch({ q: query || undefined }));
    } catch (e) {
      setHubError((e as Error).message);
    } finally {
      setHubBusy(false);
    }
  };

  const openRepo = async (repoId: string) => {
    setSelectedRepo(repoId);
    setDefinitions(null);
    setHubBusy(true);
    setHubError(null);
    try {
      setDefinitions(await clusterApi.hubDefinitions(repoId));
    } catch (e) {
      setHubError((e as Error).message);
    } finally {
      setHubBusy(false);
    }
  };

  const importFromHub = async (filename: string) => {
    if (!selectedRepo) return;
    await onHubImport({ repo_id: selectedRepo, filename });
    onClose();
  };

  return (
    <Modal id="cluster-import" isOpen={open} onClose={onClose} title="Import cluster definition">
      <div className="flex gap-1 mb-4 border-b border-slate-700">
        {(['paste', 'file', 'hub'] as Tab[]).map((t) => (
          <button
            key={t}
            onClick={() => setTab(t)}
            className={`px-3 py-1.5 text-sm rounded-t transition-colors ${
              tab === t
                ? 'bg-slate-700/60 text-slate-100'
                : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            {t === 'paste' ? 'Paste JSON' : t === 'file' ? 'Upload file' : 'Hugging Face'}
          </button>
        ))}
      </div>

      {tab === 'paste' && (
        <div className="space-y-3">
          <textarea
            value={pasted}
            onChange={(e) => setPasted(e.target.value)}
            rows={10}
            placeholder='{"kind": "mistudio.cluster-definition", "schema_version": "1", ...}'
            className="w-full px-3 py-2 bg-slate-900 border border-slate-700 rounded text-xs font-mono text-slate-100 placeholder-slate-600 focus:outline-none focus:border-cyan-500"
          />
          {parseError && <p className="text-xs text-red-400">{parseError}</p>}
          <div className="flex justify-end">
            <Button
              variant="primary"
              size="sm"
              disabled={!pasted.trim() || isImporting}
              onClick={() => void submitJson(pasted)}
            >
              Import
            </Button>
          </div>
        </div>
      )}

      {tab === 'file' && (
        <div className="space-y-3">
          <button
            onClick={() => fileRef.current?.click()}
            className="w-full border-2 border-dashed border-slate-700 hover:border-slate-500 rounded-lg p-8 text-slate-400 hover:text-slate-200 flex flex-col items-center gap-2"
          >
            <Upload className="w-6 h-6" />
            <span className="text-sm">Choose a .cluster.json or bundle file</span>
          </button>
          <input
            ref={fileRef}
            type="file"
            accept=".json,application/json"
            className="hidden"
            onChange={(e) => {
              const f = e.target.files?.[0];
              if (f) void handleFile(f);
              e.target.value = '';
            }}
          />
          {parseError && <p className="text-xs text-red-400">{parseError}</p>}
        </div>
      )}

      {tab === 'hub' && (
        <div className="space-y-3">
          <div className="flex gap-2">
            <Input
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="Search cluster packs…"
              onKeyDown={(e) => e.key === 'Enter' && void runSearch()}
            />
            <Button variant="secondary" size="sm" onClick={() => void runSearch()}>
              <Search className="w-4 h-4" />
            </Button>
          </div>
          {hubBusy && <Spinner size="sm" />}
          {hubError && <p className="text-xs text-red-400">{hubError}</p>}

          {!selectedRepo &&
            repos?.map((r) => (
              <button
                key={r.repo_id}
                onClick={() => void openRepo(r.repo_id)}
                className="w-full text-left p-2 rounded border border-slate-700/60 hover:border-slate-500 flex items-center justify-between"
              >
                <span className="text-sm text-slate-200 truncate">{r.repo_id}</span>
                <span className="text-xs text-slate-500 shrink-0">
                  ♥ {r.likes} · ↓ {r.downloads}
                </span>
              </button>
            ))}
          {!selectedRepo && repos?.length === 0 && !hubBusy && (
            <p className="text-xs text-slate-500">
              No public packs found for tag "mistudio-cluster-definition".
            </p>
          )}

          {selectedRepo && (
            <div className="space-y-2">
              <div className="flex items-center gap-2">
                <Button variant="ghost" size="sm" onClick={() => setSelectedRepo(null)}>
                  ← back
                </Button>
                <Badge variant="default">{selectedRepo}</Badge>
              </div>
              {definitions?.map((d) => (
                <div
                  key={d.file}
                  className="p-2 rounded border border-slate-700/60 flex items-center justify-between gap-2"
                >
                  <div className="min-w-0">
                    <div className="text-sm text-slate-200 truncate">
                      {d.name ?? d.file}
                    </div>
                    <div className="text-xs text-slate-500">
                      {d.file}
                      {d.member_count != null && ` · ${d.member_count} members`}
                      {d.base_model && ` · ${d.base_model}`}
                    </div>
                  </div>
                  <Button
                    variant="primary"
                    size="sm"
                    disabled={isImporting}
                    onClick={() => void importFromHub(d.file)}
                  >
                    Import
                  </Button>
                </div>
              ))}
              {definitions?.length === 0 && !hubBusy && (
                <p className="text-xs text-slate-500">No .cluster.json files in this repo.</p>
              )}
            </div>
          )}
        </div>
      )}
    </Modal>
  );
}
