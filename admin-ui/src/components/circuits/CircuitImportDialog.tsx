/**
 * CircuitImportDialog (Feature 13) — two ways in: paste JSON or upload a
 * .circuit.json file. (Hub browse is Feature 15 scope.)
 *
 * Parse errors are surfaced in-dialog so the user sees WHY a paste failed
 * without losing their input.
 */

import { useRef, useState } from 'react';
import { FileJson, Upload } from 'lucide-react';

import { Button, Modal } from '@components/common';

type Tab = 'paste' | 'file';

interface CircuitImportDialogProps {
  open: boolean;
  onClose: () => void;
  onImport: (payload: unknown) => Promise<unknown>;
  isImporting?: boolean;
}

export function CircuitImportDialog({
  open,
  onClose,
  onImport,
  isImporting = false,
}: CircuitImportDialogProps) {
  const [tab, setTab] = useState<Tab>('paste');
  const [pasted, setPasted] = useState('');
  const [parseError, setParseError] = useState<string | null>(null);
  const fileRef = useRef<HTMLInputElement>(null);

  const submitJson = async (text: string) => {
    let payload: unknown;
    try {
      payload = JSON.parse(text);
    } catch (e) {
      setParseError(
        `That is not valid JSON: ${e instanceof Error ? e.message : 'parse error'}`,
      );
      return;
    }
    setParseError(null);
    try {
      await onImport(payload);
      setPasted('');
      onClose();
    } catch {
      // The hook's onError toast already reported it; keep the dialog open so
      // the user can correct the document.
    }
  };

  const handleFile = async (file: File) => {
    await submitJson(await file.text());
  };

  return (
    <Modal id="circuit-import" isOpen={open} onClose={onClose} title="Import circuit definition">
      <div className="space-y-4">
        <div className="flex gap-2 border-b border-slate-700">
          {(['paste', 'file'] as Tab[]).map((t) => (
            <button
              key={t}
              type="button"
              onClick={() => setTab(t)}
              data-testid={`tab-${t}`}
              className={`px-3 py-1.5 text-sm border-b-2 -mb-px ${
                tab === t
                  ? 'border-cyan-400 text-slate-100'
                  : 'border-transparent text-slate-400 hover:text-slate-200'
              }`}
            >
              {t === 'paste' ? 'Paste JSON' : 'Upload file'}
            </button>
          ))}
        </div>

        {tab === 'paste' && (
          <div className="space-y-2">
            <textarea
              data-testid="paste-area"
              value={pasted}
              onChange={(e) => setPasted(e.target.value)}
              rows={10}
              spellCheck={false}
              placeholder='{"kind": "mistudio.circuit-definition", "schema_version": "1", ...}'
              className="w-full rounded border border-slate-700 bg-slate-900 p-2 font-mono text-xs text-slate-200"
            />
            <div className="flex justify-end">
              <Button
                variant="primary"
                size="sm"
                disabled={!pasted.trim() || isImporting}
                onClick={() => submitJson(pasted)}
                data-testid="import-paste"
              >
                <FileJson className="w-4 h-4 mr-1" />
                Import
              </Button>
            </div>
          </div>
        )}

        {tab === 'file' && (
          <div className="space-y-2">
            <input
              ref={fileRef}
              type="file"
              accept=".json,application/json"
              data-testid="file-input"
              onChange={(e) => {
                const f = e.target.files?.[0];
                if (f) void handleFile(f);
              }}
              className="block w-full text-sm text-slate-300 file:mr-3 file:rounded file:border-0 file:bg-slate-700 file:px-3 file:py-1.5 file:text-slate-100"
            />
            <p className="text-xs text-slate-500 flex items-center gap-1">
              <Upload className="w-3 h-3" />
              A <code className="font-mono">.circuit.json</code> exported from miStudio.
            </p>
          </div>
        )}

        {parseError && (
          <p data-testid="parse-error" className="text-xs text-red-400">
            {parseError}
          </p>
        )}
      </div>
    </Modal>
  );
}
