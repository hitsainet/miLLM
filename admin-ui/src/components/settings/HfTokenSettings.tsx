/**
 * Feature 34: the Settings page's "Hugging Face token" card.
 *
 * Save stores the token on the server, encrypted; downloads and previews then use it whenever the
 * request carries no token of its own. The server never sends the token back: this card shows the
 * masked form, which tier is active, and the last successful check (Hugging Face `whoami`).
 *
 * The typed value lives only in this component's state and is cleared after Save, so it is not
 * left sitting in the field.
 */

import { useState } from 'react';
import { KeyRound, Save, ShieldCheck, Trash2 } from 'lucide-react';
import { Badge, Button, Card, CardHeader, Input } from '@components/common';
import { ApiError, settingsApi } from '@/services/api';
import type { HfTokenStatus } from '@/types/settings';

const SOURCE_LABEL: Record<HfTokenStatus['source'], string> = {
  settings: 'Saved token',
  env: 'Server HF_TOKEN',
  none: 'None (anonymous)',
};

const STORED_ERROR: Record<string, string> = {
  key_missing:
    'A token is saved but the server has no SETTINGS_ENCRYPTION_KEY, so it cannot be read and is not used.',
  key_mismatch:
    'A token is saved under a different encryption key, so it cannot be read and is not used. Save it again.',
  corrupt: 'The saved token could not be read and is not used. Save it again.',
};

function message(error: unknown): string {
  return error instanceof ApiError || error instanceof Error ? error.message : 'Request failed';
}

function formatWhen(iso: string | null): string {
  if (!iso) return 'never';
  const d = new Date(iso);
  return Number.isNaN(d.getTime()) ? iso : d.toLocaleString();
}

export interface HfTokenSettingsProps {
  status: HfTokenStatus | null;
  onStatus: (status: HfTokenStatus) => void;
  /** Re-reads the status (after a Test, which records the verification server-side). */
  onRefresh: () => Promise<void> | void;
}

export function HfTokenSettings({ status, onStatus, onRefresh }: HfTokenSettingsProps) {
  const [draft, setDraft] = useState('');
  const [busy, setBusy] = useState<'save' | 'test' | 'clear' | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);

  const run = async (kind: 'save' | 'test' | 'clear', fn: () => Promise<void>) => {
    setBusy(kind);
    setError(null);
    setNotice(null);
    try {
      await fn();
    } catch (e) {
      setError(message(e));
    } finally {
      setBusy(null);
    }
  };

  const save = () =>
    run('save', async () => {
      const next = await settingsApi.saveHfToken(draft.trim());
      setDraft('');
      onStatus(next);
      // The server keeps the outcome of a Test of THIS token run just before the Save.
      setNotice(
        next.last_verified_at
          ? `Token saved. Verified — Hugging Face user ${next.user ?? '(unnamed)'}.`
          : next.last_check_error
            ? 'Token saved, but its last Test failed (see above).'
            : 'Token saved. Use Test to check it with Hugging Face.'
      );
    });

  const test = () =>
    run('test', async () => {
      const typed = draft.trim();
      let result;
      try {
        result = await settingsApi.testHfToken(typed || undefined);
      } catch (e) {
        // A failed check of the saved token is recorded server-side: re-read it so it shows.
        await onRefresh();
        throw e;
      }
      setNotice(
        `Valid — Hugging Face user ${result.user ?? '(unnamed)'}${
          typed && result.source === 'request'
            ? ' (the typed token; not saved — Save it now and this check is kept with it)'
            : ''
        }.`
      );
      await onRefresh();
    });

  const clear = () =>
    run('clear', async () => {
      onStatus(await settingsApi.clearHfToken());
      setNotice('Saved token cleared.');
    });

  const saveBlocked = status !== null && !status.encryption_available;

  return (
    <Card>
      <CardHeader
        title="Hugging Face token"
        subtitle="Used for gated and private models, SAEs and packs when a request brings no token"
        icon={<KeyRound className="w-5 h-5 text-slate-400" />}
      />
      <div className="stack-y-4">
        <div className="stack-y-2 text-sm" data-testid="hf-token-status">
          <div className="flex justify-between">
            <span className="text-slate-400">Active source</span>
            <Badge variant={status?.source === 'none' ? 'default' : 'success'}>
              {status ? SOURCE_LABEL[status.source] : '—'}
            </Badge>
          </div>
          <div className="flex justify-between">
            <span className="text-slate-400">Token</span>
            <span className="text-slate-200 font-mono">{status?.masked ?? '—'}</span>
          </div>
          <div className="flex justify-between">
            <span className="text-slate-400">Last verified</span>
            <span className="text-slate-200">
              {status?.last_verified_at
                ? `${formatWhen(status.last_verified_at)}${status.user ? ` · ${status.user}` : ''}`
                : 'never'}
            </span>
          </div>
        </div>

        {status?.stored_error && (
          <p className="text-xs text-yellow-400" role="alert">
            {STORED_ERROR[status.stored_error] ?? STORED_ERROR.corrupt}
          </p>
        )}
        {status?.last_check_error === 'invalid' && (
          <p className="text-xs text-red-400" role="alert">
            The saved token was rejected by Hugging Face at its last check (invalid or expired).
          </p>
        )}
        {status?.last_check_error === 'network' && (
          <p className="text-xs text-yellow-400" role="alert">
            The last check of the saved token could not reach Hugging Face; the token itself was not
            judged.
          </p>
        )}
        {saveBlocked && (
          <p className="text-xs text-yellow-400" role="alert">
            Saving is disabled: the server has no SETTINGS_ENCRYPTION_KEY, and a token is never stored
            unencrypted. See Configuration in the manual for the one-line command that creates it.
          </p>
        )}

        {/* An API token, NOT a password: the manager opt-outs stop browsers offering to save it as
            a site credential (same treatment as the download forms). */}
        <Input
          label="New token"
          name="hf-settings-token"
          type="password"
          autoComplete="off"
          data-lpignore="true"
          data-1p-ignore="true"
          data-form-type="other"
          placeholder="hf_xxxxxxxxxxxxxxxxxxxx"
          value={draft}
          onChange={(e) => setDraft(e.target.value)}
          helperText="A token typed into a download form still overrides the saved one for that request."
        />

        <div className="flex flex-wrap items-center gap-2">
          <Button
            size="sm"
            onClick={save}
            disabled={!draft.trim() || busy !== null || saveBlocked}
            loading={busy === 'save'}
            leftIcon={<Save className="w-4 h-4" />}
          >
            Save
          </Button>
          <Button
            size="sm"
            variant="secondary"
            onClick={test}
            disabled={busy !== null || (!draft.trim() && !status?.configured)}
            loading={busy === 'test'}
            leftIcon={<ShieldCheck className="w-4 h-4" />}
          >
            Test
          </Button>
          <Button
            size="sm"
            variant="danger"
            onClick={clear}
            disabled={busy !== null || !status?.stored}
            loading={busy === 'clear'}
            leftIcon={<Trash2 className="w-4 h-4" />}
          >
            Clear
          </Button>
        </div>

        {notice && (
          <p className="text-xs text-green-400" role="status">
            {notice}
          </p>
        )}
        {error && (
          <p className="text-xs text-red-400" role="alert">
            {error}
          </p>
        )}

        <p className="text-xs text-slate-500">
          Order used for every Hugging Face call: a token given with the request, then the saved
          token, then the server&apos;s HF_TOKEN, then anonymous. The token is stored encrypted and
          is never shown again or sent back to this page.
        </p>
      </div>
    </Card>
  );
}

export default HfTokenSettings;
