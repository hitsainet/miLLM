/**
 * Feature 35: the Settings page's Environment section (miForge's layout).
 *
 * MUTATION CONTROLS (each must turn this file red):
 *   * SettingsPage stops rendering <EnvironmentSettings>      -> "the Settings page shows the section"
 *   * Save sends a string for a number, or drops the reason   -> "Save sends the typed number and reason"
 *   * the refusal is not shown on the row                     -> "a refusal is shown on its row"
 *   * "Use the environment value" stops calling DELETE        -> "reset calls DELETE"
 *   * a secret renders its value                              -> "secrets read set / not set"
 *   * the effect badge or env value disappears                -> "a runtime row shows"
 */

import { beforeEach, describe, expect, it, vi } from 'vitest';
import { render, screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import type { EnvironmentListing, EnvironmentRow } from '@/types/environment';

const api = vi.hoisted(() => ({
  list: vi.fn(),
  set: vi.fn(),
  reset: vi.fn(),
  history: vi.fn(),
}));
const settings = vi.hoisted(() => ({
  getHfToken: vi.fn(),
  saveHfToken: vi.fn(),
  clearHfToken: vi.fn(),
  testHfToken: vi.fn(),
}));

vi.mock('@/services/api', async (importOriginal) => {
  const actual = await importOriginal<typeof import('@/services/api')>();
  return { ...actual, environmentApi: api, settingsApi: settings };
});

import { EnvironmentSettings } from '../EnvironmentSettings';
import { parseDraft } from '../environmentValues';
import { useEnvironmentSettings } from '@/hooks/useEnvironmentSettings';
import { SettingsPage } from '@pages/SettingsPage';
import { ApiError } from '@/services/api';

function row(over: Partial<EnvironmentRow>): EnvironmentRow {
  return {
    key: 'X',
    label: 'X',
    description: null,
    group: 'environment',
    effect: 'set in the environment; restart millm-backend',
    type: null,
    secret: false,
    set: null,
    value: null,
    env_value: null,
    source: 'env',
    runtime_writable: false,
    minimum: null,
    maximum: null,
    choices: null,
    updated_at: null,
    updated_by: null,
    reason: null,
    ...over,
  };
}

const QUEUE = row({
  key: 'MAX_PENDING_REQUESTS',
  label: 'Queue length',
  description: 'How many requests may wait for the model at once.',
  group: 'runtime',
  effect: 'next queue admission',
  type: 'int',
  value: 10,
  env_value: 10,
  runtime_writable: true,
  minimum: 1,
  maximum: 1000,
});
const LEVEL = row({
  key: 'LOG_LEVEL',
  label: 'Log level',
  group: 'runtime',
  effect: 'now',
  type: 'choice',
  value: 'WARNING',
  env_value: 'INFO',
  source: 'override',
  runtime_writable: true,
  choices: ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
  updated_at: '2026-10-20T12:00:00+00:00',
  updated_by: 'operator',
  reason: 'too chatty',
});

const SWITCH = row({
  key: 'RULE_SIGNALS_ENABLED',
  label: 'Rule signals',
  group: 'runtime',
  effect: 'next request',
  type: 'bool',
  value: true,
  env_value: true,
  runtime_writable: true,
});

function listing(): EnvironmentListing {
  const settingsRows = [
    LEVEL,
    QUEUE,
    SWITCH,
    row({ key: 'TORCH_COMPILE', group: 'model_load', effect: 'next model load', value: null }),
    row({ key: 'KV_CACHE_MODE', group: 'restart', value: 'dynamic' }),
    row({ key: 'HOST', value: '0.0.0.0' }),
    row({ key: 'HF_TOKEN', secret: true, set: true }),
    row({ key: 'WEIGHTS_TOKENS', secret: true, set: false }),
  ];
  return {
    settings: settingsRows,
    counts: { runtime: 3, model_load: 1, restart: 1, environment: 3 },
    runtime_keys: ['LOG_LEVEL', 'MAX_PENDING_REQUESTS', 'RULE_SIGNALS_ENABLED'],
  };
}

function Harness() {
  return <EnvironmentSettings state={useEnvironmentSettings()} />;
}

beforeEach(() => {
  vi.clearAllMocks();
  api.list.mockResolvedValue(listing());
  api.history.mockResolvedValue({
    changes: [
      {
        id: 1,
        key: 'LOG_LEVEL',
        action: 'set',
        previous_value: 'INFO',
        new_value: 'WARNING',
        actor: 'operator',
        approval_id: null,
        reason: 'too chatty',
        changed_at: '2026-10-20T12:00:00+00:00',
      },
    ],
  });
  settings.getHfToken.mockResolvedValue({
    configured: false,
    source: 'none',
    masked: null,
    last_verified_at: null,
    user: null,
    stored: false,
    stored_error: null,
    last_check_error: null,
    encryption_available: true,
    env_configured: false,
  });
});

describe('EnvironmentSettings', () => {
  it('a runtime row shows its name, effect, environment value and last change', async () => {
    render(<Harness />);
    const level = await screen.findByTestId('env-row-LOG_LEVEL');
    expect(within(level).getByText('Log level')).toBeInTheDocument();
    expect(within(level).getByText('Takes effect: now')).toBeInTheDocument();
    expect(within(level).getByText('override')).toBeInTheDocument();
    expect(screen.getByTestId('env-values-LOG_LEVEL')).toHaveTextContent(
      'In force: WARNING · Environment value: INFO'
    );
    expect(screen.getByTestId('env-last-LOG_LEVEL')).toHaveTextContent(/by operator.*too chatty/);
    expect(screen.getByTestId('env-values-MAX_PENDING_REQUESTS')).toHaveTextContent('Allowed: 1 to 1000');
    expect(screen.getByTestId('env-runtime-count')).toHaveTextContent('3 of 8 settings');
  });

  it('read-only sections list their keys; secrets read set / not set', async () => {
    render(<Harness />);
    await screen.findByTestId('env-section');
    expect(within(screen.getByTestId('env-model-load')).getByText('TORCH_COMPILE')).toBeInTheDocument();
    expect(within(screen.getByTestId('env-restart')).getByText('KV_CACHE_MODE')).toBeInTheDocument();
    expect(screen.getByTestId('env-fixed-HOST')).toHaveTextContent('0.0.0.0');
    expect(screen.getByTestId('env-fixed-HF_TOKEN')).toHaveTextContent(/^set$/);
    expect(screen.getByTestId('env-fixed-WEIGHTS_TOKENS')).toHaveTextContent(/^not set$/);
    expect(screen.queryByTestId('env-row-HOST')).toBeNull();
  });

  it('Save sends the typed number and reason, then re-reads', async () => {
    api.set.mockResolvedValue({ ...QUEUE, value: 4, source: 'override' });
    render(<Harness />);
    const input = await screen.findByTestId('env-input-MAX_PENDING_REQUESTS');
    await userEvent.clear(input);
    await userEvent.type(input, '4');
    await userEvent.type(screen.getByTestId('env-reason-MAX_PENDING_REQUESTS'), 'busy day');
    await userEvent.click(screen.getByTestId('env-save-MAX_PENDING_REQUESTS'));
    await waitFor(() => expect(api.set).toHaveBeenCalledTimes(1));
    expect(api.set).toHaveBeenCalledWith('MAX_PENDING_REQUESTS', 4, 'busy day');
    await waitFor(() => expect(api.list).toHaveBeenCalledTimes(2));
  });

  it('a choice saves the chosen value with no reason', async () => {
    api.set.mockResolvedValue(LEVEL);
    render(<Harness />);
    const select = await screen.findByTestId('env-input-LOG_LEVEL');
    await userEvent.selectOptions(select, 'ERROR');
    await userEvent.click(screen.getByTestId('env-save-LOG_LEVEL'));
    await waitFor(() => expect(api.set).toHaveBeenCalledWith('LOG_LEVEL', 'ERROR', undefined));
  });

  it('a switch saves a JSON boolean, never the string (039 S2)', async () => {
    api.set.mockResolvedValue({ ...SWITCH, value: false });
    render(<Harness />);
    const select = await screen.findByTestId('env-input-RULE_SIGNALS_ENABLED');
    expect((select as HTMLSelectElement).value).toBe('true');
    await userEvent.selectOptions(select, 'false');
    await userEvent.click(screen.getByTestId('env-save-RULE_SIGNALS_ENABLED'));
    await waitFor(() => expect(api.set).toHaveBeenCalledTimes(1));
    expect(api.set).toHaveBeenCalledWith('RULE_SIGNALS_ENABLED', false, undefined);
  });

  it('a refusal is shown on its row and nothing is re-read', async () => {
    api.set.mockRejectedValue(
      new ApiError('SETTING_INVALID', 'MAX_PENDING_REQUESTS must be at most 1000, got 5000.')
    );
    render(<Harness />);
    const input = await screen.findByTestId('env-input-MAX_PENDING_REQUESTS');
    await userEvent.clear(input);
    await userEvent.type(input, '5000');
    await userEvent.click(screen.getByTestId('env-save-MAX_PENDING_REQUESTS'));
    expect(await screen.findByTestId('env-problem-MAX_PENDING_REQUESTS')).toHaveTextContent(
      'must be at most 1000'
    );
    expect(api.list).toHaveBeenCalledTimes(1);
  });

  it('a non-number is refused before anything is sent', async () => {
    render(<Harness />);
    const input = await screen.findByTestId('env-input-MAX_PENDING_REQUESTS');
    await userEvent.clear(input);
    await userEvent.click(screen.getByTestId('env-save-MAX_PENDING_REQUESTS'));
    expect(await screen.findByTestId('env-problem-MAX_PENDING_REQUESTS')).toHaveTextContent(
      'Enter a number'
    );
    expect(api.set).not.toHaveBeenCalled();
  });

  it('reset calls DELETE only on an overridden row', async () => {
    api.reset.mockResolvedValue({ ...LEVEL, source: 'env', value: 'INFO' });
    render(<Harness />);
    await screen.findByTestId('env-row-LOG_LEVEL');
    expect(screen.queryByTestId('env-reset-MAX_PENDING_REQUESTS')).toBeNull();
    await userEvent.click(screen.getByTestId('env-reset-LOG_LEVEL'));
    await waitFor(() => expect(api.reset).toHaveBeenCalledWith('LOG_LEVEL', undefined));
  });

  it('shows the change history', async () => {
    render(<Harness />);
    const history = await screen.findByTestId('env-history');
    expect(history).toHaveTextContent(/LOG_LEVEL set to WARNING \(was INFO\) by operator/);
  });

  it('a failed read is reported, not guessed', async () => {
    api.list.mockRejectedValue(new ApiError('SERVER_ERROR', 'Server returned 502 error'));
    render(<Harness />);
    expect(await screen.findByTestId('env-error')).toHaveTextContent('502');
    expect(screen.queryByTestId('env-section')).toBeNull();
  });

  it('parseDraft keeps whole numbers whole', () => {
    expect(parseDraft(QUEUE, '2.5')).toEqual({ error: 'Enter a whole number' });
    expect(parseDraft({ ...QUEUE, type: 'float' }, '2.5')).toEqual({ value: 2.5 });
  });
});

describe('the Settings page', () => {
  it('shows the section', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue({ ok: false, json: async () => null }));
    render(<SettingsPage />);
    expect(await screen.findByTestId('env-row-MAX_PENDING_REQUESTS')).toBeInTheDocument();
    expect(api.list).toHaveBeenCalledTimes(1);
    expect(api.history).toHaveBeenCalledWith(50);
    vi.unstubAllGlobals();
  });
});
