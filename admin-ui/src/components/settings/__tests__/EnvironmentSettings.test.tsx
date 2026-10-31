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
 *   * the text control (ENH-006 D78) is dropped or sends a number -> "the endpoint card saves text"
 *   * the Telemetry group's note is removed                      -> "telemetry export settings ..."
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
import { TELEMETRY_NOTE, parseDraft } from '../environmentValues';
import { OTHER_GROUP, SETTING_GROUPS, groupIdOf, groupSettings } from '../environmentGroups';
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

/** ENH-006 D75/D78: the telemetry export settings are runtime rows in the Telemetry group. */
const TEL_EFFECT = 'next export; drains and restarts the export pipeline';
const TRACES = row({
  key: 'TELEMETRY_TRACES_ENABLED',
  label: 'Export traces',
  group: 'runtime',
  effect: TEL_EFFECT,
  type: 'bool',
  value: false,
  env_value: false,
  runtime_writable: true,
});
const ENDPOINT = row({
  key: 'TELEMETRY_ENDPOINT',
  label: 'Receiver endpoint',
  group: 'runtime',
  effect: TEL_EFFECT,
  type: 'text',
  // The server sends the endpoint REDACTED (scheme, host, port, path).
  value: 'https://otlp.example:4318/base',
  env_value: 'https://deploy.example:4318',
  source: 'override',
  runtime_writable: true,
  updated_at: '2026-10-28T12:00:00+00:00',
  updated_by: 'agent:planner',
});

function telemetryListing(): EnvironmentListing {
  return {
    settings: [
      LEVEL,
      TRACES,
      ENDPOINT,
      row({ key: 'TELEMETRY_HEADERS', secret: true, set: true }),
      row({ key: 'TELEMETRY_TLS_CA_FILE', value: '/etc/otel/ca.pem' }),
      row({ key: 'OTEL_SDK_DISABLED', value: false }),
    ],
    counts: { runtime: 3, model_load: 0, restart: 0, environment: 3 },
    runtime_keys: ['LOG_LEVEL', 'TELEMETRY_TRACES_ENABLED', 'TELEMETRY_ENDPOINT'],
  };
}

describe('telemetry export settings (ENH-006 D78)', () => {
  beforeEach(() => api.list.mockResolvedValue(telemetryListing()));

  it('telemetry export settings are writable cards in the Telemetry group, with the note', async () => {
    render(<Harness />);
    const group = await screen.findByTestId('env-runtime-group-telemetry');
    expect(within(group).getByTestId('env-row-TELEMETRY_TRACES_ENABLED')).toBeInTheDocument();
    expect(within(group).getByTestId('env-row-TELEMETRY_ENDPOINT')).toBeInTheDocument();
    expect(within(group).getByTestId('env-row-LOG_LEVEL')).toBeInTheDocument();
    expect(within(group).getByTestId('env-telemetry-note')).toHaveTextContent(TELEMETRY_NOTE);
    const endpoint = screen.getByTestId('env-row-TELEMETRY_ENDPOINT');
    expect(within(endpoint).getByText(`Takes effect: ${TEL_EFFECT}`)).toBeInTheDocument();
    expect(screen.getByTestId('env-values-TELEMETRY_ENDPOINT')).toHaveTextContent(
      'In force: https://otlp.example:4318/base · Environment value: https://deploy.example:4318'
    );
    expect(screen.getByTestId('env-last-TELEMETRY_ENDPOINT')).toHaveTextContent('by agent:planner');
    // The environment-only ones stay read-only, and the credentials only as set.
    for (const key of ['TELEMETRY_HEADERS', 'TELEMETRY_TLS_CA_FILE', 'OTEL_SDK_DISABLED']) {
      expect(screen.queryByTestId(`env-row-${key}`)).toBeNull();
    }
    expect(screen.getByTestId('env-fixed-TELEMETRY_HEADERS')).toHaveTextContent(/^set$/);
  });

  it('the endpoint card saves text, as typed and trimmed', async () => {
    api.set.mockResolvedValue(ENDPOINT);
    render(<Harness />);
    const input = await screen.findByTestId('env-input-TELEMETRY_ENDPOINT');
    expect(input).toHaveAttribute('type', 'text');
    await userEvent.clear(input);
    await userEvent.type(input, '  http://collector.local:4318  ');
    await userEvent.type(screen.getByTestId('env-reason-TELEMETRY_ENDPOINT'), 'pilot');
    await userEvent.click(screen.getByTestId('env-save-TELEMETRY_ENDPOINT'));
    await waitFor(() => expect(api.set).toHaveBeenCalledTimes(1));
    expect(api.set).toHaveBeenCalledWith('TELEMETRY_ENDPOINT', 'http://collector.local:4318', 'pilot');
  });

  it('a switch turns export on with a JSON boolean, and the server refusal shows on the row', async () => {
    api.set.mockRejectedValueOnce(
      new ApiError('SETTING_INVALID', 'TELEMETRY_ENDPOINT: the endpoint may not carry a query string.')
    );
    api.set.mockResolvedValueOnce({ ...TRACES, value: true, source: 'override' });
    render(<Harness />);
    const input = await screen.findByTestId('env-input-TELEMETRY_ENDPOINT');
    await userEvent.clear(input);
    await userEvent.type(input, 'https://h/?k=1');
    await userEvent.click(screen.getByTestId('env-save-TELEMETRY_ENDPOINT'));
    expect(await screen.findByTestId('env-problem-TELEMETRY_ENDPOINT')).toHaveTextContent(
      'may not carry a query string'
    );
    await userEvent.selectOptions(screen.getByTestId('env-input-TELEMETRY_TRACES_ENABLED'), 'true');
    await userEvent.click(screen.getByTestId('env-save-TELEMETRY_TRACES_ENABLED'));
    await waitFor(() =>
      expect(api.set).toHaveBeenLastCalledWith('TELEMETRY_TRACES_ENABLED', true, undefined)
    );
  });

  it('an overridden endpoint returns to the deployment value through DELETE', async () => {
    api.reset.mockResolvedValue({ ...ENDPOINT, source: 'env' });
    render(<Harness />);
    await userEvent.click(await screen.findByTestId('env-reset-TELEMETRY_ENDPOINT'));
    await waitFor(() => expect(api.reset).toHaveBeenCalledWith('TELEMETRY_ENDPOINT', undefined));
  });

  it('parseDraft keeps text as text', () => {
    expect(parseDraft(ENDPOINT, '  https://x:4318 ')).toEqual({ value: 'https://x:4318' });
    expect(parseDraft(ENDPOINT, '')).toEqual({ value: '' });
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


/** Every setting miLLM had on 2026-10-28 (`runtime_settings.all_setting_names()`, 186 keys with
 *  ENH-008's APPROVAL_TTL_HOURS). A new
 *  key is never lost — it lands in "Other settings" — but every key KNOWN today has a group. */
const KEYS_2026_10_28 = (
  'APPROVAL_TTL_HOURS AUTO_LOAD_MODEL BATCH_CHUNK_ROWS BATCH_ERRORS_SHOWN BATCH_FILES_DIR BATCH_FILE_RETENTION_DAYS ' +
  'BATCH_LEASE_TTL_S BATCH_MAX_COMPLETION_WINDOW_HOURS BATCH_MAX_FILE_BYTES BATCH_MAX_LINE_BYTES ' +
  'BATCH_MAX_ROWS BATCH_PACK_DEFAULT BATCH_PACK_MAX_ROWS BATCH_PACK_MAX_TOKENS ' +
  'BATCH_PROGRESS_MIN_INTERVAL_S BATCH_RETENTION_INTERVAL_S BATCH_WAIT_POLL_S ' +
  'CBM_DEFAULT_MAX_TOKENS CBM_DEFAULT_TEMPERATURE CBM_DEFAULT_TOP_P CBM_FORCE_SERIAL_MONITORING ' +
  'CBM_MAX_QUEUE_SIZE CIRCUIT_ALLOW_CONCURRENT CIRCUIT_INTENSITY_MAX CIRCUIT_INTENSITY_MIN ' +
  'CIRCUIT_SENSING_CONTEXT_TOKENS CIRCUIT_SENSING_EPSILON CIRCUIT_SENSING_FORCE_SERIAL ' +
  'CIRCUIT_SENSING_MAX_AGE_DAYS CIRCUIT_SENSING_MAX_BATCH_EVENTS_PER_CIRCUIT ' +
  'CIRCUIT_SENSING_MAX_EVENTS_PER_CIRCUIT CIRCUIT_SENSING_MAX_EVENTS_PER_REQUEST ' +
  'CIRCUIT_SENSING_MAX_OVERHEAD_MS CIRCUIT_SENSING_MAX_TOKEN_LAG CIRCUIT_SENSING_THETA_FLOOR ' +
  'CLUSTER_HUB_CACHE_TTL_S CLUSTER_HUB_TAG CLUSTER_INTENSITY_MAX CLUSTER_INTENSITY_MIN CORS_ORIGINS ' +
  'DATABASE_URL DEBUG EMBEDDINGS_MAX_INPUTS ENABLE_CONTINUOUS_BATCHING GGUF_CONTEXT_LENGTH ' +
  'GGUF_ENABLE_EMBEDDINGS GGUF_FLASH_ATTENTION GGUF_KV_CACHE_TYPE GGUF_TENSOR_SPLIT ' +
  'GRACEFUL_UNLOAD_TIMEOUT HF_TOKEN HOST IGNORED_FIELDS_HEADER_MAX_BYTES KV_CACHE_MODE ' +
  'LEASE_DEFAULT_TTL_SECONDS LEASE_ENDED_MEMORY LEASE_HOLDER_MAX_CHARS LEASE_MAX_TTL_SECONDS ' +
  'LEASE_REASON_MAX_CHARS LOG_FORMAT LOG_LEVEL MAX_CONCURRENT_REQUESTS MAX_PENDING_REQUESTS ' +
  'MODEL_CACHE_DIR MULTISAE_ATTACH_DTYPE MULTISAE_VRAM_ENVELOPE_MB OTEL_SDK_DISABLED PORT ' +
  'PROBE_BUDGET_HEADROOM PROBE_CAPACITY_DECODE_STEPS PROBE_COST_SAMPLES PROBE_EVENT_CONTEXT_TOKENS ' +
  'PROBE_FORCE_SERIAL PROBE_HUB_CACHE_TTL_S PROBE_HUB_TAG PROBE_LIST_PAGE_SIZE PROBE_MAX_AGE_DAYS ' +
  'PROBE_MAX_ARMED PROBE_MAX_ARMED_BEHAVIOURS PROBE_MAX_BATCH_EVENTS_PER_PROBE ' +
  'PROBE_MAX_EVENTS_PER_PROBE PROBE_MAX_IMPORTED PROBE_MAX_IMPORT_BYTES PROBE_MAX_OVERHEAD_MS ' +
  'PROBE_MAX_OVERHEAD_MS_PER_PASS PROBE_MAX_PER_BEHAVIOUR PROBE_MAX_PREFILL_MS_PER_1K ' +
  'PROBE_PARITY_MATCHED_DTYPE_FLOOR PROBE_PARITY_MATCHED_RELATIVE_FLOOR ' +
  'PROBE_PARITY_MIN_ROBUST_FRACTION PROBE_PARITY_SCORE_TOLERANCE PROBE_PARITY_TOLERANCE ' +
  'PROBE_RECOMMEND_MAX_DECODE_SHARE PROBE_RECOMMEND_MAX_PREFILL_SHARE PROBE_RECOMMEND_TARGET_SETS ' +
  'PROBE_REFERENCE_SET_SHAPE PROBE_RETENTION_INTERVAL_S PROBE_SCORE_MAX_INPUTS ' +
  'PROBE_SESSION_IDLE_MINUTES PROBE_SESSION_MAX_AGE_DAYS PROBE_TRANSCRIPTS ' +
  'PROBE_TRANSCRIPT_MAX_AGE_DAYS PROBE_TRANSCRIPT_MAX_TURNS PROBE_VERDICT_HEADER_MAX_BYTES ' +
  'QUEUE_DURATION_WINDOW RETRY_AFTER_FALLBACK_S RETRY_AFTER_LOAD_S RETRY_AFTER_MAX_S ' +
  'RETRY_AFTER_MEMORY_S RETRY_AFTER_NOT_LOADED_S RETRY_AFTER_QUEUE_DEFAULT_S ' +
  'RETRY_AFTER_READINESS_S RETRY_AFTER_UNLOAD_S RETRY_AFTER_WEIGHTS_TORN_S ' +
  'RULE_CODE_CALLS_PER_REQUEST RULE_CODE_CALL_MEMORY_MIB RULE_CODE_CALL_OUTPUT_BYTES ' +
  'RULE_CODE_CALL_WALL_MS RULE_CODE_MODULES_MAX_BYTES RULE_RUNNER_SOCKET RULE_SETS_MAX_IMPORTED ' +
  'RULE_SET_MAX_BYTES RULE_SET_MAX_SIGNALS RULE_SIGNALS_ENABLED RULE_TIME_BUDGET_MS ' +
  'SAE_ACTIVATIONS_ENCODE_CHUNK SAE_ACTIVATIONS_MAX_ENTRIES SAE_ACTIVATIONS_MAX_TOP_K SAE_CACHE_DIR ' +
  'SCORING_FLOAT32_DEVICE SCORING_FLOAT32_HEADROOM_MB SENSING_CONTEXT_TOKENS SENSING_DEDUP_HISTORY ' +
  'SENSING_EPSILON SENSING_FORCE_SERIAL SENSING_MAX_AGE_DAYS SENSING_MAX_BATCH_EVENTS_PER_CLUSTER ' +
  'SENSING_MAX_EVENTS_PER_CLUSTER SENSING_MAX_EVENTS_PER_REQUEST SENSING_MAX_OVERHEAD_MS ' +
  'SENSING_THETA_FLOOR SETTINGS_ENCRYPTION_KEY SPECULATIVE_MODEL SPECULATIVE_NUM_TOKENS ' +
  'STRUCTURED_OUTPUT_GRAMMAR_CACHE TELEMETRY_ALLOW_PLAINTEXT TELEMETRY_BATCH_SIZE ' +
  'TELEMETRY_DEPLOYMENT_ENVIRONMENT TELEMETRY_DRAIN_DEADLINE_S TELEMETRY_ENDPOINT ' +
  'TELEMETRY_EXPORT_TIMEOUT_S TELEMETRY_HEADERS TELEMETRY_HEALTH_DEGRADED_WINDOW_S ' +
  'TELEMETRY_HEALTH_UNAVAILABLE_AFTER_S TELEMETRY_LOGS_ENABLED TELEMETRY_METRICS_ENABLED ' +
  'TELEMETRY_METRIC_INTERVAL_S TELEMETRY_PROTOCOL TELEMETRY_QUEUE_SIZE TELEMETRY_RESULT_LOG_ORIGINS ' +
  'TELEMETRY_RETRY_HORIZON_S TELEMETRY_SERIES_CAP TELEMETRY_SERVICE_INSTANCE_ID ' +
  'TELEMETRY_SERVICE_NAME TELEMETRY_STATELESS_EXPORT_PERMITTED TELEMETRY_TLS_CA_FILE ' +
  'TELEMETRY_TLS_CLIENT_CERT_FILE TELEMETRY_TLS_CLIENT_KEY_FILE TELEMETRY_TRACES_ENABLED ' +
  'TELEMETRY_TRACES_SAMPLER_RATIO TORCH_COMPILE TORCH_COMPILE_MODE TRANSFORMERS_CUDA_CONTEXT_MB ' +
  'TRANSFORMERS_IDLE_CACHE_RELEASE_S TRANSFORMERS_MIN_CONTEXT WEIGHTS_CHECKPOINT_DIR ' +
  'WEIGHTS_CHECKPOINT_READ_ROOTS WEIGHTS_COMMIT_WATCHDOG_MS WEIGHTS_HEARTBEAT_INTERVAL_HINT_S ' +
  'WEIGHTS_HEARTBEAT_TIMEOUT_S WEIGHTS_HOLD_TIMEOUT_S WEIGHTS_LINEAGE_MAX_DEPTH ' +
  'WEIGHTS_QUIESCE_DRAIN_TIMEOUT_S WEIGHTS_SAVE_FREE_SPACE_FACTOR WEIGHTS_SAVE_OVERHEAD_MB ' +
  'WEIGHTS_TOKENS '
).trim().split(' ');

describe('Environment settings are grouped for operators (2026-10-28)', () => {
  it('every key miLLM has today belongs to a named group', () => {
    expect(KEYS_2026_10_28).toHaveLength(186);
    const ungrouped = KEYS_2026_10_28.filter((k) => groupIdOf(k) === OTHER_GROUP.id);
    expect(ungrouped).toEqual([]);
  });

  it.each([
    ['CIRCUIT_SENSING_EPSILON', 'sensing'],
    ['CIRCUIT_INTENSITY_MAX', 'steering'],
    ['SENSING_EPSILON', 'sensing'],
    ['PROBE_MAX_ARMED', 'probes'],
    ['RULE_SIGNALS_ENABLED', 'rules'],
    ['TELEMETRY_ENDPOINT', 'telemetry'],
    ['LOG_LEVEL', 'telemetry'],
    ['MAX_PENDING_REQUESTS', 'serving'],
    ['GGUF_CONTEXT_LENGTH', 'models'],
    ['WEIGHTS_TOKENS', 'weights'],
    ['HOST', 'server'],
    ['BATCH_MAX_ROWS', 'batch'],
    ['APPROVAL_TTL_HOURS', 'approvals'],
  ])('%s is in %s (longest prefix wins)', (key, group) => {
    expect(groupIdOf(key)).toBe(group);
  });

  it('an unknown key is kept, in Other settings, last', () => {
    const groups = groupSettings([row({ key: 'ZZ_NEW' }), row({ key: 'HOST' })]);
    expect(groups.map((g) => g.id)).toEqual(['server', 'other']);
  });

  it('groups follow the fixed order and rows sort by label, case-insensitively', () => {
    const groups = groupSettings([
      row({ key: 'PROBE_B', label: 'beta' }),
      row({ key: 'MAX_PENDING_REQUESTS', label: 'Queue length' }),
      row({ key: 'PROBE_A', label: 'Alpha' }),
      row({ key: 'PROBE_C', label: 'gamma 10' }),
      row({ key: 'PROBE_D', label: 'gamma 9' }),
      row({ key: 'LOG_LEVEL', label: 'Log level' }),
    ]);
    const order = SETTING_GROUPS.map((g) => g.id);
    const ids = groups.map((g) => g.id);
    expect(ids).toEqual(['serving', 'probes', 'telemetry']);
    expect([...ids].sort((a, b) => order.indexOf(a) - order.indexOf(b))).toEqual(ids);
    expect(groups[1].rows.map((r) => r.label)).toEqual(['Alpha', 'beta', 'gamma 9', 'gamma 10']);
  });

  it('each section renders its groups in order, with the rows sorted by label', async () => {
    render(<Harness />);
    await screen.findByTestId('env-row-LOG_LEVEL');
    const runtime = screen.getByTestId('env-runtime');
    const titles = within(runtime).getAllByRole('heading', { level: 4 }).map((h) => h.textContent);
    // MAX_PENDING_REQUESTS (serving), RULE_SIGNALS_ENABLED (rules), LOG_LEVEL (telemetry).
    expect(titles).toEqual([
      'Serving and requests (1)', 'Rule signals and operator code (1)', 'Telemetry and logging (1)',
    ]);
    const fixed = screen.getByTestId('env-environment');
    expect(within(fixed).getAllByTestId('env-fixed-key').map((d) => d.textContent)).toEqual([
      'HF_TOKEN', 'WEIGHTS_TOKENS', 'HOST',
    ]);
    expect(within(fixed).getByTestId('env-environment-group-models')).toHaveTextContent('HF_TOKEN');
  });

  it('sorts a section alphabetically within a group', async () => {
    const base = listing();
    base.settings = [
      row({ key: 'PROBE_Z', label: 'PROBE_Z' }),
      row({ key: 'PROBE_A', label: 'PROBE_A' }),
      row({ key: 'PROBE_M', label: 'PROBE_M' }),
    ];
    api.list.mockResolvedValue(base);
    render(<Harness />);
    const fixed = await screen.findByTestId('env-environment-group-probes');
    expect(within(fixed).getAllByTestId('env-fixed-key').map((d) => d.textContent)).toEqual([
      'PROBE_A', 'PROBE_M', 'PROBE_Z',
    ]);
  });
});
