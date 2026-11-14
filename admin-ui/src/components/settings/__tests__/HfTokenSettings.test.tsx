/**
 * Feature 34: the Settings page's Hugging Face token card, and the download forms' hint.
 *
 * MUTATION CONTROLS (each must turn this file red):
 *   * SettingsPage stops rendering <HfTokenSettings>         -> "the Settings page shows the card"
 *   * Save sends anything but the trimmed typed value        -> "Save sends the typed token"
 *   * Save leaves the typed value in the field               -> "clears the field"
 *   * Test sends the typed token when the field is empty      -> "Test with an empty field"
 *   * Clear stops calling DELETE                              -> "Clear removes"
 *   * Save ignores the verification the server kept           -> "Save after a Test says verified"
 *   * a failed Test does not re-read the status               -> "a failed Test re-reads the status"
 *   * the card drops the user / last-verified / check error   -> "shows the recorded check" tests
 *   * the forms ignore `tokenStatus`                          -> the two hint tests
 * All tokens are fake.
 */

import { beforeEach, describe, expect, it, vi } from 'vitest';
import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import type { HfTokenStatus } from '@/types/settings';

const api = vi.hoisted(() => ({
  getHfToken: vi.fn(),
  saveHfToken: vi.fn(),
  clearHfToken: vi.fn(),
  testHfToken: vi.fn(),
}));

vi.mock('@/services/api', async (importOriginal) => {
  const actual = await importOriginal<typeof import('@/services/api')>();
  return { ...actual, settingsApi: api };
});

import { HfTokenSettings } from '../HfTokenSettings';
import { SettingsPage } from '@pages/SettingsPage';
import { ModelLoadForm } from '@components/models/ModelLoadForm';
import { SAEDownloadForm } from '@components/sae/SAEDownloadForm';
import { tokenFieldHint } from '@/hooks/useHfTokenStatus';
import { ApiError } from '@/services/api';

const FAKE = 'hf_FAKEtyped000000000000000000000000typ5';

function status(over: Partial<HfTokenStatus> = {}): HfTokenStatus {
  return {
    configured: true,
    source: 'settings',
    masked: 'hf_…sav2',
    last_verified_at: null,
    user: null,
    stored: true,
    stored_error: null,
    last_check_error: null,
    encryption_available: true,
    env_configured: false,
    ...over,
  };
}

const NONE = status({ configured: false, source: 'none', masked: null, stored: false });

function renderCard(s: HfTokenStatus | null = NONE) {
  const onStatus = vi.fn();
  const onRefresh = vi.fn().mockResolvedValue(undefined);
  render(<HfTokenSettings status={s} onStatus={onStatus} onRefresh={onRefresh} />);
  return { onStatus, onRefresh };
}

beforeEach(() => {
  Object.values(api).forEach((fn) => fn.mockReset());
});

describe('HfTokenSettings', () => {
  it('shows the masked token, the active source and the last check — never more', () => {
    renderCard(status({ last_verified_at: '2026-10-18T12:00:00Z', user: 'alice' }));
    const box = screen.getByTestId('hf-token-status');
    expect(box).toHaveTextContent('Saved token');
    expect(box).toHaveTextContent('hf_…sav2');
    expect(box).toHaveTextContent('alice');
  });

  it('says a per-request token still overrides', () => {
    renderCard();
    expect(screen.getByText(/still overrides the saved one/)).toBeInTheDocument();
  });

  it('uses a password-type field', () => {
    renderCard();
    expect(screen.getByLabelText('New token')).toHaveAttribute('type', 'password');
  });

  it('Save sends the typed token, reports the new status and clears the field', async () => {
    const next = status();
    api.saveHfToken.mockResolvedValue(next);
    const { onStatus } = renderCard();
    const field = screen.getByLabelText('New token');
    await userEvent.type(field, `  ${FAKE} `);
    await userEvent.click(screen.getByRole('button', { name: /save/i }));
    await waitFor(() => expect(onStatus).toHaveBeenCalledWith(next));
    expect(api.saveHfToken).toHaveBeenCalledTimes(1);
    expect(api.saveHfToken).toHaveBeenCalledWith(FAKE);
    expect(field).toHaveValue('');
    expect(document.body.textContent).not.toContain(FAKE);
  });

  it('shows the server refusal when no encryption key is configured', async () => {
    api.saveHfToken.mockRejectedValue(
      new ApiError('SETTINGS_KEY_MISSING', 'Cannot store the secret encrypted: SETTINGS_ENCRYPTION_KEY is not set.', undefined, 409)
    );
    renderCard();
    await userEvent.type(screen.getByLabelText('New token'), FAKE);
    await userEvent.click(screen.getByRole('button', { name: /save/i }));
    expect(await screen.findByText(/SETTINGS_ENCRYPTION_KEY is not set/)).toBeInTheDocument();
  });

  it('disables Save and says why when the server has no key', () => {
    renderCard(status({ encryption_available: false, stored: false, source: 'none', configured: false }));
    expect(screen.getByRole('button', { name: /save/i })).toBeDisabled();
    expect(screen.getByText(/Saving is disabled/)).toBeInTheDocument();
  });

  it('Test with an empty field checks the active token and shows the user', async () => {
    api.testHfToken.mockResolvedValue({ ok: true, user: 'alice', source: 'settings', verified_at: 'x' });
    const { onRefresh } = renderCard(status());
    await userEvent.click(screen.getByRole('button', { name: /test/i }));
    expect(await screen.findByText(/Hugging Face user alice/)).toBeInTheDocument();
    expect(api.testHfToken).toHaveBeenCalledTimes(1);
    expect(api.testHfToken).toHaveBeenCalledWith(undefined);
    expect(onRefresh).toHaveBeenCalledTimes(1);
  });

  it('Test with a typed token checks that one without saving it', async () => {
    api.testHfToken.mockResolvedValue({ ok: true, user: 'bob', source: 'request', verified_at: 'x' });
    renderCard();
    await userEvent.type(screen.getByLabelText('New token'), FAKE);
    await userEvent.click(screen.getByRole('button', { name: /test/i }));
    expect(await screen.findByText(/user bob \(the typed token; not saved — Save it now/)).toBeInTheDocument();
    expect(api.testHfToken).toHaveBeenCalledWith(FAKE);
    expect(api.saveHfToken).not.toHaveBeenCalled();
  });

  it('Test shows a named failure', async () => {
    api.testHfToken.mockRejectedValue(
      new ApiError('INVALID_HF_TOKEN', 'Hugging Face rejected the token: it is invalid, expired or revoked.', undefined, 401)
    );
    renderCard(status());
    await userEvent.click(screen.getByRole('button', { name: /test/i }));
    expect(await screen.findByText(/invalid, expired or revoked/)).toBeInTheDocument();
  });

  it('Save after a Test says verified, from the status the server returns', async () => {
    const next = status({ last_verified_at: '2026-10-18T15:55:44Z', user: 'mistudio' });
    api.saveHfToken.mockResolvedValue(next);
    const { onStatus } = renderCard();
    await userEvent.type(screen.getByLabelText('New token'), FAKE);
    await userEvent.click(screen.getByRole('button', { name: /save/i }));
    await waitFor(() => expect(onStatus).toHaveBeenCalledWith(next));
    expect(await screen.findByText(/Token saved\. Verified — Hugging Face user mistudio/)).toBeInTheDocument();
  });

  it('Save with no kept check asks for a Test', async () => {
    api.saveHfToken.mockResolvedValue(status());
    renderCard();
    await userEvent.type(screen.getByLabelText('New token'), FAKE);
    await userEvent.click(screen.getByRole('button', { name: /save/i }));
    expect(await screen.findByText(/Use Test to check it/)).toBeInTheDocument();
  });

  it('a failed Test re-reads the status, so the recorded failure shows', async () => {
    api.testHfToken.mockRejectedValue(
      new ApiError('INVALID_HF_TOKEN', 'Hugging Face rejected the token: it is invalid, expired or revoked.', undefined, 401)
    );
    const { onRefresh } = renderCard(status());
    await userEvent.click(screen.getByRole('button', { name: /test/i }));
    expect(await screen.findByText(/invalid, expired or revoked/)).toBeInTheDocument();
    expect(onRefresh).toHaveBeenCalledTimes(1);
  });

  it('shows the recorded check: the time and the user', () => {
    renderCard(status({ last_verified_at: '2026-10-18T15:55:44Z', user: 'mistudio' }));
    const box = screen.getByTestId('hf-token-status');
    expect(box).toHaveTextContent('mistudio');
    expect(box).not.toHaveTextContent('never');
  });

  it('shows the recorded check: a rejection and an unreachable Hub', () => {
    const { unmount } = render(
      <HfTokenSettings status={status({ last_check_error: 'invalid' })} onStatus={vi.fn()} onRefresh={vi.fn()} />
    );
    expect(screen.getByText(/rejected by Hugging Face at its last check/)).toBeInTheDocument();
    unmount();
    renderCard(status({ last_check_error: 'network' }));
    expect(screen.getByText(/could not reach Hugging Face; the token itself was not/)).toBeInTheDocument();
  });

  it('Clear removes the saved token', async () => {
    const after = status({ source: 'none', configured: false, masked: null, stored: false });
    api.clearHfToken.mockResolvedValue(after);
    const { onStatus } = renderCard(status());
    await userEvent.click(screen.getByRole('button', { name: /clear/i }));
    await waitFor(() => expect(onStatus).toHaveBeenCalledWith(after));
    expect(api.clearHfToken).toHaveBeenCalledTimes(1);
  });

  it('Clear is disabled when nothing is saved', () => {
    renderCard(status({ source: 'env', stored: false }));
    expect(screen.getByRole('button', { name: /clear/i })).toBeDisabled();
  });

  it('names an unreadable saved token', () => {
    renderCard(status({ stored_error: 'key_mismatch', source: 'env' }));
    expect(screen.getByText(/different encryption key/)).toBeInTheDocument();
  });
});

describe('the Settings page', () => {
  it('shows the card, filled from the server', async () => {
    api.getHfToken.mockResolvedValue(status({ masked: 'hf_…zz99' }));
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue({ ok: false, json: async () => null }));
    render(<SettingsPage />);
    expect(screen.getByText('Hugging Face token')).toBeInTheDocument();
    expect(await screen.findByText('hf_…zz99')).toBeInTheDocument();
    expect(api.getHfToken).toHaveBeenCalled();
    vi.unstubAllGlobals();
  });
});

describe('download forms: the token field hint', () => {
  it('names the saved token only when one is saved', () => {
    expect(tokenFieldHint(status(), 'orig')).toMatch(/Leave empty to use the saved token \(hf_…sav2\)/);
    expect(tokenFieldHint(status({ source: 'env' }), 'orig')).toMatch(/server's HF_TOKEN/);
    expect(tokenFieldHint(NONE, 'orig')).toBe('orig');
    expect(tokenFieldHint(null, 'orig')).toBe('orig');
  });

  it('the model form says leave empty when a token is saved', () => {
    render(<ModelLoadForm onSubmit={vi.fn()} tokenStatus={status()} />);
    expect(screen.getByText(/Leave empty to use the saved token/)).toBeInTheDocument();
  });

  it('the model form keeps its wording with no token', () => {
    render(<ModelLoadForm onSubmit={vi.fn()} tokenStatus={NONE} />);
    expect(screen.getByText('Required for gated models like Llama')).toBeInTheDocument();
    expect(screen.queryByText(/Leave empty/)).not.toBeInTheDocument();
  });

  it('the SAE form says leave empty when a token is saved', () => {
    render(<SAEDownloadForm onSubmit={vi.fn()} tokenStatus={status()} />);
    expect(screen.getByText(/Leave empty to use the saved token/)).toBeInTheDocument();
  });
});
