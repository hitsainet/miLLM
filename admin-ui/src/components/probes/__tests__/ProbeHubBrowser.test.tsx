/**
 * The Hub browser.
 *
 * The test worth keeping is the rung one. A manifest that carries no rung must not render as
 * "rung 0": rung 0 means *trained, and nothing else measured*, which is a claim about the probe's
 * evidence. "Not stated" is the truth, and the difference decides whether someone imports it.
 */

import { beforeEach, describe, expect, it, vi } from 'vitest';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';

const api = {
  hubSearch: vi.fn(),
  hubDefinitions: vi.fn(),
  hubImport: vi.fn(),
};

vi.mock('@/services/api', () => ({ probesApi: api }));

const { ProbeHubBrowser } = await import('../ProbeHubBrowser');

function renderBrowser(overrides: { onImported?: () => void; onError?: () => void } = {}) {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(
    <QueryClientProvider client={client}>
      <ProbeHubBrowser
        onImported={overrides.onImported ?? vi.fn()}
        onError={overrides.onError ?? vi.fn()}
      />
    </QueryClientProvider>
  );
}

describe('ProbeHubBrowser', () => {
  beforeEach(() => {
    api.hubSearch.mockReset();
    api.hubDefinitions.mockReset();
    api.hubImport.mockReset();
  });

  it('does not search until asked', () => {
    renderBrowser();
    expect(api.hubSearch).not.toHaveBeenCalled();
  });

  it('searches on submit and lists the repos', async () => {
    api.hubSearch.mockResolvedValue([{ repo_id: 'mistudio/probes-lfm2' }]);
    renderBrowser();
    await userEvent.type(
      screen.getByLabelText(/Search Hugging Face/i),
      'high stakes{Enter}'
    );
    expect(await screen.findByText('mistudio/probes-lfm2')).toBeInTheDocument();
    expect(api.hubSearch).toHaveBeenCalledWith({ q: 'high stakes' });
  });

  it('⚠ a definition with no rung says so rather than showing rung 0', async () => {
    api.hubSearch.mockResolvedValue([{ repo_id: 'a/b' }]);
    api.hubDefinitions.mockResolvedValue([
      { filename: 'unknown.probe.json', rung: null },
      { filename: 'known.probe.json', rung: 2 },
    ]);
    renderBrowser();
    await userEvent.type(screen.getByLabelText(/Search Hugging Face/i), 'x{Enter}');
    await userEvent.click(await screen.findByTestId('hub-repo'));
    const rows = await screen.findAllByTestId('hub-definition');
    expect(rows[0]).toHaveTextContent('rung not stated');
    expect(rows[0]).not.toHaveTextContent('rung 0');
    expect(rows[1]).toHaveTextContent('rung 2');
  });

  it('imports one definition by repo AND filename', async () => {
    api.hubSearch.mockResolvedValue([{ repo_id: 'a/b' }]);
    api.hubDefinitions.mockResolvedValue([{ filename: 'p.probe.json', rung: 2 }]);
    api.hubImport.mockResolvedValue({ name: 'high-stakes' });
    const onImported = vi.fn();
    renderBrowser({ onImported });
    await userEvent.type(screen.getByLabelText(/Search Hugging Face/i), 'x{Enter}');
    await userEvent.click(await screen.findByTestId('hub-repo'));
    await userEvent.click(await screen.findByRole('button', { name: /Import p.probe.json/ }));
    // ⚠ The payload, not just that it was called: a repo id without its filename, or a filename
    // without its repo, would import a different document or none.
    expect(api.hubImport).toHaveBeenCalledWith({
      repo_id: 'a/b',
      filename: 'p.probe.json',
    });
  });

  it('a Hub failure is shown, not swallowed', async () => {
    api.hubSearch.mockRejectedValue(new Error('HUB_UNAVAILABLE'));
    renderBrowser();
    await userEvent.type(screen.getByLabelText(/Search Hugging Face/i), 'x{Enter}');
    expect(await screen.findByTestId('hub-error')).toHaveTextContent('HUB_UNAVAILABLE');
  });

  it('an empty result says the tag it searched for', async () => {
    api.hubSearch.mockResolvedValue([]);
    renderBrowser();
    await userEvent.type(screen.getByLabelText(/Search Hugging Face/i), 'x{Enter}');
    expect(await screen.findByText(/mistudio-probe-definition/)).toBeInTheDocument();
  });
});
