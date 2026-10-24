/**
 * Rule sets on the Probe Monitors page (Feature 39 S2, FTASKS 14.7).
 *
 * The API client is mocked at its module; the component runs for real. Requests are asserted by
 * PAYLOAD and CALL COUNT.
 *
 * Controls (039 controls record): attach replaces the behaviour's list instead of adding to it →
 * "attach adds this set…" red; import parses and re-serialises the document → "import sends the
 * document as pasted" red.
 */

import { beforeEach, describe, expect, it, vi } from 'vitest';
import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import type { RuleSetSummary } from '@/types/ruleSet';

const api = {
  list: vi.fn(), get: vi.fn(), importDocument: vi.fn(), validate: vi.fn(), remove: vi.fn(),
  attachment: vi.fn(), setAttachment: vi.fn(), clearAttachment: vi.fn(),
};
vi.mock('@/services/api', async (importOriginal) => {
  const real = await importOriginal<typeof import('@/services/api')>();
  return { ...real, ruleSetsApi: api };
});
const { RuleSets } = await import('../RuleSets');

function set(id: string, over: Partial<RuleSetSummary> = {}): RuleSetSummary {
  return {
    id, name: `set-${id}`, version: 1, behaviour_key: 'b-k1', origin: 'authored',
    digest: 'a'.repeat(64), vector_status: 'passed', signals: 2, expressions: ['hit'],
    imported_at: null, checked_at: null, ...over,
  };
}

beforeEach(() => {
  vi.clearAllMocks();
  api.list.mockResolvedValue({ data: [set('rs_000000000001')], next: null });
  api.attachment.mockResolvedValue({ behaviour: 'b-k1', rule_set_ids: ['rs_0000000000aa'],
                                     signals: null, rule_sets: [], history: [] });
  api.setAttachment.mockResolvedValue({});
  api.clearAttachment.mockResolvedValue({});
  api.importDocument.mockResolvedValue(set('rs_000000000002'));
});

describe('RuleSets', () => {
  it('attach adds this set to the behaviour list, keeping the sets already there', async () => {
    render(<RuleSets />);
    await userEvent.click(await screen.findByTestId('rule-set-attach-rs_000000000001'));
    await waitFor(() => expect(api.setAttachment).toHaveBeenCalledTimes(1));
    expect(api.setAttachment).toHaveBeenCalledWith(
      'b-k1', ['rs_0000000000aa', 'rs_000000000001']);
  });

  it('detaching the last set clears the attachment', async () => {
    api.attachment.mockResolvedValue({ behaviour: 'b-k1', rule_set_ids: ['rs_000000000001'],
                                       signals: null, rule_sets: [], history: [] });
    render(<RuleSets />);
    await userEvent.click(await screen.findByTestId('rule-set-detach-rs_000000000001'));
    await waitFor(() => expect(api.clearAttachment).toHaveBeenCalledTimes(1));
    expect(api.clearAttachment).toHaveBeenCalledWith('b-k1');
    expect(api.setAttachment).not.toHaveBeenCalled();
  });

  it('import sends the document as pasted', async () => {
    const doc = '{"format": "midataworks.rule-set/v1",   "name": "x"}';
    render(<RuleSets />);
    await userEvent.click(await screen.findByTestId('rule-set-document'));
    await userEvent.paste(doc);
    await userEvent.click(screen.getByTestId('rule-set-import'));
    await waitFor(() => expect(api.importDocument).toHaveBeenCalledTimes(1));
    expect(api.importDocument).toHaveBeenCalledWith(doc);
  });

  it('a refusal is shown, not swallowed', async () => {
    api.remove.mockRejectedValue(new Error('rule set is attached to 1 behaviour(s)'));
    render(<RuleSets />);
    await userEvent.click(await screen.findByTestId('rule-set-delete-rs_000000000001'));
    expect(await screen.findByTestId('rule-set-problem')).toHaveTextContent('attached');
  });

  it('a failed set and the fields serving ignores are labelled', async () => {
    api.list.mockResolvedValue({ data: [set('rs_000000000003', { vector_status: 'vectors_failed' })],
                                 next: null });
    api.get.mockResolvedValue({
      ...set('rs_000000000003', { vector_status: 'vectors_failed' }), description: 'd',
      forms: { hit: 'pattern', op: 'code_rule' }, provenance: null,
      vector_report: { case_id: 'c1' }, not_used_at_serving: { vetoes: [], outputs: [] },
      document: {},
    });
    render(<RuleSets />);
    await userEvent.click(await screen.findByText('View'));
    expect(await screen.findByTestId('vectors-failed')).toHaveTextContent('c1');
    expect(screen.getByTestId('not-used-at-serving')).toHaveTextContent('vetoes, outputs');
    expect(screen.getByText('operator code (scored after the response)')).toBeInTheDocument();
  });
});
