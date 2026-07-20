/**
 * Tests for AttachSetDialog (Feature 12): the multi-SAE attach-set picker.
 *
 * Covers render + pre-selection, multi-select, submit payload, the isAttaching
 * disabled state, the server's per-SAE warnings + VRAM readout, and a server
 * rejection surfacing the reason VERBATIM (rollback, not a generic toast).
 */

import { describe, expect, it, vi, beforeEach } from 'vitest';
import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';

import { ApiError } from '@/services/api';
import type { AttachSetItem, AttachSetResponse, SAEInfo } from '@/types';

import { AttachSetDialog, type AttachCandidate } from '../AttachSetDialog';

function makeSAE(id: string, layer: number): SAEInfo {
  return {
    id,
    repository_id: `repo/${id}`,
    revision: 'main',
    name: id,
    format: 'npz',
    d_in: 2304,
    d_sae: 16384,
    trained_on: 'gemma-2-2b',
    trained_layer: layer,
    width: '16k',
    average_l0: 50,
    file_size_bytes: 1024,
    status: 'cached',
    error_message: null,
    created_at: '2026-07-20T00:00:00Z',
    updated_at: '2026-07-20T00:00:00Z',
  };
}

const CANDIDATES: AttachCandidate[] = [
  { sae: makeSAE('sae-a', 10), layer: 10 },
  { sae: makeSAE('sae-b', 13), layer: 13 },
  { sae: makeSAE('sae-c', 20), layer: 20 },
];

function makeResponse(overrides: Partial<AttachSetResponse> = {}): AttachSetResponse {
  return {
    status: 'attached',
    entries: [
      { sae_id: 'sae-a', layer: 10, status: 'attached', memory_usage_mb: 64, warnings: [] },
      { sae_id: 'sae-b', layer: 13, status: 'attached', memory_usage_mb: 64, warnings: [] },
    ],
    attached_count: 2,
    total_memory_usage_mb: 128,
    vram_envelope_mb: 200,
    vram_warning: false,
    ...overrides,
  };
}

function renderDialog(
  props: Partial<React.ComponentProps<typeof AttachSetDialog>> = {},
) {
  const onSubmit = props.onSubmit ?? vi.fn().mockResolvedValue(makeResponse());
  const utils = render(
    <AttachSetDialog
      open
      onClose={vi.fn()}
      candidates={CANDIDATES}
      attached={[]}
      loadedModelName="gemma-2-2b"
      isAttaching={false}
      {...props}
      onSubmit={onSubmit}
    />,
  );
  return { ...utils, onSubmit };
}

describe('AttachSetDialog', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('renders one row per available SAE with its layer', () => {
    renderDialog();
    expect(screen.getAllByTestId('candidate-row')).toHaveLength(3);
    expect(screen.getByText('L10')).toBeInTheDocument();
    expect(screen.getByText('L13')).toBeInTheDocument();
    expect(screen.getByText('sae-c')).toBeInTheDocument();
    expect(screen.getByText(/gemma-2-2b/)).toBeInTheDocument();
  });

  it('pre-selects what is already attached ("edit the set")', () => {
    const attached: AttachSetItem[] = [{ sae_id: 'sae-b', layer: 13 }];
    renderDialog({ attached });

    expect(screen.getByLabelText('sae-b at layer 13')).toBeChecked();
    expect(screen.getByLabelText('sae-a at layer 10')).not.toBeChecked();
    // The already-attached row is badged so the set reads as an edit.
    expect(screen.getByText('attached')).toBeInTheDocument();
    expect(screen.getByText('1 selected')).toBeInTheDocument();
  });

  it('multi-selects and submits attachSet with the right payload', async () => {
    const user = userEvent.setup();
    const { onSubmit } = renderDialog();

    await user.click(screen.getByLabelText('sae-a at layer 10'));
    await user.click(screen.getByLabelText('sae-c at layer 20'));
    expect(screen.getByText('2 selected')).toBeInTheDocument();

    await user.click(screen.getByTestId('attach-set-submit'));

    await waitFor(() => expect(onSubmit).toHaveBeenCalledTimes(1));
    expect(onSubmit).toHaveBeenCalledWith([
      { sae_id: 'sae-a', layer: 10 },
      { sae_id: 'sae-c', layer: 20 },
    ]);
  });

  it('deselecting an attached entry removes it from the submitted set', async () => {
    const user = userEvent.setup();
    const { onSubmit } = renderDialog({
      attached: [
        { sae_id: 'sae-a', layer: 10 },
        { sae_id: 'sae-b', layer: 13 },
      ],
    });

    await user.click(screen.getByLabelText('sae-b at layer 13'));
    await user.click(screen.getByTestId('attach-set-submit'));

    await waitFor(() => expect(onSubmit).toHaveBeenCalledWith([{ sae_id: 'sae-a', layer: 10 }]));
  });

  it('disables the control and shows progress while isAttaching', () => {
    renderDialog({ isAttaching: true });

    expect(screen.getByTestId('attach-set-submit')).toBeDisabled();
    expect(screen.getByText('Attaching…')).toBeInTheDocument();
    // Rows are frozen too, so the set cannot drift under an in-flight call.
    expect(screen.getByLabelText('sae-a at layer 10')).toBeDisabled();
  });

  it("displays the server's per-SAE warnings and the VRAM total", async () => {
    const user = userEvent.setup();
    const onSubmit = vi.fn().mockResolvedValue(
      makeResponse({
        entries: [
          {
            sae_id: 'sae-a',
            layer: 10,
            status: 'attached',
            memory_usage_mb: 64,
            warnings: ['SAE was trained on layer 12, but attaching to layer 10'],
          },
          {
            sae_id: 'sae-b',
            layer: 13,
            status: 'attached',
            memory_usage_mb: 64,
            warnings: ["SAE was trained on 'gemma-2-2b', current model is 'gemma-2-9b'"],
          },
        ],
      }),
    );
    renderDialog({ onSubmit });

    await user.click(screen.getByTestId('attach-set-submit'));

    await waitFor(() => expect(screen.getByTestId('attach-result')).toBeInTheDocument());
    expect(screen.getByTestId('result-vram')).toHaveTextContent('128 MB');
    expect(screen.getByTestId('result-vram')).toHaveTextContent('200 MB');
    expect(screen.getByText('2 SAE(s) attached')).toBeInTheDocument();

    const warnings = screen.getByTestId('result-warnings');
    expect(warnings).toHaveTextContent(
      'sae-a @ L10: SAE was trained on layer 12, but attaching to layer 10',
    );
    expect(warnings).toHaveTextContent(
      "sae-b @ L13: SAE was trained on 'gemma-2-2b', current model is 'gemma-2-9b'",
    );
  });

  it('flags an over-envelope VRAM result', async () => {
    const user = userEvent.setup();
    const onSubmit = vi
      .fn()
      .mockResolvedValue(makeResponse({ total_memory_usage_mb: 256, vram_warning: true }));
    renderDialog({ onSubmit });

    await user.click(screen.getByTestId('attach-set-submit'));

    await waitFor(() =>
      expect(screen.getByTestId('result-vram-warning')).toBeInTheDocument(),
    );
    expect(screen.getByTestId('result-vram')).toHaveTextContent('256 MB');
  });

  it("shows the server's rejection reason verbatim and says nothing was attached", async () => {
    const user = userEvent.setup();
    const onSubmit = vi
      .fn()
      .mockRejectedValue(
        new ApiError(
          'INSUFFICIENT_MEMORY',
          'Attaching 3 SAE(s) needs ~412 MB but only 180 MB is free.',
          { projected_mb: 412, free_mb: 180 },
        ),
      );
    renderDialog({ onSubmit });

    await user.click(screen.getByTestId('attach-set-submit'));

    const rejection = await screen.findByTestId('attach-rejection');
    // Verbatim server message — not collapsed into a generic failure string.
    expect(rejection).toHaveTextContent(
      'Attaching 3 SAE(s) needs ~412 MB but only 180 MB is free.',
    );
    expect(rejection).toHaveTextContent('INSUFFICIENT_MEMORY');
    // The rollback is stated, not swallowed.
    expect(rejection).toHaveTextContent(/previously attached set is\s+unchanged/);
    expect(screen.queryByTestId('attach-result')).not.toBeInTheDocument();
  });

  it('surfaces an incompatibility rejection reason verbatim', async () => {
    const user = userEvent.setup();
    const onSubmit = vi
      .fn()
      .mockRejectedValue(
        new ApiError(
          'SAE_INCOMPATIBLE',
          "SAE 'sae-c' incompatible with model: Dimension mismatch: SAE d_in=2304, model hidden_size=3584",
        ),
      );
    renderDialog({ onSubmit });

    await user.click(screen.getByTestId('attach-set-submit'));

    expect(await screen.findByTestId('attach-rejection')).toHaveTextContent(
      'Dimension mismatch: SAE d_in=2304, model hidden_size=3584',
    );
  });

  it('tells the user to load a model when none is loaded', () => {
    renderDialog({ loadedModelName: null });

    expect(screen.getByTestId('no-model')).toBeInTheDocument();
    expect(screen.getByTestId('attach-set-submit')).toBeDisabled();
    expect(screen.queryByTestId('candidate-list')).not.toBeInTheDocument();
  });

  it('handles no downloaded SAEs for the loaded model', () => {
    renderDialog({ candidates: [] });

    expect(screen.getByTestId('no-candidates')).toBeInTheDocument();
    expect(screen.getByTestId('attach-set-submit')).toBeDisabled();
  });

  it('notes that a single available SAE cannot span layers', () => {
    renderDialog({ candidates: [CANDIDATES[0]] });

    expect(screen.getByTestId('single-candidate-note')).toBeInTheDocument();
    expect(screen.getAllByTestId('candidate-row')).toHaveLength(1);
    // Still submittable — attaching one SAE via the set path is legitimate.
    expect(screen.getByTestId('attach-set-submit')).not.toBeDisabled();
  });

  it('describes attach as ADDITIVE and does not claim unchecking detaches', () => {
    // attach_set is idempotent per (sae_id, layer) and "coexists with a
    // previously single-attached SAE (that SAE stays in the registry)" — its
    // own docstring. Copy claiming the set is "replaced" would be a lie about
    // what the button does, and would leave a user believing they had
    // detached a layer that is still steering.
    renderDialog({ open: true });
    const blurb = screen.getByText(/Attaching is/i).textContent ?? '';
    expect(blurb).toMatch(/additive/i);
    expect(blurb).toMatch(/does not.*detach|not.*detach it/i);
    expect(blurb).not.toMatch(/replaces the attached set/i);
  });
});
