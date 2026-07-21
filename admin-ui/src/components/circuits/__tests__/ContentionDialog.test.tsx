/**
 * Feature 19 task 5.4 — the informed refusal, and the case that has no
 * override.
 *
 * The load-bearing assertion here is the NEGATIVE one: a same-key collision
 * must render no compose action at all. Offering one would invite an override
 * that cannot be honest — one author's strength silently overwriting another's.
 */

import { describe, expect, it, vi } from 'vitest';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';

import { ContentionDialog } from '../ContentionDialog';
import type { ContentionDetails } from '../ContentionDialog';

const HAZARD = {
  source: 'GPU close-out 2026-07-20, LFM2.5-1.2B-Instruct',
  one_layer_at_strength_5: 'coherent, indistinguishable from baseline',
  two_layers_at_strength_5: 'degenerate output (repeated tokens)',
  note: 'one model, one fixture — indicative, not exhaustive',
};

const CONTENTION: ContentionDetails = {
  contended_layers: [13],
  incumbent: { id: 'circ_abc', name: 'fear→threat' },
  requested: { id: 'circ_xyz', name: 'hedging' },
  override_param: 'allow_layer_overlap',
  rung_header_suppressed_if_overridden: true,
  overridable: true,
  colliding_keys: [],
  measured_hazard: HAZARD,
};

const COLLISION: ContentionDetails = {
  ...CONTENTION,
  overridable: false,
  override_param: undefined,
  colliding_keys: [{ layer: 13, feature_idx: 42, incumbent: 'circ_abc' }],
};

describe('ContentionDialog', () => {
  it('names the incumbent so the next action is obvious', () => {
    render(
      <ContentionDialog
        details={CONTENTION}
        message="Layers [13] are already served by circuit 'fear→threat'."
        onCancel={vi.fn()}
      />,
    );
    expect(screen.getByTestId('contended-layers')).toHaveTextContent('fear→threat');
    expect(screen.getByTestId('contended-layers')).toHaveTextContent('13');
  });

  it('shows the MEASUREMENT behind the refusal', () => {
    // BR-011 binding condition: an operator who overrides has been told what
    // happened last time. A refusal stating only the fact of contention fails
    // the requirement.
    render(
      <ContentionDialog details={CONTENTION} message="msg" onCancel={vi.fn()} />,
    );
    const hazard = screen.getByTestId('measured-hazard');
    expect(hazard).toHaveTextContent('degenerate output');
    expect(hazard).toHaveTextContent('indistinguishable from baseline');
  });

  it('states the caveat as part of the data, not as a footnote', () => {
    render(
      <ContentionDialog details={CONTENTION} message="msg" onCancel={vi.fn()} />,
    );
    expect(screen.getByTestId('measured-hazard')).toHaveTextContent(
      'one model, one fixture',
    );
  });

  it('warns that composing removes the rung header', () => {
    render(
      <ContentionDialog details={CONTENTION} message="msg" onCancel={vi.fn()} />,
    );
    expect(screen.getByTestId('rung-suppression-note')).toHaveTextContent(
      'no single circuit',
    );
  });

  it('offers BOTH resolutions for ordinary contention', async () => {
    const compose = vi.fn();
    const deactivate = vi.fn();
    render(
      <ContentionDialog
        details={CONTENTION}
        message="msg"
        onComposeAnyway={compose}
        onDeactivateIncumbent={deactivate}
        onCancel={vi.fn()}
      />,
    );
    await userEvent.click(screen.getByTestId('compose-anyway'));
    expect(compose).toHaveBeenCalled();
    await userEvent.click(screen.getByTestId('deactivate-incumbent'));
    expect(deactivate).toHaveBeenCalled();
  });

  describe('same-key collision', () => {
    it('renders NO compose action', () => {
      // THE assertion this component exists for. There is no honest
      // composition of two circuits steering the same feature, so the UI must
      // not offer one even when a handler is supplied.
      render(
        <ContentionDialog
          details={COLLISION}
          message="msg"
          onComposeAnyway={vi.fn()}
          onDeactivateIncumbent={vi.fn()}
          onCancel={vi.fn()}
        />,
      );
      expect(screen.queryByTestId('compose-anyway')).not.toBeInTheDocument();
    });

    it('shows the colliding pairs so the operator can edit one circuit', () => {
      render(
        <ContentionDialog details={COLLISION} message="msg" onCancel={vi.fn()} />,
      );
      const keys = screen.getByTestId('colliding-keys');
      expect(keys).toHaveTextContent('L13');
      expect(keys).toHaveTextContent('42');
      expect(keys).toHaveTextContent('belong to neither author');
    });

    it('does not show a hazard block that implies composition is available', () => {
      render(
        <ContentionDialog details={COLLISION} message="msg" onCancel={vi.fn()} />,
      );
      expect(screen.queryByTestId('measured-hazard')).not.toBeInTheDocument();
      expect(screen.queryByTestId('rung-suppression-note')).not.toBeInTheDocument();
    });

    it('still offers deactivating the incumbent', () => {
      render(
        <ContentionDialog
          details={COLLISION}
          message="msg"
          onDeactivateIncumbent={vi.fn()}
          onCancel={vi.fn()}
        />,
      );
      expect(screen.getByTestId('deactivate-incumbent')).toBeInTheDocument();
    });
  });

  it('cancels without acting', async () => {
    const cancel = vi.fn();
    const compose = vi.fn();
    render(
      <ContentionDialog
        details={CONTENTION}
        message="msg"
        onComposeAnyway={compose}
        onCancel={cancel}
      />,
    );
    await userEvent.click(screen.getByTestId('contention-cancel-x'));
    expect(cancel).toHaveBeenCalled();
    expect(compose).not.toHaveBeenCalled();
  });
});
