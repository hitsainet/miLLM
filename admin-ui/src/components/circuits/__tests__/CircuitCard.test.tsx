/**
 * Tests for CircuitCard + CircuitActivateControl (Feature 13).
 *
 * The load-bearing assertions are the EVIDENCE HONESTY ones: the rung badge
 * renders the server's phrase verbatim, a rung<2 circuit is visibly marked
 * unvalidated and cannot be activated without ticking the acknowledgement, and
 * a slice-fallback serve is disclosed as a partial rendering.
 */

import { describe, expect, it, vi } from 'vitest';
import { fireEvent, render, screen } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';

// The card embeds EdgeSensingToggle (Feature 15), which reads sensing status.
// Stubbed so these tests stay about the card's evidence contract.
vi.mock('@/services/api', () => ({
  circuitSensingApi: {
    status: vi.fn().mockResolvedValue({
      armed: false,
      circuit_id: null,
      circuit_name: null,
      layers: [],
      sensable_edges: 0,
      unsensable_edges: [],
      max_token_lag: 4,
      last_request_overhead_ms: 0,
      events_recorded: 0,
      ws_dropped: 0,
      enabled_circuits: [],
    }),
    setEnabled: vi.fn(),
  },
}));

import { CircuitCard } from '../CircuitCard';
import type { CircuitSummary } from '@/types/circuits';

function makeCircuit(overrides: Partial<CircuitSummary> = {}): CircuitSummary {
  return {
    id: 'circ_1',
    name: 'fear→threat',
    description: null,
    rung: 2,
    rung_language: 'causally validated (edge)',
    rung_next_step: 'run circuit-level faithfulness at promotion',
    validated: true,
    edge_count: 3,
    layers: [10, 13],
    serveable: true,
    is_active: false,
    serving_mode: null,
    intensity: 1.0,
    per_sae_warnings: [],
    ...overrides,
  };
}

const noop = () => {};

function renderCard(
  circuit: CircuitSummary,
  onActivate = vi.fn(),
  composedLayers: number[] = [],
) {
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  render(
    <QueryClientProvider client={queryClient}>
      <CircuitCard
        circuit={circuit}
        composedLayers={composedLayers}
        onActivate={onActivate}
        onDeactivate={noop}
        onDelete={noop}
        onExport={noop}
      />
    </QueryClientProvider>,
  );
  return onActivate;
}

describe('CircuitCard', () => {
  it('renders identity, layers and edge count', () => {
    renderCard(makeCircuit());
    expect(screen.getByText('fear→threat')).toBeInTheDocument();
    expect(screen.getByText('L10 → L13')).toBeInTheDocument();
    expect(screen.getByText('3 edges')).toBeInTheDocument();
  });

  it('renders the server rung phrase VERBATIM', () => {
    renderCard(makeCircuit({ rung: 1, rung_language: 'suggested (attribution-supported)', validated: false }));
    expect(screen.getByTestId('rung-badge')).toHaveTextContent(
      'suggested (attribution-supported)',
    );
  });

  it('never shows causal language for a rung-0 circuit', () => {
    renderCard(makeCircuit({ rung: 0, rung_language: 'associated', validated: false }));
    expect(screen.getByTestId('rung-badge')).toHaveTextContent('associated');
    expect(document.body.textContent?.toLowerCase()).not.toContain('causal');
  });

  it('marks an unvalidated circuit', () => {
    renderCard(makeCircuit({ rung: 0, rung_language: 'associated', validated: false }));
    expect(screen.getByTestId('unvalidated-badge')).toBeInTheDocument();
  });

  it('does not mark a validated circuit as unvalidated', () => {
    renderCard(makeCircuit());
    expect(screen.queryByTestId('unvalidated-badge')).not.toBeInTheDocument();
  });

  it('shows the per-SAE compatibility verdicts', () => {
    renderCard(
      makeCircuit({
        per_sae_warnings: [
          { layer: 10, sae_id: 'sae-10', verdict: 'bind' },
          { layer: 13, sae_id: null, verdict: 'unbound', reason: 'nothing attached' },
        ],
      }),
    );
    const chips = screen.getByTestId('per-sae-verdicts');
    expect(chips).toHaveTextContent('L10 bind');
    expect(chips).toHaveTextContent('L13 unbound');
  });

  it('flags an incomplete SAE set', () => {
    renderCard(makeCircuit({ serveable: false }));
    expect(screen.getByText('SAE set incomplete')).toBeInTheDocument();
  });

  it('discloses slice-fallback serving as a partial rendering', () => {
    renderCard(makeCircuit({ is_active: true, serving_mode: 'slice_fallback' }));
    const disclosure = screen.getByTestId('slice-disclosure');
    expect(disclosure).toHaveTextContent(/per-layer slice/i);
    expect(disclosure).toHaveTextContent(/not the whole circuit/i);
  });

  it('does not show the slice disclosure for full serving', () => {
    renderCard(makeCircuit({ is_active: true, serving_mode: 'full' }));
    expect(screen.queryByTestId('slice-disclosure')).not.toBeInTheDocument();
  });

  it('reveals the next step that would raise the rung', () => {
    renderCard(makeCircuit({ rung: 0, rung_language: 'associated', validated: false,
      rung_next_step: 'run the attribution pass' }));
    fireEvent.click(screen.getByText('What would raise this rung?'));
    expect(screen.getByTestId('next-step')).toHaveTextContent('run the attribution pass');
  });
});

describe('CircuitActivateControl (via CircuitCard)', () => {
  it('activates a validated circuit directly, without an acknowledgement', () => {
    const onActivate = renderCard(makeCircuit());
    expect(screen.queryByTestId('unvalidated-ack')).not.toBeInTheDocument();
    fireEvent.click(screen.getByTestId('activate-button'));
    expect(onActivate).toHaveBeenCalledWith(false);
  });

  it('BLOCKS activation of a rung<2 circuit until acknowledged', () => {
    const onActivate = renderCard(
      makeCircuit({ rung: 0, rung_language: 'associated', validated: false }),
    );
    const button = screen.getByTestId('activate-button');
    expect(button).toBeDisabled();
    fireEvent.click(button);
    expect(onActivate).not.toHaveBeenCalled();
  });

  it('activates with acknowledgement once the box is ticked', () => {
    const onActivate = renderCard(
      makeCircuit({ rung: 1, rung_language: 'suggested (attribution-supported)', validated: false }),
    );
    fireEvent.click(screen.getByTestId('unvalidated-ack'));
    expect(screen.getByTestId('activate-button')).toBeEnabled();
    fireEvent.click(screen.getByTestId('activate-button'));
    expect(onActivate).toHaveBeenCalledWith(true);
  });

  it('states the rung in the acknowledgement label, verbatim', () => {
    renderCard(makeCircuit({ rung: 0, rung_language: 'associated', validated: false }));
    expect(screen.getByTestId('unvalidated-ack-label')).toHaveTextContent(
      'This circuit is associated — steer anyway',
    );
  });

  it('shows Deactivate instead of Activate when serving', () => {
    renderCard(makeCircuit({ is_active: true, serving_mode: 'full' }));
    expect(screen.getByText('Deactivate')).toBeInTheDocument();
    expect(screen.queryByTestId('activate-button')).not.toBeInTheDocument();
  });
});

describe('CircuitCard — composed layers (F19 R2-14)', () => {
  it('HIDES the rung badge while a layer is composed', () => {
    // The runtime suppresses `X-miLLM-Circuit-Rung` in exactly this case,
    // because no single circuit's evidence describes a summed response. A card
    // still showing the rung would contradict the header and tell the operator
    // the evidence still applies.
    renderCard(makeCircuit({ is_active: true }), vi.fn(), [13]);
    expect(screen.queryByTestId('rung-badge')).not.toBeInTheDocument();
    expect(screen.getByTestId('composed-badge')).toHaveTextContent(
      'rung suppressed',
    );
  });

  it('shows the rung badge when NOTHING is composed', () => {
    renderCard(makeCircuit({ is_active: true }), vi.fn(), []);
    expect(screen.getByTestId('rung-badge')).toBeInTheDocument();
    expect(screen.queryByTestId('composed-badge')).not.toBeInTheDocument();
  });

  it('never shows BOTH badges', () => {
    // A rung next to "composed" is the contradiction this exists to prevent.
    renderCard(makeCircuit({ is_active: true }), vi.fn(), [13]);
    const both =
      screen.queryByTestId('rung-badge') && screen.queryByTestId('composed-badge');
    expect(both).toBeFalsy();
  });
});
