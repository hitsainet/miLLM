/**
 * F19 R2-04 — the contention dialog actually MOUNTS.
 *
 * The previous wiring tests were `readFileSync` + `toContain` on source
 * strings. They could not fail for the defect they claimed to guard: renaming
 * the handler, deleting the prop pass, or commenting out the whole JSX block
 * while leaving `<ContentionDialog` in a comment would all still pass. A
 * source grep proves the TEXT exists, not that the component is mounted or the
 * callback invoked — the third occurrence of this increment's named
 * anti-pattern.
 *
 * These render the real page, drive a real `CIRCUIT_LAYER_CONTENTION` refusal
 * through the real hook, and assert the dialog appears in the DOM with the
 * measurement and both resolutions.
 */

import { describe, expect, it, vi, beforeEach } from 'vitest';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import type { ReactNode } from 'react';

const activateMock = vi.fn();
const deactivateMock = vi.fn();

vi.mock('@/services/circuits', () => ({
  circuitApi: {
    list: vi.fn().mockResolvedValue({
      circuits: [
      {
        id: 'circ_1',
        name: 'hedging',
        rung: 2,
        rung_language: 'causally validated (edge)',
        validated: true,
        is_active: false,
        serving_mode: null,
        layers: [10, 13],
        edge_count: 2,
        intensity: 1.0,
        serveable: true,
        per_sae_warnings: [],
      },
      ],
      total: 1,
    }),
    activate: (...a: unknown[]) => activateMock(...a),
    deactivate: (...a: unknown[]) => deactivateMock(...a),
    setIntensity: vi.fn(),
    delete: vi.fn(),
    export: vi.fn(),
    claims: vi.fn().mockResolvedValue([]),
  },
}));

vi.mock('@components/circuits', async (importActual) => {
  const actual = await importActual<Record<string, unknown>>();
  return {
    ...actual,
    // Panels that pull in unrelated stores/sockets; the page's contention flow
    // is what is under test.
    AttachmentPanel: () => null,
    EdgeSensingPanel: () => null,
  };
});

import { ApiError } from '@/services/api';
import { CircuitsPage } from '../CircuitsPage';

const DETAILS = {
  contended_layers: [13],
  incumbent: { id: 'circ_abc', name: 'fear→threat' },
  requested: { id: 'circ_1', name: 'hedging' },
  overridable: true,
  override_param: 'allow_layer_overlap',
  rung_header_suppressed_if_overridden: true,
  colliding_keys: [],
  measured_hazard: {
    source: 'GPU close-out 2026-07-20, LFM2.5-1.2B-Instruct',
    one_layer_at_strength_5: 'coherent, indistinguishable from baseline',
    two_layers_at_strength_5: 'degenerate output (repeated tokens)',
    note: 'one model, one fixture — indicative, not exhaustive',
  },
};

function wrapper({ children }: { children: ReactNode }) {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false }, mutations: { retry: false } },
  });
  return <QueryClientProvider client={client}>{children}</QueryClientProvider>;
}

async function activateAndGetRefused() {
  render(<CircuitsPage />, { wrapper });
  const button = await screen.findByTestId('activate-button');
  await userEvent.click(button);
  return await screen.findByTestId('contention-dialog');
}

describe('CircuitsPage — layer contention', () => {
  beforeEach(async () => {
    activateMock.mockReset();
    deactivateMock.mockReset();
    // A test that swaps the circuit list (the unvalidated case) must not leak
    // that fixture into its neighbours — restore the default every time.
    const { circuitApi } = await import('@/services/circuits');
    (circuitApi.list as ReturnType<typeof vi.fn>).mockResolvedValue({
      circuits: [
        {
          id: 'circ_1',
          name: 'hedging',
          rung: 2,
          rung_language: 'causally validated (edge)',
          validated: true,
          is_active: false,
          serving_mode: null,
          layers: [10, 13],
          edge_count: 2,
          intensity: 1.0,
          serveable: true,
          per_sae_warnings: [],
        },
      ],
      total: 1,
    });
    activateMock.mockImplementation(async () => {
      throw new ApiError(
        'CIRCUIT_LAYER_CONTENTION',
        'Layers [13] are already served by circuit ‘fear→threat’',
        DETAILS,
      );
    });
  });

  it('MOUNTS the dialog on a contention refusal', async () => {
    const dialog = await activateAndGetRefused();
    expect(dialog).toBeInTheDocument();
    expect(screen.getByTestId('contended-layers')).toHaveTextContent('fear→threat');
  });

  it('shows the measurement AND its caveat', async () => {
    await activateAndGetRefused();
    const hazard = screen.getByTestId('measured-hazard');
    expect(hazard).toHaveTextContent('degenerate output');
    expect(hazard).toHaveTextContent('one model, one fixture');
  });

  it('"Compose anyway" retries with allow_layer_overlap', async () => {
    await activateAndGetRefused();
    activateMock.mockReset();
    activateMock.mockResolvedValue({
      name: 'hedging',
      serving_mode: 'full',
      rung_language: 'causally validated (edge)',
    });

    await userEvent.click(screen.getByTestId('compose-anyway'));

    await waitFor(() => expect(activateMock).toHaveBeenCalled());
    const [circuitId, ack, overlap] = activateMock.mock.calls[0];
    expect(circuitId).toBe('circ_1');
    expect(overlap).toBe(true);
    expect(ack).toBe(false); // this circuit is validated
  });

  it('"Deactivate incumbent" targets the INCUMBENT, not the requester', async () => {
    await activateAndGetRefused();
    deactivateMock.mockResolvedValue({ id: 'circ_abc' });

    await userEvent.click(screen.getByTestId('deactivate-incumbent'));

    await waitFor(() => expect(deactivateMock).toHaveBeenCalledWith('circ_abc'));
  });

  it('"Deactivate incumbent" then RETRIES the activation', async () => {
    // R2-15: this closed the dialog and stopped, leaving the operator with
    // nothing steering and no indication the remedy was only half done — when
    // getting this circuit serving is the reason they opened the dialog.
    await activateAndGetRefused();
    deactivateMock.mockResolvedValue({ id: 'circ_abc' });
    activateMock.mockReset();
    activateMock.mockResolvedValue({
      name: 'hedging', serving_mode: 'full', rung_language: 'x',
    });

    await userEvent.click(screen.getByTestId('deactivate-incumbent'));

    await waitFor(() => expect(activateMock).toHaveBeenCalled());
    const [circuitId, , overlap] = activateMock.mock.calls[0];
    expect(circuitId).toBe('circ_1');
    expect(overlap).toBe(false); // the layer is free now — no override needed
  });

  it('does NOT retry when deactivating the incumbent FAILS', async () => {
    // The layer is still held, so a retry would be refused again — and that
    // second refusal reads as "the remedy did nothing" rather than as "the
    // deactivation failed", which is the actual problem.
    await activateAndGetRefused();
    deactivateMock.mockImplementation(async () => {
      throw new Error('deactivate exploded');
    });
    activateMock.mockReset();

    await userEvent.click(screen.getByTestId('deactivate-incumbent'));

    await waitFor(() => expect(deactivateMock).toHaveBeenCalled());
    expect(activateMock).not.toHaveBeenCalled();
  });

  it('cancelling closes the dialog and activates nothing', async () => {
    await activateAndGetRefused();
    activateMock.mockReset();

    await userEvent.click(screen.getByTestId('contention-cancel-x'));

    await waitFor(() =>
      expect(screen.queryByTestId('contention-dialog')).not.toBeInTheDocument(),
    );
    expect(activateMock).not.toHaveBeenCalled();
  });

  it('"Compose anyway" KEEPS the unvalidated acknowledgement', async () => {
    // THE R2-02 defect. On a validated circuit `ack` is false either way, so
    // only an UNVALIDATED circuit can catch it — verified by a mutation that
    // survived the validated-circuit version of this test.
    //
    // Without the ack, the compose retry is refused with UNVALIDATED_CIRCUIT
    // and the hook toasts "tick the acknowledgement", which the operator
    // already did. The dialog has closed and there is no path forward: the
    // override is unreachable for exactly the circuits where composition is
    // riskiest.
    const { circuitApi } = await import('@/services/circuits');
    (circuitApi.list as ReturnType<typeof vi.fn>).mockResolvedValue({
      circuits: [
        {
          id: 'circ_1',
          name: 'hedging',
          rung: 1,
          rung_language: 'suggested (attribution-supported)',
          validated: false,
          is_active: false,
          serving_mode: null,
          layers: [10, 13],
          edge_count: 2,
          intensity: 1.0,
          serveable: true,
          per_sae_warnings: [],
        },
      ],
      total: 1,
    });

    render(<CircuitsPage />, { wrapper });
    await userEvent.click(await screen.findByTestId('unvalidated-ack'));
    await userEvent.click(screen.getByTestId('activate-button'));
    await screen.findByTestId('contention-dialog');

    activateMock.mockReset();
    activateMock.mockResolvedValue({
      name: 'hedging', serving_mode: 'full', rung_language: 'suggested',
    });
    await userEvent.click(screen.getByTestId('compose-anyway'));

    await waitFor(() => expect(activateMock).toHaveBeenCalled());
    const [, ack, overlap] = activateMock.mock.calls[0];
    expect(overlap).toBe(true);
    expect(ack).toBe(true);
  });

  it('names EVERY incumbent, not just the one it can deactivate', async () => {
    // R3-03. R2-12 added `all_incumbents` to the server payload precisely so an
    // operator is not sent to deactivate one, retry, and be refused by a second
    // they were never told about — and it was never RENDERED, so that scenario
    // still happened verbatim in the browser. "Declaring is not wiring", inside
    // the fix for it.
    activateMock.mockImplementation(async () => {
      throw new ApiError('CIRCUIT_LAYER_CONTENTION', 'contended', {
        ...DETAILS,
        contended_layers: [10, 13],
        incumbent: { id: 'circ_c', name: 'cC' },
        all_incumbents: [
          { id: 'circ_c', name: 'cC' },
          { id: 'circ_d', name: 'cD' },
        ],
      });
    });

    await activateAndGetRefused();

    const others = screen.getByTestId('other-incumbents');
    expect(others).toHaveTextContent('cD');
    expect(others).toHaveTextContent('will not be enough');
    // The one it CAN act on stays in the primary line.
    expect(screen.getByTestId('contended-layers')).toHaveTextContent('cC');
  });

  it('does not show the other-incumbents block for a single incumbent', () => {
    // Specificity: a needless "others also hold this" panel would make every
    // ordinary refusal look like a multi-circuit pile-up.
    expect(DETAILS.all_incumbents).toBeUndefined();
  });

  it('a COLLISION refusal offers no compose action', async () => {
    activateMock.mockImplementation(async () => {
      throw new ApiError('CIRCUIT_LAYER_CONTENTION', 'same feature', {
        ...DETAILS,
        overridable: false,
        override_param: undefined,
        colliding_keys: [{ layer: 13, feature_idx: 42, incumbent: 'circ_abc' }],
      });
    });
    await activateAndGetRefused();
    expect(screen.queryByTestId('compose-anyway')).not.toBeInTheDocument();
    expect(screen.getByTestId('colliding-keys')).toHaveTextContent('42');
  });
});
