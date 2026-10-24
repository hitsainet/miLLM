/**
 * Feature 19 task 5.1 — layer → claimant.
 *
 * The strip exists so a refusal is intelligible BEFORE it happens. Its one
 * subtle requirement: a composed layer has several claimants and must read as
 * ONE composition rather than as two unrelated claims.
 */

import { describe, expect, it } from 'vitest';
import { render, screen } from '@testing-library/react';

import { ClaimsStrip } from '../ClaimsStrip';

describe('ClaimsStrip', () => {
  it('says so plainly when nothing is claimed', () => {
    render(<ClaimsStrip claims={[]} />);
    expect(screen.getByTestId('claims-strip-empty')).toBeInTheDocument();
  });

  it('shows each layer with its claimant', () => {
    render(
      <ClaimsStrip
        claims={[
          { layer: 10, circuit_id: 'a', circuit_name: 'fear→threat', composed: false },
          { layer: 13, circuit_id: 'b', circuit_name: 'hedging', composed: false },
        ]}
      />,
    );
    expect(screen.getByTestId('claim-L10')).toHaveTextContent('fear→threat');
    expect(screen.getByTestId('claim-L13')).toHaveTextContent('hedging');
  });

  it('groups a COMPOSED layer into one entry naming both circuits', () => {
    render(
      <ClaimsStrip
        claims={[
          { layer: 10, circuit_id: 'a', circuit_name: 'fear→threat', composed: true },
          { layer: 10, circuit_id: 'b', circuit_name: 'hedging', composed: true },
        ]}
      />,
    );
    const entry = screen.getByTestId('claim-L10');
    expect(entry).toHaveTextContent('fear→threat');
    expect(entry).toHaveTextContent('hedging');
    expect(screen.getByTestId('composed-badge-L10')).toBeInTheDocument();
  });

  it('badges a layer with two holders even if the flag is stale', () => {
    // Defence in depth: two live claims on one layer IS a composition
    // regardless of what the `composed` column says, and the badge is how an
    // operator learns the rung header has gone.
    render(
      <ClaimsStrip
        claims={[
          { layer: 10, circuit_id: 'a', circuit_name: 'one', composed: false },
          { layer: 10, circuit_id: 'b', circuit_name: 'two', composed: false },
        ]}
      />,
    );
    expect(screen.getByTestId('composed-badge-L10')).toBeInTheDocument();
  });

  it('does NOT badge an ordinary single-holder layer', () => {
    render(
      <ClaimsStrip
        claims={[{ layer: 10, circuit_id: 'a', circuit_name: 'one', composed: false }]}
      />,
    );
    expect(screen.queryByTestId('composed-badge-L10')).not.toBeInTheDocument();
  });

  it('falls back to the circuit id when a name is missing', () => {
    render(
      <ClaimsStrip
        claims={[{ layer: 10, circuit_id: 'circ_abc', circuit_name: null, composed: false }]}
      />,
    );
    expect(screen.getByTestId('claim-L10')).toHaveTextContent('circ_abc');
  });
});
