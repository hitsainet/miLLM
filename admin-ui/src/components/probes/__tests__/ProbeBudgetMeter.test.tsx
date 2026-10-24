import { describe, expect, it } from 'vitest';
import { render, screen } from '@testing-library/react';
import { ProbeBudgetMeter } from '../ProbeBudgetMeter';

const budget = { decode_ms_per_pass: 0.25, prefill_ms_per_1k: 0.5, headroom: 0.9 };

describe('ProbeBudgetMeter (Feature 37)', () => {
  it('renders both halves of the budget from the status', () => {
    render(<ProbeBudgetMeter budget={budget} projected={{ decode: 0.18, prefill: 0.31, over_budget: false }} />);
    expect(screen.getByText('projected 0.180 of 0.25 ms per generated token')).toBeInTheDocument();
    expect(screen.getByText('0.310 of 0.5 ms per 1,000 prompt tokens')).toBeInTheDocument();
    const meters = screen.getAllByRole('meter');
    expect(meters.map((m) => m.getAttribute('aria-valuenow'))).toEqual(['72', '62']);
    expect(screen.queryByText(/over budget/)).toBeNull();
  });

  it('says over budget, and that nothing was disarmed, when a lowered budget no longer fits', () => {
    render(<ProbeBudgetMeter budget={budget} projected={{ decode: 0.3, prefill: 0.1, over_budget: true }} />);
    expect(screen.getByText(/over budget: the armed set no longer fits; nothing was disarmed/)).toBeInTheDocument();
  });

  it('renders nothing for a backend that sends no budget', () => {
    const { container } = render(<ProbeBudgetMeter />);
    expect(container).toBeEmptyDOMElement();
  });
});
