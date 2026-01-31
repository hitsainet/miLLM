import { describe, it, expect } from 'vitest';
import { render } from '@testing-library/react';
import { Spinner } from '../Spinner';

describe('Spinner', () => {
  it('renders without crashing', () => {
    const { container } = render(<Spinner />);
    const spinner = container.querySelector('svg');
    expect(spinner).toBeInTheDocument();
  });

  it('applies small size', () => {
    const { container } = render(<Spinner size="sm" />);
    const spinner = container.querySelector('svg');
    expect(spinner).toHaveClass('w-4');
  });

  it('applies medium size', () => {
    const { container } = render(<Spinner size="md" />);
    const spinner = container.querySelector('svg');
    expect(spinner).toHaveClass('w-6');
  });

  it('applies large size', () => {
    const { container } = render(<Spinner size="lg" />);
    const spinner = container.querySelector('svg');
    expect(spinner).toHaveClass('w-8');
  });

  it('applies custom className', () => {
    const { container } = render(<Spinner className="custom-spinner" />);
    const spinner = container.querySelector('svg');
    expect(spinner).toHaveClass('custom-spinner');
  });
});
