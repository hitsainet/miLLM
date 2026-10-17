import { describe, it, expect } from 'vitest';
import { render, screen } from '@testing-library/react';
import { Card, CardHeader } from '../Card';

describe('Card', () => {
  it('renders children', () => {
    render(<Card>Card content</Card>);
    expect(screen.getByText('Card content')).toBeInTheDocument();
  });

  it('applies default padding', () => {
    render(<Card data-testid="card">Content</Card>);
    const card = screen.getByTestId('card');
    expect(card).toHaveClass('p-5');
  });

  it('applies small padding', () => {
    render(<Card padding="sm" data-testid="card">Content</Card>);
    const card = screen.getByTestId('card');
    expect(card).toHaveClass('p-3');
  });

  it('applies large padding', () => {
    render(<Card padding="lg" data-testid="card">Content</Card>);
    const card = screen.getByTestId('card');
    expect(card).toHaveClass('p-6');
  });

  it('applies custom className', () => {
    render(<Card className="custom-class" data-testid="card">Content</Card>);
    const card = screen.getByTestId('card');
    expect(card).toHaveClass('custom-class');
  });
});

describe('CardHeader', () => {
  it('renders title', () => {
    render(<CardHeader title="Test Title" />);
    expect(screen.getByText('Test Title')).toBeInTheDocument();
  });

  it('renders subtitle', () => {
    render(<CardHeader title="Title" subtitle="Test Subtitle" />);
    expect(screen.getByText('Test Subtitle')).toBeInTheDocument();
  });

  it('renders icon', () => {
    render(<CardHeader title="Title" icon={<span data-testid="icon">Icon</span>} />);
    expect(screen.getByTestId('icon')).toBeInTheDocument();
  });

  it('renders action slot', () => {
    render(<CardHeader title="Title" action={<button data-testid="action">Action</button>} />);
    expect(screen.getByTestId('action')).toBeInTheDocument();
  });
});
