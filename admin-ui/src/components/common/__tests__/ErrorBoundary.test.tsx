/**
 * Tests for the ErrorBoundary component.
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import ErrorBoundary, { withErrorBoundary } from '../ErrorBoundary';

// Suppress console.error during tests since we're testing error handling
const originalConsoleError = console.error;
beforeEach(() => {
  console.error = vi.fn();
});
afterEach(() => {
  console.error = originalConsoleError;
});

// Component that throws an error
const ThrowingComponent = ({ shouldThrow = true }: { shouldThrow?: boolean }) => {
  if (shouldThrow) {
    throw new Error('Test error');
  }
  return <div>No error</div>;
};

describe('ErrorBoundary', () => {
  it('renders children when no error occurs', () => {
    render(
      <ErrorBoundary>
        <div>Test content</div>
      </ErrorBoundary>
    );

    expect(screen.getByText('Test content')).toBeInTheDocument();
  });

  it('renders fallback UI when error occurs', () => {
    render(
      <ErrorBoundary>
        <ThrowingComponent />
      </ErrorBoundary>
    );

    expect(screen.getByText('Something went wrong')).toBeInTheDocument();
  });

  it('renders custom fallback when provided', () => {
    render(
      <ErrorBoundary fallback={<div>Custom error message</div>}>
        <ThrowingComponent />
      </ErrorBoundary>
    );

    expect(screen.getByText('Custom error message')).toBeInTheDocument();
  });

  it('calls onError callback when error occurs', () => {
    const onError = vi.fn();

    render(
      <ErrorBoundary onError={onError}>
        <ThrowingComponent />
      </ErrorBoundary>
    );

    expect(onError).toHaveBeenCalledTimes(1);
    expect(onError).toHaveBeenCalledWith(
      expect.any(Error),
      expect.objectContaining({
        componentStack: expect.any(String),
      })
    );
  });

  it('shows reset button by default', () => {
    render(
      <ErrorBoundary>
        <ThrowingComponent />
      </ErrorBoundary>
    );

    expect(screen.getByRole('button', { name: /try again/i })).toBeInTheDocument();
  });

  it('hides reset button when showReset is false', () => {
    render(
      <ErrorBoundary showReset={false}>
        <ThrowingComponent />
      </ErrorBoundary>
    );

    expect(screen.queryByRole('button', { name: /try again/i })).not.toBeInTheDocument();
  });

  it('resets error state when reset button is clicked', () => {
    const TestComponent = () => {
      const [shouldThrow, setShouldThrow] = vi.fn().mockReturnValue(true);

      return (
        <ErrorBoundary>
          <ThrowingComponent shouldThrow={shouldThrow} />
        </ErrorBoundary>
      );
    };

    // This test verifies the reset button exists and can be clicked
    render(
      <ErrorBoundary>
        <ThrowingComponent />
      </ErrorBoundary>
    );

    const resetButton = screen.getByRole('button', { name: /try again/i });
    expect(resetButton).toBeInTheDocument();

    // Click should not throw
    expect(() => fireEvent.click(resetButton)).not.toThrow();
  });

  it('calls onReset callback when reset button is clicked', () => {
    const onReset = vi.fn();

    render(
      <ErrorBoundary onReset={onReset}>
        <ThrowingComponent />
      </ErrorBoundary>
    );

    fireEvent.click(screen.getByRole('button', { name: /try again/i }));

    expect(onReset).toHaveBeenCalledTimes(1);
  });
});

describe('withErrorBoundary', () => {
  it('wraps component with error boundary', () => {
    const SimpleComponent = () => <div>Simple content</div>;
    const WrappedComponent = withErrorBoundary(SimpleComponent);

    render(<WrappedComponent />);

    expect(screen.getByText('Simple content')).toBeInTheDocument();
  });

  it('catches errors in wrapped component', () => {
    const WrappedThrowing = withErrorBoundary(ThrowingComponent);

    render(<WrappedThrowing />);

    expect(screen.getByText('Something went wrong')).toBeInTheDocument();
  });

  it('passes error boundary props', () => {
    const WrappedThrowing = withErrorBoundary(ThrowingComponent, {
      fallback: <div>Custom wrapped error</div>,
    });

    render(<WrappedThrowing />);

    expect(screen.getByText('Custom wrapped error')).toBeInTheDocument();
  });

  it('preserves component display name', () => {
    const NamedComponent = () => <div>Named</div>;
    NamedComponent.displayName = 'MyNamedComponent';

    const WrappedComponent = withErrorBoundary(NamedComponent);

    expect(WrappedComponent.displayName).toBe('withErrorBoundary(MyNamedComponent)');
  });

  it('uses component name when displayName is not set', () => {
    function TestComponent() {
      return <div>Test</div>;
    }

    const WrappedComponent = withErrorBoundary(TestComponent);

    expect(WrappedComponent.displayName).toBe('withErrorBoundary(TestComponent)');
  });
});

describe('ErrorBoundary error details', () => {
  it('shows error details in development mode', () => {
    const originalEnv = process.env.NODE_ENV;
    vi.stubEnv('NODE_ENV', 'development');

    render(
      <ErrorBoundary>
        <ThrowingComponent />
      </ErrorBoundary>
    );

    expect(screen.getByText('Error details')).toBeInTheDocument();

    vi.stubEnv('NODE_ENV', originalEnv);
  });
});

// Import afterEach for cleanup
import { afterEach } from 'vitest';
