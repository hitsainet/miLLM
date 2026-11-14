/**
 * Toast (007 task 17.7): the container renders the store's toasts with their variant, a toast
 * dismisses itself after its duration (default 5 s, 0 = sticky), and the close button removes
 * exactly that toast.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { act, fireEvent, render, screen } from '@testing-library/react';
import { ToastContainer } from '../Toast';
import { useUIStore } from '@/stores/uiStore';

const add = (t: Parameters<ReturnType<typeof useUIStore.getState>['addToast']>[0]) =>
  act(() => useUIStore.getState().addToast(t));

describe('ToastContainer', () => {
  beforeEach(() => {
    vi.useFakeTimers();
    useUIStore.setState({ toasts: [] });
  });
  afterEach(() => {
    vi.useRealTimers();
    useUIStore.setState({ toasts: [] });
  });

  it('renders nothing with no toasts', () => {
    const { container } = render(<ToastContainer />);
    expect(container).toBeEmptyDOMElement();
  });

  it('renders each toast with its variant style', () => {
    render(<ToastContainer />);
    add({ type: 'success', message: 'saved' });
    add({ type: 'error', message: 'failed' });
    expect(screen.getByText('saved').parentElement?.className).toContain('text-green-400');
    expect(screen.getByText('failed').parentElement?.className).toContain('text-red-400');
  });

  it('dismisses itself after the default 5 seconds', () => {
    render(<ToastContainer />);
    add({ type: 'info', message: 'hello' });
    act(() => vi.advanceTimersByTime(4999));
    expect(screen.getByText('hello')).toBeInTheDocument();
    act(() => vi.advanceTimersByTime(1));
    expect(screen.queryByText('hello')).not.toBeInTheDocument();
    expect(useUIStore.getState().toasts).toHaveLength(0);
  });

  it('honours a custom duration, and 0 never auto-dismisses', () => {
    render(<ToastContainer />);
    add({ type: 'warning', message: 'quick', duration: 100 });
    add({ type: 'warning', message: 'sticky', duration: 0 });
    act(() => vi.advanceTimersByTime(60_000));
    expect(screen.queryByText('quick')).not.toBeInTheDocument();
    expect(screen.getByText('sticky')).toBeInTheDocument();
  });

  it('the close button removes only that toast', () => {
    render(<ToastContainer />);
    add({ type: 'info', message: 'one', duration: 0 });
    add({ type: 'info', message: 'two', duration: 0 });
    const close = screen.getByText('one').parentElement!.querySelector('button')!;
    fireEvent.click(close);
    expect(screen.queryByText('one')).not.toBeInTheDocument();
    expect(screen.getByText('two')).toBeInTheDocument();
  });
});
