/**
 * Slider (007 task 15.7): live onChange while dragging, onChangeEnd once on release with the
 * LATEST value, and the external value ignored mid-drag so a server echo cannot yank the thumb.
 */
import { describe, expect, it, vi } from 'vitest';
import { fireEvent, render, screen } from '@testing-library/react';
import { Slider } from '../Slider';

describe('Slider', () => {
  it('shows the value with a sign and the label', () => {
    render(<Slider value={2.5} onChange={() => {}} label="Strength" />);
    expect(screen.getByText('Strength')).toBeInTheDocument();
    expect(screen.getByText('+2.5')).toBeInTheDocument();
  });

  it('shows a negative value without a plus', () => {
    render(<Slider value={-3} onChange={() => {}} min={-10} max={10} />);
    expect(screen.getByText('-3.0')).toBeInTheDocument();
  });

  it('hides the value when showValue is false', () => {
    render(<Slider value={4} onChange={() => {}} showValue={false} />);
    expect(screen.queryByText('+4.0')).not.toBeInTheDocument();
  });

  it('uses formatValue', () => {
    render(<Slider value={4} onChange={() => {}} formatValue={(v) => `${v}x`} />);
    expect(screen.getByText('+4x')).toBeInTheDocument();
  });

  it('passes min, max and step to the range input', () => {
    render(<Slider value={0} onChange={() => {}} min={-200} max={200} step={0.5} />);
    const input = screen.getByRole('slider');
    expect(input).toHaveAttribute('min', '-200');
    expect(input).toHaveAttribute('max', '200');
    expect(input).toHaveAttribute('step', '0.5');
  });

  it('calls onChange on every move and onChangeEnd once, with the last value, on release', () => {
    const onChange = vi.fn();
    const onChangeEnd = vi.fn();
    render(<Slider value={0} onChange={onChange} onChangeEnd={onChangeEnd} min={-10} max={10} />);
    const input = screen.getByRole('slider');
    fireEvent.mouseDown(input);
    fireEvent.change(input, { target: { value: '3' } });
    fireEvent.change(input, { target: { value: '7' } });
    expect(onChange).toHaveBeenNthCalledWith(1, 3);
    expect(onChange).toHaveBeenNthCalledWith(2, 7);
    expect(onChangeEnd).not.toHaveBeenCalled();
    fireEvent.mouseUp(input);
    expect(onChangeEnd).toHaveBeenCalledTimes(1);
    expect(onChangeEnd).toHaveBeenCalledWith(7);
  });

  it('touch release also commits', () => {
    const onChangeEnd = vi.fn();
    render(<Slider value={0} onChange={() => {}} onChangeEnd={onChangeEnd} />);
    const input = screen.getByRole('slider');
    fireEvent.touchStart(input);
    fireEvent.change(input, { target: { value: '12' } });
    fireEvent.touchEnd(input);
    expect(onChangeEnd).toHaveBeenCalledWith(12);
  });

  it('ignores an external value while dragging, and follows it otherwise', () => {
    const { rerender } = render(<Slider value={1} onChange={() => {}} />);
    const input = screen.getByRole('slider');
    fireEvent.mouseDown(input);
    fireEvent.change(input, { target: { value: '9' } });
    rerender(<Slider value={2} onChange={() => {}} />);
    expect(input).toHaveValue('9');
    fireEvent.mouseUp(input);
    rerender(<Slider value={5} onChange={() => {}} />);
    expect(input).toHaveValue('5');
  });

  it('is disabled when asked', () => {
    render(<Slider value={0} onChange={() => {}} disabled />);
    expect(screen.getByRole('slider')).toBeDisabled();
  });
});
