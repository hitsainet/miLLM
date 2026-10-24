import { describe, it, expect, vi } from 'vitest';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { Select } from '../Select';

const mockOptions = [
  { value: 'option1', label: 'Option 1' },
  { value: 'option2', label: 'Option 2' },
  { value: 'option3', label: 'Option 3', disabled: true },
];

describe('Select', () => {
  it('renders without crashing', () => {
    render(<Select options={mockOptions} />);
    expect(screen.getByRole('combobox')).toBeInTheDocument();
  });

  it('renders options', () => {
    render(<Select options={mockOptions} />);
    expect(screen.getByText('Option 1')).toBeInTheDocument();
    expect(screen.getByText('Option 2')).toBeInTheDocument();
    expect(screen.getByText('Option 3')).toBeInTheDocument();
  });

  it('renders with label', () => {
    render(<Select options={mockOptions} label="Select an option" />);
    expect(screen.getByText('Select an option')).toBeInTheDocument();
  });

  it('renders placeholder', () => {
    render(<Select options={mockOptions} placeholder="Choose..." defaultValue="" />);
    expect(screen.getByText('Choose...')).toBeInTheDocument();
  });

  it('displays error message', () => {
    render(<Select options={mockOptions} error="Selection is required" />);
    expect(screen.getByText('Selection is required')).toBeInTheDocument();
  });

  it('displays helper text', () => {
    render(<Select options={mockOptions} helper="Pick your favorite" />);
    expect(screen.getByText('Pick your favorite')).toBeInTheDocument();
  });

  it('hides helper when error is shown', () => {
    render(<Select options={mockOptions} helper="Helper text" error="Error" />);
    expect(screen.getByText('Error')).toBeInTheDocument();
    expect(screen.queryByText('Helper text')).not.toBeInTheDocument();
  });

  it('handles selection change', async () => {
    const user = userEvent.setup();
    const onChange = vi.fn();
    render(<Select options={mockOptions} onChange={onChange} />);

    const select = screen.getByRole('combobox');
    await user.selectOptions(select, 'option2');

    expect(onChange).toHaveBeenCalled();
  });

  it('disables individual options', () => {
    render(<Select options={mockOptions} />);
    const disabledOption = screen.getByText('Option 3');
    expect(disabledOption).toBeDisabled();
  });

  it('applies error styling when error prop is provided', () => {
    render(<Select options={mockOptions} error="Error" />);
    const select = screen.getByRole('combobox');
    expect(select).toHaveClass('border-red-500/50');
  });

  it('associates label with select via htmlFor', () => {
    render(<Select options={mockOptions} label="Category" id="category-select" />);
    const label = screen.getByText('Category');
    expect(label).toHaveAttribute('for', 'category-select');
  });
});
