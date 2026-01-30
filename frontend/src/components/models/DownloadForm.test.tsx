/**
 * Tests for DownloadForm component.
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { DownloadForm } from './DownloadForm';

describe('DownloadForm', () => {
  const mockOnDownload = vi.fn();

  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('should render form elements', () => {
    render(<DownloadForm onDownload={mockOnDownload} />);

    expect(screen.getByText('Download Model')).toBeInTheDocument();
    expect(screen.getByLabelText(/huggingface repository/i)).toBeInTheDocument();
    expect(screen.getByLabelText(/quantization/i)).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /download model/i })).toBeInTheDocument();
  });

  it('should have correct placeholder text', () => {
    render(<DownloadForm onDownload={mockOnDownload} />);

    const input = screen.getByPlaceholderText('google/gemma-2-2b');
    expect(input).toBeInTheDocument();
  });

  it('should have quantization options', () => {
    render(<DownloadForm onDownload={mockOnDownload} />);

    const select = screen.getByLabelText(/quantization/i);
    expect(select).toBeInTheDocument();

    // Check that Q4 is default selected
    expect(select).toHaveValue('Q4');
  });

  it('should show error when submitting empty repo ID', async () => {
    const user = userEvent.setup();
    render(<DownloadForm onDownload={mockOnDownload} />);

    const submitButton = screen.getByRole('button', { name: /download model/i });
    await user.click(submitButton);

    expect(screen.getByText('Please enter a repository ID')).toBeInTheDocument();
    expect(mockOnDownload).not.toHaveBeenCalled();
  });

  it('should show error when repo ID format is invalid', async () => {
    const user = userEvent.setup();
    render(<DownloadForm onDownload={mockOnDownload} />);

    const input = screen.getByLabelText(/huggingface repository/i);
    await user.type(input, 'invalid-repo-id');

    const submitButton = screen.getByRole('button', { name: /download model/i });
    await user.click(submitButton);

    expect(screen.getByText('Repository ID should be in format: owner/model-name')).toBeInTheDocument();
    expect(mockOnDownload).not.toHaveBeenCalled();
  });

  it('should call onDownload with correct parameters', async () => {
    const user = userEvent.setup();
    mockOnDownload.mockResolvedValue(undefined);
    render(<DownloadForm onDownload={mockOnDownload} />);

    const input = screen.getByLabelText(/huggingface repository/i);
    await user.type(input, 'google/gemma-2-2b');

    const submitButton = screen.getByRole('button', { name: /download model/i });
    await user.click(submitButton);

    expect(mockOnDownload).toHaveBeenCalledWith('google/gemma-2-2b', 'Q4');
  });

  it('should clear input after successful download', async () => {
    const user = userEvent.setup();
    mockOnDownload.mockResolvedValue(undefined);
    render(<DownloadForm onDownload={mockOnDownload} />);

    const input = screen.getByLabelText(/huggingface repository/i);
    await user.type(input, 'google/gemma-2-2b');

    const submitButton = screen.getByRole('button', { name: /download model/i });
    await user.click(submitButton);

    await waitFor(() => {
      expect(input).toHaveValue('');
    });
  });

  it('should show error when download fails', async () => {
    const user = userEvent.setup();
    mockOnDownload.mockRejectedValue(new Error('Download failed'));
    render(<DownloadForm onDownload={mockOnDownload} />);

    const input = screen.getByLabelText(/huggingface repository/i);
    await user.type(input, 'google/gemma-2-2b');

    const submitButton = screen.getByRole('button', { name: /download model/i });
    await user.click(submitButton);

    await waitFor(() => {
      expect(screen.getByText('Download failed')).toBeInTheDocument();
    });
  });

  it('should allow changing quantization', async () => {
    const user = userEvent.setup();
    mockOnDownload.mockResolvedValue(undefined);
    render(<DownloadForm onDownload={mockOnDownload} />);

    const select = screen.getByLabelText(/quantization/i);
    await user.selectOptions(select, 'FP16');

    const input = screen.getByLabelText(/huggingface repository/i);
    await user.type(input, 'google/gemma-2-2b');

    const submitButton = screen.getByRole('button', { name: /download model/i });
    await user.click(submitButton);

    expect(mockOnDownload).toHaveBeenCalledWith('google/gemma-2-2b', 'FP16');
  });

  it('should disable form when isLoading is true', () => {
    render(<DownloadForm onDownload={mockOnDownload} isLoading={true} />);

    const input = screen.getByLabelText(/huggingface repository/i);
    const select = screen.getByLabelText(/quantization/i);
    const button = screen.getByRole('button', { name: /download model/i });

    expect(input).toBeDisabled();
    expect(select).toBeDisabled();
    // Button may show loading state
    expect(button).toBeInTheDocument();
  });

  it('should trim whitespace from repo ID', async () => {
    const user = userEvent.setup();
    mockOnDownload.mockResolvedValue(undefined);
    render(<DownloadForm onDownload={mockOnDownload} />);

    const input = screen.getByLabelText(/huggingface repository/i);
    await user.type(input, '  google/gemma-2-2b  ');

    const submitButton = screen.getByRole('button', { name: /download model/i });
    await user.click(submitButton);

    expect(mockOnDownload).toHaveBeenCalledWith('google/gemma-2-2b', 'Q4');
  });

  it('should show helper text', () => {
    render(<DownloadForm onDownload={mockOnDownload} />);

    expect(screen.getByText('Enter a HuggingFace model repository ID')).toBeInTheDocument();
  });
});
