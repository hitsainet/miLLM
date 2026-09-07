/**
 * Choosing a quantization must select every file it needs.
 *
 * The picker's whole job is to make a QUANTIZATION the unit of choice rather
 * than a file. On a 7B repo the two are identical and any test passes; the
 * fixture here is taken from a 70B repo, where `Q6_K` is genuinely two files.
 *
 * MUTATION CONTROLS (each must turn this file red):
 *   * render one row per FILE instead of per quant -> "one row per quantization" fails
 *   * drop the parts badge                         -> "says a quant is split" fails
 *   * report `files[0].size_bytes` as the size     -> "size covers every part" fails
 *   * make fitsInVram always return 'fits'         -> "too large" fails
 */

import { describe, expect, it, vi } from 'vitest';
import { render, screen, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';

import { GGUFQuantPicker, fitsInVram, formatBytes } from '../GGUFQuantPicker';
import type { GGUFQuantInfo } from '../../../types/api';

const GB = 1024 ** 3;

// From bartowski/Qwen2.5-72B-Instruct-GGUF: Q6_K really is two files.
const QUANTS: GGUFQuantInfo[] = [
  {
    label: 'Q4_K_M',
    files: [{ path: 'X-Q4_K_M.gguf', size_bytes: 4 * GB }],
    total_size_bytes: 4 * GB,
    is_split: false,
    quant_parsed: true,
  },
  {
    label: 'Q6_K',
    files: [
      { path: 'X-Q6_K/X-Q6_K-00001-of-00002.gguf', size_bytes: 34 * GB },
      { path: 'X-Q6_K/X-Q6_K-00002-of-00002.gguf', size_bytes: 30 * GB },
    ],
    total_size_bytes: 64 * GB,
    is_split: true,
    quant_parsed: true,
  },
];

describe('GGUFQuantPicker', () => {
  it('renders one row per QUANTIZATION, not per file', () => {
    render(
      <GGUFQuantPicker quants={QUANTS} selectedLabel={null} onSelect={vi.fn()} />,
    );

    expect(screen.getByTestId('gguf-quant-Q4_K_M')).toBeInTheDocument();
    expect(screen.getByTestId('gguf-quant-Q6_K')).toBeInTheDocument();
    // Three files, two choices. A per-file list would show three.
    expect(screen.getAllByRole('radio')).toHaveLength(2);
  });

  it('says when a quantization is SPLIT, and how many parts', () => {
    render(
      <GGUFQuantPicker quants={QUANTS} selectedLabel={null} onSelect={vi.fn()} />,
    );

    const split = screen.getByTestId('gguf-quant-Q6_K');
    expect(within(split).getByText('2 parts')).toBeInTheDocument();
    // The single-file quant must NOT claim to be split.
    const single = screen.getByTestId('gguf-quant-Q4_K_M');
    expect(within(single).queryByText(/parts/)).not.toBeInTheDocument();
  });

  it('shows a size covering EVERY part, not just the first', () => {
    render(
      <GGUFQuantPicker quants={QUANTS} selectedLabel={null} onSelect={vi.fn()} />,
    );

    const split = screen.getByTestId('gguf-quant-Q6_K');
    // 64 GB total; the first part alone is 34 GB.
    expect(within(split).getByText('64.00 GB')).toBeInTheDocument();
    expect(within(split).queryByText('34.00 GB')).not.toBeInTheDocument();
  });

  it('reports the chosen label upward', async () => {
    const onSelect = vi.fn();
    render(
      <GGUFQuantPicker quants={QUANTS} selectedLabel={null} onSelect={onSelect} />,
    );

    await userEvent.click(screen.getByTestId('gguf-quant-Q6_K'));

    expect(onSelect).toHaveBeenCalledWith('Q6_K');
  });

  it('marks a name it could not parse, rather than hiding the file', () => {
    render(
      <GGUFQuantPicker
        quants={[
          {
            label: 'mystery.gguf',
            files: [{ path: 'mystery.gguf', size_bytes: GB }],
            total_size_bytes: GB,
            is_split: false,
            quant_parsed: false,
          },
        ]}
        selectedLabel={null}
        onSelect={vi.fn()}
      />,
    );

    expect(screen.getByText('unrecognised name')).toBeInTheDocument();
    expect(screen.getByTestId('gguf-quant-mystery.gguf')).toBeInTheDocument();
  });

  it('renders nothing for a repo with no GGUF files', () => {
    const { container } = render(
      <GGUFQuantPicker quants={[]} selectedLabel={null} onSelect={vi.fn()} />,
    );
    expect(container).toBeEmptyDOMElement();
  });

  it('marks what will not fit the card', () => {
    render(
      <GGUFQuantPicker
        quants={QUANTS}
        selectedLabel={null}
        onSelect={vi.fn()}
        gpuTotalBytes={24 * GB}
      />,
    );

    expect(within(screen.getByTestId('gguf-quant-Q4_K_M')).getByText('fits')).toBeInTheDocument();
    expect(within(screen.getByTestId('gguf-quant-Q6_K')).getByText('too large')).toBeInTheDocument();
  });

  it('says nothing about fit when the card is unknown', () => {
    render(
      <GGUFQuantPicker quants={QUANTS} selectedLabel={null} onSelect={vi.fn()} />,
    );
    // Silence beats a confident guess against an unknown card.
    expect(screen.queryByText('fits')).not.toBeInTheDocument();
    expect(screen.queryByText('too large')).not.toBeInTheDocument();
  });
});

describe('fitsInVram', () => {
  it('is null when the card is unknown', () => {
    expect(fitsInVram(4 * GB, null)).toBeNull();
    expect(fitsInVram(4 * GB, undefined)).toBeNull();
  });

  it('leaves headroom rather than calling a near-full card a fit', () => {
    // 21 GB of weights on a 24 GB card is not "fits" once overhead is counted.
    expect(fitsInVram(21 * GB, 24 * GB)).toBe('tight');
    expect(fitsInVram(4 * GB, 24 * GB)).toBe('fits');
    expect(fitsInVram(64 * GB, 24 * GB)).toBe('too-large');
  });
});

describe('formatBytes', () => {
  it('uses binary units, matching nvidia-smi', () => {
    expect(formatBytes(4 * GB)).toBe('4.00 GB');
    expect(formatBytes(512 * 1024 ** 2)).toBe('512 MB');
  });

  it('does not render a missing size as zero', () => {
    expect(formatBytes(0)).toBe('—');
  });
});
