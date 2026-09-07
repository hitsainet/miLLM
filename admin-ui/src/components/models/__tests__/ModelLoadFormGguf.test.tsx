/**
 * The Quantization dropdown must describe the repository in the box.
 *
 * For an ordinary safetensors model the options are bitsandbytes levels applied
 * at load time — a property of the runtime, identical for every repo. For a
 * GGUF repo they are files baked ahead of time, so they differ per repo and the
 * static list is meaningless there.
 *
 * MUTATION CONTROLS (each must turn this file red):
 *   * always render the static list      -> "lists the repo's quantizations" fails
 *   * drop the previewedRepoId guard     -> "stops describing an edited repo" fails
 *   * submit only files[0]               -> "submits every part" fails
 */

import { describe, expect, it, vi } from 'vitest';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';

import { ModelLoadForm } from '../ModelLoadForm';
import type { GGUFQuantInfo } from '../../../types/api';

const GB = 1024 ** 3;

const QUANTS: GGUFQuantInfo[] = [
  {
    label: 'Q4_K_M',
    files: [{ path: 'm.Q4_K_M.gguf', size_bytes: 18 * GB }],
    total_size_bytes: 18 * GB,
    is_split: false,
    quant_parsed: true,
  },
  {
    label: 'Q6_K',
    files: [
      { path: 'm-Q6_K/m-Q6_K-00001-of-00002.gguf', size_bytes: 34 * GB },
      { path: 'm-Q6_K/m-Q6_K-00002-of-00002.gguf', size_bytes: 30 * GB },
    ],
    total_size_bytes: 64 * GB,
    is_split: true,
    quant_parsed: true,
  },
];

async function typeRepo(repo: string) {
  const input = screen.getByLabelText(/Hugging Face Repository ID/i);
  await userEvent.clear(input);
  await userEvent.type(input, repo);
}

describe('ModelLoadForm quantization dropdown', () => {
  it('shows the runtime levels when the repo is not GGUF', () => {
    render(<ModelLoadForm onSubmit={vi.fn()} />);

    const select = screen.getByLabelText(/Quantization/i);
    expect(select).toHaveTextContent('Q4 - 4-bit (Recommended)');
    expect(select).toHaveTextContent('FP32 - Full Precision');
  });

  it("lists the repo's own quantizations, with measured sizes", async () => {
    render(
      <ModelLoadForm onSubmit={vi.fn()} ggufQuants={QUANTS} previewedRepoId="o/m-GGUF" />,
    );
    await typeRepo('o/m-GGUF');

    const select = screen.getByLabelText(/Quantization/i);
    expect(select).toHaveTextContent('Q4_K_M — 18.00 GB');
    // A split quant says so here too, not only in the modal.
    expect(select).toHaveTextContent('Q6_K — 64.00 GB (2 parts)');
    // The runtime levels are meaningless for a pre-quantized file.
    expect(select).not.toHaveTextContent('FP32 - Full Precision');
  });

  it('stops describing the repo once the box is edited', async () => {
    render(
      <ModelLoadForm onSubmit={vi.fn()} ggufQuants={QUANTS} previewedRepoId="o/m-GGUF" />,
    );
    await typeRepo('o/m-GGUF');
    expect(screen.getByLabelText(/Quantization/i)).toHaveTextContent('Q4_K_M');

    await typeRepo('someone/else');

    // Those file paths do not exist in the new repo. A preview is true of ONE
    // repository, and showing it against another is a stale verdict.
    const select = screen.getByLabelText(/Quantization/i);
    expect(select).not.toHaveTextContent('Q4_K_M');
    expect(select).toHaveTextContent('Q4 - 4-bit (Recommended)');
  });

  it('submits EVERY part of a split quantization', async () => {
    const onSubmit = vi.fn();
    render(
      <ModelLoadForm onSubmit={onSubmit} ggufQuants={QUANTS} previewedRepoId="o/m-GGUF" />,
    );
    await typeRepo('o/m-GGUF');
    await userEvent.selectOptions(screen.getByLabelText(/Quantization/i), 'Q6_K');
    await userEvent.click(screen.getByRole('button', { name: /Download & Load Model/i }));

    expect(onSubmit).toHaveBeenCalledTimes(1);
    const data = onSubmit.mock.calls[0][0];
    expect(data.gguf_label).toBe('Q6_K');
    expect(data.gguf_files).toEqual([
      'm-Q6_K/m-Q6_K-00001-of-00002.gguf',
      'm-Q6_K/m-Q6_K-00002-of-00002.gguf',
    ]);
  });

  it('sends no gguf fields for an ordinary model', async () => {
    const onSubmit = vi.fn();
    render(<ModelLoadForm onSubmit={onSubmit} />);
    await typeRepo('google/gemma-2-2b');
    await userEvent.click(screen.getByRole('button', { name: /Download & Load Model/i }));

    const data = onSubmit.mock.calls[0][0];
    // Absent means "the whole repository", which is correct here.
    expect(data.gguf_files).toBeUndefined();
    expect(data.gguf_label).toBeUndefined();
  });
});
