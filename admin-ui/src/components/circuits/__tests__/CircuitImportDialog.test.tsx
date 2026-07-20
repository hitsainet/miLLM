/**
 * Tests for CircuitImportDialog (Feature 13): tab switching, JSON parse-error
 * surfacing, successful import, and staying open on a failed import so the
 * user does not lose their input.
 */

import { describe, expect, it, vi } from 'vitest';
import { fireEvent, render, screen, waitFor } from '@testing-library/react';

import { CircuitImportDialog } from '../CircuitImportDialog';

const VALID = JSON.stringify({
  kind: 'mistudio.circuit-definition',
  schema_version: '1',
  name: 'c',
  saes: [{ layer: 10 }],
  members: [{ layer: 10 }],
});

function renderDialog(overrides: Partial<Parameters<typeof CircuitImportDialog>[0]> = {}) {
  const onImport = overrides.onImport ?? vi.fn().mockResolvedValue({});
  const onClose = overrides.onClose ?? vi.fn();
  render(
    <CircuitImportDialog open onClose={onClose} onImport={onImport} {...overrides} />,
  );
  return { onImport, onClose };
}

describe('CircuitImportDialog', () => {
  it('starts on the paste tab', () => {
    renderDialog();
    expect(screen.getByTestId('paste-area')).toBeInTheDocument();
  });

  it('switches to the file tab', () => {
    renderDialog();
    fireEvent.click(screen.getByTestId('tab-file'));
    expect(screen.getByTestId('file-input')).toBeInTheDocument();
  });

  it('disables import until something is pasted', () => {
    renderDialog();
    expect(screen.getByTestId('import-paste')).toBeDisabled();
  });

  it('surfaces a JSON parse error in-dialog without losing input', async () => {
    const { onImport } = renderDialog();
    fireEvent.change(screen.getByTestId('paste-area'), {
      target: { value: '{ not json' },
    });
    fireEvent.click(screen.getByTestId('import-paste'));
    await waitFor(() => {
      expect(screen.getByTestId('parse-error')).toBeInTheDocument();
    });
    expect(onImport).not.toHaveBeenCalled();
    // Input preserved so the user can fix it.
    expect(screen.getByTestId('paste-area')).toHaveValue('{ not json');
  });

  it('imports valid JSON and closes', async () => {
    const { onImport, onClose } = renderDialog();
    fireEvent.change(screen.getByTestId('paste-area'), { target: { value: VALID } });
    fireEvent.click(screen.getByTestId('import-paste'));
    await waitFor(() => expect(onImport).toHaveBeenCalled());
    expect(onImport.mock.calls[0][0]).toMatchObject({
      kind: 'mistudio.circuit-definition',
    });
    await waitFor(() => expect(onClose).toHaveBeenCalled());
  });

  it('stays open when the import fails', async () => {
    const onImport = vi.fn().mockRejectedValue(new Error('nope'));
    const { onClose } = renderDialog({ onImport });
    fireEvent.change(screen.getByTestId('paste-area'), { target: { value: VALID } });
    fireEvent.click(screen.getByTestId('import-paste'));
    await waitFor(() => expect(onImport).toHaveBeenCalled());
    expect(onClose).not.toHaveBeenCalled();
  });
});
