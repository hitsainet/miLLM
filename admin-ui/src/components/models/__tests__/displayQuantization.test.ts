/**
 * A model must be named by the quantization it actually is.
 *
 * `quantization` is a five-member bucket and GGUF has dozens of quantizations
 * mapping onto it many-to-one, so a GGUF model shown by its bucket is named
 * something nobody chose — the zora-v1.13 card read "Q4" for a model downloaded
 * as Q5_K_M, on both the list row and the details card.
 *
 * MUTATION CONTROLS (each must turn this file red):
 *   * return model.quantization first -> "prefers the exact label" fails
 *   * use ?? instead of ||            -> "empty label falls through" fails
 */

import { describe, expect, it } from 'vitest';

import { displayQuantization } from '../displayQuantization';

describe('displayQuantization', () => {
  it('prefers the exact GGUF label over the coarse bucket', () => {
    expect(
      displayQuantization({ quantization: 'Q8', gguf_label: 'Q5_K_M' }),
    ).toBe('Q5_K_M');
  });

  it('keeps distinct quantizations distinct', () => {
    // All three are "Q4" to the enum; showing that would merge them on screen.
    const shown = ['Q4_K_M', 'Q4_K_S', 'Q4_0'].map((gguf_label) =>
      displayQuantization({ quantization: 'Q4', gguf_label }),
    );
    expect(new Set(shown).size).toBe(3);
  });

  it('falls back to the bucket for an ordinary model', () => {
    expect(displayQuantization({ quantization: 'FP16' })).toBe('FP16');
    expect(displayQuantization({ quantization: 'FP16', gguf_label: null })).toBe('FP16');
  });

  it('treats an EMPTY label as "no GGUF selection", not as a name', () => {
    // '' is what the column stores for an ordinary model. `??` would render it
    // as a blank quantization instead of falling through.
    expect(displayQuantization({ quantization: 'Q4', gguf_label: '' })).toBe('Q4');
  });
});
