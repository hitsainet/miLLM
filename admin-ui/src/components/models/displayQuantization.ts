import type { ModelInfo } from '@/types';

/**
 * What to call a model's quantization on screen.
 *
 * `ModelInfo.quantization` is a five-member bucket — FP32/FP16/Q8/Q4/Q2 — and
 * GGUF has dozens of quantizations that map onto it many-to-one. Q4_K_M, Q4_K_S
 * and Q4_0 are all "Q4"; Q5_K_M and Q8_0 are both "Q8". So showing the bucket
 * for a GGUF model names a quantization nobody chose: a model downloaded as
 * Q5_K_M displayed as "Q4".
 *
 * ONE rule, shared by the list row and the details card. Written twice it
 * drifts, and a card and its own list entry disagreeing about what a model is
 * would be worse than either being wrong alone.
 */
export function displayQuantization(
  model: Pick<ModelInfo, 'quantization'> & { gguf_label?: string | null },
): string | null {
  // Empty string is the stored value for "not a GGUF selection", so it must
  // fall through rather than render as blank.
  return model.gguf_label || model.quantization || null;
}
