/**
 * Neuronpedia feature-link derivation from the attached SAE's metadata.
 *
 * URL format: {base}/{model-slug}/{layer}-res-{d_sae_k}k
 * e.g. https://neuronpedia.hitsai.net/lfm2.5-1.2b-instruct/12-res-8k
 *
 * Shared by the Probe page and the Steering sliders so feature links always
 * reflect the SAE actually attached (never a hardcoded model family).
 */

export const NEURONPEDIA_BASE = 'https://neuronpedia.hitsai.net';

interface SAELike {
  trained_on?: string | null;
  trained_layer?: number | null;
  d_sae?: number | null;
}

export function neuronpediaBaseUrl(attachedSAE: SAELike | null | undefined): string {
  if (!attachedSAE) return NEURONPEDIA_BASE;
  // Model slug: "LiquidAI/LFM2.5-1.2B-Instruct" → "lfm2.5-1.2b-instruct"
  const modelSlug =
    (attachedSAE.trained_on || '').split('/').pop()?.toLowerCase().replace(/_/g, '-') || '';
  const layer = attachedSAE.trained_layer ?? 0;
  const dSaeK = Math.round((attachedSAE.d_sae || 0) / 1000);
  return modelSlug ? `${NEURONPEDIA_BASE}/${modelSlug}/${layer}-res-${dSaeK}k` : NEURONPEDIA_BASE;
}

export function neuronpediaFeatureUrl(
  attachedSAE: SAELike | null | undefined,
  featureIndex: number
): string {
  return `${neuronpediaBaseUrl(attachedSAE).replace(/\/$/, '')}/${featureIndex}`;
}
