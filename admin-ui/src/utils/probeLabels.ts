/**
 * The separation a probe was fitted to make, from its training view's label mapping.
 *
 * ⚠ THE TILE NAMED NEITHER SIDE. Until 2026-10-01 a probe row read
 * `meta-llama/Llama-3.1-8B-Instruct · L11 · mean · dense residual · scope all` — where it reads
 * and nothing about what it detects. Two imported probes differed only in the run that produced
 * them, and the list could not tell them apart.
 *
 * ⚠ THIS MIRRORS miStudio's `probeConcept.ts`, DELIBERATELY AND WITHOUT SHARED CODE. The two
 * repos do not share a frontend, so the format string is the thing that can drift. Both sides
 * pin the same real mapping — `{"low-stakes": "negative", "high-stakes": "positive"}`, which is
 * what the models-under-pressure corpus actually carries — to the same output,
 * `"high-stakes vs low-stakes"`, so a divergence fails a test in whichever repo moved.
 *
 * The values are the corpus's own strings. Nothing here translates, title-cases or prettifies:
 * a reader checking the tile against the corpus must find the same token.
 */

/** The three sides of a label mapping, each sorted so a caption does not reorder. */
export interface TrainingLabels {
  /** Raw label values mapped to `positive` — what the probe fires on. */
  positive: string[];
  /** Raw label values mapped to `negative` — what it was fitted to separate them FROM. */
  negative: string[];
  /** Raw label values mapped to `excluded` — dropped before fitting, so on neither side. */
  excluded: string[];
}

function labelsMappedTo(mapping: Record<string, string>, target: string): string[] {
  return Object.keys(mapping)
    .filter((label) => String(mapping[label]).toLowerCase() === target)
    .sort();
}

export function trainingLabels(
  mapping: Record<string, string> | null | undefined
): TrainingLabels {
  const m = mapping ?? {};
  return {
    positive: labelsMappedTo(m, 'positive'),
    negative: labelsMappedTo(m, 'negative'),
    excluded: labelsMappedTo(m, 'excluded'),
  };
}

/**
 * `"high-stakes vs low-stakes"`, or `null`.
 *
 * `null` when either side is empty, because a half-stated separation is misleading in a way a
 * blank is not: "trained on high-stakes" reads as a corpus, not as a boundary, and the boundary
 * is the whole of what a linear probe is.
 */
export function labelSeparation(
  mapping: Record<string, string> | null | undefined
): string | null {
  const { positive, negative } = trainingLabels(mapping);
  if (positive.length === 0 || negative.length === 0) return null;
  return `${positive.join(' or ')} vs ${negative.join(' or ')}`;
}
