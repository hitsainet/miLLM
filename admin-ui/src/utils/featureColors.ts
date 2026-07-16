/**
 * Feature tile color palette (miStudio-aligned).
 *
 * Ordered like miStudio's steering palette (teal, blue, purple, amber, rose,
 * then more hues); features cycle through it by their position in a list.
 * Tailwind requires literal class names, so every hue carries its full class
 * set; `accent` is the raw hex used for slider tracks/thumbs.
 */

export interface FeatureColorClasses {
  /** small colored dot */
  dot: string;
  /** feature index / accent text */
  text: string;
  /** card border */
  border: string;
  /** subtle tinted card background */
  bg: string;
  /** raw hex for gradient tracks / accent-color */
  accent: string;
}

const PALETTE: FeatureColorClasses[] = [
  { dot: 'bg-teal-400', text: 'text-teal-400', border: 'border-teal-500/60', bg: 'bg-teal-500/5', accent: '#2dd4bf' },
  { dot: 'bg-blue-400', text: 'text-blue-400', border: 'border-blue-500/60', bg: 'bg-blue-500/5', accent: '#60a5fa' },
  { dot: 'bg-purple-400', text: 'text-purple-400', border: 'border-purple-500/60', bg: 'bg-purple-500/5', accent: '#c084fc' },
  { dot: 'bg-amber-400', text: 'text-amber-400', border: 'border-amber-500/60', bg: 'bg-amber-500/5', accent: '#fbbf24' },
  { dot: 'bg-rose-400', text: 'text-rose-400', border: 'border-rose-500/60', bg: 'bg-rose-500/5', accent: '#fb7185' },
  { dot: 'bg-cyan-400', text: 'text-cyan-400', border: 'border-cyan-500/60', bg: 'bg-cyan-500/5', accent: '#22d3ee' },
  { dot: 'bg-lime-400', text: 'text-lime-400', border: 'border-lime-500/60', bg: 'bg-lime-500/5', accent: '#a3e635' },
  { dot: 'bg-orange-400', text: 'text-orange-400', border: 'border-orange-500/60', bg: 'bg-orange-500/5', accent: '#fb923c' },
  { dot: 'bg-fuchsia-400', text: 'text-fuchsia-400', border: 'border-fuchsia-500/60', bg: 'bg-fuchsia-500/5', accent: '#e879f9' },
  { dot: 'bg-sky-400', text: 'text-sky-400', border: 'border-sky-500/60', bg: 'bg-sky-500/5', accent: '#38bdf8' },
  { dot: 'bg-emerald-400', text: 'text-emerald-400', border: 'border-emerald-500/60', bg: 'bg-emerald-500/5', accent: '#34d399' },
  { dot: 'bg-violet-400', text: 'text-violet-400', border: 'border-violet-500/60', bg: 'bg-violet-500/5', accent: '#a78bfa' },
  { dot: 'bg-pink-400', text: 'text-pink-400', border: 'border-pink-500/60', bg: 'bg-pink-500/5', accent: '#f472b6' },
  { dot: 'bg-indigo-400', text: 'text-indigo-400', border: 'border-indigo-500/60', bg: 'bg-indigo-500/5', accent: '#818cf8' },
  { dot: 'bg-yellow-400', text: 'text-yellow-400', border: 'border-yellow-500/60', bg: 'bg-yellow-500/5', accent: '#facc15' },
  { dot: 'bg-red-400', text: 'text-red-400', border: 'border-red-500/60', bg: 'bg-red-500/5', accent: '#f87171' },
  { dot: 'bg-green-400', text: 'text-green-400', border: 'border-green-500/60', bg: 'bg-green-500/5', accent: '#4ade80' },
];

export function featureColor(order: number): FeatureColorClasses {
  return PALETTE[((order % PALETTE.length) + PALETTE.length) % PALETTE.length];
}

/**
 * Gradient slider track for a steering range symmetric around zero:
 * danger red at both extremes, warning amber inside, the feature color
 * across the safe middle band (mirrors miStudio's warning-zone track).
 */
export function steeringTrackGradient(accent: string): string {
  return (
    `linear-gradient(to right, ` +
    `#7f1d1d 0%, #b45309 12%, ${accent}55 30%, ${accent} 50%, ` +
    `${accent}55 70%, #b45309 88%, #7f1d1d 100%)`
  );
}
