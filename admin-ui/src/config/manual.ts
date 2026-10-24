// Where the public miLLM manual lives (manual/, published to GitHub Pages under the custom domain
// docs-millm.hitsai.net). The Help page, every Help link and the sidebar's book icon point into it;
// the in-app text stays a summary. This is the ONE place the address is written: set
// VITE_MANUAL_BASE_URL at build time to point at another copy (a self-hosted build of manual/).
//
// Origin: miDataworks frontend/src/config/manual.ts @ 32cf406 (ENH-004).
//
// manualLinks.test.tsx checks that the default equals manual/docusaurus.config.ts `url + baseUrl`
// (the sidebar used to link https://hitsainet.github.io/miLLM/ while the manual said
// docs-millm.hitsai.net, and nothing compared the two), and that every page and heading a Help
// topic names exists in the manual's sources.

export const DEFAULT_MANUAL_BASE_URL = 'https://docs-millm.hitsai.net/';

/** The manual's base URL, always ending in "/". An empty or missing value takes the default. */
export function manualBaseUrl(raw: string | undefined = import.meta.env.VITE_MANUAL_BASE_URL): string {
  const value = (raw ?? '').trim() || DEFAULT_MANUAL_BASE_URL;
  return value.endsWith('/') ? value : `${value}/`;
}

export const MANUAL_BASE_URL = manualBaseUrl();

/** A manual page, e.g. "features/circuits#serving-a-circuit", as an absolute URL. */
export function manualUrl(page: string, base: string = MANUAL_BASE_URL): string {
  return `${base}${page.replace(/^\/+/, '')}`;
}
