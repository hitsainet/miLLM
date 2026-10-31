// The app's version, as package.json says. vite.config.ts and vitest.config.ts define
// __APP_VERSION__ from it, so the sidebar can no longer drift from the package (it was a
// hand-kept '0.5.0' literal; ENH-004). Origin: miDataworks frontend/src/config/version.ts @ 32cf406.
declare const __APP_VERSION__: string;

export const APP_VERSION: string = __APP_VERSION__;
