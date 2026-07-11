import {themes as prismThemes} from 'prism-react-renderer';
import type {Config} from '@docusaurus/types';
import type * as Preset from '@docusaurus/preset-classic';

const config: Config = {
  title: 'miLLM Manual',
  tagline: 'Mechanistic Interpretability LLM Server — Feature Steering & Real-Time Probing',
  favicon: 'img/favicon.svg',

  // The manual is served from the custom domain (GitHub Pages on
  // hitsainet/miLLM with CNAME docs-millm.hitsai.net). Do NOT revert these to
  // the onegaishimas.github.io/miLLM values — that breaks every asset link on
  // the live site. (These were fixed once on hitsainet and clobbered by the
  // repo sync; the source of truth is now here.)
  url: 'https://docs-millm.hitsai.net',
  baseUrl: '/',

  organizationName: 'hitsainet',
  projectName: 'miLLM',
  deploymentBranch: 'gh-pages',
  trailingSlash: false,

  onBrokenLinks: 'throw',
  onBrokenMarkdownLinks: 'warn',

  i18n: {
    defaultLocale: 'en',
    locales: ['en'],
  },

  presets: [
    [
      'classic',
      {
        docs: {
          routeBasePath: '/',
          sidebarPath: './sidebars.ts',
          editUrl: 'https://github.com/hitsainet/miLLM/tree/main/manual/',
        },
        blog: false,
        theme: {
          customCss: './src/css/custom.css',
        },
      } satisfies Preset.Options,
    ],
  ],

  themes: [
    [
      require.resolve('@easyops-cn/docusaurus-search-local'),
      {
        hashed: true,
        indexBlog: false,
        docsRouteBasePath: '/',
        highlightSearchTermsOnTargetPage: true,
        searchResultLimits: 10,
      },
    ],
  ],

  themeConfig: {
    navbar: {
      title: 'miLLM Manual',
      items: [
        {
          type: 'docSidebar',
          sidebarId: 'manualSidebar',
          position: 'left',
          label: 'Manual',
        },
        {
          to: '/api/overview',
          label: 'API Reference',
          position: 'left',
        },
        {
          to: '/tutorials/steering-gemma',
          label: 'Tutorials',
          position: 'left',
        },
        {
          href: 'https://github.com/hitsainet/miLLM',
          label: 'GitHub',
          position: 'right',
        },
      ],
    },
    footer: {
      style: 'dark',
      links: [
        {
          title: 'Learn',
          items: [
            {label: 'Quickstart', to: '/getting-started/quickstart'},
            {label: 'Concepts', to: '/concepts/interpretability'},
            {label: 'Tutorials', to: '/tutorials/steering-gemma'},
          ],
        },
        {
          title: 'Reference',
          items: [
            {label: 'API Reference', to: '/api/overview'},
            {label: 'Configuration', to: '/reference/configuration'},
            {label: 'Troubleshooting', to: '/troubleshooting'},
          ],
        },
        {
          title: 'Resources',
          items: [
            {label: 'GitHub', href: 'https://github.com/hitsainet/miLLM'},
            {label: 'Neuronpedia', href: 'https://neuronpedia.org'},
            {label: 'GemmaScope SAEs', href: 'https://huggingface.co/google/gemma-scope-2b-pt-res'},
          ],
        },
      ],
      copyright: `Copyright © ${new Date().getFullYear()} MCS Lab. Built with Docusaurus.`,
    },
    colorMode: {
      defaultMode: 'dark',
      disableSwitch: false,
      respectPrefersColorScheme: false,
    },
    prism: {
      theme: prismThemes.github,
      darkTheme: prismThemes.dracula,
      additionalLanguages: ['bash', 'python', 'typescript', 'nginx', 'yaml', 'json'],
    },
  } satisfies Preset.ThemeConfig,
};

export default config;
