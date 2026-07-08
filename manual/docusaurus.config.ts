import {themes as prismThemes} from 'prism-react-renderer';
import type {Config} from '@docusaurus/types';
import type * as Preset from '@docusaurus/preset-classic';

const config: Config = {
  title: 'miLLM Manual',
  tagline: 'Mechanistic Interpretability LLM Server — Feature Steering & Real-Time Probing',
  favicon: 'img/favicon.svg',

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
          title: 'Manual',
          items: [
            {label: 'Getting Started', to: '/getting-started/introduction'},
            {label: 'Core Features', to: '/features/model-management'},
            {label: 'Troubleshooting', to: '/troubleshooting'},
          ],
        },
        {
          title: 'Resources',
          items: [
            {label: 'GitHub', href: 'https://github.com/hitsainet/miLLM'},
            {label: 'Neuronpedia', href: 'https://neuronpedia.org'},
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
