import type {SidebarsConfig} from '@docusaurus/plugin-content-docs';

const sidebars: SidebarsConfig = {
  manualSidebar: [
    'intro',
    {
      type: 'category',
      label: 'Getting Started',
      items: [
        'getting-started/introduction',
        'getting-started/installation',
        'getting-started/dashboard',
        {
          type: 'category',
          label: 'Installation Guides',
          items: [
            'getting-started/install-guide-compose',
            'getting-started/install-guide-k8s',
          ],
        },
      ],
    },
    {
      type: 'category',
      label: 'Core Features',
      items: [
        'features/model-management',
        'features/sae-management',
        'features/feature-steering',
        'features/probe-monitoring',
        'features/profiles',
      ],
    },
    {
      type: 'category',
      label: 'API Reference',
      items: [
        'api/openai-compatible',
        'api/management-api',
      ],
    },
    'troubleshooting',
  ],
};

export default sidebars;
