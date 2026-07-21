import type {SidebarsConfig} from '@docusaurus/plugin-content-docs';

const sidebars: SidebarsConfig = {
  manualSidebar: [
    'intro',
    {
      type: 'category',
      label: 'Getting Started',
      items: [
        'getting-started/introduction',
        'getting-started/quickstart',
        'getting-started/installation',
        'getting-started/hardware',
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
      label: 'Concepts',
      items: [
        'concepts/interpretability',
        'concepts/steering',
        'concepts/monitoring',
        'concepts/architecture',
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
        'features/clusters',
        'features/circuits',
        'features/mcp-circuits',
      ],
    },
    {
      type: 'category',
      label: 'Tutorials',
      items: [
        'tutorials/steering-gemma',
        'tutorials/open-webui',
        'tutorials/python-scripting',
      ],
    },
    {
      type: 'category',
      label: 'API Reference',
      items: [
        'api/overview',
        'api/openai-compatible',
        'api/models',
        'api/saes',
        'api/monitoring',
        'api/profiles',
        'api/websockets',
        'api/management-api',
      ],
    },
    {
      type: 'category',
      label: 'Reference',
      items: [
        'reference/configuration',
        'reference/error-codes',
      ],
    },
    'troubleshooting',
  ],
};

export default sidebars;
