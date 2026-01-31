# miLLM Admin UI

Administrative web interface for the miLLM (Mechanistic Interpretability LLM) server.

## Features

- **Models Management**: Download, load, and manage Hugging Face models with quantization support
- **SAE Management**: Download and attach Sparse Autoencoders for feature steering
- **Feature Steering**: Adjust feature activation strengths to influence model behavior
- **Monitoring**: Real-time observation of feature activations during inference
- **Profiles**: Save and load steering configurations for quick switching
- **Dashboard**: Overview of system status and active configurations

## Tech Stack

- **React 19** with TypeScript
- **Vite** for fast development and building
- **Tailwind CSS** for styling
- **Zustand** for state management
- **React Query** for server state
- **Socket.IO** for real-time WebSocket communication
- **Vitest** for testing

## Getting Started

### Prerequisites

- Node.js 20+
- npm or yarn

### Installation

```bash
npm install
```

### Development

Start the development server:

```bash
npm run dev
```

The UI will be available at `http://localhost:3000`. The dev server proxies API requests to `http://localhost:8000` (the backend).

### Building

Build for production:

```bash
npm run build
```

### Testing

```bash
# Run tests once
npm run test

# Run tests in watch mode
npm run test:watch

# Run tests with coverage
npm run test:coverage
```

### Linting and Type Checking

```bash
# Lint
npm run lint

# Type check
npm run typecheck
```

## Project Structure

```
src/
├── components/          # React components
│   ├── common/         # Shared UI components (Button, Card, Input, etc.)
│   ├── dashboard/      # Dashboard page components
│   ├── layout/         # Layout components (Header, Sidebar, etc.)
│   ├── models/         # Model management components
│   ├── monitoring/     # Monitoring page components
│   ├── profiles/       # Profile management components
│   ├── sae/            # SAE management components
│   └── steering/       # Steering page components
├── hooks/              # Custom React hooks
├── pages/              # Route page components
├── services/           # API and WebSocket clients
├── stores/             # Zustand state stores
├── types/              # TypeScript type definitions
└── utils/              # Utility functions
```

## API Integration

The UI connects to the miLLM backend via:

- **REST API** (`/api/*`): CRUD operations for models, SAEs, profiles, steering
- **WebSocket** (`/socket.io`): Real-time events for progress, monitoring, and system metrics

## Environment Configuration

The development server is configured to proxy requests:

- `/api` → `http://localhost:8000` (REST API)
- `/socket.io` → `http://localhost:8000` (WebSocket)

For production, the UI should be served alongside the backend or configured with appropriate CORS settings.

## License

See the main project LICENSE file.
