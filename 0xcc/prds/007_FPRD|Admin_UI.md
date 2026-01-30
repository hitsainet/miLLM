# Feature PRD: Admin UI

## miLLM Feature 7

**Document Version:** 1.0
**Created:** January 30, 2026
**Status:** Draft
**Feature Priority:** Core (MVP)

---

## 1. Overview

### 1.1 Feature Summary

The Admin UI provides a web-based dashboard for managing and monitoring the miLLM server. It offers a real-time interface for model loading, SAE attachment, feature steering, activation monitoring, and profile management - consolidating all management operations into a single, intuitive interface.

### 1.2 Problem Statement

Managing an interpretability-focused LLM server through CLI or raw API calls is cumbersome and error-prone. Users need visual feedback when adjusting feature steering values, monitoring activations, and managing server state.

### 1.3 Solution

A React-based single-page application that:
- Connects to the miLLM Management API and WebSocket server
- Provides real-time status updates via Socket.IO
- Offers intuitive controls for all management operations
- Displays activation visualizations for monitoring
- Works as a standalone dashboard (no auth in v1)

### 1.4 Target Users

- **ML Researchers**: Exploring feature interpretability and steering effects
- **AI Safety Engineers**: Monitoring and adjusting model behavior
- **System Administrators**: Managing server resources and model lifecycle
- **Developers**: Testing integrations and debugging

---

## 2. User Stories

### 2.1 Dashboard Overview

**US-7.1:** As a user, I want to see the server status at a glance so I can understand the current system state.

**Acceptance Criteria:**
- Dashboard shows server health status (healthy/degraded/error)
- Shows loaded model name and status
- Shows attached SAE name and status
- Shows active profile name (if any)
- Shows steering active indicator (on/off with feature count)
- Shows monitoring active indicator
- All status updates in real-time via WebSocket

### 2.2 Model Management

**US-7.2:** As a user, I want to load and unload models through the UI so I can manage the inference server.

**Acceptance Criteria:**
- Form to enter model ID (HuggingFace path)
- Optional device selection (auto/cuda/cpu)
- Optional dtype selection (auto/float16/bfloat16/float32)
- Load button with loading spinner during operation
- Unload button (disabled when no model loaded)
- Progress indication for model loading
- Error display if loading fails
- Success notification on completion

### 2.3 SAE Management

**US-7.3:** As a user, I want to download, attach, and detach SAEs through the UI so I can enable interpretability features.

**Acceptance Criteria:**
- List of downloaded SAEs with metadata (name, layer, features)
- Download form with repo_id and optional filename
- Download progress indicator
- Attach button for each downloaded SAE
- Detach button when SAE is attached
- Delete button for cached SAEs
- Error handling for download failures
- Status shows currently attached SAE

### 2.4 Feature Steering

**US-7.4:** As a user, I want to adjust feature steering values through sliders so I can modify model behavior in real-time.

**Acceptance Criteria:**
- Search/filter for features by index or label (if available)
- Slider for each selected feature (-10.0 to +10.0)
- Numeric input for precise value entry
- Batch add multiple features at once
- Clear individual feature button
- Clear all steering button
- Enable/disable steering toggle
- Real-time update as sliders move (debounced)
- Visual indication of non-zero steering values

### 2.5 Activation Monitoring

**US-7.5:** As a user, I want to see real-time feature activations so I can understand model behavior.

**Acceptance Criteria:**
- Toggle to enable/disable monitoring
- Configuration for top_k features to display (1-50)
- Live activation display showing feature index and strength
- History view of recent activations
- Clear history button
- Statistics panel (mean, max, total activations per feature)
- Visual bar chart of top activations
- Pause/resume live updates

### 2.6 Profile Management

**US-7.6:** As a user, I want to save and load steering profiles so I can reuse configurations.

**Acceptance Criteria:**
- List of saved profiles with name and description
- Create new profile button with name/description form
- Save current steering to profile
- Load profile (applies steering)
- Activate/deactivate profile toggle
- Delete profile button with confirmation
- Export profile to JSON file
- Import profile from JSON file
- Visual indication of active profile

### 2.7 Settings & Configuration

**US-7.7:** As a user, I want to configure UI preferences so the dashboard works for my workflow.

**Acceptance Criteria:**
- Theme toggle (light/dark mode)
- WebSocket reconnection settings
- Monitoring throttle settings
- Local storage persistence of preferences
- Server connection status indicator
- Manual reconnect button

---

## 3. Functional Requirements

### 3.1 Navigation & Layout

| ID | Requirement | Priority |
|----|-------------|----------|
| UI-N1 | Sidebar navigation with icons and labels | Must |
| UI-N2 | Collapsible sidebar for more workspace | Should |
| UI-N3 | Breadcrumb navigation within sections | Could |
| UI-N4 | Responsive layout for different screen sizes | Should |
| UI-N5 | Persistent header with server status | Must |

### 3.2 Dashboard Page

| ID | Requirement | Priority |
|----|-------------|----------|
| UI-D1 | Status cards for model, SAE, steering, monitoring | Must |
| UI-D2 | Quick action buttons for common operations | Should |
| UI-D3 | Recent activity log from WebSocket events | Could |
| UI-D4 | System resource usage (if available from API) | Could |

### 3.3 Models Page

| ID | Requirement | Priority |
|----|-------------|----------|
| UI-M1 | Model load form with all parameters | Must |
| UI-M2 | Loading progress indicator | Must |
| UI-M3 | Current model info display | Must |
| UI-M4 | Unload button with confirmation | Must |
| UI-M5 | Model loading history (session only) | Could |

### 3.4 SAE Page

| ID | Requirement | Priority |
|----|-------------|----------|
| UI-S1 | Downloaded SAE list with metadata | Must |
| UI-S2 | Download form with progress | Must |
| UI-S3 | Attach/detach controls | Must |
| UI-S4 | Delete cached SAE with confirmation | Must |
| UI-S5 | SAE info panel when attached | Should |

### 3.5 Steering Page

| ID | Requirement | Priority |
|----|-------------|----------|
| UI-ST1 | Feature search by index | Must |
| UI-ST2 | Slider controls with numeric input | Must |
| UI-ST3 | Add/remove feature controls | Must |
| UI-ST4 | Clear all button | Must |
| UI-ST5 | Enable/disable toggle | Must |
| UI-ST6 | Batch feature entry (comma-separated) | Should |
| UI-ST7 | Feature labels display (if available) | Could |

### 3.6 Monitoring Page

| ID | Requirement | Priority |
|----|-------------|----------|
| UI-MO1 | Enable/disable toggle | Must |
| UI-MO2 | Top-K configuration | Must |
| UI-MO3 | Live activation display | Must |
| UI-MO4 | Activation history list | Must |
| UI-MO5 | Clear history button | Must |
| UI-MO6 | Statistics panel | Should |
| UI-MO7 | Visual bar chart | Should |
| UI-MO8 | Pause/resume live feed | Should |

### 3.7 Profiles Page

| ID | Requirement | Priority |
|----|-------------|----------|
| UI-P1 | Profile list with status | Must |
| UI-P2 | Create profile form | Must |
| UI-P3 | Update/delete operations | Must |
| UI-P4 | Activate/deactivate toggle | Must |
| UI-P5 | Export to file | Must |
| UI-P6 | Import from file | Must |
| UI-P7 | Profile comparison view | Could |

### 3.8 Settings Page

| ID | Requirement | Priority |
|----|-------------|----------|
| UI-SET1 | Theme toggle | Should |
| UI-SET2 | Connection status display | Must |
| UI-SET3 | Manual reconnect button | Must |
| UI-SET4 | Preference persistence | Should |

---

## 4. Non-Functional Requirements

### 4.1 Performance

| ID | Requirement |
|----|-------------|
| UI-NF1 | Initial page load < 3 seconds |
| UI-NF2 | WebSocket updates render < 100ms |
| UI-NF3 | Slider interactions feel instant (< 50ms feedback) |
| UI-NF4 | Handle 100+ activation updates per second |

### 4.2 Usability

| ID | Requirement |
|----|-------------|
| UI-NF5 | Consistent visual design across all pages |
| UI-NF6 | Clear loading states for all async operations |
| UI-NF7 | Informative error messages with recovery hints |
| UI-NF8 | Keyboard navigation support |

### 4.3 Reliability

| ID | Requirement |
|----|-------------|
| UI-NF9 | Automatic WebSocket reconnection |
| UI-NF10 | Graceful degradation when server unavailable |
| UI-NF11 | Local state preservation on page refresh |

### 4.4 Compatibility

| ID | Requirement |
|----|-------------|
| UI-NF12 | Chrome, Firefox, Safari, Edge (latest 2 versions) |
| UI-NF13 | Minimum viewport: 1024x768 |
| UI-NF14 | Tablet-friendly (1024px+ width) |

---

## 5. Technical Requirements

### 5.1 Technology Stack (per ADR)

- **Framework**: React 18 with TypeScript
- **Styling**: Tailwind CSS
- **State Management**: Zustand
- **HTTP Client**: Axios or fetch with React Query
- **WebSocket**: socket.io-client
- **Build Tool**: Vite
- **Icons**: Heroicons or Lucide React

### 5.2 API Integration

The Admin UI connects to:

**REST API (Management)**
- `GET/POST /api/models` - Model management
- `GET/POST/DELETE /api/sae` - SAE management
- `GET/POST/DELETE /api/steering` - Steering operations
- `GET/POST/DELETE /api/monitoring` - Monitoring config
- `GET/POST/PUT/DELETE /api/profiles` - Profile CRUD

**WebSocket Events (Socket.IO)**
- `model:status` - Model loading/unloading status
- `sae:status` - SAE attach/detach status
- `steering:update` - Steering value changes
- `monitoring:activation` - Live activation data
- `server:status` - Server health updates

### 5.3 File Structure

```
admin-ui/
├── src/
│   ├── components/
│   │   ├── common/         # Shared UI components
│   │   ├── dashboard/      # Dashboard-specific components
│   │   ├── models/         # Model management components
│   │   ├── sae/            # SAE management components
│   │   ├── steering/       # Steering control components
│   │   ├── monitoring/     # Monitoring display components
│   │   └── profiles/       # Profile management components
│   ├── hooks/              # Custom React hooks
│   ├── stores/             # Zustand stores
│   ├── services/           # API and WebSocket services
│   ├── types/              # TypeScript type definitions
│   ├── utils/              # Utility functions
│   ├── pages/              # Page-level components
│   ├── App.tsx
│   └── main.tsx
├── public/
├── package.json
├── tailwind.config.js
├── tsconfig.json
└── vite.config.ts
```

---

## 6. UI/UX Specifications

### 6.1 Color Palette (Dark Theme Default)

```
Background:      #0f172a (slate-900)
Surface:         #1e293b (slate-800)
Border:          #334155 (slate-700)
Text Primary:    #f8fafc (slate-50)
Text Secondary:  #94a3b8 (slate-400)
Accent:          #3b82f6 (blue-500)
Success:         #22c55e (green-500)
Warning:         #f59e0b (amber-500)
Error:           #ef4444 (red-500)
```

### 6.2 Component Specifications

**Status Card**
- 200px min-width
- Icon + label + value layout
- Colored border based on status
- Hover state for interactive cards

**Slider Control**
- Full-width track
- Value label above
- Numeric input inline
- Step: 0.1

**Action Button**
- Primary: Filled blue
- Secondary: Outlined
- Danger: Filled red
- Disabled: Grayed out with no interaction

**Data Table**
- Striped rows
- Sortable headers
- Action buttons per row
- Loading skeleton

### 6.3 Layout Specifications

**Sidebar**
- Width: 240px (expanded), 64px (collapsed)
- Fixed position
- Logo at top
- Navigation items with icons

**Main Content**
- Max-width: 1440px
- Padding: 24px
- Page title + breadcrumbs at top

**Header**
- Height: 64px
- Server status badge right-aligned
- Connection indicator

---

## 7. Out of Scope (v1)

- User authentication and authorization
- Multi-user collaboration features
- Mobile-responsive design (tablets minimum)
- Internationalization (i18n)
- Accessibility (WCAG AA) - basic keyboard nav only
- Custom theming beyond light/dark
- Dashboard customization/widget arrangement
- Data export to CSV/Excel
- Comparison views between profiles
- Feature annotation/labeling interface

---

## 8. Success Metrics

| Metric | Target |
|--------|--------|
| Page load time | < 3 seconds |
| WebSocket latency | < 100ms |
| Feature completion | 100% of Must requirements |
| Browser compatibility | All listed browsers |

---

## 9. Dependencies

- **Feature 1 (Model Management)**: API endpoints for model operations
- **Feature 3 (SAE Management)**: API endpoints for SAE operations
- **Feature 4 (Feature Steering)**: API and WebSocket for steering
- **Feature 5 (Feature Monitoring)**: WebSocket for activations
- **Feature 6 (Profile Management)**: API for profile CRUD

---

## 10. Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| WebSocket disconnections | Poor UX | Auto-reconnect with exponential backoff |
| High activation frequency | Browser lag | Throttling and virtualization |
| Large feature count | Slow rendering | Virtual scrolling for lists |
| State synchronization | Stale UI | Optimistic updates + server reconciliation |

---

**Document Status:** Complete
**Next Document:** 007_FTDD|Admin_UI.md (Technical Design Document)
