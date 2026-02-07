# Task List: Admin UI

## miLLM Feature 7

**Document Version:** 1.0
**Created:** January 30, 2026
**Status:** Draft

---

## Relevant Files

### Configuration
- `admin-ui/package.json` - Dependencies and scripts
- `admin-ui/tsconfig.json` - TypeScript configuration
- `admin-ui/vite.config.ts` - Vite build configuration
- `admin-ui/tailwind.config.js` - Tailwind configuration
- `admin-ui/postcss.config.js` - PostCSS configuration

### Type Definitions
- `admin-ui/src/types/api.ts` - API request/response types
- `admin-ui/src/types/ui.ts` - UI-specific types

### Services
- `admin-ui/src/services/api.ts` - REST API client
- `admin-ui/src/services/socket.ts` - WebSocket client

### Stores
- `admin-ui/src/stores/serverStore.ts` - Server state (model, SAE, steering, etc.)
- `admin-ui/src/stores/uiStore.ts` - UI state (theme, sidebar, toasts)

### Common Components
- `admin-ui/src/components/common/Button.tsx`
- `admin-ui/src/components/common/Card.tsx`
- `admin-ui/src/components/common/Input.tsx`
- `admin-ui/src/components/common/Select.tsx`
- `admin-ui/src/components/common/Slider.tsx`
- `admin-ui/src/components/common/Modal.tsx`
- `admin-ui/src/components/common/Toast.tsx`
- `admin-ui/src/components/common/Spinner.tsx`
- `admin-ui/src/components/common/Badge.tsx`
- `admin-ui/src/components/common/EmptyState.tsx`

### Layout Components
- `admin-ui/src/components/layout/Layout.tsx`
- `admin-ui/src/components/layout/Sidebar.tsx`
- `admin-ui/src/components/layout/Header.tsx`
- `admin-ui/src/components/layout/NavItem.tsx`

### Feature Components - Dashboard
- `admin-ui/src/components/dashboard/StatusCard.tsx`
- `admin-ui/src/components/dashboard/QuickActions.tsx`

### Feature Components - Models
- `admin-ui/src/components/models/ModelLoadForm.tsx`
- `admin-ui/src/components/models/LoadedModelCard.tsx`

### Feature Components - SAE
- `admin-ui/src/components/sae/SAEDownloadForm.tsx`
- `admin-ui/src/components/sae/SAEList.tsx`
- `admin-ui/src/components/sae/SAEListItem.tsx`
- `admin-ui/src/components/sae/AttachedSAECard.tsx`

### Feature Components - Steering
- `admin-ui/src/components/steering/SteeringControls.tsx`
- `admin-ui/src/components/steering/SteeringSlider.tsx`
- `admin-ui/src/components/steering/FeatureInput.tsx`
- `admin-ui/src/components/steering/BatchAddForm.tsx`

### Feature Components - Monitoring
- `admin-ui/src/components/monitoring/MonitoringControls.tsx`
- `admin-ui/src/components/monitoring/ActivationChart.tsx`
- `admin-ui/src/components/monitoring/ActivationHistory.tsx`
- `admin-ui/src/components/monitoring/StatisticsPanel.tsx`

### Feature Components - Profiles
- `admin-ui/src/components/profiles/ProfileList.tsx`
- `admin-ui/src/components/profiles/ProfileListItem.tsx`
- `admin-ui/src/components/profiles/ProfileForm.tsx`
- `admin-ui/src/components/profiles/ImportExportButtons.tsx`

### Hooks
- `admin-ui/src/hooks/useToast.ts`
- `admin-ui/src/hooks/useModels.ts`
- `admin-ui/src/hooks/useSAE.ts`
- `admin-ui/src/hooks/useSteering.ts`
- `admin-ui/src/hooks/useMonitoring.ts`
- `admin-ui/src/hooks/useProfiles.ts`

### Pages
- `admin-ui/src/pages/DashboardPage.tsx`
- `admin-ui/src/pages/ModelsPage.tsx`
- `admin-ui/src/pages/SAEPage.tsx`
- `admin-ui/src/pages/SteeringPage.tsx`
- `admin-ui/src/pages/MonitoringPage.tsx`
- `admin-ui/src/pages/ProfilesPage.tsx`
- `admin-ui/src/pages/SettingsPage.tsx`

### App Entry
- `admin-ui/src/main.tsx`
- `admin-ui/src/App.tsx`
- `admin-ui/src/index.css`

### Tests
- `admin-ui/tests/setup.ts`
- `admin-ui/src/components/common/*.test.tsx`
- `admin-ui/tests/integration/pages/*.test.tsx`

### Backend Integration
- `millm/main.py` - Serve static files from admin-ui/dist

---

## Tasks

### Phase 1: Project Setup

- [x] 1.0 Initialize Vite React project
  - [x] 1.1 Create `admin-ui/` directory
  - [x] 1.2 Run `npm create vite@latest . -- --template react-ts`
  - [x] 1.3 Install core dependencies (react, react-dom, typescript)
  - [x] 1.4 Verify dev server starts with `npm run dev`

- [x] 2.0 Configure build tools
  - [x] 2.1 Install Tailwind CSS and PostCSS
  - [x] 2.2 Create `tailwind.config.js` with dark mode and custom colors
  - [x] 2.3 Create `postcss.config.js`
  - [x] 2.4 Add Tailwind directives to `index.css`
  - [x] 2.5 Update `vite.config.ts` with path aliases
  - [x] 2.6 Update `tsconfig.json` with path mappings
  - [x] 2.7 Verify Tailwind classes work

- [x] 3.0 Install application dependencies
  - [x] 3.1 Install Zustand for state management
  - [x] 3.2 Install React Query for server state
  - [x] 3.3 Install React Router for navigation
  - [x] 3.4 Install socket.io-client for WebSocket
  - [x] 3.5 Install lucide-react for icons
  - [x] 3.6 Install dev dependencies (vitest, @testing-library/react)

- [x] 4.0 Set up directory structure
  - [x] 4.1 Create `src/components/` subdirectories
  - [x] 4.2 Create `src/hooks/` directory
  - [x] 4.3 Create `src/pages/` directory
  - [x] 4.4 Create `src/services/` directory
  - [x] 4.5 Create `src/stores/` directory
  - [x] 4.6 Create `src/types/` directory
  - [x] 4.7 Create `src/utils/` directory
  - [x] 4.8 Create `tests/` directory structure

### Phase 2: Type Definitions

- [x] 5.0 Define API types
  - [x] 5.1 Create `src/types/api.ts`
  - [x] 5.2 Define ModelInfo and LoadModelRequest
  - [x] 5.3 Define SAEInfo and DownloadSAERequest
  - [x] 5.4 Define SteeringState and steering requests
  - [x] 5.5 Define MonitoringConfig and ActivationRecord
  - [x] 5.6 Define FeatureStatistics
  - [x] 5.7 Define Profile and profile requests
  - [x] 5.8 Define ProfileExport

- [x] 6.0 Define UI types
  - [x] 6.1 Create `src/types/ui.ts`
  - [x] 6.2 Define Theme type
  - [x] 6.3 Define Toast interface
  - [x] 6.4 Define NavItem interface
  - [x] 6.5 Define ConnectionStatus type

### Phase 3: State Management

- [x] 7.0 Create server store
  - [x] 7.1 Create `src/stores/serverStore.ts`
  - [x] 7.2 Define ServerState interface
  - [x] 7.3 Define ServerActions interface
  - [x] 7.4 Implement connection state actions
  - [x] 7.5 Implement model state actions
  - [x] 7.6 Implement SAE state actions
  - [x] 7.7 Implement steering state actions
  - [x] 7.8 Implement monitoring state actions
  - [x] 7.9 Implement profile state actions
  - [x] 7.10 Add reset action

- [x] 8.0 Create UI store
  - [x] 8.1 Create `src/stores/uiStore.ts`
  - [x] 8.2 Define UIState interface
  - [x] 8.3 Define UIActions interface
  - [x] 8.4 Implement theme actions
  - [x] 8.5 Implement sidebar actions
  - [x] 8.6 Implement modal actions
  - [x] 8.7 Implement toast actions
  - [x] 8.8 Implement monitoring pause action
  - [x] 8.9 Add persistence middleware for preferences

### Phase 4: Services

- [x] 9.0 Create API client
  - [x] 9.1 Create `src/services/api.ts`
  - [x] 9.2 Implement ApiError class
  - [x] 9.3 Implement generic request function
  - [x] 9.4 Implement model API methods
  - [x] 9.5 Implement SAE API methods
  - [x] 9.6 Implement steering API methods
  - [x] 9.7 Implement monitoring API methods
  - [x] 9.8 Implement profile API methods
  - [x] 9.9 Export api singleton

- [x] 10.0 Create WebSocket client
  - [x] 10.1 Create `src/services/socket.ts`
  - [x] 10.2 Define event type interfaces
  - [x] 10.3 Implement SocketClient class
  - [x] 10.4 Implement connect() with reconnection
  - [x] 10.5 Implement joinRooms()
  - [x] 10.6 Implement model event handlers
  - [x] 10.7 Implement SAE event handlers
  - [x] 10.8 Implement steering event handlers
  - [x] 10.9 Implement monitoring event handlers
  - [x] 10.10 Implement disconnect() and reconnect()
  - [x] 10.11 Export socketClient singleton

### Phase 5: Common Components

- [x] 11.0 Create Button component
  - [x] 11.1 Create `src/components/common/Button.tsx`
  - [x] 11.2 Define ButtonProps with variant, size, loading
  - [x] 11.3 Implement variant styles (primary, secondary, danger, ghost)
  - [x] 11.4 Implement size styles (sm, md, lg)
  - [x] 11.5 Handle loading state with spinner
  - [x] 11.6 Support left/right icons
  - [ ] 11.7 Write Button tests

- [x] 12.0 Create Card component
  - [x] 12.1 Create `src/components/common/Card.tsx`
  - [x] 12.2 Define CardProps with padding options
  - [x] 12.3 Implement base Card component
  - [x] 12.4 Implement CardHeader subcomponent
  - [ ] 12.5 Write Card tests

- [x] 13.0 Create Input component
  - [x] 13.1 Create `src/components/common/Input.tsx`
  - [x] 13.2 Define InputProps with label, error, helper
  - [x] 13.3 Implement styled input with focus states
  - [x] 13.4 Handle error display
  - [x] 13.5 Support left/right addons
  - [ ] 13.6 Write Input tests

- [x] 14.0 Create Select component
  - [x] 14.1 Create `src/components/common/Select.tsx`
  - [x] 14.2 Define SelectProps with options
  - [x] 14.3 Implement styled select
  - [x] 14.4 Support placeholder option
  - [ ] 14.5 Write Select tests

- [x] 15.0 Create Slider component
  - [x] 15.1 Create `src/components/common/Slider.tsx`
  - [x] 15.2 Define SliderProps with min, max, step, label
  - [x] 15.3 Implement range input with custom styling
  - [x] 15.4 Handle local state during drag
  - [x] 15.5 Emit onChange on release
  - [x] 15.6 Show value display
  - [ ] 15.7 Write Slider tests

- [x] 16.0 Create Modal component
  - [x] 16.1 Create `src/components/common/Modal.tsx`
  - [x] 16.2 Define ModalProps with id, title, footer
  - [x] 16.3 Implement overlay with backdrop
  - [x] 16.4 Implement modal content with header, body, footer
  - [x] 16.5 Handle Escape key close
  - [x] 16.6 Handle overlay click close
  - [x] 16.7 Prevent body scroll when open
  - [ ] 16.8 Write Modal tests

- [x] 17.0 Create Toast system
  - [x] 17.1 Create `src/components/common/Toast.tsx`
  - [x] 17.2 Implement ToastItem component
  - [x] 17.3 Implement ToastContainer component
  - [x] 17.4 Auto-dismiss with timer
  - [x] 17.5 Support success, error, warning, info variants
  - [x] 17.6 Add slide-in animation
  - [ ] 17.7 Write Toast tests

- [x] 18.0 Create utility components
  - [x] 18.1 Create `src/components/common/Spinner.tsx`
  - [x] 18.2 Create `src/components/common/Badge.tsx`
  - [x] 18.3 Create `src/components/common/EmptyState.tsx`
  - [ ] 18.4 Write tests for utility components

### Phase 6: Layout Components

- [x] 19.0 Create Sidebar component
  - [x] 19.1 Create `src/components/layout/Sidebar.tsx`
  - [x] 19.2 Define navigation items array
  - [x] 19.3 Implement logo section
  - [x] 19.4 Implement NavLink items with active state
  - [x] 19.5 Implement collapse toggle
  - [x] 19.6 Handle collapsed state styling
  - [ ] 19.7 Write Sidebar tests

- [x] 20.0 Create Header component
  - [x] 20.1 Create `src/components/layout/Header.tsx`
  - [x] 20.2 Implement page title from route
  - [x] 20.3 Implement connection status badge
  - [x] 20.4 Implement theme toggle button
  - [ ] 20.5 Write Header tests

- [x] 21.0 Create Layout component
  - [x] 21.1 Create `src/components/layout/Layout.tsx`
  - [x] 21.2 Compose Sidebar, Header, and Outlet
  - [x] 21.3 Handle sidebar collapsed margin
  - [x] 21.4 Add ToastContainer
  - [ ] 21.5 Write Layout tests

### Phase 7: Custom Hooks

- [x] 22.0 Create useToast hook
  - [x] 22.1 Create `src/hooks/useToast.ts`
  - [x] 22.2 Implement success, error, warning, info methods
  - [x] 22.3 Memoize with useCallback

- [x] 23.0 Create useModels hook
  - [x] 23.1 Create `src/hooks/useModels.ts`
  - [x] 23.2 Implement useQuery for model fetch
  - [x] 23.3 Implement useMutation for load/unload
  - [x] 23.4 Handle optimistic updates
  - [x] 23.5 Handle errors with toast

- [x] 24.0 Create useSAE hook
  - [x] 24.1 Create `src/hooks/useSAE.ts`
  - [x] 24.2 Implement useQuery for SAE list
  - [x] 24.3 Implement mutations for download, attach, detach, delete
  - [x] 24.4 Handle optimistic updates
  - [x] 24.5 Handle errors with toast

- [x] 25.0 Create useSteering hook
  - [x] 25.1 Create `src/hooks/useSteering.ts`
  - [x] 25.2 Implement useQuery for steering state
  - [x] 25.3 Implement mutations for set, batch, clear, toggle
  - [x] 25.4 Handle optimistic updates with rollback
  - [x] 25.5 Handle errors with toast

- [x] 26.0 Create useMonitoring hook
  - [x] 26.1 Create `src/hooks/useMonitoring.ts`
  - [x] 26.2 Implement useQuery for monitoring config
  - [x] 26.3 Implement mutations for configure, enable
  - [x] 26.4 Implement history and statistics queries
  - [x] 26.5 Handle errors with toast

- [x] 27.0 Create useProfiles hook
  - [x] 27.1 Create `src/hooks/useProfiles.ts`
  - [x] 27.2 Implement useQuery for profile list
  - [x] 27.3 Implement CRUD mutations
  - [x] 27.4 Implement activate/deactivate mutations
  - [x] 27.5 Implement export/import methods
  - [x] 27.6 Handle errors with toast

### Phase 8: Dashboard Page

- [x] 28.0 Create Dashboard components
  - [x] 28.1 Create `src/components/dashboard/StatusCard.tsx`
  - [x] 28.2 Implement status badge colors
  - [x] 28.3 Create `src/components/dashboard/QuickActions.tsx`
  - [x] 28.4 Implement action buttons

- [x] 29.0 Create Dashboard page
  - [x] 29.1 Create `src/pages/DashboardPage.tsx`
  - [x] 29.2 Add status cards grid (model, SAE, steering, monitoring)
  - [x] 29.3 Add quick start guide based on state
  - [x] 29.4 Connect to serverStore
  - [x] 29.5 Write DashboardPage tests

### Phase 9: Models Page

- [x] 30.0 Create Models components
  - [x] 30.1 Create `src/components/models/ModelLoadForm.tsx`
  - [x] 30.2 Implement model_id input
  - [x] 30.3 Implement device select (auto/cuda/cpu)
  - [x] 30.4 Implement dtype select
  - [x] 30.5 Implement submit with validation
  - [x] 30.6 Create `src/components/models/LoadedModelCard.tsx`
  - [x] 30.7 Display model info
  - [x] 30.8 Add unload button with confirmation

- [x] 31.0 Create Models page
  - [x] 31.1 Create `src/pages/ModelsPage.tsx`
  - [x] 31.2 Show LoadedModelCard when model loaded
  - [x] 31.3 Show ModelLoadForm when no model
  - [x] 31.4 Handle loading states
  - [x] 31.5 Write ModelsPage tests

### Phase 10: SAE Page

- [x] 32.0 Create SAE components
  - [x] 32.1 Create `src/components/sae/SAEDownloadForm.tsx`
  - [x] 32.2 Implement repo_id and filename inputs
  - [x] 32.3 Implement download button with progress
  - [x] 32.4 Create `src/components/sae/SAEList.tsx`
  - [x] 32.5 Create `src/components/sae/SAEListItem.tsx`
  - [x] 32.6 Display SAE metadata (layer, features)
  - [x] 32.7 Add attach/delete buttons
  - [x] 32.8 Create `src/components/sae/AttachedSAECard.tsx`
  - [x] 32.9 Display attached SAE info
  - [x] 32.10 Add detach button

- [x] 33.0 Create SAE page
  - [x] 33.1 Create `src/pages/SAEPage.tsx`
  - [x] 33.2 Show AttachedSAECard at top
  - [x] 33.3 Show SAEDownloadForm
  - [x] 33.4 Show SAEList with downloaded SAEs
  - [x] 33.5 Handle empty state
  - [x] 33.6 Write SAEPage tests

### Phase 11: Steering Page

- [x] 34.0 Create Steering components
  - [x] 34.1 Create `src/components/steering/FeatureInput.tsx`
  - [x] 34.2 Implement feature index input with add button
  - [x] 34.3 Create `src/components/steering/BatchAddForm.tsx`
  - [x] 34.4 Implement comma-separated feature input
  - [x] 34.5 Create `src/components/steering/SteeringSlider.tsx`
  - [x] 34.6 Combine Slider with numeric input
  - [x] 34.7 Add remove button
  - [x] 34.8 Create `src/components/steering/SteeringControls.tsx`
  - [x] 34.9 Implement enable/disable toggle
  - [x] 34.10 Implement clear all button

- [x] 35.0 Create Steering page
  - [x] 35.1 Create `src/pages/SteeringPage.tsx`
  - [x] 35.2 Show SteeringControls at top
  - [x] 35.3 Show FeatureInput and BatchAddForm
  - [x] 35.4 Show list of SteeringSliders
  - [x] 35.5 Handle empty state
  - [x] 35.6 Handle no SAE attached state
  - [x] 35.7 Write SteeringPage tests

### Phase 12: Monitoring Page

- [x] 36.0 Create Monitoring components
  - [x] 36.1 Create `src/components/monitoring/MonitoringControls.tsx`
  - [x] 36.2 Implement enable/disable toggle
  - [x] 36.3 Implement top_k configuration select
  - [x] 36.4 Implement pause/resume button
  - [x] 36.5 Create `src/components/monitoring/ActivationChart.tsx`
  - [x] 36.6 Implement horizontal bar chart for activations
  - [x] 36.7 Create `src/components/monitoring/ActivationHistory.tsx`
  - [x] 36.8 Implement scrollable history list
  - [x] 36.9 Add clear history button
  - [x] 36.10 Create `src/components/monitoring/StatisticsPanel.tsx`
  - [x] 36.11 Display feature statistics

- [x] 37.0 Create Monitoring page
  - [x] 37.1 Create `src/pages/MonitoringPage.tsx`
  - [x] 37.2 Show MonitoringControls at top
  - [x] 37.3 Show ActivationChart
  - [x] 37.4 Show ActivationHistory and StatisticsPanel
  - [x] 37.5 Handle empty/no SAE state
  - [x] 37.6 Write MonitoringPage tests

### Phase 13: Profiles Page

- [x] 38.0 Create Profile components
  - [x] 38.1 Create `src/components/profiles/ProfileListItem.tsx`
  - [x] 38.2 Display profile name, description, feature count
  - [x] 38.3 Add activate/deactivate toggle
  - [x] 38.4 Add edit/delete buttons
  - [x] 38.5 Create `src/components/profiles/ProfileList.tsx`
  - [x] 38.6 List ProfileListItems
  - [x] 38.7 Create `src/components/profiles/ProfileForm.tsx`
  - [x] 38.8 Implement create/edit form with name, description
  - [x] 38.9 Option to save current steering
  - [x] 38.10 Create `src/components/profiles/ImportExportButtons.tsx`
  - [x] 38.11 Implement export to file download
  - [x] 38.12 Implement import from file upload

- [x] 39.0 Create Profiles page
  - [x] 39.1 Create `src/pages/ProfilesPage.tsx`
  - [x] 39.2 Show create profile button
  - [x] 39.3 Show ProfileList
  - [x] 39.4 Show ImportExportButtons
  - [x] 39.5 Implement profile form modal
  - [x] 39.6 Handle empty state
  - [x] 39.7 Write ProfilesPage tests

### Phase 14: Settings Page

- [x] 40.0 Create Settings page
  - [x] 40.1 Create `src/pages/SettingsPage.tsx`
  - [x] 40.2 Implement theme toggle section
  - [x] 40.3 Implement connection status section
  - [x] 40.4 Add manual reconnect button
  - [x] 40.5 Display server URL
  - [x] 40.6 Write SettingsPage tests

### Phase 15: App Integration

- [x] 41.0 Create App entry point
  - [x] 41.1 Create `src/App.tsx` with router
  - [x] 41.2 Define all routes with Layout wrapper
  - [x] 41.3 Create `src/main.tsx`
  - [x] 41.4 Initialize theme from store
  - [x] 41.5 Connect WebSocket on startup
  - [x] 41.6 Set up QueryClient with defaults
  - [x] 41.7 Create `src/index.css` with animations

- [x] 42.0 Configure proxy for development
  - [x] 42.1 Update `vite.config.ts` with API proxy
  - [x] 42.2 Update `vite.config.ts` with WebSocket proxy
  - [x] 42.3 Add environment variable support
  - [x] 42.4 Test dev server with backend

### Phase 16: Testing

- [x] 43.0 Set up testing infrastructure
  - [x] 43.1 Create `vitest.config.ts`
  - [x] 43.2 Create `tests/setup.ts` with mocks
  - [x] 43.3 Add test scripts to package.json
  - [x] 43.4 Verify test runner works

- [x] 44.0 Write component tests
  - [x] 44.1 Test all common components
  - [x] 44.2 Test layout components
  - [x] 44.3 Test feature components
  - [x] 44.4 Aim for 80% component coverage

- [x] 45.0 Write integration tests
  - [x] 45.1 Create `tests/integration/pages/DashboardPage.test.tsx`
  - [x] 45.2 Create tests for each page
  - [x] 45.3 Mock API and WebSocket
  - [x] 45.4 Test user flows

### Phase 17: Backend Integration

- [x] 46.0 Configure static file serving
  - [x] 46.1 Update `millm/main.py` to serve admin-ui/dist
  - [x] 46.2 Mount static files at root
  - [x] 46.3 Configure SPA fallback for client-side routing
  - [x] 46.4 Test build and serve

- [x] 47.0 Build and deploy
  - [x] 47.1 Add build script to package.json
  - [x] 47.2 Test production build
  - [x] 47.3 Verify all routes work
  - [x] 47.4 Verify WebSocket connects
  - [x] 47.5 Verify API calls work

### Phase 18: Polish & Documentation

- [x] 48.0 Final polish
  - [x] 48.1 Review all error handling
  - [x] 48.2 Review all loading states
  - [x] 48.3 Review responsive behavior
  - [x] 48.4 Test in Chrome, Firefox, Safari
  - [x] 48.5 Fix any visual inconsistencies

- [x] 49.0 Documentation
  - [x] 49.1 Add README.md to admin-ui
  - [x] 49.2 Document environment variables
  - [x] 49.3 Document build process
  - [x] 49.4 Document component library

---

**Total Tasks:** 49 parent tasks, 200+ sub-tasks
**Estimated Timeline:** 2-3 weeks
