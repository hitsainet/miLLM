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

- [ ] 2.0 Configure build tools
  - [ ] 2.1 Install Tailwind CSS and PostCSS
  - [ ] 2.2 Create `tailwind.config.js` with dark mode and custom colors
  - [ ] 2.3 Create `postcss.config.js`
  - [ ] 2.4 Add Tailwind directives to `index.css`
  - [ ] 2.5 Update `vite.config.ts` with path aliases
  - [ ] 2.6 Update `tsconfig.json` with path mappings
  - [ ] 2.7 Verify Tailwind classes work

- [ ] 3.0 Install application dependencies
  - [ ] 3.1 Install Zustand for state management
  - [ ] 3.2 Install React Query for server state
  - [ ] 3.3 Install React Router for navigation
  - [ ] 3.4 Install socket.io-client for WebSocket
  - [ ] 3.5 Install @heroicons/react for icons
  - [ ] 3.6 Install dev dependencies (vitest, @testing-library/react)

- [ ] 4.0 Set up directory structure
  - [ ] 4.1 Create `src/components/` subdirectories
  - [ ] 4.2 Create `src/hooks/` directory
  - [ ] 4.3 Create `src/pages/` directory
  - [ ] 4.4 Create `src/services/` directory
  - [ ] 4.5 Create `src/stores/` directory
  - [ ] 4.6 Create `src/types/` directory
  - [ ] 4.7 Create `src/utils/` directory
  - [ ] 4.8 Create `tests/` directory structure

### Phase 2: Type Definitions

- [ ] 5.0 Define API types
  - [ ] 5.1 Create `src/types/api.ts`
  - [ ] 5.2 Define ModelInfo and LoadModelRequest
  - [ ] 5.3 Define SAEInfo and DownloadSAERequest
  - [ ] 5.4 Define SteeringState and steering requests
  - [ ] 5.5 Define MonitoringConfig and ActivationRecord
  - [ ] 5.6 Define FeatureStatistics
  - [ ] 5.7 Define Profile and profile requests
  - [ ] 5.8 Define ProfileExport

- [ ] 6.0 Define UI types
  - [ ] 6.1 Create `src/types/ui.ts`
  - [ ] 6.2 Define Theme type
  - [ ] 6.3 Define Toast interface
  - [ ] 6.4 Define NavItem interface
  - [ ] 6.5 Define ConnectionStatus type

### Phase 3: State Management

- [ ] 7.0 Create server store
  - [ ] 7.1 Create `src/stores/serverStore.ts`
  - [ ] 7.2 Define ServerState interface
  - [ ] 7.3 Define ServerActions interface
  - [ ] 7.4 Implement connection state actions
  - [ ] 7.5 Implement model state actions
  - [ ] 7.6 Implement SAE state actions
  - [ ] 7.7 Implement steering state actions
  - [ ] 7.8 Implement monitoring state actions
  - [ ] 7.9 Implement profile state actions
  - [ ] 7.10 Add reset action

- [ ] 8.0 Create UI store
  - [ ] 8.1 Create `src/stores/uiStore.ts`
  - [ ] 8.2 Define UIState interface
  - [ ] 8.3 Define UIActions interface
  - [ ] 8.4 Implement theme actions
  - [ ] 8.5 Implement sidebar actions
  - [ ] 8.6 Implement modal actions
  - [ ] 8.7 Implement toast actions
  - [ ] 8.8 Implement monitoring pause action
  - [ ] 8.9 Add persistence middleware for preferences

### Phase 4: Services

- [ ] 9.0 Create API client
  - [ ] 9.1 Create `src/services/api.ts`
  - [ ] 9.2 Implement ApiError class
  - [ ] 9.3 Implement generic request function
  - [ ] 9.4 Implement model API methods
  - [ ] 9.5 Implement SAE API methods
  - [ ] 9.6 Implement steering API methods
  - [ ] 9.7 Implement monitoring API methods
  - [ ] 9.8 Implement profile API methods
  - [ ] 9.9 Export api singleton

- [ ] 10.0 Create WebSocket client
  - [ ] 10.1 Create `src/services/socket.ts`
  - [ ] 10.2 Define event type interfaces
  - [ ] 10.3 Implement SocketClient class
  - [ ] 10.4 Implement connect() with reconnection
  - [ ] 10.5 Implement joinRooms()
  - [ ] 10.6 Implement model event handlers
  - [ ] 10.7 Implement SAE event handlers
  - [ ] 10.8 Implement steering event handlers
  - [ ] 10.9 Implement monitoring event handlers
  - [ ] 10.10 Implement disconnect() and reconnect()
  - [ ] 10.11 Export socketClient singleton

### Phase 5: Common Components

- [ ] 11.0 Create Button component
  - [ ] 11.1 Create `src/components/common/Button.tsx`
  - [ ] 11.2 Define ButtonProps with variant, size, loading
  - [ ] 11.3 Implement variant styles (primary, secondary, danger, ghost)
  - [ ] 11.4 Implement size styles (sm, md, lg)
  - [ ] 11.5 Handle loading state with spinner
  - [ ] 11.6 Support left/right icons
  - [ ] 11.7 Write Button tests

- [ ] 12.0 Create Card component
  - [ ] 12.1 Create `src/components/common/Card.tsx`
  - [ ] 12.2 Define CardProps with padding options
  - [ ] 12.3 Implement base Card component
  - [ ] 12.4 Implement CardHeader subcomponent
  - [ ] 12.5 Write Card tests

- [ ] 13.0 Create Input component
  - [ ] 13.1 Create `src/components/common/Input.tsx`
  - [ ] 13.2 Define InputProps with label, error, helper
  - [ ] 13.3 Implement styled input with focus states
  - [ ] 13.4 Handle error display
  - [ ] 13.5 Support left/right addons
  - [ ] 13.6 Write Input tests

- [ ] 14.0 Create Select component
  - [ ] 14.1 Create `src/components/common/Select.tsx`
  - [ ] 14.2 Define SelectProps with options
  - [ ] 14.3 Implement styled select
  - [ ] 14.4 Support placeholder option
  - [ ] 14.5 Write Select tests

- [ ] 15.0 Create Slider component
  - [ ] 15.1 Create `src/components/common/Slider.tsx`
  - [ ] 15.2 Define SliderProps with min, max, step, label
  - [ ] 15.3 Implement range input with custom styling
  - [ ] 15.4 Handle local state during drag
  - [ ] 15.5 Emit onChange on release
  - [ ] 15.6 Show value display
  - [ ] 15.7 Write Slider tests

- [ ] 16.0 Create Modal component
  - [ ] 16.1 Create `src/components/common/Modal.tsx`
  - [ ] 16.2 Define ModalProps with id, title, footer
  - [ ] 16.3 Implement overlay with backdrop
  - [ ] 16.4 Implement modal content with header, body, footer
  - [ ] 16.5 Handle Escape key close
  - [ ] 16.6 Handle overlay click close
  - [ ] 16.7 Prevent body scroll when open
  - [ ] 16.8 Write Modal tests

- [ ] 17.0 Create Toast system
  - [ ] 17.1 Create `src/components/common/Toast.tsx`
  - [ ] 17.2 Implement ToastItem component
  - [ ] 17.3 Implement ToastContainer component
  - [ ] 17.4 Auto-dismiss with timer
  - [ ] 17.5 Support success, error, warning, info variants
  - [ ] 17.6 Add slide-in animation
  - [ ] 17.7 Write Toast tests

- [ ] 18.0 Create utility components
  - [ ] 18.1 Create `src/components/common/Spinner.tsx`
  - [ ] 18.2 Create `src/components/common/Badge.tsx`
  - [ ] 18.3 Create `src/components/common/EmptyState.tsx`
  - [ ] 18.4 Write tests for utility components

### Phase 6: Layout Components

- [ ] 19.0 Create Sidebar component
  - [ ] 19.1 Create `src/components/layout/Sidebar.tsx`
  - [ ] 19.2 Define navigation items array
  - [ ] 19.3 Implement logo section
  - [ ] 19.4 Implement NavLink items with active state
  - [ ] 19.5 Implement collapse toggle
  - [ ] 19.6 Handle collapsed state styling
  - [ ] 19.7 Write Sidebar tests

- [ ] 20.0 Create Header component
  - [ ] 20.1 Create `src/components/layout/Header.tsx`
  - [ ] 20.2 Implement page title from route
  - [ ] 20.3 Implement connection status badge
  - [ ] 20.4 Implement theme toggle button
  - [ ] 20.5 Write Header tests

- [ ] 21.0 Create Layout component
  - [ ] 21.1 Create `src/components/layout/Layout.tsx`
  - [ ] 21.2 Compose Sidebar, Header, and Outlet
  - [ ] 21.3 Handle sidebar collapsed margin
  - [ ] 21.4 Add ToastContainer
  - [ ] 21.5 Write Layout tests

### Phase 7: Custom Hooks

- [ ] 22.0 Create useToast hook
  - [ ] 22.1 Create `src/hooks/useToast.ts`
  - [ ] 22.2 Implement success, error, warning, info methods
  - [ ] 22.3 Memoize with useCallback

- [ ] 23.0 Create useModels hook
  - [ ] 23.1 Create `src/hooks/useModels.ts`
  - [ ] 23.2 Implement useQuery for model fetch
  - [ ] 23.3 Implement useMutation for load/unload
  - [ ] 23.4 Handle optimistic updates
  - [ ] 23.5 Handle errors with toast

- [ ] 24.0 Create useSAE hook
  - [ ] 24.1 Create `src/hooks/useSAE.ts`
  - [ ] 24.2 Implement useQuery for SAE list
  - [ ] 24.3 Implement mutations for download, attach, detach, delete
  - [ ] 24.4 Handle optimistic updates
  - [ ] 24.5 Handle errors with toast

- [ ] 25.0 Create useSteering hook
  - [ ] 25.1 Create `src/hooks/useSteering.ts`
  - [ ] 25.2 Implement useQuery for steering state
  - [ ] 25.3 Implement mutations for set, batch, clear, toggle
  - [ ] 25.4 Handle optimistic updates with rollback
  - [ ] 25.5 Handle errors with toast

- [ ] 26.0 Create useMonitoring hook
  - [ ] 26.1 Create `src/hooks/useMonitoring.ts`
  - [ ] 26.2 Implement useQuery for monitoring config
  - [ ] 26.3 Implement mutations for configure, enable
  - [ ] 26.4 Implement history and statistics queries
  - [ ] 26.5 Handle errors with toast

- [ ] 27.0 Create useProfiles hook
  - [ ] 27.1 Create `src/hooks/useProfiles.ts`
  - [ ] 27.2 Implement useQuery for profile list
  - [ ] 27.3 Implement CRUD mutations
  - [ ] 27.4 Implement activate/deactivate mutations
  - [ ] 27.5 Implement export/import methods
  - [ ] 27.6 Handle errors with toast

### Phase 8: Dashboard Page

- [ ] 28.0 Create Dashboard components
  - [ ] 28.1 Create `src/components/dashboard/StatusCard.tsx`
  - [ ] 28.2 Implement status badge colors
  - [ ] 28.3 Create `src/components/dashboard/QuickActions.tsx`
  - [ ] 28.4 Implement action buttons

- [ ] 29.0 Create Dashboard page
  - [ ] 29.1 Create `src/pages/DashboardPage.tsx`
  - [ ] 29.2 Add status cards grid (model, SAE, steering, monitoring)
  - [ ] 29.3 Add quick start guide based on state
  - [ ] 29.4 Connect to serverStore
  - [ ] 29.5 Write DashboardPage tests

### Phase 9: Models Page

- [ ] 30.0 Create Models components
  - [ ] 30.1 Create `src/components/models/ModelLoadForm.tsx`
  - [ ] 30.2 Implement model_id input
  - [ ] 30.3 Implement device select (auto/cuda/cpu)
  - [ ] 30.4 Implement dtype select
  - [ ] 30.5 Implement submit with validation
  - [ ] 30.6 Create `src/components/models/LoadedModelCard.tsx`
  - [ ] 30.7 Display model info
  - [ ] 30.8 Add unload button with confirmation

- [ ] 31.0 Create Models page
  - [ ] 31.1 Create `src/pages/ModelsPage.tsx`
  - [ ] 31.2 Show LoadedModelCard when model loaded
  - [ ] 31.3 Show ModelLoadForm when no model
  - [ ] 31.4 Handle loading states
  - [ ] 31.5 Write ModelsPage tests

### Phase 10: SAE Page

- [ ] 32.0 Create SAE components
  - [ ] 32.1 Create `src/components/sae/SAEDownloadForm.tsx`
  - [ ] 32.2 Implement repo_id and filename inputs
  - [ ] 32.3 Implement download button with progress
  - [ ] 32.4 Create `src/components/sae/SAEList.tsx`
  - [ ] 32.5 Create `src/components/sae/SAEListItem.tsx`
  - [ ] 32.6 Display SAE metadata (layer, features)
  - [ ] 32.7 Add attach/delete buttons
  - [ ] 32.8 Create `src/components/sae/AttachedSAECard.tsx`
  - [ ] 32.9 Display attached SAE info
  - [ ] 32.10 Add detach button

- [ ] 33.0 Create SAE page
  - [ ] 33.1 Create `src/pages/SAEPage.tsx`
  - [ ] 33.2 Show AttachedSAECard at top
  - [ ] 33.3 Show SAEDownloadForm
  - [ ] 33.4 Show SAEList with downloaded SAEs
  - [ ] 33.5 Handle empty state
  - [ ] 33.6 Write SAEPage tests

### Phase 11: Steering Page

- [ ] 34.0 Create Steering components
  - [ ] 34.1 Create `src/components/steering/FeatureInput.tsx`
  - [ ] 34.2 Implement feature index input with add button
  - [ ] 34.3 Create `src/components/steering/BatchAddForm.tsx`
  - [ ] 34.4 Implement comma-separated feature input
  - [ ] 34.5 Create `src/components/steering/SteeringSlider.tsx`
  - [ ] 34.6 Combine Slider with numeric input
  - [ ] 34.7 Add remove button
  - [ ] 34.8 Create `src/components/steering/SteeringControls.tsx`
  - [ ] 34.9 Implement enable/disable toggle
  - [ ] 34.10 Implement clear all button

- [ ] 35.0 Create Steering page
  - [ ] 35.1 Create `src/pages/SteeringPage.tsx`
  - [ ] 35.2 Show SteeringControls at top
  - [ ] 35.3 Show FeatureInput and BatchAddForm
  - [ ] 35.4 Show list of SteeringSliders
  - [ ] 35.5 Handle empty state
  - [ ] 35.6 Handle no SAE attached state
  - [ ] 35.7 Write SteeringPage tests

### Phase 12: Monitoring Page

- [ ] 36.0 Create Monitoring components
  - [ ] 36.1 Create `src/components/monitoring/MonitoringControls.tsx`
  - [ ] 36.2 Implement enable/disable toggle
  - [ ] 36.3 Implement top_k configuration select
  - [ ] 36.4 Implement pause/resume button
  - [ ] 36.5 Create `src/components/monitoring/ActivationChart.tsx`
  - [ ] 36.6 Implement horizontal bar chart for activations
  - [ ] 36.7 Create `src/components/monitoring/ActivationHistory.tsx`
  - [ ] 36.8 Implement scrollable history list
  - [ ] 36.9 Add clear history button
  - [ ] 36.10 Create `src/components/monitoring/StatisticsPanel.tsx`
  - [ ] 36.11 Display feature statistics

- [ ] 37.0 Create Monitoring page
  - [ ] 37.1 Create `src/pages/MonitoringPage.tsx`
  - [ ] 37.2 Show MonitoringControls at top
  - [ ] 37.3 Show ActivationChart
  - [ ] 37.4 Show ActivationHistory and StatisticsPanel
  - [ ] 37.5 Handle empty/no SAE state
  - [ ] 37.6 Write MonitoringPage tests

### Phase 13: Profiles Page

- [ ] 38.0 Create Profile components
  - [ ] 38.1 Create `src/components/profiles/ProfileListItem.tsx`
  - [ ] 38.2 Display profile name, description, feature count
  - [ ] 38.3 Add activate/deactivate toggle
  - [ ] 38.4 Add edit/delete buttons
  - [ ] 38.5 Create `src/components/profiles/ProfileList.tsx`
  - [ ] 38.6 List ProfileListItems
  - [ ] 38.7 Create `src/components/profiles/ProfileForm.tsx`
  - [ ] 38.8 Implement create/edit form with name, description
  - [ ] 38.9 Option to save current steering
  - [ ] 38.10 Create `src/components/profiles/ImportExportButtons.tsx`
  - [ ] 38.11 Implement export to file download
  - [ ] 38.12 Implement import from file upload

- [ ] 39.0 Create Profiles page
  - [ ] 39.1 Create `src/pages/ProfilesPage.tsx`
  - [ ] 39.2 Show create profile button
  - [ ] 39.3 Show ProfileList
  - [ ] 39.4 Show ImportExportButtons
  - [ ] 39.5 Implement profile form modal
  - [ ] 39.6 Handle empty state
  - [ ] 39.7 Write ProfilesPage tests

### Phase 14: Settings Page

- [ ] 40.0 Create Settings page
  - [ ] 40.1 Create `src/pages/SettingsPage.tsx`
  - [ ] 40.2 Implement theme toggle section
  - [ ] 40.3 Implement connection status section
  - [ ] 40.4 Add manual reconnect button
  - [ ] 40.5 Display server URL
  - [ ] 40.6 Write SettingsPage tests

### Phase 15: App Integration

- [ ] 41.0 Create App entry point
  - [ ] 41.1 Create `src/App.tsx` with router
  - [ ] 41.2 Define all routes with Layout wrapper
  - [ ] 41.3 Create `src/main.tsx`
  - [ ] 41.4 Initialize theme from store
  - [ ] 41.5 Connect WebSocket on startup
  - [ ] 41.6 Set up QueryClient with defaults
  - [ ] 41.7 Create `src/index.css` with animations

- [ ] 42.0 Configure proxy for development
  - [ ] 42.1 Update `vite.config.ts` with API proxy
  - [ ] 42.2 Update `vite.config.ts` with WebSocket proxy
  - [ ] 42.3 Add environment variable support
  - [ ] 42.4 Test dev server with backend

### Phase 16: Testing

- [ ] 43.0 Set up testing infrastructure
  - [ ] 43.1 Create `vitest.config.ts`
  - [ ] 43.2 Create `tests/setup.ts` with mocks
  - [ ] 43.3 Add test scripts to package.json
  - [ ] 43.4 Verify test runner works

- [ ] 44.0 Write component tests
  - [ ] 44.1 Test all common components
  - [ ] 44.2 Test layout components
  - [ ] 44.3 Test feature components
  - [ ] 44.4 Aim for 80% component coverage

- [ ] 45.0 Write integration tests
  - [ ] 45.1 Create `tests/integration/pages/DashboardPage.test.tsx`
  - [ ] 45.2 Create tests for each page
  - [ ] 45.3 Mock API and WebSocket
  - [ ] 45.4 Test user flows

### Phase 17: Backend Integration

- [ ] 46.0 Configure static file serving
  - [ ] 46.1 Update `millm/main.py` to serve admin-ui/dist
  - [ ] 46.2 Mount static files at root
  - [ ] 46.3 Configure SPA fallback for client-side routing
  - [ ] 46.4 Test build and serve

- [ ] 47.0 Build and deploy
  - [ ] 47.1 Add build script to package.json
  - [ ] 47.2 Test production build
  - [ ] 47.3 Verify all routes work
  - [ ] 47.4 Verify WebSocket connects
  - [ ] 47.5 Verify API calls work

### Phase 18: Polish & Documentation

- [ ] 48.0 Final polish
  - [ ] 48.1 Review all error handling
  - [ ] 48.2 Review all loading states
  - [ ] 48.3 Review responsive behavior
  - [ ] 48.4 Test in Chrome, Firefox, Safari
  - [ ] 48.5 Fix any visual inconsistencies

- [ ] 49.0 Documentation
  - [ ] 49.1 Add README.md to admin-ui
  - [ ] 49.2 Document environment variables
  - [ ] 49.3 Document build process
  - [ ] 49.4 Document component library

---

**Total Tasks:** 49 parent tasks, 200+ sub-tasks
**Estimated Timeline:** 2-3 weeks
