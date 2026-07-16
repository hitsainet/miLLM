# Rule: Generating a Task List from a PRD

## Goal

To guide an AI assistant in creating a detailed, step-by-step task list in Markdown format based on an existing Feature PRD and its companion Technical Design Document (TDD) and Technical Implementation Document (TID). The task list is the feature's complete backlog: every functional requirement, acceptance criterion, edge case, and technical concern from the source documents must map to at least one task — or be explicitly marked out of scope with a reason.

## Inputs

- **Required:** Feature PRD (`[###]_FPRD|[feature-name].md`) — requirements, user stories, acceptance criteria, edge cases
- **Required:** Feature TDD (`[###]_FTDD|[feature-name].md`) and TID (`[###]_FTID|[feature-name].md`) — these carry the migration, deployment, configuration, error-handling, integration, and utility work that the PRD alone does not. If either is missing, stop and recommend creating it first (`005_create-tdd.md` / `006_create-tid.md`). Only proceed without them at the user's explicit request, and note in the task list that coverage is limited to the PRD.
- **Reference:** Project ADR (`000_PADR|[project-name].md`) for standards, and AGENT.md for project test commands

## Output

- **Format:** Markdown (`.md`)
- **Location:** `0xcc/tasks/`
- **Filename:** `[###]_FTASKS|[feature-name].md` (matching the corresponding PRD's feature name)

## Process

1.  **Receive Document References:** The user points the AI to a Feature PRD and its TDD and TID.
2.  **Analyze Source Documents:** Read the PRD (functional requirements, user stories, acceptance criteria, edge cases, dependencies, open questions), the TDD (architecture, data design, API design, deployment/DevOps), and the TID (file organization, implementation patterns, error handling, configuration, integration strategy).
3.  **Assess Current State:** Review the existing codebase to understand existing infrastructure, architectural patterns and conventions established in the project's ADR. Identify existing components, utilities, and files that can be leveraged or need modification. **If the codebase is empty or nearly empty** (e.g., this is the project's first feature), check whether `0xcc/tasks/000_FTASKS|Project_Foundation.md` exists. If it does not, either generate it first (see "Project Foundation Task List" below) or include the required bootstrap work in this feature's list as an explicit "Foundation" parent task — never generate implementation tasks that have no runnable project to land in.
4.  **Phase 1: Generate Parent Tasks:** Based on the document analysis and current state assessment, create the file and generate the main, high-level tasks required to implement the feature. **Scale the number of parent tasks to the feature's complexity** (use the PRD's "Implementation Considerations" assessment): a simple feature may need 3–4, a complex one 8 or more. Do not compress a complex feature into a fixed number of tasks. **Each parent task must cite the functional requirement numbers it covers** (e.g., "covers FR-1, FR-3"). Present these tasks to the user in the specified format (without sub-tasks yet). Inform the user: "I have generated the high-level tasks based on the PRD. Ready to generate the sub-tasks? Respond with 'Go' to proceed."
5.  **Wait for Confirmation:** Pause and wait for the user to respond with "Go".
6.  **Phase 2: Generate Sub-Tasks:** Break down each parent task into smaller, actionable sub-tasks. Granularity rule: **one sub-task ≈ one commit** — completable in a single sitting, with a verifiable outcome. Ensure sub-tasks logically follow from the parent task, cover the implementation details from the PRD/TDD/TID, and consider existing codebase patterns where relevant without being constrained by them. **For every acceptance criterion and edge case in the PRD, create both an implementing sub-task and a testing sub-task.**
7.  **Apply the Category Checklist:** Work through the checklist below. Every category must either produce tasks or be recorded as N/A with a one-line reason in the "Category Checklist Results" section of the output.
8.  **Convert Open Questions to Tasks:** Any unresolved item in the PRD's "Open Questions" section becomes an explicit research/spike/decision sub-task, placed before the work that depends on its answer.
9.  **Derive Integration Tasks:** From the PRD's "Dependencies" section and the TID's "Integration Strategy" section, enumerate the modifications required to *existing* features and files as their own tasks — cross-feature ripple work is part of this feature's backlog.
10. **Add the Feature Acceptance Parent Task:** The final parent task always verifies the feature end-to-end: check each PRD success criterion and acceptance criterion, run the full test suite, and update AGENT.md's Document Inventory.
11. **Run the Coverage Audit:** Complete the audit below and fix any gaps before presenting the final list.
12. **Identify Relevant Files:** Based on the tasks and source documents, identify files to be created or modified, including corresponding test files.
13. **Generate Final Output:** Combine parent tasks, sub-tasks, relevant files, checklist results, and notes into the final Markdown structure.
14. **Save Task List:** Save the document in `0xcc/tasks/` as `[###]_FTASKS|[feature-name].md`, where `[feature-name]` matches the base name of the input Feature PRD file (e.g., input `001_FPRD|User_Profile_Editing.md` → output `001_FTASKS|User_Profile_Editing.md`).
15. **Update AGENT.md:** Record the new task list in the Document Inventory (add/flip its entry to ✅) and update Current Status/Next Steps.

## Category Checklist

Every task list must consider each category. Produce tasks, or mark the category N/A with a stated reason — silent omission is not allowed:

- **Data layer:** schema changes, migrations, seed data (PRD §Data Requirements, TDD §Data Design, TID §Database Implementation)
- **Backend/API:** endpoints, request/response handling, validation, auth integration (PRD §API/Integration Specifications, TDD §API Design, TID §API Implementation)
- **Frontend/UI:** components, state management, accessibility, responsive behavior (PRD §User Experience Requirements, TID §Frontend Implementation)
- **Business logic:** core algorithms, transformations, validation rules (PRD §Functional Requirements, TID §Business Logic)
- **Integration wiring:** modifications to existing features and files (PRD §Dependencies, TID §Integration Strategy)
- **Error handling & logging:** failure paths for each edge case, user-facing error states (PRD §User Stories edge cases, TID §Error Handling and Logging)
- **Testing:** unit **and** integration/end-to-end/performance tests as required by PRD §Testing Requirements — never unit tests alone if the PRD asks for more
- **Performance & security hardening:** benchmarks, optimization, sanitization, access control (PRD §Non-Functional Requirements, TDD §Security / §Performance & Scalability)
- **Configuration / environment / deployment:** env vars, feature flags, pipeline changes, monitoring (TDD §Deployment & DevOps, TID §Configuration and Environment)
- **Documentation:** README/API docs/user-facing docs affected by this feature

## Coverage Audit

Before saving, verify — and correct — the following:

1. Every numbered functional requirement in the PRD is cited by at least one parent task. List any uncovered requirements and add tasks for them.
2. Every acceptance criterion and edge case has both an implementing sub-task and a testing sub-task.
3. Every applicable TDD section (Data Design through Deployment & DevOps) and TID section is represented by at least one task or an N/A checklist entry.
4. Every unresolved Open Question from the PRD appears as a spike/decision task.
5. The final parent task is Feature Acceptance.

## Output Format

The generated task list _must_ follow this structure:

```markdown
## Relevant Files

- `path/to/potential/file1.ts` - Brief description of why this file is relevant (e.g., Contains the main component for this feature).
- `path/to/file1.test.ts` - Unit tests for `file1.ts`.
- `path/to/another/file.tsx` - Brief description (e.g., API route handler for data submission).
- `path/to/another/file.test.tsx` - Unit tests for `another/file.tsx`.
- `lib/utils/helpers.ts` - Brief description (e.g., Utility functions needed for calculations).
- `lib/utils/helpers.test.ts` - Unit tests for `helpers.ts`.

### Notes

- Unit tests should typically be placed alongside the code files they are testing (e.g., `MyComponent.tsx` and `MyComponent.test.tsx` in the same directory).
- Run tests with the project's test command as defined in the ADR / AGENT.md (e.g., `npm test`, `pytest`). Running without a path executes the full suite.

### Category Checklist Results

- Data layer: tasks 2.x
- Backend/API: tasks 3.x
- Frontend/UI: tasks 4.x
- Business logic: tasks 3.x
- Integration wiring: tasks 5.x
- Error handling & logging: tasks 3.4, 4.5
- Testing: sub-tasks throughout; integration tests in 6.x
- Performance & security: N/A — [one-line reason]
- Configuration/deployment: tasks 1.x
- Documentation: task 7.2

## Tasks

- [ ] 1.0 Parent Task Title (covers FR-1, FR-3)
  - [ ] 1.1 [Sub-task description 1.1]
  - [ ] 1.2 [Sub-task description 1.2]
- [ ] 2.0 Parent Task Title (covers FR-2)
  - [ ] 2.1 [Sub-task description 2.1]
- [ ] 3.0 Parent Task Title (may not require sub-tasks if purely structural or configuration)
- [ ] N.0 Feature Acceptance (always the final parent task)
  - [ ] N.1 Verify each acceptance criterion from the PRD, one by one
  - [ ] N.2 Run the full test suite
  - [ ] N.3 Update AGENT.md Document Inventory (mark this feature's documents ✅) and file any discovered follow-up work as new tasks
```

## Project Foundation Task List (once per project)

Work that belongs to no single feature — repo and toolchain scaffolding, framework/app initialization, database provisioning and base schema, CI/CD pipeline, test runner configuration, authentication foundation, logging and error-handling utilities, design system setup, deployment environments — gets its own backlog: `0xcc/tasks/000_FTASKS|Project_Foundation.md`.

Generate it once, right after the ADR is complete and before the first feature, using this same document's process with these substitutions:

- **Inputs:** Project PRD (`000_PPRD|...`) and ADR (`000_PADR|...`) instead of a Feature PRD/TDD/TID
- **Requirement source:** the ADR's Technology Stack, Development Standards, and infrastructure decisions; the Project PRD's high-level and non-functional requirements
- **Final parent task:** "Foundation Acceptance" — a clean clone builds, tests run green, CI passes, and a walking-skeleton deploy succeeds

Feature task lists may then assume the foundation exists; if a feature discovers missing foundation work, file it in the foundation list rather than duplicating it per-feature.

## Interaction Model

The process explicitly requires a pause after generating parent tasks to get user confirmation ("Go") before proceeding to generate the detailed sub-tasks. This ensures the high-level plan aligns with user expectations before diving into details.

## Target Audience

Assume the primary reader of the task list is a **junior developer** who will implement the feature with awareness of the existing codebase context.
