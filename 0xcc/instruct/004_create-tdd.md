# Rule: Generating a Technical Design Document (TDD)

## Goal

To guide an AI assistant in creating a detailed Technical Design Document (TDD) in Markdown format, based on an existing Product Requirements Document (PRD). The TDD should provide the technical architecture, design patterns, and implementation approach for the feature, serving as a bridge between business requirements and actual code implementation.

## Process

1. **Receive PRD Reference:** The user provides a reference to an existing PRD file.
2. **Analyze PRD:** Read and understand the functional requirements, user stories, and business objectives from the PRD.
3. **Assess Current Codebase:** Review existing architecture, design patterns, technologies, and conventions used in the project.
4. **Ask Technical Clarifying Questions:** Before writing the TDD, the AI *must* ask technical clarifying questions to understand implementation preferences and constraints. Provide options in letter/number lists for easy selection.
5. **Generate TDD:** Create a comprehensive technical design document using the structure outlined below.
6. **Save TDD:** Save the generated document as `tdd-[feature-name].md` inside the `/tasks` directory.

## Technical Clarifying Questions (Examples)

The AI should adapt questions based on the PRD and existing codebase, but common areas include:

* **Architecture Pattern:** "Which architectural pattern should we follow for this feature?"
  - A) Follow existing MVC/MVVM pattern
  - B) Implement using Component-based architecture
  - C) Use microservices approach
  - D) Other: [specify]

* **Data Layer:** "How should we handle data persistence and retrieval?"
  - A) Extend existing database schema
  - B) Create new tables/collections
  - C) Use external API integration
  - D) In-memory storage only

* **API Design:** "What API approach should we use?"
  - A) REST endpoints following existing conventions
  - B) GraphQL queries/mutations
  - C) WebSocket for real-time features
  - D) Combination approach

* **State Management:** "How should we handle application state?"
  - A) Use existing state management solution (Redux, Vuex, etc.)
  - B) Local component state only
  - C) Context API / Provide/Inject pattern
  - D) New state management implementation

* **Authentication/Authorization:** "What security considerations apply?"
  - A) Use existing auth system
  - B) Add new permission levels
  - C) Public feature (no auth required)
  - D) External auth integration needed

* **Performance Requirements:** "Are there specific performance constraints?"
  - A) Standard performance expectations
  - B) High-throughput requirements
  - C) Real-time/low-latency needs
  - D) Offline functionality required

* **Testing Strategy:** "What testing approach should we prioritize?"
  - A) Unit tests for core logic
  - B) Integration tests for data flow
  - C) End-to-end user journey tests
  - D) All of the above

## TDD Structure

The generated TDD should include the following sections:

1. **Executive Summary:** Brief overview linking business goals from PRD to technical approach.

2. **System Architecture:**
   - High-level architecture diagram (text description)
   - Component relationships and data flow
   - Integration points with existing systems

3. **Technical Stack:**
   - Technologies, frameworks, and libraries to be used
   - Justification for technology choices
   - Dependencies and version requirements

4. **Data Design:**
   - Database schema considerations and approach
   - Data relationship patterns to follow
   - Validation strategy and consistency hints
   - Migration approach and data preservation strategy

5. **API Design:**
   - API design patterns and conventions to follow
   - Data flow and transformation hints
   - Error handling strategy and consistency approach
   - Security and performance design principles

6. **Component Architecture:**
   - Component organization and hierarchy approach
   - Reusability patterns and abstraction hints
   - Data flow and communication patterns
   - Separation of concerns guidance

7. **State Management:**
   - Application state organization principles
   - State flow patterns and update strategies
   - Side effects handling approach
   - Caching strategy and data consistency hints

8. **Security Considerations:**
   - Authentication and authorization strategy
   - Data validation and sanitization approach
   - Security best practices to follow
   - Privacy and compliance guidance

9. **Performance & Scalability:**
   - Performance optimization principles
   - Caching strategy and invalidation approach
   - Database optimization hints
   - Scalability design considerations

10. **Testing Strategy:**
    - Testing approach and coverage philosophy
    - Test organization and dependency management
    - Testing patterns and best practices
    - Mock and fixture strategy guidance

11. **Deployment & DevOps:**
    - Deployment pipeline changes
    - Environment configurations
    - Monitoring and logging requirements
    - Rollback strategy

12. **Risk Assessment:**
    - Technical risks and mitigation strategies
    - Dependencies and potential blockers
    - Complexity assessment
    - Alternative approaches considered

13. **Development Phases:**
    - High-level implementation phases
    - Dependencies between phases
    - Milestone definitions
    - Estimated effort and timeline

## Target Audience

The TDD should be understandable by:
- **Senior developers** who need to review the technical approach
- **Junior developers** who will implement portions of the feature
- **Technical leads** who need to assess feasibility and resource allocation
- **QA engineers** who need to understand testing requirements

## Output

* **Format:** Markdown (`.md`)
* **Location:** `/tasks/`
* **Filename:** `tdd-[feature-name].md` (should match the corresponding PRD name pattern)

## Final Instructions

1. Do NOT start implementing code or create the TDD immediately
2. First ask the technical clarifying questions to understand implementation preferences
3. Review the existing codebase to understand current patterns and architecture
4. Take the user's answers and the PRD content to create a comprehensive TDD
5. Ensure the TDD provides enough technical detail for the next phase (Technical Implementation Document)