# Rule: Generating a Technical Implementation Document (TID)

## Goal

To guide an AI assistant in creating a detailed Technical Implementation Document (TID) in Markdown format, based on an existing Product Requirements Document (PRD) and Technical Design Document (TDD). The TID should provide specific, actionable implementation details that directly inform the task generation process.

## Process

1. **Receive PRD and TDD References:** The user provides references to existing PRD and TDD files.
2. **Analyze PRD and TDD:** Read and synthesize the business requirements from PRD and technical architecture from TDD.
3. **Deep Codebase Analysis:** Perform comprehensive analysis of existing code structure, patterns, utilities, and conventions.
4. **Ask Implementation-Specific Questions:** Before writing the TID, ask detailed implementation questions. Provide options in letter/number lists for easy selection.
5. **Generate TID:** Create a comprehensive implementation document using the structure outlined below.
6. **Save TID:** Save the generated document as `tid-[feature-name].md` inside the `/tasks` directory.

## Implementation Clarifying Questions (Examples)

Focus on specific implementation details that will directly impact task creation:

* **File Organization:** "How should we organize the new code files?"
  - A) Follow existing directory structure patterns
  - B) Create new feature-specific directories
  - C) Group by functionality (components, services, utils)
  - D) Mixed approach based on file types

* **Naming Conventions:** "What naming patterns should we follow?"
  - A) Existing project conventions only
  - B) Feature-prefixed naming (e.g., `UserProfile*`)
  - C) Domain-driven naming
  - D) Functional naming patterns

* **Code Generation Approach:** "Should we prioritize any specific implementation approach?"
  - A) Extend existing components/classes
  - B) Create new standalone implementations
  - C) Refactor existing code for reusability
  - D) Hybrid approach

* **Error Handling:** "How should errors be handled throughout the implementation?"
  - A) Use existing error handling patterns
  - B) Implement feature-specific error handling
  - C) Global error boundary approach
  - D) Custom error handling strategy

* **Validation Strategy:** "Where and how should data validation occur?"
  - A) Frontend validation only
  - B) Backend validation only
  - C) Both frontend and backend validation
  - D) Schema-based validation approach

* **Code Reusability:** "What level of code reusability should we target?"
  - A) Maximize use of existing utilities and components
  - B) Create new reusable components for future features
  - C) Feature-specific implementation (minimal reusability)
  - D) Balanced approach

## TID Structure

The generated TID should include the following sections:

1. **Implementation Overview:**
   - Summary of implementation approach
   - Key implementation principles and patterns to follow
   - Integration points with existing codebase

2. **File Structure and Organization:**
   - Directory organization approach and naming patterns
   - File naming conventions and grouping strategy
   - Dependency organization and import patterns
   - Configuration integration hints

3. **Component Implementation Hints:**
   - Component design patterns and abstraction levels
   - Interface design principles and consistency
   - Lifecycle management and state handling hints
   - Composition patterns and reusability approach

4. **Database Implementation Approach:**
   - Schema design patterns and field organization
   - Migration strategy and rollback considerations
   - Query optimization patterns and indexing hints
   - Data integrity and constraint strategies

5. **API Implementation Strategy:**
   - Endpoint organization and RESTful design hints
   - Request/response handling patterns
   - Validation layer organization and error patterns
   - Authentication integration and middleware approach

6. **Frontend Implementation Approach:**
   - Component composition and hierarchy hints
   - State management integration patterns
   - Event handling and user interaction strategies
   - Styling organization and responsive design approach

7. **Business Logic Implementation Hints:**
   - Core algorithm approach and processing patterns
   - Data transformation strategies and validation patterns
   - External service integration patterns
   - Caching and performance optimization strategies

8. **Testing Implementation Approach:**
   - Test organization patterns and coverage strategy
   - Test data management and isolation patterns
   - Mock and stub strategies and dependency injection
   - Assertion patterns and verification approaches

9. **Configuration and Environment Strategy:**
   - Environment-specific configuration patterns
   - Feature flag integration and toggle strategies
   - Build process integration hints
   - Deployment configuration approaches

10. **Integration Strategy:**
    - Existing code modification patterns and backwards compatibility
    - API integration patterns and data synchronization
    - Event handling and messaging strategies
    - Third-party service integration approaches

11. **Utilities and Helpers Design:**
    - Reusable utility organization and naming patterns
    - Helper abstraction levels and composition hints
    - Validation and transformation strategy patterns
    - Common pattern extraction and reusability approaches

12. **Error Handling and Logging Strategy:**
    - Error categorization and handling patterns
    - Logging strategy and information capture approach
    - User feedback patterns and error communication
    - Recovery and fallback strategy patterns

13. **Performance Implementation Hints:**
    - Optimization technique selection and implementation hints
    - Caching strategy implementation and invalidation patterns
    - Lazy loading and resource management approaches
    - Database interaction optimization strategies

14. **Code Quality and Standards:**
    - Code organization and documentation patterns
    - Consistency and maintainability approaches
    - Refactoring strategy and technical debt management
    - Review and validation integration hints

## Target Audience

The TID serves as the direct input for:
- **Task generation process** (consumed by generate-tasks.md)
- **Junior developers** who need specific implementation guidance
- **Code reviewers** who need to understand implementation decisions
- **QA engineers** who need to understand testing requirements

## Output

* **Format:** Markdown (`.md`)
* **Location:** `/tasks/`
* **Filename:** `tid-[feature-name].md` (should match corresponding PRD and TDD name patterns)

## Integration with Task Generation

The TID should be structured to directly inform the task generation process by:
- Providing clear, actionable implementation steps
- Identifying all files that need creation or modification
- Specifying dependencies between implementation components
- Including testing and validation requirements for each component

## Final Instructions

1. Do NOT start implementing code or create the TID immediately
2. First ask the implementation-specific clarifying questions
3. Perform thorough analysis of existing codebase patterns and conventions
4. Take the user's answers, PRD content, and TDD content to create a comprehensive TID
5. Ensure the TID provides sufficient detail for accurate and comprehensive task generation
6. Include specific implementation details that will translate directly into actionable tasks