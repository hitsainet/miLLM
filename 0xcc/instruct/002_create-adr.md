# Rule: Generating an Architecture Decision Record (ADR) and Tech Stack Specification

## Goal

To guide an AI assistant in creating a comprehensive Architecture Decision Record (ADR) that establishes foundational technology choices, development principles, and architectural standards for the entire project. This document standardizes technical decisions before feature-level development begins.

## Process

1. **Receive Project PRD Reference:** The user provides a reference to the existing Project PRD file.
2. **Analyze Project Requirements:** Review the project goals, user base, scale requirements, and constraints from the Project PRD.
3. **Assess Technical Landscape:** Consider available technology options for each decision area.
4. **Ask Technical Evaluation Questions:** Present technology choices and trade-offs for user decision-making. Provide options in letter/number lists for easy selection.
5. **Generate ADR:** Create a comprehensive architecture decision record using the structure outlined below, including both a full documentation version and a condensed CLAUDE.md section.
6. **Save ADR Files:** Save both the complete document as `adr-[project-name].md` and generate the CLAUDE.md content section.

## Technical Evaluation Questions (Examples)

The AI should present technology options with pros/cons for each decision area:

* **Application Architecture:** "What overall architecture pattern fits your project best?"
  - A) Monolithic application (simpler deployment, faster initial development)
  - B) Microservices (scalable, team independence, complex deployment)
  - C) Modular monolith (balanced approach, easier to split later)
  - D) Serverless/Function-based (auto-scaling, pay-per-use)

* **Frontend Technology:** "Which frontend approach aligns with your requirements?"
  - A) React (large ecosystem, flexible, steep learning curve)
  - B) Vue.js (gentle learning curve, good documentation)
  - C) Angular (full framework, enterprise features, complex)
  - D) Svelte (performance, smaller bundle sizes, newer ecosystem)
  - E) Streamlit (rapid prototyping, Python-centric, limited customization)
  - F) Server-rendered (Django/Rails templates, SEO-friendly, simpler)

* **Backend Technology:** "Which backend technology suits your needs?"
  - A) Node.js/Express (JavaScript everywhere, fast development)
  - B) Python/Django (rapid development, admin interface, batteries included)
  - C) Python/Flask (lightweight, flexible, minimal)
  - D) Python/FastAPI (modern, automatic documentation, type hints)
  - E) Java/Spring Boot (enterprise features, strong typing)
  - F) Go (performance, simple deployment, growing ecosystem)

* **Database Choice:** "What database approach fits your data needs?"
  - A) PostgreSQL (powerful relational, ACID compliance, extensions)
  - B) MySQL (widespread adoption, good performance)
  - C) SQLite (simple setup, file-based, good for prototypes)
  - D) MongoDB (document store, flexible schema, NoSQL)
  - E) Redis (in-memory, caching, sessions)
  - F) Hybrid approach (multiple databases for different needs)

* **API Design:** "How should your application expose APIs?"
  - A) REST APIs (standard, cacheable, stateless)
  - B) GraphQL (flexible queries, single endpoint, learning curve)
  - C) gRPC (performance, type safety, binary protocol)
  - D) WebSocket (real-time, bidirectional, connection overhead)
  - E) Mixed approach (different APIs for different needs)

* **State Management:** "How should application state be managed?"
  - A) Built-in framework state (React hooks, Vue composition)
  - B) Dedicated library (Redux, Vuex, Zustand)
  - C) Server state management (React Query, SWR)
  - D) Simple/minimal state management

* **Authentication & Authorization:** "What authentication approach do you prefer?"
  - A) Session-based (traditional, server-side sessions)
  - B) JWT tokens (stateless, scalable, token management needed)
  - C) OAuth/Social login (user convenience, external dependency)
  - D) Custom authentication (full control, more development)
  - E) Third-party service (Auth0, Firebase Auth, AWS Cognito)

* **Testing Strategy:** "What testing approach should be prioritized?"
  - A) Unit testing focus (Jest, pytest, fast feedback)
  - B) Integration testing focus (API testing, database interactions)
  - C) End-to-end testing (Cypress, Playwright, user workflows)
  - D) Balanced testing pyramid (all levels with emphasis on unit)

* **Deployment & DevOps:** "What deployment strategy fits your project?"
  - A) Cloud platforms (Heroku, Vercel, Netlify - simple deployment)
  - B) Container orchestration (Docker + Kubernetes - scalable, complex)
  - C) Virtual machines (DigitalOcean, AWS EC2 - traditional, flexible)
  - D) Serverless deployment (AWS Lambda, Vercel Functions)
  - E) On-premise (full control, infrastructure management)

* **Development Principles:** "Which development principles should guide the project?"
  - A) Rapid prototyping (speed over perfection, iterate quickly)
  - B) Enterprise-grade (robust, maintainable, comprehensive testing)
  - C) Performance-first (optimization-focused, lean codebase)
  - D) Developer experience (good tooling, clear patterns, productivity)

## ADR Structure

The generated ADR should include the following sections in the **full documentation**, with key elements summarized in the **CLAUDE.md content section**:

### Full ADR Document Structure:

1. **Decision Summary:**
   - Date and project context
   - Key architectural decisions overview
   - Decision-making criteria and priorities

2. **Technology Stack Decisions:**

   **Frontend Stack:**
   - Primary framework/library choice and rationale
   - UI component approach and design system
   - State management solution
   - Build tools and development environment

   **Backend Stack:**
   - Server technology and framework choice
   - API design approach and standards
   - Authentication and authorization strategy
   - Background job processing (if needed)

   **Database & Data:**
   - Primary database choice and rationale
   - Data modeling approach
   - Caching strategy
   - Data migration and backup approach

   **Infrastructure & Deployment:**
   - Deployment platform and strategy
   - Container strategy (if applicable)
   - Environment management approach
   - Monitoring and logging strategy

3. **Development Standards:**

   **Code Organization:**
   - Directory structure and file naming conventions
   - Module organization and dependency management
   - Code style and formatting standards
   - Documentation requirements

   **Quality Assurance:**
   - Testing strategy and coverage expectations
   - Code review process and standards
   - Continuous integration approach
   - Performance monitoring and optimization

   **Development Workflow:**
   - Version control strategy and branching model
   - Development environment setup
   - Package management and dependency handling
   - Release and deployment procedures

4. **Architectural Principles:**
   - Core design principles to follow
   - Scalability and performance considerations
   - Security and privacy requirements
   - Maintainability and code quality standards

5. **Package and Library Standards:**
   - Approved libraries and frameworks for common tasks
   - Package selection criteria and evaluation process
   - Version management and update strategy
   - Custom vs. third-party solution guidelines

6. **Integration Guidelines:**
   - API design standards and conventions
   - Data exchange formats and protocols
   - Error handling and logging standards
   - Cross-service communication patterns

7. **Development Environment:**
   - Required development tools and IDEs
   - Local development setup and configuration
   - Testing environment requirements
   - Debugging and profiling tools

8. **Security Standards:**
   - Authentication and authorization patterns
   - Data validation and sanitization requirements
   - Secure coding practices
   - Vulnerability management approach

9. **Performance Guidelines:**
   - Performance targets and monitoring
   - Optimization strategies and best practices
   - Caching policies and implementation
   - Resource management standards

10. **Decision Rationale:**
    - Trade-offs considered for major decisions
    - Alternative options evaluated and rejected
    - Risk assessment and mitigation strategies
    - Future flexibility and evolution considerations

11. **Implementation Guidelines:**
    - How these decisions should be applied in feature development
    - Exception handling and decision review process
    - Documentation and knowledge sharing requirements
    - Team training and onboarding considerations

### CLAUDE.md Content Section Structure:

The condensed section for CLAUDE.md should follow this format:

```markdown
# Project Standards

## Technology Stack
- **Frontend:** [Chosen technology and key libraries]
- **Backend:** [Chosen framework and key dependencies]
- **Database:** [Database choice and key patterns]
- **Testing:** [Testing frameworks and approach]
- **Deployment:** [Deployment platform and strategy]

## Development Standards

### Code Organization
- [Directory structure patterns]
- [File naming conventions]
- [Import and dependency patterns]

### Coding Patterns
- [State management approach]
- [Error handling patterns]
- [API design conventions]
- [Component organization principles]

### Quality Requirements
- [Testing coverage expectations]
- [Code review standards]
- [Documentation requirements]
- [Performance considerations]

## Architecture Principles
- [Core design principles]
- [Security requirements]
- [Scalability considerations]
- [Integration guidelines]

## Implementation Notes
- [Key constraints and limitations]
- [Package and library standards]
- [Environment and configuration approach]
```

## Target Audience

The ADR serves as a reference for:
- **All developers** working on any part of the project
- **Technical leads** making implementation decisions
- **New team members** understanding project standards
- **Feature PRD creators** who need to understand technical constraints

## Output

* **Format:** Two files will be generated:
  1. **Full ADR:** `adr-[project-name].md` in `/tasks/` directory (complete documentation)
  2. **CLAUDE.md Content:** Condensed standards for inclusion in project's `CLAUDE.md` file

## CLAUDE.md Integration

The ADR will generate a **"Project Standards"** section specifically formatted for CLAUDE.md that includes:

- **Technology Stack Summary:** Chosen technologies and key libraries
- **Development Standards:** Coding patterns, file organization, naming conventions
- **Architecture Principles:** Key design decisions and constraints
- **Testing Requirements:** Testing approach and coverage expectations
- **Code Quality Guidelines:** Formatting, documentation, and review standards

This content should be **copied into the project's CLAUDE.md file** to ensure all AI interactions follow established project standards.

## Integration with Workflow

This ADR should be:
- **Created after** the Project PRD is complete
- **Content added to CLAUDE.md** immediately after generation
- **Referenced by** all subsequent feature PRDs, TDDs, and TIDs
- **Updated occasionally** as major architectural decisions evolve
- **Consulted during** task generation to ensure consistency

## Final Instructions

1. Do NOT start implementing any code or creating feature documents
2. Focus on establishing foundational decisions that will guide all development
3. Present technology options with clear trade-offs for user decision-making
4. Create both a comprehensive ADR and a condensed CLAUDE.md section
5. Ensure the CLAUDE.md content captures the essential standards for AI memory
6. Include clear, actionable guidelines that can be easily referenced during development
7. Format the CLAUDE.md section for easy copy-paste into the project's CLAUDE.md file
8. Consider the project's specific requirements, timeline, and team capabilities when presenting options