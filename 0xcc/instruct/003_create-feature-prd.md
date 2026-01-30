# Rule: Generating a Feature-Level Product Requirements Document (PRD)

## Goal

To guide an AI assistant in creating a detailed, feature-specific Product Requirements Document (PRD) in Markdown format, based on a feature identified in the Project PRD and informed by the established Architecture Decision Record (ADR). The PRD should be clear, actionable, and suitable for a junior developer to understand and implement the specific feature.

## Process

1. **Receive Feature Request:** The user specifies which feature from the Project PRD they want to detail, or describes a new feature.
2. **Reference Project Context:** Review the Project PRD and ADR to understand overall project goals and established technical standards.
3. **Ask Feature-Specific Clarifying Questions:** Before writing the Feature PRD, the AI *must* ask clarifying questions to gather sufficient detail about this specific feature. Provide options in letter/number lists for easy selection.
4. **Generate Feature PRD:** Based on the feature request, project context, and user answers, generate a PRD using the structure outlined below.
5. **Save Feature PRD:** Save the generated document as `prd-[feature-name].md` inside the `/tasks` directory.

## Feature-Specific Clarifying Questions (Examples)

The AI should adapt its questions based on the feature description and project context:

* **Feature Priority:** "What is the priority level of this feature?"
  - A) Core/MVP feature (essential for project success)
  - B) Important feature (valuable but not critical)
  - C) Nice-to-have feature (future enhancement)
  - D) Experimental feature (testing concept/approach)

* **User Stories:** "Can you describe the primary user scenarios for this feature?"
  - A) Single primary user workflow
  - B) Multiple user types with different workflows
  - C) Admin/management functionality
  - D) Integration/API feature for other systems

* **Feature Scope:** "What is the intended scope for this feature?"
  - A) Simple/minimal implementation
  - B) Standard feature with common functionality
  - C) Advanced feature with complex workflows
  - D) Enterprise-level feature with extensive options

* **Data Requirements:** "What kind of data does this feature work with?"
  - A) User-generated content and interactions
  - B) System-generated data and analytics
  - C) External data integration
  - D) Configuration and settings data
  - E) Real-time or streaming data

* **Integration Needs:** "How does this feature interact with other parts of the system?"
  - A) Standalone feature with minimal integration
  - B) Integrates with existing user management
  - C) Requires new database tables/collections
  - D) Needs external API or service integration
  - E) Affects multiple existing features

* **User Interface Requirements:** "What are the UI/UX expectations for this feature?"
  - A) Follow existing design patterns and components
  - B) Requires new UI components or patterns
  - C) Mobile-responsive considerations
  - D) Accessibility requirements
  - E) Real-time updates or interactive elements

* **Performance Considerations:** "Are there specific performance requirements?"
  - A) Standard performance expectations
  - B) High-throughput or high-volume requirements
  - C) Real-time or low-latency needs
  - D) Offline capability requirements
  - E) Caching or optimization needs

* **Security and Permissions:** "What security considerations apply to this feature?"
  - A) Public feature (no authentication required)
  - B) User-specific feature (standard authentication)
  - C) Role-based access control needed
  - D) Admin-only or restricted access
  - E) Special security requirements (encryption, compliance)

## Feature PRD Structure

The generated Feature PRD should include the following sections:

1. **Feature Overview:**
   - Feature name and brief description
   - Problem statement specific to this feature
   - Feature goals and user value proposition
   - Connection to overall project objectives

2. **User Stories & Scenarios:**
   - Primary user stories with acceptance criteria
   - Secondary user scenarios
   - Edge cases and error scenarios
   - User journey flows specific to this feature

3. **Functional Requirements:**
   - Specific functionalities this feature must provide (numbered list)
   - Input and output specifications
   - Business logic and validation rules
   - Integration requirements with existing features

4. **User Experience Requirements:**
   - UI/UX specifications and guidelines
   - Interaction patterns and user flows
   - Responsive design considerations
   - Accessibility requirements
   - Reference to established design system (from ADR)

5. **Data Requirements:**
   - Data models and relationships needed
   - Data validation and constraints
   - Data persistence and retrieval needs
   - Data migration considerations (if applicable)

6. **Technical Constraints:**
   - Reference to relevant ADR decisions and standards
   - Technology stack constraints from project decisions
   - Performance and scalability requirements
   - Security and compliance requirements

7. **API/Integration Specifications:**
   - External API requirements
   - Internal API endpoints needed
   - Data exchange formats and protocols
   - Authentication and authorization requirements

8. **Non-Functional Requirements:**
   - Performance expectations and benchmarks
   - Scalability requirements
   - Reliability and availability needs
   - Security and privacy considerations

9. **Feature Boundaries (Non-Goals):**
   - What this feature explicitly will NOT include
   - Future enhancements that are out of scope
   - Related features that are handled separately
   - Technical limitations and accepted trade-offs

10. **Dependencies:**
    - Dependencies on other features or system components
    - External service or library dependencies
    - Data or infrastructure dependencies
    - Timeline dependencies and prerequisites

11. **Success Criteria:**
    - Quantitative success metrics
    - User satisfaction indicators
    - Performance benchmarks
    - Completion and acceptance criteria

12. **Testing Requirements:**
    - Unit testing expectations
    - Integration testing scenarios
    - User acceptance testing criteria
    - Performance testing requirements

13. **Implementation Considerations:**
    - Complexity assessment and risk factors
    - Recommended implementation approach (high-level)
    - Potential technical challenges
    - Resource and timeline estimates

14. **Open Questions:**
    - Remaining questions needing clarification
    - Technical decisions requiring further research
    - Business decisions pending stakeholder input
    - Design decisions requiring user feedback

## Target Audience

Assume the primary reader of the Feature PRD is a **junior developer** who will implement this feature within the established project architecture. Requirements should be explicit, unambiguous, and reference the established technical standards from the ADR.

## Output

* **Format:** Markdown (`.md`)
* **Location:** `/tasks/`
* **Filename:** `prd-[feature-name].md`

## Integration with Project Context

The Feature PRD should:
- **Reference the Project PRD** for overall context and goals
- **Adhere to the ADR** for technical standards and constraints
- **Be consistent** with other feature PRDs in the project
- **Provide sufficient detail** for TDD and TID creation

## Final Instructions

1. Do NOT start implementing the PRD or creating technical documents
2. Make sure to ask feature-specific clarifying questions
3. Review the Project PRD and ADR to understand context and constraints
4. Take the user's answers to create a comprehensive, implementable Feature PRD
5. Ensure the PRD provides clear requirements for the subsequent TDD/TID/Tasks workflow