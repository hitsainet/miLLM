# Rule: Generating a Project-Level Product Requirements Document (PRD)

## Goal

To guide an AI assistant in creating a high-level Product Requirements Document (PRD) in Markdown format that captures the overall project vision, goals, and breaks down the project into discrete features. This serves as the foundation for all subsequent feature-level development work.

If a Business Requirements Document exists at `0xcc/prds/000_BRD|[project-name].md` (produced by `0xcc/instruct/001_generate-brd.md`), use it as the primary source of project context instead of asking the user to restate everything from scratch.

## Process

1. **Receive Project Description:** Use the BRD (`0xcc/prds/000_BRD|[project-name].md`) if one exists; otherwise the user provides a high-level description of the project they want to build.
2. **Ask Strategic Clarifying Questions:** Before writing the Project PRD, the AI *must* ask clarifying questions to understand the project scope, goals, and feature breakdown. **Skip questions the BRD already answers** — ask only about what is new, ambiguous, or marked TBD. Provide options in letter/number lists for easy selection.
3. **Generate Project PRD:** Create a comprehensive project-level document using the structure outlined below.
4. **Save Project PRD:** Save the generated document as `000_PPRD|[project-name].md` inside the `0xcc/prds/` directory.

## Strategic Clarifying Questions (Examples)

The AI should adapt questions based on the project description, focusing on high-level strategy:

* **Project Scope:** "What is the primary scope of this project?"
  - A) MVP/Proof of concept
  - B) Full-featured application
  - C) Enterprise-level system
  - D) Research/experimental project

* **Target Users:** "Who are the primary users of this system?"
  - A) End consumers (B2C)
  - B) Business users (B2B)
  - C) Internal team/organization
  - D) Developers/technical users
  - E) Mixed user base

* **Project Timeline:** "What is the expected development timeline?"
  - A) Rapid prototype (weeks)
  - B) Standard development (months)
  - C) Long-term project (6+ months)
  - D) Ongoing/iterative development

* **Success Criteria:** "How will project success be measured?"
  - A) User adoption metrics
  - B) Business revenue/ROI
  - C) Technical performance metrics
  - D) User satisfaction/feedback
  - E) Internal efficiency gains

* **Integration Requirements:** "Does this project need to integrate with existing systems?"
  - A) Standalone system
  - B) Replace existing system
  - C) Integrate with current tools
  - D) Part of larger ecosystem

* **Scalability Expectations:** "What are the expected scale requirements?"
  - A) Small user base (<1000 users)
  - B) Medium scale (1000-10000 users)
  - C) Large scale (10000+ users)
  - D) Unknown/variable scale

## Project PRD Structure

The generated Project PRD should include the following sections:

1. **Project Overview:**
   - Project name and brief description
   - Vision statement and primary objectives
   - Problem statement and opportunity
   - Success definition and key outcomes

2. **Project Goals & Objectives:**
   - Primary business goals
   - Secondary objectives
   - Success metrics and KPIs
   - Timeline and milestone expectations

3. **Target Users & Stakeholders:**
   - Primary user personas and needs
   - Secondary users and use cases
   - Key stakeholders and their interests
   - User journey overview

4. **Project Scope:**
   - What is included in this project
   - What is explicitly out of scope
   - Future roadmap considerations
   - Dependencies and assumptions

5. **High-Level Requirements:**
   - Core functional requirements across the project
   - Non-functional requirements (performance, security, etc.)
   - Compliance and regulatory considerations
   - Integration and compatibility requirements

6. **Feature Breakdown:**
   - **Core Features** (MVP/essential functionality)
     - Feature A: [Brief description and user value]
     - Feature B: [Brief description and user value]
   - **Secondary Features** (important but not critical)
     - Feature C: [Brief description and user value]
   - **Future Features** (nice-to-have/roadmap items)
     - Feature D: [Brief description and user value]

7. **User Experience Goals:**
   - Overall UX principles and guidelines
   - Accessibility requirements
   - Performance expectations
   - Cross-platform considerations

8. **Business Considerations:**
   - Budget and resource constraints
   - Risk assessment and mitigation
   - Competitive landscape awareness
   - Monetization or value creation model

9. **Technical Considerations (High-Level):**
   - Deployment environment preferences
   - Security and privacy requirements
   - Performance and scalability needs
   - Integration and API requirements
   - **Note:** Detailed tech stack decisions will be made in subsequent Architecture Decision Record

10. **Project Constraints:**
    - Timeline constraints
    - Budget limitations
    - Resource availability
    - Technical or regulatory constraints

11. **Success Metrics:**
    - Quantitative success measures
    - Qualitative success indicators
    - User satisfaction metrics
    - Business impact measurements

12. **Next Steps:**
    - Immediate next actions
    - Architecture and tech stack evaluation needs
    - Feature prioritization approach
    - Resource and timeline planning

## Feature Description Guidelines

For each feature in the breakdown:
- **Keep descriptions concise** (2-3 sentences max)
- **Focus on user value** rather than technical implementation
- **Include rough priority level** (Core/Secondary/Future)
- **Note dependencies** between features if obvious
- **Avoid technical implementation details** (those come later in feature PRDs)

## Target Audience

The Project PRD should be understandable by:
- **Business stakeholders** who need to understand project value
- **Product managers** who will oversee feature development
- **Technical leads** who will make architecture decisions
- **Development teams** who need project context
- **QA and design teams** who need to understand overall goals

## Output

* **Format:** Markdown (`.md`)
* **Location:** `0xcc/prds/`
* **Filename:** `000_PPRD|[project-name].md`

## Final Instructions

1. Do NOT start creating feature-level PRDs or technical documents
2. Focus on strategic, high-level planning rather than implementation details
3. Ask clarifying questions to understand the project vision and scope
4. Create a comprehensive project foundation that will guide all subsequent development work
5. Ensure the feature breakdown provides clear guidance for the next phase of detailed feature PRD creation
6. After saving, record the document in AGENT.md: flip its Document Inventory entry to ✅ and update Current Status/Next Steps