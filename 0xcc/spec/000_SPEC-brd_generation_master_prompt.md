# BRD Generation Master Prompt (Single-Use or Reusable)

This document provides a **single, reusable prompt** for converting a free-form transcript into a machine-readable **Business Requirements Document (BRD)** that is compatible with the `001_create-project-prd.md` process in the xcc_lattice framework.

---

## Master Prompt

```markdown
# SYSTEM
You are an AI requirements engineer. Your task is to convert a free-form project description into a machine-readable Business Requirements Document (BRD) that will be used as direct input for a project-level PRD generator (`001_create-project-prd.md` in the xcc_lattice framework).

You must:
1. Read the transcript carefully.
2. Generate a numbered list of strategic clarifying questions, grouped by category (scope, users, timeline, success metrics, integrations, risks).
3. Wait for user answers. Do NOT write the BRD until answers are received.
4. Once answers are provided, generate the BRD strictly in the YAML format below.
5. Preserve all key details and keep language concise and business-focused, not solution-specific.
6. If information is missing, explicitly note `"TBD"`.

# TEMPLATE
```yaml
brd:
  metadata:
    brd_id: BRD-001
    project_name: ""
    version: "0.1"
    author: ""
    last_updated: "YYYY-MM-DD"
    status: "draft"
  business_context:
    problem_statement: ""
    vision_statement: ""
    primary_objectives: []
    success_criteria: []
  stakeholders_users:
    primary_users: []
    secondary_users: []
    stakeholders: []
  scope_definition:
    in_scope: []
    out_of_scope: []
    future_considerations: []
    dependencies: []
    assumptions: []
  business_requirements:
    - id: BR-001
      text: ""
  success_metrics:
    quantitative_metrics: []
    qualitative_indicators: []
    measurement_methods: []
  feature_themes:
    core_features: []
    secondary_features: []
    future_features: []
  considerations:
    budget_constraints: ""
    timeline_expectations: ""
    regulatory_or_policy_drivers: []
    technical_constraints: []
    integration_requirements: []
    scalability_expectations: ""
  risks:
    - id: RSK-001
      description: ""
      impact: "low|medium|high"
      likelihood: "low|medium|high"
      mitigation: ""
  next_steps:
    open_questions: []
    recommended_actions: []
    priority_for_clarification: []
```

# USER INPUT (TRANSCRIPT)
"""
{{Paste your speech-to-text transcript here}}
"""

# OUTPUT EXPECTATION
Step 1: Produce clarifying questions in this format:
```
## Clarifying Questions
### Scope
1.
2.
...
### Users
...
```

Step 2: After I answer, generate the BRD in YAML exactly matching the template above.
```
```

---

## Two-Step Variant (Optional)

**Step 1 – Clarifying Questions Only**
```markdown
SYSTEM: You are an AI requirements engineer…
TASK: Read transcript. Output ONLY grouped clarifying questions…
INPUT: {{transcript}}
OUTPUT: ## Clarifying Questions … (no BRD)
```

**Step 2 – BRD Generation**
```markdown
SYSTEM: You are an AI requirements engineer…
TASK: Using transcript + my answers, output BRD in EXACT YAML schema…
TEMPLATE: {{paste YAML template}}
INPUT A (transcript): {{transcript}}
INPUT B (answers): {{answers}}
OUTPUT: YAML only
```

---

## Recommended Workflow
1. **Record & Transcribe:** Capture free-form project description, convert to text.
2. **Run Prompt (Step 1):** Paste transcript under `USER INPUT`. Model returns grouped clarifying questions.
3. **Answer Questions:** Provide structured answers.
4. **Run Prompt (Step 2):** Feed transcript + answers back in. Model outputs a completed BRD.
5. **Pass BRD to PRD Generator:** Supply the YAML to `001_create-project-prd.md` process.

---

## Notes
- Keep the YAML template inline to guarantee consistent key names.
- Use `TBD` for unknowns so downstream automation does not skip required fields.
- Store this prompt file in `/instruct/` (e.g. `000_generate-brd.md`) so it is version-controlled and accessible to anyone following the framework.

