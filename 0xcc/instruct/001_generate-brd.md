# Rule: Generating a Business Requirements Document (BRD) — Step 1 of the XCC Workflow

## Goal

To guide a developer through initial repo setup for a brand-new project, then guide an AI assistant in converting free-form project description material into a structured, machine-readable Business Requirements Document (BRD). The BRD becomes the direct input to the Project PRD step (`0xcc/instruct/002_create-project-prd.md`).

This is **step 1** of the XCC framework workflow — the first thing a developer does after cloning the template, before any other instruction document.

## Repo Setup (Before Running This Prompt)

1. Clone the XCC template repo.
2. Delete the template's git history: `rm -rf .git` (or the PowerShell equivalent, `Remove-Item -Recurse -Force .git`).
3. Rename the root folder to your project's name.
4. Initialize a fresh git repository: `git init`.
5. Gather your source material: a free-form transcript (e.g. a speech-to-text recording of you describing the project and its requirements), plus any other documents that inform the business requirements — research notes, existing specs, meeting notes, competitor analysis, etc.

Once the source material is ready, run this prompt.

## Process

1. **Gather Source Material:** Collect the free-form transcript and any supporting documents (research, notes, existing requirements, etc.).
2. **Ask Clarifying Questions:** The AI must first read all source material, then generate a numbered list of strategic clarifying questions grouped by category (scope, users, timeline, success metrics, integrations, risks). Do NOT write the BRD until the user answers.
3. **Generate the BRD:** Once answers are provided, produce the BRD strictly in the YAML format below, preserving key details in concise, business-focused (not solution-specific) language. Mark missing information as `"TBD"`.
4. **Save the BRD:** Save the generated document as `000_BRD|[project-name].md` inside the `0xcc/prds/` directory.

## BRD Template

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

## Two-Step Variant (Optional)

Some users prefer to split clarifying questions and BRD generation into two separate turns:

**Step 1 – Clarifying Questions Only**
```
TASK: Read all source material. Output ONLY grouped clarifying questions (no BRD).
```

**Step 2 – BRD Generation**
```
TASK: Using the source material plus the user's answers, output the BRD in the exact YAML schema above. Output YAML only.
```

## Output

* **Format:** YAML embedded in Markdown (`.md`)
* **Location:** `0xcc/prds/`
* **Filename:** `000_BRD|[project-name].md`

## Final Instructions

1. Do NOT generate the BRD until the user has answered the clarifying questions.
2. Keep the YAML schema exact — the next step (`002_create-project-prd.md`) expects these keys.
3. Use `"TBD"` for unknowns rather than omitting fields or guessing.
4. After saving, record the document in AGENT.md: flip its Document Inventory entry to ✅ and update Current Status/Next Steps.
5. Once the BRD is saved, proceed to `0xcc/instruct/002_create-project-prd.md`, referencing the BRD file as context for the Project PRD.
