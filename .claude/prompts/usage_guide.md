# Essential Prompts Usage Guide

## The Five Expert Analysis Templates

### 1. Product Context Analysis
**File:** `.claude/prompts/analysis/product/context_analysis_template.md`
**Use for:** Requirements analysis, user story validation, business alignment

### 2. Code Quality Review  
**File:** `.claude/prompts/analysis/quality/code_review_template.md`
**Use for:** Code reviews, quality gates, production readiness

### 3. Architecture Assessment
**File:** `.claude/prompts/analysis/architecture/assessment_template.md` 
**Use for:** System design review, scalability planning, technical debt

### 4. Testing Strategy Analysis
**File:** `.claude/prompts/analysis/testing/strategy_analysis_template.md`
**Use for:** Test coverage analysis, risk assessment, testing strategy

### 5. Integration Review
**File:** `.claude/prompts/analysis/integration/review_template.md`
**Use for:** System integration health, workflow validation, performance

## How to Use

### Step 1: Copy Template
```bash
# Copy template to working file
cp .claude/prompts/analysis/[type]/[template].md analysis_[scope]_$(date +%Y%m%d).md
```

### Step 2: Load Context
```
@CLAUDE.md
@.claude/context/agents/[relevant-agent].md  
@.claude/context/health/dashboard.md
```

### Step 3: Complete Analysis
- Fill in all template sections
- Use objective scoring where provided
- Document specific examples and locations
- Focus on actionable recommendations

### Step 4: Update Agent Contexts
After analysis, update relevant agent files with findings.

### Step 5: Update Project Health
Reflect analysis results in health dashboard.

## Quick Access via /analyze Command

- `/analyze product feature-name` - Product context analysis
- `/analyze quality component-name` - Code quality review
- `/analyze architecture system-part` - Architecture assessment  
- `/analyze testing coverage-area` - Testing strategy analysis
- `/analyze integration workflow-name` - Integration review

## Integration with Agent System

Each analysis type corresponds to agent expertise:
- **Product Analysis** → Update Product Engineer agent context
- **Quality Analysis** → Update QA Engineer agent context
- **Architecture Analysis** → Update Architect agent context
- **Testing Analysis** → Update Test Engineer agent context
- **Integration Analysis** → Update multiple agent contexts

## Best Practices

### Analysis Preparation
- Define scope clearly before starting
- Load all relevant context files
- Set aside adequate time for thoroughness

### During Analysis  
- Be objective in scoring and assessment
- Document specific examples rather than generalities
- Focus on actionable insights over theoretical issues

### Post-Analysis Actions
- Update agent contexts immediately
- Create action items with owners and timelines
- Share findings with stakeholders
- Schedule follow-up analysis as needed

---

The prompt system provides expert-level analysis frameworks that integrate seamlessly with the agent system and project health monitoring - all using pure Markdown templates.
