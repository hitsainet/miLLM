# /feature - Feature Development Workflow

**Usage:** `/feature [feature-name] [mode]`
**Modes:** quick, standard, comprehensive

## Feature Development Workflow

### Step 1: Agent Coordination Setup

Load and update agent contexts for the feature:

```
@.claude/context/agents/product_engineer.md
```
Update with:
- Feature requirements analysis
- User story breakdown
- Business logic requirements

```
@.claude/context/agents/architect.md  
```
Update with:
- System design approach
- Integration points
- Architecture decisions

```
@.claude/context/agents/qa_engineer.md
```
Update with:
- Quality requirements
- Testing strategy
- Acceptance criteria

```
@.claude/context/agents/test_engineer.md
```
Update with:
- Test planning approach
- Coverage targets
- Risk assessment

### Step 2: XCC Framework Integration (if available)

If XCC framework exists:
```
@!xcc/instruct/003_create-feature-prd.md
@!xcc/instruct/004_create-tdd.md
@!xcc/instruct/005_create-tid.md
@!xcc/instruct/006_generate-tasks.md
```

### Step 3: Feature Session Tracking

```
@.claude/context/sessions/session_template.md
```
Copy to `feature_[name]_[date].md` and update with feature-specific context.

### Step 4: Development Mode Configuration

**Quick Mode:**
- Focus on Product Engineer and basic Architect input
- Minimal documentation requirements
- Rapid prototyping approach

**Standard Mode:**
- All four agents provide input
- Standard documentation and testing
- Balanced approach

**Comprehensive Mode:**
- Deep agent analysis and collaboration
- Extensive documentation and testing
- Enterprise-grade development approach

## Feature Context Management

Each feature gets:
1. **Agent Analysis:** All agents provide perspective
2. **Session Tracking:** Feature-specific session files
3. **Health Monitoring:** Feature impact on project health
4. **Documentation:** Integrated with XCC or standalone docs

## What This Accomplishes
- ✅ Multi-agent feature development coordination
- ✅ Structured approach based on development mode
- ✅ Integration with existing project frameworks
- ✅ Complete context preservation for features
- ✅ No external dependencies or scripts
