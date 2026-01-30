# /review - Multi-Agent Code Review

**Usage:** `/review [scope]`
**Scopes:** feature, code, architecture, tests, all

## Multi-Agent Review Process

### Step 1: Load Agent Contexts
```
@.claude/context/agents/product_engineer.md
@.claude/context/agents/qa_engineer.md
@.claude/context/agents/architect.md
@.claude/context/agents/test_engineer.md
```

### Step 2: Review Coordination

**Product Engineer Review:**
- Requirements alignment check
- User story coverage validation
- Business logic correctness
- Context completeness assessment

**QA Engineer Review:**  
- Code quality standards compliance
- Error handling adequacy
- Security best practices
- Testing coverage analysis

**Architect Review:**
- Design pattern consistency
- System integration quality
- Scalability implications
- Technical debt assessment

**Test Engineer Review:**
- Test strategy effectiveness
- Coverage gap identification
- Risk area analysis
- Debugging capability assessment

### Step 3: Review Documentation

Create review session file:
```
@.claude/context/sessions/session_template.md
```
Copy to `review_[scope]_[date].md`

### Step 4: Update Agent Contexts

After review, update each agent's .md file with:
- Findings from their perspective
- Recommendations for improvement
- Priority levels for issues found
- Next review focus areas

### Step 5: Health Impact Assessment

Update project health based on review findings:
```
@.claude/context/health/dashboard.md
```

## Review Scopes

**Feature Review:**
- Focus on specific feature implementation
- Cross-agent validation of feature requirements
- Feature-level quality gates

**Code Review:**
- Broad code quality assessment
- Standards compliance check
- Technical debt identification

**Architecture Review:**
- System design consistency
- Integration patterns
- Scalability bottlenecks

**Test Review:**
- Testing strategy effectiveness
- Coverage analysis
- Risk assessment

**All (Comprehensive Review):**
- Complete multi-agent analysis
- Cross-cutting concern identification
- Full project health assessment

## What This Accomplishes
- ✅ Systematic multi-perspective code review
- ✅ Agent-specific expertise application
- ✅ Structured review documentation
- ✅ Continuous project health monitoring
- ✅ Pure Markdown workflow - no scripts
