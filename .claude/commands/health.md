# /health - Project Health Assessment

## Quick Health Check

Load and review these files to assess project health:

1. **Load Health Dashboard:**
   ```
   @.claude/context/health/dashboard.md
   ```

2. **Review Agent Status:**
   ```
   @.claude/context/agents/product_engineer.md
   @.claude/context/agents/qa_engineer.md  
   @.claude/context/agents/architect.md
   @.claude/context/agents/test_engineer.md
   ```

3. **Analyze Current State:**
   - Check git status: `git status`
   - Review project structure: `ls -la`
   - Check documentation: `ls *.md`

## Health Assessment Workflow

### Step 1: Load Assessment Template
```
@.claude/context/health/assessment_template.md
```

### Step 2: Fill Out Assessment
Copy the template to a new file and complete the assessment:
```bash
cp .claude/context/health/assessment_template.md .claude/context/health/assessment_$(date +%Y%m%d).md
```

### Step 3: Update Dashboard
After completing assessment, update:
```
@.claude/context/health/dashboard.md
```

## What This Command Does
- Provides structured approach to health assessment
- Uses only file loading and Markdown templates
- No scripts required - pure Claude interaction
- Creates historical health tracking

## Quick Status Indicators
- 🟢 **Excellent (18-20/20):** Well-organized, documented, tested
- 🟡 **Good (14-17/20):** Solid foundation, minor improvements needed  
- 🟠 **Fair (10-13/20):** Functional but needs attention
- 🔴 **Needs Attention (6-9/20):** Multiple issues to address
- 💀 **Critical (0-5/20):** Requires immediate restructuring
