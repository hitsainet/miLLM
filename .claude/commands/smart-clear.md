# /smart-clear - Intelligent Context Management

## Pre-Clear Context Capture

### Step 1: Update Current Session Context
```
@CLAUDE.md
```
Update the "Current Status" section with:
- What you're currently working on
- Next immediate steps
- Any blockers or important decisions

### Step 2: Update Agent Context
Load and update relevant agent files:
```
@.claude/context/agents/product_engineer.md
@.claude/context/agents/qa_engineer.md
@.claude/context/agents/architect.md  
@.claude/context/agents/test_engineer.md
```

### Step 3: Create Session Snapshot
```
@.claude/context/sessions/session_template.md
```
Copy template to new session file with current timestamp.

### Step 4: Update Health Status (if needed)
```
@.claude/context/health/dashboard.md
```

## Post-Clear Resume Sequence

After using `/clear`, load files in this order:

### Essential Context
```
@CLAUDE.md
@.claude/context/health/dashboard.md
```

### Current Work Context
```
@.claude/context/sessions/[latest-session-file].md
```

### Agent Context (as needed)
```
@.claude/context/agents/product_engineer.md
@.claude/context/agents/qa_engineer.md
@.claude/context/agents/architect.md
@.claude/context/agents/test_engineer.md
```

### Project Context (if using XCC)
```
@!xcc/tasks/[current-task-file].md
@!xcc/prds/[current-prd-file].md
```

## What This Accomplishes
- ✅ Preserves all important context in Markdown files
- ✅ No scripts to fail or dependencies to break
- ✅ Human-readable context that can be edited
- ✅ Structured approach to context management
- ✅ Historical session tracking

## Session Template Location
The session template is at `.claude/context/sessions/session_template.md`
