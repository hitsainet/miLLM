# /collaborate - Multi-Agent Collaboration

**Usage:** `/collaborate [challenge-type] [description]`
**Types:** design, problem-solving, optimization, debugging

## Collaboration Workflow

### Step 1: Define Challenge

Create collaboration session:
```
@.claude/context/sessions/session_template.md
```
Copy to `collaboration_[type]_[date].md` and document:
- Challenge description
- Desired outcome
- Constraints or requirements
- Timeline expectations

### Step 2: Agent Consultation

Load each agent for their perspective:

**Product Engineer Input:**
```
@.claude/context/agents/product_engineer.md
```
Focus on:
- Business impact analysis
- User experience implications
- Requirements clarification
- Product vision alignment

**QA Engineer Input:**
```
@.claude/context/agents/qa_engineer.md
```
Focus on:
- Quality implications
- Testing considerations
- Risk assessment
- Standards impact

**Architect Input:**
```
@.claude/context/agents/architect.md
```
Focus on:
- Technical approach options
- System design implications
- Integration considerations
- Scalability impact

**Test Engineer Input:**
```
@.claude/context/agents/test_engineer.md
```
Focus on:
- Testing strategy implications
- Debugging considerations
- Reliability impact
- Risk mitigation

### Step 3: Synthesis and Decision

Document in collaboration session file:
- Each agent's input and recommendations
- Conflicting viewpoints and resolutions
- Consensus decisions
- Implementation plan
- Success criteria

### Step 4: Update Project Context

Update relevant files based on decisions:
- `CLAUDE.md` - Update project direction
- Agent files - Update with decision outcomes
- Health dashboard - Note any health impacts

## Collaboration Types

**Design Challenges:**
- System architecture decisions
- User interface design choices
- Data model design
- Integration approach selection

**Problem-Solving:**
- Bug resolution strategies
- Performance issue diagnosis
- Technical roadblock solutions
- Process improvement initiatives

**Optimization:**
- Performance enhancement strategies
- Code quality improvement plans
- Workflow efficiency improvements
- Resource utilization optimization

**Debugging:**
- Complex issue diagnosis
- Root cause analysis
- Solution strategy development
- Prevention strategy planning

## What This Accomplishes
- ✅ Structured multi-agent problem solving
- ✅ Documented decision-making process
- ✅ Preserved reasoning and rationale
- ✅ Cross-functional perspective integration
- ✅ No external tools or scripts required
