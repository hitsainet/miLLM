# QA Engineer Agent

**Role:** Code quality, testing strategy, standards enforcement
**Focus:** Quality gates, testing coverage, security, performance

## Current Analysis

### Code Quality Assessment  
**Last Reviewed:** [DATE]
**Overall Quality:** [EXCELLENT/GOOD/FAIR/POOR]

**Standards Compliance:**
- [ ] Coding conventions followed consistently
- [ ] Error handling implemented properly  
- [ ] Security best practices applied
- [ ] Performance considerations addressed
- [ ] Code is maintainable and readable

**Quality Metrics:**
- **Code Consistency:** [HIGH/MEDIUM/LOW]
- **Error Handling:** [COMPREHENSIVE/ADEQUATE/LACKING]
- **Security Score:** _/10
- **Performance Score:** _/10

### Testing Strategy
**Test Coverage Goals:**
- Unit Tests: _% target, _% actual
- Integration Tests: _% target, _% actual  
- End-to-End Tests: _% target, _% actual

**Testing Health:**
- [ ] Test structure organized and maintainable
- [ ] Tests run automatically and reliably
- [ ] Test data managed properly
- [ ] Edge cases covered in tests

### Current Issues
**Blocking Issues:**
1. 
2. 

**Critical Issues:**
1. 
2. 

**Quality Improvements:**
1. 
2. 

### Session Context
**Current Review Scope:** [What's being reviewed]
**Quality Gates Status:** [PASSING/FAILING]
**Next QA Actions:**
1. 
2. 

---

## Usage with Claude Code

### Loading This Agent Context
```markdown
@.claude/context/agents/qa_engineer.md
@CLAUDE.md
@0xcc/adrs/000_PADR|[project-name].md
```

### Integration with Commands
- Primary agent for `/analyze quality [scope]`
- Essential for `/review` workflows
- Update before and after test execution
- Include in quality-focused `/collaborate` sessions

### Agent Activation Phrase
"Load the QA Engineer agent context for code quality analysis and testing strategy"

### Best Used For
- Code review assessments
- Testing coverage analysis
- Quality gate evaluations
- Performance and security reviews

### Integration with Testing Tools
- Works with project test commands from CLAUDE.md
- References test standards from ADR
- Tracks coverage metrics and quality trends

---
*Update this context during code reviews and quality assessments*
