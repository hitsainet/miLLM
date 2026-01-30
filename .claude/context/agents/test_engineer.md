# Test Engineer Agent

**Role:** Testing strategy, troubleshooting, reliability engineering
**Focus:** Test coverage, automation, debugging, system reliability

## Current Analysis

### Testing Health Assessment
**Last Assessed:** [DATE] 
**Testing Maturity:** [COMPREHENSIVE/GOOD/BASIC/MINIMAL]

**Coverage Analysis:**
- **Unit Tests:** _% coverage (Target: _%)
- **Integration Tests:** _% coverage (Target: _%)
- **End-to-End Tests:** _% coverage (Target: _%)
- **Performance Tests:** [YES/NO/PLANNED]
- **Security Tests:** [YES/NO/PLANNED]

**Test Quality Metrics:**
- **Test Reliability:** [HIGH/MEDIUM/LOW]
- **Test Speed:** [FAST/ACCEPTABLE/SLOW]
- **Maintenance Burden:** [LOW/MEDIUM/HIGH]

### Testing Strategy
**Current Focus:**
- [ ] Unit test expansion for core logic
- [ ] Integration test implementation
- [ ] End-to-end workflow testing
- [ ] Performance benchmarking
- [ ] Security vulnerability testing

**Test Automation Status:**
- **CI/CD Integration:** [FULL/PARTIAL/NONE]
- **Automated Test Runs:** [YES/NO]
- **Test Reporting:** [COMPREHENSIVE/BASIC/NONE]

### Risk Assessment
**High-Risk Areas:**
1. **[COMPONENT]:** [Risk description and testing approach]
2. **[COMPONENT]:** [Risk description and testing approach]

**Testing Gaps:**
1. 
2. 
3. 

### Troubleshooting & Debugging
**Current Issues:**
- **P0 (Critical):** 
- **P1 (High):** 
- **P2 (Medium):** 

**Debugging Tools Status:**
- [ ] Logging comprehensive and structured
- [ ] Error tracking implemented
- [ ] Performance monitoring active
- [ ] Test environment stable

### Session Context
**Current Testing Focus:** [What you're testing]
**Test Development Status:** [What tests you're writing/updating]
**Next Testing Actions:**
1. 
2. 

---

## Usage with Claude Code

### Loading This Agent Context
```markdown
@.claude/context/agents/test_engineer.md
@CLAUDE.md
@0xcc/adrs/000_PADR|[project-name].md
```

### Integration with Commands
- Primary agent for `/analyze testing [scope]`
- Essential for comprehensive `/review` workflows
- Update during test development and debugging sessions
- Include in reliability-focused `/collaborate` sessions

### Agent Activation Phrase
"Load the Test Engineer agent context for testing strategy and reliability analysis"

### Best Used For
- Test coverage analysis and planning
- Debugging and troubleshooting issues
- Performance and reliability assessments
- Risk assessment and mitigation planning

### Integration with Development Workflow
- Works with test commands defined in CLAUDE.md
- Coordinates with QA Engineer agent on quality aspects
- References testing standards from ADR
- Tracks testing metrics and reliability trends

### Hardcore Debugging Mode
When complex issues arise, this agent can work with the hardcore-debugger agent:
```markdown
@.claude/agents/hardcore-debugger.md
@.claude/context/agents/test_engineer.md
```

---
*Update this context when working on testing and quality assurance*
