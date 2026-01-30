# /analyze - Deep Analysis Using Expert Prompts

**Usage:** `/analyze [type] [scope]`
**Types:** product, quality, architecture, testing, integration

## Quick Analysis

### Product Context Analysis
```
@.claude/prompts/analysis/product/context_analysis_template.md
```
Copy to `product_analysis_[scope]_[date].md` and complete.

### Code Quality Analysis  
```
@.claude/prompts/analysis/quality/code_review_template.md
```
Copy to `quality_analysis_[scope]_[date].md` and complete.

### Architecture Analysis
```
@.claude/prompts/analysis/architecture/assessment_template.md
```
Copy to `architecture_analysis_[scope]_[date].md` and complete.

### Testing Strategy Analysis
```
@.claude/prompts/analysis/testing/strategy_analysis_template.md
```
Copy to `testing_analysis_[scope]_[date].md` and complete.

### Integration Analysis
```
@.claude/prompts/analysis/integration/review_template.md
```
Copy to `integration_analysis_[scope]_[date].md` and complete.

## Multi-Perspective Analysis

For comprehensive evaluation:
1. Load all five prompt templates
2. Complete systematic analysis from each perspective
3. Update relevant agent contexts with findings
4. Prioritize actions by impact and effort

## What This Accomplishes
- ✅ Expert-level systematic analysis
- ✅ Comprehensive quality assessment
- ✅ Actionable recommendations with priorities
- ✅ Integration with agent system
- ✅ Historical tracking of analysis results
