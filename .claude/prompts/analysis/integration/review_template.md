# Integration Review Template

**Review Date:** [DATE]
**Reviewer:** [YOUR NAME]
**Integration Scope:** [System/Service/Component integrations being reviewed]

## Cross-Component Integration Analysis

### Component Communication Patterns
- [ ] Communication patterns consistent across similar integrations
- [ ] Data serialization/deserialization efficient
- [ ] Protocol selection justified and documented
- [ ] Message schemas versioned and backward compatible

**Communication Quality Score:** ___/10

### Data Flow and Transformation
- [ ] Data transformations preserve semantic meaning
- [ ] Data validation comprehensive at integration boundaries
- [ ] Data consistency maintained across integrations
- [ ] Data lineage traceable for auditing

**Data Flow Score:** ___/10

## External System Integration

### Third-Party Service Integration
**External Dependencies:**
1. **Service:** [external service name]
   **Purpose:** [why integrated]
   **Reliability:** [SLA/uptime expectations]
   **Fallback Strategy:** [backup plan]
   **Security:** [authentication/data protection]

**External Integration Score:** ___/10

### Integration Resilience
- [ ] Circuit breaker patterns implemented
- [ ] Timeout and retry logic configured
- [ ] Rate limiting respected and handled
- [ ] Graceful degradation when services unavailable

**Resilience Score:** ___/10

## End-to-End Workflow Validation

### User Journey Implementation
**Primary Workflows:**
1. **Workflow:** [end-to-end user journey]
   **Systems Involved:** [list of systems]
   **Integration Points:** [where systems connect]
   **Error Handling:** [how failures handled]

**Workflow Integration Score:** ___/10

## Integration Quality Gates

### System-Breaking Issues
1. **Issue:** [critical integration failure]
   **Systems Affected:** [what stops working]
   **Fix:** [immediate resolution needed]

### User-Impacting Issues
1. **Issue:** [integration problem affecting users]
   **Impact:** [how UX degraded]
   **Fix:** [proper resolution]

### Performance Issues  
1. **Bottleneck:** [performance issue]
   **Current Performance:** [baseline]
   **Target:** [required performance]
   **Optimization:** [how to improve]

## Overall Assessment
- **Communication Patterns:** ___/10
- **Data Flow:** ___/10
- **External Integration:** ___/10
- **Resilience:** ___/10
- **Workflow Integration:** ___/10

**Overall Integration Health:** ___/50
**Integration Maturity:** [SEAMLESS/SOLID/FUNCTIONAL/FRAGILE/BROKEN]

---
*Focus on seamless system integration that delivers reliable user experiences and operational excellence.*
