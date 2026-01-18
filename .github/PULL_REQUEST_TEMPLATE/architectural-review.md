<!--
This PR template is for architectural review submissions.
For regular code changes, delete this template and use the default.
-->

## Architectural Review Submission

**Review Type**: [Code Implementation / System Architecture / Testing / Documentation / Performance]
**Priority**: [HIGH / MEDIUM / LOW]
**Review Period**: [YYYY-MM]

---

### 📄 Review Document
- **File**: `docs/reviews/YYYY-MM/[NAME]-REVIEW.md`
- **Related Issue**: Closes #XXX
- **Tracking Index**: `docs/reviews/YYYY-MM/REVIEW-TRACKING-INDEX.md`

### 📋 Executive Summary
<!-- 2-3 sentence overview of what this review documents -->



### 🎯 Scope
**Systems Affected**:
- [ ] Core solvers (Stokes, Poisson, Advection-Diffusion)
- [ ] Variable system (Mesh/Swarm variables, arrays)
- [ ] Units & non-dimensionalization
- [ ] Parallel operations (MPI, collective operations)
- [ ] Testing infrastructure
- [ ] Documentation & examples
- [ ] Other: _________________

**Files Changed**:
- New files: [count]
- Modified files: [count]
- Total LOC: ~[count]

### ✅ Testing Status
- [ ] All tests passing: `pixi run underworld-test`
- [ ] New tests added: [count] tests
- [ ] Test coverage: [X]% (or N/A)
- [ ] Regression tests validated
- [ ] Performance benchmarks run (if applicable)

**Test Results**:
```bash
# Command used
pixi run -e default pytest tests/test_XXXX*.py -v

# Results summary
[X] passed, [Y] failed, [Z] skipped
```

### 📊 Key Metrics & Impact
<!-- Quantify the changes: performance, coverage, complexity -->

**Performance**:
- Before: [metric]
- After: [metric]
- Improvement: [X]x faster / [Y]% reduction

**Quality**:
- Test coverage: [X]%
- Code complexity: [increase/decrease/unchanged]
- Documentation: [X pages / X examples]

### 🔍 Review Checklist (for Reviewers)

**Design & Architecture**:
- [ ] Design rationale is clear and well-justified
- [ ] Trade-offs are documented with alternatives considered
- [ ] System architecture is comprehensible
- [ ] Integration points are identified

**Implementation**:
- [ ] Implementation matches documented design
- [ ] Code quality meets project standards
- [ ] Breaking changes are identified and justified
- [ ] Backward compatibility is addressed

**Testing & Validation**:
- [ ] Testing strategy is adequate
- [ ] Test coverage is sufficient
- [ ] Edge cases are covered
- [ ] Performance impact is assessed

**Documentation**:
- [ ] Known limitations are clearly documented
- [ ] Benefits are quantified
- [ ] User-facing changes are documented
- [ ] Migration guide provided (if needed)

### ⚠️ Known Limitations
<!-- List any current constraints or future work needed -->

1.
2.

### 🔗 Dependencies & Related Work
**Depends on**:
- [ ] Review #XXX (must be approved first)

**Blocks**:
- [ ] Review #YYY (waiting on this review)

**Related**:
- Discussion: [Link to GitHub Discussion]
- Prior art: [Links to related reviews]

---

## 🖊️ Sign-Off

**This PR merge represents formal approval of the architectural review.**

Reviewers: Please review the full document at `docs/reviews/YYYY-MM/[NAME]-REVIEW.md` and approve this PR only when satisfied with the design, implementation, and documentation.

| Role | GitHub Handle | Sign-Off Date | Status |
|------|---------------|---------------|--------|
| **Author** | @username | YYYY-MM-DD | ✅ Submitted |
| **Primary Reviewer** | @reviewer1 | | ⏳ Pending |
| **Secondary Reviewer** | @reviewer2 | | ⏳ Pending |
| **Project Lead** | @lead | | ⏳ Pending |

---

### 📝 Additional Context
<!-- Optional: Any other information reviewers should know -->



<!--
REVIEW HISTORY:
- Review #10 (2025-11-17): Template created as part of Review System Infrastructure review
-->
