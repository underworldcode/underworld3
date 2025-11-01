# Underworld3 Code Review Process

**Status**: Active
**Version**: 1.0
**Last Updated**: 2025-10-25
**Audience**: Developers, Code Reviewers, Project Maintainers

## Overview

This document defines the formal code review process for Underworld3 contributions, including implementation work, documentation, and tests. The process ensures quality, correctness, and maintainability while maintaining efficient workflows.

## Purpose and Goals

The code review process serves to:

1. **Ensure Correctness**: Verify that implementations work as intended and don't introduce bugs
2. **Validate Documentation**: Confirm that user-facing documentation is accurate and helpful
3. **Maintain Consistency**: Ensure code style, patterns, and conventions are followed
4. **Catch Regressions**: Identify potential impacts on existing functionality
5. **Knowledge Transfer**: Enable team members to learn from each other's work
6. **Create Accountability**: Document decisions and sign-offs for future reference

## Review Categories and Scope

### 1. Code Implementation Reviews

**When Applied To**:
- New features and functionality
- Bug fixes and patches
- Refactoring and optimization
- Solver modifications and enhancements
- Performance improvements

**Review Checklist** (See template below)
- [ ] Does the code implement the intended functionality?
- [ ] Are all edge cases handled?
- [ ] Does it follow Underworld3 coding conventions?
- [ ] Are there any performance concerns?
- [ ] Does it maintain backward compatibility (if required)?
- [ ] Are corresponding tests included and passing?

### 2. Documentation Reviews

**When Applied To**:
- User guides and tutorials
- Technical documentation
- API documentation and docstrings
- Migration guides and breaking change notices
- Advanced feature documentation

**Review Checklist** (See template below)
- [ ] Is the documentation accurate and complete?
- [ ] Are examples working and tested?
- [ ] Is the language clear and appropriate for the audience?
- [ ] Are technical concepts explained clearly?
- [ ] Are caveats and limitations documented?
- [ ] Are cross-references and links current?
- [ ] Is the documentation discoverable and well-indexed?

### 3. Test Coverage Reviews

**When Applied To**:
- New test suites
- Test modifications and improvements
- Regression test additions
- Integration test updates

**Review Checklist** (See template below)
- [ ] Do tests validate the intended functionality?
- [ ] Are test assertions correct and meaningful?
- [ ] Is test coverage adequate for the feature?
- [ ] Do tests follow Underworld3 conventions?
- [ ] Are all tests passing consistently?
- [ ] Do tests handle edge cases and error conditions?
- [ ] Is test documentation clear?

## Review Process Workflow

### Phase 1: Preparation

1. **Author**: Prepare change materials
   - Code changes with clear commits
   - Documentation and docstring updates
   - Test suite with comprehensive coverage
   - Git history showing progression

2. **Author**: Create review package
   - Summary document (see "Review Summary Template" below)
   - Links to code changes (GitHub PR or branch)
   - Links to documentation files
   - Links to test files

3. **Author**: Self-review
   - Walk through changes personally
   - Check for obvious issues
   - Verify tests pass locally
   - Ensure documentation is complete

### Phase 2: Review Assignment

1. **Project Lead**: Assign reviewers
   - Select appropriate reviewers based on expertise
   - Assign 1-2 reviewers minimum
   - Consider code ownership and critical paths

2. **Reviewers**: Confirm and schedule
   - Acknowledge assignment
   - Indicate expected review timeline
   - Flag any conflicts or concerns

### Phase 3: Review Execution

1. **Primary Reviewer**: Conduct thorough review
   - Read all materials (code, docs, tests)
   - Run tests locally if necessary
   - Check against review checklists
   - Document findings and questions

2. **Secondary Reviewer**: Execute focused review
   - Concentrate on areas of concern from primary review
   - Verify critical functionality
   - Check for performance implications

3. **Reviewers**: Document feedback
   - Provide specific, actionable comments
   - Reference line numbers or specific sections
   - Distinguish between blocking issues and suggestions
   - Document approval or requested changes

### Phase 4: Response and Resolution

1. **Author**: Address feedback
   - Respond to each comment
   - Make requested changes
   - Update documentation if needed
   - Re-run tests to verify fixes

2. **Author**: Create response summary
   - Document what was changed
   - Explain why certain suggestions weren't adopted (if applicable)
   - Reference updated files/commits

3. **Reviewers**: Verify resolutions
   - Check that feedback was addressed
   - Approve if satisfied
   - Request additional changes if needed
   - Track remaining concerns

### Phase 5: Sign-Off and Merge

1. **Reviewers**: Final approval
   - Both reviewers sign off
   - Document review completion date and names
   - Flag any remaining caveats or limitations

2. **Project Lead**: Final check
   - Verify review was thorough
   - Check all checklists completed
   - Confirm test suite passing
   - Authorize merge

3. **Author**: Merge and archive
   - Merge to main branch
   - Archive review documentation
   - Update project status if needed
   - Communicate results to team

## Review Documentation

### Folder Structure

```
docs/reviews/
├── 2025-10/
│   ├── REDUCTION-OPERATIONS-REVIEW.md
│   ├── SWARM-INTEGRATION-REVIEW.md
│   └── TEST-COVERAGE-ANALYSIS.md
├── 2025-11/
│   └── [future reviews]
└── README.md                          # Index of all reviews
```

### File Naming Convention

```
[FEATURE-NAME]-REVIEW.md
[COMPONENT]-CODE-REVIEW.md
[SYSTEM]-TEST-ANALYSIS.md
[DOCUMENTATION]-VERIFICATION.md
```

## Review Templates

### Review Summary Template

Create one summary document per feature or significant change:

```markdown
# Code Review Summary: [Feature Name]

## Overview
[1-2 sentence description of what was implemented/changed]

## Changes Made

### Code Changes
- [List of files modified, new files created]
- [Summary of major changes in each file]

### Documentation Changes
- [List of documentation files added/modified]
- [New sections or examples added]

### Test Coverage
- [Test files created/modified]
- [Test count and coverage metrics]

## Review Scope

**Primary Focus Areas**:
- [Area 1 for careful review]
- [Area 2 for careful review]
- [Area 3 for careful review]

**Known Limitations/Caveats**:
- [Limitation 1 with explanation]
- [Limitation 2 with explanation]

## Relevant Resources

- [Link to code changes/PR]
- [Link to main documentation]
- [Link to test suite]
- [Link to related issues]

## Testing Instructions

```bash
[Command to run tests]
[Expected output]
```

## Sign-Off

| Role | Name | Date | Status |
|------|------|------|--------|
| Author | [Name] | [Date] | Submitted |
| Primary Reviewer | [Name] | [Date] | [Approved/Changes Requested] |
| Secondary Reviewer | [Name] | [Date] | [Approved/Changes Requested] |
| Project Lead | [Name] | [Date] | [Merged/Blocked] |

## Review Comments and Resolutions

[Captured via review tools or attached as REVIEW-COMMENTS.md]
```

### Code Review Template

For detailed code reviews:

```markdown
# Code Review: [File Name]

**Reviewer**: [Name]
**Date**: [Date]
**Status**: [In Progress / Complete]

## Summary
[Overview of what this review covers]

## File-by-File Analysis

### [File 1]

**Line [X-Y]: [Section Description]**
```
[Code snippet]
```
- **Finding**: [Issue description]
- **Severity**: [Blocking / Major / Minor / Suggestion]
- **Recommendation**: [What should be changed]
- **Resolution**: [What was changed / status]

### [File 2]

[Similar analysis]

## Testing Verification

- [ ] Tests pass locally: `pixi run -e default pytest [test-files] -v`
- [ ] No new warnings or errors introduced
- [ ] Performance impact assessed (if applicable)
- [ ] Edge cases tested

## Overall Assessment

**Approval Status**: [Approved / Approved with Conditions / Changes Required]

**Rationale**: [Summary of review findings]

**Conditions for Approval** (if applicable):
1. [Condition 1]
2. [Condition 2]

## Reviewer Signature
[Reviewer Name], [Date]
```

### Documentation Verification Template

```markdown
# Documentation Review: [Document Name]

**Reviewer**: [Name]
**Date**: [Date]
**Status**: [Complete]

## Content Verification

- [ ] All technical information is accurate
- [ ] Examples are tested and working
- [ ] Cross-references are current
- [ ] Code snippets follow conventions
- [ ] Caveats and limitations are documented

## Clarity and Usability

- [ ] Language is clear and appropriate for audience
- [ ] Concepts are explained adequately
- [ ] Technical terms are defined or linked
- [ ] Organization and flow are logical
- [ ] Visual aids (if any) are clear and helpful

## Completeness Check

- [ ] All related topics are covered
- [ ] Future improvements/TODOs are noted
- [ ] Edge cases are mentioned
- [ ] Performance considerations noted (if applicable)

## Issues Found

| Line/Section | Issue | Severity | Resolution |
|--------------|-------|----------|-----------|
| [Location] | [Issue] | [Level] | [Status] |

## Approval

**Approved**: [Yes / With Conditions]

**Reviewer**: [Name], [Date]
```

### Test Coverage Analysis Template

```markdown
# Test Coverage Analysis: [Feature/System]

**Analyzer**: [Name]
**Date**: [Date]
**Coverage Target**: [e.g., 90%]

## Test Files Reviewed

| File | Tests | Status | Coverage |
|------|-------|--------|----------|
| [test_file_1.py] | [Count] | ✓ Passing | [%] |
| [test_file_2.py] | [Count] | ✓ Passing | [%] |

## Functionality Coverage

### Feature: [Feature Name]
- **Happy Path**: ✓ [Test name]
- **Error Cases**: ✓ [Test name]
- **Edge Cases**: [Status]

### Feature: [Feature Name]
[Similar breakdown]

## Coverage Assessment

**Total Tests**: [Count]
**Pass Rate**: [%]
**Coverage**: [%]

**Gaps Identified**:
1. [Gap 1 - Test suggestion]
2. [Gap 2 - Test suggestion]

## Performance Testing

- [ ] Benchmarks run for critical paths
- [ ] No regressions detected
- [ ] Scaling behavior verified (if applicable)

## Recommendation

[Assessment and approval status]
```

## Issue Tracking and Resolution

### Feedback Categories

**Blocking Issues** (must resolve before merge)
- Correctness errors or logic bugs
- Test failures
- Documentation that conflicts with implementation
- Security or performance problems

**Major Issues** (should resolve before merge)
- Design improvements
- Code style violations
- Incomplete documentation
- Inadequate test coverage

**Minor Issues** (nice to have)
- Typos and grammar
- Code formatting
- Suggestions for future improvement
- Code comments and clarity

### Response Process

**For Blocking Issues**:
1. Author makes immediate fix
2. Reviewer verifies resolution
3. If not resolved, merge is blocked

**For Major Issues**:
1. Author evaluates feedback
2. Author either makes change or documents rationale
3. Reviewer accepts explanation or requests change

**For Minor Issues**:
1. Author may defer to future work
2. Document as "acknowledged for future improvement"
3. Don't block on resolution

## Best Practices for Reviewers

### Do

✓ Ask clarifying questions if something is unclear
✓ Give specific, actionable feedback with examples
✓ Acknowledge good code and design decisions
✓ Consider reviewer fatigue - don't request perfect polish
✓ Review in timely manner - aim for 48 hours
✓ Provide context for suggestions (explain the "why")
✓ Test the code locally when possible
✓ Check both happy path and error cases

### Don't

✗ Approve without reading all materials
✗ Get caught up in style debates (defer to linter)
✗ Request changes without clear justification
✗ Approve changes you don't fully understand
✗ Mix high-level questions with style comments
✗ Leave reviews in draft form - complete them

## Best Practices for Authors

### Do

✓ Prepare thorough review materials
✓ Include clear summary of changes
✓ Provide testing instructions
✓ Be responsive to feedback
✓ Explain design decisions when questioned
✓ Include documentation and tests with code
✓ Reference related issues and PRs
✓ Test locally before submitting

### Don't

✗ Submit incomplete or untested work
✗ Ignore feedback or dismiss concerns
✗ Merge approved changes without checking
✗ Rush reviews - quality takes time
✗ Submit massive changes (break into smaller parts)
✗ Make code changes after approval

## Emergency Procedures

### Hot-Fix Bypass (Only for Critical Bugs)

In case of critical production bugs:

1. **Rapid assessment** (15 min): Is this truly critical?
2. **Emergency review** (1 lead reviewer, not 2)
3. **Testing**: Thorough validation of fix specifically
4. **Post-merge review**: Full review conducted after merge
5. **Documentation**: Full documentation of emergency and its resolution

**Emergency Fix Log**: `docs/reviews/EMERGENCY-FIXES.md`

This should be exceptionally rare. If using more than once per month, the process has failed.

## Metrics and Reporting

### Review Metrics

Track these to improve the process:

| Metric | Measurement | Target |
|--------|-------------|--------|
| Review Time | Days to complete review | < 2 days |
| Feedback Rounds | Average rounds to approval | < 2 rounds |
| Approval Rate | % of first-submission approvals | > 60% |
| Issues Found | Avg issues per 100 lines | > 0.5 |
| Issue Severity | % blocking issues | < 20% |

### Monthly Review Report

Generate monthly using this template:

```markdown
# Code Review Report: [Month/Year]

**Period**: [Dates]

## Summary
- Total Reviews: [Count]
- Total Changes: [Files, LOC]
- Approval Rate: [%]

## Metrics
[Table of metrics from above]

## Trends
[Notable patterns or improvements]

## Recommendations
[For process improvement]
```

## Escalation Path

If review is stuck:

1. **Disagreement (1 week)**: Author and reviewer discuss directly
2. **Impasse (2 weeks)**: Escalate to project lead
3. **Blocked (3 weeks)**: Project lead makes final decision, documents rationale
4. **Resolution**: Document outcome for future reference

## Integration with CI/CD

### Automated Checks (must pass before review)

```yaml
- Tests: pixi run -e default pytest tests/ -v
- Linting: [configured linter]
- Documentation: Builds without errors
```

### Manual Review (after automated checks pass)

Review process begins only after CI passes.

## Archives and Record Keeping

### Review Document Storage

All review documents stored in `docs/reviews/[YYYY-MM]/`

**Retention**: Permanent (reference and learning)
**Access**: Team members and project history
**Archival**: Annual cleanup of old summaries

### Review History

Maintain index file: `docs/reviews/README.md`

```markdown
# Code Review Archive

## 2025

- **October**: [Reduction Operations](#), [Swarm Integration](#)
- **November**: [Pending]

## Summary Statistics

- Total reviews (2025): [Count]
- Avg review time: [Days]
- Most common issues: [Top 3]
```

## Amendment and Versioning

**This document version**: 1.0
**Effective date**: 2025-10-25
**Last updated**: 2025-10-25

To propose changes to this process:
1. Create issue in project tracking system
2. Discuss proposed changes with team
3. Update document and version number
4. Archive old version for reference
5. Communicate change to all reviewers

---

**Maintained by**: Project Leadership
**Questions**: Contact project lead or review coordinator
