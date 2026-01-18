# GitHub Integration for Architectural Reviews

**Status**: Active
**Version**: 1.0
**Last Updated**: 2025-11-17
**Related**: [`CODE-REVIEW-PROCESS.md`](CODE-REVIEW-PROCESS.md)

## Overview

This document describes how Underworld3's formal architectural review process integrates with GitHub's collaboration tools. Unlike standard pull request code reviews (which focus on specific line-by-line changes), architectural reviews evaluate system design, implementation rationale, trade-offs, and long-term maintainability.

## Why GitHub Integration?

**GitHub provides**:
- 📋 **Issue Tracking** - Review lifecycle management, assignment, status
- 🗂️ **Project Boards** - Visual progress tracking across reviews
- 💬 **Discussions** - Architectural discourse and decision documentation
- 🔀 **Pull Requests** - Line-by-line commenting on review documents
- 🏷️ **Labels & Milestones** - Categorization and release tracking
- 🤖 **Actions** - Automated validation and workflow enforcement

## Integration Strategy

### 1. GitHub Issues - Individual Review Tracking

**Purpose**: Track each review's lifecycle from submission to approval.

**How to Use**:
1. **Submit Review**: Create issue using "Architectural Review" template
2. **Assign Reviewers**: Use GitHub assignments (1-2 reviewers)
3. **Track Progress**: Update labels as review progresses
4. **Document Feedback**: Use issue comments for high-level discussion
5. **Close on Approval**: Link to merged PR when review is approved

**Labels**:
```
architectural-review       # Identifies architectural reviews
review:submitted          # Ready for review assignment
review:in-progress        # Under active review
review:changes-requested  # Author needs to address feedback
review:approved           # Passed review, ready to archive
priority:high/medium/low  # Urgency level
type:architecture/code/test/docs  # Review category
```

**Example Issue**:
```
Title: [REVIEW] Function Evaluation System
Labels: architectural-review, review:submitted, priority:high, type:architecture

Body:
Review Document: docs/reviews/2025-11/FUNCTION-EVALUATION-SYSTEM-REVIEW.md
Priority: HIGH
Category: System Architecture

Summary: Merger of evaluate() and global_evaluate() code paths with
automatic lambdification optimization providing ~10,000x speedup...

Key Metrics:
- Performance: 22s → 0.003s (7,400x faster)
- Test coverage: 20 comprehensive tests, all passing
- New module: pure_sympy_evaluator.py (~360 lines)
```

### 2. GitHub Projects - Review Dashboard

**Purpose**: Visualize review progress, track monthly review cycles.

**Board Structure**:
```
Project: "November 2025 Architectural Reviews"

Columns:
┌──────────────┬──────────────┬──────────────┬──────────────┐
│ Submitted    │ In Review    │ Changes Req. │ Approved     │
├──────────────┼──────────────┼──────────────┼──────────────┤
│ [Issues]     │ [Issues]     │ [Issues]     │ [Issues]     │
└──────────────┴──────────────┴──────────────┴──────────────┘
```

**Custom Fields**:
- **Review Document**: Link to `docs/reviews/` file
- **Review Type**: Architecture / Code / Testing / Docs
- **Priority**: High / Medium / Low
- **LOC Changed**: Number
- **Test Coverage %**: Number
- **Target Date**: Date

**Views**:
- **Board View**: Kanban-style status tracking
- **Table View**: Detailed metadata for all reviews
- **Timeline View**: Review deadlines and dependencies
- **Roadmap View**: Month-over-month review planning

### 3. GitHub Discussions - Architectural Discourse

**Purpose**: Community discussion of design decisions, trade-offs, and alternatives.

**Categories**:
```
💡 Architecture Decisions
   - Design trade-offs and alternatives
   - Technology choices and rationale
   - System architecture debates

🔍 Review Feedback
   - Public discussion of review findings
   - Clarifying questions from reviewers
   - Design iteration based on feedback

❓ Q&A
   - Questions about reviewed systems
   - Implementation guidance
   - Best practices

📢 Announcements
   - Approved reviews
   - Process changes
   - Policy updates
```

**Example Discussion**:
```
Category: Architecture Decisions
Title: "Units System: Pint vs Custom Implementation?"

Initial Post (Author):
We're evaluating two approaches for units awareness:
1. Pint library (full-featured, battle-tested)
2. Custom lightweight implementation (minimal overhead)

Trade-offs:
[Detailed analysis...]

Comments:
├─ Performance benchmarks showing negligible Pint overhead
├─ Extensibility considerations for future features
├─ Community support and maintenance burden
└─ Resolution: Selected Pint, documented in review
```

### 4. Pull Requests - Review Document Submission

**Purpose**: Treat review documents like code - line-by-line commenting, version control.

**Workflow**:
```
1. Author creates branch:
   git checkout -b review/2025-11-function-evaluation

2. Author adds review document:
   docs/reviews/2025-11/FUNCTION-EVALUATION-SYSTEM-REVIEW.md

3. Author updates tracking index:
   docs/reviews/2025-11/REVIEW-TRACKING-INDEX.md

4. Author creates PR using "Architectural Review" template

5. Reviewers comment on specific lines in review document

6. Author addresses feedback via commits

7. Reviewers approve PR

8. PR merge = Review formally approved
```

**Benefits**:
- **Line-by-line comments** on design decisions
- **Version history** of review document evolution
- **Required approvals** enforced by branch protection
- **Automated checks** via GitHub Actions
- **Permanent record** in git history

**When to Use PR vs Issue**:
- **PR**: When review document needs detailed feedback, multiple revision rounds
- **Issue**: When review is straightforward, mainly needs assignment/tracking

### 5. Labels & Milestones - Organization

**Labels** (`.github/labels.yml`):
```yaml
# Review Status
- name: architectural-review
  color: 0E8A16
  description: Formal architectural review

- name: review:submitted
  color: 1D76DB
  description: Ready for review assignment

- name: review:in-progress
  color: FBCA04
  description: Under active review

- name: review:changes-requested
  color: D93F0B
  description: Author needs to address feedback

- name: review:approved
  color: 0E8A16
  description: Passed review

# Priority
- name: priority:high
  color: D93F0B
  description: Urgent review required

- name: priority:medium
  color: FBCA04
  description: Normal priority

- name: priority:low
  color: 0E8A16
  description: Low urgency

# Review Type
- name: type:architecture
  color: 5319E7
  description: System architecture review

- name: type:code
  color: 1D76DB
  description: Code implementation review

- name: type:testing
  color: C5DEF5
  description: Testing infrastructure review

- name: type:documentation
  color: BFD4F2
  description: Documentation review
```

**Milestones**:
```
Milestone: "November 2025 Review Cycle"
- Due Date: 2025-11-30
- Description: Architectural reviews for major systems revised in November 2025
- Issues: 9 reviews (3 complete, 6 in progress)
- Progress: 33%
```

### 6. GitHub Actions - Automation

**Automated Workflows**:

**1. Review Validation** (`.github/workflows/architectural-review-validation.yml`):
```yaml
Triggers: PR to docs/reviews/
Validates:
  ✓ Required sections present (Overview, Changes, Testing, Sign-Off)
  ✓ Sign-off table included
  ✓ Tracking index updated
  ✓ Proper markdown formatting
```

**2. Review Checklist Posting**:
```yaml
Triggers: New review issue/PR created
Posts:
  - Reviewer checklist as comment
  - Links to review process documentation
  - Reminder about approval requirements
```

**3. Test Suite Validation** (Optional):
```yaml
Triggers: Review PR opened
Runs:
  - All tests mentioned in review document
  - Performance benchmarks (if applicable)
  - Coverage analysis
Posts:
  - Test results as PR comment
  - Pass/fail status for reviewers
```

**4. Status Sync**:
```yaml
Triggers: PR merged or closed
Updates:
  - GitHub Issue status
  - Project board column
  - Milestone progress
  - Master README.md index
```

## Recommended Workflow

### For Authors

**1. Prepare Review Document**:
```bash
# Create review branch
git checkout -b review/2025-11-your-feature

# Write review document following template
vim docs/reviews/2025-11/YOUR-FEATURE-REVIEW.md

# Update tracking index
vim docs/reviews/2025-11/REVIEW-TRACKING-INDEX.md

# Commit
git add docs/reviews/2025-11/
git commit -m "Add architectural review: Your Feature"
git push origin review/2025-11-your-feature
```

**2. Submit for Review**:

**Option A - Via Pull Request** (Recommended for complex reviews):
- Create PR using "Architectural Review" template
- Fill in all sections (summary, scope, metrics, etc.)
- Link related issues/discussions
- Assign reviewers
- Wait for line-by-line feedback

**Option B - Via Issue** (For straightforward reviews):
- Create issue using "Architectural Review" template
- Link to review document in repo
- Assign reviewers via GitHub
- Use issue comments for feedback

**3. Address Feedback**:
```bash
# Make changes based on reviewer comments
vim docs/reviews/2025-11/YOUR-FEATURE-REVIEW.md

# Commit updates
git add docs/reviews/2025-11/YOUR-FEATURE-REVIEW.md
git commit -m "Review feedback: Clarify design rationale for X"
git push

# Respond to each comment explaining changes
```

**4. Merge on Approval**:
- Wait for required approvals (typically 2 reviewers + project lead)
- Merge PR (or close issue if using issue-only workflow)
- Review document is now in `main` branch = formally approved

### For Reviewers

**1. Review Assignment**:
- Receive GitHub notification (issue assignment or PR review request)
- Acknowledge within 24 hours
- Indicate expected review timeline

**2. Conduct Review**:
```bash
# Read review document
cat docs/reviews/2025-11/FEATURE-REVIEW.md

# Check referenced code (if code review)
git diff main...review-branch src/

# Run tests mentioned in review
pixi run -e default pytest tests/test_XXXX*.py -v

# Review against checklist:
# ✓ Design rationale clear?
# ✓ Trade-offs documented?
# ✓ Testing adequate?
# ✓ Known limitations documented?
```

**3. Provide Feedback**:

**In PR** (line-by-line):
- Click on specific lines in review document
- Add comments, questions, suggestions
- Mark comments as "Request changes" or "Comment"

**In Issue** (high-level):
- Add comments to issue thread
- Reference specific sections of review doc
- Summarize overall assessment

**4. Approve**:
- Once satisfied, approve PR or add "Approved" comment to issue
- Update sign-off table with your name and date
- Project lead merges when all approvals received

### For Project Leads

**1. Triage New Reviews**:
- Assign appropriate reviewers based on expertise
- Set priority and milestone
- Move to "In Review" on project board

**2. Monitor Progress**:
- Check project board weekly
- Follow up on stalled reviews
- Resolve conflicts or impasses

**3. Final Approval**:
- Verify all reviewer approvals received
- Check that feedback was addressed
- Merge PR or close issue
- Update master README index
- Archive in permanent review collection

## Comparison: GitHub Tools

| Tool | Best For | Review Stage | Complexity |
|------|----------|--------------|-----------|
| **Issues** | Individual tracking, assignment | All stages | Low |
| **Projects** | Progress visualization, planning | Planning, monitoring | Medium |
| **Discussions** | Design debates, community input | Pre-review, post-approval | Low |
| **Pull Requests** | Detailed document review | Review, revision | High |
| **Labels** | Categorization, filtering | All stages | Low |
| **Milestones** | Release tracking, grouping | Planning, monitoring | Low |
| **Actions** | Automation, validation | Submission, approval | High |

## Integration with External Tools

### Documentation Sites
```yaml
# Auto-deploy approved reviews to documentation site
on:
  push:
    paths:
      - 'docs/reviews/**/*.md'
    branches:
      - main

steps:
  - Deploy to: https://underworld3.readthedocs.io/reviews/
```

### Slack/Discord Notifications
```yaml
# Notify team of new reviews, approvals
- uses: slackapi/slack-github-action@v1
  with:
    payload: |
      {
        "text": "New architectural review submitted: ${{ github.event.issue.title }}"
      }
```

### Metrics Dashboard
```yaml
# Generate review metrics for project dashboard
- Track: Time to approval, review backlog, approval rate
- Export: To GitHub Pages or external dashboard
```

## Best Practices

### Do ✅

- **Use templates** - Ensure consistency across reviews
- **Link liberally** - Connect issues, PRs, discussions, commits
- **Update status promptly** - Keep labels and project board current
- **Automate where possible** - Use Actions for validation
- **Archive permanently** - Approved reviews stay in `main` forever
- **Communicate clearly** - Use @mentions, clear commit messages

### Don't ❌

- **Mix review types** - Keep architectural reviews separate from code PRs
- **Skip templates** - They ensure completeness
- **Leave reviews orphaned** - Always track in project board
- **Rush approvals** - Quality over speed
- **Forget to update index** - Keep master README current
- **Delete review history** - It's reference documentation

## FAQ

**Q: Should every review use both an issue AND a PR?**
A: No. Use PR for complex reviews needing detailed feedback. Use issue-only for straightforward reviews that just need tracking.

**Q: What if reviewers disagree on approval?**
A: Escalate to project lead. Lead makes final decision and documents rationale in review.

**Q: Can reviews be revised after approval?**
A: Yes, but rarely. Create a new addendum review document if major changes needed post-approval.

**Q: How long should reviews take?**
A: Target 2-5 days for initial review, 1-2 rounds of revision. See metrics in CODE-REVIEW-PROCESS.md.

**Q: Who has merge permission for review PRs?**
A: Only project leads should merge review PRs. This ensures proper approval verification.

## Templates Location

- **Issue Template**: `.github/ISSUE_TEMPLATE/architectural-review.yml`
- **PR Template**: `.github/PULL_REQUEST_TEMPLATE/architectural-review.md`
- **Workflow**: `.github/workflows/architectural-review-validation.yml`
- **Labels**: Create via GitHub UI or `.github/labels.yml`

## Getting Started

**1. Create labels** (one-time setup):
```bash
# Via GitHub CLI
gh label create "architectural-review" --color 0E8A16
gh label create "review:submitted" --color 1D76DB
# ... (see Labels section above for full list)
```

**2. Create project board** (monthly):
- Go to: Projects → New Project → "November 2025 Reviews"
- Add columns: Submitted, In Review, Changes Requested, Approved
- Add custom fields: Review Document, Priority, Type, etc.

**3. Submit first review**:
- Create review document in `docs/reviews/YYYY-MM/`
- Create PR or Issue using template
- Assign reviewers
- Track progress on project board

## Support

- **Questions**: Open a GitHub Discussion in "Q&A" category
- **Process Issues**: Create issue with `process-improvement` label
- **Template Updates**: Submit PR to `.github/` templates

---

**Related Documentation**:
- [CODE-REVIEW-PROCESS.md](CODE-REVIEW-PROCESS.md) - Full review process
- [Review Templates](../../.github/ISSUE_TEMPLATE/) - GitHub templates
- [Review Archive](../reviews/README.md) - Historical reviews

**Last Updated**: 2025-11-17
**Maintained By**: Project Leadership

---

**Reviewed for**: Review System Infrastructure & GitHub Integration (Review #10, 2025-11-17)  
**Part of**: Formal architectural review process implementation

---

## Review History
- **Review #10** (2025-11-17): Document created as part of Review System Infrastructure review
