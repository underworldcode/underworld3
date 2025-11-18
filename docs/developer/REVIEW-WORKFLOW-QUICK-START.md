# Architectural Review - Quick Start Guide

**For**: Authors submitting reviews | Reviewers evaluating reviews | Project leads approving reviews

## 🚀 Quick Reference

### For Authors: Submit a Review

**Step 1: Prepare Review Document**
```bash
# Create branch
git checkout -b review/2025-11-your-feature

# Write review using template
cp docs/developer/CODE-REVIEW-PROCESS.md docs/reviews/2025-11/YOUR-FEATURE-REVIEW.md
vim docs/reviews/2025-11/YOUR-FEATURE-REVIEW.md

# Update tracking index
vim docs/reviews/2025-11/REVIEW-TRACKING-INDEX.md

# Verify tests pass
pixi run underworld-test

# Commit
git add docs/reviews/2025-11/
git commit -m "Add architectural review: Your Feature"
git push origin review/2025-11-your-feature
```

**Step 2: Choose Submission Method**

**Option A: Pull Request** (Recommended for complex reviews needing detailed feedback)
```bash
# Create PR via GitHub UI or CLI
gh pr create --title "[REVIEW] Your Feature" \
             --template architectural-review.md \
             --assignee @reviewer1,@reviewer2

# Fill in PR template sections
# Reviewers can comment on specific lines in review document
```

**Option B: GitHub Issue** (For straightforward reviews needing tracking only)
```bash
# Create issue via GitHub UI or CLI
gh issue create --title "[REVIEW] Your Feature" \
                --template architectural-review \
                --assignee @reviewer1,@reviewer2

# Fill in issue template
# Discussion happens in issue comments
```

**Step 3: Monitor & Respond**
- Respond to reviewer comments within 48 hours
- Update review document based on feedback
- Push changes to branch (for PR) or update in repo (for issue)
- Request re-review when ready

### For Reviewers: Review Submission

**Step 1: Acknowledge Assignment**
```bash
# Comment on issue/PR
"Acknowledged. Will review by YYYY-MM-DD."
```

**Step 2: Review Document**
```bash
# Clone/pull latest
git checkout review/2025-11-feature-name

# Read full review document
cat docs/reviews/2025-11/FEATURE-REVIEW.md

# Check referenced code changes
git diff main src/underworld3/path/to/changed/files.py

# Run tests
pixi run -e default pytest tests/test_XXXX*.py -v

# Verify metrics claimed in review
# (performance improvements, test coverage, etc.)
```

**Step 3: Provide Feedback**

**In Pull Request** (line-by-line comments):
```
1. Click "Files changed" tab
2. Click line number in review document
3. Add comment: Question, suggestion, or request
4. Submit review: "Approve" or "Request changes"
```

**In Issue** (general comments):
```
1. Scroll to comment section
2. Reference specific sections: "In 'System Architecture' section..."
3. Provide overall assessment
4. Tag author: @author please address X, Y, Z
```

**Step 4: Approve**
```bash
# Once satisfied:
# - PR: Click "Approve" in review interface
# - Issue: Comment "LGTM - Approved" or similar

# Update sign-off table in review document
```

### For Project Leads: Manage Reviews

**Monthly Review Cycle Setup**
```bash
# 1. Create milestone
gh milestone create "November 2025 Review Cycle" \
                    --due "2025-11-30" \
                    --description "Architectural reviews for November 2025"

# 2. Create project board
# Via GitHub UI: Projects → New Project → "November 2025 Reviews"
# Columns: Submitted | In Review | Changes Requested | Approved

# 3. Monitor submitted reviews
gh issue list --label "architectural-review,review:submitted"

# 4. Assign reviewers
gh issue edit 123 --add-assignee @reviewer1,@reviewer2
```

**Review Triage**
```bash
# Check new submissions daily
gh issue list --label "architectural-review" --state open

# Set priority
gh issue edit 123 --add-label "priority:high"

# Add to milestone
gh issue edit 123 --milestone "November 2025 Review Cycle"

# Move on project board
# (Manual via GitHub UI)
```

**Final Approval**
```bash
# Verify all approvals received
gh pr view 456 --json reviews

# Merge review PR (= formal approval)
gh pr merge 456 --squash --delete-branch

# Update master index
vim docs/reviews/README.md
git add docs/reviews/README.md
git commit -m "Archive approved review: Feature Name"
git push
```

## 📋 Review Checklists

### Author Pre-Submission Checklist
- [ ] Review document follows template structure
- [ ] All required sections present (Overview, Changes, Testing, Sign-Off, Limitations)
- [ ] Sign-off table included
- [ ] Testing instructions provided with commands
- [ ] Known limitations documented honestly
- [ ] Metrics are quantified (performance, coverage, LOC)
- [ ] All referenced tests are passing
- [ ] Tracking index (REVIEW-TRACKING-INDEX.md) updated
- [ ] Related issues/PRs linked
- [ ] Code changes committed to version control

### Reviewer Evaluation Checklist
**Design & Architecture** (30 min):
- [ ] Design rationale is clear and well-justified
- [ ] Trade-offs documented with alternatives considered
- [ ] System architecture is comprehensible
- [ ] Integration points are clearly identified
- [ ] Complexity is justified by benefits

**Implementation** (45 min):
- [ ] Implementation matches documented design
- [ ] Code quality meets project standards
- [ ] Breaking changes identified and justified
- [ ] Backward compatibility properly addressed
- [ ] Error handling is adequate

**Testing & Validation** (30 min):
- [ ] Testing strategy is adequate for changes
- [ ] Tests pass locally: `pixi run -e default pytest tests/test_XXXX*.py -v`
- [ ] Test coverage is sufficient (>80% for new code)
- [ ] Edge cases are properly covered
- [ ] Performance impact assessed with benchmarks

**Documentation** (15 min):
- [ ] Known limitations clearly documented
- [ ] Benefits quantified with real metrics
- [ ] User-facing changes documented
- [ ] Migration guide provided (if needed)
- [ ] Code comments adequate for maintainability

**Total Time**: ~2 hours for thorough review

### Project Lead Final Approval Checklist
- [ ] All required reviewers have approved (typically 2)
- [ ] All "Request changes" feedback has been addressed
- [ ] Author has responded to all comments
- [ ] Tests are passing in CI
- [ ] No merge conflicts with main branch
- [ ] Review document is complete and well-written
- [ ] Sign-off table is fully filled out
- [ ] Master index (README.md) will be updated post-merge

## 🏷️ Label Reference

| Label | Meaning | When to Use |
|-------|---------|-------------|
| `architectural-review` | Identifies architectural reviews | Always (auto-applied by template) |
| `review:submitted` | Ready for review | Initial submission |
| `review:in-progress` | Under active review | Reviewer assigned and working |
| `review:changes-requested` | Needs revision | Reviewer requested changes |
| `review:approved` | Passed review | All approvals received |
| `priority:high` | Urgent | Blocks other work or release |
| `priority:medium` | Normal | Standard priority |
| `priority:low` | Can wait | Nice to have |
| `type:architecture` | System design | Architecture/design review |
| `type:code` | Implementation | Code quality review |
| `type:testing` | Test infrastructure | Testing system review |
| `type:documentation` | Docs | Documentation review |

## 🔄 Status Progression

```
┌─────────────┐
│ SUBMITTED   │  Issue/PR created with review:submitted label
└──────┬──────┘
       │ Author: Submit review
       │ Lead: Assign reviewers, set priority
       ▼
┌─────────────┐
│ IN REVIEW   │  Change label to review:in-progress
└──────┬──────┘
       │ Reviewers: Evaluate, provide feedback
       │ Author: Monitor, respond to questions
       ▼
  ┌────┴────┐
  │         │
  ▼         ▼
APPROVED  CHANGES REQUESTED
  │         │
  │         │ Author: Address feedback
  │         │ Reviewers: Re-evaluate
  │         └─────────┐
  │                   │
  │         ┌─────────▼─────────┐
  │         │ RE-REVIEW         │
  │         └─────────┬─────────┘
  │                   │
  │◄──────────────────┘
  │
  ▼
┌─────────────┐
│ MERGED      │  PR merged OR issue closed
└─────────────┘  Review document in main branch = formally approved
```

## 💡 Tips & Best Practices

### For Authors
- **Write iteratively**: Don't wait for perfection, get early feedback
- **Be honest about limitations**: Builds trust, helps reviewers focus
- **Quantify everything**: "Faster" → "10x faster", "Better" → "80% coverage"
- **Provide context**: Explain WHY not just WHAT
- **Link liberally**: Reference issues, PRs, discussions, commits
- **Test locally first**: Don't submit if tests are failing

### For Reviewers
- **Respond quickly**: Acknowledge within 24h, complete within 2-5 days
- **Be specific**: "Line 45: Consider X instead of Y because Z"
- **Ask questions**: "Why did you choose approach A over B?"
- **Acknowledge good work**: "Nice solution to the caching problem!"
- **Distinguish**: Blocking issues vs. suggestions for future improvement
- **Test locally**: Don't rely only on reading, run the code

### For Project Leads
- **Assign expertise**: Match reviewers to their areas of knowledge
- **Monitor progress**: Weekly check on project board
- **Unblock quickly**: Don't let reviews sit idle for weeks
- **Communicate**: Keep team informed of review status and priorities
- **Archive promptly**: Update master index when reviews are approved

## 🆘 Common Issues & Solutions

**Issue**: Review is taking too long (>1 week)
**Solution**:
```bash
# Lead: Ping reviewers, offer to help or reassign
gh issue comment 123 --body "@reviewer1 @reviewer2 Status update? Can I help unblock?"

# If stuck, escalate to synchronous meeting
```

**Issue**: Reviewers disagree on approval
**Solution**:
```bash
# Lead makes final call, documents decision
gh issue comment 123 --body "After discussion, approving with rationale: [explanation]"
```

**Issue**: Review document is incomplete/poorly written
**Solution**:
```bash
# Change label to review:changes-requested
gh issue edit 123 --remove-label "review:submitted" \
                  --add-label "review:changes-requested"

# Provide specific guidance on what's missing
gh issue comment 123 --body "Please add: 1) Testing instructions 2) Performance metrics 3) Known limitations"
```

**Issue**: Tests are failing but review claims they pass
**Solution**:
```bash
# Request CI run or local test evidence
gh issue comment 123 --body "Tests appear to be failing. Please provide output of: pixi run underworld-test"

# Block approval until resolved
```

## 📚 Resources

- **Full Process**: [CODE-REVIEW-PROCESS.md](CODE-REVIEW-PROCESS.md)
- **GitHub Integration**: [GITHUB-REVIEW-INTEGRATION.md](GITHUB-REVIEW-INTEGRATION.md)
- **Review Archive**: [docs/reviews/README.md](../reviews/README.md)
- **Templates**: [.github/ISSUE_TEMPLATE/](../../.github/ISSUE_TEMPLATE/)

## 🤖 Automation

**Review Validation Workflow**:
- Automatically checks review document structure
- Posts reviewer checklist as comment
- Validates tests are passing
- Checks tracking index is updated

**Manual Triggers**:
```bash
# Re-run validation workflow
gh workflow run architectural-review-validation.yml

# Check workflow status
gh run list --workflow=architectural-review-validation.yml
```

---

**Quick Links**:
- Create Issue: https://github.com/underworldcode/underworld3/issues/new?template=architectural-review.yml
- View Reviews: https://github.com/underworldcode/underworld3/tree/main/docs/reviews
- Project Board: https://github.com/underworldcode/underworld3/projects
- Review Process: [CODE-REVIEW-PROCESS.md](CODE-REVIEW-PROCESS.md)

**Last Updated**: 2025-11-17
