---
description: Analyze units test failures and classify for reliability tiers
---

Run units tests and classify any failures by reliability tier:

```bash
pixi run -e default pytest tests/test_07*_units*.py tests/test_08*_*.py -v --tb=short 2>&1 | head -200
```

## Classification Criteria

### Tier A Candidates (Production-Ready)
- [ ] Test has existed for >3 months
- [ ] Test passes consistently
- [ ] Functionality is stable and production-used
- [ ] Failure would indicate definite bug

### Tier B Candidates (Validated)
- [ ] Test passes at least once
- [ ] Functionality appears correct
- [ ] Needs more production validation
- [ ] Test is new (<3 months)

### Tier C Candidates (Experimental)
- [ ] Feature is not fully implemented
- [ ] Test documents expected future behavior
- [ ] Known issues exist

## Analysis Required

For each failing test:
1. Identify root cause (test issue vs code issue)
2. Classify into appropriate tier
3. Recommend: fix, mark xfail, or remove

## Output Format

| Test File | Status | Tier | Recommendation |
|-----------|--------|------|----------------|
| test_0700_* | Pass/Fail | A/B/C | Action needed |

## Reference

See `docs/developer/TEST-CLASSIFICATION-2025-11-15.md` for current status.
