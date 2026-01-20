---
description: Run only Tier A (production-ready) tests - trusted for TDD and CI
---

Run only the most reliable, production-ready tests:

```bash
pixi run -e default pytest -m "tier_a" -v --tb=short
```

## What This Tests

**Tier A tests are:**
- Long-lived tests with proven track record (>3 months)
- Consistently passing across multiple environments
- Testing stable, well-understood functionality
- Failure indicates **DEFINITE regression** in production code

## When to Use

- Test-Driven Development (TDD) sprints
- Continuous Integration (CI) pipelines
- Release validation
- Bisecting regressions (these tests can be trusted)

## If a Tier A Test Fails

**HIGH PRIORITY** - This indicates a real regression:
1. Investigate immediately
2. Bisect to find breaking commit
3. Fix production code (or demote test if incorrectly promoted)

## Reference

See `docs/developer/TESTING-RELIABILITY-SYSTEM.md` for tier definitions.
