---
description: Run Tier A+B tests - full validation suite
---

Run production-ready and validated tests (excludes experimental):

```bash
pixi run -e default pytest -m "tier_a or tier_b" -v --tb=short
```

## What This Tests

**Tier A:** Production-ready (failures = definite regression)
**Tier B:** Validated but newer (failures need investigation)

## When to Use

- Full validation before merging
- Feature completion verification
- Manual review of new functionality

## Interpreting Results

| Tier | If Fails | Action |
|------|----------|--------|
| A    | HIGH priority | Investigate immediately - real regression |
| B    | MEDIUM priority | Could be test OR code issue - investigate |

## Reference

See `docs/developer/TESTING-RELIABILITY-SYSTEM.md` for tier definitions.
