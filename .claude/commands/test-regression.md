---
description: Check regression test suite status
---

Run the regression test suite and report on any failures:

```bash
pixi run -e default pytest tests/test_06*_regression.py -v --tb=short
```

## Analysis Required

- Count total tests and number passing/failing
- For each failure, identify:
  - Test name and location
  - Error type (AttributeError, AssertionError, etc.)
  - Root cause category (API assumption issue, actual bug, naming convention)
- Provide summary with recommended next actions

## Output Format

- Summary: X/Y tests passing
- Failures grouped by category
- Recommended fixes for each failure type
