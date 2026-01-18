---
description: Validate complex solver tests
---

Run the complex solver test suite to validate core physics functionality:

```bash
# Test Stokes solvers
pixi run -e default pytest tests/test_101*_Stokes*.py -v --tb=short

# Test advection-diffusion solvers
pixi run -e default pytest tests/test_110*_AdvDiff*.py -v --tb=short

# Test other complex systems
pixi run -e default pytest tests/test_11*_*.py -v --tb=short
```

## Analysis Required

- Report pass/fail counts for each category (Stokes, AdvDiff, Other)
- For failures, categorize by:
  - Solver type
  - Error pattern (convergence, numerical accuracy, API usage)
- Identify if failures are related to recent changes

## Critical Note

These tests validate core physics. Any failures are **high priority**.

## Output Format

| Test Category | Total | Passed | Failed |
|---------------|-------|--------|--------|
| Stokes        |       |        |        |
| AdvDiff       |       |        |        |
| Other         |       |        |        |

- Detailed failure analysis grouped by solver type
- Assessment: Are failures new regressions or pre-existing?
