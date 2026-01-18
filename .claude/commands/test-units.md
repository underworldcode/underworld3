---
description: Quick validation of units system
---

Run the complete units test suite to ensure dimensional analysis is working:

```bash
pixi run -e default pytest tests/test_07*_units*.py tests/test_08*_*.py -v --tb=short
```

## Expected Results

Most tests should pass. Check `docs/developer/TEST-CLASSIFICATION-2025-11-15.md` for current baseline.

## Analysis Required

If failures occur, identify which subsystem:
- **Core units** (test_0700): Fundamental units system
- **Unit conversion** (test_0801): Utility functions
- **Workflow integration** (test_0803): End-to-end workflows
- **Stokes ND** (test_0818): Non-dimensional solver integration
- **Coordinate units**: Dimensional analysis of mesh coordinates

## Output Format

- Quick status: X/Y tests passing
- If failures: detailed breakdown by subsystem
- Assessment: regression vs known issue
- Recommended fixes if needed
