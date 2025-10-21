# TODO: Update view() Methods to Remove Deprecated References

**Date Created**: 2025-10-14
**Priority**: High
**Status**: ✅ Complete

## Issue

The `mesh.view()` and `swarm.view()` methods likely still reference the deprecated `mesh.points` and `swarm.points` properties.

## User Request

> "update mesh.view() [probably swarm.view too] so it does not reference mesh.points any more."

## Action Items

- [x] Find and update `mesh.view()` method
  - ✅ Replaced `mesh.points` references with `mesh.X.coords`
  - Location: `src/underworld3/discretisation/discretisation_mesh.py:670-671`
  - Fixed display messages in unit-aware coordinate section

- [x] Find and update `swarm.view()` method
  - ✅ Checked - no coordinate references found
  - Location: `src/underworld3/swarm.py:1594-1600`
  - Method only displays index information, no coordinate access

- [x] Rebuild after updates
  - ✅ `pixi run underworld-build` completed successfully

## Search Strategy

```bash
grep -n "def view" src/underworld3/discretisation/discretisation_mesh.py
grep -n "def view" src/underworld3/swarm.py
```

Then check inside those methods for:
- `self.points`
- `.points`
- Any display/print of coordinate information

## Related Work

This is part of the broader deprecation cleanup effort where we've been replacing:
- `mesh.points` → `mesh.X.coords`
- `mesh.data` → `mesh.X.coords`
- `swarm.points` → `swarm._particle_coordinates.data`

## Context

Identified during the final stages of deprecation warning elimination before starting geographic mesh work.
