# TODO: Fix Notebook 13 - Coordinate Units and Clarity

**Date Created**: 2025-10-14
**Priority**: Medium
**Status**: Pending

## Issues to Address

### 1. Strange Arrow Diagrams
Notebook 13 has diagrams with arrows that need revision for clarity.

### 2. Over-Complicated Explanations
**User's Intent** (from introduction):
> "We'll solve this problem on two different meshes, one a mesh with coordinates in metres, and on a second with coordinates in kilometres. The gradient values should differ by a factor of 1000 (K/m vs K/km), but the physical value they represent is, of course, the same. We will try to show that this can be done without us needing to mess with unit conversions at all."

**Key Principle**: Make it **as obvious as possible**
- ❌ Avoid: Detailed print statements explaining every step
- ❌ Avoid: Complicated comments over-explaining the obvious
- ✅ Goal: Clean, clear demonstration that "it just works"

The notebook should show the natural elegance of the units system, not explain it to death.

## Location

Check: `docs/examples/*13*.ipynb` or similar numbered notebook

## Action Items

- [ ] Locate notebook 13 in the documentation/examples
- [ ] Review and fix the arrow diagrams
- [ ] **Simplify explanations** - remove verbose print statements
- [ ] **Remove excessive comments** - let the code speak for itself
- [ ] Ensure the "it just works" message is clear through simplicity
- [ ] Use UK/Australian spelling (metres, kilometres)

## Design Philosophy

The notebook should demonstrate that:
1. You can work in different coordinate units (metres vs kilometres)
2. The system handles unit conversions automatically
3. Gradients scale correctly (1000× difference)
4. **You don't need to think about it** - it just works

Show, don't tell. Less commentary, more clarity.

## Context

This note was created during the deprecation warning cleanup session, just before starting work on the geographical mesh functionality.

## Next Steps

When ready to address this:
1. Find and read notebook 13
2. **Strip out verbose explanations**
3. Fix arrow diagrams
4. Test that the clean version is actually clearer
5. Ensure it demonstrates elegance through simplicity
