# Declined proposal: quantity-valued coordinates / `coord_units` for evaluate()

**Status**: Declined — maintainer ruling D7, 2026-07-06 (units-family scope
decision, `docs/reviews/2026-07/REMEDIATION-WORKLIST.md`). Recorded here so
the idea is findable if it is ever revisited; the placeholder tests that
used to stand in for it were deleted (LE-08, LE-10, LE-11 / WA-23, WA-24).

## What was proposed

A family of unit-aware coordinate conveniences, which existed only as
skipped test placeholders (never implemented, no design doc):

1. **`coord_units=` parameter for `uw.function.evaluate()`** — interpret a
   plain coordinate array in a named unit, e.g.
   `evaluate(T.sym, pts_km, coord_units="km")`.
2. **Quantity-valued coordinate lists** — accept
   `[(uw.quantity(0.5, "m"), uw.quantity(50, "cm"))]` (and flat
   `[x_qty, y_qty]`) as evaluate/global_evaluate coordinates.
3. **`UnitAwareArray` returns from coordinate-expression evaluation**, and
   Pint-quantity coordinate support in `mesh.points_in_domain()` /
   `mesh.get_closest_cells()`.
4. **Constructor-set mesh units** — `units=` on the Cartesian mesh
   constructors as a live feature, plus `mesh.to_units()` conversion.

## The ruling

The family is **unsupported**. The model owns the unit system
(`docs/developer/design/UNITS_SIMPLIFIED_DESIGN_2025-11.md`): coordinates
passed to evaluation are plain arrays in model (non-dimensional) units, or
whole `UnitAwareArray`/`UWQuantity` arrays (which the evaluate gateway
non-dimensionalises). Physical locations are converted explicitly with
`uw.scaling.non_dimensionalise`. The mesh-constructor `units=` kwarg is
deprecated (DeprecationWarning) and ignored; `mesh.units` reports the
model's coordinate unit.

## What was removed (Wave A, 2026-07)

- 16 placeholder skips across 7 units test files (`test_0730`,
  `test_0800_unit_aware_functions`, `test_0803_*` ×2, `test_0804`,
  `test_0811`); files that contained nothing else were deleted.
- `test_0630_mesh_units_demonstration.py` (pure demonstration of the
  superseded proposal); `test_0620_mesh_units_interface.py` rewritten to
  assert the implemented model-owned semantics.
- `test_0812_poisson_with_units.py` rewritten with supported coordinate
  forms (the Poisson-with-units coverage is regained).

If the feature is revisited, start from the deleted tests in git history —
they are a usable statement of the intended user experience.
