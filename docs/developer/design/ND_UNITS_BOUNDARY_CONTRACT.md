# The non-dimensional ↔ units boundary (contract)

This page defines **where the units system stops and the non-dimensional (ND)
solver world begins** in Underworld3, for both users and developers. It is the
companion "rules of engagement" to the
[units design](UNITS_SIMPLIFIED_DESIGN_2025-11.md).

The one sentence to remember:

> **Everything from the DM downward — PETSc, the solvers, `evaluate` /
> `global_evaluate`, point-location, the JIT residuals — runs in
> non-dimensional (model-unit) space. Units live *above* that line, in the
> user-facing `MeshVariable` API.**

## The boundary, concretely

The mesh DM stores **non-dimensional** coordinates (e.g. `0..1000` model units,
*not* `0..1e9` metres). `evaluate` / `global_evaluate` treat plain arrays as
those DM/ND coordinates, and the JIT residual/Jacobian kernels see ND field
values. None of that machinery knows about Pint units.

Units are a deliberate **duality on the variable API**:

| dimensional (units side) | non-dimensional (solver/DM side) |
| --- | --- |
| `var.array`  — `UnitAwareArray` | `var.data` — plain ND values (what the solver sees) |
| `var.coords` — `UnitAwareArray` | `var.coords_nd` — plain ND DM coordinates |
| `uw.quantity(...)`, Pint quantities | plain floats / numpy arrays |

The **crossing functions** between the two sides are:

- `uw.non_dimensionalise(...)` — dimensional → ND (going *down* to the solver)
- `uw.dimensionalise(value, units)` — ND → dimensional (coming *up* to the user)
- the `.data` / `.coords_nd` accessors (already-crossed ND views)

## The rule for developers

Inside any **solver-internal** code (anything that builds DM coordinates,
calls `evaluate`/`global_evaluate`, locates points, assembles residuals, or does
coordinate/velocity arithmetic for the DM):

1. **Work in ND space.** Use `var.coords_nd`, not `var.coords`; use `var.data`,
   not `var.array`.
2. **Cross at the boundary, explicitly.** If you start from a dimensional
   quantity, call `uw.non_dimensionalise(...)` *before* the value reaches the
   DM / `evaluate`. Convert back with `uw.dimensionalise(...)` only when handing
   a result back to the user.
3. **Never pass a `UnitAwareArray` into `evaluate` / `global_evaluate` / the
   DM.** See the trap below for why this is silently wrong rather than an error.

For **users**: pass dimensional quantities (`uw.quantity(...)`) or plain
model-unit values; read `.array`/`.coords` for dimensional results and
`.data`/`.coords_nd` for the ND values the solver used. You should not need to
nondimensionalise by hand — that is the library's job at this boundary.

## The trap (why this is a contract, not just advice)

`evaluate` / `global_evaluate` **strip the `UnitAwareArray` subclass with
`np.array()` but keep the numeric value.** So if you hand them a *dimensional*
array (`coords` ≈ `1e9` m), they keep `1e9` and locate it against a `0..1000` DM
— **no exception, wrong location, wrong answer.** The unit label is the only
thing that gets dropped; the dimensional magnitude survives and silently
mislocates.

This is exactly the bug that was fixed in
[#267](https://github.com/underworldcode/underworld3/issues/267): the
semi-Lagrangian trace-back carried dimensional `.coords` and `m/s` velocity into
the ND DM. The crash (`'meter' − 'meter/second'`) was the *lucky* symptom; the
deeper failure was mislocation when units happened to be compatible.

## The canonical correct pattern (from the #267 fix)

The semi-Lagrangian trace-back in `systems/ddt.py` is the reference example —
the whole trace-back is computed in ND/DM space regardless of whether the model
carries units:

```python
# Departure point in the mesh's ND (DM) coordinate space.
# coords_nd == .coords for a non-units model; the ND reduction of the
# dimensional coordinates when units are active — matching what
# evaluate / the DM point-location expect.
coords = np.asarray(self.psi_star[i].coords_nd)

# Velocity non-dimensionalised to the same space before the arithmetic.
v_nondim = uw.non_dimensionalise(v_at_node_pts, model)   # -> plain ND array

# dt reduced to non-dimensional model-time.
mid_pt = coords - v_nondim * (0.5 * dt_nd)
v_mid  = uw.function.global_evaluate(self.V_fn, mid_pt)  # plain ND coords in
```

What this code does **not** do (and you should not do): reach for `.coords`
(dimensional), keep velocities in `m/s`, or pass a `UnitAwareArray` to
`global_evaluate`.

## Enforcement (in progress)

Today the boundary is **convention** — the rule above is correct but not
machine-checked, which is how #267 slipped in. Making `evaluate` /
`global_evaluate` (and the DM coordinate setters) **reject or correctly
non-dimensionalise** a `UnitAwareArray` input — instead of stripping the label
and keeping the value — would turn this contract into a guard rail. That
enforcement work is tracked separately; until it lands, the rule above is on you
to follow in solver-internal code.

## See also

- [Units system design](UNITS_SIMPLIFIED_DESIGN_2025-11.md) — the authoritative
  units architecture and the `.array` vs `.data` semantics.
- [Why units, not dimensionality](WHY_UNITS_NOT_DIMENSIONALITY.md).
- [Coordinate units technical note](../ai-notes/COORDINATE-UNITS-TECHNICAL-NOTE.md).
