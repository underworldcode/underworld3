---
title: "Mesh.OT_adapt() — public API proposal"
date: 2026-05-24
status: retired (2026-07 — OT_adapt removed; superseded by the MMPDE mover)
---

> **RETIRED movers note (2026-07):** the spring / Monge-Ampère / OT-step /
> anisotropic-Winslow interior movers this document discusses were retired
> (superseded by `method="mmpde"`, the default). Kept as the R&D record;
> the code lives in git history before the retirement commit.

# `mesh.OT_adapt()` — public API proposal

## Background

The validated production pattern for time-stepping convection with
metric-driven mesh adaptation (per the 2026-05-23/24 investigation,
see `project_ot_reset_validated.md`) is, internally:

1. Reset the mesh to its IC uniform coords
2. FE-remap the driving field T onto the uniform-mesh DOFs
3. Build the gradient-based metric ρ
4. Run the OT mover from the uniform canvas
5. FE-remap simulation fields onto the OT-adapted positions

But the "reset" is an **implementation detail** of what makes the
OT mover behave well across many adapt cycles. From the user's
point of view this is just "adapt the mesh to track ∇T". The API
should reflect that.

## API surface

A single method on the mesh:

```python
moved = mesh.OT_adapt(
    field,
    *,
    refinement=3.0,
    coarsening="auto",
    grad_smoothing_length="auto",
    metric_choice="front-following",
    fields_to_remap=None,
    fields_to_zero=None,
    skip_threshold=None,
    verbose=False,
)
```

**Required:**

- `field` — the scalar MeshVariable whose gradient drives
  refinement (typically `T`)

**Metric knobs** (production defaults validated):

- `refinement` — cell-size envelope (h0/R). Primary feature knob;
  validated range 1.5–5, 3 ≈ Nu sweet spot
- `coarsening` — `"auto"` (= refinement^(1/d)) or numeric
- `grad_smoothing_length` — screened-Poisson de-noising length L for
  ∣∇field∣ (the most effective sliver lever). `"auto"` (default) ≈ the
  mesh's uniform cell size — needed to keep R≈3 sliver-free; `None` = off;
  ≈ 2·h0 stronger. User-supplied lengths are **unit-aware** (Pint /
  non-dimensionalised via the projection)
- `metric_choice` — `"front-following"` or `"gradient-uniform"`

**Field handling:**

- `fields_to_remap` — list of MeshVariables to FE-remap onto the
  adapted positions (preserve as continuous fields). `field`
  itself is always remapped implicitly
- `fields_to_zero` — list of MeshVariables to zero post-adapt
  (e.g. velocity, pressure on a cold restart)

**Behaviour:**

- `skip_threshold` — if mesh is already aligned within this
  misalignment, skip the whole step and return False
- Returns `True` if the mesh moved, `False` otherwise

## Reference coordinates — function-managed cache

The "reset target" coords are cached lazily on first call:

```python
def OT_adapt(self, field, **kwargs):
    if not hasattr(self, "_ot_adapt_reference_coords"):
        # First call — snapshot the current mesh state as
        # the reset target for all future OT_adapt calls
        self._ot_adapt_reference_coords = \
            np.asarray(self.X.coords).copy()
    ...
```

This keeps mesh classes (`Annulus`, `Box`, …) unchanged — no
modifications to their `__init__`. The cache lives on the mesh
instance and survives across calls.

**Staleness caveat.** If anyone (not OT_adapt itself) deforms the
mesh between calls — e.g. a user manually calling
`mesh._deform_mesh(some_X)` for unrelated purposes — the cached
"pristine" coords will diverge from whatever the user thinks of
as the reference state. The cached value is still well-defined
(it's whatever the mesh was on the *first* OT_adapt call), but
may no longer match the user's intent.

Mitigations:

- Document the caveat in the docstring (above the example)
- Provide a `mesh.OT_adapt_reset_reference(coords=None)` method
  to invalidate / override the cache (None → re-cache from
  current state; explicit coords → use those as the new
  reference). Lets the user opt-in to a deliberate re-baseline
- Optional `reference_coords` kwarg on `OT_adapt` itself for
  one-off override (doesn't update the cache)

## Boundary-slip as a DOF constraint (analogous to Stokes BCs)

Same framing as Stokes `essential_bc` — at each boundary node,
constrain certain DOFs of the **displacement** to zero, leave
others free:

| BC type | Constraint on displacement at boundary node |
|---|---|
| pinned | all components zero (no motion) |
| slip | component along outward normal = 0 (only tangential motion) |
| free | no constraint |

For curved boundaries this is first-order: zeroing the normal
component of an arbitrary displacement leaves the node *near*
but not *exactly on* the boundary. A small snap-back projection
restores the node to the surface after each step (the same
"snap to fixed \|r\|" that the current ring code does, but now
derived from the BC rather than hardcoded).

**Use the existing `mesh.Gamma_N`** — UW3 already exposes a
`sympy.Matrix` row of the normalised outward boundary normal as
`mesh.Gamma_N` (in `discretisation_mesh.py:2213`). Every mesh
class already defines it. No new method required.

`OT_adapt` evaluates `mesh.Gamma_N` at the boundary node
coordinates to get the per-node normal, then zeros the normal
component of the OT displacement at those nodes — same pattern
Stokes uses with `add_essential_bc`. No geometry-specific
'box' / 'ring' strings, no per-class snap-back projection
bespoke code.

```python
def _apply_slip_constraint(self, displacement, boundary_mask):
    """Zero the normal component of displacement at boundary
    nodes. Tangential motion left free."""
    boundary_coords = self.X.coords[boundary_mask]
    n_hat = np.asarray(uw.function.evaluate(
        self.Gamma_N, boundary_coords)).reshape(-1, self.cdim)
    disp_bnd = displacement[boundary_mask]
    disp_normal = (disp_bnd * n_hat).sum(axis=1, keepdims=True)
    displacement[boundary_mask] -= disp_normal * n_hat
    return displacement
```

This is just first-order — for curved boundaries (Annulus,
SphericalShell), zeroing the normal component leaves the node
*near* but not *exactly on* the boundary surface. A small
snap-back projection restores nodes to the surface after each
step (snap-to-fixed-\|r\| for radial cases — the current 'ring'
code generalised). The snap-back can also be derived from the
mesh's coordinate system (e.g. `mesh.CoordinateSystem`).

**Sphere2D edge case**: its normal is also `Gamma_N`
(radial-outward everywhere); the difference is that the
constraint applies to *every node*, not just boundary nodes. A
`mesh.is_manifold` flag (or `mesh.constraint_mask` exposing
"these nodes need the slip projection") tells OT_adapt to
extend the projection. The API hook (`Gamma_N`) is the same;
only the "which nodes get constrained" logic differs.

See the **Boundary-slip as a DOF constraint** section below for
how the per-mesh `boundary_normal` method makes this uniform
across geometries.

## What's *not* in the API

- The legacy `incremental` OT path (slivers accumulate)
- The post-OT spring polish (converged spring loses Nu)
- The escalating-R chain (no gain over single R)
- `metric_degree > 1` (broken on reset path — cached projection
  goes stale)
- `boundary_slip` mode string — automatic: uses
  `mesh._boundary_tangent_project` if defined, else falls back
  to pinned boundaries
- The "reset" itself — caller doesn't see it; just calls
  `mesh.OT_adapt(...)`

## What still needs upstream work

Two production-readiness gaps remain (per
`project_ot_production_blockers.md`):

1. **Sphere2D constrained-manifold OT** — the only true manifold
   mesh in the table; OT mover needs to constrain *every* node
   (not just boundary nodes) to the spherical surface. The
   NotImplementedError hook is the API contract; the actual
   implementation is research.
2. **Parallel JIT determinism error** — blocks ANY parallel UW3
   run, not specific to OT_adapt.

## Caller code

After the API lands, the harness's `_adapt_step` becomes:

```python
def _adapt_step():
    return mesh.OT_adapt(
        T,
        refinement=args.refinement,
        coarsening=args.coarsening,
        grad_smoothing_length=args.grad_smooth_length,
        metric_choice=args.metric_choice,
        fields_to_remap=[T],
        fields_to_zero=[V, P],
        verbose=True,
    )
```

A user wanting a one-shot adapt (no time loop) writes:

```python
mesh = uw.meshing.Annulus(radiusOuter=1.0, radiusInner=0.5,
                          cellSize=1/16, qdegree=3)
T = uw.discretisation.MeshVariable(...)
# ... initialise T somehow ...
mesh.OT_adapt(T, refinement=3.0, fields_to_remap=[T])
```

## Implementation location

- Method lives on `Mesh` base class in
  `src/underworld3/discretisation/discretisation_mesh.py`
- Common implementation in
  `src/underworld3/meshing/_ot_adapt.py` (new file), called
  from the method
- Per-mesh hooks implemented in each mesh class file in
  `src/underworld3/meshing/`
- The existing `_winslow_equidistribute`'s box/ring handling
  becomes legacy; new code uses `mesh._boundary_tangent_project`

## Open questions

1. Should `fields_to_remap` default to `[field]` (i.e. just remap
   the driving field if nothing else specified)?
2. Should the post-adapt FE-remap zero out V,P automatically when
   the mesh changes topology? (Probably not — user knows their
   physics; explicit `fields_to_zero` is cleaner.)
3. Should there be a class-level constant on the mesh advertising
   whether boundary slip is supported, so the caller can check
   without try/except? E.g. `mesh.supports_boundary_slip`?
4. Naming: `OT_adapt` (PascalCase to match `CoordinateSystem`
   etc.) vs `ot_adapt` (snake_case, matches most UW3 method
   conventions)? UW3 codebase mixes both — what's the project
   preference?

## Test plan

- Unit test: `Annulus.OT_adapt(T)` on a fixed T moves mesh and
  preserves T's spatial pattern within FE-remap tolerance
- Regression test: harness using API matches hand-rolled current
  version bit-for-bit
- Negative test: `Sphere2D.OT_adapt(T)` raises
  NotImplementedError with the expected message (the
  constrained-manifold case)
- Resume test: save + restart, call `OT_adapt` — the cache
  initialises lazily from the *loaded* mesh's current coords
  (which is the deformed state at the snapshot point). For
  resume-from-snapshot scenarios the user should
  `mesh.OT_adapt_reset_reference(coords=loaded_init_coords)`
  with the explicit IC mesh, or otherwise document that resumed
  runs use the snapshot's mesh as the "reference"
