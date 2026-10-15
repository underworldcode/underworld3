---
name: faults-in-convection
description: Recipe for a split-node fault in a TIME-DEPENDENT convection model (yielding lid, fault placed as the structured ladder by place_fault_ribbon_2d on a copy of the base's finest level, Stokes on the faulted mesh, temperature on the unfaulted base). Covers the multigrid hierarchy that keeps the faulted velocity solve bounded (the lean split-ladder chain, "C on the ladder"), the warm-start rules of the rotated nonlinear loop, rebuilding the faulted mesh for a moving trench, and the traps measured on the 4 x 1 box. Reach for THIS before writing or tuning a fault-in-convection driver; for the fault mechanics on an adapted mesh see `adapt-on-top-faults`, for the yield-law solver see `nonlinear-solver`.
families: [Stokes, AdvDiffusion]
kind: recipe
---

# Faults in convection

A split-node fault in a convecting, yielding lid. The pattern measured on a 4 x 1 box
(2026-10, branch `feature/box-faults`), with the fault-multigrid library of
`feature/fault-split-multigrid`:

- the base mesh carries the temperature and its transport and is never cut;
- the faulted mesh is a copy of the base's finest level with the fault placed as the
  structured ladder, the spine split and in frictionless contact from three rung
  spacings below the surface, the last rungs to the surface a weak isotropic bridge;
- Stokes is solved on the faulted mesh, and the velocity is evaluated back onto the
  base for the transport step.

## The multigrid hierarchy: split the coarse levels

`set_custom_fmg(stokes, base_levels, field_id=0)` with the base's own levels gives the
WELDED hierarchy (A in `design/FAULT_MULTIGRID_HIERARCHIES.md`): only the finest level
carries the fault. Its coarse spaces cannot represent the slip, the weak bridge or the
ladder cells, and the velocity sub-solve caps at 200 iterations once Newton starts on a
mobilising lid. A placed ladder is not replayed onto the levels by `add_fault`, so the
split levels have to be built by hand: the same ladder at twice and four times the rung
spacing, placed on the base levels one and two steps below the finest and split along
their own rungs, as the mid levels.

```python
from underworld3.utilities.custom_mg import set_custom_fmg
from underworld3.utilities.place_surface import place_fault_ribbon_2d

base_levels = base._coarse_level_meshes()[:-1] + [base]   # coarsest .. finest

def placed_ladder(host, coarsening):
    # the spine and trace sampled at coarsening x the rung spacing S; the finest
    # sample count is a multiple of 2**chain, so every coarser sampling is a subset
    spine, trace = fault_trace(coarsening)
    width = 2 * S * coarsening                 # two of the level's own rungs
    mesh, info = place_fault_ribbon_2d(host, [("Fault", trace)], width,
                                       spines=[("Fault", spine)], margin_rings=1,
                                       clearance=0.7, split=True, mesher="ladder")
    return mesh, info

chain = 2
mids = [placed_ladder(base_levels[len(base_levels) - k], 2 ** k)[0]
        for k in range(chain, 0, -1)]
faulted, info = placed_ladder(base, 1)
stokes = uw.systems.Stokes(faulted, ...)
set_custom_fmg(stokes, base_levels[:len(base_levels) - chain] + mids, field_id=0)
```

Measured on one hard state (a mobilising lid, slip ~270, one solve from rest, 4 ranks;
velocity iterations as the rotated loop reports them per Newton increment):

| Hierarchy | Levels (cells) | Newton increments | Velocity its, total | Velocity its per increment, Newton phase | Caps (200) |
|---|---|---|---|---|---|
| A, welded | 642, 2568, 10272, 11604 | 39 | 1849 | 71-200 | 2 |
| ladder chain all on the finest | 642, 2568, 10272, 10573, 10966, 11612 | 42 | 636 | 26-29 | 0 |
| **lean chain** (above) | 642, 2924, 10950, 11612 | 42 | 695 | 29-38 | 0 |

A cycle of the chain placed entirely on the finest level visits three levels of ~11k
cells, about twice the work of a lean or welded cycle, so the lean chain is the cheapest
per solve. In the time-stepping run every velocity sub-solve capped under A once the lid
mobilised; under the lean chain none did, at 4-5 Newton increments per step and the same
slip. A band of constant width (two finest rungs) on every level gave the same counts as
the widened bands. The band patch smoother (`fac_zone=band`) did not help the welded
hierarchy: 114-150 velocity iterations per increment against 91-162 without.

## The rotated nonlinear loop

`add_fault_bc` sends the solve through the rotated strong-constraint loop
(`rotated_bc.solve_rotated_freeslip`). Walls with plain essential BCs compose with it;
block-constraint multipliers do not. A closed box needs
`petsc_use_pressure_nullspace = True`.

- Picard steps on this path need `stokes.consistent_jacobian = "continuation"`;
  `solve(picard=N)` with the default tangent raises.
- The first solve on a faulted mesh starts from rest. A warm start from a velocity
  computed without the fault stalls the Picard line search (measured 8e-2 of the
  reference residual after 2 iterations); from rest the solve converges in 18-34 Picard
  and 5-8 Newton iterations on the states measured.
- The loop switches from Picard to Newton when the residual has fallen by
  `newton_switch_rtol` from THIS solve's starting residual, while its convergence test
  is relative to the cold reference. A warm start already near convergence therefore
  stays in Picard to the end (measured 13-27 iterations at ~0.85 per iteration). After
  the first solve set `stokes.newton_switch_rtol = 1.0`: warm steps then take 2 Newton
  iterations.

## A moving trench

A fault fixed in space loses its slab once the convergence zone migrates; the slab
stays on the fault when the trench moves with the overriding plate (the median surface
velocity over a window beyond the fault, not the sliver next to it). The faulted mesh is
then rebuilt as the trench moves:

- carry the solution across a rebuild by evaluating the old velocity and pressure at
  each new node shifted back by the trench displacement, so the old slip lands on the
  new fault;
- start the rebuilt solve with Picard to a residual reduction of 1e-2 of its own start.
  Newton from the transferred field stalls (measured 40 iterations at 0.1 of the
  reference residual), and so does Newton after one order of Picard reduction;
- rebuild every few rungs (four, half a base cell here), not every rung: a rebuild step
  costs about twice an ordinary one.

## Cost and diagnostics

- Through an overturn the nonlinear solve, not the multigrid, sets the cost: 15-25
  Newton increments per step on the moving-trench run while the velocity sub-solve
  stayed bounded.
- Compare solver cost by iteration counts (Newton increments, velocity cycles), not
  seconds: on a laptop the wall time moves with power state and load.
- A refined mesh read back from its checkpoint has its cells joined to the wrong
  vertices (open roadmap item). Renderers rebuild the mesh from the run's parameters
  and remap the fields by coordinate; draw P1 fields on the mesh's own cells, not on a
  Delaunay of their DOFs.
