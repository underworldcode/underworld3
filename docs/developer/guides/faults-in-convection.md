---
name: faults-in-convection
description: Recipe for a split-node fault in a TIME-DEPENDENT convection model (yielding lid, fault placed as the structured ladder by place_fault_ribbon_2d on a copy of the base's finest level, Stokes on the faulted mesh, temperature on the unfaulted base). Covers the multigrid hierarchy that keeps the faulted velocity solve bounded (the welded base with the FAC LU patch on the split finest level, ruled 2026-11-01; the split-ladder chain is the measured alternative), the warm-start rules of the rotated nonlinear loop, rebuilding the faulted mesh for a moving trench, and the traps measured on the 4 x 1 box. Reach for THIS before writing or tuning a fault-in-convection driver; for the fault mechanics on an adapted mesh see `adapt-on-top-faults`, for the yield-law solver see `nonlinear-solver`.
families: [Stokes, AdvDiffusion]
kind: recipe
---

# Faults in convection

A split-node fault in a convecting, yielding lid. The pattern measured on a 4 x 1 box
(2026-10, on `feature/box-faults`), with the fault-multigrid library that has been on
`development` since 2026-10-31 (#889, #890, #912):

- the base mesh carries the temperature and its transport and is never cut;
- the faulted mesh is a copy of the base's finest level with the fault placed as the
  structured ladder, the spine split and in frictionless contact from three rung
  spacings below the surface, the last rungs to the surface a weak isotropic bridge;
- Stokes is solved on the faulted mesh, and the velocity is evaluated back onto the
  base for the transport step.

## The multigrid hierarchy: the welded base, the split finest level, the LU patch

The production tail (Louis, 2026-11-01, on #892) is the base's own levels under the
faulted mesh: `set_custom_fmg(stokes, base.multigrid_levels, field_id=0)`, where
`Mesh.multigrid_levels` returns the levels coarsest first with the base last, and the
faulted mesh, a placement on the base, is the finest level above them. Only that
finest level carries the fault. What makes the welded coarse spaces enough is the
patch on the finest level: an additive-Schwarz component over the split's coincident
pairs, installed without being asked for when the sole split level is the finest, its
blocks solved by LU with a nonzero shift (#912).

```python
from underworld3.utilities.custom_mg import set_custom_fmg
from underworld3.utilities.place_surface import place_fault_ribbon_2d

spine, trace = fault_trace()
faulted, info = place_fault_ribbon_2d(base, [("Fault", trace)], 2 * S,
                                      spines=[("Fault", spine)], margin_rings=1,
                                      clearance=0.7, split=True, mesher="ladder")
stokes = uw.systems.Stokes(faulted, ...)
set_custom_fmg(stokes, base.multigrid_levels, field_id=0)   # coarsest .. base; faulted on top
```

Measured on UWTN 2026-022's fixtures in the tail sweep and reproduced on #912's build
(velocity iterations as the rotated loop reports them; the band ratio is the base's
finest cell size over the rung spacing $S$):

| | band ratio | welded base + LU patch | lean split chain |
|---|---|---|---|
| linear fault, np 1, over all applications | 6, 12, 24 | 66, 65, 67 | 83, 118, 166 |
| box with a fault, np 4, per Newton increment | 8, 16, 25 | 7, 7, 7 | 28-40, 45-53, 67-76 |

The Newton path and the peak slip are the same under both tails, and the welded tail's
counts do not move with the ratio.

**The split chain, the alternative.** Until 2026-11-01 this guide recommended the lean
chain: the same ladder at $2S$ placed into a copy of the base's finest level and at
$4S$ into a copy of the level below, each split along its rungs, under the faulted
mesh, with the coarser base levels welded (`uw_chain = 2` in the drivers). That rested
on counts measured while the finest patch's blocks ran unshifted ILU(0) instead of the
configured LU (#892): under those the welded tail capped at 200 velocity iterations
once the lid mobilised and the chain did not. The chain's levels never carry the patch
(a pair touching a split coarse level skips FAC detection), so its counts did not
change with the fix, and with the LU blocks it costs 3 to 16 times the welded tail's
iterations on the same states. It stays in the drivers as the measured alternative,
for the case where the LU patch is unavailable or its block too large (the block grows
with the band's rows; 3-D is not measured). On the production faulted annulus at
np 16 (kaiju, `development` 25d62812, steps from a TAf checkpoint; the welded arm with
a 3584-row patch on the one rank the default gather placement gives the band to, the
chain with none) the two tails do the same velocity
work, 8.65 against 8.74 F-cycles per solve, in the same Stokes time within the ±10 %
run-to-run spread, with the same Newton counts and answers; the box's gain does not
show there because that model's band ratio is 3.9, below the smallest ratio in the
table. The welded tail is no worse and has one hierarchy fewer
to build, so the drivers' `uw_chain` default moves to 0 as the campaign builds move
to `development`. Whether split coarse levels stay a library option is not decided. The record, with the superseded
table, is the journal topic
[Fault multigrid](https://github.com/underworldcode/underworld3-journal/blob/main/topics/fault-multigrid.md)
(entries 2026-10-29, 2026-10-31 and 2026-11-01).

## The rotated nonlinear loop

`add_fault_bc` sends the solve through the rotated strong-constraint loop
(`rotated_bc.solve_rotated_freeslip`). Walls with plain essential BCs compose with it;
block-constraint multipliers do not. A closed box needs
`petsc_use_pressure_nullspace = True`.

- `solve(picard=N)` with the default (consistent Newton) tangent takes at least N
  Picard iterations on a blended Jacobian, then Newton;
  `stokes.consistent_jacobian = "continuation"` switches from Picard to Newton by
  itself.
- The first solve on a faulted mesh starts from rest. A warm start from a velocity
  computed without the fault stalls the Picard line search (measured 8e-2 of the
  reference residual after 2 iterations); from rest the solve converges in 18-34 Picard
  and 5-8 Newton iterations on the states measured.
- The loop switches from Picard to Newton when the residual has fallen by
  `newton_switch_rtol` relative to the same reference as its convergence test (the
  larger of this solve's starting residual and the rest-state residual), so a warm
  start near convergence reaches Newton without first shrinking its own residual a
  hundredfold under Picard. (Before 2026-10-23 the switch was measured from the warm
  start's own residual, warm steps stayed in Picard to the end, and the advice was
  `newton_switch_rtol = 1.0`; that is no longer needed.)

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
