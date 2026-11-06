# Solver sequencing with pluggable DDt schemes

**Status:** notes for later, not a plan. Written 2026-11-06 during the MMPDE annulus
clean-up. Nothing here is implemented.

## The problem

Which values a field holds between a mesh move and the next transport solve depends on
the DDt plugged into its solver. Any other solver that reads that field in between gets
a different answer depending on the scheme, and nothing in the API says so.

| Scheme | Transfer of the unknown | Between the move and the next transport solve |
|---|---|---|
| `Eulerian`, `EulerianSUPG` | `REMAP` | The field on the new nodes |
| `SemiLagrangian` | `CARRY`, managed by the DDt | The old values, riding the moved nodes |

Under `CARRY` this is by design. The unknown and its history stay on the nodes, and the
next transport solve corrects for the motion with $\mathbf{u} - \mathbf{u}_\text{mesh}$.
The field is only right again after that solve.

## Measured

Annulus, radii 0.5 to 1, cell size 1/8, three levels, a sharp thermal layer, buoyancy
from $T$, rotated free slip with the radial normal, one mover call before the Stokes
solve. $v_\text{rms}$ on the uniform mesh is 1.62238.

| | $v_\text{rms}$ of the Stokes solve after the move |
|---|---|
| `SemiLagrangian` | 1.67325 |
| `EulerianSUPG` | 1.62263 |
| $T$ reset from its formula | 1.62256 |

These are from `scripts/sessions/mmpde_annulus/probe_state.py`, with PR #959's fix for a
separate stale-vector bug in the build. Before that fix the `REMAP` case read old values
as well, so the two schemes looked the same.

## Why it matters

The same model script gives different physics when the time-derivative scheme is
swapped, and it can look healthy: the solve converges and iterations do not change.
Solver order is already awkward in coupled models. Pluggable DDt schemes add a hidden
dependence, because the safe order depends on a choice made elsewhere.

## For now

One order works for every scheme: **move, then transport, then the solvers that read the
transported field.** The field is current under either transfer when they run. Transport
uses the velocity from before the move, which is the usual lag of explicit coupling.
This should go in the `adaptive-meshing` guide with the reason.

## Ideas to weigh later

1. **The DDt exposes the current field.** A read such as "the unknown on the current
   nodes" that is correct under any transfer, and that other solvers use in their
   kernels. Order stops mattering for readers. `CARRY` would need a remapped copy kept
   beside the carried values, which costs a variable and a remap per move, and the copy
   must not feed back into the history.
2. **A state flag on the variable.** The DDt marks its unknown as awaiting correction
   after a move, and a solver whose kernels read it warns. Cheap, and it makes the
   problem visible without solving it. Kernels would have to be inspected for which
   variables they read.
3. **The orchestration owns the order.** A time-stepping layer knows the dependencies
   between solvers and runs them in a valid order. The largest change, and the one that
   addresses solver sequencing in general, not only after a move.

## Questions

- Does a Stokes solve between the move and the transport step ever want the carried
  values? If not, idea 1 has no downside beyond its cost.
- The same question applies to history in a constitutive model (stress), and to several
  solvers sharing one mesh whose DDts use different transfers. Neither has been measured.
- `SemiLagrangian` on a moving mesh also drifts in this annulus case: mean $T$ rises about
  1 % over six steps where the static mesh holds it, and $v_\text{rms}$ grows faster. That
  is not explained yet, and may or may not be related.

## See also

- {doc}`eulerian-supg-transport`, moving meshes, phase 1
- {doc}`lagged-clone-sl-history`
- {doc}`/developer/guides/adaptive-meshing`
