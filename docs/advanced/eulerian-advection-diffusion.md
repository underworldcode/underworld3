# Eulerian advection-diffusion (SUPG): a drop-in for SLCN

`uw.systems.AdvDiffusionSUPG` solves the same scalar transport equation as the
semi-Lagrangian solver `uw.systems.AdvDiffusionSLCN`,

$$
\frac{\partial \phi}{\partial t} + \mathbf{u}\cdot\nabla\phi
    - \nabla\cdot(\kappa\nabla\phi) = f ,
$$

but assembles every term on the mesh, implicit in time, with streamline-upwind
(SUPG) stabilisation. There is no trace-back and no departure point. The two
classes share their interface, so switching is one line:

```python
adv = uw.systems.AdvDiffusionSUPG(mesh, T, v.sym, order=1)   # was AdvDiffusionSLCN
adv.constitutive_model = uw.constitutive_models.DiffusionModel
adv.constitutive_model.Parameters.diffusivity = 1.0e-3
adv.add_dirichlet_bc(1.0, "Bottom")
adv.add_dirichlet_bc(0.0, "Top")

dt = adv.estimate_dt()          # accuracy-based: 2% of the field's range per step
adv.solve(timestep=dt)
```

The one deliberate difference is the timestep estimate. The semi-Lagrangian
`estimate_dt` reports the cell-crossing time, which for this solver is neither
a stability limit nor an accuracy one. The Eulerian solver's `estimate_dt`
instead returns the step at which the field changes by a given fraction of its
range (0.02 by default), from the advective rate before the first solve and
from the rate the last step actually produced after it. It does not depend on
the mesh, so cells refined for the Stokes problem do not shrink it. A script
that sizes its step in Courant numbers can still ask for
`estimate_dt(basis="resolution")`.

## What carries over

| SLCN | SUPG | note |
|---|---|---|
| `order=1, theta=0.5` | same | Crank-Nicolson, the default for both |
| `order=1, theta=1.0` | same | backward Euler |
| `order=2, theta=1.0` | same | SL-BDF2 becomes BDF2 |
| `order=2, theta=0.5` | refused | refused for the same reason: a BDF stencil does not pair with a centred flux |
| `f`, `V_fn`, `constitutive_model`, `delta_t` | same | |
| `estimate_dt()` | accuracy-based by default | the field may change by `fraction` (0.02) of its range per step; `basis="resolution"` returns the cell-crossing time SLCN reports |
| `solve(zero_init_guess, timestep, ...)` | same | |
| `DuDt.set_initial_history(values, dt)` | same | plant an exact history to start at full order |
| `restore_points_func`, `monotone_mode`, `old_frame_traceback`, `DFDt` | ignored, with a warning | they configure the trace-back |

`order=3` (BDF3) is available; see below for when it is safe.

## When to use which

Both solvers are free of any stability limit on the timestep, so cells refined
for the Stokes problem never dictate the transport step. They differ in what
bounds their accuracy and in what a step costs.

**Eulerian SUPG.** The error is set by how far the transported feature moves per
step relative to its own width, as $(\mathbf{u}\Delta t)^2$ for the second-order
schemes. It does not depend on the cell size at all: on a rotating Gaussian a band
refined to $h/9$, with its cells at a local Courant number of 13, changes the error
in the third digit only. A step costs one nonsymmetric solve, four to six times
less than a semi-Lagrangian step in serial, and it needs no departure points in
parallel. On a moving mesh the field and its history are re-interpolated by the
ordinary remesh transfer, so no special staging is needed.

**Semi-Lagrangian.** The error is nearly independent of the timestep but
accumulates one interpolation per step, so at small Courant numbers it is the
worse scheme (21% against 0.6% after one revolution at Courant 0.5 on the same
mesh). Its limit is the arc a characteristic turns per step, about 10 degrees for
the RK2 trace-back, a property of the flow rather than the mesh. Above roughly
Courant 2 on the feature's own scale it keeps its accuracy where the Eulerian
scheme loses it.

In parallel the Eulerian step stays seven times cheaper in serial and about five
times at eight ranks (the departure-point search parallelises perfectly, the
ILU preconditioner a little less), and its answer is identical to ten digits
at every rank count, where the semi-Lagrangian answer moves with the partition.

A practical rule: if the timestep is chosen so that the temperature field itself
is resolved in time (a fraction of a feature width per step), the Eulerian solver
is cheaper and more accurate; if the step is deliberately long relative to the
transported features, the semi-Lagrangian solver is the one that survives it.

## Choosing the time scheme

Measured on a rotating Gaussian, one revolution, relative $L_2$ error; the full
tables are in the design note.

| scheme | behaviour |
|---|---|
| Crank-Nicolson (`order=1`) | three to four times more accurate than BDF2 at the same timestep below Courant 2; rings once the feature is under-resolved in time |
| BDF2 (`order=2`) | damped and stable at every Courant number; the choice for sharp or under-resolved fields |
| BDF3 (`order=3`) | the most accurate scheme below Courant 1 when diffusion is present; on pure advection it grows slowly at any Courant number, so use it only with diffusion |
| backward Euler (`order=1, theta=1.0`) | 20 to 40% error at any practical timestep; not for transport |
| Adams-Moulton 2, 3 (not offered) | third and fourth order below Courant 1 but blow up on advection from about Courant 1, which is why there is no knob for them |

All schemes cost the same per step: the history terms are extra kernel inputs,
not extra solves. Changing the timestep between steps changes a runtime constant
of the compiled kernels; nothing is recompiled.

## Details that differ from SLCN

- The strong residual used in the SUPG term carries the time derivative and the
  advection but no diffusion term, because PETSc's pointwise kernels see first
  derivatives only. For linear elements the missing term is identically zero.
- The stabilisation parameter uses the local cell size (`mesh.cell_size()`) and
  three weights that are runtime constants (`solver.tau_weights`);
  `solver.supg_weight = 0` gives the plain Galerkin scheme for comparison.
- The linear system is nonsymmetric, so the solver uses GMRES with an
  additive-Schwarz ILU preconditioner, with the Krylov tolerance matched to the
  SNES tolerance so that a step is one Newton iteration. Measured, this is the
  cheaper solve at every Courant number up to eight ranks and its iteration
  count does not grow with the rank count. `solver.preconditioner = "fmg"`
  switches to geometric multigrid over the mesh's refinement hierarchy
  (`refinement >= 1`) for very large rank counts. Every option can be
  overridden through `solver.petsc_options`.

## Further reading

- Design note and measurements: `docs/developer/design/eulerian-supg-transport.md`
- The semi-Lagrangian schemes: {doc}`semi-lagrangian-time-integration`
- Example: `docs/examples/convection/advanced/Ex_AdvectionDiffusionSUPG_RotationTest.py`
