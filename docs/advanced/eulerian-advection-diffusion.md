# Advection-diffusion composed from a transport manager (Eulerian SUPG by default)

`uw.systems.AdvDiffusion` is the general scalar transport solver. It assembles the
diffusive flux and the source itself and takes the transport from the history manager
it holds (`DuDt`); with the default manager, `uw.systems.ddt.EulerianSUPG`, it is the
Eulerian SUPG scheme this page describes, a drop-in for the semi-Lagrangian solver
`uw.systems.AdvDiffusionSLCN`. Both solve

$$
\frac{\partial \phi}{\partial t} + \mathbf{u}\cdot\nabla\phi
    - \nabla\cdot(\kappa\nabla\phi) = f ,
$$

but assembles every term on the mesh, implicit in time, with streamline-upwind
(SUPG) stabilisation. There is no trace-back and no departure point. The two
classes share their interface, so switching is one line:

```python
adv = uw.systems.AdvDiffusion(mesh, T, v.sym, order=1)   # was AdvDiffusionSLCN
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

A practical rule: if the timestep is chosen so that the temperature field itself
is resolved in time (a fraction of a feature width per step), the Eulerian solver
is cheaper and more accurate; if the step is deliberately long relative to the
transported features, the semi-Lagrangian solver is the one that survives it.

## The three transport managers

`DuDt` selects the transport, and three managers are worth considering for a
scalar field. The choice turns on the Courant number the model runs at and on
whether the model is carrying particles for another reason.

**Eulerian SUPG** (`uw.systems.ddt.EulerianSUPG`, the default) is the general
choice. Its error falls as $\Delta t^2$, it puts no lower limit on the Courant
number, and it conserves the integral of the transported field to solver
tolerance. On the LeVeque deformation test at its standard period it matches the
integration-point history's accuracy at a third of the cost, and holds the
enclosed volume to 4e-5 against that scheme's 5e-3. Use it unless something
below applies.

**Semi-Lagrangian on the integration points**
(`uw.systems.ddt.IntegrationPointSemiLagrangian`) is the accurate choice at
larger Courant numbers. Its error is flat between Courant 0.5 and 2, so a model
that takes long steps keeps its accuracy where the Eulerian scheme loses it, and
it loses 45 times less of the second moment than the nodal scheme does. It
carries a history at the integration points rather than at nodes, which is the
reason to prefer it for tensor transport, where the extra sub-cell resolution
has more to represent. It has a low Courant number limit: the fit it performs is
not contractive under pure advection, and below about Courant 0.5 a mode grows.
The growth is suppressed by physical diffusion and is unreachable when the
timestep comes from the Courant condition, but it is a real limit for
advection-dominated flow with sharp interfaces. Adding diffusivity to damp it
makes the answer worse at every strength, so the limit is a reason to choose a
different manager rather than something to correct. The measurements are in
`docs/developer/subsystems/integration-point-variables.md`.

**Lagrangian on a swarm** (`uw.systems.ddt.Lagrangian_Swarm`) transports the
field on particles. It is worth using when the model already carries a swarm for
material tracking, so the transport rides on particles it is advecting anyway.
We would not introduce particles in order to use it.

**Semi-Lagrangian at the nodes** (`uw.systems.ddt.SemiLagrangian`) remains the
historical default of `AdvDiffusionSLCN`. It re-interpolates once per step, which
costs it accuracy at small Courant numbers, and on a deforming flow with a sharp
interface it diverges below a Courant number that depends on the problem. Prefer
one of the three above.

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
  `solver.supg_weight = 0` gives the plain Galerkin scheme for comparison. The term is
  also weighted by the cell Péclet number, $Pe^2/(Pe^2 + Pe_c^2)$ with
  $Pe = |\mathbf{u}| h / 2\kappa$ and $Pe_c$ the `peclet_weight` argument (default 4),
  so the stabilisation is off where a cell is diffusion-dominated and full where advection
  dominates (pure advection, $\kappa = 0$, is unaffected); `peclet_weight=0` gives the
  uniform weight.
- The linear system is nonsymmetric, so the solver uses GMRES with an
  additive-Schwarz ILU preconditioner, with the Krylov tolerance matched to the
  SNES tolerance so that a step is one Newton iteration. Measured, this is the
  cheaper solve at every Courant number up to eight ranks and its iteration
  count does not grow with the rank count. `solver.preconditioner = "fmg"`
  switches to geometric multigrid over the mesh's refinement hierarchy
  (`refinement >= 1`) for very large rank counts. Every option can be
  overridden through `solver.petsc_options`.

## The history manager is the transport plugin

The solver does not assemble its transport itself. Its history manager (`solver.DuDt`)
contributes three symbolic terms, and the solver composes its residual from them:
the time derivative of the scheme, the advection, and the stabilisation flux of the
strong residual. The default manager is `uw.systems.ddt.EulerianSUPG`, which owns the
advecting velocity (`V_fn` is data on it), the time scheme (`order`, `theta`), and the
stabilisation knobs (`supg_weight`, `tau_weights`, `tau_shape`, `peclet_weight`); the
solver's properties of the same names pass through to it.

Any history manager that follows the contract can be supplied instead. A semi-Lagrangian
manager answers zero for the advection and the stabilisation, because its history is
already traced back along the characteristics, so the same solver becomes a
semi-Lagrangian scheme on the field history:

```python
history = uw.systems.ddt.SemiLagrangian(mesh, T.sym, v.sym, vtype=uw.VarType.SCALAR,
                                        degree=T.degree, continuous=True, order=1)
adv = uw.systems.AdvDiffusion(mesh, T, v.sym, DuDt=history)   # no assembled advection
```

On pure advection this reproduces `AdvDiffusionSLCN` to the solver tolerance; with
diffusion the two differ in where the diffusive flux history comes from (the traced-back
field here, the traced-back flux there). The manager works for a vector or tensor unknown
as well (`vtype`), applying the advection component by component, which is how the
Navier-Stokes solver and a transported stress use it.

## Further reading

- Design note and measurements: `docs/developer/design/eulerian-supg-transport.md`
- The semi-Lagrangian schemes: {doc}`semi-lagrangian-time-integration`
- Example: `docs/examples/convection/advanced/Ex_AdvectionDiffusionSUPG_RotationTest.py`
