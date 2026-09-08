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

The comparisons and implicit timestep policies below describe the default
`EulerianSUPG` manager. The optional `EulerianSUPGPC` manager has a different
update and timestep policy; see [Predictor-corrector transport](#predictor-corrector-transport).

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

## Predictor-corrector transport

For continuous-P1 scalar transport, select the predictor-corrector manager
explicitly and pass it to the same `uw.systems.AdvDiffusion` solver:

```python
Tdot = uw.discretisation.MeshVariable("Tdot", mesh, 1, degree=1)
transport = uw.systems.ddt.EulerianSUPGPC(
    mesh, T, v.sym,
    method="citcoms",
    temperature_rate_field=Tdot,
)
adv = uw.systems.AdvDiffusion(mesh, T, v.sym, DuDt=transport)
adv.constitutive_model = uw.constitutive_models.DiffusionModel
adv.constitutive_model.Parameters.diffusivity = 1.0
adv.add_dirichlet_bc(0.0, "Upper")
adv.add_dirichlet_bc(1.0, "Lower")
adv.solve(timestep=adv.estimate_dt())
```

Here `T` must also be continuous P1. The manager owns the rate, startup state,
correction controls and transport policy; the solver owns the constitutive
model, source and boundary conditions. Select `method="citcoms"` for the
fixed-correction benchmark method or `method="pc_converged"` for a
residual-converged reference. These are manager methods, not solver time
integrator arguments. CN and BDF remain on the default `EulerianSUPG` manager.

### Predictor and corrections

Writing the temperature rate as $q$, the predictor is
$T^{(0)} = T^n + (1-\gamma)\Delta t\,q^n$, followed by resetting $q$ to zero.
Each correction assembles the full finite-element residual $F(T,q)$ and applies

$$
\delta q = -D^{-1}F(T,q), \qquad
q \leftarrow q + \delta q, \qquad
T \leftarrow T + \gamma\Delta t\,\delta q,
$$

where $D$ is the positive row-lumped mass. Dirichlet values are reinserted at
each correction. `method="citcoms"` defaults to `adv_gamma=0.5` and
`corrector_steps=2`, with a single lumped correction to initialise the rate.

Both PC methods use the steady directional simplex stabilisation

$$
\tau = \frac{h}{2|\mathbf{u}|}\max(0,1-1/Pe), \qquad
Pe = \frac{|\mathbf{u}|h}{2\kappa}, \qquad
h = \frac{2|\mathbf{u}|}{\sum_a |\mathbf{u}\cdot\nabla N_a|}.
$$

Zero velocity gives zero tau; zero diffusivity uses the advective limit.
This is not the default manager's transient norm tau or cell-Peclet weighting.
`tau=None` selects this automatic rule; a scalar symbolic expression or number
overrides it. `supg_weight=1.0` is the PC default.
Automatic geometry supports 2-D triangles and 3-D tetrahedra and currently
requires a non-empty volume partition on every rank. Unsupported layouts
are rejected collectively; use fewer ranks or a sufficiently resolved mesh.

After attachment to the solver, `transport.estimate_dt()` returns
`0.9*min(dt_adv, dt_diff)`, using the directional advective rate and a row-sum
bound on the lumped diffusion operator; `adv.estimate_dt()` delegates to it.
The implicit field-change estimate described above is not a stability bound
for this update. Fixed comparison timesteps must respect the PC bound;
residual convergence does not make diagonal correction converge for arbitrary
steps. SUPG does not guarantee a nodal maximum principle: check temperature
bounds and heat balance.

Diffusion remains in the Galerkin flux but is absent from the strong SUPG
residual. This omission is exact for affine P1 fields with elementwise
constant diffusivity, not for arbitrary curved mappings, variable
coefficients or P2 temperature.

### Finite-correction accuracy

The correction mass is lumped, but the time derivative in the residual uses
the consistent finite-element mass. Consequently `adv_gamma=0.5` and two
corrections do **not** guarantee second-order temporal convergence for a
nonuniform field at fixed mesh. For pure diffusion, let $M$ be the consistent
mass, $K$ the stiffness and $D=\operatorname{diag}(M\mathbf{1})$. Two corrections
approach the operator $(2I-D^{-1}M)D^{-1}K$ as $\Delta t$ vanishes, generally
different from both $M^{-1}K$ and $D^{-1}K$. The startup rate $-D^{-1}KT$ is
also only an approximation to the consistent rate $-M^{-1}KT$.

The regression in `tests/test_1118_pc2_diffusion_time.py` isolates these
effects with independently integrated element
matrices and exact discrete eigenmode/matrix-exponential solutions on tiny
triangular and tetrahedral meshes. It records first-order timestep
differences in serial and MPI. Uniform scalar decay, where the two masses
agree, is not sufficient evidence of PDE time accuracy. The same reference
records temporal order 2.00 for an actual UW3 consistent-mass CN update in
both geometries, in serial and on eight ranks, with its nodal amplification
map agreeing within 1.6e-14. The DDt-manager migration reproduced these
results on 8 September 2026. These are isolated numerical checks, not
production-scale validation.

### Residual-converged reference

Construct `EulerianSUPGPC` with `method="pc_converged"` to keep the same SUPG
residual and gamma update while using the lumped mass only as an iterative
preconditioner. At startup it converges the consistent Petrov-Galerkin rate
equation with temperature held fixed; after prediction it converges the
coupled rate/temperature correction. The full residual must be no larger
than `max(corrector_atol, corrector_rtol*initial_residual)`. Defaults are
`corrector_rtol=1e-10`, `corrector_atol=1e-12` and
`max_corrector_steps=100`. Non-convergence raises `RuntimeError` instead of
accepting the step. Inspect `transport.temperature_rate`,
`transport.last_corrector_iterations`, `transport.last_corrector_residual`
and `transport.corrector_target` on the manager.

With `adv_gamma=0.5`, this supplies a separate second-order reference rather
than changing the fixed-correction CitcomS method. The discrete diffusion
regression records order 2.00 in 2-D and 3-D, in serial and on
eight ranks, and agreement with the trapezoidal amplification map below
5.2e-14. At relative tolerance `1e-12`, those small meshes needed 48-63
corrections per step in 2-D and 63-81 in 3-D. This is an accuracy reference,
not evidence that diagonal iteration is the most efficient production
consistent-mass solve. Changing residual mass or correction count changes
the fixed-correction method and must not be presented as unchanged paper
reproduction.

### Checkpoint state

```python
orchestration_model = uw.get_default_model()
orchestration_model.save_state(file="checkpoint.h5")
# Reconstruct the matching model, fields and manager before loading.
orchestration_model.load_state("checkpoint.h5")
```

An exact PC restart needs temperature plus the manager's rate, startup state
and correction controls. A temperature-only checkpoint is insufficient.
Implicit integration instead needs its DDt fields, timestep history, theta
and field-change estimator state. Disk snapshots require the same model
layout and MPI rank count; snapshots with a different solver/manager layout
require migration. The manager owns PC restart state, not a solver alias.

## Further reading

- Design note and measurements: `docs/developer/design/eulerian-supg-transport.md`
- The semi-Lagrangian schemes: {doc}`semi-lagrangian-time-integration`
- Example: `docs/examples/convection/advanced/Ex_AdvectionDiffusionSUPG_RotationTest.py`
