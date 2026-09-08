# Eulerian SUPG transport: design and measurements

**Status**: implemented on `feature/eulerian-supg-transport` (2026-09-02), static mesh.

**Credit.** The SUPG weak form used here (the test-function perturbation written as
a flux, so PETSc needs no modified test space), its first working implementation on
PetscDS with P2 elements, the LeVeque swirling-flow comparison against SLCN and the
conservative level-set pipeline that motivated it are NengLu's, on the `levelset`
branch of issue #657. This note builds on that prototype: same formulation and
stabilisation parameter, time integration moved onto the symbolic history
machinery, and the measurements added.

## Why an Eulerian scheme

Underworld3 meshes are usually refined for the momentum problem: faults, viscosity
jumps, boundary layers. A transported scalar rarely needs that resolution, so a
scheme whose timestep is bounded by the smallest cell pays for cells it does not
use. The semi-Lagrangian solver (`AdvDiffusionSLCN`) escapes that bound but pays
for departure points, which are expensive per step and irregular in parallel, and
its moving-mesh staging needs a lagged copy of the previous geometry.

An implicit Eulerian scheme has no stability bound at all. Its cost is a
nonsymmetric solve per step, and its accuracy is bounded by how far the transported
feature moves in one step. The measurements below say when each is the better tool.

## The scheme

The equation is

$$
\frac{\partial \phi}{\partial t} + \mathbf{u}\cdot\nabla\phi
    - \nabla\cdot(\kappa\nabla\phi) = f .
$$

Every past time level $\phi^{n}, \phi^{n-1}, \dots$ is a mesh variable held by an
`Eulerian` history manager, so first derivatives of past states are available in
the kernels and two multistep families share one code path:

| family | time derivative | spatial operator |
|---|---|---|
| BDF, order $N$ (`order=2, 3`) | $\frac{1}{\Delta t}\sum_{k=0}^{N} c_k\,\phi^{n+1-k}$ | at $n+1$ only |
| theta rule (`order=1`; Adams-Moulton of $N$ steps internally) | $\frac{\phi^{n+1}-\phi^{n}}{\Delta t}$ | $\sum_{k=0}^{N} a_k\,S(\phi^{n+1-k})$ |

with $S(\phi) = \mathbf{u}\cdot\nabla\phi - \nabla\cdot(\kappa\nabla\phi)$ and the
coefficients those the history manager already maintains (`theta` is the
Adams-Moulton weight at order 1; 0.5 is Crank-Nicolson). Both families ramp from
first order over the opening steps unless `solver.DuDt.set_initial_history` plants
the history. The pointwise residual is

$$
f_0 = R(\phi), \qquad
\mathbf{f}_1 = \sum_k w_k\,\kappa\nabla\phi^{n+1-k} + \tau\,R(\phi)\,\mathbf{u},
$$

where $R$ is the strong residual of the chosen scheme (time derivative, advection
and source) and $w_k$ the spatial weights of the family. The SUPG term is the
Petrov-Galerkin test-function perturbation $\tau\,\mathbf{u}\cdot\nabla w$ written
as a flux against $\nabla w$, so PETSc needs no modified test space.

$$
\tau = \left[\left(\frac{2 c_0}{\Delta t}\right)^2
    + \left(\frac{2|\mathbf{u}|}{h}\right)^2
    + \left(\frac{4\kappa}{h^2}\right)^2\right]^{-1/2},
\qquad h = \texttt{mesh.cell\_size()} .
$$

### Decisions and their reasons

- **No diffusion in the strong residual.** PETSc's pointwise kernels see first
  derivatives only, so $-\nabla\cdot(\kappa\nabla\phi)$ cannot appear in $R$. For
  linear elements it vanishes identically; for higher orders this is the usual
  inconsistency of SUPG without a Laplacian reconstruction. Diffusion enters as the
  Galerkin flux only.
- **Every knob is a runtime constant.** The timestep, the multistep coefficients,
  the three weights in $\tau$ and the overall SUPG weight are UW expressions routed
  through PETSc's `constants[]` array. A change of timestep costs nothing; the
  prototype recompiled its kernels on every change (1.2 s against 0.03 s for a step).
- **Diffusivity on the constitutive model**, as for every scalar solver, starting at
  $\kappa = 0$. The prototype carried a float attribute with a warning bridge.
- **Additive-Schwarz ILU, one Newton iteration per step.** The operator is
  nonsymmetric, so the smoother and the outer Krylov solver have to be safe for
  one. Measured (below), GMRES with an additive-Schwarz ILU preconditioner is the
  cheaper linear solve at every Courant number from 1/2 to 32 and its iteration
  count does not change between one and eight ranks; geometric multigrid's cycle
  count grows with the Courant number nearly as fast, and a cycle costs about
  three Schwarz iterations. The linear solve is under a tenth of a step either
  way; assembly is the rest. What did matter was the tolerance pair: the Krylov
  default (1e-5) does not reach the SNES tolerance (1e-8), so the SNES took a
  second Newton step on a linear operator, and that Jacobian assembly cost more
  than every linear solve of the step. The Krylov tolerance is now 1e-9.
  `preconditioner = "fmg"` hands the block to the managed multigrid route
  (custom-P transfers over the refinement hierarchy or an adapt child's coarse
  tail, flexible GMRES outside), for the rank count where a one-level method
  runs out of coarse space. The solver's `solve()` builds through the base
  `_build`, which is where a preconditioner choice is resolved; the
  semi-Lagrangian solvers run the three setup stages directly and their
  `preconditioner` property is inert as a result (#683).
- **Moving meshes, phase 1.** The unknown and its history stay on the default
  `REMAP` transfer policy with the material velocity. The remap re-interpolates old
  states onto the new nodes, so the Eulerian form is already correct to
  interpolation accuracy. The `CARRY` + $\mathbf{u} - \mathbf{u}_\text{mesh}$ form
  is phase 2 and must not be mixed with `REMAP`.
- **Not yet:** discontinuity capturing (the prototype's residual omitted the time
  derivative and added first-order diffusion everywhere; a correct lagged residual
  needs $\phi^{n-1}$), a streamline element length from a mesh-owned metric tensor,
  the ALE hook, and vector or tensor unknowns: the solver is scalar, where the
  semi-Lagrangian trace-back carries vectors and tensors through the same machinery.

## Measurements

Rotating Gaussian (`uw.analytic.RotatingGaussian`, $\sigma = 0.12$, orbit radius
0.5), P2 field, unstructured simplex box, one revolution; relative $L_2$ error at
the end. "Courant" is on the cell size. Study scripts and CSVs are in
`~/+Simulations/supg_vs_slcn_657/`.

### Eulerian against semi-Lagrangian (the #657 prototype, Crank-Nicolson)

| mesh | Courant | SUPG CN | SLCN | cost per step SUPG : SLCN |
|---|---|---|---|---|
| uniform 32 | 0.5 | 0.6% | 21% | 1 : 6.3 |
| uniform 32 | 2 | 9.8% | 7.7% | 1 : 6.4 |
| uniform 32 | 8 | 66%, min $-0.35$ | 8.8% | 1 : 5.6 |
| uniform 32 | 32 | 113% | 93%, mass $-32$% | 1 : 5.7 |
| uniform 64 | 2 | 2.5% | 2.2% | 1 : 3.6 |
| uniform 64 | 8 | 31% | 2.2% | 1 : 3.6 |
| band $h/9$ at $x = 0$ | 0.5 / 2 | 0.6% / 9.8% | 18% / 6.5% | 1 : 5.6 |

Three facts follow.

1. The implicit scheme is stable at any cell Courant number, and cells the scalar
   does not need are free: the band refined to $h/9$ sits at local Courant 13 and
   changes the error in the third digit only.
2. Its accuracy is set by $\mathbf{u}\Delta t$ against the feature width. The error
   scales as $\Delta t^2$ for Crank-Nicolson, which is A-stable but not L-stable
   and rings once the feature is under-resolved in time.
3. SLCN's error is flat in $\Delta t$ but accumulates at small Courant (one
   interpolation per step), so it is the worse scheme exactly where it is not meant
   to run; its limit is the arc a characteristic turns per step, about 10 degrees
   for the RK2 trace-back, a property of the flow rather than the mesh.

The new class reproduces the prototype's Crank-Nicolson numbers to four digits
(0.5993% and 9.777% at Courant 0.5 and 2 on the uniform mesh).

### BDF against Adams-Moulton

`time_integrator_study.py`: the same rotating Gaussian, res 32, every scheme
the class offers, at Courant 0.25 to 8; relative $L_2$ error after one
revolution, "X" where the run blew up (with the step). Pure advection first,
then $\kappa = 10^{-3}$ (cell Peclet about 40).

| scheme | C 0.25 | 0.5 | 1 | 2 | 4 | 8 |
|---|---|---|---|---|---|---|
| BDF1 = backward Euler | 19% | 30% | 44% | 57% | 68% | 77% |
| BDF2 | 0.6% | 2.4% | 9.3% | 28% | 53% | 73% |
| BDF3 | 0.32% | 0.28% | 2.7% | 18% | X | X |
| Crank-Nicolson (`am`, 1, theta 0.5) | 0.27% | 0.6% | 2.5% | 9.8% | 31% | 66% |
| Adams-Moulton 2 (third order) | 0.28% | 0.24% | 0.24% | X@68 | X@41 | X@32 |
| Adams-Moulton 3 (fourth order) | 0.28% | 0.25% | X@155 | X@32 | X@22 | X@19 |

| scheme, $\kappa = 10^{-3}$ | C 0.25 | 0.5 | 1 | 2 | 4 | 8 |
|---|---|---|---|---|---|---|
| BDF1 = backward Euler | 12% | 20% | 31% | 45% | 58% | 69% |
| BDF2 | 0.27% | 0.71% | 3.3% | 13% | 35% | 59% |
| BDF3 | 0.31% | 0.45% | 0.87% | 4.4% | 51% | X |
| Crank-Nicolson | 0.38% | 0.51% | 0.63% | 2.5% | 13% | 42% |
| Adams-Moulton 2 | 0.42% | 0.71% | 1.3% | X | X | X |
| Adams-Moulton 3 | 0.42% | 0.71% | X | X | X | X |

At res 64 (pure advection, Courant 1 to 8, 590 to 74 steps per revolution):

| scheme, res 64 | C 1 | 2 | 4 | 8 |
|---|---|---|---|---|
| BDF1 = backward Euler | 30% | 43% | 57% | 68% |
| BDF2 | 2.5% | 9.1% | 27% | 53% |
| BDF3 | 3100% (slow growth) | 1.9% | 17% | 130% |
| Crank-Nicolson | 0.62% | 2.5% | 9.5% | 31% |
| Adams-Moulton 2 | 310% (slow growth) | X@76 | X@49 | X@38 |
| Adams-Moulton 3 | X@116 | X@37 | X@25 | X@22 |

BDF2 and Crank-Nicolson track their res-32 values at the same $\mathbf{u}\Delta t$
(the error is set by the timestep, not the mesh). BDF3 is not safe for pure
advection at any Courant number: its stability region misses the imaginary axis
near the origin, so the low-frequency modes a finer mesh carries grow slowly (31
times the exact field after 590 steps at Courant 1, where the coarser mesh with
half the steps still looked fine); with $\kappa = 10^{-3}$ it behaved. Use it
only with diffusion and below Courant 2.

Cost per step is the same for every scheme (0.058 to 0.068 s at res 32, 0.32 to
0.36 s at res 64): the
history terms are extra kernel inputs, not extra solves. BDF1 and backward Euler
agree to every digit, which checks that the two families are assembled
consistently.

What the table says:

- **Adams-Moulton above order 1 is unusable for advection.** Its stability region
  is bounded and covers only a short segment of the imaginary axis, so on a pure
  advection operator it blows up once the Courant number reaches about 1, and
  diffusion at this Peclet number does not rescue it. The assembly code handles
  it, but no public argument reaches it.
- **BDF3 is the most accurate scheme below Courant 1 with diffusion present**
  (0.3%, on the spatial floor) but it is not A-stable, fails from Courant 4, and
  on pure advection grows slowly at any Courant number (the res-64 rows).
- **Crank-Nicolson is three to four times more accurate than BDF2 at the same
  timestep** across the usable range, because it does not damp; the price is
  ringing once the feature is under-resolved in time (minimum $-0.35$ at Courant 8
  against $-0.20$ for BDF2), and no damping of stiff modes at all.
- **BDF2 is the robust choice**: stable at every Courant number, damped, second
  order, and the error is still set by $\mathbf{u}\Delta t$ against the feature
  width.

**Interface and default.** The class is a drop-in replacement for the
semi-Lagrangian solver: the same constructor, and `order` and `theta` with the same
meaning (`order=1, theta=0.5` is Crank-Nicolson and the default, as for SLCN;
`order=2, theta=1.0` is BDF2, the counterpart of SL-BDF2; `order=2, theta=0.5` is
refused for the reason the SLCN documentation gives). There is no `integrator`
argument: the family follows the order, and the only schemes that argument would
have added, Adams-Moulton at orders 2 and 3, are the ones the table rules out.
The study reached them by switching the family on the instance. The choice of
Crank-Nicolson as the default follows the drop-in contract and the table: it is
the more accurate scheme wherever the answer is good, and where it rings the
answer is already wrong for every scheme. A user who wants damping asks for
`order=2`; below Courant 1 with diffusion, `order=3`. Backward Euler is not a
sensible choice for transport.

### Temporal convergence (tests/test_1100)

Quarter-turn error on the uniform res-32 mesh with the exact history planted:
BDF1 slopes 0.80 and 0.88 between $\Delta t$ = 0.02, 0.01, 0.005; BDF2 slopes above
1.65 between 0.04, 0.02, 0.01.

### Preconditioner

Level-set advection step (`uw.systems.level_set`, a two-cell band, P2,
Crank-Nicolson) on a structured quad box built with a refinement hierarchy, so
every solver sees the same finest operator; the vortex velocity field of the
level-set study. Wall time per step over ten steps after a warm-up step, on a
sixteen-core workstation. Script and logs:
`~/+Simulations/supg_vs_slcn_657/parallel/fmg_timing.py`, `fmg.log`.

**Schwarz against geometric multigrid at matched tolerances** (Krylov 1e-9,
SNES 1e-8; one Newton iteration per step for both), 256², three levels:

| Courant | GMRES + ASM-ILU, its (np 1 / 8) | s/step (np 1 / 8) | fgmres + FMG, cycles (np 1 / 8) | s/step (np 1 / 8) |
|---|---|---|---|---|
| 1/2 | 5 / 5 | 0.913 / 0.121 | 1 / 1 | 0.943 / 0.145 |
| 2 | 8.9 / 8.6 | 0.925 / 0.141 | 3.4 / 3.6 | 1.079 / 0.178 |
| 8 | 16.6 / 16.5 | 0.971 / 0.146 | 12.8 / 12.8 | 1.657 / 0.299 |
| 32 | 37 / 37.8 | 1.128 / 0.172 | 23.8 / 24.1 | 2.347 / 0.457 |

The multigrid smoother is the managed bundle's gmres/4 + SOR with Galerkin coarse
operators, which inherit the fine-grid $\tau$; four levels instead of three
changes nothing at Courant 1/2 (one cycle, 0.935 s either way), so the coarse
operators are not under-stabilised there. Above Courant 8 the scheme rings (the
range of $\phi$ reaches $-0.29$ to $1.29$ at Courant 8), so the rows where
multigrid's cycle count is closest to the Schwarz count are rows nobody runs.

**Where the step goes** (`-log_view`, np 1, Courant 1/2, eleven solves): residual
evaluation 4.0 s, Jacobian evaluation 4.4 s, `KSPSolve` 0.36 s under Schwarz and
0.95 s under multigrid. With the Krylov tolerance left at its default of 1e-5 the
Schwarz solver stopped at three iterations, the SNES took a second Newton step
(22 Jacobian assemblies over eleven solves), and the step cost 1.54 s; one
multigrid cycle happens to reduce the residual below the SNES tolerance, so it
took one. That looked like a 1.65x win for multigrid and was a Jacobian
assembly.

**Controls** (Krylov tolerance at its default, 256², np 1 / 8): algebraic
multigrid (the managed GAMG bundle) 5 iterations, 2.07 / 0.245 s; the "fast"
smoother (richardson/3 + SOR) 0.933 s, the same as gmres/4; gmres/2 needs two
cycles and costs 1.61 s; an ILU smoother 1.62 s. At 512² with four levels the
unmatched rows read 6.12 / 0.84 s (Schwarz, two Newton steps) against 3.71 /
0.58 s (multigrid); matched, with the shipped defaults, 3.51 / 0.48 s (Schwarz,
5 iterations) against 3.62 / 0.54 s (multigrid, one cycle).

## Navier-Stokes with SUPG momentum transport

`uw.systems.NavierStokes` (`systems/navier_stokes_eulerian.py`) is the vector
form of the scalar solver on the Stokes saddle-point class: the momentum advection
is assembled implicitly and the streamline term stabilises it. The residual is

$$
\mathbf{f}_0 = \rho\,\big(\dot{\mathbf{u}} + \textstyle\sum_k w_k (\mathbf{a}_k\cdot\nabla)\mathbf{u}^{(k)}\big) - \mathbf{f},
\qquad
\mathbf{F}_1 = \textstyle\sum_k w_k\,\boldsymbol{\tau}(\mathbf{u}^{(k)}) - p_\mathrm{mech}\mathbf{I} + \tau_s\,\mathbf{R}\otimes\mathbf{a},
$$

with $\mathbf{R} = \mathbf{f}_0 + \nabla p$ the strong residual the SUPG term sees,
$\mathbf{a}$ the advecting velocity at the new level and $\mathbf{a}_k = \mathbf{u}^{(k)}$
at the stored ones, $w_k$ the weights of the spatial operator (Adams-Moulton at
order 1, all on n+1 for BDF2), and $\tau_s$ the scalar formula with $\nu = \eta/\rho$.
The pressure equation is the Stokes constraint; Taylor-Hood needs no pressure
stabilisation. Since 2026-09-06 the term carries the cell-Péclet weight
$Pe^2/(Pe^2 + Pe_c^2)$ with $Pe_c = 4$ by default (Louis: "the code is still 100% local,
so we should probably just switch to this strategy right away"), measured in "The weight
by cell Péclet number" below; every table before that subsection was made with the
uniform weight (`peclet_weight=0`), and the Pe_c = 4 column there gives the change. The
scalar transport solver carries the same weight (its convection rows are in that
subsection). Decisions, and what they rest on:

- **No stress history.** The semi-Lagrangian solver carries a stress history
  because its Crank-Nicolson viscous term needs the old flux at the departure
  points. On the grid the old flux is needed where it was formed, so
  $\boldsymbol{\tau}(\mathbf{u}^n) = 2\eta\,\dot\varepsilon(\mathbf{u}^n)$ is rebuilt
  from the stored velocity level with the current effective viscosity (exact for
  a constant viscosity; use BDF2 with a strain-rate dependent one). Pressure has
  no history. A history-dependent stress is the constitutive model's business.
- **The advecting velocity is pluggable.** `advection="extrapolated"` (default),
  $\mathbf{a} = 2\mathbf{u}^n - \mathbf{u}^{n-1}$, makes each step one linear Oseen
  solve through the Stokes fieldsplit, with a second-order lag and no explicit
  stability limit; `picard_iterations=n` re-solves with the latest iterate for the
  fully implicit fixed point without a tangent; `advection="implicit"` puts
  $\mathbf{u}^{n+1}$ in the term and the SNES takes Newton steps with the symbolic
  tangent. At a steady state all three coincide, which the Kovasznay rows below
  confirm to every digit; the cylinder wake is where they differ.
- **The pressure gradient belongs in the SUPG residual.** Without it $\mathbf{R}$ is
  O(1) at the exact solution and the stabilisation injects an O($\tau$) error:
  Kovasznay at 1/16 read 1.9e-3 against 6.6e-4 with it (three times). The viscous
  term needs second derivatives the kernels do not see and is the remaining
  inconsistency, O($h^2$) for P2 velocity in diffusion-limited cells; a recovered
  Laplacian would close it and is deferred.
- **`mesh.cell_size()` was partition-dependent** (#687): the kd-tree radius picked
  the nearest centroid among the rank's own cells, so $\tau$ differed across a
  partition seam (two-rank Kovasznay error 5e-4 off serial, 1e-15 with a
  constant $h$), and after a deform it read stale vertex coordinates against fresh
  centroids. The field now reports each cell's own radius from the DM's
  coordinates; the kd-tree radii still feed `get_min_radius`.
- **`solve()` builds through `_build`**, one Newton iteration per step at matched
  tolerances, as for the scalar solver.

### Kovasznay flow (Re 40)

Exact steady Navier-Stokes on $[-0.5, 1] \times [-0.5, 0.5]$, Dirichlet velocity
from the exact solution, P2-P1, 40 steps at Courant 1 from the exact solution
(or 80 from rest); relative $L_2$ velocity error at the end
(`~/+Simulations/navier_stokes_supg/kovasznay/`).

| h | SUPG, CN | Galerkin, CN | SUPG, BDF2 | SLCN | s/step SUPG / SLCN |
|---|---|---|---|---|---|
| 1/16 | 6.6e-4 | 1.1e-4 | 6.3e-4 | 5.8e-3 | 0.27 / 1.9 |
| 1/32 | 2.6e-4 | 1.6e-5 | 2.5e-4 | 2.9e-3 | 1.5 / 3.7 |
| 1/64 | 6.7e-5 | | | | 6.8 / |

Galerkin converges at third order here (the interpolation error), SUPG at 1.4
rising to 2.0 (the missing viscous term), SLCN at first order. At Re 40 the
element Reynolds number is below three on every mesh and the stabilisation is not
needed; it costs a factor of six to sixteen against Galerkin and is still nine
times more accurate than the semi-Lagrangian scheme at seven times less cost per
step. Newton (`advection="implicit"`), two Picard passes, Courant 4, and the
from-rest starts all reach the same steady state (6.60e-4 at 1/16); BDF2 sits at
an exact fixed point (step change 0) where Crank-Nicolson keeps a 5e-5 flicker.
Two ranks reproduce the serial error to 1e-7 (test_1078).

### Lid-driven cavity

Unit square, no-slip walls, unit lid (singular at the corners), P2-P1 on an
unstructured mesh, marched from rest; centreline extrema (u on x = 0.5, v on
y = 0.5) against Ghia, Ghia and Shin (1982). `~/+Simulations/navier_stokes_supg/cavity/`.

| Re | h | scheme | Courant | Picard | u_min | v_max | v_min | steps | s/step |
|---|---|---|---|---|---|---|---|---|---|
| 100 | Ghia | | | | -0.2109 | 0.1753 | -0.2453 | | |
| 100 | 1/32 | SUPG | 1 | 0 | -0.2025 | 0.1710 | -0.2437 | 543 (fixed point) | 0.73 |
| 100 | 1/32 | SLCN | 1 | | -0.1977 | 0.1695 | -0.2365 | 1000 (still moving) | 2.9 |
| 400 | Ghia | | | | -0.3273 | 0.3020 | -0.4499 | | |
| 400 | 1/48 | SUPG | 2 | 0 | -0.3076 | 0.2832 | -0.4288 | 1500 | 2.3 |
| 400 | 1/48 | SUPG | 2 | 1 | -0.3076 | 0.2831 | -0.4288 | 976 (fixed point) | 2.8 |
| 400 | 1/48 | SUPG | 1 | 0 | -0.3075 | 0.2828 | -0.4284 | 1500 (change 3e-6) | 2.1 |
| 1000 | Ghia | | | | -0.3829 | 0.3709 | -0.5155 | | |
| 1000 | 1/64, 3-level FMG | SUPG | 1 | 0 | -0.3413 | 0.0613 | -0.4687 | 1200 (t = 19, still moving) | 3.3 (np 4) |
| 1000 | 1/64, 3-level FMG | Galerkin | 1 | 0 | -0.1437 | 0.0695 | -0.2031 | 300 (t = 4.7) | 3.7 (np 4) |
| 1000 | 1/64, 3-level FMG | SUPG | 2 | 1 | -0.3620 | 0.3491 | -0.4940 | 2281 (t = 71, steady) | 3.3 (np 4) |

At Re 100 SUPG is within 4% of Ghia on every extremum on a 1/32 mesh and
reaches an exact fixed point; SLCN on the same mesh sits a little further out and
has not settled after 1000 steps at four times the cost. At Re 400 on 1/48 both
runs give the same extrema, 5 to 6% below Ghia (the mesh, not the scheme: the
extrema are steady to four digits), but the extrapolated step alone never
becomes stationary: the max-norm change per step grows to 0.1 and saturates, an
alternating mode of the lagged coefficient fed by the lid singularity while the
interior sits still. One Picard pass removes it (step change exactly zero) for
20% more per step, and so does Courant 1 without any pass. That is the regime the
Picard option was built for: Courant 2 with an element Reynolds number near eight.

At Re 1000 (element Reynolds number 16) on a 1/64 mesh built with a two-level
refinement so the velocity block runs geometric multigrid, the extrapolated step
takes one Newton and one Krylov iteration per step at 3.3 s on four ranks, and
the Galerkin form runs just as stably for its 300 steps: neither oscillates on
this mesh. The Courant-2, one-Picard run reaches the steady tolerance at step 2281
(t = 71) with the three extrema at 94 to 96% of Ghia and their positions within 0.01
(u_min at y 0.175, v_max at x 0.163, v_min at x 0.907), the same shortfall as Re 400
on 1/48, and the flow (primary vortex, both bottom-corner eddies) as the reference
shows it. The v_max of 0.05 to 0.07 the earlier rows print is the driver reading rank
0's own `evaluate` on four ranks (the left-wall upflow sits in another partition); the
driver now reduces the extrema across ranks, and the row above is a serial
re-evaluation of the final checkpoint. Two earlier four-rank attempts stalled at
their first logged step, which was the driver calling the collective centreline
evaluation on rank 0 only, and a third was killed by the hang watchdog on a rank
that never prints; none of those said anything about the solver.

### Cylinder wake (DFG 2D-2, Re 100)

Channel 2.2 by 0.41, cylinder of radius 0.05 at (0.2, 0.2), parabolic inflow with
mean velocity 1, $\nu = 10^{-3}$; mesh 1/20 in the channel and 1/80 on the
cylinder, P2-P1, Courant 1 on the cylinder cells (dt 0.0083), twelve time units
from the parabolic profile; drag and lift from the traction integral on the
cylinder, the Strouhal number from the lift zero crossings over the last three
units. Reference (Schaefer and Turek 1996): $C_D$ max 3.22 to 3.24, $C_L$ max
0.99 to 1.01, St 0.295 to 0.305, $\Delta p$ 2.46 to 2.50.
`~/+Simulations/navier_stokes_supg/cylinder/`.

Every drag value in the first table below is the PRESSURE drag only: the boundary
integral of the traction dropped the viscous part, because the viscosity is a runtime
expression and the integral kernels read every expression as zero (#695, found through
this benchmark and fixed on this branch). The lift and the Strouhal number were never
affected (the lift is pressure-dominated and the frequency does not go through an integral).

| scheme | advecting velocity | St | $C_L$ max | $C_D$ max (pressure part only, #695) | $\Delta p$ | s/step |
|---|---|---|---|---|---|---|
| SUPG | extrapolated | 0.298 | 0.82 | 2.33 | 2.41 | 0.73 |
| SUPG | extrapolated + 1 Picard pass | 0.296 | 0.75 | 2.30 | 2.39 | 1.02 |
| SUPG | implicit (Newton) | 0.295 | 0.76 | 2.30 | 2.39 | 0.99 |
| SLCN | (trace-back) | 0.259 | 0.68 | 2.72 | 2.30 | 2.36 |
| SUPG, mesh 1/40 and 1/160, np 4 | extrapolated | 0.304 | 0.89 | 2.48 | | 1.0 (np 4) |

The shedding frequency and the pressure difference are on the reference at both meshes.
The two fully implicit forms agree with each other to three digits, and the extrapolated
step differs from them by 1% in frequency and 8% on the lift peak: at Courant 1 the lag is
visible on a time-dependent wake but small, and a single Picard pass, or Newton, removes
it at 40% more per step. The semi-Lagrangian solver on the same mesh and step has the
shedding 13% too slow (St 0.259) at three times the cost; the frequency is the quantity
the time integration owns, and there the Eulerian scheme is the accurate one.

**The drag deficit was a measurement, not the scheme.** The drag read 28% low on the
1/20 mesh and 23% on 1/40, and did not move with the SUPG weights: with the velocity
block solved by LU (the GAMG fallback on this gmsh mesh spins at weak stabilisation, so
the "Galerkin cannot run" of the first attempt was the preconditioner, not the
discretisation), the SUPG weight from 1 to 0 and the tau weights over a factor of four
moved the peak by 2.6%, with the reaction-form drag (the momentum residual integrated
against a hat function on the cylinder nodes) 6% above the traction integral throughout.
The log then showed the total traction drag equal to its pressure part to four digits.
With the integrals fixed, the channel mesh held at 1/20 and only the cylinder cells
refined through gmsh (Louis's prescription: refine the cylinder, keep the step), all at
dt 0.0083 (Courant 1 on the 1/80 cylinder cells, 8 on the 1/640 ones), velocity block LU,
serial; the last row is the whole mesh at 1/40 on four ranks at its own Courant-1 step:

| cylinder cells | Courant at the cylinder | $C_D$ max traction / reaction | $C_L$ max | $\Delta p$ | St | steps | s/step |
|---|---|---|---|---|---|---|---|
| 1/80 (SUPG) | 1 | 3.046 / 3.115 | 0.897 | 2.41 | 0.298 | 1440 | 0.38 |
| 1/80 (Galerkin) | 1 | 3.098 / 3.168 | 0.909 | 2.41 | 0.295 | 1440 | 0.38 |
| 1/160 | 2 | 3.134 / 3.155 | 0.866 | 2.43 | 0.300 | 1440 | 0.64 |
| 1/160, dt 0.0042 | 1 | 3.131 / 3.153 | 0.841 | 2.43 | 0.298 | 2880 | 0.47 |
| 1/320 | 4 | 3.198 / 3.204 | 0.969 | 2.48 | 0.300 | 1440 | 1.0 |
| 1/640 | 8 | 3.237 / 3.252 | 1.067 | 2.49 | 0.299 | 1440 | 1.6 |
| 1/640, one Picard pass | 8 | 3.218 / 3.220 | 1.018 | 2.48 | 0.296 | 1440 | 1.6 |
| whole mesh 1/40 (cylinder 1/160), np 4 | 1 | 3.182 / 3.204 | 0.979 | | 0.304 | 2880 | 1.5 (np 4) |
| FMG: base 1/10 refined once (cylinder 1/80) | 1 | 3.108 / 3.156 | 0.976 | 2.49 | 0.303 | 1440 | 1.6 |
| FMG: base 1/10 refined once (cylinder 1/160) | 2 | 3.181 / 3.202 | 1.006 | 2.49 | 0.302 | 1440 | 2.4 |
| reference | | 3.22 to 3.24 | 0.99 to 1.01 | 2.46 to 2.50 | 0.295 to 0.305 | | |

The drag, the pressure difference and the frequency converge onto the reference bands as
the cylinder cells alone are refined, the two force measurements close on each other (6%
apart with the wall shear in one cell, 0.2% at 1/320), and the time step does not enter:
the 1/160 rows at Courant 1 and 2 give the same drag to three digits. Refining the whole
mesh to 1/40 (four ranks, twice the steps) buys less than the 1/320 cylinder cells do on
one core with the 1/20 channel. The lift peak converges from below and overshoots the band
by 6% at 1/640 (Courant 8 on those cells); one Picard pass on the same mesh brings it to
1.018 with the drag at 3.218 and the two force measurements 0.1% apart, so the overshoot is
the extrapolated advecting velocity's lag at that local Courant number, not the mesh. That
is the regime the Picard option exists for, and at Courant 8 on the cells that set the
forces it is worth its 40% per step. Earlier reads of this benchmark (drag "23 to 28% low, not
closing with the mesh, not moving with tau") were the missing viscous traction (#695): the
SUPG weight from 1 to 0 and the tau weights over a factor of four move the drag peak by
2.6%, and the Galerkin form that "could not run" was the GAMG fallback spinning inside the
Schur complement at weak stabilisation (native stack), not the discretisation.

FMG on this gmsh mesh: building the base mesh at 1/10 and refining once through the circle
callback (`-uw_refinement 1`, the callback snaps the new vertices to the circle) gives the
velocity block its geometric hierarchy, one Krylov iteration per Newton step and no
fallback (the two FMG rows). The refined mesh also gives a better lift and pressure
difference than the directly meshed 1/20 channel with the same cylinder cell: at 1/160 on
the cylinder every quantity but the drag (1% low) is inside the reference band. The
per-step times were taken with twelve cores busy and are not a like-for-like comparison
with LU. LU on the velocity block is serial-only: on more than one rank PETSc's native
factorisation has no parallel path and the run dies in the first solve, so the multigrid
hierarchy is the parallel route on this mesh.

The same cylinder-only refinement through FMG (base 1/10 refined once, channel 1/20,
fixed dt 0.0083, no LU), the route that scales:

| cylinder cells | Courant there | advecting velocity | $C_D$ max traction / reaction | $C_L$ max | $\Delta p$ | St | s/step |
|---|---|---|---|---|---|---|---|
| 1/320 | 4 | extrapolated | 3.208 / 3.215 | 1.004 | 2.48 | 0.300 | 4.9 |
| 1/320 | 4 | one Picard pass | 3.187 / 3.194 | 0.954 | 2.47 | 0.297 | 6.1 |
| 1/640 | 8 | one Picard pass | 3.204 / 3.206 | 0.969 | 2.47 | 0.297 | 10.9 |
| 1/640, np 4 | 8 | one Picard pass | 3.205 / 3.206 | 0.969 | | 0.297 | 6.2 (np 4) |
| 1/640 | 8 | Newton | 3.204 / 3.205 | 0.969 | 2.47 | 0.296 | 7.4 |
| 1/640 | 8 | one Picard pass, BDF2 | 3.196 / 3.198 | 0.939 | 2.47 | 0.295 | 8.6 |
| reference | | | 3.22 to 3.24 | 0.99 to 1.01 | 2.46 to 2.50 | 0.295 to 0.305 | |

(Times with the machine shared by five runs.) The drag and the pressure difference sit
within 1% of the bands with the two force measurements 0.05% apart at 1/640; the
frequency is in band throughout. The lift is the sensitive quantity: the extrapolated step
reads 1.004 at Courant 4 on the cylinder cells and 1.067 at Courant 8 on the unrefined
mesh, the implicit forms 0.954 to 0.969, and BDF2 0.939, so at these local Courant numbers
the extrapolation's lag and BDF2's damping each move the lift peak by 3 to 5% and the
Crank-Nicolson implicit forms are the ones to compare with the reference. One Picard
pass and Newton agree to three digits at 1/640 and Newton is the cheaper of the two.
Serial and four ranks agree to four digits (3.204 / 3.205, 0.9692 / 0.9689), the
partition independence the assembled operator should give, at 1.8x on four ranks with
the machine loaded. `figures/13_cylinder_Re100_supg_c320_picard_tracers.mp4` is the wake
at the 1/320-cell, one-Picard setup with tracers released in the central band.

Parallel tracers (#693) work with the empty-rank guard from #680: the two further
failures reported there were the driver's (an advection before the first release, on a
swarm that had never been populated and so carries the DMSwarm local size of −1, which
fails in serial in the same way; and a timing variable shadowed by a rank-local array).

### Vortex decay (Taylor-Green)

The exact unsteady solution on $[0,\pi]^2$, $\mathbf{u} = (-\sin x\cos y,\ \cos x\sin y)\,e^{-2\nu t}$,
$p = \tfrac14(\cos 2x + \cos 2y)\,e^{-4\nu t}$, has no normal flow and no tangential
stress on the walls, so free-slip walls (the normal component fixed) are exact and carry
no time dependence. Relative $L_2$ velocity error at $t = 1$ against the exact solution,
$\nu = 0.01$, P2-P1 on a regular simplex mesh, velocity block by LU, from the exact
initial state (`~/+Simulations/navier_stokes_supg/vortex_decay/`, `scripts/taylor_green.py`).
The interpolation error of the exact field is 1.7e-5 on the 1/32 mesh and 2.2e-6 on 1/64.

| dt (mesh 1/32) | SUPG, CN | Galerkin, CN | SUPG, BDF2 | Galerkin, BDF2 |
|---|---|---|---|---|
| 0.2 | 4.3e-4 | 6.8e-5 | 2.5e-4 | 5.3e-5 |
| 0.1 | 2.1e-4 | 4.8e-5 | 2.0e-4 | 4.9e-5 |
| 0.05 | 1.7e-4 | 4.9e-5 | 1.4e-4 | 4.9e-5 |
| 0.025 | 1.2e-4 | 4.9e-5 | 9.2e-5 | 4.9e-5 |
| 0.0125 | 7.8e-5 | 4.9e-5 | 6.5e-5 | 4.9e-5 |

| h (dt 0.0125) | SUPG | Galerkin | interpolation |
|---|---|---|---|
| 1/8 | 9.0e-3 | 9.3e-3 | 1.1e-3 |
| 1/16 | 6.7e-4 | 6.6e-4 | 1.4e-4 |
| 1/32 | 7.8e-5 | 4.9e-5 | 1.7e-5 |
| 1/64 | 1.6e-5 | 4.0e-6 | 2.2e-6 |

The Galerkin form is spatially limited at every time step in the table: its error is the
same at dt 0.2 as at dt 0.0125 and converges at third order in $h$, three times the
interpolation error. The time integration is not what limits this problem, because the
pattern is steady and only the amplitude decays, and Crank-Nicolson integrates
$e^{-2\nu t}$ with $2\nu\,\Delta t \le 0.004$ almost exactly. What the SUPG column measures
is the stabilisation's consistency error, and it scales with $\tau_s$: halving the time
step raises the transient term in $\tau_s$ and lowers the error by 1.5 to 1.8 until the
advective term takes over, and on the 1/64 mesh the floor is four times the Galerkin
error. The advecting-velocity choices coincide to four digits (1.723e-4 at dt 0.05 for
extrapolated, three Picard passes and Newton). Across the viscosity range at dt 0.025 on
1/32 the SUPG error is 4.2e-4 at $\nu = 1$ (the energy has decayed to 1.8%), 1.9e-5 at
0.1, 1.2e-4 at 0.01 and 5.1e-4 at 0.001 (element Reynolds number 100). The kinetic energy
ratio follows $e^{-4\nu t}$ to 1e-6 with free-slip walls for both forms.

Imposing the exact velocity on the walls instead (`-uw_bc dirichlet`, the time a runtime
expression in the condition) gives 1.13e-4 at dt 0.025 on 1/32, the free-slip value. It
first gave 4.7e-3 on every mesh and at every time step, with the decay 10% too slow, and
freezing the time deliberately reproduced that number to four digits: the time expression
had been created at the value zero, and sympy's automatic evaluation, reading the
expression's `is_zero` assumption from its value, had evaluated $e^{-2\nu t}$ out of the
boundary formula before the JIT saw it (issue #696, since fixed: a `UWexpression` no longer
reports `is_zero`, `is_positive` or `is_negative` from its current value, so sympy cannot fold on
them; `tests/test_0503` carries the `exp(c)` control).

### The recovered viscous term: measured and withdrawn

The SUPG column above is the stabilisation's consistency error: the strong residual the
term weights lacks $\nabla\cdot\boldsymbol{\sigma}$ (second derivatives the kernels do
not see). The Péclet turn-down of $\tau_s$ does not remove it: at low Péclet number
$\tau_s \to h^2/(4\nu)$ while the missing term is $\nu\nabla^2\mathbf{u}$, and the product
is $O(h^2)$ with no $\nu$ in it. Three ways of supplying the term were built and measured
(velocity error at $t = 1$, dt 0.0125; Kovasznay at Re 40; the cylinder on the 1/20 mesh):

| case | SUPG | Galerkin | projected stress | balance form (Louis) | balance, smoothed L = 0.01 to 0.2 |
|---|---|---|---|---|---|
| vortex 1/32 | 7.8e-5 | 4.9e-5 | 7.8e-5 | 4.9e-5 | 7.8e-5 |
| vortex 1/64 | 1.6e-5 | 4.0e-6 | diverged | 4.1e-6 | |
| Kovasznay 1/16 | 6.6e-4 | 1.1e-4 | | 1.1e-4 | |
| Kovasznay 1/32 | 2.6e-4 | 1.6e-5 | diverged | 1.6e-5 | |
| cylinder $C_D$ / $C_L$ max | 3.046 / 0.897 | 3.098 / 0.909 | | diverged (step 25 to 50) | 3.04 / 0.85 to 0.87 |

- **Projected stress**: the deviatoric stress of the advecting velocity fitted to a
  continuous P2 tensor and differentiated. No change at 1/32, unstable finer: a
  differentiated fit to a discontinuous strain rate is not a Laplacian.
- **Balance form**: $\nabla\cdot\boldsymbol{\sigma}^n = \rho(D\mathbf{u}/Dt)^n + \nabla p^n
  - \mathbf{f}$ from the stored levels and a stored pressure, so the residual is the
  increment of the out-of-balance force between levels. It returns the Galerkin accuracy
  to two digits on every resolved case, and it does so because it is a tautology: for any
  slowly varying discrete solution the residual it builds is zero, so it does not recover
  the viscous term, it switches the stabilisation off. Where the stabilisation is needed it
  fails the same way, with the lagged residual fed back as a source (cylinder, drag 7% high
  at step 25, linear solve diverged before step 50, on a mesh where plain Galerkin runs).
- **Balance form projected with a smoothing length** (screened Poisson, 0.01 to 0.2 on the
  vortex, one to two cylinder cells on the cylinder): the projected term matches the exact
  $\nu\nabla^2\mathbf{u}$ to a few per cent and the SUPG error does not move at any length,
  while the cylinder stays stable and within 1% of plain SUPG on drag. A continuous
  recovery of the viscous term, however accurate, does not touch the error.

What the three say together: the consistency error on resolved P2 flow is not the smooth
part of the missing viscous term. It is the pointwise, element-wise residual of the
discrete solution (the piecewise-constant P1 pressure gradient and the second derivatives
of the P2 velocity, both O(h) pointwise) that $\tau_s\,\mathbf{a}\cdot\nabla\mathbf{w}$
integrates; only a term that cancels it pointwise removes it, and that term cancels the
stabilisation with it. The remedy the measurements support is not a recovered Laplacian
but the weight: where the cell Péclet number is small the term is not needed and costs a
fixed multiple of the Galerkin error, second order in $h$ (`supg_weight`, or the Galerkin
form; Kovasznay's recommendation stands). A Péclet-dependent weight is the design
question that remains. The options were removed from the solver after the measurement (a
knob that quietly disables the stabilisation should not ship); the drivers' `-uw_recovered`
switches went with them and the runs are in the study directory (`rec_*`, `bal_*`, `sm_*`).

### The shape of tau (`tau_shape`)

The inverse-sum $\tau_s$ is above the optimal 1-D curve at cell Péclet numbers of order
1 to 10 (Louis: the shape of the correction was always the debated trade-off between
accuracy and cost). Two further shapes are selectable, each combined with the same
transient cap $[(C_t c_0/\Delta t)^2 + \tau^{-2}]^{-1/2}$: Brooks-Hughes,
$\tau = (h/2|a|)(\coth Pe - 1/Pe)$, and the doubly asymptotic $(h/2|a|)\min(Pe/3, 1)$,
$Pe = |a|h/2\nu$. Same cases as above (cell Péclet number in brackets):

| case | inverse sum | Brooks-Hughes | doubly asymptotic | Galerkin |
|---|---|---|---|---|
| vortex 1/32, dt 0.0125 (Pe 5) | 7.8e-5 | 7.8e-5 | 7.8e-5 | 4.9e-5 |
| vortex 1/32, dt 0.1 | 2.1e-4 | 1.7e-4 | 1.8e-4 | 4.8e-5 |
| vortex 1/64, dt 0.0125 (Pe 2.5) | 1.6e-5 | 1.4e-5 | 1.4e-5 | 4.0e-6 |
| Kovasznay 1/16 (Pe 1 to 3) | 6.6e-4 | 4.1e-4 | 4.8e-4 | 1.1e-4 |
| Kovasznay 1/32 | 2.6e-4 | 1.2e-4 | 1.2e-4 | 1.6e-5 |
| cylinder $C_D$ / $C_L$ max (Pe 10 at the wall) | 3.046 / 0.897 | 3.057 / 0.903 | 3.046 / 0.896 | 3.098 / 0.909 |

The shape matters where the cell Péclet number is near one: on Kovasznay the optimal
form halves the error (1.6 to 2.3 times) and on the 1/64 vortex it takes 15% off; where
the transient term caps $\tau_s$ (the 1/32 vortex at dt 0.0125) or advection dominates
(the cylinder, where all three shapes are $h/2|a|$) nothing moves. What remains after the
optimal shape is still seven times the Galerkin error on Kovasznay at 1/32: the shape
reduces the excess of $\tau_s$ over the 1-D optimum, it cannot remove the $O(h^2)$ product
of $\tau_s$ and the missing viscous term. The two 1-D shapes are exposed as options; the
inverse sum stays the default (smooth, no per-cell Péclet evaluation), and the weight,
by cell Péclet number, remains the lever that reaches the Galerkin value.

### The weight by cell Péclet number (`peclet_weight`)

The other side of the same lever: leave $\tau_s$ alone and multiply the term by
$w = Pe^2/(Pe^2 + Pe_c^2)$, $Pe = |a|h/2\nu$, so it is off where the cell is
diffusion-dominated and full where advection dominates. Same cases, three thresholds:

| case (cell Péclet) | SUPG | $Pe_c = 2$ | $Pe_c = 4$ | $Pe_c = 8$ | Galerkin |
|---|---|---|---|---|---|
| vortex 1/32 (Pe 5) | 7.8e-5 | 6.3e-5 | 5.3e-5 | 4.9e-5 | 4.9e-5 |
| vortex 1/64 (Pe 2.5) | 1.6e-5 | 8.6e-6 | 5.0e-6 | 4.1e-6 | 4.0e-6 |
| Kovasznay 1/16 (Pe 1 to 3) | 6.6e-4 | 3.6e-4 | 1.6e-4 | 1.1e-4 | 1.1e-4 |
| Kovasznay 1/32 | 2.6e-4 | 5.2e-5 | 2.1e-5 | 1.6e-5 | 1.6e-5 |
| cylinder $C_D$ / $C_L$ max (Pe 10 wall, 37 channel) | 3.046 / 0.897 | 3.061 / 0.908 | 3.080 / 0.919 | 3.094 / 0.917 | 3.098 / 0.909 |
| cylinder St | 0.298 | 0.297 | 0.296 | 0.296 | 0.295 |

This is the measurement that closes the trade-off. At $Pe_c = 8$ every resolved case
sits on the Galerkin value and the cylinder is still stable and within 0.2% of Galerkin
on drag with the weight at 0.6 on the wall cells and 0.95 in the channel; at $Pe_c = 4$
the resolved cases are within 1.3 times Galerkin and the wall cells keep 86% of the
term. The weight does what neither the recovered viscous term nor the shape of $\tau_s$
could: it removes the cost of stabilisation where the 1-D analysis says none is needed
and leaves it where it is. The default stays at zero (uniform weight) so that the
recorded benchmarks and the test references do not move; $Pe_c = 4$ is the recommended
setting for resolved or mixed problems. Louis's ruling (2026-09-06, the code being
unreleased): $Pe_c = 4$ is the default of both solvers. The scalar solver on the convection
benchmarks with it (`~/+Simulations/supg_vs_slcn_657/convection_benchmarks/`, runs
`*_pew4`; Blankenbach 1a reference Nu 4.884, Vrms 42.865):

| case | uniform weight: Vrms / Nu cold / Nu mid | $Pe_c = 4$: Vrms / Nu cold / Nu mid | transport s/step |
|---|---|---|---|
| box, Ra 1e4, 1/32 | 42.790 / 4.913 / 4.872 | 42.868 / 4.920 / 4.884 | 0.066 / 0.070 |
| annulus, Ra 1e4, 0.03 | 38.39 / 2.514 / 2.500 | 38.61 / 2.525 / 2.514 | 0.179 / 0.178 |

The box lands on the reference to four digits in Vrms and in the mid-plane Nusselt
number (the cells there sit at a Péclet number near one, where the term was costing
accuracy); the annulus moves 0.6% in the same direction; the cost does not move. The
parallel test's serial reference for the Navier-Stokes solver (Kovasznay at 1/8) goes
from 3.83e-3 to 1.42e-3; the pure-advection references are unchanged (weight 1).

### A defect in the integrals (#695)

The first error metric of this benchmark, an integral of $|\mathbf{v} - \mathbf{u}(t)|^2$
with the time as a runtime expression, returned $1 - e^{-2\nu t}$ at every time step: the
exact field inside the integral never left $t = 0$. `uw.maths.Integral`, `BdIntegral` and
`CellWiseIntegral` compile through the same JIT as the solvers, which routes every
`uw.function.expression` to PETSc's constants array, but none of them set the constants on
the DS they integrate with, so the kernels read zeros: any expression in an integrand
integrated to nothing, and a fresh Integral returned the cached zero. The constitutive
viscosity is such an expression, which is where the cylinder drag went (next section). Fixed
on this branch (`petsc_maths.pyx`, the boundary integral sets them on its sandbox DS);
`tests/test_0503_integral_expression_constants.py`.

## The DDt as the transport plugin

Louis asked (2026-09-06) whether the history manager could be the object that decides
how transport is done, so that one solver takes SUPG where it is needed and a
semi-Lagrangian history where it is not. It can, and it now is. Every solver that owns an
unknown composes its residual from three contributions of its `DuDt`:

| contribution | `EulerianSUPG` | `SemiLagrangian`, `Eulerian`, `Lagrangian` |
|---|---|---|
| `time_derivative()` | $(\psi^{n+1}-\psi^n)/\Delta t$ (theta rule) or the BDF stencil over the history | the same, over its own history |
| `advection()` | $\sum_k w_k\,(\mathbf{a}_k\cdot\nabla)\psi^{(k)}$, entry by entry of the unknown | zero (the history carries it) |
| `stabilisation_flux(R)` | $\tau\,R\otimes\mathbf{a}$, one flux row per component of $R$ | zero |
| `states()`, `spatial_weights()` | the levels and the weights $w_k$ of the scheme, for the solver's own flux | the same |

The scalar solver assembles $f_0 = \dot\phi + \mathbf{u}\cdot\nabla\phi - f$ and
$\mathbf{f}_1 = \sum_k w_k\kappa\nabla\phi^{(k)} + \tau R\mathbf{u}$ from these; the
Navier-Stokes solver multiplies the first two by $\rho$, adds $\nabla p$ to the residual
the flux sees, and keeps the viscous flux of the scheme and the pressure as its own. The
manager owns what the transport needs: the advecting velocity as data (`V_fn`, and
`V_fn_history` for the stored levels, which is the stored velocity itself for momentum),
the timestep as a runtime constant (`delta_t`, written by every flavour's
`update_pre_solve`), the time scheme, the diffusivity that $\tau$ sees (set by the solver
from its constitutive model), and the stabilisation knobs. The nonlinearity of a
self-advected unknown lives in what `V_fn` is: the extrapolated field, the Picard iterate,
or the unknown's own symbol for Newton.

What this bought, measured: the refactor moved no physics. The Péclet-weight rows of
Kovasznay at 1/16 and 1/32 (1.596e-4, 2.082e-5), the vortex decay at 1/32 (5.272e-5,
energy ratio 0.960789) and the Blankenbach box (42.8675 / 4.9204 / 4.8840) reproduce to
every printed digit, and the two-rank tests keep their serial constants. The cylinder at
1/20 with LU keeps its mean drag, lift extrema, reaction drag and Strouhal number to the
printed digits (3.0532, 0.9193 / -0.9652, 3.1219, 0.2964) while the drag peak moves from
3.0797 to 3.0802 and the pressure difference at peak lift from 2.4134 to 2.4125: the
assembled expressions are the same terms in a different order, and a shedding wake
amplifies the last bits over 1400 steps where a steady state does not. A `SemiLagrangian` manager dropped into `AdvDiffusion` reproduces
`AdvDiffusionSLCN` to the solver tolerance on pure advection (test_1057): the solver's
equation with zero advection and zero stabilisation is the semi-Lagrangian one. A tensor
unknown, flattened to its independent components on a `MATRIX` variable, is transported
through the multi-component solver with a residual that is nothing but the manager's
terms (test_1057, uniform translation of a Gaussian stress to 5%). That is stress
transport without rotation; the rotation of a transported tensor is a constitutive
matter and stays out of the transport.

Two things the plain `Eulerian` manager keeps: with a velocity it still applies the
explicit splitting correction to the history (its `_advection_mode` is `"split"`), which
is what the Richards and Darcy solvers rely on; `EulerianSUPG` sets the mode to
`"assembled"` and the solver's residual carries the advection instead. And the
semi-Lagrangian Stokes stress history (`DFDt` on a viscoelastic Stokes solve) is
untouched: it advects a stress that is not an unknown of the solve, which is a different
job from the one the contract describes.

## What the timestep estimate means

The cell-crossing time is not a stability limit for either scheme and says
nothing about this one's accuracy, so the Eulerian solver's `estimate_dt` measures
the field instead:

$$
\Delta t = f\,\frac{\max\phi - \min\phi}{\max|\dot\phi|},
$$

with $\dot\phi$ the advective rate $|\mathbf{u}\cdot\nabla\phi|$ before the first
solve and the realised rate $|\phi^{n+1}-\phi^{n}|/\Delta t$ after it (diffusion
and sources included). On the rotating Gaussian the fraction at Courant 0.5 on
the res-32 mesh is about 0.03 (0.6% Crank-Nicolson error) and at Courant 1 about
0.07 (2.5%); the default $f = 0.02$ therefore sits at a few tenths of a per cent.
The estimate is mesh-independent by construction, which is the property the
transport note's section 1 asks for; `basis="resolution"` still returns the
semi-Lagrangian solver's cell-crossing time. For SLCN the honest limit is the
trace-back arc, $\Delta t \lesssim 0.25 / \max|\nabla\mathbf{u}|$, which is a separate
change to that solver.

## A defect found on the way

The API test was flaky only after a test that dropped mesh variables. The cause
is general and predates this work: `mesh.vars` holds variables weakly, a
garbage-collected variable leaves its PETSc field in the DM, and both
`Mesh.update_lvec` and the JIT's auxiliary-field offsets assumed the registry and
the DM fields line up by position. Every later variable was then packed into, and
read from, the wrong slots. Fixed in the same branch (pack by field name, offsets
from the DM's field list) with `tests/test_1058_dropped_meshvariable_aux_layout.py`.

## Predictor-corrector manager (#689)

`uw.systems.ddt.EulerianSUPGPC(mesh, T, V_fn, method="citcoms", ...)` is a
scalar continuous-P1 transport manager supplied as `DuDt=` to
`uw.systems.AdvDiffusion`. `method="pc_converged"` selects a separate
residual-converged reference. The existing `EulerianSUPG` manager retains
CN/BDF, its transient tau and field-change timestep policy; the measurements
above describe that implicit path, not the PC update.

The ownership follows the transport contract: the manager supplies the
rate time derivative, advection, stabilisation and current spatial state;
the solver assembles the constitutive diffusion flux, source and boundary
terms. A generic stepping hook lets the manager execute corrections using
the solver's assembled residual and boundary handling. PC-specific rate,
startup state, correction controls, geometry and reusable workspaces belong
to the manager, without solver-side method aliases or unused BDF history.

For rate $q$, predict $T^{(0)}=T^n+(1-\gamma)\Delta t q^n$, reset $q=0$,
then apply $\delta q=-D^{-1}F(T,q)$, $q\leftarrow q+\delta q$ and
$T\leftarrow T+\gamma\Delta t\delta q$, reinserting Dirichlet values.
Here $D$ is positive row-lumped mass but $F$ retains the consistent
Petrov-Galerkin time derivative. `citcoms` defaults to `adv_gamma=0.5` and
`corrector_steps=2`, with one lumped startup correction. `pc_converged`
converges the rate equation at fixed temperature at startup and the coupled
correction after prediction, using $D$ only as a preconditioner. It stops at
`max(corrector_atol, corrector_rtol*initial_residual)` (defaults `1e-12`,
`1e-10`) or raises `RuntimeError` after `max_corrector_steps` (default 100).

Both PC methods retain the steady directional simplex tau and the
`0.9*min(dt_adv, dt_diff)` timestep estimate. Automatic geometry is limited
to triangles/tetrahedra with non-empty volume partitions on every rank;
unsupported layouts must be rejected collectively. The missing strong
diffusion term is exact only for affine P1 with elementwise constant
diffusivity. Neither SUPG nor residual convergence guarantees a nodal
maximum principle or unrestricted diagonal-iteration timesteps.

Finite corrections are not a consistent-mass solve: for pure diffusion,
two corrections approach $(2I-D^{-1}M)D^{-1}K$, not generally $M^{-1}K$,
as $\Delta t\to0$. A lumped startup rate adds another discrepancy. The
[user guide](../../advanced/eulerian-advection-diffusion.md#finite-correction-accuracy)
records the mathematical regression results: first-order timestep
differences for fixed corrections, and order 2.00 for consistent CN and
`pc_converged` at gamma 0.5 on tiny triangles/tetrahedra in serial and on
eight ranks. The DDt-manager migration reproduced these isolated results
on 8 September 2026; they are not coupled production acceptance.

Restart registration must capture the manager's rate, startup flag and
correction controls as well as temperature. Derived PETSc workspaces can be
rebuilt; rate history cannot be replaced with zero on continuation. A
matching model/manager layout and MPI rank count are required for disk
replay. Migration checks should cover frozen numerical equivalence, exact
discrete time-order references, snapshot continuation in a fresh process,
and workspace reuse in serial and MPI, separately from coupled production
benchmarks.
