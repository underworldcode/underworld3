# Eulerian SUPG transport: design and measurements

**Status**: implemented on `feature/eulerian-supg-transport` (2026-09-02), static mesh.
Supersedes the Crank-Nicolson prototype of issue #657 as the implementation route
while keeping its weak-form idea.

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

| `integrator` | time derivative | spatial operator |
|---|---|---|
| `bdf`, order $N$ | $\frac{1}{\Delta t}\sum_{k=0}^{N} c_k\,\phi^{n+1-k}$ | at $n+1$ only |
| `am`, order $N$ | $\frac{\phi^{n+1}-\phi^{n}}{\Delta t}$ | $\sum_{k=0}^{N} a_k\,S(\phi^{n+1-k})$ |

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
- **Own preconditioner.** The operator is nonsymmetric, so GMRES with an
  additive-Schwarz ILU preconditioner replaces the managed GAMG block. The solver
  sets `_pc_option_prefix = None`, and the mesh-owned multigrid pickup on adapt
  children now respects that (it segfaulted otherwise).
- **Moving meshes, phase 1.** The unknown and its history stay on the default
  `REMAP` transfer policy with the material velocity. The remap re-interpolates old
  states onto the new nodes, so the Eulerian form is already correct to
  interpolation accuracy. The `CARRY` + $\mathbf{u} - \mathbf{u}_\text{mesh}$ form
  is phase 2 and must not be mixed with `REMAP`.
- **Not yet:** discontinuity capturing (the prototype's residual omitted the time
  derivative and added first-order diffusion everywhere; a correct lagged residual
  needs $\phi^{n-1}$), a streamline element length from a mesh-owned metric tensor,
  the ALE hook.

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
  diffusion at this Peclet number does not rescue it. It is kept in the class for
  the record and for diffusion-dominated use, with that warning in the docstring.
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
refused for the reason the SLCN documentation gives). `integrator` is inferred and
only needs setting to reach Adams-Moulton above order 1. The choice of
Crank-Nicolson as the default follows the drop-in contract and the table: it is
the more accurate scheme wherever the answer is good, and where it rings the
answer is already wrong for every scheme. A user who wants damping asks for
`order=2`; below Courant 1 with diffusion, `order=3`. Backward Euler is not a
sensible choice for transport.

### Temporal convergence (tests/test_1100)

Quarter-turn error on the uniform res-32 mesh with the exact history planted:
BDF1 slopes 0.80 and 0.88 between $\Delta t$ = 0.02, 0.01, 0.005; BDF2 slopes above
1.65 between 0.04, 0.02, 0.01.

## What the timestep estimate means

`estimate_dt` returns the cell-crossing time, the same resolution estimate the
semi-Lagrangian solver reports, because that is the only quantity the mesh knows.
It is not a stability limit for either scheme. Choose the Eulerian timestep from
the transported feature: $|\mathbf{u}|\Delta t$ a fraction of its width. For SLCN
the honest limit is the trace-back arc, $\Delta t \lesssim 0.25 / \max|\nabla\mathbf{u}|$,
which is a separate change to that solver.

## A defect found on the way

The API test was flaky only after a test that dropped mesh variables. The cause
is general and predates this work: `mesh.vars` holds variables weakly, a
garbage-collected variable leaves its PETSc field in the DM, and both
`Mesh.update_lvec` and the JIT's auxiliary-field offsets assumed the registry and
the DM fields line up by position. Every later variable was then packed into, and
read from, the wrong slots. Fixed in the same branch (pack by field name, offsets
from the DM's field list) with `tests/test_1058_dropped_meshvariable_aux_layout.py`.
