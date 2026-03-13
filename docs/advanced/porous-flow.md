---
title: "Porous Media Flow"
---

# Porous Media Flow

Underworld3 provides a hierarchy of solvers for groundwater and variably-saturated
porous media flow.  This guide explains when to use each solver, how to configure
retention curves, and practical tips for nonlinear convergence.

## Solver Hierarchy

Three solvers are available, each building on the previous one:

| Solver | Equation | Use case |
|--------|----------|----------|
| {class}`~underworld3.systems.solvers.SNES_Darcy` | $-\nabla\cdot[K\nabla h - \mathbf{s}] = f$ | Steady-state, fully saturated |
| {class}`~underworld3.systems.solvers.SNES_TransientDarcy` | $S_s\,\partial h/\partial t - \nabla\cdot[K\nabla h - \mathbf{s}] = f$ | Transient, constant storage |
| {class}`~underworld3.systems.solvers.SNES_Richards` | $\partial\theta/\partial t - \nabla\cdot[K(\psi)(\nabla\psi - \mathbf{s})] = f$ | Variably-saturated, nonlinear |

All three use the `DarcyFlowModel` constitutive model, which defines
permeability $K$ and a gravity-like source vector $\mathbf{s}$.

### Choosing a Solver

- **Steady-state, constant permeability** — use `SteadyStateDarcy` (alias for `SNES_Darcy`).
  No time stepping needed; just call `solve()`.
- **Transient, constant storage** — use `TransientDarcy`.
  Set `solver.storage` to the specific storage coefficient $S_s$ and advance with
  `solve(timestep=dt)`.
- **Variably-saturated** — use `Richards`.
  Permeability and storage depend nonlinearly on pressure head $\psi$.
  This solver handles the stiff nonlinearities arising from soil-water retention curves.

Python access:

```python
import underworld3 as uw

darcy     = uw.systems.SteadyStateDarcy(mesh, h_Field=h, v_Field=v)
transient = uw.systems.TransientDarcy(mesh, h_Field=h, v_Field=v, order=1)
richards  = uw.systems.Richards(mesh, psi_Field=psi, v_Field=v, order=1)
```

## Retention Curves

The Richards equation requires soil-water retention curves that describe how
moisture content $\theta$ and hydraulic conductivity $K$ vary with
pressure head $\psi$.

Underworld3 provides three models in
{mod}`underworld3.utilities.retention_curves`:

### Van Genuchten--Mualem

The most widely used model in hydrology.  Parameters: $\alpha$, $n$, $K_s$,
$\theta_r$, $\theta_s$.

```python
from underworld3.utilities.retention_curves import (
    van_genuchten_K,
    van_genuchten_theta,
)

psi_sym = psi.sym[0]

K_expr = van_genuchten_K(psi_sym, Ks=1e-4, alpha=3.35, n=2.0)
theta_expr = van_genuchten_theta(
    psi_sym, theta_r=0.045, theta_s=0.43, alpha=3.35, n=2.0
)
```

Typical parameter ranges (SI units, $\psi$ in metres):

| Soil type | $\alpha$ (1/m) | $n$ | $K_s$ (m/s) | $\theta_r$ | $\theta_s$ |
|-----------|----------------|-----|-------------|-------------|-------------|
| Sand      | 14.5           | 2.68 | $8.25 \times 10^{-5}$ | 0.045 | 0.43 |
| Loam      | 3.6            | 1.56 | $2.89 \times 10^{-6}$ | 0.078 | 0.43 |
| Clay      | 0.8            | 1.09 | $5.56 \times 10^{-7}$ | 0.068 | 0.38 |

### Gardner Exponential

A simpler model with an analytical steady-state solution — ideal for
verification:

$$K(\psi) = K_s \, e^{\alpha\psi}$$

```python
from underworld3.utilities.retention_curves import (
    gardner_K,
    gardner_theta,
    gardner_steady_state_psi,
)

K_expr = gardner_K(psi_sym, Ks=1.0, alpha=2.0)
theta_expr = gardner_theta(psi_sym, theta_r=0.05, theta_s=0.4, alpha=2.0)

# Exact steady-state profile for benchmarking
psi_exact = gardner_steady_state_psi(y_coords, psi_0=-3.0, psi_L=-0.5, L=1.0, alpha=2.0)
```

### Haverkamp

A rational-function model where retention ($\alpha$, $\beta$) and
conductivity ($A$, $B$) have **independent** parameters, giving extra
flexibility when fitting laboratory data.  Used in the
Vauclin (1979) water-table recharge benchmark.

$$\theta(\psi) = \theta_r + \frac{\alpha\,(\theta_s - \theta_r)}{\alpha + |\psi|^{\beta}},
\qquad
K(\psi) = K_s\,\frac{A}{A + |\psi|^B}$$

```python
from underworld3.utilities.retention_curves import (
    haverkamp_K,
    haverkamp_theta,
    haverkamp_C,
)

# Vauclin (1979) benchmark parameters (CGS, ψ in cm)
K_expr = haverkamp_K(psi_sym, Ks=9.44e-5, A=1.175e6, B=4.74)
theta_expr = haverkamp_theta(
    psi_sym, theta_r=0.075, theta_s=0.287, alpha=1.611e6, beta=3.96
)
```

### Choosing a Retention Model

| Model | Strengths | Typical use |
|-------|-----------|-------------|
| Van Genuchten--Mualem | Widely validated, coupled $K$--$\theta$ | General-purpose simulations |
| Gardner exponential | Admits analytical solutions | Verification benchmarks |
| Haverkamp | Independent $K$ and $\theta$ params | Lab data fitting, Vauclin benchmark |

## Setting Up a Richards Solver

### Mixed Form (Recommended)

The **mixed form** discretises the storage term as
$(\theta(\psi^{n+1}) - \theta(\psi^n))/\Delta t$, which is exactly
mass-conservative.  This is the preferred approach.

```python
richards = uw.systems.Richards(mesh, psi_Field=psi, v_Field=v, order=1, theta=0.5)

# Constitutive model (permeability + gravity)
richards.constitutive_model = uw.constitutive_models.DarcyFlowModel
richards.constitutive_model.Parameters.permeability = van_genuchten_K(
    psi.sym[0], Ks=1e-4, alpha=3.35, n=2.0
)
richards.constitutive_model.Parameters.s = sympy.Matrix([0, -1]).T

# Mixed form: provide θ(ψ) directly
richards.water_content = van_genuchten_theta(
    psi.sym[0], theta_r=0.045, theta_s=0.43, alpha=3.35, n=2.0
)

# Source term
richards.f = 0.0
```

When `water_content` is set, the solver computes the Jacobian
$\partial\theta/\partial\psi = C(\psi)$ automatically via PETSc's
finite-difference colouring.  You do **not** need to provide $C(\psi)$
separately.

### Head-Based Form (Backward Compatible)

The head-based form discretises the storage as
$C(\psi)(\psi^{n+1} - \psi^n)/\Delta t$.  This is simpler but not
mass-conservative when $C(\psi)$ varies sharply.

```python
from underworld3.utilities.retention_curves import van_genuchten_C

richards.capacity = van_genuchten_C(
    psi.sym[0], theta_r=0.045, theta_s=0.43, alpha=3.35, n=2.0
)
```

If both `water_content` and `capacity` are set, the mixed form takes precedence.

## Time Stepping

All transient porous flow solvers use BDF (Backward Differentiation Formula)
time integration with automatic order ramping:

```python
# First call: BDF-1 (backward Euler)
richards.solve(timestep=dt)

# Second call onwards: BDF-2 (if order=2 was requested)
richards.solve(timestep=dt)
```

The solver automatically:
- Initialises time-derivative history on the first solve call
- Ramps BDF order from 1 up to the requested `order`
- Tracks variable timesteps for correct BDF coefficients

### Timestep Estimation

`TransientDarcy` and `Richards` provide a diffusive CFL estimate:

```python
dt = richards.estimate_dt()
```

For the Richards equation with strongly nonlinear retention curves,
you may need to use a smaller timestep than this estimate, especially
near wetting fronts.

## Convergence Tips

The Richards equation with Van Genuchten curves is a **stiff nonlinear problem**.
Here are practical strategies for reliable convergence:

### 1. Use Backtracking Line Search

```python
richards.petsc_options["snes_linesearch_type"] = "bt"
```

The backtracking line search (default is `basic`) helps SNES find a
descent direction when the initial Newton step overshoots.

### 2. Increase SNES Iterations

```python
richards.petsc_options["snes_max_it"] = 50  # default is 20
```

Nonlinear problems near saturation may need more iterations.

### 3. Start From a Smooth Initial Condition

A linear profile between boundary values is a good starting guess:

```python
y = mesh.X[1]
psi_init = psi_bottom + (psi_top - psi_bottom) * y
psi.array = uw.function.evaluate(psi_init, psi.coords)
```

Abrupt initial conditions (e.g., step functions) cause convergence
difficulties.

### 4. Use Small Timesteps Initially

Start with small $\Delta t$ and increase gradually, especially when
wetting fronts are developing:

```python
dt = 0.001
for step in range(n_steps):
    richards.solve(timestep=dt)
    dt = min(dt * 1.2, dt_max)  # gradual increase
```

### 5. Monitor Convergence

```python
richards.petsc_options["snes_monitor"] = None
richards.petsc_options["snes_converged_reason"] = None
```

## Boundary Conditions

Boundary conditions follow the standard Underworld3 pattern:

```python
# Fixed head / pressure head (Dirichlet)
richards.add_dirichlet_bc([0.0], "Top")      # saturated surface
richards.add_dirichlet_bc([-5.0], "Bottom")  # deep water table

# Natural (Neumann) boundary conditions are set via richards.f
# and the constitutive model's flux term
```

For the Richards equation, typical boundary conditions are:
- **Saturated surface**: $\psi = 0$ (water table at the surface)
- **Deep dry condition**: $\psi = \psi_{\mathrm{init}}$ (initial pressure head)
- **No-flow boundaries**: Natural BC (the default on boundaries without Dirichlet conditions)

## Tutorials

For worked examples with complete code:

- [Tutorial 16 — Richards Equation: Groundwater](../beginner/tutorials/16-Richards-Equation-Groundwater.ipynb):
  Steady-state drainage with Gardner curves, comparison to analytical solution.
- [Tutorial 17 — Richards: Transient Wetting Front](../beginner/tutorials/17-Richards-Transient-Wetting-Front.ipynb):
  Transient infiltration with Van Genuchten curves, wetting front propagation.

## API Reference

- {class}`~underworld3.systems.solvers.SNES_Darcy` — Steady-state Darcy
- {class}`~underworld3.systems.solvers.SNES_TransientDarcy` — Transient Darcy
- {class}`~underworld3.systems.solvers.SNES_Richards` — Richards equation
- {mod}`underworld3.utilities.retention_curves` — Retention curve functions
