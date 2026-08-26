---
title: "SUPG Scalar Transport"
---

# SUPG scalar transport

`AdvDiffusionSUPG` solves

$$
\frac{\partial T}{\partial t} + \mathbf{u}\cdot\nabla T
- \nabla\cdot(\kappa\nabla T) = f
$$

on simplex volume meshes. It adds streamline-upwind Petrov-Galerkin (SUPG)
stabilization to the continuous finite-element residual. Advection remains a
local finite-element operation: the solver does not trace departure points or
interpolate a semi-Lagrangian history.

## Minimal example

This example transports and diffuses a continuous P1 scalar in a prescribed
velocity field. Automatic stabilization is the default.

```python
import numpy as np
import underworld3 as uw

mesh = uw.meshing.UnstructuredSimplexBox(
    minCoords=(0.0, 0.0),
    maxCoords=(1.0, 1.0),
    cellSize=0.1,
)

temperature = uw.discretisation.MeshVariable("T", mesh, 1, degree=1)
velocity = uw.discretisation.MeshVariable("U", mesh, mesh.dim, degree=1)

with mesh.access(temperature, velocity):
    x = temperature.coords[:, 0]
    y = temperature.coords[:, 1]
    temperature.data[:, 0] = np.sin(np.pi * x) * np.sin(np.pi * y)
    velocity.data[:, 0] = 1.0
    velocity.data[:, 1] = 0.0

thermal = uw.systems.AdvDiffusionSUPG(
    mesh,
    u_Field=temperature,
    V_fn=velocity.sym,
)
thermal.constitutive_model = uw.constitutive_models.DiffusionModel
thermal.constitutive_model.Parameters.diffusivity = 0.01

for boundary in ("Left", "Right", "Top", "Bottom"):
    thermal.add_dirichlet_bc(0.0, boundary)

for _ in range(10):
    thermal.solve(timestep=1.0e-3, zero_init_guess=False)
```

The default `time_integrator="bdf"` uses an implicit Eulerian BDF method and
the generic transient SUPG stabilization parameter. `order=1` and `order=2`
select BDF1 and BDF2 respectively.

## CitcomS-compatible predictor-corrector

For continuous P1 temperature, UW3 also provides the row-sum-mass
predictor-corrector used for the Zhong mantle-convection benchmark:

```python
temperature_rate = uw.discretisation.MeshVariable(
    "Tdot", mesh, 1, degree=1
)

thermal = uw.systems.AdvDiffusionSUPG(
    mesh,
    u_Field=temperature,
    V_fn=velocity.sym,
    time_integrator="citcoms",
    temperature_rate_field=temperature_rate,
)
thermal.constitutive_model = uw.constitutive_models.DiffusionModel
thermal.constitutive_model.Parameters.diffusivity = 0.01

dt = thermal.estimate_dt()
thermal.solve(timestep=dt)
```

This path uses `adv_gamma=0.5`, two residual-correction iterations, positive
row-sum mass, and the clipped CitcomS stabilization parameter by default. The
timestep is explicit and must satisfy the returned advection-diffusion limit.
Supplying a named `temperature_rate_field` makes the additional restart state
visible and straightforward to checkpoint. Exact restart requires `T`,
`Tdot`, and the solver snapshot metadata.

## Choosing a transport solver

| Method | Strength | Main cost or limitation | Restart state |
| --- | --- | --- | --- |
| `AdvDiffusionSUPG`, implicit BDF | Local assembly, no trace-back interpolation, automatic simplex stabilization | Timestep accuracy still requires convergence testing; automatic tau currently assumes scalar isotropic diffusivity on volume simplices | Temperature plus BDF history |
| `AdvDiffusionSUPG`, CitcomS predictor-corrector | Second-order explicit update, row-lumped P1 mass, close to CitcomS mantle-convection numerics | Continuous P1 only; explicit advection-diffusion timestep limit | `T`, `Tdot`, solver metadata |
| `AdvDiffusionSLCN` | Stable characteristic transport at large advective Courant number | Departure-point search/interpolation, flux history, and higher MPI memory/runtime | Temperature plus characteristic and flux histories |
| `AdvDiffusionSLCN` with SL-BDF2 | Second-order characteristic history without Crank-Nicolson flux ringing | Two departure points and greater history/interpolation cost | Two-level characteristic history plus flux history |
| `AdvDiffusionSLCN` with BDF1/Backward Euler | Robust, L-stable diffusion baseline | First-order time integration and trace-back interpolation | One characteristic history level |

Use the CitcomS predictor-corrector when reproducing a continuous-P1 CitcomS
benchmark. Use implicit SUPG when local streamline stabilization is desired
without the explicit predictor-corrector restriction. Use SLCN when large
advective timesteps are more important than trace-back cost. For every method,
verify timestep and mesh convergence using the physical diagnostics of the
problem; the solver name alone does not establish accuracy.

## Stabilization controls

- Omit `tau` for automatic stabilization.
- `tau_model="generic"` combines transient, advective, and diffusive scales.
- `tau_model="citcoms"` selects the clipped steady CitcomS relation on a
  simplex streamline length.
- Pass an explicit scalar `tau` for unsupported element or constitutive-model
  combinations. `tau=0` recovers the unstabilized Galerkin residual.
- Automatic tau supports two- and three-dimensional simplex volume meshes and
  scalar isotropic non-negative diffusivity.

## Related documentation

- [Semi-Lagrangian time integration](semi-lagrangian-time-integration.md)
- [State snapshots and restore](snapshot-restore.md)
- [Parallel computing](parallel-computing.md)

