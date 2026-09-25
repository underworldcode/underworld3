# Solvers

## Stokes Flow

### SNES_Stokes

```{eval-rst}
.. autoclass:: underworld3.systems.solvers.SNES_Stokes
   :members:
   :show-inheritance:
```

### SNES_VE_Stokes

Viscoelastic extension of the Stokes solver.

```{eval-rst}
.. autoclass:: underworld3.systems.solvers.SNES_VE_Stokes
   :members:
   :show-inheritance:
```

## Scalar Equations

### Integrated Boundary Flux

After solving a continuous scalar problem, call collectively on all ranks:

```python
total_flux = poisson.boundary_flux_integral("Top")
```

This sums consistent nodal volume reactions on the requested essential
boundary. It has the same raw CBF sign as `boundary_flux()`, without mass
recovery, a temporary flux field, or boundary quadrature. Boundary membership
is propagated through the PETSc point SF so ranks sharing a boundary node
include their partial reactions even when they hold no labelled facet.

The return value is an integral: no mean removal, area division or Nusselt
normalization is applied. Divide by the boundary area and the appropriate
reference conductive flux when a normalized diagnostic is required. At
intersections of driven walls, a shared nodal reaction mixes contributions
from both walls; this method does not separate those by facet. Use
`boundary_flux()` or `boundary_flux_field()` for pointwise values instead.

### SNES_Poisson

```{eval-rst}
.. autoclass:: underworld3.systems.solvers.SNES_Poisson
   :members:
   :show-inheritance:
```

### SNES_Darcy

```{eval-rst}
.. autoclass:: underworld3.systems.solvers.SNES_Darcy
   :members:
   :show-inheritance:
```

### SNES_AdvectionDiffusion

```{eval-rst}
.. autoclass:: underworld3.systems.solvers.SNES_AdvectionDiffusion
   :members:
   :show-inheritance:
```

### SNES_AdvectionDiffusion_Composed (`uw.systems.AdvDiffusion`)

The scalar transport solver composed from a DDt transport manager; with the
default `EulerianSUPG` manager it is the implicit Eulerian SUPG scheme.

```{eval-rst}
.. autoclass:: underworld3.systems.advection_diffusion_eulerian.SNES_AdvectionDiffusion_Composed
   :members:
   :show-inheritance:
```

### EulerianSUPG (the transport manager)

```{eval-rst}
.. autoclass:: underworld3.systems.ddt.EulerianSUPG
   :members:
   :show-inheritance:
```

### SNES_Diffusion

```{eval-rst}
.. autoclass:: underworld3.systems.solvers.SNES_Diffusion
   :members:
   :show-inheritance:
```

## Projection Solvers

### SNES_Projection

```{eval-rst}
.. autoclass:: underworld3.systems.solvers.SNES_Projection
   :members:
   :show-inheritance:
```

### SNES_Vector_Projection

```{eval-rst}
.. autoclass:: underworld3.systems.solvers.SNES_Vector_Projection
   :members:
   :show-inheritance:
```

## Navier-Stokes

### SNES_NavierStokes

```{eval-rst}
.. autoclass:: underworld3.systems.solvers.SNES_NavierStokes
   :members:
   :show-inheritance:
```

### SNES_NavierStokes_Composed (`uw.systems.NavierStokes`)

Navier-Stokes composed from a DDt transport manager (Eulerian SUPG momentum
transport by default); the semi-Lagrangian class above is `uw.systems.NavierStokesSLCN`.

```{eval-rst}
.. autoclass:: underworld3.systems.navier_stokes_eulerian.SNES_NavierStokes_Composed
   :members:
   :show-inheritance:
```
