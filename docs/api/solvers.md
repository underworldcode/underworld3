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
