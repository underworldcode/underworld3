# Navier-Stokes with Eulerian SUPG momentum transport

`uw.systems.NavierStokesSUPG` solves the incompressible Navier-Stokes equations on
the mesh, with the momentum advection assembled implicitly in the Stokes
saddle-point residual and stabilised by the streamline-upwind Petrov-Galerkin
term. It is the vector counterpart of {doc}`eulerian-advection-diffusion` and
takes the same constructor as `uw.systems.Stokes` plus the density and the time
scheme:

```python
ns = uw.systems.NavierStokesSUPG(mesh, v, p, rho=1.0, order=1)   # Crank-Nicolson
ns.constitutive_model = uw.constitutive_models.ViscousFlowModel
ns.constitutive_model.Parameters.shear_viscosity_0 = 1.0 / Re
ns.add_dirichlet_bc((0.0, 0.0), "Bottom")
...
for step in range(n):
    ns.solve(timestep=dt)
```

`order=1` is the theta rule (Crank-Nicolson at the default `theta=0.5`), `order=2`
is BDF2. The velocity history lives on the mesh; there is no stress history, the
viscous stress at an earlier level is rebuilt from the stored velocity. Pressure
has no history.

## The advecting velocity

The nonlinear term is $(\mathbf{a}\cdot\nabla)\mathbf{u}^{n+1}$ with $\mathbf{a}$
chosen by `advection=`:

- `"extrapolated"` (default): $\mathbf{a} = 2\mathbf{u}^n - \mathbf{u}^{n-1}$, a
  second-order lag. Each step is one linear solve through the Stokes fieldsplit.
- `picard_iterations=n` re-solves up to `n` more times with the latest iterate as
  $\mathbf{a}$, stopping when the velocity stops changing (`picard_tolerance`).
  The fixed point is the fully implicit scheme.
- `"implicit"`: $\mathbf{a} = \mathbf{u}^{n+1}$ and the SNES takes Newton steps on
  the quadratic term.

The stabilisation parameter is
$\tau = [(C_t/\Delta t)^2 + (C_u |\mathbf{a}|/h)^2 + (C_\nu \nu/h^2)^2]^{-1/2}$
with $h$ the local cell size and the three weights in `ns.tau_weights`;
`ns.supg_weight = 0` gives the plain Galerkin scheme. The strong residual the
term acts on carries the time derivative, the advection, the pressure gradient
and the body force, but not the viscous term (the kernels see first derivatives
only), so on a smooth, well-resolved flow the Galerkin form is the more accurate
one and the stabilisation earns its place where the element Reynolds number
$\rho|\mathbf{a}|h/\eta$ exceeds one.

## Timestep

`ns.estimate_dt()` returns the step at which the velocity changes by a fraction
(default 0.02) of its range, from the realised rate of the last step; before the
first solve, and with `basis="resolution"`, it returns the Stokes solver's
cell-crossing time.

## Further reading

- Design note and measurements: `docs/developer/design/eulerian-supg-transport.md`
- The semi-Lagrangian Navier-Stokes solver: `uw.systems.NavierStokes`
