# Per-iteration solver callbacks

Any UW3 solver can run a user callback **at the start of every nonlinear (Newton/SNES)
iteration**, via PETSc's `SNESSetUpdate`. This is the right hook whenever something
the solve depends on must be *recomputed from the current iterate* as the nonlinear
solve proceeds — rather than only once per timestep.

```python
solver.add_update_callback(my_callback)   # my_callback(solver, iteration)
```

Before each call the current Newton iterate is scattered into the solver's field
variable(s) — its single unknown for a scalar/vector solver, or velocity, pressure
(and any block-constraint multipliers) for Stokes — so the callback can read the
fields at the current iterate. After the call the (possibly modified) fields are
gathered back into the iterate, and the mesh auxiliary vector is refreshed so the
next residual evaluation sees any changes. The callbacks are also applied once to the
final converged iterate. With no callback registered the solve path is unchanged.

The scattered fields are correct **everywhere**, including on driven (non-zero
Dirichlet) boundaries: the scatter mirrors the post-solve copy-back —
`globalToLocal` followed by `DMPlexSNESComputeBoundaryFEM` — so a callback that reads
a field next to a driven boundary sees the imposed value, not a stale one.

```{note}
Callbacks run in registration order and receive `(solver, iteration)`. They may
**read** any field, **modify** a field (e.g. shift the pressure), or **fire another
solver** (e.g. a projection/Helmholtz smoother).
```

## Use case 1 — fix the pressure gauge (zero mean pressure on a surface)

On enclosed / all-Dirichlet-velocity problems the pressure is determined only up to
an additive constant. A constant pressure null space makes the system *solvable*, but
leaves the solver free to pick any representative. To pin a **specific, physical**
gauge — for example zero mean pressure on the top surface — use:

```python
stokes.petsc_use_pressure_nullspace = True   # solvability
stokes.set_pressure_gauge("Top")             # gauge: mean pressure on "Top" -> 0
stokes.solve()
```

`set_pressure_gauge(boundary, reference=0.0)` registers a callback that, every
iteration, subtracts the surface-mean pressure from the whole pressure field so that

$$\frac{1}{|\Gamma|}\int_\Gamma p\,dS = \texttt{reference}.$$

After the solve the mean pressure over the chosen boundary equals `reference` to
machine precision:

```python
area = uw.maths.BdIntegral(mesh, 1.0, "Top").evaluate()
mean_top = uw.maths.BdIntegral(mesh, p.sym[0, 0], "Top").evaluate() / area
# mean_top  ~  0
```

## Use case 2 — fire a Helmholtz/Projection smoother each iteration

This is the mechanism behind gradient-plasticity / shear-band stabilisation: the
yield viscosity depends on a *smoothed* (nonlocal) strain-rate field obtained by a
screened-Poisson projection of the local strain rate. To keep that smoothed field
consistent with the velocity *as the nonlinear solve converges*, fire the projection
each iteration:

```python
ebar = uw.discretisation.MeshVariable("ebar", mesh, 1, degree=1)

edot     = stokes.strainrate
e_local  = sympy.sqrt((edot*edot).trace()/2 + e_min**2)

smoother = uw.systems.Projection(mesh, ebar)
smoother.uw_function     = e_local
smoother.smoothing_length = ell          # internal length scale

# yield viscosity uses the SMOOTHED field
stokes.constitutive_model.Parameters.shear_viscosity_0 = \
    1 / (1/eta0 + 2*ebar.sym[0,0]/tau_y)

# re-fire the smoother at every Newton iteration
stokes.add_update_callback(lambda solver, iteration: smoother.solve())

stokes.solve()
```

At convergence `ebar` is the nonlocal average of the strain rate of the converged
velocity, and the regularised yield law has been applied self-consistently within the
nonlinear solve (rather than lagged across timesteps). Because the scatter is
boundary-correct (see above), the smoother sees the imposed velocity even where the
shear band meets a driven wall.

## When to use this vs. a timestep update

- Use a **per-iteration callback** when the residual genuinely depends on a quantity
  that must track the current iterate (regularised rheology, a per-iterate gauge).
- Use the **per-timestep** `DDt.update_pre_solve` / `update_post_solve` hooks for
  history variables that are, by definition, lagged in time (advected fields,
  accumulated plastic strain).
```
