# Viscoelastic stress transport

A viscoelastic constitutive model carries the stress from one step to the next. The
solver owns the unknowns (velocity, pressure); the stress history is a transported
field managed by one of the `DDt` flavours, chosen with `solver.stress_transport`
before the constitutive model is assigned. This page says what each flavour does,
what limits it, and how to keep a run inside those limits.

```python
stokes = uw.systems.Stokes(mesh, velocityField=v, pressureField=p)
stokes.stress_transport = "integration_point"       # or "semi_lagrangian" (the default), "forward", "eulerian"
stokes.constitutive_model = uw.constitutive_models.ViscoElasticPlasticFlowModel(
    stokes.Unknowns, order=1, integrator="bdf", objective_rate="upper_convected")
stokes.constitutive_model.Parameters.shear_viscosity_0 = eta_p
stokes.constitutive_model.Parameters.shear_modulus = G
stokes.constitutive_model.Parameters.solvent_viscosity = eta_s    # Oldroyd-B; omit for Maxwell
stokes.constitutive_model.Parameters.dt_elastic = dt
```

## The four histories

| `stress_transport` | storage | carried by | stable at | fails by |
|---|---|---|---|---|
| `semi_lagrangian` (nodal) | continuous P1 at the vertices | vertex trace-back, interpolation at the foot | any Courant number | excess stress in the first cells off a no-slip wall; on the confined cylinder that excess loses the conformation and the solve hangs |
| `integration_point` | continuous P1 store, sampled at the quadrature points | trace-back of every quadrature point | Courant near one, or below one with store smoothing | a cell-scale mode of the stress that grows below Courant one when the solvent viscosity is small |
| `forward` | discontinuous P1 per cell, fitted from the arrivals | fixed launch set of interior points (the integration points), one forward trajectory a step; the flux is read back at the launch points through a continuous P1 projection; an inflow cell's uncovered share is filled with the inflow value | the cylinder walls at dt 0.04; below Courant one with `flux_smoothing` at c = 0.023 (Waters-King 1/16, dt 0.0125: 0.9543 at t 1 and 0.5185 at t 6.5, against nodal 0.9622 and 0.5171) | the same cell-scale mode as the integration-point history without that smoothing (diverges at t 2.4 there); first order only; does not cross a periodic seam or follow a moving mesh |
| `eulerian` (SUPG grid) | continuous P1 | assembled transport equation with streamline upwinding | with DEVSS | without DEVSS the velocity block loses its preconditioner as the stress grows |

The nodal and integration-point flavours store the stress after every solve by
the same global L2 projection onto the continuous space. The integration-point
flavour differs only in where it samples that field: the quadrature points along
their own characteristics, rather than the vertices. Nothing is carried at the
points from one step to the next. The forward flavour is the exception: its
launch values persist, and a cell that receives no fit keeps its previous one.

The momentum equation sees the stress only through $\int \sigma : \nabla v$, so
only its per-cell P1 projection matters, and the viscous part is rebuilt from
$\nabla u$ every step while the memory part decays on the relaxation time. That
is why an interpolating history that would be too diffusive for temperature or
velocity is acceptable for stress. Where it fails is where the projection itself
is wrong: sub-cell layers and no-slip walls.

## The timestep is set by the wall strain rate, not the far-field Courant number

The objective-rate source $L\sigma^* + \sigma^* L^T$ acts on the carried stress
with the current velocity gradient, so over one step it stretches the conformation
by about $(1 + \Delta t\,\dot\gamma)^2$ before relaxation acts. When
$\Delta t\,\dot\gamma$ is of order one that update loses the positive-definiteness
of the conformation $c = \sigma^*/G + I$ in the first step, and no arrangement of
the split recovers it. On the confined cylinder at Courant one on the far-field
mesh the wall shear rate is ten times the far-field one: every history lost the
conformation at the cylinder top in step one, the nodal history then ran away and
the solve hung, and the integration-point history gave out at Wi 0.6. At a step
ten times smaller the integration-point history is admissible everywhere to Wi 0.6,
and at Wi 0.8 the conformation is mildly indefinite on two percent of its points.

```python
dt = min(courant_dt, stokes.constitutive_model.max_elastic_timestep(safety=0.3))
```

`max_elastic_timestep` is the safety factor over the largest strain-rate magnitude
$\sqrt{2\,\mathbf{D}:\mathbf{D}}$ (the shear rate in simple shear), read from the
continuous projection of the strain rate at the points of the carried stress. That
projection sits a little below the per-cell gradient at a wall, which the safety
factor covers: 0.3 kept the conformation positive on the cylinder. The value is a
physical time when reference scales are active, as `estimate_dt` is.

## Watching the conformation

```python
health = stokes.constitutive_model.conformation_min_eigenvalue()
# {'min': 0.29, 'max': 79.3, 'fraction_negative': 0.0, 'where': (0.001, 1.07)}
```

An Oldroyd-B or Maxwell stress is $G(c - I)$ with $c$ positive-definite, so the
most compressive eigenvalue of the polymer stress is bounded by $-G$, and the
symmetric part of the momentum tangent stays positive exactly as long as that
holds. A negative `min` is a discretisation defect and it tells the two failure
modes apart: a solve that has lost its preconditioner (the non-symmetric,
co-rotational part of the tangent grows with $|W||\sigma^*|/G$ and is a solver
setting) from a solve that has lost its problem (nothing recovers it). Print this
line every step on a new problem.

## The recommended configuration

Integration-point history, the step set by the wall strain rate
(`max_elastic_timestep`), store smoothing at c = 0.07 when the solvent viscosity
is a small fraction of the total, DEVSS off. That combination is characterised on
Waters and King (regular and irregular meshes) and on the confined cylinder,
admissible to Wi 0.6 and mildly indefinite at Wi 0.8. The coefficient 0.023 is
the least that holds the mode on a regular mesh; 0.07 holds it on an irregular
one as well and costs half a percent, so it is the recommended value. The forward flavour is the same scheme with a per-cell fit and interior
launch points, measured at 17 s a step against 28 on the cylinder, parallel by
handing the arrivals that cross a seam to the rank that owns them; it needs its read-back smoothing (`DFDt.flux_smoothing = 0.023 * mesh.cell_size()**2`). Neither transports its memory without a cell-scale mode below Courant
one on a Maxwell element: a version that did not ring turned out not to be
transporting the memory at all.

## Store smoothing for the integration-point history below Courant one

The store cycle of the integration-point flavour, sample at the points then
project, is a consistent-mass Galerkin transport of the carried stress. It has no
dissipation at the cell scale, so below Courant one a cell-scale mode grows from
round-off at a rate $\gamma$ set by the elastic feedback: about 2.4 per unit time
on the Maxwell Waters-King start-up, 1.8 with a solvent fraction of 0.2, and not
observed at 0.59 over the run lengths used. The vertex interpolation of the nodal
flavour damps that mode; a discontinuous store (private nodes per cell) frees it
and diverges; DEVSS does not touch it.

A Laplacian term in the store projection does. It multiplies wavenumber $k$ by
$1/(1 + \alpha k^2)$ once per step, so the mode is held when
$\alpha \approx \gamma\,\Delta t\,(h/\pi)^2$. The coefficient is a field, so the
dose follows the local cell on a graded mesh:

```python
stokes.DFDt.store_smoothing = 0.07        # alpha = 0.07 * mesh.cell_size()**2
```

Measured on the Waters-King start-up at 1/32 and $\Delta t = 0.01$: $c = 0.023$
($\alpha = 10^{-5}$) holds the mode for eight time units at 0.1% on the peak,
$c = 0.07$ holds it unconditionally at 0.5%, and $c = 0.23$ (ten times 0.023)
costs 3%. The irregular mesh needs 0.07. Zero, the default, is the plain projection. When the
solvent viscosity is a fair fraction of the total, as on the cylinder benchmark,
the smoothing is unnecessary and costs nothing if left on.

## DEVSS

`solver.devss_viscosity = eta_a` adds $2\eta_a(\mathbf{D} - \bar{\mathbf{D}})$
to the momentum flux, with $\bar{\mathbf{D}}$ the continuous projection of the
strain rate, iterated within the solve until the pair cancels. It is a viscosity
on cell-scale velocity modes only. It rescues the grid (SUPG) history on the
cylinder at Wi 0.4 (80,000 inner solves at the iteration cap without it, none
with), is harmless on the integration-point history, does nothing for the nodal
collapse, and does nothing for the store-cycle mode above. Use it with the grid
history; treat it as optional elsewhere.

## Waters and King as the discriminating test

The start-up of plane Poiseuille flow of a Maxwell fluid (Waters and King 1970)
has a reference solution and separates the histories: nodal converges as the
step falls, the integration-point history rings then diverges as the step falls
without smoothing, and the grid history diverges at every step. Note that the
exact solution is independent of $x$, so $u\cdot\nabla\sigma \equiv 0$ there:
the test exercises the store cycle, not the trace-back. A pure Maxwell element is
also the most demanding case for anything explicit in the stress.

Related: [constitutive models](constitutive-models.md),
[integration-point variables](integration-point-variables.md),
[solvers](solvers.md). Tests: `tests/test_1059_stress_transport.py` (the
histories, the conformation check, the elastic timestep), `test_1060` (store
smoothing below Courant one), `test_1061` (the forward flavour on Waters and
King), `tests/parallel/test_1062` (the forward flavour at np 2), `test_1063`
(save and restore).
