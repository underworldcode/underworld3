# Conservative level sets

`uw.systems.LevelSetSolver` carries a material interface as the 0.5 contour of a
smoothed indicator

$$
\psi = \tfrac12\left(1 + \tanh\frac{\varphi}{2\varepsilon}\right),
$$

where $\varphi$ is the signed distance to the interface (positive inside) and
$\varepsilon$ the interface thickness, a fraction of the local cell size. The
field is transported by an ordinary scalar solver; what makes it a level set is
what happens after each step:

- **reinitialisation** restores the $\tanh$ profile without moving the 0.5
  contour (Parameswaran and Mandal 2023, integrated in pseudo-time with SSP-RK3);
- **mass correction** restores the enclosed volume by a uniform, clipped shift
  found by bisection (Zhang, Zou and Greaves 2010).

Neither depends on the transport scheme, so the solver takes either the Eulerian
SUPG solver (the default) or the semi-Lagrangian one.

```python
from underworld3.systems import level_set

psi = uw.discretisation.MeshVariable("psi", mesh, 1, degree=2)
eps = level_set.interface_thickness(mesh, psi, scale=0.35)
level_set.initialise_psi(psi, eps, interface_geometry="polygon",
                         interface_coordinates=circle_points)   # or signed_distance=...

ls = uw.systems.LevelSetSolver(psi, velocity=v.sym, epsilon=eps)  # advection="slcn" to compare
for step in range(n_steps):
    ls.solve(dt)                     # advect, reinitialise when due, restore the volume

viscosity = level_set.material_property_field(psi.sym[0], [eta_outside, eta_inside], "geometric")
```

## Choices

| argument | meaning |
|---|---|
| `advection` | `"supg"` (default) or `"slcn"`; both run pure advection |
| `order`, `theta` | the transport solver's time scheme; Crank-Nicolson by default, which preserves the profile's amplitude between reinitialisations |
| `reini_frequency`, `reini_steps`, `reini_dt` | how often, how many pseudo-time steps, and how long each is (half the smallest $\varepsilon$ by default) |
| `far_field` | the value of $\psi$ imposed on the domain boundary; set it whenever the flow crosses the boundary (an inflow boundary with no value lets mass in) |
| `conserve_mass` | `"auto"` (default): the global correction is on for `"slcn"`, which loses volume by interpolation, and off for `"supg"`, which conserves it to solver tolerance on its own; the clip to [0, 1] of the ringing at a one-cell band then costs about 0.2% per revolution, which `volume_drift` reports |
| `adv_solver_bc` | box wall labels on which a zero normal gradient is imposed by copying the neighbouring interior nodes |

**Band thickness.** `interface_thickness(scale=0.35)`, the g-adopt default,
gives a band well under one cell, which a continuous-Galerkin transport rings
at. Measured on a rotating circle at 32 cells across, one revolution, SUPG with
no mass correction:

| `scale` | $\varepsilon / h$ | volume drift |
|---|---|---|
| 0.35 | 0.12 | +0.84% (clipped ringing) |
| 1.0 | 0.36 | +0.28% |
| 2.0 | 0.71 | -0.18% (ringing gone) |
| 3.0 | 1.07 | -0.85% (reinitialisation curvature error) |

For the SUPG transport a `scale` of 1.5 to 2, a band of two to three cells, is
the sensible setting; the thickness trades interface resolution for a clean
transport.

`initialise_psi` accepts a precomputed signed distance, or a polygon, curve or
`shapely` geometry (the latter three need the optional `shapely` package).
`material_property_field` blends a property across one or more level sets with
a sharp, arithmetic, geometric or harmonic transition.

## Cost

Per step at 64 by 64 (LeVeque flow, Courant 0.5, reinitialisation every fifth
step): the SUPG advection takes 0.13 s and the SLCN advection 1.24 s; the
reinitialisation 0.06 to 0.11 s averaged; the mass correction 0.13 to 0.24 s.
Since the Eulerian transport does not need the correction, its level-set step
costs about 0.19 s against 1.59 s for the semi-Lagrangian one.

## Which transport solver

On the LeVeque swirling flow at 64 by 64 and Courant 0.5 (period 2), the SUPG
level set returns with a shape error of 0.028 against 0.051 for the
semi-Lagrangian one, at half the wall time; the mass correction pins both to
the same volume. The Eulerian solver's advantage is the same as for any scalar:
no interpolation loss per step, and cells refined for the Stokes problem cost
nothing. See {doc}`eulerian-advection-diffusion`. The example
`docs/examples/convection/advanced/Ex_LevelSet_LeVeque_SUPG_vs_SLCN.py` runs the
comparison.

## Credit

The level-set pipeline, its SUPG transport and the LeVeque comparison are
NengLu's contribution (issue #657); this module unifies the two variants of
that work on the shared solver interface.
