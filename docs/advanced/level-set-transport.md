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
| `conserve_mass` | apply the global correction after every step |
| `adv_solver_bc` | box wall labels on which a zero normal gradient is imposed by copying the neighbouring interior nodes |

`initialise_psi` accepts a precomputed signed distance, or a polygon, curve or
`shapely` geometry (the latter three need the optional `shapely` package).
`material_property_field` blends a property across one or more level sets with
a sharp, arithmetic, geometric or harmonic transition.

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
