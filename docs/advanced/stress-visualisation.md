# Visualising the Stress Tensor

Scalar fields get colormaps and velocity gets arrows; the stress tensor
needs its own glyph. `underworld3.visualisation` provides
principal-stress glyphs — sampled at seed points, the way velocity
arrows sample the velocity — and stress trajectories, the curvilinear
net traced by the principal directions. Both work from the same
recovered stress fields you already checkpoint.

## Principal-stress glyphs

At each seed point we diagonalise the stress and draw one bar per
principal axis: bar length proportional to the principal-value
magnitude, blue for compressive ($\lambda < 0$), red for tensile. In
2-D this is the classical stress cross; in 3-D each seed carries three
orthogonal bars, best drawn on one or two section planes rather than a
filled volume.

```python
import underworld3 as uw
import underworld3.visualisation as vis
import sympy

# Recovered stress components (P1 projections of the deviatoric
# stress) plus pressure give the full stress. Project once after the
# solve; do not pass raw solver derivative expressions to a plot.
# A scalar variable's .sym is a 1x1 Matrix — index it before
# assembling the tensor.
stress = sympy.Matrix([[Txx.sym[0] - P.sym[0], Txy.sym[0]],
                       [Txy.sym[0], Tyy.sym[0] - P.sym[0]]])

pl = vis.plot_stress_glyphs(mesh, stress, num_seeds=24,
                            save_png=True, dir_fname="stress_glyphs.png")
```

Seeds default to a regular grid over the mesh bounding box, filtered
to points inside the mesh (an annulus seeds no glyphs in its hole).
Pass `seeds` explicitly to sample section planes in 3-D or to avoid
regions — in fault models, keep seeds out of the weak zones, where the
recovered stress mixes materials across the interface.

```{figure} figures/stress_glyphs_thrust.png
:alt: Two-panel figure for a thrust-ramp model. Top panel, principal stress crosses on a 26 by 13 seed grid over a 2 by 1 section with two grey parabolic ramps rising from a basal decollement. Far-field crosses are blue-horizontal (compression); in the wedges riding each ramp the crosses turn red and near-vertical (tension), with the largest red bars just above the blind ramp tips. Bottom panel, stress trajectories: dark sigma-1 lines run horizontally and arch smoothly over each ramp tip, pale sigma-3 lines rise near-vertically between them, and the two families cross at right angles everywhere.
:name: fig-stress-glyphs-thrust

Principal-stress crosses (top) and stress trajectories (bottom) for a
blind-thrust model. The colour convention matches the RdBu_r field
convention used across the documentation: blue compressive, red
tensile. The trajectory panel shows the $\sigma_1$ family (dark)
arching over the ramp tips with the $\sigma_3$ family (pale)
orthogonal to it.
```

### The pressure gauge matters for colours, not directions

For incompressible models the full stress is
$\sigma = \tau - p\,I$ and the pressure datum is a gauge choice.
Shifting that datum shifts every principal value equally, so it can
flip bars between red and blue — but it cannot rotate the principal
directions or reorder the principal values. State the gauge in your
caption (the examples here demean the pressure), or add the
lithostatic reference before plotting if absolute compression
matters.

The same argument answers a common question about map-view regime
colouring in the style of the World Stress Map (red normal, green
strike-slip, blue thrust): the regime classification is
gauge-invariant, but in a 2-D incompressible plane-strain model it is
also *degenerate* — the out-of-plane deviatoric stress is zero while
the in-plane deviatoric principals are $\pm s$, so the out-of-plane
stress is always the intermediate principal stress and every map-view
point classifies as strike-slip. Regime colouring only carries
information in 3-D models.

## Stress trajectories (2-D)

Trajectories integrate the principal *direction* field into curves —
the classical stress-trajectory diagrams of structural geology. A
principal direction is defined only modulo 180°, so ordinary
streamline tools cannot draw this field: the integrator in
`direction_trajectories` sign-aligns each evaluated eigenvector with
the previous heading, and places lines evenly (Jobard–Lehmann
occupancy) so the figure stays legible. The two families cross at
right angles wherever both are drawn — a built-in correctness check.

The direction callable is yours to build, which keeps the integrator
independent of how the stress is stored. From nodal arrays:

```python
import numpy as np

def make_direction(interpolate_stress, family="compressive"):
    # interpolate_stress(p) -> (sxx, syy, sxy) at point p, or None
    def direction_at(p):
        values = interpolate_stress(p)
        if values is None:
            return None
        sxx, syy, sxy = values
        mean_dev = 0.5 * (sxx - syy)
        radius = np.hypot(mean_dev, sxy)
        if radius < 1.0e-12:      # isotropic point: direction undefined
            return None
        angle = 0.5 * np.arctan2(sxy, mean_dev)   # most-tensile axis
        if family == "compressive":
            angle += 0.5 * np.pi
        return np.array([np.cos(angle), np.sin(angle)])
    return direction_at

lines = vis.direction_trajectories(
    direction_at, candidate_seeds, inside,
    step=0.008, separation=0.04,
)
trajectory_lines = vis.trajectories_to_pv_lines(lines)
```

Draw the compressive family dark and the tensile family pale, as in
the figure above. In 3-D the analogue of a trajectory is a surface;
we do not attempt those — draw glyphs on section planes instead.

## 3-D glyphs

`principal_stress_glyphs` accepts `(n, 3, 3)` tensors and returns
three bars per seed. Seed one or two planes through the feature of
interest:

```python
u = np.linspace(0.05, 0.95, 13)
gx, gz = np.meshgrid(u, u)
plane = np.column_stack([gx.ravel(),
                         np.full(gx.size, 0.5),   # y = centre plane
                         gz.ravel()])
pl = vis.plot_stress_glyphs(mesh, stress, seeds=plane)
```

```{figure} figures/stress_glyphs_sinker3d.png
:alt: A unit cube drawn in outline with a grey sphere of radius 0.16 near the centre, slightly above mid-height. Three-bar principal stress glyphs are drawn on a vertical section plane and a horizontal section plane through the sphere. Above the sphere the bars are red and near-vertical (tension as material is pulled down behind the sinker); below and beside it they are blue (compression), fanning outward on the horizontal plane beneath the sphere. Bar length decays with distance from the sphere.
:name: fig-stress-glyphs-sinker

Three-bar principal-stress glyphs on two section planes through a
Stokes sinker: a tensile (red) column above the sinking sphere, a
compressive (blue) fan below and around it.
```

## Building blocks

The plot function is a convenience wrapper; every step is available
separately for custom figures:

| Function | Purpose |
|---|---|
| `tensor_fn_to_pv_points(pv_mesh, uw_fn)` | Evaluate a `dim`×`dim` sympy tensor at points |
| `principal_stress_glyphs(coords, stress, scale)` | Bar segments + `"tensile"` cell array |
| `direction_trajectories(direction_at, seeds, inside, step, separation)` | Evenly spaced mod-180° trajectories |
| `trajectories_to_pv_lines(lines)` | Bundle polylines for `add_mesh` |
| `plot_stress_glyphs(mesh, stress, ...)` | One-call cross plot |

The figures on this page come from checkpointed models (a blind-thrust
fault network and a Stokes sinker): the solve writes the mesh, the
velocity, the pressure, and the recovered stress components with
`mesh.write_timestep`, and the glyph plots load them back with
`read_timestep` — no re-solving to restyle a figure.
