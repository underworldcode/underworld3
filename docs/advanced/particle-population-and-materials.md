---
title: "Particles: population control and materials"
---

# Particles: population control and materials

Two things a particle method has to get right, and how Underworld3 does them.

**Population control** keeps every cell holding enough particles to be worth
integrating, however hard the flow deforms the swarm.

**A material index read where the assembler actually looks** keeps a material
interface where the particles put it, instead of smearing it over a cell.

Both are demonstrated below with runnable scripts.

## Population control

A swarm is not conserved cell by cell. Particles leave through outflow
boundaries and nothing arrives through inflow boundaries unless you put it
there; strong deformation sweeps them out of some cells and piles them into
others. A cell that ends up with too few particles cannot support the
reconstruction that gives the mesh its material properties, and one with none
carries no information at all.

`Swarm.repopulate()` takes a census of the particles per cell and refills the
starved ones from the cell's own lattice, choosing the lattice points furthest
from the particles already there. A new particle takes each swarm variable
from its neighbours: a bounded reconstruction for a continuous field, and its
nearest neighbour's value whole for an integer field, because the average of
two material labels is not a material label.

Set `swarm.population_control` and every `advection()` ends with a
repopulation, before anything reads the swarm again:

```python
swarm.population_control = dict()                     # back to the populate() density
swarm.population_control = dict(min_per_cell=8)       # or a floor you choose
swarm.population_control = dict(min_per_cell=8, values={T: 0.0})   # an inflow datum
```

Or call it yourself, which is what you want if the swarm moves by some route
of your own:

```python
added, removed = swarm.repopulate(min_per_cell=8)
```

`values` overrides what a new particle receives, per variable, with a constant
or a callable of the coordinates. That is how an inflow boundary gets the
right material or temperature rather than a copy of whatever drifted past.
`max_per_cell` will also thin over-full cells, by dropping the particles
closest to a neighbour; it is off by default because discarding particles
costs accuracy.

### Demonstration

`docs/examples/utilities/Ex_Swarm_Population_Control.py`. Pure-shear extension
$\mathbf{v} = (x, -y)$ on a fixed mesh, so the side walls are outflow and the
top and bottom are inflow, with a marker layer through the middle. Particles
that leave are deleted (`mesh.return_coords_to_bounds = None`), which is what
an open boundary means.

```{figure} figures/repopulation_particles.png
:alt: Particles in an extending box, with and without population control

Without population control (top) the box drains: after two time units 1020 of
the original 7640 particles remain and 626 of the 764 cells are empty, and the
marker layer has shredded into streaks. With it (bottom) no cell is ever
empty, and the layer thins as the extension requires rather than falling apart.
```

```{figure} figures/repopulation_counts.png
:alt: Particle count and starved-cell count against time

The count in the box and the number of starved cells. Population control is
not fighting the outflow, which is physical and correct; it is refilling the
inflow side, where the flow brings nothing.
```

## A material index at the integration points

`IndexSwarmVariable` carries a material label per particle and presents the
mesh with one level set per material, so that `createMask` can build a
material-weighted property:

```python
material = uw.swarm.IndexSwarmVariable("M", swarm, indices=2, proxy_degree=1)
viscosity = material.createMask([1.0, 1000.0])
```

Where those level sets live is now a choice:

| `proxy_location` | the level sets are | at an interface |
|---|---|---|
| `"nodes"` (default) | a continuous field per material | a node on the interface averages both materials, so the cells either side see a viscosity that is neither |
| `"integration_points"` | the material of the nearest particle at every integration point | exactly 0 or 1, and the interface keeps its sub-cell position |
| `"cells"` | a polynomial material fraction per cell | a sharp step at cell edges, with a gradient inside the cell |

The integration-point option is the classic particle-in-cell material mapping
of Ellipsis and Underworld: the assembler reads each material property at the
point where it evaluates the weak form, from the particle nearest that point.
Nothing is averaged, so nothing can overshoot into a negative viscosity, and
the masks sum to one by construction.

```python
material = uw.swarm.IndexSwarmVariable(
    "M", swarm, indices=2, proxy_location="integration_points")
```

### Demonstration

`docs/examples/utilities/Ex_Swarm_Material_Index.py`. Two viscosity layers,
1 and 1000, carried as a material index and driven from the top. With the
interface on mesh edges the exact velocity is piecewise linear and lies in the
P2 velocity space, so the only error in the solve is how the material is
represented.

```{figure} figures/material_index.png
:alt: The material mask across the interface and the resulting velocity error

Left: the upper-material mask along a line crossing the interface, as the weak
form sees it. The nodal level set ramps linearly across a whole cell. At the
integration points it is a step in the right place. Right: the resulting error
in the velocity against the exact layered flow.
```

| `proxy_location` | assembled $\int \eta$ (exact 500.5) | velocity $L_2$ error |
|---|---|---|
| `"nodes"` | 500.5000 | 8.0e-2 |
| `"integration_points"` | 500.5000 | **1.8e-7** |
| `"cells"` | 500.5000 | 4.9e-2 |

All three integrate the viscosity correctly in the mean, which is why the
error does not show up in a bulk diagnostic. Only the integration-point
mapping puts the viscosity contrast in the right place, and it solves the
layered problem to solver tolerance.

The `"cells"` option is the one to reach for when the material property needs
a gradient (the level sets are polynomials, and can be differentiated) or when
the field is a fraction rather than a label. For a pure material index at an
interface, `"integration_points"` is the sharper of the two.

## What this rests on

The reconstruction behind `"cells"`, the delta element behind
`"integration_points"`, and the guards that keep both well posed are described
in {doc}`../developer/subsystems/integration-point-variables`. The same
machinery carries semi-Lagrangian and fully Lagrangian histories, including
the viscoelastic stress history.
