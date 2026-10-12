---
title: "Particles: population control and materials"
---

# Particles: population control and materials

Two things a particle method has to get right, and how Underworld3 does them.

**Population control** keeps every cell holding enough particles to be worth
integrating, however hard the flow deforms the swarm.

**A material index read where the assembler actually looks** keeps a material
interface where the particles put it, instead of smearing it over a cell.

Both are demonstrated below with runnable scripts. Neither changes how you
write anything: a particle field reaches the mathematics through a proxy, and
what the proxy changes is where the sampling happens, not what you can say.

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

`docs/examples/utilities/intermediate/Ex_Swarm_Population_Control.py`. Pure-shear extension
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

### What it is worth, in a number

The figures above are particle scatters. The field-level cost depends on how
the material is read, and it is worth knowing which case needs the refilling.
Pure shear thins a marker layer by $e^{-t}$, so after $t = 2$ its area should
be 13.5% of where it started; anything else is the representation failing:

| `proxy_location` | population control | layer area vs exact |
|---|---|---|
| `"cells"` | on | 1.03x |
| `"cells"` | off | **5.03x** |
| `"integration_points"` | on | 1.06x |
| `"integration_points"` | off | 1.07x |

Starvation wrecks the per-cell fit, which needs particles *in that cell*. The
nearest-particle mapping degrades gracefully — an empty cell still finds a
plausible particle nearby — so with the default mapping the field-level answer
barely moves, and losing particles shows up only in the swarm's own
bookkeeping. Population control is cheap and always safe; `"cells"` is where
it is *necessary*.

## Materials

Name the materials, say where they are, and stop.

```python
materials = uw.swarm.MaterialSwarm(mesh, fill_param=3)

materials.add("mantle", shear_viscosity_0=1.0,   density=3300)
materials.add("slab",   shear_viscosity_0=1.0e3, density=3400)

materials["slab"] = mesh.X[1] > 0.53

stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
stokes.materials = materials
stokes.bodyforce = -materials.density * mesh.CoordinateSystem.unit_e_1
```

That is the whole interface. `stokes.materials = materials` sets every
constitutive-model parameter the materials declare *and* the model recognises,
by name — here `shear_viscosity_0`. A property the model does not own,
`density`, is not pushed anywhere; it is available as a blended symbol for the
model script to use where it belongs. Properties can be changed and regions
repainted afterwards: the blend is symbolic and the push repeats.

A `MaterialSwarm` **is** a `Swarm`, so everything on the previous page still
applies to it. Declare any extra per-particle state **before** the materials
are first read, though — the first read allocates the particles, and a swarm
cannot gain variables once it holds them:

```python
materials = uw.swarm.MaterialSwarm(mesh, fill_param=3)
materials.add("mantle", shear_viscosity_0=1.0)
materials.add("slab",   shear_viscosity_0=1.0e3)

strain = uw.swarm.SwarmVariable("eps", materials, 1,          # per-particle state
                                proxy_location="integration_points",
                                proxy_sampling="share")

materials["slab"] = mesh.X[1] > 0.53      # reads nothing yet
stokes.materials = materials              # this allocates

materials.population_control = dict(min_per_cell=8)
materials.advection(v.sym, dt)
```

```{warning}
Allocating the level sets is **collective**: every rank must reach it
together. It happens on the first *read* of any material property —
`materials.density`, `materials["x"].mask`, `materials.index`, or
`stokes.materials = ...` — so a read inside a rank-local branch
(`if rank == 0: ...`) deadlocks. Call `materials.build()` at a point where
every rank is together if the ordering is ever in doubt.
```

Regions can be a symbolic condition on the mesh coordinates (and `&`, `|`, `~`
combinations of them), a boolean array over the particles, or a callable of the
coordinate array. `materials.add(..., where=...)` is the same thing said in one
line.

Assignment **replaces**: `materials["slab"] = A` followed by
`materials["slab"] = B` leaves the material at B, and anything it held outside
B reverts to the first material declared. Across *different* materials the
later assignment wins where they overlap.

A few things are refused rather than half-done, because each of them used to
produce a plausible-looking wrong answer:

| | |
|---|---|
| adding or deleting a material after the first read | the index of a material *is* which level set is its |
| two distributions sharing an explicit `name` on one mesh | they would share level sets, silently |
| `mixing(...)` naming a property no material declares | it would have no effect |
| harmonic mixing of a property some material sets to zero | `1/Σ(φᵢ/vᵢ)` divides by it |
| a mesh label value that does not exist | asking PETSc for it aborts the run |

Dimensional values are non-dimensionalised when they are **read**, not when
they are declared, so a registry written before the model's reference
quantities are set means the same thing as one written after.

### What a material is, and where it is

Those are two questions, and Underworld3 keeps them apart.

**What** is a `uw.MaterialRegistry` entry: a name and a table of properties,
with optional description and reference. A registry knows nothing about
geometry, so it can be written before there is a mesh, shared between models,
and exported and read back as plain data.

```python
rocks = uw.MaterialRegistry()
rocks.add("mantle", shear_viscosity_0=1.0,   density=3300,
          description="upper mantle", reference="Turcotte & Schubert (2014)")
rocks.add("slab",   shear_viscosity_0=1.0e3, density=3400)
```

**Where** is a *distribution*, and there are two. `MaterialSwarm` carries the
materials on particles, so they advect with the flow. `MaterialRegions` ties
them to the mesh:

```python
materials = uw.MaterialRegions(mesh, registry=rocks)
materials["slab"] = "Slab"                  # a gmsh physical group
materials["slab"] = mesh.X[1] > 0.53        # or a geometric condition

stokes.materials = materials
```

Both take the same registry, declare materials the same way, and hand the same
thing to the solver. `materials.add(...)` on either is shorthand that writes
into its registry, so a two-material script never has to mention one.

| | `MaterialSwarm` | `MaterialRegions` |
|---|---|---|
| the material moves | yes | no |
| level sets are | sampled from the particles | exact, 0 or 1 |
| needs population control | yes | no |
| interface position | sub-cell, to the particle density | sub-cell, to the rule |
| cost per step | a proxy fill | nothing; built once |

Use regions when the geometry is fixed — a layered model, an inclusion, a
basin. It is exact and free. Use the swarm when the material is carried by
the flow, or when it has to carry per-particle history as well.

### A property can be a law, not just a number

This is the whole reason for the machinery underneath:

```python
materials.add("crust", shear_viscosity_0=eta_0 * sympy.exp(-T.sym[0]))
```

There is no number to store at an integration point for that material, so it
cannot be handled by sampling a viscosity field. It is carried symbolically and
combined with the other materials' laws by the level sets described below.

### There is no route that puts the property on the particles

Carrying the viscosity itself on the particles and sampling it looks simpler
and is the wrong shape. PETSc calls the compiled pointwise functions with
values tabulated at the rule points, so a solver handed a sampled *answer*
cannot reach the parts of the constitutive law it needs — the tangent, the
yield surface, the history update. Keeping the law symbolic and letting the
materials weight it is what makes a material model composable. (The mechanical
symptom is easy to see too: the linear-exact reconstruction a smooth field
wants overshoots a factor-1000 viscosity jump to −219, and a negative viscosity
is not a viscosity. Asking for nearest-particle sampling on a plain
`SwarmVariable` raises, and the error says so.)

### What is underneath

A `MaterialSwarm` carries one integer label per particle and presents the mesh
with one *level set* per material — a partition of unity — so that a property
is `Σ φᵢ · valueᵢ`. Where the masks are 0 or 1 that sum is a select, and if
every property were a number a single stored coefficient field would do the
same job; the sum earns its place the moment a property is a law, because there
is then no other way to combine N expressions into one symbol the assembler can
compile.

That machinery is `uw.swarm.IndexSwarmVariable` and its `createMask`. Models
written before `MaterialSwarm` use it directly and still work; new models
should not need to see it. The two things worth knowing about it are the two
arguments `MaterialSwarm` passes through, below.

#### `proxy_location` — where the level sets live

Where the level sets are stored is what the assembler actually reads:

```python
materials = uw.swarm.MaterialSwarm(mesh, proxy_location="cells")
```

| `proxy_location` | the level sets are | at an interface |
|---|---|---|
| `"integration_points"` **(default)** | a value at each point of the quadrature rule | exactly 0 or 1, and the interface keeps its sub-cell position |
| `"cells"` | a polynomial material fraction per cell | a sharp step at cell edges, with a gradient inside the cell |
| `"nodes"` | a continuous field per material | a node on the interface averages both, so the cells either side see a viscosity that is neither |

The default is the integration points: it is the classic particle-in-cell
material mapping of Ellipsis and Underworld, and it is measurably the best of
the three. The nodal option is kept for continuity with existing models, but
its smear is about one cell wide *however many particles you add* — that is a
property of the basis, not of the swarm — and the current distance-weighted
fill actually *widens* the band as the swarm is refined.

#### `proxy_sampling` — what each point reads

How the particles in a cell become the value at each of its integration
points. Applies only to `proxy_location="integration_points"`.

| `proxy_sampling` | each point takes | masks |
|---|---|---|
| `"nearest"` **(default)** | the material of its nearest particle | exactly 0 or 1 |
| `"share"` | the material fractions of the particles it speaks for — those whose nearest integration point *within their own cell* is this one | fractional where a cell is crossed |

`"nearest"` is sharp and assumption-free: nothing is mixed, so no mixing rule
is implied. It sub-samples, though — at ten particles per cell and six rule
points, most particles never reach the assembly.

`"share"` is the cell-restricted Voronoi share, and it is the closest thing to
the true Voronoi integration that a fixed quadrature rule allows: the rule
points partition their own cell between them, every particle lands in exactly
one part, and nothing crosses a cell boundary. For an *identity* that buys
fractional masks in the crossed cells, and then the answer depends on how you
blend — see the trap below.

#### What the choice is worth

Interface at $y = 0.53$ on an irregular mesh, so no scheme can be exact (the
P2 velocity cannot hold a kink inside a cell), viscosity contrast 1000,
against the exact layered Couette profile:

| particles per cell | `"nearest"` | `"share"` + `createMask` | `"share"`, blended harmonically |
|---|---|---|---|
| 3 (2 420) | 3.48e-2 | 3.53e-2 | 1.83e-2 |
| 8 (10 890) | **1.85e-2** | 3.53e-2 | 1.27e-2 |
| 15 (32 912) | 1.85e-2 | 3.61e-2 | **1.24e-2** |

```{figure} figures/material_share.png
:alt: Identity error against particle density, and the history overshoot

Left: only `"nearest"` and the harmonically-blended share improve as particles
are added; the arithmetic (Voigt) blend of fractional masks is flat. Right: the
share is bounded by the particle values because it is an average of them, while
the reconstruction's overshoot grows with the swarm.
```

Three things to read off it.

**`"nearest"` converges to a floor and then stops.** By about eight particles
per cell the error stops moving: what limits it is the quadrature rule and the
velocity space, not the swarm. Past that point, refine the mesh — more
particles buy nothing.

**Fractional masks need a mixing rule you have chosen.** `createMask` blends
*arithmetically*, which for a flux at a common strain rate is a Voigt average,
and on a sharp contrast a Voigt average does not converge with particle
density at all — the middle column is flat. The same masks blended
harmonically (Reuss, `1.0 / material.createMask([1/η₀, 1/η₁])`) converge, and
past the `"nearest"` floor. So fractions are worth having when the material
genuinely *is* a sub-cell mixture and you have picked the rule on physical
grounds; for an interface, `"nearest"` says the true thing (nothing is mixed)
without you having to.

**Cost is not the discriminator.** Filling the level sets took 1.0 / 1.7 /
3.7 ms for `"nearest"` and 0.7 / 1.3 / 3.9 ms for `"share"` at the three
densities.

With the interface on mesh edges instead, the exact velocity lies in the P2
space and the only error left is the material representation:

| `proxy_location` | assembled $\int \eta$ (exact 500.5) | velocity $L_2$ error |
|---|---|---|
| `"nodes"` | 500.5000 | 8.0e-2 |
| `"integration_points"` | 500.5000 | **1.8e-7** |
| `"cells"` | 500.5000 | **1.8e-7** |

All three integrate the viscosity correctly *in the mean*, which is exactly
why a bulk diagnostic cannot see the difference. Only the placement differs,
and the placement is what the solve feels.

```{figure} figures/material_index.png
:alt: The material mask across the interface and the resulting velocity error

Left: the upper-material mask along a line crossing the interface, as the weak
form sees it. The nodal level set ramps linearly across a whole cell; the other
two are a step in the right place and lie on top of each other. Right: the
resulting error in the velocity against the exact layered flow.
```

`docs/examples/utilities/intermediate/Ex_Swarm_Material_Index.py` runs it.

### Mixing, when the masks are fractional

`materials.mixing(shear_viscosity_0="harmonic")` chooses how a property is
blended. With the default sampling exactly one mask is 1 at every integration
point, so every rule gives the same answer and there is nothing to choose. With
`proxy_sampling="share"` there is, and the table above is the reason to choose
it on physical grounds rather than by trying both.

## Material state and history

Identity is not the only thing particles carry. A viscoelastic stress, an
accumulated strain, a damage variable — these are *state*, earned by being
advected, and every particle's is different. They ride on ordinary
`SwarmVariable`s, and they get the same choice of where the assembler reads
them and how:

```python
stress = uw.swarm.SwarmVariable(
    "tau", materials, (2, 2),      # the MaterialSwarm is the swarm
    proxy_location="integration_points",
    proxy_sampling="share")        # every particle's history contributes
```

`Lagrangian_Swarm` takes the same argument, so a viscoelastic stress history
carried on the particles is read the same way:

```python
DFDt = uw.systems.ddt.Lagrangian_Swarm(
    swarm=materials, psi_fn=sympy.Matrix.zeros(2, 2),
    vtype=uw.VarType.SYM_TENSOR, degree=1, continuous=False,
    order=2, step_averaging=1,
    proxy_location="integration_points", proxy_sampling="share")
```

Build it before anything reads the materials: it adds swarm variables, and a
swarm cannot gain variables once it holds particles.

For state, `"share"` is the one to reach for, and for a different reason than
it was rejected for identity:

- **It uses the whole swarm.** Each point averages the particles it represents
  rather than adopting one of them, so a history that varies within a cell is
  represented by all of it. This is the "more PIC than not" part: the
  quadrature rule is fixed, but what it reads can still be a weighted account
  of every particle.
- **It cannot cross a material boundary.** The default `"reconstruct"` gathers
  from the nearest particles by distance, and that stencil ignores cell walls;
  across a jump it both smears and overshoots. Measured on a discontinuous
  particle field read at the integration points (interface at $y = 0.53$,
  $h = 0.1$):

  | particles per cell | `"reconstruct"` range | `"share"` range |
  |---|---|---|
  | 3 | −0.051 … +1.137 | 0.000 … 1.000 |
  | 8 | −0.068 … +1.108 | 0.000 … 1.000 |
  | 15 | −0.109 … +1.097 | 0.000 … 1.000 |

  The share is bounded by the particle values by construction — it is an
  average of them — so it cannot invent a stress the swarm never held. The
  reconstruction's overshoot *grows* as particles are added.

Keep `"reconstruct"` (the default) for a field that really is smooth: it is
exact for linear fields, where the share carries a small averaging error.

**Cost.** The share needs to know each particle's cell, and locating particles
is the expensive half: 18.8 ms for 32 912 particles against 3.2 ms for the
share itself, on a mesh of 242 cells. That location is cached on the swarm
until the particles move and is reused by every share variable and by
`repopulate`, so a model with several history fields and population control
pays it once per step. For comparison, the `"reconstruct"` path costs 8.1 ms
per step on the same swarm (its cached operator is geometry-only, so it is
rebuilt whenever the particles move).

## What can be said about a particle field

A particle field is a first-class citizen of the symbolic algebra: it carries a
symbol, and that symbol goes wherever a mesh variable's symbol goes. The one
exception is a derivative of the integration-point form.

| | `"nodes"` | `"integration_points"` | `"cells"` |
|---|---|---|---|
| arithmetic with mesh variables and `sympy` | yes | yes | yes |
| `uw.maths.Integral` | yes | yes | yes |
| `uw.function.evaluate` anywhere | yes | yes | yes |
| projection onto a mesh variable | yes | yes | yes |
| viscosity, body force, any solver term | yes | yes | yes |
| a **gradient** of the expression | yes | refused | yes |

The element that holds a value at each integration point has no gradient to
give, so the compiler refuses one rather than returning the silent zero its
tabulation would produce. The gradient is still available from a projection of
the same data, and `"cells"` is that projection: its level sets are a
least-squares polynomial per cell, so they differentiate directly and with no
global solve. Recovering $\partial_x$ of a quadratic particle field:

| where the proxy lives | gradient error |
|---|---|
| `"nodes"`, degree 1 | 2.1e-3 |
| `"nodes"`, degree 2 | 1.1e-4 |
| `"cells"`, degree 1 | 1.3e-3 |
| `"cells"`, degree 2 | **2.4e-7** |
| a global L2 projection onto P2, then differentiate | 1.1e-4 |
| `"integration_points"` | refused |

The per-cell fit is the most accurate of them because it is local and exact for
polynomials up to its degree, and no further projection follows it.

**Two paths, and the difference is deliberate.** A weak form and a query are
not the same thing:

- **In a weak form**, a derivative of an integration-point field is *refused*.
  Answering would mean either the silent zero its own tabulation gives, or a
  reconstruction chosen behind your back and paid for at every assembly.
  Which discretisation the gradient comes from is a modelling decision, so it
  stays yours: build the variable with `proxy_location="cells"` and the level
  sets are already polynomials.
- **`uw.function.evaluate`** is a query, and it *answers*. It fits the
  integration-point values cell by cell and differentiates that, once, for
  this call. The result converges (2.4e-3, 6.1e-4, 2.6e-4 for a quadratic
  field as the cell size halves from 1/5 to 1/20) but it is a *recovered*
  gradient, so treat it as a diagnostic rather than as the field's own
  derivative. The `"cells"` route reaches 2.4e-7 on the same field.

So the rule is: sample at the integration points when you want a value placed
exactly, fit per cell when you want to differentiate, and expect `evaluate` to
help you look at a gradient either way.

## What this rests on

The reconstruction behind `"cells"`, the delta element behind
`"integration_points"`, the share, and the guards that keep them well posed
are described in {doc}`../developer/subsystems/integration-point-variables`.
The same machinery carries semi-Lagrangian and fully Lagrangian histories,
including the viscoelastic stress history.
