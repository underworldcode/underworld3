# Listric faults over a common detachment: two ways to build a fault

The classic geometry of a continental rift: listric normal faults — steep
near the surface, flattening with depth — soling into a **common
detachment** in weak rock, with the hanging-wall blocks stepping down
between them. This example builds that system twice on the same base mesh,
with the two fault representations UW3 provides, and the comparison is the
lesson: **how you treat the junctions decides whether the system behaves
like linked tectonics or like isolated cracks.**

Scripts: `~/+Simulations/listric_extension/` (`listric_lines.py`,
`listric_ribbon.py`, shared geometry in `listric_common.py`).

## The geometry

A listric trace flattening toward the detachment depth $y_d$:

```python
DETACH_Y = 0.4                      # the common detachment depth
LAM = 0.18                          # flattening length

def listric_trace(x0, y_lo, n=30):
    y = np.linspace(1.0 - TOP_GAP, y_lo, n)
    x = x0 + LAM * np.log((1.0 - DETACH_Y) / (y - DETACH_Y))
    return np.column_stack([x, y])
```

Two faults (`x0 = 0.35` and `1.05`) and a horizontal detachment segment
spanning beneath both. The instantaneous Stokes problem is plain extension:
sides pulled apart at ±0.5, free-slip base, free top, viscosity 1 in the
country rock and $10^{-3}$ in the fault material, assigned **per cell**
(P0, discontinuous — interpolating a nodal viscosity across a sharp
interface smears exactly what the conforming mesh resolves).

## Way one: finite-width ribbons, soled into the detachment

```python
from underworld3.utilities.place_surface import place_thin_volume

dm, info = place_thin_volume(mesh0.dm, [sole, f1, f2], width=0.02,
                             label="Zone", label_value=31)
```

One call places the whole network. The traces are thickened to ribbons of
the given **constitutive width** and resolved against one another in the
CAD kernel — the fault–detachment junctions become ordinary cells of the
union, with no geometric treatment at all. The zone's cells carry the
label; the weak viscosity is read straight off it.

```{figure} figures/listric_ribbon.png
:width: 90%

Ribbons soled into the detachment. The hanging-wall blocks move as
coherent units (middle), and the detachment is ACTIVE between the two
junctions (bottom) — the textbook rift cartoon, reproduced.
```

## Way two: zero-thickness surfaces, with the junction gap

```python
from underworld3.utilities.place_surface import place_along_lines

dm, _ = place_along_lines(mesh0.dm, [sole], label="Sole", label_value=31)
dm, _ = place_along_lines(dm, [f1], label="F1", label_value=32)
dm, _ = place_along_lines(dm, [f2], label="F2", label_value=33)
```

Zero-thickness surfaces are placed **one at a time**, and two surfaces
closer than a cell are refused — converging surfaces compete for the same
cavity — so each listric fault stops ~1.5 elements above the sole: the
standard **gap convention** for zero-thickness junctions. The weak region
is the facet support (one cell each side of each surface).

```{figure} figures/listric_lines.png
:width: 90%

The same geometry as zero-thickness surfaces with gaps at the sole. Each
fault localises strain individually (bottom), but the velocity field
(middle) shows only weak block individuation and the detachment is nearly
silent: the gaps break the kinematic linkage, and slip cannot transfer
from the faults into the sole through 1.5 elements of strong rock.
```

## The comparison, and when to use which

| | ribbons (`place_thin_volume`) | zero-thickness (`place_along_lines`) |
|---|---|---|
| junctions | in the volume — the rheology decides | gap convention (or explicit junction machinery) |
| fault width | a REAL parameter (sub-$h$ supported); $V = 2\dot\varepsilon\, w$ makes it constitutive | tracks the mesh (facet support ≈ one element each side) |
| system connectivity | native — the detachment engages | broken at the gaps unless closed by other means |
| sharp velocity jump | smeared over $w$ | exact at the facet chain |
| this example | reproduces the linked block tectonics | under-connects the system |

For a junction-dominated structure — which a rift with a common detachment
is — the **connectivity of the weak network controls the tectonics**, and
the ribbon representation carries connectivity natively. Zero-thickness
surfaces remain the right tool where the interface itself is the object of
study (contact, exact slip, dynamic topography from a discrete fault), and
the two compose: both live in the same placement family, on the same
meshes, with the same lifecycle (add, remove, re-place — serial or
parallel).

Both representations also support faults that **outcrop at the surface**
in 3-D (specify the surface a little long; prep clips it and remeshes the
cap conformingly). In 2-D the ribbons here stop just below the top — a
recorded limitation, not a design one.
