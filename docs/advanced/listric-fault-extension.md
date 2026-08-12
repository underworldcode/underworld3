# Listric faults over a common detachment

The classic geometry of a continental rift: listric normal faults — steep
near the surface, flattening with depth — soling into a **common
detachment** in weak rock, the hanging-wall blocks stepping down between
them. This example builds the system as **finite-width fault ribbons**
(`place_thin_volume`): one call places the whole network, the
fault–detachment junctions are resolved in the CAD stage and become
ordinary cells of the union, and the zone's cells carry the weak rheology
directly.

Scripts: `~/+Simulations/listric_extension/`.

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

Two faults and a horizontal detachment spanning beneath both, soled a
fraction of the ribbon width into it so the union is clean:

```python
from underworld3.utilities.place_surface import place_thin_volume

dm, info = place_thin_volume(mesh0.dm, [sole, f1, f2], width=0.02,
                             label="Zone", label_value=31)
```

The width is a **real parameter** — constitutive, in the finite-width
formulation ($V = 2\dot\varepsilon\, w$) — not something that tracks the
mesh. The instantaneous Stokes problem is plain extension: sides pulled at
±0.5, free-slip base, free top, viscosity 1 in the country rock and
$10^{-3}$ in the zone, assigned **per cell** off the zone label (P0,
discontinuous — interpolating a nodal viscosity across a sharp interface
smears exactly what the conforming zone resolves).

```{figure} figures/listric_ribbon.png
:width: 90%

Weak zones (top), horizontal velocity (middle), strain rate (bottom). The
hanging-wall blocks move as coherent units and the detachment is ACTIVE
between the two junctions — the textbook rift cartoon, reproduced. Note
the listric–sole junctions are *glancing* intersections, so the layer
mesh's minimum angle there equals the local intersection angle (≈5° in
this geometry); the solve is untroubled, and steepening the trace's
terminal dip is the lever if a stiffer rheology objects.
```

## The comparison to come

The other fault representation — the **zero-thickness fault**, meaning the
*split mesh*: duplicated nodes along the surface, a true velocity
discontinuity, contact conditions on the interface — is the interesting
counterpart for this geometry, and the comparison arrives with the
fault-contact machinery. The expectation to test: the two should agree
closely on the block kinematics, with the differences carried by the
**junction treatment** — the ribbon network is through-going by
construction, while split surfaces meet the sole through an explicit
glue / through-going choice.

(A third construction — a conforming *weak band*, the placed surface with
its facet-support cells weakened — exists and is sometimes convenient, but
its width tracks the mesh and its junctions under-connect; it is not a
useful model of this system and is not shown.)
