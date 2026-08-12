# Listric faults over a common detachment

The classic geometry of a continental rift: listric normal faults — steep
near the surface, flattening with depth — soling into a **common
detachment** in weak rock, the hanging-wall blocks stepping down between
them.

Underworld3 can build this system two ways, and this page builds it both
ways from the *same traces, the same base mesh and the same drive* so the
two can be compared directly:

- **finite-width ribbons** ([`place_thin_volume`](conforming-surfaces-and-fault-zones.md))
  — each fault is a thin band of weak material, of a width you choose;
- **zero-thickness faults** ([split-node](split-node-faults.md), assembled
  by [`FaultNetwork`](fault-networks.md)) — each fault is a surface across
  which the velocity jumps, with no width at all.

Scripts: `~/+Simulations/listric_extension/`.

## The geometry

A listric trace flattening toward the detachment depth $y_d$, then a
straight tail continuing that dip *through* the detachment:

```python
DETACH_Y, LAM, SOLE_DIP = 0.4, 0.18, 0.066

def listric_trace(x0, n=30):
    y = np.linspace(1.0 - TOP_GAP, DETACH_Y + SOLE_DIP, n)
    x = x0 + LAM * np.log((1.0 - DETACH_Y) / (y - DETACH_Y))
    ...                                  # straight tail across the sole
```

Both representations consume this same polyline. Neither reaches the free
surface: 2-D faults cannot daylight in either formulation yet, so the tips
stop a little below the top.

The drive is plain instantaneous extension — sides pulled at $\pm 0.5$,
free-slip base, free top — and the country rock has viscosity 1
throughout.

## Building it both ways

**Ribbons.** One call places the whole network. The junctions are
resolved in the CAD stage and become ordinary cells of the union, so the
fault runs *into* the detachment as continuous weak material:

```python
from underworld3.utilities.place_surface import place_thin_volume

dm, info = place_thin_volume(mesh0.dm, [sole, f1, f2], width=0.02,
                             label="Zone", label_value=31)
```

The width is a **real parameter** — constitutive in this formulation
($V = 2\dot\varepsilon\,w$) — not something that tracks the mesh. The
zone's cells carry $\eta = 10^{-3}$, assigned per cell off the label (P0,
discontinuous: interpolating a nodal viscosity across a sharp interface
smears exactly what the conforming zone resolves).

**Split nodes.** The split formulation refuses shared vertices at a
junction, so `FaultNetwork` converts each T abutment to the *offset*
form — the junior trace is pulled back and a small damage plug bridges
the gap:

```python
net = uw.meshing.FaultNetwork(traces(), hierarchy=["Sole", "F1", "F2"])
net.prepare(h=CELL, ligament=1.0)
net.mesh = base_mesh().add_fault(net.prepared)   # same base as the ribbons
...
stokes.constitutive_model.Parameters.yield_stress = net.damage_yield(v, dial=0.05)
net.apply_contact(stokes, conds=ETA_F)
```

The interface law is **matched to the band**: a shear band of viscosity
$\eta_{\rm band}$ and width $w$ has zero-thickness limit
$\eta_f = \eta_{\rm band}/w = 10^{-3}/0.02 = 0.05$.

```{figure} figures/listric_construction.png
:width: 100%

The F1/detachment junction, at the same scale. **Left**: the ribbon union
is through-going — the fault arrives as weak cells and the detachment
continues through them. **Right**: the split mesh, with the prepared trace
(dark) stopping short of where the raw trace would have met the sole (thin
red), and the damage plug (peach) bridging the gap.
```

The pull-back is the thing to watch here. It is quoted as a *ligament* in
multiples of $h$, but that is a clearance measured **across** the senior
fault, so the retreat **along** the junior trace is $\ell h/\sin\theta$ at
an intersection angle $\theta$. A listric fault sole is a glancing
junction by definition — $\theta \approx 20°$ here — so one element of
clearance costs $0.10$ of trace, three times what the same setting costs
at an orthogonal crossing. The ribbon union pays nothing, but its
junction cells are correspondingly slivered (minimum angle $0.9°$; the
solve is untroubled, and steepening the terminal dip is the lever if a
stiffer rheology objects).

## What the two give

```{figure} figures/listric_compare.png
:width: 100%

The same system, both representations. Horizontal velocity (top), vertical
velocity (middle), strain rate (bottom, log scale).
```

The **kinematics agree**. Both build the same two hanging-wall blocks with
the same sharp velocity contrast across each fault, and the vertical
velocity — the subsidence pattern, the geologically meaningful signal —
matches in shape and in magnitude ($-0.648$ against $-0.635$ peak).

The **strain rate does not, and cannot**. This is the clearest statement
of what separates the two formulations: the ribbon spends the
deformation as strain *inside* a band, so the fault is a bright line in
$\dot\varepsilon_{II}$; the split mesh spends it as a *jump*, which no
strain measure can see. The only thing lit up on the split panel is the
junction glue. Neither picture is more correct — they are different
currencies, and a comparison has to be made in a quantity both can pay.

```{figure} figures/listric_slip.png
:width: 100%

Slip on each fault, measured in each representation's own way: the
velocity difference across the band edges (ribbon) against the contact
pair jump (split node). The split curves stop short at the detachment —
that is the ligament.
```

Peak slip:

| fault | ribbon | split node | ratio |
|-------|--------|------------|-------|
| F1    | 0.498  | 0.401      | 0.80  |
| F2    | 0.417  | 0.390      | 0.93  |
| Sole  | 0.317  | 0.154      | 0.49  |

On the listric faults the two agree to within 7–20% and the profile
*shapes* are the same. On the detachment the split model slips half as
much.

## The junction is the difference

That halving is not the interface law, and it is worth being explicit
about how we know. Two controls:

- **Remove the law entirely** ($\eta_f = 0$, a frictionless fault): peak
  slip moves by 2–4% (sole $0.154 \to 0.161$). The matched law is a
  small correction, as it should be when both representations are already
  weak.
- **Double the ligament** ($1h \to 2h$): the sole loses 16%
  ($0.154 \to 0.130$) and F1 loses 12%.

So what the detachment feels is the **gap**, and on a glancing junction
the gap is large. This is the practical rule for detachment systems: the
junction connectivity controls how much slip transfers, and in the offset
form you buy fidelity with elements, not with physical size — make $h$
smaller and the ligament follows it down.

## Which to use

- **Ribbons** when the width is physics — a gouge zone with its own
  rheology, a damage zone that another equation (porous flow, thermal)
  has to see, or a network whose junctions must be through-going without
  further thought.
- **Split nodes** when slip is the quantity of interest — earthquake
  cycles, stress transfer, frictional and rate-and-state laws — and when
  conditioning matters: there is no thin feature and no viscosity
  contrast for the Schur complement to fight.

Junctions are the place to think hardest. The ribbon's are free but
slivered; the split's are clean but offset, and on glancing intersections
that offset is what sets the answer.

```{note}
The split solves here report an inner velocity-block KSP reaching its
iteration cap. The answers were checked against the direct-LU route
(`solver._rotated_use_lu`) and agree to 0.2%, and are unchanged by a
1000x tighter tolerance — the outer Krylov is flexible, so the warning
reflects preconditioner quality on this mesh, not the solution.
```
