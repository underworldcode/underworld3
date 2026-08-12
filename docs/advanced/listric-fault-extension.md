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

## Chasing the detachment deficit

The obvious suspect is the junction, and it is a large part of the
answer — but not all of it, and the way that came out is worth showing,
because the tempting conclusion is the wrong one. Every knob was turned,
one at a time, reading peak slip on the sole:

| control | sole slip | effect |
|---------|-----------|--------|
| ribbon reference | 0.317 | — |
| split baseline ($h$ = 0.035, ligament 1$h$, dial 0.05) | 0.154 | — |
| interface law removed ($\eta_f = 0$) | 0.161 | +5% |
| ligament $1h \to 2h$ | 0.130 | −16% |
| refine $h$: 0.035 → 0.0233 → 0.0175 | 0.154 → 0.181 → 0.187 | +21%, converging |
| glue dial 0.05 → 0.01 → 0.002 | 0.181 → 0.204 → 0.210 | +16%, saturating |
| glue plug made *larger* at fixed strength | 0.181 → 0.167 | −8% |
| best case (finest $h$, dial at the ceiling) | **0.216** | 68% of the ribbon |

Reading down that table:

- **It is not the law.** Removing the interface condition entirely moves
  slip by 5%. Both representations are already weak, as intended.
- **The junction gap is the biggest single lever.** Doubling the ligament
  costs 16%; refining the mesh — which shrinks the gap, since the
  ligament is measured in $h$ — buys 21% back. This is the practical
  rule: buy fidelity with elements, not with physical size.
- **But closing the gap does not close the difference.** The refinement
  sequence is converging to ≈0.19, not to the ribbon's 0.317, and the
  ribbon itself is essentially $h$-independent over the same range
  (0.3173 → 0.3155) — as it should be, since its width is a physical
  parameter, which makes it a clean control.
- **The glue helps, and then stops helping.** Weakening the plug raises
  transfer to a ceiling near 0.21 — the inviscid-plug limit the network
  documentation describes. Making the plug *bigger* at the same strength
  makes things *worse*: a larger weak region lets the corner deform
  instead of passing slip along the sole. A junction plug is a hinge, not
  a bridge.

Two explanations we tested and discarded, in case they occur to you too.
The ribbon's band, being weak and sub-horizontal in horizontal extension,
might simply be stretching along its own length, with our probe reading
that as slip — but shrinking $w$ by 4$\times$ at fixed
$\eta_f = \eta_{\rm band}/w$ moves the sole by only 4.5% (0.317 → 0.303),
so it is not that. (That the ribbon is nearly $w$-independent at fixed
$\eta_f$ is a good result on its own: it says $\eta_f$ really is the
similarity parameter linking the two formulations.) And the plug does
shrink away under refinement — but as the table shows, a bigger plug
transmits less, not more.

So a residual of roughly 30% survives every control we have. The best the
offset form achieves here is 68% of the through-going detachment slip,
and no single parameter closes the rest.

## What that means for choosing

The two formulations agree on what a fault *is* — the block kinematics,
the subsidence, the slip on the individual listric faults — and disagree
on what a *junction* is, in a way that is structural rather than
mistuned. A through-going weak band and a network of zero-thickness
contacts joined at offset junctions are not the same mechanical object
where faults meet, and on this geometry the sole — the one fault that has
to *receive* slip through two junctions — is where the disagreement
shows. The listric faults themselves reach 86–97% of the ribbon's slip at
the glue ceiling; the sole reaches 68%.

- **Ribbons** when the width is physics — a gouge zone with its own
  rheology, a damage zone another equation has to see — and, on this
  evidence, when the system is **junction-dominated**: a detachment fed
  by splays is exactly that, and the union carries the connectivity
  natively.
- **Split nodes** when slip on individual faults is the quantity of
  interest — earthquake cycles, stress transfer, frictional and
  rate-and-state laws — and when conditioning matters: there is no thin
  feature and no viscosity contrast for the Schur complement to fight.

```{warning}
The residual is not explained. If you are modelling a detachment system
in the offset form, calibrate the transfer rather than assuming it: the
numbers above are for one geometry at one intersection angle, and the
glancing junction is the hard case.
```

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
slivered; the split's are clean but offset, and on a glancing
intersection that offset is expensive and only partly recoverable.

```{note}
The split solves here report an inner velocity-block KSP reaching its
iteration cap. The answers were checked against the direct-LU route
(`solver._rotated_use_lu`) and agree to 0.2%, and are unchanged by a
1000x tighter tolerance — the outer Krylov is flexible, so the warning
reflects preconditioner quality on this mesh, not the solution.
```
