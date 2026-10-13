# Gouge zones

A fault with a gouge zone can be built either way. As a **ribbon** it is a
band of crushed rock of width $w$ with its own material viscosity
$\eta_1$. As a **contact** it is a surface with an interface law
$\eta_I$, and the two agree when

$$
\eta_I = \frac{\eta_1}{w},
$$

which is the matched collapse the campaign established and this page
re-checks rather than assumes.

The collapse is exact for the slip. What it is not exact for is anything
that depends on the state *inside* the zone, and the reason is visible in
the formula: it preserves the **ratio** $\eta_1/w$ and destroys the pair.
Any quantity that needs $\eta_1$ and $w$ separately is gone.

Scripts: `~/+Simulations/ribbon_network_2d/`.

## The premise, checked

One straight normal fault in an extending unit box, both ways, at four
widths with the gouge viscosity held at $\eta_1 = 10^{-3}$ so the
interface law changes with the width:

| $w$ | $\eta_I = \eta_1/w$ | ribbon peak slip | contact peak slip | ratio |
|---|---|---|---|---|
| 0.020 | 0.05 | 0.50255 | 0.44947 | 111.8% |
| 0.010 | 0.10 | 0.48200 | 0.45374 | 106.2% |
| 0.005 | 0.20 | 0.46351 | 0.44800 | 103.5% |
| 0.002 | 0.50 | 0.42800 | 0.42101 | **101.7%** |

```{figure} figures/gouge_mechanics.png
:width: 100%

Left: slip along the fault, ribbon (solid) against the matched contact
(dashed), at each width. Right: the two agree in the limit and not
before it.
```

The collapse is a $w \to 0$ limit and behaves like one: it is worth 12% at
$w = 0.02$ and 1.7% at $w = 0.002$. Quote it as an identity only when the
zone is thin against the fault length.

```{note}
The contact runs through the rotated machinery, which builds its **own**
prefixed KSP that `stokes.petsc_options` never reaches. Left on its
iterative default the velocity sub-solve here hit its 200-iteration cap
and warned, while the outer SNES still reported convergence. Set
`stokes._rotated_use_lu = True` on problems this size.

The verdict then has to be physical, because the outer SNES reason does
not describe that solver either. Use the constraint: the no-opening
condition is strong, so the **normal** part of the pair jump is machine
zero when the increment was really solved. It reads $5.5\times10^{-17}$
against a tangential jump of 0.47 here.
```

## What the collapse loses

Dissipation per unit **length** of fault is $\eta_I V^2$ — preserved. The
far-field temperature is therefore the same either way. Dissipation per
unit **volume** in the gouge is

$$
\Phi = \eta_1 \left(\frac{V}{w}\right)^2 = \frac{\eta_I V^2}{w},
$$

which is not preserved: it rises as the gouge narrows at fixed interface
law. Feed that into conduction across a thin band with its edge
temperatures as boundary data and the excess at the centre is

$$
\Delta T = \frac{\Phi\, w^2}{8\kappa} = \frac{\eta_1 V^2}{8\kappa}
         = \frac{\eta_I V^2}{8\kappa}\, w .
$$

At fixed $\eta_I$ that is **proportional to the width**. Two models with
the same interface law and different widths have the same slip and the
same far field, and gouge temperatures in the ratio of their widths. For
a contact it is zero at every width, because a contact has no width.

Peak gouge temperature is what controls thermal weakening and frictional
melting, so this is not a diagnostic nicety.

```{figure} figures/gouge_thermal.png
:width: 100%

Left: the gouge's own temperature excess, measured as the departure from
the straight line through its two edges — the part a contact leaves out.
Right: measured against predicted, with nothing fitted. $\eta_1$ and
$\kappa$ are set, $V$ and $\Delta T$ are measured independently.
```

| $w$ | $V$ measured | $\eta_1V^2/8\kappa$ | $\Delta T$ measured | agreement |
|---|---|---|---|---|
| 0.020 | 0.50255 | 0.003157 | 0.004119 | 130.5% |
| 0.010 | 0.48200 | 0.002904 | 0.002945 | **101.4%** |
| 0.005 | 0.46351 | 0.002686 | 0.002715 | **101.1%** |
| 0.002 | 0.42800 | 0.002290 | 0.002300 | **100.4%** |

The $w = 0.02$ row is the thin-band reduction failing where $w/L = 0.033$
is not small; the exact 1-D quadrature on the measured dissipation
profile converges the same way (58.1%, 78.6%, 92.3%, 98.1%).

```{warning}
Do not take the dissipation profile through `uw.function.evaluate`. It
L2-projects any derivative composite, and viscous dissipation is one, so
the smoothing lands exactly on the across-band structure being measured.
Project to a discontinuous degree-0 variable instead — on that space the
L2 projection is the cell average and nothing crosses a cell boundary.

The first attempt at an oracle here also failed for a physical reason
worth recording: treating the band as a uniformly heated slab against a
uniform *background*, and subtracting a far-field $\Phi$, misses by a
factor of six and gets the sign wrong. There is no background to
subtract — the wall rock enters through the measured edge temperatures,
and the source inside the band is the gouge's own dissipation alone.
```

## A split mesh is not a home for a second equation

The other failure is structural rather than quantitative, and it does not
involve the mechanics at all. Solve plain conduction between a hot wall
and a cold wall — uniform diffusivity, no fault properties anywhere — on
a plain mesh and on the split-node mesh:

```{figure} figures/gouge_insulation.png
:width: 100%

The same problem, twice. A split-node mesh is split for the **mechanics**;
heat does not jump across a fault, but the doubled nodes let it.
```

| | flux through the cold wall | $T$ across the fault line |
|---|---|---|
| plain mesh | −0.010000 (analytic $-\kappa\Delta T/L$ = −0.01) | +0.008660, the smooth ramp over the probe offset |
| split mesh | −0.008091, 19% of the heat stopped | **+0.471241**, a real discontinuity |

The plain row reproduces the analytic flux exactly, so the split row is
the mesh and not the solver. The fault behaves as a perfect insulator,
and no choice of fault parameter changes that — there is no thermal
interface condition on a split node to set.

That is the sharpest form of the argument for the ribbon. Where a second
equation has to see the fault, the finite-width zone gives it cells, and
everything is continuous across them.

## Choosing

- **Ribbon** when the width is physics: a gouge with its own rheology, a
  damage zone another equation has to see, anything where a quantity
  *inside* the zone is wanted. It costs DOFs as $\mathrm{length}/w$.
- **Contact** when the fault is genuinely a surface at the scale of the
  problem and slip on it is the quantity of interest. It costs DOFs as
  $\mathrm{length}$, with no $w$ in it at all.

## See also

- [Crossing fault zones](crossing-fault-zones.md) — networks, where the
  ribbon's other advantage lives.
- [Split-node faults](split-node-faults.md) — the contact representation.
- [Fault networks](fault-networks.md) — the zero-thickness toolkit.
