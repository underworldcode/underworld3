# Branching faults, junctions, and what a crack model cannot do

The [crossing-zones page](crossing-fault-zones.md) compared overlap
rheologies for fused fault zones in pure extension and found the
differences modest at ordinary angles. That test had a hidden weakness:
the configuration is maximally symmetric about the load, and the drive
carries **no rotation**. This page repeats the question with both
removed — and the verdict inverts. It ends at the reason: two of our
fault representations are crack models, one is not, and the difference
is not a numerical detail but the mechanics of branching itself.

Scripts and caches: `~/+Simulations/ribbon_network_2d/`
(`branch_study.py`, `branch_width.py`, `branch_damage.py`).

## The shear box

Simple shear supplies what extension cannot: spin, $\omega =
\dot\gamma/2$. The test frame is a driven box — top plate $v = (+0.5,
0)$, bottom $v = (-0.5, 0)$, sides open (traction-free) — holding an
asymmetric Y: a **sub-horizontal main fault** at $\theta = 20/30/40^\circ$
and a **horizontal branch** running in from the side and fusing into it.
Nothing is mirror-symmetric; the branch lies in the shear plane
(optimally oriented) while the main degrades with $\theta$ (resolved
drive $\propto \cos 2\theta$). Bands at the union-jack default
resolution, structure verified per build; the isotropic weak band is the
reference throughout, with transversely isotropic (TI) variants against
it. Everything is measured from the P0 cell strain rate — local, no
recovery step, no probes.

```{figure} figures/branch_maps_z20.png
:width: 100%

Strain rate (P0 invariant, log scale, one range). Rows: $\theta$ =
20/30/40$^\circ$; columns: the isotropic reference and three TI
variants differing only at the junction. In the reference the junction
is the hottest point of the network and intensifies with $\theta$; in
every TI variant it cools, and the misoriented main fades until its
lower arm is a ghost.
```

## The junction is a slip exchanger — and the crack constraint cannot run it

Slip on each fault, at the junction, TI variants as a fraction of the
reference:

| $\theta$ | main fault | branch |
|---|---|---|
| $20^\circ$ | 77% | ~75% |
| $30^\circ$ | **46%** | ~78% |
| $40^\circ$ | **19%** | ~52% |

Three facts organise this. First, the reference's main-fault slip at the
junction **rises** as the fault becomes misoriented (0.30 → 0.36): the
optimally-oriented branch delivers its offset *into* the main through
the junction, the isotropic corner rotating the deformation between the
two orientations. Second, the three TI overlap treatments (nearest
director, harmonic average, isotropic overlap cells) are
indistinguishable from each other while all collapsing against the
reference — the overlap rule was never the dominant term. Third, the
mechanism is general: **a gradient of slip along a fault is a
volumetric exchange between band and surroundings**, and that is
precisely the deformation mode the TI constraint (like the slip
surface) removes. Crack kinematics carry slip *along* a fault
faithfully and cannot hand it *between* faults.

Imposed isotropic "damage halos" around the junction recover the
exchange roughly linearly in their radius (42% at $3w$, 73% at $5w$)
with no plateau: there is no compact junction patch, because the
constraint taxes the mechanics wherever slip varies.

```{figure} figures/halo_recovery.png
:width: 100%

Slip profiles and recovery against halo radius: freeing the constraint
where the slip gradient lives, by hand.
```

## The exit throat, and a law with nothing fitted

Where is the reference's peak? Not at the junction point. It sits on
the main's up-dip arm at $r \approx 2.1\,w$ — a **self-similar**
position (2.05–2.11 $w$ across a fourfold width sweep) — where the
fused Y narrows back to a single band. The combined slip of both faults
must exit through that neck, and flux conservation predicts its
intensity outright:

$$
\gamma_{\rm peak} \;\approx\; \frac{s_{\rm main} + s_{\rm branch}}{w},
$$

measured at 0.96–1.09 of the prediction over the three resolved widths.

```{figure} figures/junction_peak_zoom.png
:width: 100%

The peak (circled) at four band widths, one log scale: always at the
exit throat, always at $r \approx 2.1\,w$, in healthy cells.
```

The limit is the point: as $w \to 0$ the throat slides onto the corner
and its amplitude diverges. **A branch point is a corner singularity of
the continuum problem, and the band width is its regularisation
length.** Neither description converges there — the crack models
*exclude* the corner physics; the continuum model *concentrates* it
without bound. The junction's width is therefore not a numerical
parameter to refine away but a physical one: the process-zone size. (It
is the [gouge page's](gouge-zones.md) "width is physics", at the place
where it matters most.)

## What the country rock feels

Deviatoric stress in the matrix (where $\tau = 2\dot\varepsilon$
exactly), zones masked:

```{figure} figures/branch_stress_matrix.png
:width: 100%

Matrix stress, one linear scale. Means agree to 2–4% across all six
cases — the load is conserved; its **location** is not.
```

The locked TI junctions park a wall-rock stress concentration at the
elbow (the tail rises 11%) that the reference does not possess; freeing
the junction trades it back for **tip lobes**, which is where the
reference — and a real fault — concentrates stress. Ranking these
models by peak stress alone would order them backwards; the location is
the diagnostic. A constrained-network model read at face value predicts
wall-rock damage and seismicity at junctions where the physics puts
none, and under-predicts tip process zones.

## Offered plasticity, the crack model builds a bypass

A deliberately primitive damage law — any matrix cell whose stress
exceeds $\tau_c$ becomes isotropic $\eta_1$, instantly and irreversibly;
re-solve; repeat — asks what a yielding background does with each
representation. With a negative control:

```{figure} figures/damage_control_compare.png
:width: 100%

Damaged cells coloured by the pass at which they yielded, same law and
threshold. Left, the isotropic reference: growth at the free **tips**
only — the physical pattern — with the junction silent throughout.
Right, the fully-TI network: the same tip growth **plus** a
junction-nucleated front marching along the shear plane — a
through-going bypass that retires the misoriented fault.
```

Both models grow faults at tips; that part is physics. Only the
constrained model manufactures growth at the junction — nucleated by
the stress its own kinematic restriction parks there — and its response
to that stress is not to heal the junction but to **re-route the
network around it**. The law is a cartoon (no softening, no length
scale, grid-width bands), so the nucleation sites and topology are the
trustworthy content, not rates or widths. One further consequence of
the corner singularity: at realistic (thinner) widths the *reference's*
corner stress also exceeds any finite yield threshold — so in a
yielding crust a branch point **must** develop a damage zone. The
damage halo at a junction is not a modelling patch; it is the
material's own regularisation of a corner that crack mechanics cannot
represent and continuum mechanics cannot bound.

## Choosing, finally

One of these is a crack model twice over — the split surface imposes
slip-only kinematics exactly; the TI band imposes the same as a
$10^{3}$ constitutive penalty — and one is not: the isotropic band is a
model of fault *rock*, not fault *kinematics*. The crack assumptions
fight each other at branches. Hence:

- **Large-scale fault networks**: build from surfaces (contacts) — the
  crack idealisation at its best, where segments are distinct,
  well-oriented, and slip-dominated.
- **Zones with interior physics** (gouge, heating, damage evolution):
  thin volumes/ribbons — with TI where the zone should stay
  crack-like, isotropic where it must evolve internally.
- **Crossings and joins**: make them kinematically compatible by
  construction — bend faults in to meet tangentially rather than
  abutting at high angle, and surround the joins with isotropic damage
  zones sized as process zones. Do not expect any director rule, patch,
  or refinement to buy back a high-angle locked junction: the failure
  is the crack assumption itself.
- **Evolution in time** — where a misoriented fault would dismember
  into en-échelon segments and the network would rebuild itself through
  its damage field — requires transporting faults as material chains
  and re-deriving them with the evolving damage. That is a programme,
  not a feature; this page records the mechanics that motivates it.
