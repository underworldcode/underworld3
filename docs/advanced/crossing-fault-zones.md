# Fault zones that cross

Two fault zones that cross are the geometry the finite-width
representation exists for. Thickening each trace and letting the CAD
kernel fuse the results turns the junction into ordinary cells of a
single region: nothing downstream is told which branch a cell came from,
and no junction geometry is ever constructed. The zero-thickness
representation cannot take this route at all — the split refuses
overlapping cuts.

That ignorance is deliberate, and it has a price. The rheology at the
overlap comes entirely from the `Surface` lookup, which gives every point
the **director of the nearest surface**. Where the two zones cross, the
material is weak in two directions and a single director represents one
of them. This page works out when that matters.

Scripts: `~/+Simulations/ribbon_network_2d/`.

## Building the crossing

One call places the whole network. The two ribbons may overlap — that is
the point — and `assembly="fuse"` returns their union as one region with
no internal seam:

```python
from underworld3.utilities.place_surface import place_thin_volume

dm, info = place_thin_volume(mesh0.dm, [trace_a, trace_b], width=0.02,
                             label="Zone", label_value=31,
                             size=0.005, assembly="fuse")
```

Two details in that call decide whether the experiment can measure
anything.

**`size` must be passed.** It defaults to `0.9 * width`, which puts about
two elements across the band at every width. Two elements cannot resolve
which half of an overlap a rule owns, and the ownership split is the
whole question. `width/4` is used throughout here.

**`fragment` is the wrong assembly for a crossing.** It keeps each
overlap piece as its own region, so the mesh conforms to the boundaries
of the overlap; where two zones converge at a shallow angle the overlap
is a spike, and its fragmented tip meshes to arbitrarily bad angles. The
fused union has no such tip.

## What the overlap should be

Before solving anything, the question has a closed-form answer in 2-D,
and the closed form says where to look.

In plane strain with an incompressible medium the deviatoric strain-rate
space is **two-dimensional**, spanned by

$$
\mathbf{e}_1 = \tfrac{1}{\sqrt 2}\begin{pmatrix} 1 & 0\\ 0 & -1\end{pmatrix},
\qquad
\mathbf{e}_2 = \tfrac{1}{\sqrt 2}\begin{pmatrix} 0 & 1\\ 1 & 0\end{pmatrix},
$$

and rotating the material by $\theta$ rotates a state in that space by
$2\theta$. A weak plane with normal at $\phi$ is compliant in exactly one
mode — on-plane shear — so it occupies a single **line** in that space,
at angle $2\phi + 90^\circ$.

Everything follows from the doubling. Two planes separated by $\Delta$ in
the rock are separated by $2\Delta$ there, so:

- $\Delta = 0^\circ$ and $\Delta = 90^\circ$ put both compliances on the
  **same line**. Shear on a plane with normal $\hat{\mathbf x}$ *is*
  shear on a plane with normal $\hat{\mathbf y}$ — the same $\tau_{xy}$.
  A 90-degree X is degenerate in 2-D.
- $\Delta = 45^\circ$ makes the two compliances **orthogonal**. Together
  they span the whole deviatoric space, and the overlap is isotropic.

So the geometry the phrase "high angle" suggests — the square X — is the
one geometry where the question cannot be asked. The tensors to compare
differ most at 45 degrees.

```{warning}
**This degeneracy is two-dimensional and does not carry to 3-D.** It
exists because the deviatoric space here has only two dimensions, so two
lines in it must either coincide or be distinct in one way. In 3-D that
space has five dimensions, two weak planes at any angle occupy different
subspaces, and there is no null. Do not carry "a 90-degree crossing is
safe" into a 3-D model — the 3-D question is open and nothing on this
page answers it.
```

### The orthotropic tensor is a TI tensor

Summing the two planes' compliances and inverting gives a tensor with two
eigenvalues and one principal direction in the deviatoric plane, which is
precisely the structure of
{py:class}`~underworld3.constitutive_models.TransverseIsotropicFlowModel`.
So the orthotropic treatment needs no new constitutive machinery in 2-D:
read off an effective director and effective $\eta_0$, $\eta_1$ and use
the shipped model.

```python
S = np.eye(2) / (2 * eta_0)
s = 1 / (2 * eta_1) - 1 / (2 * eta_0)
for phi, lam in zip(normal_angles, weights):
    u = weak_mode(phi)                    # unit vector at 2*phi + 90
    S = S + lam * s * np.outer(u, u)
C = np.linalg.inv(S)
eta_1_eff, eta_0_eff = sorted(np.linalg.eigvalsh(C) / 2)
```

### Two candidate rules, not one

The weights are a modelling choice and the two ends of the family behave
differently:

| $\Delta$ | compliance **sum** ($\lambda = 1$ each) | compliance **average** ($\lambda = \tfrac12$ each) |
|---|---|---|
| $0^\circ$ | anisotropy 199 — twice as weak as one plane | exactly the single plane |
| $45^\circ$ | isotropic, $\eta_1$ | isotropic, $2\eta_1$ |
| $90^\circ$ | anisotropy 199 | exactly the single plane |

The sum is what "add the compliances" means literally: two independent
fabrics superposed. Read it at $\Delta = 0$ — two *coincident* weak
planes come out twice as weak as one — and note that the excess does not
switch off away from coincidence. The average treats the overlap as a
50/50 mixture of the two fabrics and degenerates to the single plane
exactly. Both are carried below; neither is obviously right.

```{important}
There is **no external reference solution** here, and there cannot be one
without modelling the fabric itself. Both constructions are assumptions,
so what is measured below is the distance from the shipped rule to a
*choice*, not to a truth. What makes that worth reading is that the two
choices differ by a factor of two in how weak they make the overlap and
still bracket the answer tightly — 61.8% against 70.0% at the worst
angle, 92.3% against 93.0% at $45^\circ$. The conclusion is insensitive
to the part that had to be assumed.
```

## The experiment

Two straight ribbons crossing at the centre of a unit box, fused into one
zone, driven in horizontal extension with a free top. The crossing angle
$\Delta$ is swept, and five rheologies are solved **on the same mesh**:

| case | the zone | the overlap |
|---|---|---|
| `none` | uniform $\eta_0$ | — (the no-fault control) |
| `isotropic` | weak in every mode | weak in every mode |
| `nearest` | TI, nearest trace's director | TI, nearest trace's director |
| `ortho_sum` | TI, nearest trace's director | the compliance-sum tensor |
| `ortho_avg` | TI, nearest trace's director | the compliance-average tensor |

The last three are identical everywhere except the overlap cells, so any
difference between them is the overlap rheology and nothing else.
`isotropic` weakens the arms too, so it is not a candidate rule — only a
bracket on how compliant the network could be.

The metric is the **dissipation weakening**

$$
W = 1 - \frac{D}{D_{\rm no\ fault}},
$$

referenced to the `none` control on the same mesh at the same angle. It
is a single number with no probe geometry in it. (The control dissipates
exactly 4.000000 in this box, which is the analytic value, so the
integral is checked as well.)

Note what $W$ is made of at a shallow crossing: at $\Delta = 10^\circ$
the two dissipations are 3.9521 and 3.9225 against a control of 4.0000,
so the whole signal is a difference of about 0.03 between numbers that
agree to 1%. The exact control and the resolution study below are what
make that difference readable; a single run at one resolution would not
be.

### What comes out

| $\Delta$ | $W$ nearest | $W$ orthotropic | $W$ isotropic | nearest, as % of orthotropic |
|---|---|---|---|---|
| $10^\circ$ | 1.198% | 1.710% | 3.628% | **70.0%** |
| $20^\circ$ | 3.687% | 4.520% | 6.256% | 81.6% |
| $30^\circ$ | 6.618% | 7.679% | 9.240% | 86.2% |
| $45^\circ$ | 11.566% | 12.435% | 13.646% | 93.0% |
| $60^\circ$ | 15.789% | 16.253% | 17.110% | 97.1% |
| $75^\circ$ | 18.322% | 18.476% | 19.121% | 99.2% |
| $90^\circ$ | 18.918% | 18.918% | 19.465% | **100.0%** |

```{figure} figures/crossing_weakening.png
:width: 100%

Left: what the two faults are worth, against the no-fault control on the
same mesh. Right: the overlap rule on its own — the two runs share the
mesh and the arms exactly, so the ratio is the overlap and nothing else.
```

Read the last row first. At $\Delta = 90^\circ$ the two dissipations
agree to seven significant figures (3.243286 against 3.243286). That is
not a tuned result: the algebra says the orthotropic overlap tensor there
is *identically* the single plane's, and a 2-D director is only defined
mod $90^\circ$, so the nearest-fault rule assigns that same tensor on
both sides of the medial axis. The experiment reproduces it exactly, and
that is what licenses the other six rows.

Now read up the column. The nearest-fault rule loses a substantial part
of the network's weakening at a shallow crossing, and the error closes
monotonically as the crossing squares up. The percentages in that table
are computed at a zone mesh of $w/4$ and are **bounds**, not estimates —
refining the zone mesh moves the rule closer to the orthotropic one, and
the converged values are about 76% at $10^\circ$ and 95% at $45^\circ$
(see the resolution control below).

```{figure} figures/crossing_map.png
:width: 100%

The same three answers in physical space at $\Delta = 45^\circ$, on one
colour scale. They are indistinguishable by eye, and the region in
dispute is 34 cells out of 1480 (right). A difference of this kind is
only visible in an integral — which is why the metric above is a
dissipation and not a picture.
```

### The error is largest where the issue expected none

The per-cell tensors differ most at $\Delta = 45^\circ$, and the measured
curve shows no feature there at all. Both facts are right, and the
resolution is geometric: the overlap **area** goes as $w^2/\sin\Delta$.
A shallow crossing has a long lens of shared material, a square one has
almost none — 141 overlap cells at $10^\circ$ against 24 at $90^\circ$.
The tensor error peaks at $45^\circ$, the area falls monotonically, and
the product is dominated by the area.

So the geometry to worry about is the **shallow** crossing, not the
high-angle one — the opposite of what "high angle" suggests, and the
opposite of the case a designer would think to check.

## Measuring the slip is harder than it looks

Slip is read as the tangential velocity jump between probes either side
of the band. A **fixed** standoff — some fraction of the width — is wrong
at a crossing: near a shallow one, a probe placed at $0.75\,w$ from one
trace sits *inside the other ribbon*, in weak material, and reports
almost no jump. At $\Delta = 10^\circ$ that read 0.006 against a peak of
0.18 on the same fault.

The probes have to be walked clear of **every** ribbon before they are
read. The jump then means the same thing at every angle: the velocity
difference across the whole weak region.

Even walked out, the local probe **cannot** be trusted at the crossing
itself: that is exactly where the standoff changes discontinuously, and
the reading there dips sharply — at $\Delta = 90^\circ$ it goes negative
once the no-fault baseline is subtracted, which is not a physical slip.
This is the reason the metric above is a dissipation integral. Use the
slip probe for the arms and for pictures, not for the number that decides
the question.

Two further cautions on reading absolute slip here:

- The pair is symmetric about the loading axes, so changing $\Delta$
  necessarily changes how well each plane is oriented for slip. At
  $\Delta = 10^\circ$ both planes are nearly vertical — nearly principal
  planes — and barely slip at all. Compare ratios at fixed angle, not
  slip across angles.
- The no-fault control read through the same probes gives the wall rock's
  own contribution over the probe separation. Subtract it.

## How much of this is the mesh

The overlap is a few tens of cells, so the numbers above are worth
nothing without a resolution control. At $\Delta = 45^\circ$:

| host cell | zone $h$ | zone cells | overlap | $W$ nearest | $W$ orthotropic | ratio |
|---|---|---|---|---|---|---|
| 0.030 | 0.0100 | 380 | 12 | 10.9522% | 12.4148% | 88.2% |
| 0.030 | 0.0050 | 1480 | 34 | 11.5662% | 12.4351% | 93.0% |
| 0.030 | 0.0025 | 5342 | 130 | 11.7717% | 12.4565% | 94.5% |
| 0.040 | 0.0050 | 1480 | 34 | 11.5593% | 12.4278% | 93.0% |
| 0.020 | 0.0050 | 1480 | 34 | 11.5714% | 12.4404% | 93.0% |

The **host** mesh does not enter: 0.04, 0.03 and 0.02 give 93.0% three
times over, with $W$ agreeing to four significant figures. The embedded
zone mesh is identical in all three, which is the embedding working as
designed and worth having as a measurement rather than a claim.

The **zone** mesh does. Halving it twice moves the ratio 88.2 → 93.0 →
94.5%, and the increments fall by about a factor of three, so the
converged value here is near 95%.

The same check at the worst angle, $\Delta = 10^\circ$, where the overlap
is largest and the resolution matters most:

| host cell | zone $h$ | zone cells | overlap | $W$ nearest | $W$ orthotropic | ratio |
|---|---|---|---|---|---|---|
| 0.030 | 0.0050 | 1374 | 141 | 1.1980% | 1.7103% | 70.0% |
| 0.030 | 0.0025 | 4910 | 522 | 1.2731% | 1.7252% | 73.8% |
| 0.030 | 0.0013 | 19562 | 2051 | 1.3030% | 1.7361% | 75.1% |

Increments of +3.8 then +1.3, the same threefold contraction, giving a
converged value near 76%.

Refinement always moves the nearest-fault rule *closer* to the
orthotropic one, so the sweep table earlier on this page **overstates the
error** everywhere. The corrected statement is that the rule loses about
a quarter of the network's weakening at the worst crossing angle, not a
third — which changes the size of the effect and none of its structure.

## `clearance` is not monotone

The carve deletes host vertices near the assembly's skin, and whether
what is left is one simple hole is a topological accident of the host
mesh against a plus-shaped cavity. It does not improve monotonically with
`clearance`: a 30-degree crossing builds only at `clearance=1.3`, a
45-degree crossing at 1.0 but *not* at 1.3. Walk a ladder and record
which value was used.

```python
for clearance in (1.0, 0.9, 1.1, 0.8, 1.2, 0.7, 1.3):
    try:
        dm, info = place_thin_volume(..., clearance=clearance)
    except RuntimeError:
        continue
    break
```

## Postscript: this test was too kind

Everything above is measured in pure extension with the fault pair
symmetric about the load — a drive with **no rotation**. Repeating the
question in a shear box with an asymmetric branching geometry (spin in
the kinematics, nothing mirror-symmetric) inverts the verdict: the
difference between director rules stays immaterial, but the *entire
TI-arms family* loses factors — not percent — against the isotropic
reference wherever the network must hand slip between differently
oriented members. The overlap tensor was never the dominant term; the
arms' constraint is. See
[Branching Faults and Junctions](fault-branching-junctions.md), which
supersedes the recommendation implied here for any network that
redistributes slip.

## See also

- [Fault networks](fault-networks.md) — the zero-thickness toolkit, and
  the offset-junction preparation the ribbon does not need.
- [Split-node faults](split-node-faults.md) — the contact representation.
- [Transverse isotropy](vep-transverse-isotropy-faults.md) — the weak-plane
  rheology and its parameters.
