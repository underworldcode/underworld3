# Fault networks: hierarchy, junctions, and damage-zone glue

Real fault systems cross, branch, and abut. The split-node formulation
(see [Split-node faults](split-node-faults.md)) deliberately refuses
shared vertices — a junction vertex would need a non-binary contact
pairing — so a network is represented in the **offset form**: at every
junction the junior trace stops a cell or two short of the senior one,
and a small **damage zone** of viscoplastic material connects them.
`FaultNetwork` packages that whole recipe.

```python
import underworld3 as uw

net = uw.meshing.FaultNetwork(
    [("Main", main_pts), ("Splay", splay_pts), ("Cross", cross_pts)],
    hierarchy=["Main", "Splay", "Cross"])   # seniority order

mesh = net.prepare(h=0.006).build(width=0.01)  # junctions -> mesh -> split

v = uw.discretisation.MeshVariable("V", mesh, 2, degree=2)
p = uw.discretisation.MeshVariable("P", mesh, 1, degree=0,
                                   continuous=False)
stokes = uw.systems.Stokes(mesh, velocityField=v, pressureField=p)
stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
stokes.constitutive_model.Parameters.shear_viscosity_0 = \
    net.junction_patch(eta_0=1.0)            # the junction glue (linear)
net.apply(stokes)                            # no-opening pairs, all pieces
# ... wall boundary conditions ...
info = net.solve(stokes)
print(net.slips(stokes))                     # peak slip per piece
```

## One specification, two realisations

A fault is specified once — a trace, its rank in the hierarchy, and the
properties it carries — and then *realised*. Which realisation you get
is a keyword on `build`, not a different set of calls:

```python
net.prepare(h=0.006)
mesh = net.build(width=0.01)                      # cut, node-pair contact
mesh = net.build(width=0.002, realisation="ti")   # volumetric weak plane
net.apply(stokes, eta_1=0.01)                     # eta_1: TI only
```

Both realisations place the same ribbon band along the same prepared
pieces, so the cells are identical and results from the two may be
compared directly. The band is meshed around the trace's own points and
segments, which become mesh vertices and edges, so **the mesh can be cut
whatever the width is** — measured complete, with exact vertex
coincidence, down to a band a tenth of the background element size. The
realisation is a free choice, not something the mesh grants or refuses.

What differs is what `width` *means*. For the split it is a resolution
parameter: the band exists to give the cut its own vertices, and its
thickness is not physics. For the weak plane it is constitutive — the
layer thickness that sets the slip rate through `V = 2 e_nt w` — so it
wants two or three elements across it. That is the whole asymmetry, and
it is about the rheology rather than the mesh.

`slips()` reports each realisation in its own quantity: the tangential
jump between the two nodes of a cut pair, or the jump in tangential
velocity across the layer, sampled one half-width plus a cell either
side of the spine. Both are the fault's own throughput; a probe placed
further out reads the surrounding flow as well and over-reads short
strands.

`build(width=None)` keeps the older no-band path — graded refinement
cut directly. It is split-only, and its mesh is not the one a weak
plane would use, so do not compare across that choice.

**The band is material, not scaffolding.** It is easy to read the band
as something the weak plane needs and the split merely tolerates. It is
not: the band is a meshed region of material *around* the fault, and a
segmented fault does its interesting work exactly there — at the strand
tips, and in the ligaments where one cut stops short of the next. Damage
in those places needs cells to live in, and the band is where they are.
`net.band` is the mask, `net.footprints` the per-strand ones, in either
realisation.

`net.band_yield(tau_y)` gives a rheology for the whole band: von
Mises yield confined to it, everything outside set far too strong to
yield. Read the next paragraph before using it as glue.

```python
stokes.constitutive_model = uw.constitutive_models.ViscoPlasticFlowModel
stokes.constitutive_model.Parameters.yield_stress = net.band_yield(4.0)
stokes.consistent_jacobian = True
```

A released fault flank sits far below `tau_y` and is untouched, so
the breakdown appears where the mechanics puts it — but that is tips
and bends as much as joints. Measured on the S-fault rig: at the
strength that repairs a stepover, more than half the yielded band cells
were on strand flanks and free tips, and one step weaker the whole
main strand had become a weak fault. A uniform threshold cannot pick
out the welds alone, because the stress concentration at a weld is not
far enough above the tip and bend concentrations. `band_yield` is a
damage model for the band; the junction glue is `junction_patch`.

The two realisations' interface parameters correspond, which is worth
keeping in view when comparing them: the zero-thickness limit of a band
of viscosity `eta_band` and width `w` is an interface viscosity
`eta_f = eta_band / w`, which is the `conds` argument of
`add_fault_bc`. The weak plane's `V = 2 e_nt w` is precisely what the
contact replaces with a genuine slip rate.

**Properties belong to the fault.** `net.surface(name)` returns the
retained {class}`~underworld3.meshing.surfaces.Surface` for a piece.
Friction, accumulated slip, a damage state live there, on the fault,
and outlive any one realisation of it:

```python
main = net.surface("Main")
friction = main.add_variable("mu", size=1)
friction.data[:] = 0.6
```

## The recipe, and why each piece is the way it is

**Hierarchy.** At an X crossing the senior fault runs through and the
junior is severed and pulled back (`hierarchy`, pairwise). A trace that
*ends* on another (T abutment) is always the one trimmed — an endpoint
cannot run through. Absolute masters (`through=`) override rank.

**Small junctions.** The pull-back is `ligament * h` (default 2
elements). Measured on a collinear bridge: gaps from 1 to 6 elements
all solve at identical cost, slip transmission varies by only ~7%
(monotone — shorter is better), and one element of gap resolves at the
same answer when the junction patch is refined 2x. Make the join as
small as the mesh allows; buy fidelity with elements, not physical
size.

**The glue, and where it goes.** The split only goes wrong at the
joints: away from them the cut *is* the target every volumetric
representation converges to, and adding weakness along a whole strand
makes the fault over-weak in a way that depends on the band width. So
the glue is placed, not found. `junction_cells()` reads the places off
the mesh itself: the ribbon (the band with its extrapolated margins)
is everything the weak-plane realisation would treat as fault, the cut
chains are what the split sliced, and a band cell whose nearest spine
point lies in a piece's margin *and* which sits inside a second
piece's ribbon is where two pieces meet without being joined — a
kissing branch, an abutting pair, the intact bridge of a stepover.
Free tips are excluded on purpose: a margin that runs into intact
material is a tip, and damage there lengthens the fault instead of
joining it (measured: with the free tips included, nearly every
yielded cell was at a tip and the main strand grew 1-6% longer in
slip). The cells are dilated by one vertex ring, and that ring is
not optional: the weld's stiffness lives in the intact material
around the two tips, and the bare junction cells recover only a
fifth to a quarter of the deficit even when fully plastic.

Two pieces that continue one another along a line — an abutting pair,
a stepover's continuation — are placed on **one spine**: two ribbons
laid along the same line interleave their vertices into sliver cells
(measured: 7800 cells below 1e-6 in area, and the velocity solve
five times slower). `build()` groups such pieces (end tangents within
25 degrees, the far start within half a width of the line, within the
margins' reach), bridges the gap with spine vertices at the local rung,
and cuts each piece at its own ends; the gap is spine the split does
not cut, which is exactly what the junction rule reads.

For the rule to see a joint, the ribbons have to meet across it.
`build()` sees to that: at an end that sits on a prepared junction the
tip margin is extended until the ribbon reaches the other piece's cut,
so the whole ligament lies in both ribbons; free tips keep the default
margin. An abutting pair that `prepare()` did not record as a junction
(a gap wider than the ligament) is covered as far as the default
margins overlap — a gap wider than that is two faults, and stays
welded, which is what a gap of intact rock means.

`junction_patch(eta_0, ratio=0.01)` then makes those cells weak
isotropic material, `eta = ratio * eta_0`. A viscosity ratio rather
than a yield stress, because the joint only has to be broken and a
ratio needs no stress scale — nothing about the block or the loading
has to be known to set it. Measured against the two end members on
the S-fault rig (the fault *longer*, one continuous cut, and the fault
*cut*, abutting cuts, at two resolutions): the patch recovers
0.8-0.97 of the continuous fault's transmission across the joint; the
slip crosses on the cut itself (the segment's pair jump reaches the
continuous fault's); the rest of the network keeps the split's answer
(main strand within 2.5%); the weak patch reproduces a fully plastic
patch on the same cells to 1-2% and is insensitive to the ratio from
0.01 to 0.001; the solve is linear and costs the split's velocity
iterations, with only the pressure block noticing the contrast
(hence 0.01, not smaller). Gluing a joint does change the partition
between the strands that meet there — a reconnected main line takes
back slip a through-going branch was carrying past the weld — which
is the junction working, not the patch leaking.

`damage_yield` is the older glue: a viscoplastic plug of radius
`max(2.5 h, 1.2 pull)` at each *prepared* junction point, yield
`dial * (1 + 2 * edot_II)` inside, strength and rate-regularisation on
one dial, sharp `Piecewise` boundaries. It stays available for
studies of the glue itself, and it does not see stepover bridges,
which are not prepared junctions.

**No prescribed reconnection.** Nothing tells the network how to link
up: the stress lobes of the abutting tips decide. A collinear gap
grows a straight band; a branch junction feeds whichever limb the
drive favours, and switches when the drive rotates.

**Isotropic glue, deliberately.** Oriented (transversely isotropic)
weakness in the plug cannot relax the corner-turning deformation a
junction must accommodate — a weak-plane fabric only softens shear ON
the plane, and the measured junction stress stays high no matter how
weak the plane is made. Use TI damage along fault *zones* (where
deformation is plane-parallel); junctions get isotropic breakdown,
which is also what junction breccia looks like in the field.

## Baseline discipline

The stateless control for any network experiment is a plug of constant
weak viscosity — it transmits and partitions slip essentially
identically to the damage glue (measured within ~5%). What damage adds
is **state**: wear-in and healing with a slip-rate memory, which only
changes the *instantaneous* mechanics once elasticity (stored stress)
enters. Judge any refinement of the glue against that control.

## 3-D networks: planar patches

Pass triangulated `FaultSurface` objects instead of 2-D traces and the
same object runs the 3-D pipeline:

```python
fsA = uw.meshing.FaultSurface("Main", main_pts);  fsA.triangulate()
fsB = uw.meshing.FaultSurface("Cross", cross_pts); fsB.triangulate()
net = uw.meshing.FaultNetwork([fsA, fsB], hierarchy=["Main", "Cross"])
mesh = net.prepare(h=0.06).build()      # trim -> embed -> split each
```

In 3-D two planar patches meet along a **segment** (the plane–plane
line clipped to both rims). The junior patch is cut into two pieces by
removing a ligament band about that line — *in its own plane* — so the
mesher (`BoxInternalPatch`, which now embeds a list of disjoint
patches with gmsh grading from every patch) never sees intersecting
surfaces: prepare-first, exactly as in 2-D. The junction glue becomes
a **tube** (distance to the segment) instead of a disc; `damage_yield`
handles both automatically. Pieces that fall below a minimum area
after trimming are dropped and reported — small patches with large
ligaments can disappear entirely, so watch the report.

Two meshers share the contract (`build(mesher=...)`):
`"embed"` (default) is the gmsh conforming multi-patch embed — fast
and proven for networks; `"place"` is the native placed-sheet route
(`place_surface.place_sheet`), non-cumulative from a static base (the
fault position is a design variable) and parallel-capable end to end
— the composed chain place -> split -> contact converges at np=2 at
serial speed with the serial answer (ptest_0852). Placement needs
sheet triangulations with no all-rim triangles; the toolkit
triangulates prepared rims itself (interior grid, centreline points
for narrow strips, edge flips) so users never meet that contract
directly. Place is currently OPT-IN for networks: on graded
(edge_split) bases the composed mesh builds and converges but the
solve is pathological — an open operator-health work item; on uniform
bases it is healthy.

v1 scope, refused loudly outside it: planar patches (the
`rim_polygon` contract), convex rims, genuine X crossings (a
near-miss — close but not crossing — is refused rather than guessed
at); parallel MULTI-fault splitting (the pairing does not yet migrate
through redistribution — single faults are parallel-validated).

## Limitations

- 3-D: planar convex patches, X crossings only, serial (above); the
  weak-plane realisation is 2-D for now — place 3-D zones with
  `place_thin_volume` directly.
- One damage dial per network in `damage_yield` (per-junction values:
  build the expression with `uw.meshing.damage_zone_yield` directly).
- Time-dependent damage (wear-in/healing) is study-level for now: see
  `~/+Simulations/fault_junction_rheology/damage_switch.py` for the
  pattern (per-fault scalar states driven by each fault's slip rate).
