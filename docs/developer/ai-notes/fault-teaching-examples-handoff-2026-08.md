# Fault-mechanics teaching examples: curation handoff

Written 2026-08-05 for the session curating the teaching materials.
Everything described here is committed and pushed on
`origin/feature/fault-split-node` (through the commit adding this
file). Read this document first; every path below is repo-relative on
that branch.

## The one critical dependency

**These examples run ONLY on `feature/fault-split-node`.** They are
built on the split-node fault capability (`fault_split.py`,
`fault_contact.py`, the pair blocks in `rotated_bc.py`), which is not
on `development` yet. Curate the *placement and prose* freely, but the
examples cannot execute from any other branch, and they should not be
merged into user-facing docs ahead of the capability itself. The
figures, however, are committed PNGs/GIFs — prose pages that embed
them render fine anywhere.

## Where everything is

- **Page**: `docs/advanced/fault-mechanics-examples.md` (in the
  `docs/advanced` toctree). Nine figures + three animations, each
  with a short teaching narrative and a closing "what these models
  are, and are not" caveat section.
- **Scripts, figures, caches**: `docs/advanced/figures/fault-examples/`
  — one Python script per example, committed output PNG/GIF alongside,
  probe caches as `_*.npz` (committed: they let anyone re-style every
  figure and both animations without re-running ~100 solves).
  `_*.png` frame intermediates and `*.log` files are gitignored.
- **Shared harness**: `figures/fault-examples/common.py` — every
  example builds on it (see contracts below).
- **Companion user documentation**: `docs/advanced/split-node-faults.md`
  (the API guide the examples assume) and the method/benchmark
  write-up `docs/developer/design/SPLIT_NODE_FAULT_METHOD_2026-08.md`
  (with its own figure set under
  `docs/developer/design/figures/split-node-faults/` — cetz sources,
  generators, and PNGs for the split anatomy, pair transform, mesh
  stack, and grid hierarchy; those serve the method paper / blog
  posts rather than the teaching page).

## The example inventory (teaching order)

1. **`ladder.py` → `ladder.png`** — *the fault-strength ladder.* One
   fault, one shear drive, five solves down the constitutive ladder
   (frictionless / viscous eta/a / Coulomb-weak / rate-state /
   Coulomb-stuck), slip profiles against the elliptical crack shape,
   plus a stacked column of per-rung shear-stress panels. Teaches: what
   an interface law IS. ~5 solves.
2. **`mohr_circle.py` → `mohr-circle.png`** — welded faults as passive
   stress probes trace the Mohr circle (fitted radius 1.411 vs analytic
   sqrt(2)). Teaches: the probe instrument + the circle exists.
3. **`mohr_animate.py` → `mohr-circle-build.gif`** — the circle built
   frame by frame as the fault rotates; the measured traction vector
   snaps to the fault normal at the principal orientations, flagged on
   both panels. Teaches: the 2-theta rule as motion. 25 solves, cached.
4. **`mohr_friction.py` → `mohr-friction.png` + `-build.gif`** — give
   the rotating fault Coulomb friction: stuck probes reproduce the
   circle, sliding probes pin to the envelope, tensile probes are
   HELD SHUT (unphysical — the bilateral constraint glues an opening
   fault; the tell is the sign of the normal traction). Teaches: the
   yield envelope truncates the circle.
5. **`mohr_cohesion.py` → `mohr-cohesion.png` + `-build.gif`** —
   cohesion: strength declines through mild tension to zero at
   sigma = -C/mu; stuck arcs at both poles; the law is registered as a
   four-line sympy expression (the extension path for any rheology).
6. **`mohr_graded.py` → `mohr-graded.png`** — hydrostatic load: each
   welded fault becomes a depth-coloured STREAK spanning the family of
   Mohr circles between its shallowest and deepest points. Teaches:
   per-node stress recovery = depth-dependent strength along one fault.
7. **`orientations.py` → `orientations.png`** — frictionless faults at
   swept orientations: peak slip follows |cos 2 theta| with an exact
   zero at 45 degrees. Teaches: resolved stress controls slip.
8. **`interacting_faults.py` → `interacting-faults.png` +
   `interacting-rotation.png`** — King-style stress transfer: the
   source slips, the receiver is welded in the source's along-strike
   tip lobe. Delta CFF as a field (linear RdBu_r at +-1, P0 cells)
   AND as the receiver's probe cloud moving in the Mohr plane, with
   the loaded near-tip nodes CROSSING the cohesive envelope into the
   shaded failure zone while the far end retreats. The friction
   dressing (P0 = 1 confining pressure, C = 0.75 cohesion) enters the
   ENVELOPE only, never Delta CFF — both docs say so explicitly. The
   rotation sweep's message: the regional stress orientation controls
   the MARGIN to the envelope (grazing at phi = 20, crossing at 45,
   parked safe at 70). 6 solves at h = 0.016, 13-18 s each.
9. **`california.py` → `california.png`** — schematic southern
   California: the San Andreas as ONE continuous dextral trace with a
   smooth tanh S-bend (the Big Bend — the smoothed stepover; left
   step = restraining). The curved trace is sampled as a polyline and
   carries the smooth curve's ANALYTIC NORMAL via
   `add_fault_bc(..., normal=...)` — the capability that made curved
   traces viable (see the rendering/roughness rules below). The
   restraining bend fills with strong compression: a dCFF bowtie
   exactly where the Transverse Ranges belong — the model puts the
   mountains where the mountains are. Garlock (resolves sinistral
   from the kinematics), three ECSZ strands and a San Jacinto
   analogue welded as probes, all inboard. Verdicts: SJF relaxed
   (-0.41), Garlock and ECSZ mildly loaded (+0.05/+0.06); slip
   arrows from the MEASURED jump (+0.242 right-lateral — a
   continuous trace slips more than the two offset segments did).
   ~50 s of compute at h = 0.012. The capstone.

   RENDERING RULES the interaction fields obey (hard-won, measured):
   stress fields are P0 (cell) projections — continuous-P1 projection
   of rough near-fault stress rings at node scale (residual rms 0.26
   at half-wavelength h/2) — rendered as CELL data on the split
   mesh's TRUE connectivity (common.split_mesh_cell_render /
   split_mesh_cell_rows); never Delaunay the DOF point cloud of a
   split mesh (coincident fault pairs + trace-crossing edges paint
   false beading). Colour: linear RdBu_r at +-1 (dCFF per unit
   stress drop) — log scaling makes the far field "overflow".

## Later additions (branch `feature/fault-teaching-animations`)

Three animated examples added 2026-08-06 for the EMSC-3002 teaching
decks, on a worktree branched from this one at `03055633`. They use the
harness unchanged — no edits to `common.py`, so they merge cleanly.

10. **`mohr_two_fields.py` → `mohr-circle-build-A/B.gif` +
    `mohr-two-circles.png`** — the same rotating probe swept through TWO
    stress states (`shear_plus_stretch(0.5, 1.0)`, R = sqrt(2); and
    `pure_shear_drive(phi=60, tau0=0.8)`, R = 0.8), on identical Mohr
    axes. Adds: the applied field drawn as wall tractions; the principal
    axes struck through the model as LINES when the probe crosses
    tau = 0, taken from the measured crossings (both sweeps recovered
    their drive's axes exactly — 22.5/112.5 and 60/150 degrees); the
    probe restyled as an instrument. Contains a general `probe_under
    (drive, theta)` — `common.mohr_probe` is hardwired to
    shear_plus_stretch. 2 x 25 solves, cached.
11. **`california_clocks.py` → `california-clocks.gif`** — the
    california.py geography with a rotating welded GAUGE at each of the
    three neighbour sites (half-length 0.05; max safe 0.0617, asserted
    against trace/wall/mutual clearance before solving). All gauges share
    ONE solve per angle. Swept twice, trunk welded then free, so dCFF is
    the motion of each circle. Result worth knowing: the ambient circles
    are IDENTICAL at all three sites (the regional field is uniform and
    matches `ambient_sigma_n_simple` exactly) — the whole signal is in
    the post-slip pass. dCFF range over orientations: Garlock -0.64 to
    +0.54, ECSZ -0.01 to +0.15, SJF -0.75 to +0.66; the medians are near
    zero and say almost nothing. A fourth FIXED gauge in the quiet
    corner is the pressure reference (the constant runs ~+1.97 welded but
    ~+35.7 when the trunk slips — the drift the gauge discipline exists
    for). 2 x 25 solves at h = 0.012, ~20 min; cache is 6 KB.
12. **`mohr_failure_field.py` → `mohr-failure-field.gif`** — the
    rotating fault with the FIELD beside the Mohr panel: colour is
    d(tau_max), the change in the LOCAL Mohr radius, with ticks along the
    most-compressive principal direction. Stuck orientations leave the
    field perfectly uniform; sliding ones grow the tip lobes and swing
    the axes. Uses the COHESIVE law of mohr_cohesion.py, not bare
    friction — a purely deviatoric drive puts half the sweep in tension,
    where bare friction is held-shut everywhere (measured: 12 of 25
    orientations unphysical with bare friction, 7 with cohesion). Fields
    are cached per angle (3.1 MB) and rendered in matplotlib
    `PolyCollection` off the split-mesh triangles rather than pyvista.
    25 solves at h = 0.025, 150 s.

GIF sizing note: a continuous field render needs palette quantisation to
stay near the docs budget, and **dithering must be off** — Floyd-Steinberg
scatters pixel noise that defeats GIF run-length compression and made the
file three times larger (1.5 MB dithered vs 0.65 MB not).

## The harness contracts (`common.py`)

- `base_mesh(h)` + `mesh.add_fault([...])`: one static base, re-faulted
  per case (the non-cumulative pattern). Networks = list form;
  segments must not share vertices (ligament >= 2h).
- `stokes_on(child, drive)`: P2 velocity / P0 *discontinuous* pressure
  (the fault pressure-space ruling), all-wall Dirichlet drive,
  `petsc_use_pressure_nullspace = True`, `stokes.tolerance` (never raw
  ksp_rtol).
- Drives: `simple_shear`, `shear_plus_stretch(a, gamma)` (Mohr radius
  eta sqrt(4a^2+gamma^2)), `pure_shear_drive(phi)` (phi = COMPRESSION
  axis — an earlier sign error made it the extension axis; fixed),
  `boundary_simple_shear(trend)` (right-lateral along a plate-boundary
  trend; dextral verified from the measured jump).
- **Read fault quantities through the DOF pairing only**
  (`slip_vs_position`, `probe_nodes`, `fault_pair_jumps`): the two
  sides are geometrically coincident — coordinate queries see one side.
  `probe_nodes` selects the named fault's nodes via its own pairing
  (several law-carrying faults share one assembler).
- **Pressure-gauge discipline** (closed velocity-driven boxes fix
  pressure only up to a per-solve constant; observed drifts up to
  ~300 stress units on large slip events): differenced fields are
  anchored with `far_field_anchor` (a slip event changes nothing far
  away), absolute probe pressures with `ambient_sigma_n` /
  `ambient_sigma_n_simple` (analytic ambient states). Every removed
  constant is printed, never silently absorbed.
- Field rendering (the standing rules, measured not guessed): stress
  fields are P0 (cell) projections rendered as CELL data on the split
  mesh's TRUE connectivity via `split_mesh_cell_render` /
  `split_mesh_cell_rows`. Never Delaunay the DOF point cloud of a
  split mesh (coincident fault pairs + trace-crossing edges paint
  false beading), and never project rough near-fault stress to
  continuous P1 (node-scale ringing, residual rms 0.26 at
  half-wavelength h/2). Colour: linear RdBu_r at +-1 — dCFF per unit
  stress drop, pale far field. (`signed_log` remains in the harness
  if a far-field-emphasis view is ever wanted, but the maintainer
  rejected it for these figures: it makes the far field "overflow".)
- Curved faults carry their ANALYTIC NORMAL
  (`add_fault_bc(..., normal=sympy 1×dim Matrix in mesh.X)`). The kink
  roughness of a sampled curve was diagnosed (2026-08-05,
  ~/+Simulations/curved_fault_roughness/): NOT the meshing, NOT the
  integration — the default per-node normal AVERAGES the adjacent
  facet normals and zig-zags at the sampling kinks, so the no-opening
  constraint forbids smooth slip past each kink (sawteeth that GROW
  under refinement). The analytic normal on the same kinked mesh cuts
  the sawtooth 7-17x and restores h-convergence. Straight faults need
  nothing (the average is exact there); a deliberately kinked fault
  should NOT be given a smooth normal — the kink response is then the
  physics.

## Conventions (consistent across every figure)

- Mohr planes in the GEOLOGICAL sign convention: compression positive,
  tension on the negative axis (solver tractions are tension-positive;
  only plots flip).
- The interaction figures dress friction with a declared confining
  pressure P0 = 1 and cohesion C = 0.75 (envelope
  tau = +-(C + mu' sigma)); **neither enters Delta CFF** (constants
  under differencing) — they place the failure line where teaching
  needs it, and the pages say so.
- mu' = 0.4 (King's value) for all Delta CFF.
- The California geography is SCHEMATIC but it is not arbitrary:
  which SIDE of the San Andreas a named fault sits on is a real
  claim, and students check it. The Garlock and the three ECSZ
  strands are inboard (NE, North American side); the SAN JACINTO
  is OUTBOARD (SW, Pacific side) -- it took a share of the
  boundary slip on the far side of the master fault. This was
  wrong until 2026-08-25 (the San Jacinto sat NE with the ECSZ,
  where a fourth ECSZ strand would go) and it propagated into
  five published figures. If you add or move a fault, check its
  perpendicular offset against saf_trace() before solving.
- Fields render in pyvista (RdBu_r, white background, lighting off);
  line/scatter plots in matplotlib. Method-paper figures (cetz) use
  sans labels (Noto Sans -> Helvetica fallback) at small sizes.

## Regeneration

Everything runs inside the worktree env, with its bin on PATH (the JIT
needs mpicc):

    cd <worktree>/docs/advanced/figures/fault-examples
    PATH=<worktree>/.pixi/envs/runtime/bin:$PATH \
        <worktree>/.pixi/envs/runtime/bin/python -u <script>.py

With the committed `_*.npz` caches present, every script regenerates
its figures/animations in seconds (plot-only). Delete a cache to
re-measure. Measured costs on a laptop: ladder ~5 solves of ~30 s;
each Mohr sweep 25 solves of ~30-60 s; the en echelon sweep 6 solves
of 13-18 s (h = 0.016); California ~50 s total (h = 0.012: 37 s
slipping solve, 11 s welded + projections). pyvista emits harmless
`__del__` AttributeError noise at interpreter exit; ignore it.

## Physics caveats (ready-made prose on the page)

The page's closing section states: static incompressible elasticity
and Stokes are the same mathematics, so these are exactly the elastic
nu = 1/2 patterns with slip rate as coseismic slip; deliberately
absent are depth/half-space geometry, topography, gravity (except the
graded fault), compressibility, and postseismic processes; slipping
faults are completely weak during their event, and Delta CFF scales
linearly with the dropped stress (partial drop = the Coulomb rung on
the source). Keep that section attached to wherever the interaction
examples land.

## Curation notes

- If the examples move (e.g. into a tutorials tree), keep script +
  figure + cache TOGETHER, and update the relative image paths in the
  page. `common.py` must travel with the scripts.
- Notebook conversion is mechanical: each script is linear
  (setup -> sweep(cached) -> figures) and the caches make cells fast.
- The animations are GIFs sized for docs (~200 KB); they embed with a
  plain image directive.
- Known remaining candidates (listed on the page): tip-to-tip vs
  overlapped en echelon side by side, denser networks, gravity-loaded
  interactions, and — further out — 3-D examples (the capability
  exists: see `docs/advanced/split-node-faults.md`).
