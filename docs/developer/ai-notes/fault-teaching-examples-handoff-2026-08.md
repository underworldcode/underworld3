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
  `docs/advanced` toctree). Thirteen figures + two animations, each
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
   `interacting-rotation.png`** — King-style stress transfer: source
   slips, receiver welded; Delta CFF as a field (symmetric-log colour)
   AND as the receiver's probe cloud moving in the Mohr plane, nodes
   crossing the cohesive envelope. The rotation sweep shows orientation
   controls the MARGIN to failure. 6 solves + projections.
9. **`california.py` → `california.png`** — schematic southern
   California: the San Andreas as TWO parallel offset dextral segments
   (left step = restraining; the stepover is the Transverse Ranges,
   beside the Garlock). Straight segments by design: curved polyline
   trunks leave kink roughness (tried, measured, rejected). Garlock
   (resolves sinistral from the kinematics), three ECSZ strands and a
   San Jacinto analogue welded as probes, all inboard. Verdicts: SJF
   relaxed (-0.45), ECSZ neutral, the Garlock cloud STRETCHED
   (stepover end loaded, far end relaxed); slip arrows from the
   MEASURED jump (+0.167 right-lateral). ~50 s of compute at
   h = 0.012. The capstone.

   RENDERING RULES the interaction fields obey (hard-won, measured):
   stress fields are P0 (cell) projections — continuous-P1 projection
   of rough near-fault stress rings at node scale (residual rms 0.26
   at half-wavelength h/2) — rendered as CELL data on the split
   mesh's TRUE connectivity (common.split_mesh_cell_render /
   split_mesh_cell_rows); never Delaunay the DOF point cloud of a
   split mesh (coincident fault pairs + trace-crossing edges paint
   false beading). Colour: linear RdBu_r at +-1 (dCFF per unit
   stress drop) — log scaling makes the far field "overflow".

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
- `signed_log` / `signed_log_annotations`: the symmetric-log colour
  transform for Delta CFF fields (linear inside |x| = 0.02), with
  scalar bars annotated in true units.

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
re-measure (costs: ladder ~5 solves; each Mohr sweep 25; interactions
6; California 2 at h = 0.02 — minutes each on a laptop). pyvista emits
harmless `__del__` AttributeError noise at interpreter exit; ignore it.

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
