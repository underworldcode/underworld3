# Underworld3 Development Changelog

This log tracks significant development work at a conceptual level, suitable for quarterly reporting to CIG and stakeholders. For detailed commit history, see git log.

---

## 2026 Q3 (July – September)

### The Multiplier Was Not the Whole Traction (August 2026)

**`Stokes_Constrained.topography()` now returns the traction the boundary is
actually held with**, and a new `traction()` exposes it directly. The momentum
row carries `λ + r(n·u − g)`, so the bare multiplier is short by the
augmented-Lagrangian share — `r` times the discrete constraint residual. With the
viscosity-weighted default `r = 1e4·μ(x)` that share is a few per cent of the
surface traction on a uniform-viscosity annulus and most of it across a `1e6`
viscosity step, where `λ` alone reads a tenth of the exact SolCx topography and
is anti-correlated with it. `multiplier()` still returns `λ` and now says what it
is not.

The defect survived because the validation scored a **correlation** (0.9999)
between the multiplier and the recovered normal stress. A correlation is
scale-free and cannot see a systematic amplitude deficit, which is precisely what
a missing share of the load is. The new guard,
`tests/test_1063_constrained_traction.py`, scores a relative `l2` against the
exact SolCx surface topography and carries the bare multiplier as its negative
control.

The corrected quantity is the consistent boundary flux: at convergence
`M_Γ(λ + r(n·u − g))` balances the volume residual restricted to the boundary,
which is the CBF nodal load (Zhong, Gurnis & Hulbert 1993). So the multiplier
route and the rotated constraint's `boundary_normal_traction` are the same
computation, and they agree to 3–5% — inside each route's own error against the
exact answer.

Documentation: `docs/advanced/curved-boundary-conditions.md` now writes the
penalty free-slip recipe against `mesh.boundary_normal` rather than `mesh.Gamma`.
A penalty against the per-facet normal over-constrains the shared nodes and does
not converge — measured on an annulus at coefficient `1e6`, the velocity error
stays at 0.60 and the surface-stress error grows from 0.21 to 0.26 as the mesh is
refined, while the leak reads 1e-5 throughout. (underworld3#607, #608, #614)

### A Singular Recovery Mass, Mistaken for a Penalty Defect (August 2026)

**The grad-div penalty default stays off**, but the reason it was held off turned out to
be a defect somewhere else entirely (#633) — so the objection that had blocked it is gone,
and a different one took its place.

With the penalty at 10, the spherical dynamic topography recovered from the rotated
free-slip reaction dropped 28% at *vertices* while the facet-integrated value stayed
correct. The natural reading — that grad-div augmentation corrupts the de-smearing from
reaction loads to pointwise stress — was wrong.

The de-smearing mass for a 3-D **P2 triangular** trace has vertex rows that sum to
**exactly zero**. Those rows annihilate a constant, so solving `M σ = R` amplifies any
perturbation of the nodal load at vertices by O(1) — and, being an instability rather
than a discretisation error, does so independently of mesh resolution. The recovery was
already 7.6% low with no penalty at all; the penalty only made it large enough to fail a
test whose 12% tolerance had been hiding it.

The discrimination needed a case that was curved but not 3-D. A 2-D annulus reproduces
the signature exactly and then parts company under refinement: its error falls ~O(h²)
while the shell's stays flat at ~0.28 over a 3.2× node-count range. The 2-D P2 **line**
mass has positive vertex row sums, which is why 2-D never showed the defect and why
dimension, curvature and the rotated constraint were all red herrings.

- The zero-mean P2 vertex basis was **already known and documented in #414**, which
  recorded the same drift-away-under-refinement we re-measured here. Its mechanism is
  the sharper one and is adopted: because the vertex basis has zero surface mean, the
  vertex reaction carries essentially only the O(h) facet-normal/geometry error, and the
  consistent solve *faithfully reconstructs that error* — it is not amplifying noise.
  What #633 adds is the separation from the grad-div penalty (which was blamed for it)
  and the fix below, which is #414's own unactioned recommendation (2).
- So `mass="auto"` stops asking. On a 3-D P2 trace it now takes the **consistent** solve,
  keeps its superconvergent midpoints, and **reconstructs the vertices from them**: the
  three midpoints of a facet determine a unique linear function, so a vertex reads its
  two adjacent midpoints and subtracts the opposite one, averaged over incident facets.
  Worst-node error against the analytic coefficient, over cellSize 0.25 → 0.11:

  | | 0.25 | 0.20 | 0.16 | 0.13 | 0.11 |
  |---|---|---|---|---|---|
  | surface, P1-projected | 0.041 | 0.026 | 0.018 | 0.013 | 0.008 |
  | surface, reconstructed | 0.016 | 0.012 | 0.012 | 0.003 | 0.004 |
  | CMB, P1-projected | 0.116 | 0.067 | 0.047 | 0.030 | 0.025 |
  | CMB, reconstructed | 0.094 | 0.058 | 0.043 | 0.024 | 0.015 |

  Better at every resolution on both boundaries, by 1.8x to 4.9x, and converging. The
  simpler P1-projected recovery stays available as `mass="p1"` — it is sound, it just
  discards the good data along with the bad.
- `FreeSurface` already used the P1-projected recovery in 3-D, so production dynamic
  topography was never affected. The exposure was `mass="auto"`.
- The spherical topography test is refined (cellSize 0.25 → 0.13) and its tolerances
  tightened from 0.10/0.12 to 0.01/0.05, set from measured discretisation error with
  ~2× headroom, and now assert every node class rather than the aggregate — the failure
  was confined to one class and an aggregate assertion passed straight through it.
- Filed #637: 3-D recovery accepts only P1/P2 triangular traces, so dynamic topography
  has exactly one supported discretisation there and cannot be cross-validated. That
  blocked the P3/hex arm of this investigation.
- `Stokes.DEFAULT_PENALTY` was flipped to 10 on the #625 evidence and then **reverted**.
  Three tier-A/B tests fail at 10 and pass at 0, and the same three run **8.3x slower**
  (9.27 s to 76.92 s, warm cache both ways). The #625 win needs an FMG hierarchy; without
  `refinement>=1` the velocity block falls back to GAMG, which is where grad-div
  augmentation drives the solve into its iteration cap. The default path is the one
  without a hierarchy, so the default serves it; set `penalty=10` explicitly where FMG
  is available.
- Two of those three failures are not penalty defects. The Nitsche free-slip leak
  (1.234e-4 against a 1e-4 bound) is augmentation perturbing a *weakly* imposed
  constraint — a strong rotated constraint is untouched. The swarm one exposed #641:
  `evaluate` returns −0.4976 for `sqrt((E**2).trace()/2)` at in-domain points near the
  lid-corner singularity, at `penalty=0` as well; the penalty merely moved an accumulated
  total across zero.

### The Free Surface Reaches the Spherical Shell (July 2026)

**`uw.systems.FreeSurface` now runs in 3D on a spherical shell** — the same
exponential three-number integrator, held-lid σ_nn recovery and strong
material-boundary datum, with the surface machinery made dimension-general
rather than ported piecewise:

- The datum gauge (mean removal) is an FE trace-mass reduction over
  owned boundary facets — no ordered ring, no gather; the same code is the
  2D line gauge and the 3D area gauge. On the way it resolved a real 2D
  defect: the deforming-ring strong-datum solves used to stall at a ~2e-3
  residual floor, which turned out to be three stacked causes (arc-length vs
  FE trace weights; the datum's *directed* mean flux through the deformed
  facet normals, now stripped with the same FE surface integral the residual
  uses; and the constant-pressure gauge mode, which the inner solver projects
  and the outer loop therefore now measures in the quotient space). With all
  three closed, every step of the power-law acceptance run converges.
- σ_nn on a 3D P2 boundary is recovered by **P1 projection** (edge-midpoint
  loads folded exactly onto vertices, sound P1 lumped triangle mass) — chosen
  over the consistent P2 mass because its vertex-integral checkerboard sits
  exactly at the vertices the P1 topography field consumes.
- The two genuinely 2D features (ring Taubin filter, tangential transport)
  are refused explicitly in 3D; everything else is shared code.

First 3D evidence (spherical Y20 topographic relaxation, constant-density
shell): exponential decay at an O(1) shell correction below the half-space
Cathles rate, in the physically correct direction, with the equilibrium
modal bias falling 16% → 2% of the initial amplitude over one resolution
step (the known discrete recovery defect, resolution-convergent). The
detailed benchmarking — analytic shell-rate comparison, convergence study,
low-Ra spherical convection, 3D parallel — is deliberately left to the
review pass.

### One Owner for the Geometric-Multigrid Option Bundle (July 2026)

**The PETSc option bundle that configures a Stokes velocity block's multigrid
now lives in exactly one module, and all three routes that reach that block read
it from there** (#468), **and rotated free-slip now picks up a mesh-owned
multigrid hierarchy instead of silently discarding it** (#467).

Three routes reach a multigrid velocity block: native (PETSc interpolation
between refined DMPlex levels), custom-P on the standard solve path, and custom-P
through the rotated free-slip path. They are the same preconditioner reached
three ways, not alternatives — custom-P is *mandatory* wherever native cannot go,
namely rotated boundary conditions and `adapt()` children. The bundle was
written in two places and had drifted: the native path had been moved to a
`gmres`+`sor` smoother on a recorded measurement, and the custom-P routes had
not. Worse, the custom-P writer never *set* the smoother iteration count at all,
so it inherited whatever had last written that options prefix — 3 left behind by
the GAMG bundle on the standard path, PETSc's own default of 2 on the rotated
path. The same function smoothed differently depending on what had run before
it.

Unifying the bundle recovers, on the same operator, right-hand side and coarse
solve: rotated custom-P velocity-block iterations 11 → 5 (0.68 s → 0.39 s of
linear solve, timed in isolation), standard custom-P 5 → 4, on a *two-level*
hierarchy — the depth at which the native measurement says the gmres margin is
smallest. The bundle also now derives which stale keys it must clear rather than
carrying a hand-maintained list, which is what let the iteration count go unset
in the first place.

Separately, `mesh.adapt()` leaves a coarse tail on its refinement child so that
every solver on an adapted mesh gets geometric multigrid with no per-solver call.
The rotated path never consulted it — the standard path's injection hook runs
after the rotated dispatch has already returned — so an adapt child under rotated
free-slip fell back to algebraic multigrid, indistinguishable from having no
hierarchy at all. That is the `adapt-on-top-faults` workflow's own configuration
(a fault resolved by local refinement, with rotated free-slip chosen because it
composes with transverse isotropy). Both paths now resolve the hierarchy through
one shared rule, with the same opportunistic degrade-to-GAMG behaviour.

The regression test reads the smoother configuration back off the **live PETSc
objects** for all three routes and asserts they agree. An options-database
assertion would not have caught the original drift, because the drift was
precisely a key nobody wrote.

### One Rotated Free-Slip Path, Now With a Prescribed Wall-Normal Velocity (July 2026)

**Rotated strong free-slip now takes a prescribed wall-normal velocity datum
(`add_rotated_freeslip_bc(conds, boundary, ...)` with non-zero `conds`) through
the full nonlinear Newton machinery, and the separate linear and nonlinear
solve paths have been unified into one** (#438; #403 items 2 and 4).

The rotated constraint `u·n̂ = ũ_n` is the primitive behind both the held free-slip
lid and the free-surface material-boundary condition — they differ only in the
constraint right-hand side. Previously the non-zero datum existed only on a
linear one-shot path, so a power-law or anisotropic rheology silently fell back
to a weak penalty (which leaks worst exactly where anisotropy makes it matter).
Now every rotated solve runs a single Newton/Picard loop in which accepted
iterates carry the datum exactly; a cold start imposes it through the first
increment's affine lift at the rest-state tangent (snapping a zero state onto a
datum creates a boundary strain state a shear-thinning tangent cannot recover
from — measured, not assumed); a linear model simply converges after that first
increment, so the up-front nonlinearity probe is gone from the dispatch and
every linear rotated solve saves two Jacobian assemblies.

Three latent solver defects were found and fixed by making the loop report
honestly along the way:

- Branching on rank-local datum bookkeeping desynchronised the ranks'
  collective sequences (an np>1 deadlock class); the datum-activity decision is
  now a collective PETSc reduction.
- A rigid-rotation mode pinned by an essential condition on *another* boundary
  could still enter the solver null space (only the rotated rows were checked),
  silently projecting an irreducible component out of every increment — Newton
  converged superlinearly and then floored, far above tolerance. Candidate
  modes are now verified as null vectors of the assembled operator, which
  catches any form of pinning.
- A tiny Newton step was reported as convergence even when the residual was
  still large (a stiff tangent also produces tiny steps); the step-norm exit is
  now verified against the problem's rest-state residual scale, which is also
  the convergence reference for warm starts (rtol relative to a good warm
  start's own small initial residual demands ever-more absolute accuracy).

The free-surface manager's `consistent_constraint="strong"` therefore works for
nonlinear rheologies: the penalty fallback is removed, and the consistent solve
warm-starts from the free solve's converged fields. Acceptance: power-law
annulus free-surface convection holds the material boundary at the strong-datum
level (5e-3, versus 4e-2 for the penalty) with net surface flux 1e-4; a
transversely isotropic fault-bearing smoke test converges through the same
path. Direct LU per Newton increment remains available as a serial,
preconditioner-free diagnostic (`solver._rotated_use_lu`).

New subsystem documentation: `subsystems/rotated-freeslip.md`.

### Local Interpolation That Reproduces Linear Fields (July 2026)

**The local scattered-point interpolator now has a linear-reproduction
guarantee**, and the swarm proxy variables use it by default (#430).

Underworld's local interpolator was inverse-distance (Shepard) weighting.
Its weights are positive and sum to one, so it reproduced a *constant* exactly
but not a linear field: any field with a gradient was smeared, and the error
did not fall as the points crowded together. Measured on an exactly linear
field — which lies inside both the P1 and P2 proxy space, so the finite element
discretisation contributes nothing and all of the error is particle-to-node
transfer — the swarm proxy carried 1e-3 to 2e-2 relative error, falling only
first order with refinement and **not at all** with stencil size.

- New `order=1` scheme on `uw.kdtree.KDTree.rbf_interpolator_local`: a
  polyharmonic kernel with an affine tail, solved per target point on its own
  nearest-neighbour stencil. Constants and linear fields are exact by
  construction, and the result stays sparse at `nnn` non-zeros per row. The
  existing inverse-distance path is unchanged and remains the KDTree default.
- Proxy error on a linear field falls to round-off; on a quadratic field it
  improves roughly ninety-fold in 2D and thirty-fold in 3D. Swarm proxy
  variables now default to `order=1`.
- `KDTree.interpolation_matrix()` returns the transfer as a sparse operator.
  The weights depend only on geometry, so one build serves every field and
  component — the form a multigrid prolongation or a remesh transfer wants.
- Stencils that cannot support an affine fit (collinear in 2D, coplanar in 3D)
  are detected, retried on a wider neighbourhood, and only then fall back to
  inverse distance with a warning. They never return `NaN`, and never silently.
- An opt-in limiter bounds the non-affine part of the interpolant while leaving
  the linear reconstruction untouched, so limiting does not cost the guarantee.

Two deliberate exclusions, both measured rather than assumed:
`MeshVariable.rbf_interpolate` keeps inverse distance because it is the
fallback rung of the point-location ladder, whose documented contract is that
it is bounded; and `IndexSwarmVariable` material level sets keep it because
they estimate a fraction from a handful of *integer* samples, where the error
is dominated by variance rather than bias. Signed weights amplify that variance
by roughly an order of magnitude and push level sets outside `[0, 1]`, while
the bias they would remove is already negligible.

Related: swarm proxy refresh no longer fails under an active units model
(#426, #434); the units the proxy advertises are tracked separately (#439).
New subsystem documentation: `subsystems/interpolation.md`.

### Purposeful Adapt / Redistribution Naming (July 2026)

**User-facing mesh-modification names now state the capability** (maintainer
naming ruling 2026-07-16); the algorithm names (NVB, MMPDE) stay in internals
and docstrings:

- New user entry `uw.meshing.node_redistribution(mesh, metric, ...)`,
  dispatching through the mesh-controlled `Mesh.redistribute_nodes(metric)`
  method — the architecture by which each mesh type controls how it can be
  modified. The base implementation supports 2D simplex (triangle) meshes
  (via the MMPDE mover); quad/hex, 3D and manifold meshes raise an honest
  `NotImplementedError` stating what exists. `smooth_mesh_interior` remains
  as the machinery underneath.
- `mesh.adapt(metric, max_levels=...)` no longer needs `engine=`: the graded
  newest-vertex-bisection engine is the default on 2D meshes (NVB is 2D-only
  this pass, so 3D meshes resolve to SBR); `engine=` stays as the
  advanced/internal selector.

### Retired Interior Movers — MMPDE Is the Mover (July 2026)

**The superseded fixed-topology interior movers were retired** (maintainer
ruling 2026-07): the spring-equilibrium, Monge–Ampère, OT-improvement-step
and anisotropic-Winslow movers were deleted, together with `mesh.OT_adapt()`
(built on the OT step; closes #346, whose latent MPI deadlock dies with the
spring mover, and #353, whose `strategy=` TypeError dies with the dispatch).

- `smooth_mesh_interior(method=...)` now defaults to **`"mmpde"`** (was
  `"spring"`) — a sanctioned behaviour change: with a scalar metric the
  MMPDE mover reproduces the retired movers' isotropic equidistribution
  (the isotropic-metric equivalence), and with a tensor metric it clusters
  and aligns where they could not. Retired spellings raise a `ValueError`
  naming the replacement.
- `follow_metric(...)` (the two-knob adapter) now drives the MMPDE mover;
  `mesh.OT_adapt()` raises a `RuntimeError` tombstone pointing at
  `follow_metric` / `smooth_mesh_interior` / `mesh.adapt`.
- The graph-Laplacian Jacobi smoother and the Taubin surface-field smoother
  (`smooth_surface_field`) are separate, current tools and are unchanged.
- The boundary-facet / boundary-slip primitives shared with surviving code
  moved from `meshing/_ot_adapt.py` into `meshing/smoothing/graph.py`;
  the style-gate allowlist shrank by the deleted files' entries.


### July 2026 Quality Campaign — Audit, Style Charter, Remediation Waves (July 2026)

**A systematic post-development-burst quality campaign**: six adversarially
verified review dimensions (loose ends, API consistency, readability, swarm
subsystem, docs coherence, branch triage) produced a ranked remediation
worklist, and the first waves have landed (#317, #322, #325, #326, #309–#312,
#334).

- **UW3 Style Charter** adopted as the normative coding contract for every
  development session, human or AI (`docs/developer/UW3_STYLE_CHARTER.md`),
  with the campaign's review documents under `docs/reviews/2026-07/` (#317).
  The maintainer's 18 decision rulings are recorded in the worklist (#322).
- **Track-0 bug fixes**: RBF derivative path now uses the requested component
  rather than component 0 (#312); `CellWiseIntegral` evaluates on the mesh DS
  instead of a mis-cloned P1 DM (#311); Lagrangian history writes use the
  modern data layout (#310); honest 2D-only guard on the MMPDE mover (#309);
  and a medium-severity batch — parallel `BoxInternalBoundary`, SL theta
  restore, viewer crash, projection double-count, units-boundary honesty (#326).
- **Wave A deletions**: the `persistence.py` stub module removed
  (behaviour-neutral), the unsupported coordinate-units feature family pruned
  with the mesh `units=` kwarg deprecated, and the design-directory experiment
  artifacts re-filed under `design/experiments/` (#325).
- **Wave C API harmonization**: the newer BC methods (`add_nitsche_bc`,
  `add_rotated_freeslip_bc`, `add_constraint_bc`) migrated to the canonical
  value-first signature `(conds, boundary, ...)` with one-warning deprecation
  shims for the legacy boundary-first / `g=` spellings; `conds` is the single
  BC datum name across all BC methods (#334).

### Rotated Strong Free-Slip, Boundary Traction and Dynamic Topography (July 2026)

**New `solver.add_rotated_freeslip_bc(...)`** imposes free-slip
(v·n̂ = 0) to machine precision by rotating boundary velocity DOFs into a
per-node (normal, tangential) frame and constraining the rotated normal
component exactly — correct on curved, tilted, and deformed boundaries (#293).

- The constraint **reaction** is the consistent boundary normal traction
  σ_nn: `boundary_normal_traction()` / `dynamic_topography()` recover surface
  topography with no augmented-Lagrangian splitting (#293), and the rotated
  path now runs **inside the nonlinear SNES**, with a numerical nonlinearity
  probe that fails fast instead of silently returning a single linearisation
  for a nonlinear rheology (#298).
- Schur-preconditioning parity: native 1/μ-mass USER Schur preconditioner,
  3D rotation nullspace, and an SVD-robust FMG coarse solve took the Zhong
  spherical benchmark from 44 outer iterations to 1 (#306).
- A general **consistent boundary flux (CBF) primitive** recovers boundary
  fluxes for any solver — surface heat flux / Nusselt number for scalar
  diffusion, boundary traction σ·n for Stokes (#294).
- Three-dimensional CBF recovery now assembles the exact triangular trace mass:
  P1 supports lumped or consistent recovery, while P2 uses the required
  consistent six-node surface-mass solve. The default `mass="auto"` selects the
  valid method; explicit P2 lumping and non-triangular 3D traces raise instead
  of returning a non-pointwise reaction scaling. Strict MPI invariance of a
  vector normal projection requires an analytic normal; geometric facet-normal
  seam sensitivity is unchanged (#404).
- Recorded as the preferred free-slip BC in the project guidance (#300);
  conda PETSc floor raised to ≥ 3.25 for FMG/rotation API consistency (#304).
- `uw.postprocessing.geoid` provides generic spherical-shell geoid and
  self-gravity coefficient functions. Its rotated-Stokes adapter projects the
  existing boundary traction onto an axisymmetric harmonic; the pure functions
  also accept coefficients recovered by other methods and an optional internal
  load.
- Rotated free slip exposes
  `Stokes.boundary_normal_traction_integral(boundary, fn)` for a distributed
  weak contraction of the assembled normal reaction. Cylindrical-annulus
  Stokes responses use this fitted integral and its matching finite-element
  boundary norm instead of gathering pointwise samples for angular quadrature.
- The spherical-shell geoid adapter accepts `projection="reaction"` to use
  the same fitted integral without pointwise P2 recovery or a rank-zero
  surface triangulation; `projection="centroid"` remains the default.
- `uw.analytic.Zhong2008` implements the Hager--O'Connell propagator-matrix
  oracle used for the Zhong et al. spherical-shell response benchmark. It
  supports piecewise-constant radial viscosity and reproduces every analytical
  response printed in Zhong Tables 2 and 3; geoid and self-gravity are delegated
  to the generic postprocessing functions above.

### Generalized Geometric Multigrid via Custom Prolongation (July 2026)

**Geometric multigrid decoupled from the nested `refine()` hierarchy**: custom
(barycentric or RBF) prolongation operators drive PCMG across independent —
possibly non-nested — coarse/fine mesh pairs, with coarse operators assembled
by Galerkin RAP (#290).

- Ties native FMG iteration-for-iteration on nested hierarchies while
  supporting graded and adapted meshes that native FMG cannot nest.
- Native geometric FMG is locked out for single-field solvers where its DM
  assumptions are invalid (#297).

### Consistent Jacobian Tangent — Opt-In Newton (July 2026)

**New opt-in `solver.consistent_jacobian`** (default `False` keeps the
bit-identical Picard tangent) fixes the unwrap-before-differentiate order so
the assembled Jacobian sees the strain-rate dependence of nonlinear
viscosities, plus a `"continuation"` mode that stages Picard → Newton for
robustness far from the solution (#258).

### Swarm Correctness: Stale Caches and Parallel Checkpoint Restore (July 2026)

**Swarm data-pipeline hardening across serial and parallel paths.**

- Three stale-cache bugs after swarm particle addition fixed (#216), followed
  by the campaign's Track-0 batch: cache invalidation ahead of the migrate()
  early return, migration-suppression semantics, empty-rank KDTree guards,
  and a pre-solve proxy refresh (#313).
- Parallel `read_timestep` restores each particle exactly once via a
  rank-0-routed read (#329); reduction return types aligned to the
  MeshVariable per-component contract, `NodalPointSwarm` deprecated, and the
  never-functional `recycle_rate > 1` machinery excised (#323).

### NumPy 2 and Environment Support (July 2026)

**NumPy ≥ 2.0 supported** (#301), with a follow-up fix for 2-D `np.cross` on
`UnitAwareArray` under numpy 2 (#305). The repository now ships its Claude
Code skills for AI-assisted development sessions (#299). `UWQuantity` handles
offset temperature units (degC/degF) correctly (#295), and boundary rebuilds
avoid a PETSc IS size query that could abort on empty strata (#288).

---

## 2026 Q2 (April – June)

### Mesh Adaptation: Metric-Driven Movers and MMPDE Robustness (May – June 2026)

**A family of metric-driven mesh-adaptation movers** landed and hardened:
`smooth_mesh_interior` (Winslow Jacobi smoother, #190), `follow_metric` with
optimal-transport movers and `mesh.OT_adapt()` (#209), and anisotropic movers
with mesh-owned boundary tangent-slip and MPI robustness (#228).

- Parallel adaptive "seam-spike" fixed: mover heap corruption, point-locator
  hardening, and a redesigned remesh field transfer (#213).
- MMPDE metric hardening: SPD floor stops silent NaN-bail on deformed meshes
  (#259); monotone RBF metric bake from nodal values (#266).
- Deformed-mesh correctness: boundary normals and domain-membership tests
  track mesh deformation (#264); `on_boundary` acceptance for on-face point
  queries (#207); gmsh `spacedim` no longer leaks across imports (#238).

### Moving-Mesh Field Transfer and deform() (June 2026)

**Mesh coordinate mutation made foolproof**: a capability gate plus the
public `mesh.deform()` entry point, with semi-Lagrangian CARRY field transfer
across deformation (#246, locked by regression test in #249).

- Old-frame semi-Lagrangian reach-back for moving meshes (SLCN / SL-BDF2)
  traces advected histories in the pre-deformation frame (#251).
- Evaluate / DMInterp / topology caches are invalidated on mesh deformation
  (#188).

### Semi-Lagrangian Accuracy and Timestep Controls (May 2026)

**Fixed the semi-Lagrangian trace-back FE overshoot and added a monotone
limiter** to the DDt schemes (#186), exposed as `monotone_mode` on
`AdvDiffusionSLCN` (#189) and promoted to a universal evaluator flag (#208).

- `theta` exposed on the SemiLagrangian DDt for backward-Euler /
  Crank-Nicolson selection (#187).
- Tensor evaluate path in `_project_to_work_variable` (#185); NavierStokes
  SLCN projection shape mismatch fixed (#183); vector DMInterpolation
  overshoot at cell-shared boundaries fixed (#164).
- `estimate_dt` gained opt-in median/percentile cell reduction for
  sliver-robust timesteps (#220).

### Snapshot and Checkpoint Toolkit (May 2026)

**A snapshot toolkit — "git stash for timesteps"**: in-memory snapshots
(#195), `Model.tracker` for snapshot-managed run state (#196), and an on-disk
snapshot format v1.1 with a metadata wrapper around PETSc bulk data (#198,
docs in #199). PETSc DMPlex checkpoint reload for mesh variables landed as
the underlying primitive (#146, T. Gollapalli).

### Stokes_Constrained: Multiplier Free-Slip and Parallel Correctness (June 2026)

**In-saddle Lagrange-multiplier free-slip with surface topography recovery**
in `Stokes_Constrained` (#224), then made parallel-correct.

- `selfp` Schur preconditioner default, viscosity-scaled penalty, and
  nullspace re-setup fix (#229); over-conservative serial guard removed
  (#240); gauge, convergence, knockout, and rotation-gauge fixes (#265).
- The main `Stokes.penalty` (augmented-Lagrangian grad-div) is likewise
  viscosity-scaled since June 2026: the parameter is now a dimensionless
  O(1) number, not a large constant tuned against the viscosity magnitude.
  Migration note for older scripts in `docs/advanced/troubleshooting.md`
  (#292).

### Boundary Conditions: Local-h Nitsche and Boundary-Slip Surfaces (June 2026)

**Nitsche penalty scaled by local per-cell mesh size** rather than the global
minimum radius, restoring correct stiffness on graded and adapted meshes
(#275).

- `mesh.boundary_slip` API with `BoundingSurface` objects for boundary
  tangent-slip (#225); `Surface.influence_function` respects finite edges
  (#241).

### Units System: Quantity Interoperability and the ND Boundary Contract (June 2026)

**UWQuantity operands now work across the API surface**: MeshVariable
arithmetic (#283), the Stokes bodyforce setter (#284), and units-active
semi-Lagrangian trace-back (#277).

- The non-dimensional ↔ units boundary contract is documented as a design
  contract (#278).
- Tutorials and examples repaired for strict units: thermal convection
  (#263), dimensionality demo (#261), unit-aware coordinate evaluation (#262).

### Memory, Evaluation and Solver Infrastructure (May – June 2026)

**Comprehensive memory-leak fixes** in solver setup, interpolation caching,
and SubDM synchronisation (#178), Cython deallocation and callback hardening
(#181), and a `memprobe` diagnostic module (#179); cached spatial indexing
(KDTree) consolidated (#182).

- `global_evaluate`: faithful parallel `evaluate()` fixing out-of-domain
  mislocation (#222); swarm particle loss across rank boundaries during
  advection fixed (#177); empty-partition reshape crash in parallel
  `read_timestep` fixed (#221).
- Manifold-mesh PDE support with `Mesh.extract_surface` for solving on
  embedded surfaces (#237).
- Exponential time-differencing VE/VEP integrators, ETD-1 default (#161);
  per-iteration SNES update callbacks with pressure gauge and
  boundary-correct scatter (#250); SolCx analytic solution ported as
  `uw.function.analytic.SolCx` with its exact stress tensor (#223, #226);
  projection gained unit-aware `smoothing_length` (#234) and an opt-in
  `linear_solver()` (#281).
- XDMF output moved to PETSc-native topology with explicit cell-to-vertex
  connectivity for ParaView (#218, #205); Gadi Singularity container build
  files (#133); maturity-gated release tooling `./uw dev` (#233); `-uw_*`
  CLI overrides applied on all platforms (#280).

### DDt.set_initial_history — Public API for BDF Restart (April 2026)

**New `set_initial_history(values, dt=...)` method on `SemiLagrangian` and
`Eulerian` DDt classes** to plant BDF history at the start of a run.
Two use cases:

- **Analytical IC for benchmarks** — populate ψ* from a known closed-form
  solution so the very first solve runs at full BDF order with no startup
  transient. The `bench_ve_harmonic.py` peak-start benchmark used the manual
  pattern (poking four private attributes including `psi_star[k].array`,
  `_n_solves_completed`, `_dt_history`); the new API wraps that cleanly.
- **Checkpoint restart** — resume a multistep history from disk without
  re-ramping `effective_order` from BDF-1 over the first `order` steps.

Sets `psi_star[0..order-1].array`, marks history initialised,
seeds `_dt_history` for variable-dt BDF coefficients, and warns when
`order >= 2` is called without `dt`. Six unit tests cover bookkeeping,
scalar broadcast, length validation, and the warning path.

**Files**: `src/underworld3/systems/ddt.py`,
`docs/advanced/benchmarks/bench_ve_harmonic.py`,
`tests/test_1052_ddt_set_initial_history.py`,
`docs/api/systems_ddt.md`.

### Multi-Component Projection Solver (April 2026)

**New `SNES_MultiComponent_Projection` solver** that projects N scalar components in a single PETSc SNES solve sharing one DM, replacing the per-component cycling in `SNES_Tensor_Projection` (which tore down and rebuilt the DM on each inner iteration). The underlying `SNES_MultiComponent` Cython base decouples the FE component count from `mesh.dim` — PETSc's pointwise callback interface accepts any DOF count per node; the new class exposes that directly.

- Wired into `SNES_VE_Stokes` via `_setup_tau_projection` for the symmetric-tensor tau projection (Nc=3 in 2D, Nc=6 in 3D). User-facing tau variable remains a `SYM_TENSOR` so downstream `.array[:, i, j]` reads are unchanged; a flat `(1, Nc)` MATRIX drives the actual solve and results fan out after each solve.
- DM build count scales with outer solves rather than `Nc × outer_solves` — the dominant cost in `SNES_Tensor_Projection` on the VE square-wave benchmark.
- 10 validation tests: `Nc=1` agrees with `SNES_Projection`, `Nc=3` symmetric-tensor agrees with `SNES_Tensor_Projection`, `Nc=4` full-tensor agreement, DM-rebuild count invariant, smoothing-parametrised agreement (1e-4, 1e-2, 1.0).

**Files**: `cython/petsc_generic_snes_solvers.pyx` (new `SNES_MultiComponent` class, VE tau wiring), `systems/solvers.py` (new `SNES_MultiComponent_Projection`), `systems/__init__.py` (export), `tests/test_multicomponent_projection.py`.

### PETSc Pointwise Jacobian Layout Fix (April 2026)

**Documented PETSc's `[fc, gc, df, dg]` flat-index convention** for pointwise Jacobian arrays and fixed a latent layout bug in `SNES_Vector`. The `SNES_Vector` permutations `(0, 3, 1, 2)` for g3 and `(2, 1, 0)` for g1/g2 did not match PETSc's element-assembly index order (fe.c:2639–2790) — the bug was hidden by the trial-side symmetry of every in-repo consumer's F1 (strain-rate-based smoothing, deviatoric Stokes stress, divergence penalty).

- Migrated `SNES_Vector._setup_pointwise_functions` (main residual and natural-BC Jacobian paths) from `derive_by_array` + `permutedims` to explicit nested-loop construction that writes directly into PETSc's expected row-major 2D layout. Same pattern as the new `SNES_MultiComponent`.
- Regression test with `F1 = smoothing * Unknowns.L` (raw gradient, not symmetrised) guards against the layout bug returning: at `smoothing > 0`, identical targets must give identical components, and results must match `SNES_MultiComponent_Projection` to rel-L2 ≤ 1e-8.
- Audit of other solvers: `SNES_Scalar` trivially correct (Nc=1); `SNES_Stokes_SaddlePt` and `SNES_NavierStokes` already use the correct `(0, 2, 1, 3)` permutation.
- New developer documentation: `docs/developer/subsystems/petsc-jacobian-layout.md` captures the convention, the sympy-to-PETSc axis mapping, and a checklist for new solvers (identical-targets + non-zero-smoothing validation tests required).

**Files**: `cython/petsc_generic_snes_solvers.pyx` (`SNES_Vector` migration), `tests/test_snes_vector_asymmetric_jacobian.py`, `docs/developer/subsystems/petsc-jacobian-layout.md`, `docs/developer/index.md` (toctree).

---

## 2026 Q1 (January – March)

### v3.0.0 Release (March 2026)

**Underworld3 v3.0.0 released**: Merged 398 commits from development to main. Major release incorporating 18 months of work since the JOSS v0.99 publication, including units system overhaul, symbol disambiguation, boundary integrals, mathematical mixin, platform-conditional MPI, and comprehensive CI/CD automation.

- Tagged `v0.99` at previous main HEAD (pixi-compatible JOSS snapshot) for binder compatibility
- Deleted obsolete `uw3-release-candidate` branch
- Cleaned up 10 merged feature/bugfix branches

### Binder Infrastructure Overhaul (March 2026)

**Versioned binder links** with full CI automation for tag-based releases.

- Four launcher branches: `v0.99`, `v3.0.0`, `main`, `development` — each with frozen Dockerfile
- CI workflow handles `v*` tag builds with automatic launcher branch creation via `repository_dispatch`
- Manual dispatch overrides (`uw3_branch`, `image_tag`) for building images from old tags
- Dockerfile made version-resilient: versioned lib subdirectories (vtk-X.Y, openvino-X.Y.Z) use wildcards instead of hardcoded paths
- Launcher dispatch payload fixed: field names now match target workflow (`branch`/`ref_type`)
- README badges updated with three versioned binder launch links

**Files**: `binder-image.yml`, `Dockerfile.base.optimized`, `binder_wizard.py`, `containers.md`

### Checkpoint XDMF Fix (March 2026)

**`petsc_save_checkpoint()` now uses modern XDMF output** (fixes #80). Previously used legacy `generateXdmf()` which missed vertex/cell compatibility groups, field projection (P2→P1), and tensor repacking for ParaView.

- Refactored as thin wrapper around `write_timestep()` — single checkpoint code path
- Output file layout changes from single HDF5 to per-variable files (consistent with `write_timestep()`)

**Files**: `discretisation_mesh.py`

### Boundary Integral Support (March 2026)

**New `uw.maths.BdIntegral` class** for boundary and surface integrals (closes #47). Wraps PETSc's `DMPlexComputeBdIntegral` with MPI Allreduce and units support. Works on external boundaries and internal boundaries (e.g. `AnnulusInternalBoundary`). Integrands can reference the outward unit normal via `mesh.Gamma`.

- PETSc patch (`plexfem-internal-boundary-ownership-fix.patch`): fixes ghost facet ownership and part-consistent assembly in boundary residual, integral, and Jacobian paths. Resolves rank-dependent L2 norms for internal boundary natural BCs (fixes #77). Contributed by gthyagi.
- C wrapper simplified: ghost filtering delegated to PETSc patch, wrapper retains MPI Allreduce only
- 20 tests across external/internal boundaries, normal vectors, mesh variables
- MPI regression test for internal boundary circumference

**Files**: `petsc_compat.h`, `petsc_maths.pyx`, `petsc_extras.pxi`, `maths/__init__.py`, `petsc-custom/patches/`

### Binder Image Fix (March 2026)

**Fixed Dockerfile building from stale branch** (fixes #71). The binder Dockerfile hardcoded `uw3-release-candidate` as the clone branch, but the CI workflow triggers on `main` and `development` pushes. The image was missing recent dependencies (e.g. `python-xxhash`).

- Dockerfile now uses `ARG UW3_BRANCH=development` instead of hardcoded branch
- CI workflow passes the triggering branch name via `--build-arg`
- Binder wizard script default updated to `development`

### Worktree Symlink Safety (March 2026)

**Prevented worktree symlinks from being accidentally committed**. The `./uw worktree create` command creates `.pixi` and `petsc-custom/petsc` symlinks that could be picked up by `git add -A`, breaking CI.

- `.gitignore` patterns now match both directories and symlinks (removed trailing `/`)
- `./uw worktree create` writes exclusions to the worktree's `.git/info/exclude`

### MeshVariable Data Cache Bug Fix (February 2026)

**Self-validating `.data` cache**: Fixed a critical bug where the `.data` property could return stale (zero) values after PETSc DM rebuilds. When new MeshVariables are added to a mesh, PETSc requires a new DM — destroying and recreating all existing variables' local vectors (`_lvec`). The cached `_canonical_data` array (a NumPy view into the old `_lvec`) would silently read freed memory, returning zeros even though the solver correctly wrote results to the new vector.

- Root cause: Early `.data` access cached a view that became invalid after DM rebuild
- Fix: `.data` property now tracks `id(self._lvec)` and auto-rebuilds when stale
- Self-healing design: no code path that replaces `_lvec` needs to manually invalidate the cache
- Eager invalidation in DM rebuild loop and `mesh.adapt()` preserved as performance optimization

**Files**: `discretisation_mesh_variables.py` (`.data` property), `discretisation_mesh.py` (`mesh.adapt()`)

### Binder/Docker CI Automation (January 2026)

**Automated container build pipeline**: Implemented full GitHub Actions automation for Docker image builds and mybinder.org integration.

- **Binder images** (`binder-image.yml`): Builds to GHCR on push to main/development
  - Triggers on Dockerfile, pixi.toml, Cython, or setup.py changes
  - Pushes to `ghcr.io/underworldcode/uw3-base:<branch>-slim`
  - Cross-repo dispatch updates launcher repository automatically

- **Command-line images** (`docker-image.yml`): Separate workflow for GHCR (micromamba-based)

- **Launcher auto-update**: `uw3-binder-launcher` receives `repository_dispatch` events and updates its Dockerfile reference automatically

- **Container consolidation**: All container files now in `container/` directory with comprehensive README

**Key infrastructure**:
- `LAUNCHER_PAT` secret enables cross-repo communication
- Branch-specific image tags enable testing different versions
- nbgitpuller allows any repository to use pre-built images

**Documentation**: `docs/developer/subsystems/containers.md` — comprehensive guide covering both binder and command-line container strategies.

---

## 2025 Q4 (October – December)

### Symbol Disambiguation (December 2025)

**Clean multi-mesh symbol identity**: Replaced the invisible whitespace hack (`\hspace{}`) with SymPy-native symbol disambiguation using `_uw_id` in `_hashable_content()`. This follows the `sympy.Dummy` pattern.

- Variables on different meshes with same name are now symbolically distinct without display name pollution
- Clean LaTeX rendering — no more invisible whitespace artifacts
- Proper serialization/pickling support
- Coordinate symbols (`mesh.N.x`, etc.) also isolated per-mesh to prevent cache pollution bugs

**Key technical insight**: SymPy's `Symbol.__new__` has an internal cache that runs *before* `_hashable_content()`. Solution: use `Symbol.__xnew__()` to bypass the cache, same as `sympy.Dummy` does.

**Expression rename capability**: Added `UWexpression.rename()` method to customize display names without changing symbolic identity. Uses SymPy's custom printing protocol (`_latex()`, `_sympystr()`) to separate display from identity. Useful for multi-material models where parameters need distinct LaTeX labels:
```python
viscosity.rename(r"\eta_{\mathrm{mantle}}")  # Custom LaTeX display
```

**Files**: `expressions.py`, `_function.pyx`, `discretisation_mesh_variables.py`
**Design doc**: `docs/developer/design/SYMBOL_DISAMBIGUATION_2025-12.md`

### Units System Overhaul (November 2025)

**Gateway pattern implementation**: Units are now handled at system boundaries (user input/output) rather than propagating through internal symbolic operations. This eliminates unit-related errors during solver execution while preserving dimensional correctness for users.

- `UWQuantity` provides lightweight Pint-backed quantities
- `UWexpression` wraps symbolic expressions with lazy unit evaluation
- Linear algebra dimensional analysis replaces fragile pattern-matching
- Proper non-dimensional scaling throughout advection-diffusion solvers
- **Pint-only arithmetic policy**: All unit conversions delegated to Pint — no manual fallbacks that could lose scale factors

**Key fixes:**
- `delta_t` setter correctly converts between unit systems (Pint's `.to_reduced_units()`)
- `estimate_dt()` properly non-dimensionalizes diffusivity parameters
- Data cache invalidation after PETSc solves (buffer pointer changes)
- JIT compilation unwrapping respects `keep_constants` parameter
- Subtraction chain unit propagation fixed (chained operations preserve correct units)

### Automatic Expression Optimisation (November 2025)

**Lambdification for pure sympy expressions**: `uw.function.evaluate()` now automatically detects pure sympy expressions (no UW3 MeshVariables) and uses cached lambdified functions for dramatic performance improvements.

- 10,000x+ speedups for analytical solutions — no code changes required
- Automatic detection: UW3 variables use RBF interpolation, pure sympy uses lambdify
- Cached compilation: repeated evaluations reuse compiled functions
- Transparent fallback: mixed expressions still work correctly

### Timing System (November 2025)

**Unified PETSc timing integration**: Refactored timing system to route all profiling through PETSc's event system, eliminating environment variable complexity.

- `uw.timing.start()` / `uw.timing.print_summary()` API for simple profiling
- Filters PETSc internals to show only UW3-relevant operations
- Now Jupyter-friendly — no environment variables needed
- Programmatic access via `uw.timing.get_summary()`

### Solver Robustness (November 2025)

**Quad mesh boundary interpolation**: Fixed Semi-Lagrangian advection scheme failing on `StructuredQuadBox` meshes. The point location algorithm was receiving coordinates exactly on element boundaries. Solution: use pre-computed centroid-shifted coordinates for evaluation.

### Test Infrastructure (November 2025)

- Strict units mode enforcement in test collection
- All advection-diffusion tests now pass across mesh types (StructuredQuadBox, UnstructuredSimplex regular/irregular)
- **Dual test classification system**: Levels (0000-9999 complexity prefixes) + Tiers (A/B/C reliability markers)
  - Tier A: Production-ready, trusted for TDD
  - Tier B: Validated but recent, use with caution
  - Tier C: Experimental, development only

### Build System & Developer Experience (December 2025)

**`./uw` wrapper script**: Unified command-line interface for all underworld3 operations. Replaces fragmented pixi/mamba instructions with a single entry point.

- `./uw setup` — Interactive wizard installs pixi, configures environment, builds underworld3
- `./uw build` — Smart rebuild with automatic dependency chain handling
- `./uw test` / `./uw test-all` — Tiered test execution
- `./uw doctor` — Diagnoses configuration issues (PETSc mismatches, missing deps)
- `./uw status` — Check for updates on GitHub without pulling
- `./uw update` — Pull latest changes and rebuild

**Documentation overhaul**: Rewrote installation docs to focus on `./uw` workflow. The 3-line install now appears on the landing page. Removed outdated mamba/conda instructions; Docker and system PETSc kept as alternatives for specific use cases.

### Documentation & Planning (November 2025)

- Reorganised `planning/` → `docs/developer/design/` to distinguish from strategic planning
- Hub-spoke planning system integration for cross-project coordination
- This changelog established for quarterly reporting

---

## Format Guide

Each quarter should capture:

1. **Major features or capabilities** — What can users do now that they couldn't before?
2. **Architectural improvements** — What's better about the system design?
3. **Significant bug fixes** — Only those affecting correctness of results
4. **Infrastructure changes** — Testing, documentation, build system

Keep entries conceptual. Technical details belong in design documents or commit messages.
