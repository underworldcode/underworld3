# Curved-wall Stokes: upstream integration review

Date: 2026-11-10. Prepared by the Underworld development team with AI support from Codex; maintainer review remains outstanding.

## Overview

We merged upstream development into feature/cylindrical-wall-boundaries_sph, preserving upstream solver contracts and the branch's pressure/wall corrections. Local-only development toolchain commits are excluded from this PR. Numerical tests use a regular installed Linux/arm64 PETSc 3.25.4 int32 package, never an editable source import. The upstream FAC-only Python delta was tested separately. The later merge through `a6eebb61` adds a Richards docstring clarification and two timestep regressions; its library AST is identical after removing docstrings. Compiled extension hashes are preserved.

## Changes Made

- Six initial merge conflicts: resolved shared-base versus Stokes-only methods without restoring duplicate subclass implementations; retained upstream closure tables, batch sharding, residual-only final iterations, vector dispatch, adjoint/cache lifetime and in-place interface operator refresh.
- CI reach: the spherical pressure/norm/diagnostic files lacked upstream batch coverage; added explicit globs. All 440 test files are reachable; 22 unrelated ptests retain upstream documented deferrals.
- Removed an obsolete rotated-solver exception allowance; the deprecated-pattern gate gains no new exemptions.
- Analytical gravity kernels lacked native CI coverage: added 28 independent shell-theorem, Poisson-jump, finite-boundary and radial-quadrature checks. All pass. An in-memory +1% sheet normalization mutant fails all four selected Newton-shell cases, confirming sensitivity; production source stays unchanged. These test analytic reference kernels, not numerical gravity integration.
- Pressure-equation loss: test_1018_rotated_pressure_nullspace rejects a false constant-pressure gauge on both left and right; viscosity cannot mask the pressure scale. Coarse-correction tests compare explicit nonsymmetric actions and detect stale state without dropping a physical equation.
- Geometry/assembly: fitted-side topology, true-sphere sheet work, cut cells, curtains and physical wall reactions have independent mathematical oracles. Unsupported physical P2 rows are completed only in the background norm; tests compare the no-void operator bitwise and void support against an explicit patch.
- Finalized-field residuals are recomputed against the original equations; observer callbacks and metrics borrow objects without replacing equations. Serial tests include nonzero flow against an independent LU reference and repeated solves.
- Parallel attacks cover rank-local coordinate identity/halo refresh, empty wall ownership, deterministic JIT, pressure/coarse collectives and distributed geometry. All pass; counts are recorded below.

- Upstream test contracts: the adjoint test now supplies measured pressure verification; the deliberately faulty bisector control retains its algebraic defect check when a non-null rotation is explicitly refused. The cache fixture now asserts actual initial convergence before automatic warming. Its old slid wall had finalized residual3.8345e-8 above target1.9953e-11 and correctly started cold; no solver guard is weakened. The closure-table oracle now checks global non-emptiness and exact empty local tables on ranks with no cells.

### PR #1033 CI repair audit

The first full CI run (`42753082647`, head `7e4d3f71`) found 30 distinct failing cases. The earlier targeted counts below were not full-suite acceptance. Follow-up repairs address the failures rather than weakening PDE or solver tolerances:

- Coordinate DM clones retain the embedding dimension. Independent surface P1/P2/DG0 coordinate oracles failed before the repair; manifold areas, Killing fields, membrane motion and surface transport then passed. Affine basis coordinates reject substantial bounding-box excursions collectively and clip only excursions within 64 machine epsilons, fixing PETSc point-location failures.
- Pressure-response pseudoinverses project a copied inner RHS only in verified rigid gauges, preserving the original coarse equations. The native adapter releases its velocity-KSP wrapper on refresh/destruction, fixing the MPI multigrid-retention regression.
- Gauge candidates must be left and right kernels of the actual normal-constrained operator. A declared geometric rotation does not waive this test. Updated rotation/projection tests independently recompute the finalized original equations; exact analytic-normal rotations retain positive controls.
- Fault-pair sheet data use the registered boundary name, not the numeric datum, when scheduling collective evaluation. Both copies now carry the prescribed motion. Reactions use the same surface conormal as the constraint; the original membrane-resultant tolerance passes.
- Direct LU selects one globally agreed pressure row, including ranks with no pressure DOFs. The old per-rank pins removed extra mass equations; the existing two-rank LU/multigrid slip comparison now passes without changing its tolerance.
- Weak-pressure correction is a preconditioner policy, separate from gauge verification. Responses below the requested relative residual scale are left to the original fieldsplit/FGMRES solve instead of amplified by an exact coarse inverse. Original A/RHS, pressure-gauge rejection and finalized residual targets are unchanged. The nonlinear radial-datum reproducer changes from a pressure offset near -7.7e8 and an unconverged residual to a converged original residual of 3.00e-10; the historical annulus accuracy anchor is unchanged.
- Kramer is included in the analytical sweep; its body-force sign mutation uses a nonzero smooth source. Collective context checks execute outside rank-local guards. XDMF midpoint checks allow only machine roundoff at zero; the FAC oracle checks its mathematical three-quarter coverage rule. Parallel deformation fixtures select a coordinate with an untouched-rank witness instead of assuming one partitioner layout.

### Second CI repair audit

Run `42782858215` at `f8927628` failed 11 distinct cases (MPI duplicates counted once), across three shards plus the aggregate check. The follow-up fixes preserve solver tolerances and existing physical reference values:

- Seven scalar/SUPG/builder cases share one branch regression: `_get_coords_for_basis` now needs the coordinate degree, but `_DMLevelView` did not supply it. That exception prevented transfer construction and forced GAMG. The level view now carries the same geometry-degree metadata as its source mesh. Requested scalar FMG and its expected collective retry/degrade warning paths pass.
- Three rotated tests require an exact rotation kernel. Their analytic radial-normal fixtures now supply that kernel; the pressure gauge is still admitted only after actual left/right verification. The adjoint test checks both zero kernel actions against a matrix-scaled rounding bound, retaining its absolute bound, instead of demanding one-percent agreement between two unrelated rounding errors. Parallel LU and coarse-GAMG reference values/tolerances are unchanged.
- The Cartesian transport test queried a whole-domain profile with rank-local evaluation. In MPI2 this already gives an initial interpolation error near one. Owner-aware `global_evaluate` gives initial error 0.03495 and final error 0.01598 against the unchanged 0.05 tolerance. A separate exactly representable P2 polynomial regression checks interior profile queries to rounding precision and fails with local evaluation. Existing expected-failure annotations are not expanded. Physical-boundary query nudges are outside this ownership regression; PR #859 addresses locator-position behavior separately.
- Upstream advanced to `abe5e0ca`; merged #1036 (multigrid lifecycle), #959 (mesh-move field refresh), and #1035 (example parse checks), without conflicts. One incremental native rebuild includes their Cython lifecycle hooks. Open adjoint PRs #1038/#1040 were reviewed for overlap; their production adjoint guards are not duplicated here.

Focused validation: 33 serial adjoint/scalar/SUPG checks pass; five rotated/builder checks pass on each MPI2 rank; 14 upstream lifecycle/mesh-move checks pass serially and seven pass on each MPI2 rank. The transport profile and exact polynomial checks pass on each MPI2 rank (three passes plus the existing mesh0 XPASS). The deprecated-pattern scan and all 316 Python examples/27 notebooks pass. Fresh full GitHub CI remains required; these targeted runs do not claim complete CI acceptance.

## System Architecture

Public Stokes methods delegate to shared private state, observation and field-layout helpers, with distinct cylindrical/spherical geometry and algebra modules. Native diagnostics borrow solved state; they do not replace equations.

## Testing Instructions

- Pre-CI baseline: 352 unique serial checks passed; 13 skipped (12 optional assess comparisons and one x86-only OpenBLAS-kernel probe on arm64). This did not substitute for full CI.
- MPI8: 191 unique checks passed on every rank; two serial-only direct-LU datum checks skipped on every rank.
- MPI3: 13 unique checks passed on every rank, covering empty wall ownership/JIT, tolerance/cache retention, upstream FAC blocks, and Stokes/ThinSheet interface operator refresh.
- All 440 test files are reachable from scripts/test.sh; deprecated-pattern scan, Python compilation, shell syntax and git diff whitespace checks pass.
- CI-repair numerical head `b57a4cdc13a0`: regular installed int32 package; 52 passed/one x86-only skip in the focused serial pressure/options/projection gate, 19 passed on every MPI2 rank in the original failing pressure/datum/LU/cleanup gate, and 17 passed on every MPI2 rank in the pressure-gauge/LU lifecycle gate. Twelve native objects remain byte-identical to the fresh `f063f9f58de5` build; Python-only repairs required no native recompilation.
- Additional repair evidence: 25 serial rotation-gauge checks pass; nine XDMF checks and the DMField point-location reproducer pass. Full surface transport and the 192-step Maxwell tide passed after the embedding-coordinate repair; those long runs are not represented as reruns of later fault-only/preconditioner edits. Final short surface checks pass all nine selected cases. The geometry/rotation gate passes 41 cases; its old singular-coarse fixture required an analytic normal to supply the true kernel it claims to test. That corrected fixture passes its unchanged SVD-value tolerance. No full-CI result is inferred from this evidence.
- NCI explicitly reports scheduled maintenance; no fresh merged-head Gadi/int64 result is claimed.

Twelve optional assess comparisons are skipped when assess is unavailable; independent native analytical tests run. Earlier 336 benchmark records remain pinned to their original numerical versions and are not represented as reruns of this merge.

## Known Limitations

We make no new issue-closure claims from this integration. Upstream fixes for #920/#924 (caps/tolerances), #907 (failure reporting), #963 (exact-mode projection), #894 (operator refresh), #936 (constrained reaction memory), and #1007 (cleanup) are preserved, with regression evidence where noted; that does not substitute for a development-only issue-closing probe.

The following remain open or unverified by this PR, rather than being inferred fixed:

- #348/#899: the global-datum regression and existing two-rank split-fault LU/multigrid comparison now pass. This is targeted regression evidence, not an issue-closing claim for every direct-LU configuration.
- #910/#911/#985/#625/#755: broader cached/roundoff/Schur-cost and viscoelastic-preconditioner issues; explicit rotated inner options do not claim to solve them generally.
- #98/#608/#939/#931/#923/#962: other pressure, corner and host/partition-specific wall/topography failures; Kramer/coupled-wall component checks are not those reproducers.
- #883/#884: wider cold datum/reference and gauge-reference issues; retained targeted regression contracts do not establish every mixed-forcing/continuation case.
- #781/#769/#844/#817/#759: adjoint-target, nonlinear sensitivity, omitted-pressure constructor, constrained warm-up and zero-viscosity behavior outside the supported constant-viscosity P2/P1 methods.
- #876/#877/#887/#950/#624/#892/#629/#1010: unrelated ThinSheet API/units/documentation, option lifetime, default Krylov/FAC and hierarchy/feature integration claims are not closed here.
- #157: direct spherical diagnostics avoid generic tensor projection; the generic projector's memory behavior is not fixed.

Uniform mixed/multiplier inf-sup stability is unproven; the new coupled methods remain opt-in with explicit supported-space/geometry restrictions. Numerical topography/gravity integration is outside this change. The explicit experimental gap methods do not claim adjoint or automatic Model.step transcript integration; the ordinary solver description/adjoint contracts are tested separately. Fresh merged-head Gadi/int64 validation is unavailable during NCI maintenance; earlier int64 evidence is not relabelled as current. Merge should await required GitHub CI and maintainer review, with int64 validation explicitly tracked if required for release.

## Sign-Off

| Role | Name | Date | Status |
| --- | --- | --- | --- |
| Automated integration review | Codex | 2026-11-10 | Component/MPI review completed; limits above |
| Maintainer review | Pending | — | Pending; no approval claimed |
