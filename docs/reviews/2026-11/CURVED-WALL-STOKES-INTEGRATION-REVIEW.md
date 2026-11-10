# Curved-wall Stokes: upstream integration review

Date: 2026-11-10. Prepared by the Underworld development team with AI support from Codex; maintainer review remains outstanding.

## Scope and provenance

We merged upstream development into feature/cylindrical-wall-boundaries_sph, preserving upstream solver contracts and the branch's pressure/wall corrections. Local-only development toolchain commits are excluded from this PR. Numerical tests use a regular installed Linux/arm64 PETSc 3.25.4 int32 package, never an editable source import. The subsequent upstream FAC-only Python delta is tested separately; compiled extension hashes are preserved.

## Findings and attacks

- Six initial merge conflicts: resolved shared-base versus Stokes-only methods without restoring duplicate subclass implementations; retained upstream closure tables, batch sharding, residual-only final iterations, vector dispatch, adjoint/cache lifetime and in-place interface operator refresh.
- CI reach: the spherical pressure/norm/diagnostic files lacked upstream batch coverage; added explicit globs. All 440 test files are reachable; 22 unrelated ptests retain upstream documented deferrals.
- Removed an obsolete rotated-solver exception allowance; the deprecated-pattern gate gains no new exemptions.
- Analytical gravity kernels lacked native CI coverage: added 28 independent shell-theorem, Poisson-jump, finite-boundary and radial-quadrature checks. All pass. An in-memory +1% sheet normalization mutant fails all four selected Newton-shell cases, confirming sensitivity; production source stays unchanged. These test analytic reference kernels, not numerical gravity integration.
- Pressure-equation loss: test_1018_rotated_pressure_nullspace rejects a false constant-pressure gauge on both left and right; viscosity cannot mask the pressure scale. Coarse-correction tests compare explicit nonsymmetric actions and detect stale state without dropping a physical equation.
- Geometry/assembly: fitted-side topology, true-sphere sheet work, cut cells, curtains and physical wall reactions have independent mathematical oracles. Unsupported physical P2 rows are completed only in the background norm; tests compare the no-void operator bitwise and void support against an explicit patch.
- Finalized-field residuals are recomputed against the original equations; observer callbacks and metrics borrow objects without replacing equations. Serial tests include nonzero flow against an independent LU reference and repeated solves.
- Parallel attacks cover rank-local coordinate identity/halo refresh, empty wall ownership, deterministic JIT, pressure/coarse collectives and distributed geometry. All pass; counts are recorded below.

- Upstream test contracts: the adjoint test now supplies measured pressure verification; the deliberately faulty bisector control retains its algebraic defect check when a non-null rotation is explicitly refused. The cache fixture now asserts actual initial convergence before automatic warming. Its old slid wall had finalized residual3.8345e-8 above target1.9953e-11 and correctly started cold; no solver guard is weakened. The closure-table oracle now checks global non-emptiness and exact empty local tables on ranks with no cells.

## Validation

- Serial: 352 unique checks passed; 13 skipped (12 optional assess comparisons and one x86-only OpenBLAS-kernel probe on arm64).
- MPI8: 191 unique checks passed on every rank; two serial-only direct-LU datum checks skipped on every rank.
- MPI3: 13 unique checks passed on every rank, covering empty wall ownership/JIT, tolerance/cache retention, upstream FAC blocks, and Stokes/ThinSheet interface operator refresh.
- All 440 test files are reachable from scripts/test.sh; deprecated-pattern scan, Python compilation, shell syntax and git diff whitespace checks pass.
- The final src tree is identical to tested numerical source51e01ba267e9; subsequent commits adjust tests/documentation only. Installed package d13975522fb7 is a regular int32 install. Twelve compiled native objects remain byte-identical to the fresh f063f9f58de5 build.
- NCI explicitly reports scheduled maintenance; no fresh merged-head Gadi/int64 result is claimed.

Twelve optional assess comparisons are skipped when assess is unavailable; independent native analytical tests run. Earlier 336 benchmark records remain pinned to their original numerical versions and are not represented as reruns of this merge.

## Issue adjudication and limits

We make no new issue-closure claims from this integration. Upstream fixes for #920/#924 (caps/tolerances), #907 (failure reporting), #963 (exact-mode projection), #894 (operator refresh), #936 (constrained reaction memory), and #1007 (cleanup) are preserved, with regression evidence where noted; that does not substitute for a development-only issue-closing probe.

The following remain open or unverified by this PR, rather than being inferred fixed:

- #348/#899: partition-dependent direct-LU datum and split-fault LU behavior; the direct-LU field oracle here is serial.
- #910/#911/#985/#625/#755: broader cached/roundoff/Schur-cost and viscoelastic-preconditioner issues; explicit rotated inner options do not claim to solve them generally.
- #98/#608/#939/#931/#923/#962: other pressure, corner and host/partition-specific wall/topography failures; Kramer/coupled-wall component checks are not those reproducers.
- #883/#884: wider cold datum/reference and gauge-reference issues; retained targeted regression contracts do not establish every mixed-forcing/continuation case.
- #781/#769/#844/#817/#759: adjoint-target, nonlinear sensitivity, omitted-pressure constructor, constrained warm-up and zero-viscosity behavior outside the supported constant-viscosity P2/P1 methods.
- #876/#877/#887/#950/#624/#892/#629/#1010: unrelated ThinSheet API/units/documentation, option lifetime, default Krylov/FAC and hierarchy/feature integration claims are not closed here.
- #157: direct spherical diagnostics avoid generic tensor projection; the generic projector's memory behavior is not fixed.

Uniform mixed/multiplier inf-sup stability is unproven; the new coupled methods remain opt-in with explicit supported-space/geometry restrictions. Numerical topography/gravity integration is outside this change. The explicit experimental gap methods do not claim adjoint or automatic Model.step transcript integration; the ordinary solver description/adjoint contracts are tested separately. Fresh merged-head Gadi/int64 validation is unavailable during NCI maintenance; earlier int64 evidence is not relabelled as current. Merge should await required GitHub CI and maintainer review, with int64 validation explicitly tracked if required for release.
