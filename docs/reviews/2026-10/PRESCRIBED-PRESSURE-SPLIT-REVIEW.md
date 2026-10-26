# Prescribed pressure lifting: pre-PR audit

27 October 2026. Agent audit of `feature/cylindrical-pressure-jump` against
development `7222caf34c1e`; not maintainer approval. Runtime source
`55ea3ffea4e8`, Mac Linux ARM64 container, PETSc 3.25.4 int32.

## Findings resolved

- We found repeated extraction reused mesh-field names and overwrote retained
  lifts. Solver/call-specific names now preserve previous fields; the analytic/
  numerical/sign-reversal test asserts retained nodal arrays are unchanged.
- We propagate invalid rank-local samples and incompatible cycles before the
  next exchange; finite tips and contradictory junction orientations raise on
  every rank. Each requested interface must contain fitted facets.
- We removed direct-solver overrides from the manufactured test after its
  no-jump control failed too. The normal Stokes configuration satisfies both
  controls without changing library solver defaults.
- We keep exact-circle integration separate from fitted-facet P1 interpolation;
  neither the API nor its documentation equates the geometries or attributes
  the vertex-star construction to Chang and Kwak's harmonic extension.

## Attacks and evidence

- Both `test_1099*` modules: 14 tests pass per rank at 1, 2 and 8 ranks.
- Prescribed analytic expressions, continuous numerical fields, callable data,
  finite segments, crossing interfaces and tetrahedral traces satisfy independent
  facet-jump/continuity and native-volume-integral checks.
- Manufactured two-interface Stokes problems recover velocity below 1e-9 and
  pressure below 1e-8 RMS with essential or exterior natural data. Refinement
  at h=1/8, 1/16, 1/32 gives velocity order 3 and pressure order 2 on all tested
  rank counts. No analytical Stokes field constructs the lift.
- Vertex-owned exchanges carry interface stars, not replicated global meshes.
  Sorted interface iteration, global vertex/cell numbering, PointSF membership
  and ghost-cell replies are exercised by the parallel tests.
- Serial and MPI entries in `scripts/test.sh` include both new test files.
  Public entry is a Stokes method; boundary-condition capabilities are updated.
  No solver class, constraint mechanism or solve event is added. Prescribed
  data are frozen inputs; differentiability through lifting is not claimed.

## Open issues in the pressure/solver territory

We do not claim a fix or change their status: #98 concerns spherical boundary
pressure, #608 mixed-wall corner errors, #348 partition-dependent LU pressure
pinning, #625 the Schur iteration cap, and #755 viscoelastic preconditioner
scaling. Those code paths are unchanged; their reproducers were not rerun in
this pressure-extraction audit. The small manufactured meshes emit the existing
GAMG/no-hierarchy warning associated with #625, while meeting every error bound.

## Merge conditions

- Review the prescribed-jump sign and natural-datum contract and fitted-simplex
  scope. Unknown coupled jumps, unfitted cuts, nonzero finite-tip singularities
  and differentiation through numerical lifting are outside this API.
- Retain the serial/MPI mathematical tests and the specialised circle helper's
  independent weak-load checks. Broader benchmark acceptance is separate from
  these component checks; no default benchmark workflow changes here.
