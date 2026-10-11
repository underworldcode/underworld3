# Coupled volume fields in Stokes: adversarial review record

Branch `feature/shear-band-length-scale`, commits `003d3b95` (`add_coupled_field`) and
`6f0627ce` (`add_coupled_dirichlet_bc`, the shared essential-BC helper, the Cosserat
shear-layer test), written overnight on Hyperion (Linux, conda-forge PETSc). Reviewed on
the Mac (`runtime` environment) against the installed build of those commits.

## What the branch does

`SNES_Stokes_SaddlePt.add_coupled_field(field, F0, F1, momentum_flux)` adds an extra
volume field to the Stokes DM. Its weak form is `∫ F0 ψ + F1·∇ψ = 0`, and it is solved
in the same Newton system as velocity and pressure. Every Jacobian block touching the
field is derived from the residuals, through `_jacobian_source`. `momentum_flux` is added
to the momentum flux (a Cosserat skew stress). Every new path is guarded on
`_coupled_fields`, so an ordinary Stokes solve registers and compiles exactly what it did
before.

The driver behind it is the notch length scale: a micromorphic (gradient-plasticity)
field and a 2-D Cosserat micro-rotation. Both converge on the notch at ref 3 with no rate
strengthening; the campaign report is `notch_length_scale/REPORT.md` in uw3-campaigns.

## Checks that hold

- **Ordinary Stokes C is byte-identical.** The fingerprint script
  (`notch_length_scale/drivers/stokes_c_md5.py`) hashes the header of three ordinary
  solvers: isoviscous Picard, Drucker-Prager Newton, and block-constrained. On the Mac it
  reproduces Hyperion's branch and base values exactly
  (`655ff38a…`, `d5a5ba4b…`, `4fc089ca…`), including after the fixes below.
- **The generated C indexes the coupled field correctly.** `petsc_u[3]` is χ and
  `petsc_u[2]` is p, against DS components `[2, 1, 1]` (offsets 0, 2, 3).
- **Residual slots are wired to the matching compiled functions.**
- **test_1070 passes at np = 3 on every rank.**

## Findings and their state

1. **The two-way Taylor test failed on the Mac (fixed: test).** Remainder slopes were
   0.11 and 0.26. The χ and pressure rows had slope 2; the velocity row did not respond
   to pressure at all. The cause is the fixture, not the plumbing:
   - The box is closed on all four walls, so the pressure is defined only up to a
     constant, and no null space was declared.
   - MUMPS LU of the singular system let that constant drift to −2.3e14 over the Newton
     steps. At that magnitude adjacent doubles are about 0.03 apart, so the residual cannot
     be resolved below |F| ≈ 2.
   - The solve then stopped on step size (`SNES_CONVERGED_SNORM_RELATIVE`) without
     converging, and the test never checked.
   - Hyperion's MUMPS kept the constant small, so the test passed there.

   Fix: the fixture declares `petsc_use_pressure_nullspace`, and the test asserts
   |F| < 1e-8 before the Taylor steps. The negative control (frozen tangent) still fails,
   at slope 1.00.
2. **Rotated free-slip with a coupled field stopped at iteration 0 with no
   explanation (fixed: refusal).** The rotated driver splits velocity and pressure by field
   number, so a coupled field lies outside its split; this is the #464 situation the
   multipliers are already refused for. `_reject_rotated_with_coupled_fields` now refuses
   the pair in `add_coupled_field`, `add_rotated_freeslip_bc`, `add_fault_bc` and
   `solve`. Tested in both orders.
3. **`momentum_flux` was silently dropped by solvers that build their own `F1`
   (fixed: refusal).** Only the `SNES_Stokes` template adds it. `SNES_NavierStokes` and
   `SNES_NavierStokes_Composed` do not. The class flag `_F1_carries_coupled_momentum_flux`
   (True on `SNES_Stokes`, False on the composed Navier-Stokes) makes `add_coupled_field`
   raise instead. Tested.
4. **The same field could be coupled twice (fixed: refusal).** That would have registered
   two DM fields for one variable. Tested.
5. **The default solver path was never exercised: every test and run used exact LU
   (fixed: test).** It converges for two-way coupling in 6 Newton steps to |F| = 8e-9.
   Its split 0 is velocity plus coupled fields under the velocity multigrid, and split 1
   is pressure. Pinned by a test. Hyperion has since run FMG on the notch: same Newton
   counts as LU when every linear solve is tight.
6. **The stale strict xfail in test_1064 is pre-existing** (XPASS on the base
   `f57c2a27` too). The branch's `TODO(BUG)` comment there is reverted: it is outside this
   change and needs its own decision.
7. **`add_natural_bc`'s value is minus the traction (pre-existing; docs).** It enters
   PETSc's boundary f0 as given, and the docstring says "traction". The Cosserat test
   passes the negated traction, with a comment. Not changed here.

8. **The default fieldsplit solve with a coupled field hung at np = 3 (fixed: `3181baa8`).**
   `test_two_way_coupling_converges_with_the_default_fieldsplit_solver` passed serially
   but hung at np = 3.
   - **Cause.** With no mesh hierarchy the velocity block runs GAMG, and its option
     bundle sets `fieldsplit_velocity_mat_block_size 2`. `_setup_block_fieldsplit_options`
     mirrored that onto split 0, which also holds the coupled field. Its DOFs interleave
     with the velocity's, so split 0 is not node-blocked: the block size was wrong
     everywhere.
   - **Why it hung rather than failed.** PETSc refused the block size only where a rank's
     local split-0 size was odd ("Local size 215 not compatible with block size 2",
     `PCSetUp_FieldSplit` -> `MatSetFromOptions` -> `PetscLayoutSetBlockSize`). The ranks
     that raised dropped out, and the remaining rank waited in a collective
     (`MPI_Allreduce`: "Message truncated"). Serially the size happened to be even, so
     the wrong block size passed silently.
   - **Fix.** The velocity block size is not mirrored when coupled fields share split 0.
   - **Regression test.**
     `test_default_fieldsplit_with_a_coupled_field_has_no_velocity_block_size`: χ held on
     one wall makes split 0 odd (705 DOFs) on one process, which failed before the fix
     with the same PETSc error.
   - **Two test fixes in the same commit.**
     - The default-solver test's χ bound was rank-local (`chi.array.max()`, 0.74 on the
       rank far from the lid); it is now the global `chi.max()`.
     - The driven-box fixture leaves its right wall traction-free, so the pressure is
       fixed. See finding 10 for why the declared null space was not enough.
   - **Result.** test_1070: 9/9 serially, and 9/9 on every rank at np = 3, five runs
     out of five (Hyperion) and three out of three (Mac).

9. **OPEN (latent): the multipliers-only split keeps the mirrored block size.** With block
   constraints but no coupled field, split 0 is the velocity alone, and
   `_setup_block_fieldsplit_options` still mirrors `fieldsplit_velocity_mat_block_size`
   to `fieldsplit_0_mat_block_size`. `_withdraw_block_size_if_not_node_blocked` runs
   afterwards and deletes only the `fieldsplit_velocity_` key, so when the velocity is not
   node-blocked (an odd local size from mixed component-wise BCs) the `fieldsplit_0_` key
   survives and PETSc would refuse it, on the affected ranks only. Not reproduced yet;
   marked `TODO(BUG)` in `_setup_block_fieldsplit_options`.

10. **OPEN: the closed-box pressure drifted at np = 3 with the null space declared.** Before
    the fixture change, the two-way Taylor test (closed box, `petsc_use_pressure_nullspace`,
    monolithic MUMPS LU) reached |F| = 2.1 instead of < 1e-8 at np = 3 in one run of five
    (Hyperion), the same symptom as the Mac's serial drift without the null space. So the
    declared null space is not always effective under monolithic exact LU in parallel.
    **Plain Stokes does not reproduce it.**
    - The check, `notch_length_scale/drivers/nullspace_drift_probe.py` in uw3-campaigns:
      plain Stokes with no coupled field, the same closed box, the null space declared,
      exact Newton with MUMPS LU, and viscosity 1 + ε̇_II² so the solve takes several
      Newton steps.
    - Result at np = 3, five runs: all identical, 7 steps, |F| = 1.9e-11, the same
      pressure in each run.
    - So the drift is not a general failure of the declared null space under parallel
      MUMPS. It appeared only with a coupled field in the closed box, once in five runs,
      and was not reproduced in isolation (the Taylor test alone passed 4 runs out of 4).
    - Left open with that evidence. The fixture change (a traction-free wall) removes the
      singular system from test_1070 either way.

11. **Every frozen-tangent path assembled NaN on a cold coupled solve (fixed, with
    test).** A coupled source written with the bare invariant `Unknowns.Einv2 =
    sqrt(...)` has a derivative of 0/0 at rest.
    - The frozen (Picard) form of a coupled row is differentiated as written: the
      `expr` passed to `_jacobian_source`.
    - The Newton form is guarded by `_jacobian_unwrap` (`sqrt(g)` becomes
      `sqrt(g + 1e-36)`).
    - So a cold solve's Picard warm-up (`solve(picard=n)`), `consistent_jacobian = False`
      and the continuation blend `J_P + alpha (J_N - J_P)` all assembled NaN; alpha * NaN
      is NaN at alpha = 0.
    - Symptom: DIVERGED_LINEAR_SOLVE at iteration 0, 2-D and 3-D (GAMG sub-PC failure on
      split 0, or LU failure).
    - The Stokes rows never showed it: there the viscosity is an opaque expression in the
      frozen form.
    - The default cold solve escaped because it took no warm-up here, and the notch runs
      start from a viscous seed.
    - Fix: `_guard_sqrts` lifted to a module-level helper, unchanged, and applied to the
      coupled rows' frozen source. The Newton source is the raw form unwrapped, so the
      full-Newton tangent is unchanged; the residual is unchanged.
    - Ordinary Stokes C is byte-identical to the base (`stokes_c_md5.py`: 655ff38a,
      d5a5ba4b, 4fc089ca).
    - Tests: `test_cold_frozen_tangent_of_a_bare_invariant_source_is_finite` (picard 1
      and 3, frozen, continuation; 2-D and 3-D) and
      `test_cold_coupled_solve_with_a_picard_warmup_converges` (picard 1 and 3; 2-D and
      3-D). All 12 fail without the fix and pass with it; the file is 24/24 serially and
      at np = 3.
    - Not fixed here, and not the NaN: on this strongly two-way problem pure Picard
      (`False`) stops with DIVERGED_LINE_SEARCH after 1-4 steps, and the continuation ramp
      reaches only |F| = 1.5e-4 in 58 steps in 2-D and stalls near 1.9 in 3-D. The frozen
      tangent drops the u-chi cross blocks, so its direction need not descend.
      **Note, nothing to fix:** coupled-field problems use Newton only, plus the automatic
      cold Picard warm-up step (ruling, 2026-10-11).
    - Where the frozen tangent runs on a coupled problem under that ruling: only in the
      automatic cold warm-up, and that is armed only when the STOKES flux is nonlinear in
      u. `_flux_is_linear_in_unknowns()` does not count coupled fields as unknowns.
      - In the test_1070 problem (eta = 1 + 0.5 chi^2) it returns True, so a cold solve
        takes no warm-up. That is why the default path converged before the fix.
      - A cold notch (a viscoplastic flux, nonlinear in u) does arm it. That is the case
        this fix protects, for example a cold 3-D notch.
      - The campaign's notch runs start from a viscous seed and take no warm-up.

## Not covered

- Coupled fields at np ≥ 3 on the notch.
- FMG with a multi-component coupled field (3-D Cosserat).
- `add_coupled_dirichlet_bc` with the units model active: numbers are taken as
  non-dimensional, as documented.
