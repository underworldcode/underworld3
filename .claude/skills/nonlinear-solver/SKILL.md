---
name: nonlinear-solver
description: How to make a hard nonlinear Stokes solve (Drucker-Prager / yield-stress viscoplastic) CONVERGE reliably in Underworld3 the way the working recipe actually does it — automatic warm-start (one Picard step on a cold start) plus a MULTI-SOLVE δ-continuation (constant δ per solve, warm-start the next, sharper δ), the consistent-Newton tangent, and a non-symmetry-safe multigrid smoother. Reach for THIS when a viscoplastic solve stalls / diverges and you are about to hand-tune PETSc options, ramp δ, or "just add a monitor". It carries the CONFIG TRAP LIST — the setup mistakes that each produce a different failure a few steps in — and the one thing you must NOT do (ramp δ inside a single SNES solve). For the yield-law maths and which tangent per model, see `plasticity-solvers`.
---

# nonlinear-solver

The recipe that gets a **hard viscoplastic (Drucker–Prager) Stokes** problem to
converge, and — more importantly — the list of setup mistakes that stop it. The
central lesson from the Spiegelman hard-case study (`η_bg=1e26`, `V=10`): every
failure was a **solver-configuration** error, not a bad Jacobian. If the correct
setup is a minefield for an expert, that is an API regression — so the goal is to
make the correct path the default path.

Design of record: `docs/developer/design/nonlinear-solver-homotopy-warmstart.md`.
Yield-law maths, tangent-per-model, quadratic-convergence check: `plasticity-solvers`.

---

## The recipe (what actually converges)

1. **Warm start.** A cold plastic start (`v=0 ⇒ ε̇=0 ⇒ τ_y/0 ⇒ NaN`) is avoided by
   starting the continuation at **large δ** (the power-mean is then the harmonic
   mean, bounded by `η_bg` even at `ε̇→0`) and taking **one Picard step** into the
   Newton basin. One Picard step is defect-correction iteration 1 — contractive,
   cheap. From a *warm* iterate, take **no** Picard step (it wastes the good
   quadratic start).

2. **Multi-solve δ-continuation** (NOT an in-solve ramp). Hold δ **constant** for a
   full nonlinear solve to tolerance; warm-start the next, smaller δ from that
   converged state; march δ down to the sharp surface. δ is a `constants[]` atom,
   so each step is a recompile-free `PetscDSSetConstants` update. Reference driver:

   ```python
   from underworld3.systems import yield_continuation
   cm.yield_mode = "softmin"; cm.yield_smoother = "powermean"
   stokes.consistent_jacobian = True                      # per plasticity-solvers
   # non-symmetry-safe smoother (see trap list):
   stokes.petsc_options["fieldsplit_velocity_mg_levels_ksp_type"] = "gmres"
   stokes.petsc_options["fieldsplit_velocity_mg_levels_pc_type"]  = "sor"
   # a viscous (no-yield) presolve gives the warm start the driver expects
   result = yield_continuation(stokes, delta0=1.0, down=0.5, dmin=1e-3)
   # result["settled_delta"] is the smallest δ that converged
   ```

3. **Consistent-Newton tangent** for non-elastic DP (`consistent_jacobian=True`);
   **Picard** for elastic VEP — see `plasticity-solvers` for the per-model table.

4. **`bt` line search** with the consistent tangent on a smooth (δ>0) surface.

The δ-march is cheap: with the power-mean smoother a converged δ warm-starts every
sharper δ in ≈0 Newton iterations, so a residual-guided auto-descent costs almost
nothing.

---

## DO NOT ramp δ inside one SNES solve

Ramping δ **inside a single SNES solve** (a `SNESSetUpdate` callback that sharpens
the yield surface between Newton iterations) is **proven dead** — it diverges
`DIVERGED_LINEAR_SOLVE` after ~2 iterations and grinds for ~2 hours, **even on the
proven solver config**. Mechanism: the continuation only sharpens δ from a
**converged**, well-conditioned iterate; the in-SNES ramp sharpens δ **mid-solve**
at a far-from-solution iterate where the consistent-Newton Jacobian on a sharpening
surface is ill-conditioned and the linear solve fails. **Hide the *continuation*,
not the *ramp*.** (This supersedes the `enable_yield_homotopy()` in-SNES ramp still
described in `plasticity-solvers`; that path is retained only as a dead-experiment
record — use the multi-solve continuation above.)

---

## CONFIG TRAP LIST

Each of these produces a *different* failure a few steps in — that is why the hard
case felt like whack-a-mole. Check them first.

| Trap | Symptom | Fix |
|---|---|---|
| Consistent Newton makes the velocity block **non-symmetric**; default Chebyshev/Richardson MG smoother assumes SPD | smoother diverges / stalls → `DIVERGED_LINEAR_SOLVE` or an endless grind | `fieldsplit_velocity_mg_levels_ksp_type=gmres` + `mg_levels_pc_type=sor` |
| `preconditioner="fmg"` (vs explicit `pc_type=mg` + manual mg opts) | outer KSP "converges" in **1 iteration** → no real Newton correction → stall → `DIVERGED_LINE_SEARCH` | use explicit `pc_type=mg` with the smoother opts above; bound the outer KSP (`ksp_max_it`~80) so a hostile step fails fast |
| Cold plastic start `v=0` | `ε̇=0 → τ_y/0 → NaN` on iteration 0 | start δ **large** (power-mean → harmonic, bounded by `η_bg`) + one Picard step |
| LU velocity block with all-Dirichlet-ish BC | pressure nullspace singular | attach the Stokes nullspace / avoid a bare LU there |
| Hand-rolled `snes_monitor` to "see what's happening" | you read residuals but miss the tell | use `solve_with_diagnostics` / `get_snes_diagnostics` instead (below) |

**The diagnostic tell:** `solver.get_snes_diagnostics()["linear_iterations"] ≈ 1`
per Newton step means the linear solve is doing **no real work** (the FMG-1-iteration
trap). A healthy consistent-Newton solve does real Krylov work each step and
converges quadratically. Use `solve_with_diagnostics()`, not a hand-rolled monitor.

---

## Automatic warm-start (Layer 1 — landed)

`solver.has_solution` is a **public, read-only** status flag: `True` only after a
solve whose SNES converged; reset on a structural rebuild (remesh / adapt /
mesh-mover — the `is_setup=False` hook); kept through coefficient changes (viscosity,
δ, BC values, time step). A **diverged** solve leaves it `False`, so the next solve
auto-cold-starts rather than warming off a corrupted iterate.

On a **cold** (`zero_init_guess=True`) Stokes solve under the **consistent-Newton
tangent**, a single Picard step is now taken automatically (reusing the existing
`picard=1` machinery). The default (frozen) tangent path is bit-identical.

```python
stokes.consistent_jacobian = True
stokes.solve()                 # cold → one automatic Picard step, then Newton
if stokes.has_solution:
    ...
```

---

## Implementation status (this line of work)

- **Layer 1a — DONE:** `has_solution` + cold consistent-Newton Picard warm-up
  (`petsc_generic_snes_solvers.pyx`; test `test_0201`).
- **Layer 1b — planned:** flip `zero_init_guess` default to auto-detect (cold-vs-warm
  from `has_solution`). Benchmark; gate the free-surface chain-of-solves and the
  mover/adapt reset first.
- **Layer 3 — planned:** default the FMG velocity smoother to non-symmetry-safe
  (gmres/sor) when `consistent_jacobian` is on. Benchmark vs the consistent-Newton tests.
- **Layer 2 — planned:** constitutive model advertises the homotopy
  (`supports_yield_homotopy` / `_yield_homotopy_control`) and
  `stokes.solve(homotopy=True, homotopy_options=...)` runs the continuation
  (folding in `yield_continuation`) with residual-guided auto-descent.

---

## Gotchas

- **`./uw build` → `amr-dev` env**; verify `uw.__file__` is the worktree site-packages.
- **Run VEP/consistent-Newton tests UNFORKED** — `pytest --forked` SIGABRTs (fork of
  multithreaded PETSc).
- Benchmark **every** default change — "Solver Stability is Paramount".
- ξ (rate-strengthening) is a **non-homotopic** regularisation: put a user loop
  *around* `solve()`, never inside the δ-march.

## Reference

- Design: `docs/developer/design/nonlinear-solver-homotopy-warmstart.md`,
  `jacobian-consistent-tangent.md`, `solver-strategies-catalogue.md`.
- Continuation driver: `underworld3.systems.yield_continuation`.
- Diagnostics: `SNES_*.get_snes_diagnostics()` / `solve_with_diagnostics()`.
- Related skills: `plasticity-solvers` (yield law + tangent per model),
  `free-surface-convection`, `adaptive-meshing`.
