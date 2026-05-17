# Metric-driven mesh redistribution (`smooth_mesh_interior`)

Topology-preserving node redistribution toward a target *size /
density* field. Vertex indices, DOF maps and the parallel partition
are **unchanged** — only coordinates move (contrast `mesh.adapt()`,
which remeshes / changes topology).

```python
import underworld3 as uw
from underworld3.meshing import smooth_mesh_interior

smooth_mesh_interior(mesh, metric=f, method="spring")   # fast
smooth_mesh_interior(mesh, metric=f, method="ma")       # robust
```

## When to use it

- **Restore the grading of a previously-adapted mesh** after it has
  deformed (free-surface evolution, large strain): a *Lagrangian*
  metric rides the material points and pulls the design grading
  back.
- **Concentrate resolution at a feature** — bunch nodes by a factor
  of ~2 around a high-gradient region (e.g. a moving fault, a
  thermal boundary layer) without adding points.

```{important}
With a **fixed node count** the achievable grading is bounded:
≈1.3–1.8× deep/near on the test problems. The optimal-transport
ideal (≈10× for an 8× density target) requires *more nodes* — a
topology change (`mesh.adapt`), not this smoother. A ×2-ish
bunching is squarely in range; do not expect extreme refinement
from redistribution alone.
```

## The metric

A strictly-positive density expression; larger ⇒ smaller cells.
For Lagrangian behaviour, build it from a frozen state variable
set **once** to the reference coordinate and never reassigned, so
its value rides each material point through deformation:

```python
r0 = uw.discretisation.MeshVariable("r0", mesh,
        vtype=uw.VarType.SCALAR, degree=1, continuous=True)
r0.data[:, 0] = np.linalg.norm(mesh.X.coords, axis=1)   # set once
f = 1 + 8 * sympy.exp(-((r0.sym[0] - 1.0) / 0.12) ** 2)  # design grading
```

`metric=None` (default) is the original graph-Laplacian Jacobi
smoother (equalises connectivity; no grading) — unchanged.

## The two solvers

| | `method="spring"` (default) | `method="ma"` |
|---|---|---|
| Operator | Volumetric elastic-spring equilibrium | Benamou–Froese–Oberman Monge–Ampère |
| Idea | *equal* edge springs (shape → equant cells, no slivers) **+** per-cell area constraint `A0 ∝ 1/ρ_tgt` (size) | `det(I+D²φ)=g`, move by ∇φ, recovered-Hessian damped Picard, pure-Neumann + constant nullspace |
| Cost (res-16 Annulus) | **~0.3 s** | ~12–20 s (~60×) |
| Grading (AMP=8 / 20) | 1.65 / 1.79 | 1.71 / 1.54 |
| Interior-feature fidelity | good, slightly anisotropic | **clean, isotropic** |
| Mesh quality | healthy, never degenerates | healthy |
| Boundary sensitivity | high (see `boundary_slip`) | low (natural Neumann handles it) |

**Recommendation:** `spring` for routine per-step use in
time-stepping (cheap, robust). `ma` when refinement *quality*
around a localised feature matters and the cost is affordable
(it is the bullet-proof answer; its efficiency is the subject of
follow-up work — see *Open items*).

### `boundary_slip`

Off by default. When on, boundary nodes slide *tangentially* along
their boundary and are snapped back to it every step (the radial
DOF is removed — they **cannot** leave the surface; drift is
machine-ε; circular/spherical boundaries only, serial). It
**strongly helps the spring** (~+10 % grading, ~3× faster — its
hard-pinned boundary was the bottleneck) and is a **near-no-op for
`ma`**. It is off by default because for a free surface the
boundary *is* the moving surface and sliding interacts with the
free-surface coupling — enable per use-context.

```{warning}
The per-ring radius projection is exact only for
circular/spherical boundaries. A general deformed / free-surface
boundary needs projection onto the boundary *polyline* instead —
not yet implemented (matters for the spring; low priority for MA,
which is insensitive to the boundary treatment).
```

## Implementation notes

- Spring equilibrium = minimise `½Σ_e((|x_i-x_j|-L̄)/L̄)² +
  size_w·Σ_t((A_t-A0_t)/A0_t)²` by Jacobi-preconditioned nonlinear
  CG (Polak–Ribière⁺) with an Armijo line search that **rejects
  cell-inverting steps** (the tangle guard is inside the
  optimiser). `shape_w/size_w` default 1/8 — results are robust to
  them.
- MA uses the core `SNES_Scalar.constant_nullspace` hook
  (`petsc_generic_snes_solvers.pyx`) and a variationally-consistent
  weak Hessian recovery (`_hessian_recovery_class`, an SPD
  mass-matrix `SNES_MultiComponent` solve — only first derivatives
  of φ, since UW3 forbids second derivatives of mesh-variable
  functions).
- **MA solver efficiency** (`_use_direct_solver`, 2026-05-17). The
  Picard loop fixes the mesh, so the φ-Poisson Laplacian and the
  Hessian-recovery mass matrix are *constant operators* re-solved
  ~25× with only the RHS changing. The UW3 default (GMRES + GAMG)
  paid a full multigrid **setup every inner solve** (the constant
  near-nullspace re-attach forces it) — ~0.9 s/iter for the Hessian
  alone. The cached φ/Hessian/∇φ sub-solvers are therefore put on:

  | option | φ Poisson | Hessian / ∇φ |
  |---|---|---|
  | `snes_type` | `ksponly` | `ksponly` |
  | `ksp_type` | `preonly` | `preonly` |
  | `pc_type` | `lu` | `lu` |
  | `pc_factor_mat_solver_type` | `mumps` | `mumps` |
  | `snes_lag_jacobian` | `-2` | `-2` |
  | `snes_lag_preconditioner` | `-2` | `-2` |
  | `mat_mumps_icntl_24` | `1` (null-pivot) | — |

  The lag (`-2` = compute once, never again) confines the
  factorisation to the **first** inner solve; the rest are MUMPS
  back-substitutions. `_deform_mesh` rebuilds the SNES
  (`is_setup=False`) so the lag counter resets and the operator is
  correctly re-factorised on the next call's first solve — reuse
  never spans a coordinate change. A direct solve is *exact*
  (tighter than the GMRES rtol) so the Picard fixed point — hence
  the grading/quality — is **bit-for-bit unchanged** (validated
  `ma_cost_grading.py`: d/n 1.02/1.43/1.71/1.54 identical to the
  GAMG baseline). `n_picard` default 40→25 (grading flat from
  iter ≈20).
- **φ order: `phi_degree` default 3 → 2.** The deep/near grading is
  set by the φ *order*, not the solver: P2 ≡ P3 to ~3 dp across
  AMP 0/2/8/20 (matches the recorded baseline; AMP=0 no-op exact;
  no tangle) at **~2× lower cost** (smaller matrices — which also
  *help* the direct factorisation scale). **P1 is not
  grading-equivalent** (≈1.40 vs 1.71 at AMP=8, ~18 % weaker) — P2
  is the floor. Net with the reuse work: cold ~12–18 s → **~0.7–0.9 s**
  (canonical `cost_compare.py`), grading bit-for-bit, ≈15–20×.

  ```{warning}
  Sparse direct factorisation does not scale to large-3D parallel
  per-timestep use, so a `linear_solver="gamg"` path applying the
  same factor/setup-once-reuse to FGMRES + GAMG was prototyped
  (selectable; `"direct"` remains the default and sole validated
  path). Findings: the constant nullspace **is** correctly wired
  (verified — not the failure); P3 was a **major GAMG confound**
  (P2+gamg converges where P3+gamg catastrophically diverges to a
  no-op); but even at P2 the *warm* (post-`_deform_mesh`) GAMG
  re-solve stays erratic, and this build has no alternative AMG
  (hypre/ML absent). The reuse *pattern* is sound; GAMG on this
  pure-Neumann operator is not robust here. Accepted position:
  **MUMPS direct is fine for now** (it is itself MPI-parallel) and
  the P2 size reduction only helps it. A robust iterative path would
  still need the pure-Neumann operator de-fragilised (single
  Dirichlet pin, not the constant nullspace — ∇φ unaffected by the
  additive constant) and/or hypre, and is **gated behind** the
  smoother first becoming parallel-exact in *assembly* + 3D (the
  solver is not the parallel bottleneck yet). Full results:
  `docs/developer/design/ma-newton-cofactor-exploration.md`.
  ```
- Both paths are **serial-exact**; spring/MA edge & cell sums are
  accumulated over locally-visible entities, so rank-partition
  boundaries under-count in parallel (the Jacobi `metric=None` path
  *is* parallel-exact). Cross-rank assembly is future work.

## Validation & diagnostics

`scripts/` (not packaged): `show_metric_mesh.py` /
`plot_metric_meshes.py` (Annulus surface band, Spring vs MA, honest
metric + mesh pictures), `interior_refine.py` (localised interior
blob — the realistic case), `slip_test.py` / `ma_slip_test.py`
(boundary-slip A/B), `setup_sanity.py` (metric/pinning sanity),
`ma_analytic_check.py` (exact radial equidistribution ground
truth), `cost_compare.py`. Figures land in `/tmp/metric_mesh/`.

```{note}
The honest grading metric is **per-node mean incident edge length
binned by final radius** (deep/near). An earlier centroid-band
metric averaged the thin strong layer with the bulk Lagrangian
shift and understated grading ~40% — use the per-node metric.
```

## Open items (future sessions)

- **Monge–Ampère efficiency** — *done* (2026-05-17): ~10× via
  factor-once-reuse direct sub-solves, grading bit-for-bit unchanged
  (see the Implementation note). Two follow-on directions were then
  explored and **both closed negative** (see the design doc
  `ma-newton-cofactor-exploration.md`):
  - *Newton / cofactor linearisation* — tested (Phase 0). Same MA
    equation ⇒ same fixed-node grading (settled, not a lever), and it
    is *less robust* than BFO at the recovered-Hessian quality
    available (all standard convexity safeguards fail to reach BFO's
    fixed point). UW3 forbids 2nd derivatives of mesh-var functions
    so a sharp `D²φ` isn't available. Do not pursue.
  - *BFO + GAMG-reuse* (parallel path) — prototyped as
    `linear_solver="gamg"`. Reuse pattern fires; nullspace verified
    correctly wired (not the failure); P3 was a major confound
    (P2+gamg converges where P3+gamg diverges) but even at P2 the
    warm re-solve stays erratic; no alternative AMG in this build.
    Accepted: MUMPS direct (itself MPI-parallel) for now, and the
    P2 size cut only helps it. A robust iterative path needs a
    single-Dirichlet-pin (not the constant nullspace) and/or hypre,
    gated behind parallel-exact assembly + 3D (the solver is not the
    bottleneck yet). *Bankable spin-off: `phi_degree` default 3→2 —
    grading-identical, ~2× cheaper, see the Implementation note.*
- General deformed / free-surface boundary slip (polyline
  projection).
- Parallel-exact spring/MA assembly.
