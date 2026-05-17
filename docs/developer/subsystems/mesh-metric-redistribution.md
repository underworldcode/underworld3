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
  GAMG baseline). Net: cold ~12–18 s → ~1–2 s, warm ~34 s → ~1–2.5 s
  (the warm≫cold GAMG-resetup pathology is eliminated). `n_picard`
  default 40→25 (grading flat from iter ≈20).

  ```{warning}
  This is a **serial / modest-size** optimisation. Sparse direct
  factorisation (even parallel MUMPS) does not scale to large 3D
  per-timestep use (fill-in, memory, communication). The portable
  insight is *operator-constant-in-the-loop* → **factor/setup once,
  reuse**: in a parallel setting keep an iterative PC (GAMG / `gamg`
  with a constant near-nullspace) but apply the same
  `snes_lag_jacobian -2` / `KSPSetReusePreconditioner` so GAMG setup
  is paid once per call, plus warm-start the Krylov solve from the
  previous Picard φ. The direct solver is the serial expedient; the
  reuse pattern is what generalises. See *Open items*.
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

- **Monge–Ampère efficiency** — *largely done* (2026-05-17, see the
  Implementation note): ~10× via factor-once-reuse direct sub-solves;
  grading bit-for-bit unchanged. Remaining: the serial direct solver
  must become a **parallel-scalable** scheme. The win was *not*
  "direct vs iterative" — it was eliminating the per-Picard-iteration
  preconditioner re-setup on a constant operator. Port that to
  parallel by keeping GAMG (constant near-nullspace, already wired)
  but with `snes_lag_jacobian -2` / `KSPSetReusePreconditioner` so
  the GAMG hierarchy is built once per call, and warm-start the
  Krylov solve from the previous Picard φ. Quantify the parallel
  GAMG-reuse path against the serial MUMPS numbers.
- **Newton / cofactor linearisation** — replace the damped Picard
  with a quasi-Newton step on `cof(D²φ):D²(δφ)=f−det(D²φ)`
  (`cof(H)=det(H)H⁻ᵀ`). Same MA equation ⇒ *same* fixed-node grading
  (≈1.5–1.8×, settled — not a grading lever); the gain is far fewer
  outer iterations (a handful vs ~20) ⇒ an efficiency/robustness
  lever, and the per-step operator is a variable-coefficient SPD
  elliptic problem that is **AMG-friendly** (good for the parallel
  requirement above). Needs convexity safeguarding (convex guess /
  line search / projection onto convex Hessians / continuation in
  `f`); BFO's `+√` branch already supplies the convex selection in
  Picard form. Uses the existing recovered Hessian.
- General deformed / free-surface boundary slip (polyline
  projection).
- Parallel-exact spring/MA assembly.
