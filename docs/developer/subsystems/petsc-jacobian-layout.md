---
title: "PETSc Pointwise Jacobian Layout"
---

# PETSc Pointwise Jacobian Layout

This page documents the exact index convention PETSc expects for pointwise
Jacobian arrays (`g0`, `g1`, `g2`, `g3`) registered via
`PetscDSSetJacobian`. Getting this wrong produces silent numerical errors
— the solve may converge to a wrong answer (often in ways that look
plausible for symmetric problems) or stagnate without ever reaching
tolerance. This was the origin of a real bug fixed in
`SNES_MultiComponent_Projection` on 2026-04-20.

Read this before writing or modifying any class that registers a
`PetscDSSetJacobian` callback.

## The four pointwise Jacobian arrays

For a single-field problem with `Nc` components per node and `dim` spatial
dimensions, the four pointwise Jacobians couple the residual pieces
(`f0` value, `F1` flux) to the unknowns (`u` value, `∇u` gradient):

| Array | Semantic                                             | Size            |
|-------|------------------------------------------------------|-----------------|
| `g0`  | $\partial f_0[\mathrm{fc}] / \partial u[\mathrm{gc}]$                 | `Nc × Nc`       |
| `g1`  | $\partial f_0[\mathrm{fc}] / \partial(\nabla u)[\mathrm{gc}, \mathrm{df}]$          | `Nc × Nc × dim` |
| `g2`  | $\partial F_1[\mathrm{fc}, \mathrm{df}] / \partial u[\mathrm{gc}]$              | `Nc × dim × Nc` |
| `g3`  | $\partial F_1[\mathrm{fc}, \mathrm{df}] / \partial(\nabla u)[\mathrm{gc}, \mathrm{dg}]$ | `Nc × dim × Nc × dim` |

Notation:
- **`fc`** = test (output) component index
- **`gc`** = trial (input) component index
- **`df`** = test (residual-side) spatial derivative index
- **`dg`** = trial (unknown-side) spatial derivative index

## The flat-index convention

PETSc's element-matrix assembly walks these arrays as **flat** buffers
with the two *component* indices on the outside and the two *derivative*
indices on the inside:

```
g0[fc*Nc + gc]
g1[(fc*Nc + gc) * dim + df]
g2[(fc*Nc + gc) * dim + df]
g3[((fc*Nc + gc) * dim + df) * dim + dg]
```

Source of truth: `PetscFEUpdateElementMat_Internal` in
`petsc-custom/petsc/src/dm/dt/fe/interface/fe.c:2639`. The element-matrix
kernels at lines 2690 (g0), 2718 (g1), 2748 (g2), 2779 (g3) read the
arrays with exactly these index expressions.

**The key point**: the two component indices come first (outer), then
the two derivative indices (inner). The layout is `[fc, gc, df, dg]`,
**not** `[fc, df, gc, dg]`. This is easy to get wrong because the
"natural" mathematical reading of $\partial F_1[\mathrm{fc}, \mathrm{df}] / \partial L[\mathrm{gc}, \mathrm{dg}]$
pairs each function argument with its derivative — but PETSc's storage
pairs like-kind indices.

## Sympy's native convention

```{warning}
An earlier revision of this page stated the opposite convention
("F's indices first, then x's") and audited the Stokes `permutedims`
path as correct on that basis. That was wrong — it is how the issue
#457 transposed-tangent bug survived the 2026-04-21 audit.
```

`sympy.derive_by_array(F, x)` is **dx-FIRST**: the returned array has
**`x`'s indices first, then `F`'s** (see the sympy documentation for
`derive_by_array`; verified numerically against finite differences in
the #457 investigation):

```python
G3 = sympy.derive_by_array(F1, L)
#    F1 shape (Nc, dim), L shape (Nc, dim)
#  → G3 shape (Nc, dim, Nc, dim)
#  → G3[k, l, i, j] = ∂F1[i, j] / ∂L[k, l]
#  → index order [gc, dg, fc, df]
```

Translating sympy → PETSc's `[fc, gc, df, dg]` therefore needs the
permutation **`(2, 0, 3, 1)`** before flattening. The historically-used
`(0, 2, 1, 3)` yields `[gc, fc, dg, df]` — the **major transpose**
`∂F1[gc, dg]/∂L[fc, df]` in the `[fc, gc, df, dg]` slot. That is
identical to the true tangent whenever the tangent has major symmetry
(any frozen major-symmetric `C`, the isotropic Newton term
`η'·(ε̂⊗ε̂)`, linear transverse isotropy) and silently wrong when it
does not (the transverse-isotropic Newton term is `B⊗a` with `B ∦ a` —
issue #457).

Do not reach for `permutedims` at all — use the explicit construction
below.

## Alternative: explicit construction

Instead of `derive_by_array` + `permutedims`, you can loop the indices
and write each entry directly into a matrix shaped for row-major flatten
into PETSc's layout. This is what `SNES_MultiComponent` does — the code
reads like the PETSc formula and removes any chance of getting the sympy
convention wrong on future sympy upgrades:

```python
G3 = sympy.zeros(Nc * Nc, dim * dim)
for fc in range(Nc):
    for gc in range(Nc):
        for df in range(dim):
            for dg in range(dim):
                G3[fc * Nc + gc, df * dim + dg] = sympy.diff(F1[fc, df], L[gc, dg])
```

Row index `fc * Nc + gc`, col index `df * dim + dg`. Row-major flatten
gives `((fc * Nc + gc) * dim + df) * dim + dg` — PETSc's expected index.

See `src/underworld3/cython/petsc_generic_snes_solvers.pyx` — the
`_setup_pointwise_functions` method of `SNES_MultiComponent`.

## Worked example — why symmetric problems hide the bug

Consider `F1[i, j] = L[i, j]` (raw gradient, *not* symmetrised) and
`Nc = dim = 2`. Then $\partial F_1[i, j] / \partial L[k, l] = \delta_{ik}\,\delta_{jl}$.
The non-zero entries in PETSc's correct layout `g3[fc, gc, df, dg]` are
the positions where `fc == gc` **and** `df == dg`:

```
(fc=0, gc=0, df=0, dg=0) → flat idx 0
(fc=0, gc=0, df=1, dg=1) → flat idx 3
(fc=1, gc=1, df=0, dg=0) → flat idx 12
(fc=1, gc=1, df=1, dg=1) → flat idx 15
```

Now consider the stress-like symmetric case
$F_1[i, j] = \tfrac{1}{2}(L[i, j] + L[j, i])$. The Jacobian is
$\partial F_1 / \partial L[k, l] = \tfrac{1}{2}(\delta_{ik}\delta_{jl} + \delta_{il}\delta_{jk})$.
Because this expression is **invariant under the swap $(k, l) \leftrightarrow (l, k)$**,
a Jacobian array that swaps the trial-side indices (`gc`↔`dg`) still
integrates to the same stiffness matrix. Every current consumer of
`SNES_Vector` uses an F1 with this symmetry — Stokes stress, symmetric
strain-rate smoothing, divergence penalty — so the bug is hidden.

An asymmetric F1 like raw `smoothing * L` exposes it immediately.

## Audit of existing solvers

Results of walking every solver in `petsc_generic_snes_solvers.pyx` as
of 2026-04-21:

| Solver | Construction | Status |
|--------|--------------|--------|
| `SNES_Scalar` | no permutation (Nc=1) | ✅ correct — single component makes the swap a no-op |
| `SNES_Vector` | explicit per-entry construction | ✅ correct (2026-04-21 migration) |
| `SNES_MultiComponent` | explicit per-entry construction | ✅ correct (2026-04-20 fix) |
| `SNES_Stokes_SaddlePt` | explicit per-entry construction | ✅ correct (2026-08 migration, issue #457 — the earlier `permutedims (0, 2, 1, 3)` form assembled the **major transpose** of `uu_G3` and was mis-audited as correct on 2026-04-21 because this page then stated the sympy convention backwards) |
| `SNES_NavierStokes` | inherits Stokes | ✅ correct (inherited) |

### `SNES_Stokes_SaddlePt` migration (2026-08, issue #457)

The saddle-point solver used `derive_by_array + permutedims((0,2,1,3))`
for the `uu`, `up`, and natural-BC Jacobian blocks. Under the true
(dx-first) sympy convention this places `∂F1[gc,dg]/∂L[fc,df]` in
PETSc's `[fc,gc,df,dg]` slot — the major transpose. Every isotropic
rheology (and linear transverse isotropy) has a major-symmetric tangent,
so the transpose was invisible to the whole test suite; the
transverse-isotropic Newton tangent is the first in-repo tangent without
major symmetry, and `-snes_test_jacobian` flagged it at ~1e-1 relative
(vs FD-limited ~1e-4 after the fix). All blocks now use the same
explicit-index loops as `SNES_Vector`/`SNES_MultiComponent`. The
scalar-`p` blocks (`up_*`, `pu_*`) were layout-safe only where a size-1
axis made the swap a no-op; they are explicit now too.

### `SNES_Vector` migration (2026-04-21)

`SNES_Vector` originally used `derive_by_array` followed by
`permutedims(·, (0, 3, 1, 2))` for `g3` and `(2, 1, 0)` for `g1`/`g2`.
These permutations do not match PETSc's `[fc, gc, df, dg]` layout, but
the discrepancy is invisible whenever F1 is symmetric under the
trial-side swap `(k, l) ↔ (l, k)`. Every in-repo consumer of
`SNES_Vector` — `SNES_Vector_Projection` (using `Unknowns.E` plus a
divergence penalty), the Nitsche BC path, the VE stress tau projection
— supplies exactly that kind of F1, so the bug never fired in tests.

Empirical reproduction (before the migration):

```text
SNES_Vector_Projection subclass with F1 = smoothing * L (raw gradient)
two identical targets sin(x)·cos(y) on StructuredQuadBox(4, 4):

  smoothing = 0.1  →  |u0 − u1| = 1.59   (should be 0)
  vs. SNES_MultiComponent_Projection → rel-L2 disagreement ≈ 0.30
```

**Fix applied**: `SNES_Vector._setup_pointwise_functions` now builds all
four Jacobians (and the natural-BC Jacobians) with explicit nested
loops that write directly into the PETSc-ordered matrix (row-major
`[fc*Nc + gc, df*dim + dg]` etc.). The code now reads the same as
`SNES_MultiComponent` — see the companion method — and no
`permutedims` is applied to any residual or BC Jacobian.

After the migration: `tests/test_snes_vector_asymmetric_jacobian.py`
passes (identical targets produce identical components, and the
asymmetric-F1 solve matches `SNES_MultiComponent_Projection` to
rel-L2 ≤ 1e-8 across smoothing ∈ {1e-4, 1e-2, 1e-1}). The full
Stokes (`test_1010_stokesCart`) and VE Stokes (`test_1050_VEstokesCart`)
suites continue to pass, confirming no regression on symmetric-F1
consumers.

## Checklist for new solvers

When writing a new class that registers a `PetscDSSetJacobian` callback:

1. **Use the explicit-index pattern from `SNES_MultiComponent` — always.**
   It reads like the PETSc documentation and is robust against sympy
   convention drift. There is no remaining in-repo
   `derive_by_array + permutedims` Jacobian path to copy from, by design:
   the last one (Stokes) carried a transposed tangent for months because
   every test tangent happened to be major-symmetric (issue #457).

2. **Write a validation test.** For every solver that can be reached at
   `Nc > 1` with `smoothing > 0`, include a test that:
   - Runs with **identical targets** for each component and checks the
     returned components are identical (decouples the component-coupling
     bug from target geometry).
   - Runs with **non-zero smoothing** (not just the L2 projection limit
     — `smoothing = 0` makes g3 the zero matrix and the bug can't show).
   - Compares against a known-good reference solver at the same
     tolerances. Use `ksp_type=preonly` + `pc_type=lu` to eliminate
     iterative-solver tolerance as a confounder.

3. **Keep a pointer to this doc** in a code comment above the Jacobian
   construction. Authors reaching for `permutedims` should see a
   reminder that PETSc's layout is `[fc, gc, df, dg]`, not the sympy
   default.

## References

- PETSc element-matrix kernels (authoritative):
  `petsc-custom/petsc/src/dm/dt/fe/interface/fe.c:2583–2790`
  (`petsc_elemmat_kernel_g0/g1/g2/g3` macros and the assembly loop in
  `PetscFEUpdateElementMat_Internal`).
- `PetscDSSetJacobian` API:
  [petsc.org](https://petsc.org/release/manualpages/DT/PetscDSSetJacobian/).
- Bug-fix history: `SNES_MultiComponent_Projection` initially used a
  `[fc, df, gc, dg]` reshape (same layout as `SNES_Vector`). The bug
  was invisible at `smoothing = 0` because g3 vanishes; it became
  obvious when a validation test with `smoothing > 0` showed
  components with identical targets producing different results.
  Fixed in the commit that introduced the solver — see the companion
  tests in `tests/test_multicomponent_projection.py`.
