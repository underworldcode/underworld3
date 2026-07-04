"""Standalone symbolic isolation of the TI Jacobian-permutation bug.

The solver builds the uu Jacobian block as
    G3[i,j,k,l] = d F1[i,j] / d L[k,l]            (sympy derive_by_array; F1=stress)
    M = permutedims(G3, PERM).reshape(d*d, d*d)   (PERM currently (0,2,1,3))
and hands M to PETSc as g3.

derive_by_array is exact, so any error is in PERM/reshape vs PETSc's g3 layout.
The isotropic identity I_ijkl is invariant under index swaps that the TI
director term breaks, so a wrong PERM passes isotropic and fails TI.

This script:
 1. builds C (TI) and C_iso symbolically (no solver),
 2. forms F1 = C : sym(L), G3 = d F1/d L,
 3. reports which index symmetries G3 has (iso vs TI) — the swaps the wrong
    PERM relies on,
 4. for every one of the 24 permutations, tests whether the assembled M
    reproduces the TRUE directional derivative  d/de [ Gv : F1(L + e*dL) ]
    = sum_ijkl Gv[i,j] G3[i,j,k,l] dL[k,l]  under the PETSc-style contraction
    vec(Gv)^T M vec(dL)  with row-major vec — for BOTH iso and TI.
The correct PERM passes iso AND TI; the current (0,2,1,3) is expected to pass
iso only. (The row-major-vec contraction is validated against iso, where the
current code is known correct by snes_test_jacobian ~1e-8.)
"""
import itertools
import numpy as np
import sympy

d = 2
# symbolic velocity gradient L (full, NOT symmetric) and a test/trial pair
L = sympy.Matrix(d, d, lambda i, j: sympy.Symbol(f"L{i}{j}"))
symL = [L[i, j] for i in range(d) for j in range(d)]
symE = (L + L.T) / 2     # strain rate edot = sym(L)


def rank4_identity(d):
    I = sympy.MutableDenseNDimArray.zeros(d, d, d, d)
    for i in range(d):
        for j in range(d):
            for k in range(d):
                for l in range(d):
                    I[i, j, k, l] = (sympy.Rational(1, 2)
                                     * ((i == k) * (j == l) + (i == l) * (j == k)))
    return I


def C_iso(eta0):
    I = rank4_identity(d)
    return sympy.MutableDenseNDimArray(
        [[[[2 * eta0 * I[i, j, k, l] for l in range(d)]
           for k in range(d)] for j in range(d)] for i in range(d)])


def C_ti(eta0, eta1, n):
    """Muhlhaus-Moresi TI tensor, exactly as constitutive_models.py builds it."""
    I = rank4_identity(d)
    Delta = eta0 - eta1
    C = sympy.MutableDenseNDimArray.zeros(d, d, d, d)
    for i in range(d):
        for j in range(d):
            for k in range(d):
                for l in range(d):
                    base = 2 * I[i, j, k, l] * eta0
                    aniso = 2 * Delta * (
                        (n[i] * n[k] * int(j == l) + n[j] * n[k] * int(l == i)
                         + n[i] * n[l] * int(j == k) + n[j] * n[l] * int(k == i)) / 2
                        - 2 * n[i] * n[j] * n[k] * n[l])
                    C[i, j, k, l] = base - aniso
    return C


def stress(C):
    # F1[i,j] = sum_kl C[i,j,k,l] edot[k,l]
    F1 = sympy.zeros(d, d)
    for i in range(d):
        for j in range(d):
            F1[i, j] = sum(C[i, j, k, l] * symE[k, l]
                           for k in range(d) for l in range(d))
    return F1


def G3_of(C):
    F1 = stress(C)
    return sympy.derive_by_array(F1, L)   # index (i,j,k,l) = dF1[i,j]/dL[k,l]


# numeric params
import math
n0 = sympy.Matrix([math.cos(0.6), math.sin(0.6)])     # tilted director
G3i = G3_of(C_iso(3.0))
G3t = G3_of(C_ti(3.0, 1.0, n0))


def numic(G3, Lval):
    f = sympy.lambdify(symL, sympy.Array(G3), "numpy")
    return np.array(f(*Lval), dtype=float).reshape(d, d, d, d)


rng = [0.37, -0.21, 0.13, 0.44]   # arbitrary L values (linear, so value irrelevant)
Ai = numic(G3i, rng)
At = numic(G3t, rng)


def sym_report(A, tag):
    swaps = {"ij (first pair)": (1, 0, 2, 3),
             "kl (last pair)": (0, 1, 3, 2),
             "ik<->jl (pair swap)": (2, 3, 0, 1),
             "i<->k": (2, 1, 0, 3),
             "j<->l": (0, 3, 2, 1)}
    print(f"  [{tag}] index symmetries of G3:")
    for name, p in swaps.items():
        ok = np.allclose(A, np.transpose(A, p))
        print(f"     {name:22s}: {'HOLDS' if ok else 'BROKEN'}")


print("=== which index symmetries does G3 have? ===")
sym_report(Ai, "ISO")
sym_report(At, "TI ")

# True directional derivative tensor is G3 itself (index i,j,k,l).
# Test each permutation's assembled M under row-major vec contraction.
print("\n=== permutation test: does vec(Gv).T @ M @ vec(dL) == sum Gv G3 dL ? ===")
Gv = np.array([[0.5, -0.3], [0.2, 0.9]])
dL = np.array([[0.11, 0.7], [-0.4, 0.25]])


def true_dirderiv(A):
    return sum(Gv[i, j] * A[i, j, k, l] * dL[k, l]
              for i in range(d) for j in range(d)
              for k in range(d) for l in range(d))


def assembled_contraction(A, perm):
    M = np.transpose(A, perm).reshape(d * d, d * d)
    return Gv.reshape(-1) @ M @ dL.reshape(-1)


ti_true = true_dirderiv(At)
iso_true = true_dirderiv(Ai)
good = []
for perm in itertools.permutations(range(4)):
    iso_ok = abs(assembled_contraction(Ai, perm) - iso_true) < 1e-9
    ti_ok = abs(assembled_contraction(At, perm) - ti_true) < 1e-9
    if iso_ok:
        flag = "  <-- passes ISO AND TI" if ti_ok else "  (iso only)"
        print(f"  perm {perm}: iso OK, ti {'OK' if ti_ok else 'FAIL'}{flag}")
        if ti_ok:
            good.append(perm)
print(f"\ncurrent code perm = (0,2,1,3)")
print(f"permutations correct for BOTH iso and TI: {good}")
