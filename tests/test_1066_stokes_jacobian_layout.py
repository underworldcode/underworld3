"""Stokes velocity-block Jacobian layout and Picard-freezing contracts (issue #457).

Two independent defects hid behind major-symmetric tangents until the
transverse-isotropic Newton tangent (the first in-repo tangent WITHOUT major
symmetry) exposed them:

1. ``SNES_Stokes_SaddlePt`` assembled ``uu_G3`` as the MAJOR TRANSPOSE of the
   true tangent (``derive_by_array`` is dx-first; the old
   ``permutedims((0,2,1,3))`` translation assumed dx-last). Test 1 pins the
   assembled ``_uu_G3`` against a finite-difference oracle built from the
   RESIDUAL flux — the same comparison ``-snes_test_jacobian`` makes — using a
   tangent whose major-asymmetric part is provably non-trivial, so the test
   cannot pass by symmetry.

2. ``TransverseIsotropicFlowModel._build_c_tensor`` baked the UNWRAPPED
   ``.sym`` contents of its viscosity Parameters into the c-tensor, exposing
   the strain-rate dependence to ``sympy.diff`` and silently un-freezing the
   default (Picard) tangent. Tests 2/3 pin the freezing contract: coefficients
   hide inside UWexpression atoms under Picard; only the Newton unwrap sees
   through them.

See docs/developer/subsystems/petsc-jacobian-layout.md.
"""

import numpy as np
import pytest
import sympy

import underworld3 as uw
from underworld3.function.expressions import UWexpression, unwrap_expression


ETA1_FACTOR = 0.01  # weak plane: anisotropy strongly active


def _ti_stokes(consistent_jacobian):
    """Tiny TI Stokes with a strain-rate-dependent eta_0 (assigned as raw
    sympy — the Parameter setter wraps it into the Parameter atom)."""
    mesh = uw.meshing.StructuredQuadBox(elementRes=(4, 4))
    v = uw.discretisation.MeshVariable("V_jl", mesh, mesh.dim, degree=2)
    p = uw.discretisation.MeshVariable("P_jl", mesh, 1, degree=1)
    stokes = uw.systems.Stokes(mesh, velocityField=v, pressureField=p)

    x, y = mesh.X
    grad = sympy.Matrix(
        [[v.sym[0].diff(x), v.sym[0].diff(y)],
         [v.sym[1].diff(x), v.sym[1].diff(y)]]
    )
    edot = (grad + grad.T) / 2
    eII = sympy.sqrt(
        sympy.Rational(1, 2) * (edot[0, 0] ** 2 + edot[1, 1] ** 2)
        + edot[0, 1] ** 2
    )
    eta0 = (sympy.Float(0.001) + eII) ** sympy.Rational(-2, 3)

    stokes.constitutive_model = uw.constitutive_models.TransverseIsotropicFlowModel
    stokes.constitutive_model.Parameters.shear_viscosity_0 = eta0
    stokes.constitutive_model.Parameters.shear_viscosity_1 = ETA1_FACTOR * eta0
    rt2 = sympy.sqrt(2)
    stokes.constitutive_model.Parameters.director = sympy.Matrix([[1 / rt2, 1 / rt2]])

    stokes.add_dirichlet_bc((1.0, 0.0), "Top")
    stokes.add_dirichlet_bc((0.0, 0.0), "Bottom")
    stokes.consistent_jacobian = consistent_jacobian
    return stokes, v, p, mesh


def _numeric_fns(exprs, stokes, v, p, mesh, Lsyms):
    """Fully unwrap each expression, replace gradient/field/coordinate objects
    with plain symbols/numbers, lambdify over the 4 velocity-gradient symbols."""
    L = stokes.Unknowns.L
    N = mesh.CoordinateSystem.N
    flat_L = [Lsyms[k][l] for k in range(2) for l in range(2)]
    rep = {L[k, l]: Lsyms[k][l] for k in range(2) for l in range(2)}
    rep[v.sym[0]] = sympy.Float(0.1)
    rep[v.sym[1]] = sympy.Float(-0.2)
    rep[p.sym[0]] = sympy.Float(0.7)
    rep[N[0]] = sympy.Float(0.4)
    rep[N[1]] = sympy.Float(0.55)

    fns = []
    for e in exprs:
        eu = unwrap_expression(e, mode="nondimensional")
        eu = sympy.sympify(eu).xreplace(rep)
        leftovers = [a for a in eu.free_symbols if a not in flat_L]
        assert not leftovers, f"unresolved symbols {leftovers} in {e}"
        fns.append(sympy.lambdify(flat_L, eu, "numpy"))
    return fns


@pytest.mark.level_1
@pytest.mark.tier_a
def test_stokes_uu_g3_matches_fd_oracle():
    """The assembled Newton uu_G3, read back through PETSc's flat
    ((fc*d+gc)*d+df)*d+dg convention, must match a central-difference
    derivative of the residual flux. The TI Newton tangent has no major
    symmetry, so a transposed assembly fails this at O(1e-1)."""
    d = 2
    stokes, v, p, mesh = _ti_stokes(consistent_jacobian=True)
    stokes._setup_pointwise_functions()

    Lsyms = [[sympy.Symbol(f"l{k}{l}") for l in range(2)] for k in range(2)]
    Lval = np.array([[0.31, -1.17], [0.73, 0.11]])
    flat_val = Lval.flatten()

    # FD ground truth from the residual flux (what -snes_test_jacobian FDs).
    res_fns = _numeric_fns(list(stokes._u_F1), stokes, v, p, mesh, Lsyms)
    FD = np.zeros((d, d, d, d))  # [fc, gc, df, dg] = dF1[fc, df]/dL[gc, dg]
    h = 1.0e-6
    for gc in range(d):
        for dg in range(d):
            up = flat_val.copy()
            dn = flat_val.copy()
            up[gc * d + dg] += h
            dn[gc * d + dg] -= h
            for fc in range(d):
                for df in range(d):
                    idx = fc * d + df
                    FD[fc, gc, df, dg] = (res_fns[idx](*up) - res_fns[idx](*dn)) / (2 * h)

    scale = np.abs(FD).max()

    # Guard against a true-by-construction pass: the oracle tangent must have
    # a non-trivial major-asymmetric part, else this test cannot see a
    # transposed assembly at all.
    major_asym = max(
        abs(FD[a, b, c, e] - FD[b, a, e, c])
        for a in range(d) for b in range(d) for c in range(d) for e in range(d)
    )
    assert major_asym / scale > 1.0e-2, (
        "oracle tangent is (near-)major-symmetric — test cannot detect a "
        "transposed uu_G3; strengthen the anisotropy in the setup"
    )

    # The solver's assembled uu_G3, read exactly as PETSc reads it.
    G3M = stokes._uu_G3
    g3_fns = {}
    entries = []
    order = []
    for fc in range(d):
        for gc in range(d):
            for df in range(d):
                for dg in range(d):
                    entries.append(G3M[fc * d + gc, df * d + dg])
                    order.append((fc, gc, df, dg))
    fns = _numeric_fns(entries, stokes, v, p, mesh, Lsyms)
    G3 = np.zeros((d, d, d, d))
    for (fc, gc, df, dg), f in zip(order, fns):
        G3[fc, gc, df, dg] = f(*flat_val)

    rel = np.abs(G3 - FD).max() / scale
    assert rel < 1.0e-5, (
        f"assembled uu_G3 differs from the FD tangent by {rel:.3e} rel — "
        "layout transposition or missing tangent terms (issue #457)"
    )


@pytest.mark.level_1
@pytest.mark.tier_a
def test_ti_picard_tangent_is_frozen():
    """Under the default (Picard) tangent the TI viscosity coefficients must
    stay hidden inside UWexpression atoms: no raw velocity-gradient
    (Derivative) atoms may appear in the assembled uu_G3. The Newton build of
    the same problem MUST expose them (positive control that the dependence
    exists and the Newton unwrap sees it)."""
    d = 2

    stokes, _, _, _ = _ti_stokes(consistent_jacobian=False)
    stokes._setup_pointwise_functions()
    # NOTE: velocity gradients are UnderworldFunction symbols, not
    # sympy.Derivative atoms — detect dependence the way the tangent
    # machinery does: differentiate w.r.t. the L entries.
    L = stokes.Unknowns.L
    picard_live = [
        (i, k, l)
        for i, e in enumerate(stokes._uu_G3)
        for k in range(d)
        for l in range(d)
        if sympy.diff(sympy.sympify(e), L[k, l]) != 0
    ]
    assert not picard_live, (
        "Picard uu_G3 depends on the velocity gradient — the TI c-tensor "
        "build has un-frozen the default tangent (issue #457): first hits "
        f"{picard_live[:4]}"
    )

    # Frozen also means major-symmetric: C has minor+major symmetry, so the
    # Picard block must satisfy G3[fc,gc,df,dg] == G3[gc,fc,dg,df] exactly.
    G3M = stokes._uu_G3
    for fc in range(d):
        for gc in range(d):
            for df in range(d):
                for dg in range(d):
                    a = G3M[fc * d + gc, df * d + dg]
                    b = G3M[gc * d + fc, dg * d + df]
                    residue = sympy.expand(a - b)
                    if residue != 0:
                        # expand() is the cheap canonicaliser; fall back to
                        # simplify only on the (rare) unproven entry.
                        residue = sympy.simplify(residue)
                    assert residue == 0, (
                        f"Picard uu_G3 not major-symmetric at "
                        f"[{fc},{gc},{df},{dg}]"
                    )

    stokes_n, _, _, _ = _ti_stokes(consistent_jacobian=True)
    stokes_n._setup_pointwise_functions()
    Ln = stokes_n.Unknowns.L
    newton_live = any(
        sympy.diff(sympy.sympify(e), Ln[k, l]) != 0
        for e in stokes_n._uu_G3
        for k in range(d)
        for l in range(d)
    )
    assert newton_live, (
        "Newton uu_G3 has no velocity-gradient dependence — the unwrap "
        "path is not seeing the strain-rate-dependent viscosity, so this "
        "test's Picard assertion is vacuous"
    )


@pytest.mark.level_1
@pytest.mark.tier_a
def test_ti_vep_c_tensor_coefficients_are_frozen():
    """TI-VEP with yield active: the c-tensor COEFFICIENTS (including the
    yield-limited eta_1_eff, whose raw form carries grad-v through the
    resolved fault-plane shear rate) must live inside UWexpression atoms.
    E_eff's strain-rate content is residual structure and is exercised
    elsewhere; here only the coefficient freezing is pinned."""
    mesh = uw.meshing.StructuredQuadBox(elementRes=(4, 4))
    v = uw.discretisation.MeshVariable("V_tv", mesh, mesh.dim, degree=2)
    p = uw.discretisation.MeshVariable("P_tv", mesh, 1, degree=1)
    stokes = uw.systems.Stokes(mesh, velocityField=v, pressureField=p)
    cm = uw.constitutive_models.TransverseIsotropicVEPFlowModel(stokes.Unknowns)
    stokes.constitutive_model = cm
    cm.Parameters.shear_viscosity_0 = 1.0
    cm.Parameters.shear_viscosity_1 = 0.01
    cm.Parameters.shear_modulus = 100.0
    cm.Parameters.dt_elastic = 0.1  # BDF eta_0 composite is nan at the oo default
    cm.Parameters.yield_stress = 0.5
    cm.Parameters.director = sympy.Matrix([0.0, 1.0])
    cm.Parameters.strainrate_inv_II_min = 1.0e-6

    assert cm.is_viscoplastic
    cm._build_c_tensor()

    # Velocity gradients are UnderworldFunction symbols (not sympy.Derivative
    # atoms) — detect dependence via d/dL, as the tangent machinery does.
    # NOTE: iterating a rank-4 sympy Array yields SUB-ARRAYS — index scalars
    # explicitly.
    import itertools

    L = stokes.Unknowns.L
    d = mesh.dim
    c = sympy.Array(cm.c)
    live = [
        (idx, k, l)
        for idx in itertools.product(range(d), repeat=4)
        for k in range(d)
        for l in range(d)
        if sympy.diff(sympy.sympify(c[idx]), L[k, l]) != 0
    ]
    assert not live, (
        "TI-VEP c-tensor depends on the velocity gradient (yield-limited "
        "eta_1_eff baked unwrapped) — Picard tangent un-frozen (issue #457): "
        f"first hits {live[:4]}"
    )

    # Positive control: the dependence must exist INSIDE the atoms — the
    # Newton unwrap of the same tensor must reveal it.
    from underworld3.cython.generic_solvers import _jacobian_unwrap

    c_unwrapped = _jacobian_unwrap(c)
    revealed = any(
        sympy.diff(sympy.sympify(c_unwrapped[idx]), L[k, l]) != 0
        for idx in itertools.product(range(d), repeat=4)
        for k in range(d)
        for l in range(d)
    )
    assert revealed, (
        "Newton unwrap of the TI-VEP c-tensor reveals no strain-rate "
        "dependence — the yield law has been lost from the tangent chain"
    )
