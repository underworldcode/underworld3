#!/usr/bin/env python3
"""multiplier_schur_pc: verified live where it can be, instrumented where it is inert (#486).

The flag swaps only the Pmat (h,h) block of each Lagrange-multiplier field
(``_setup_solver``: the DS JacobianPreconditioner term becomes pressure's
``1/mu`` mass instead of the screening ``eps`` mass). Tracing PETSc's
fieldsplit.c resolves where that block is actually read:

* ``pc_fieldsplit_schur_precondition = "a11"`` — the Schur preconditioner is
  the grouped ``[p,h]`` **Pmat** block: the flag is LIVE (matrix-level oracle
  below, immune to "both converge in 4 iterations").
* class defaults (``selfp`` + ``diag_use_amat``) — Sp is assembled from
  **Amat** sub-blocks; the Pmat (h,h) block is provably never read: the flag
  is INERT. That inertness (the issue's original measurement) is codified as
  a negative control pinning the PETSc semantics, and the solver now records
  + warns when the opt-in cannot reach the PC.
"""
import numpy as np
import pytest
import scipy.sparse as sp
import sympy

import underworld3 as uw

pytestmark = [pytest.mark.level_2, pytest.mark.tier_b]


def _schur_pre_matrix(solver):
    """The assembled Schur preconditioner matrix Sp, as scipy CSR."""
    schur_ksp = solver.snes.getKSP().getPC().getFieldSplitSubKSP()[1]
    Sp = schur_ksp.getOperators()[1]
    i, j, v = Sp.getValuesCSR()
    return sp.csr_matrix((v, j, i), shape=Sp.getSize())


def _rel_diff(A, B):
    denom = sp.linalg.norm(A) + 1e-300
    return sp.linalg.norm(A - B) / denom


def _constrained(mesh, name, schur_pre, flag):
    """Open-top box, TWO constraint arcs (Left/Right), viscosity contrast,
    linear (ksponly) so the extracted PC is free of nonlinear noise."""
    xx, yy = mesh.X
    v = uw.discretisation.MeshVariable(f"v{name}", mesh, mesh.dim, degree=2)
    p = uw.discretisation.MeshVariable(f"p{name}", mesh, 1, degree=1)
    s = uw.systems.Stokes_Constrained(mesh, velocityField=v, pressureField=p)
    s.constitutive_model = uw.constitutive_models.ViscousFlowModel
    s.constitutive_model.Parameters.shear_viscosity_0 = sympy.exp(
        sympy.log(1000.0) * xx)                      # 3 decades across the box
    s.bodyforce = sympy.Matrix(
        [0.0, sympy.sin(sympy.pi * xx) * sympy.cos(sympy.pi * yy)])
    s.add_dirichlet_bc((0.0, 0.0), "Bottom")
    s.add_constraint_bc(0.0, "Left", normal=sympy.Matrix([[-1.0, 0.0]]))
    s.add_constraint_bc(0.0, "Right", normal=sympy.Matrix([[1.0, 0.0]]))
    s.petsc_options["snes_type"] = "ksponly"
    if schur_pre is not None:
        s.petsc_options["pc_fieldsplit_schur_precondition"] = schur_pre
    s.multiplier_schur_pc = flag
    return s


@pytest.fixture(scope="module")
def solves():
    """All four (regime x flag) arms on one mesh, warnings captured per arm."""
    import warnings as _w

    mesh = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0, 0), maxCoords=(1, 1), cellSize=0.2, qdegree=3)
    out = {}
    for key, schur_pre, flag in (
            ("selfp_off", None, False),
            ("selfp_on", None, True),
            ("a11_off", "a11", False),
            ("a11_on", "a11", True)):
        s = _constrained(mesh, key, schur_pre, flag)
        with _w.catch_warnings(record=True) as caught:
            _w.simplefilter("always")
            s.solve()
        out[key] = (s, [str(w.message) for w in caught])
    return out


# --------------------------------------------------------------------------- #
#  Live regime: under a11 the flag provably changes the Schur preconditioner
# --------------------------------------------------------------------------- #
def test_flag_is_live_under_a11(solves):
    """Matrix-level oracle: the assembled Schur pre differs flag-on vs
    flag-off under a11 — the DS JacobianPreconditioner path for multiplier
    fields reaches the PC. (If this ever finds NO difference, that path is
    broken upstream of the flag — file that as its own bug.)"""
    Sp_off = _schur_pre_matrix(solves["a11_off"][0])
    Sp_on = _schur_pre_matrix(solves["a11_on"][0])
    assert _rel_diff(Sp_on, Sp_off) > 1e-6, (
        "multiplier_schur_pc changed nothing under a11 — the multiplier "
        "JacobianPreconditioner path is broken upstream of the flag")


def test_probe_is_regime_aware_under_a11(solves):
    """Where the flag IS live, there must be no inertness record or warning."""
    s_on, warned = solves["a11_on"]
    assert "multiplier_schur_pc" not in s_on.pc_fallbacks
    assert not [w for w in warned if "multiplier_schur_pc" in w]


# --------------------------------------------------------------------------- #
#  Inert regime codified (the issue's measurement, now a pinned control)
# --------------------------------------------------------------------------- #
def test_flag_is_inert_under_class_defaults_and_says_so(solves):
    """selfp + diag_use_amat: Sp identical flag-on/off (pins the fieldsplit.c
    semantics we traced — if PETSc changes FieldSplitSchurPre, this fails),
    and the solver records + warns that the opt-in cannot reach the PC."""
    Sp_off = _schur_pre_matrix(solves["selfp_off"][0])
    s_on, warned = solves["selfp_on"]
    Sp_on = _schur_pre_matrix(s_on)
    assert _rel_diff(Sp_on, Sp_off) == 0.0, (
        "selfp read the Pmat (h,h) block — the PETSc fieldsplit semantics "
        "this class's defaults rely on have changed")
    rec = s_on.pc_fallbacks["multiplier_schur_pc"]
    assert rec["reason"] == "declined"
    assert "selfp" in rec["installed"]
    assert [w for w in warned if "multiplier_schur_pc" in w], (
        "an explicit opt-in doing nothing must warn")


def test_no_flag_means_no_record_and_no_warning(solves):
    """Negative control: with the flag off, the probe stays silent."""
    s_off, warned = solves["selfp_off"]
    assert "multiplier_schur_pc" not in s_off.pc_fallbacks
    assert not [w for w in warned if "multiplier_schur_pc" in w]


# --------------------------------------------------------------------------- #
#  Setter contract
# --------------------------------------------------------------------------- #
def test_toggling_after_a_solve_forces_ds_reregistration(solves):
    s_off, _ = solves["selfp_off"]
    assert s_off.is_setup
    s_off.multiplier_schur_pc = True
    assert not s_off.is_setup, (
        "toggling multiplier_schur_pc must force the DS term to re-register")
    s_off.multiplier_schur_pc = False   # restore for any later use


def test_probe_reads_the_flag_value_not_its_presence():
    """diag_use_amat set to FALSE under selfp means the Pmat (h,h) block IS
    read — the opt-in is live and the inertness warning must stay silent.
    The #534 review measured the Schur pre differing by rel-Frobenius 0.30
    flag-on/off in this regime while the probe (reading hasName, not the
    value) still recorded 'declined'."""
    import warnings as _warnings

    mesh = uw.meshing.StructuredQuadBox(elementRes=(6, 6))
    s = _constrained(mesh, "valrd", None, True)
    s.petsc_options["pc_fieldsplit_diag_use_amat"] = False
    with _warnings.catch_warnings(record=True) as caught:
        _warnings.simplefilter("always")
        s.solve()
    assert "multiplier_schur_pc" not in s.pc_fallbacks, (
        "probe read the key's presence, not its value: flag=False makes the "
        "opt-in live, not inert")
    assert not [w for w in caught if "multiplier_schur_pc" in str(w.message)]
