"""Parallel regression test for rotated strong free-slip (``add_rotated_freeslip_bc``).

The rotated free-slip solve (per-node DOF rotation → strong ``v_n=0`` on the rotated
normal rows → rotate back → gauge removal) was previously validated serially only
(``tests/test_1018_rotated_freeslip.py``). At np>1 it crashed: the RHS constrained-row
zeroing indexed the *local* slice with *global* row indices, which overflows on any rank
whose ownership does not start at 0 — an asymmetric crash that masqueraded as a hang.
With that fixed, the whole global system (and hence the velocity solve and the wall-normal
leakage) is partition-independent.

This test verifies that the parallel solve reproduces the serial reference to a tight
tolerance for two geometries:

  * **box** — 4-wall rotated free-slip on axis-aligned walls (GAMG velocity block); the
    velocity L2 ``∫ v·v`` must match serial (bit-identical up to the parallel reduction
    order), and the analytic SolCx velocity error must stay small.
  * **annulus** — per-node *radial* free-slip on both arcs with the analytic normal (the
    rotation-nullspace + gauge-removal path); the velocity L2 and the radial leakage
    ``∫ (v·r̂)²`` on each arc must match serial.

All diagnostics use the parallel-safe ``uw.maths.Integral`` / ``BdIntegral`` reductions
(no rank-local ``v.data``, which is per-partition, and no ``uw.function.evaluate`` on
arbitrary points, which deadlocks np>1).

Run with:
    mpirun -n 2 python -m pytest --with-mpi \\
      tests/parallel/test_1064_rotated_freeslip_parallel.py
    mpirun -n 4 python -m pytest --with-mpi \\
      tests/parallel/test_1064_rotated_freeslip_parallel.py
"""

import numpy as np
import sympy
import pytest

import underworld3 as uw
from underworld3.function import analytic as A
from underworld3.utilities import custom_mg

pytestmark = [pytest.mark.mpi(min_size=2), pytest.mark.timeout(180)]

# SERIAL (np=1) reference diagnostics — the partition-independent ground truth.
# Recompute with `python <thisfile> {box,annulus}`.
# box:     (velocity L2, analytic velocity error)
# annulus: (velocity L2, radial-leakage L2 on Lower arc, radial-leakage L2 on Upper arc)
GOLDEN_BOX = (1.275109036912e-03, 1.529545e-05)
GOLDEN_ANNULUS = (1.897011154231e-02, 4.563841e-05, 9.341699e-06)
# annulus driven by CUSTOM GEOMETRIC FMG on the velocity block (nested hierarchy):
# (velocity L2, radial-leakage L2 on Lower arc, radial-leakage L2 on Upper arc)
GOLDEN_ANNULUS_FMG = (1.906961759626e-02, 5.428193e-06, 1.177002e-06)


def _wrap(dm, m0):
    """Wrap a refined DMPlex as a UW3 mesh carrying the base mesh's boundaries and
    coordinate system (for building a nested annulus MG hierarchy)."""
    return uw.discretisation.Mesh(
        dm.clone(), simplex=True,
        coordinate_system_type=m0.CoordinateSystem.coordinate_type,
        qdegree=3, boundaries=m0.boundaries)


def _box_diagnostics():
    """Box SolCx with rotated free-slip on all four axis-aligned walls (GAMG velocity
    block). Returns (velocity L2, analytic velocity error)."""
    mesh = uw.meshing.StructuredQuadBox(
        elementRes=(24, 24), minCoords=(0, 0), maxCoords=(1, 1), qdegree=3)
    sol = A.SolCx(mesh, eta_A=1.0, eta_B=1.0e3, x_c=0.5, n=1)
    v = uw.discretisation.MeshVariable("vB", mesh, mesh.dim, degree=2, continuous=True)
    p = uw.discretisation.MeshVariable("pB", mesh, 1, degree=1, continuous=False)
    s = uw.systems.Stokes(mesh, velocityField=v, pressureField=p)
    s.constitutive_model = uw.constitutive_models.ViscousFlowModel
    s.constitutive_model.Parameters.shear_viscosity_0 = sol.fn_viscosity
    s.bodyforce = sol.fn_bodyforce
    s.penalty = 0.0
    s.tolerance = 1e-9
    for wall in ("Top", "Bottom", "Left", "Right"):
        s.add_rotated_freeslip_bc(wall)
    s.petsc_use_pressure_nullspace = True
    s.petsc_options["snes_type"] = "ksponly"
    s.solve()

    L2 = float(np.sqrt(uw.maths.Integral(mesh, v.sym.dot(v.sym)).evaluate()))
    verr = float(sol.velocity_error(v))
    return L2, verr


def _annulus_diagnostics():
    """Annulus with per-node radial free-slip on both arcs (analytic normal). Returns
    (velocity L2, radial-leakage L2 on Lower arc, radial-leakage L2 on Upper arc)."""
    RI, RO = 0.5, 1.0
    mesh = uw.meshing.Annulus(radiusInner=RI, radiusOuter=RO, cellSize=0.1, qdegree=3)
    x, y = mesh.X
    r = sympy.sqrt(x**2 + y**2)
    th = sympy.atan2(y, x)
    v = uw.discretisation.MeshVariable("Va", mesh, mesh.dim, degree=2, continuous=True)
    p = uw.discretisation.MeshVariable("Pa", mesh, 1, degree=1, continuous=True)
    s = uw.systems.Stokes(mesh, velocityField=v, pressureField=p)
    s.constitutive_model = uw.constitutive_models.ViscousFlowModel
    s.constitutive_model.Parameters.shear_viscosity_0 = 1.0
    s.bodyforce = sympy.Matrix([[x / r * sympy.cos(4 * th) * (r - RI) * (RO - r) * 40.0,
                                 y / r * sympy.cos(4 * th) * (r - RI) * (RO - r) * 40.0]])
    nhat = sympy.Matrix([[x / r, y / r]])
    s.add_rotated_freeslip_bc("Lower", normal=nhat)
    s.add_rotated_freeslip_bc("Upper", normal=nhat)
    s.tolerance = 1e-9
    s.petsc_use_pressure_nullspace = True
    s.petsc_options["snes_type"] = "ksponly"
    s.solve()

    L2 = float(np.sqrt(uw.maths.Integral(mesh, v.sym.dot(v.sym)).evaluate()))
    vr = v.sym[0] * x / r + v.sym[1] * y / r      # radial velocity v·r̂
    leak_lo = float(np.sqrt(uw.maths.BdIntegral(
        mesh=mesh, fn=vr**2, boundary="Lower").evaluate()))
    leak_up = float(np.sqrt(uw.maths.BdIntegral(
        mesh=mesh, fn=vr**2, boundary="Upper").evaluate()))
    return L2, leak_lo, leak_up


def _annulus_fmg_diagnostics():
    """Annulus radial free-slip whose velocity block is CUSTOM GEOMETRIC FMG on a
    nested hierarchy (coarse annulus -> refine -> refine), via set_custom_fmg — no
    GAMG, no direct solve. Returns (velocity L2, leakage L2 on Lower, on Upper)."""
    RI, RO = 0.5, 1.0
    m0 = uw.meshing.Annulus(radiusInner=RI, radiusOuter=RO, cellSize=0.2, qdegree=3)
    dm1 = m0.dm.refine()
    dm2 = dm1.refine()
    coarse = [m0, _wrap(dm1, m0)]
    fine = _wrap(dm2, m0)

    x, y = fine.X
    r = sympy.sqrt(x**2 + y**2)
    th = sympy.atan2(y, x)
    v = uw.discretisation.MeshVariable("Vf", fine, fine.dim, degree=2, continuous=True)
    p = uw.discretisation.MeshVariable("Pf", fine, 1, degree=1, continuous=True)
    s = uw.systems.Stokes(fine, velocityField=v, pressureField=p)
    s.constitutive_model = uw.constitutive_models.ViscousFlowModel
    s.constitutive_model.Parameters.shear_viscosity_0 = 1.0
    s.bodyforce = sympy.Matrix([[x / r * sympy.cos(4 * th) * (r - RI) * (RO - r) * 40.0,
                                 y / r * sympy.cos(4 * th) * (r - RI) * (RO - r) * 40.0]])
    nhat = sympy.Matrix([[x / r, y / r]])
    s.add_rotated_freeslip_bc("Lower", normal=nhat)
    s.add_rotated_freeslip_bc("Upper", normal=nhat)
    s.tolerance = 1e-9
    s.saddle_preconditioner = 1.0
    s.petsc_use_pressure_nullspace = True
    s.petsc_options["snes_type"] = "ksponly"
    custom_mg.set_custom_fmg(s, coarse, builder="barycentric", field_id=0)
    s.solve()

    assert s._rotated_freeslip_info["ksp_reason"] > 0, "custom-FMG rotated solve did not converge"
    L2 = float(np.sqrt(uw.maths.Integral(fine, v.sym.dot(v.sym)).evaluate()))
    vr = v.sym[0] * x / r + v.sym[1] * y / r
    leak_lo = float(np.sqrt(uw.maths.BdIntegral(
        mesh=fine, fn=vr**2, boundary="Lower").evaluate()))
    leak_up = float(np.sqrt(uw.maths.BdIntegral(
        mesh=fine, fn=vr**2, boundary="Upper").evaluate()))
    return L2, leak_lo, leak_up


def test_rotated_freeslip_box_partition_independent():
    """Box: the parallel rotated free-slip solve reproduces the serial velocity L2 and
    keeps the analytic velocity error small."""
    L2, verr = _box_diagnostics()
    L2_ref, verr_ref = GOLDEN_BOX
    assert np.isclose(L2, L2_ref, rtol=1e-8, atol=0), (
        f"box velocity L2 differs serial vs np={uw.mpi.size}: {L2_ref} vs {L2}")
    # analytic accuracy is preserved (partition may change the exact digit but not
    # the order of magnitude of the SolCx error)
    assert verr < 1e-3, f"box velocity error {verr:.2e} too large at np={uw.mpi.size}"


def test_rotated_freeslip_annulus_partition_independent():
    """Annulus: the parallel radial free-slip solve reproduces the serial velocity L2
    and the (partition-independent) radial leakage on both arcs."""
    L2, leak_lo, leak_up = _annulus_diagnostics()
    L2_ref, leak_lo_ref, leak_up_ref = GOLDEN_ANNULUS
    # velocity L2 is iterative-solver-tolerance reproducible (~1e-8 rel), not the
    # box's 1e-10 — the annulus carries a rotation null space + gauge removal.
    assert np.isclose(L2, L2_ref, rtol=1e-6, atol=0), (
        f"annulus velocity L2 differs serial vs np={uw.mpi.size}: {L2_ref} vs {L2}")
    assert np.isclose(leak_lo, leak_lo_ref, rtol=1e-4, atol=0), (
        f"annulus Lower leakage differs serial vs np={uw.mpi.size}: "
        f"{leak_lo_ref} vs {leak_lo}")
    assert np.isclose(leak_up, leak_up_ref, rtol=1e-4, atol=0), (
        f"annulus Upper leakage differs serial vs np={uw.mpi.size}: "
        f"{leak_up_ref} vs {leak_up}")


def test_rotated_freeslip_annulus_fmg_partition_independent():
    """The full stack: rotated radial free-slip on the annulus with the velocity block
    driven by CUSTOM GEOMETRIC FMG (set_custom_fmg) reproduces the serial velocity L2
    and radial leakage in parallel — FMG x rotated x annulus x np>1."""
    L2, leak_lo, leak_up = _annulus_fmg_diagnostics()
    L2_ref, leak_lo_ref, leak_up_ref = GOLDEN_ANNULUS_FMG
    assert np.isclose(L2, L2_ref, rtol=1e-6, atol=0), (
        f"FMG annulus velocity L2 differs serial vs np={uw.mpi.size}: {L2_ref} vs {L2}")
    assert np.isclose(leak_lo, leak_lo_ref, rtol=1e-4, atol=0), (
        f"FMG annulus Lower leakage differs serial vs np={uw.mpi.size}: "
        f"{leak_lo_ref} vs {leak_lo}")
    assert np.isclose(leak_up, leak_up_ref, rtol=1e-4, atol=0), (
        f"FMG annulus Upper leakage differs serial vs np={uw.mpi.size}: "
        f"{leak_up_ref} vs {leak_up}")


if __name__ == "__main__":
    # Recompute the serial GOLDEN references: `python <thisfile> {box,annulus,annulus_fmg}`.
    import sys
    _kind = sys.argv[1] if len(sys.argv) > 1 else "box"
    if _kind == "annulus":
        _L2, _lo, _up = _annulus_diagnostics()
        if uw.mpi.rank == 0:
            print(f"DIAG_ANNULUS {_L2:.12e} {_lo:.6e} {_up:.6e}")
    elif _kind == "annulus_fmg":
        _L2, _lo, _up = _annulus_fmg_diagnostics()
        if uw.mpi.rank == 0:
            print(f"DIAG_ANNULUS_FMG {_L2:.12e} {_lo:.6e} {_up:.6e}")
    else:
        _L2, _verr = _box_diagnostics()
        if uw.mpi.rank == 0:
            print(f"DIAG_BOX {_L2:.12e} {_verr:.6e}")
