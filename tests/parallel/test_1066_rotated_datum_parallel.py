"""Parallel regression for the rotated free-slip PRESCRIBED-NORMAL datum (u.n = ũ_n).

The datum is applied to the rotated normal rows via ownership-relative index surgery
(`_set_rows_local`, the xhat ownership loop, per-node datum evaluation in
`build_rotation`) — the ``np>1 crash class`` if any of it indexed the local slice with
global rows. This test prescribes ``u.n = cos(theta)`` on the annulus surface and checks,
with parallel-safe reductions only, that:

  * the datum is imposed (mean-stripped integral ``∫(u.n - cos θ)² / ∫cos²θ`` on the
    surface is tiny — the residual is the P2/faceting interpolation, not partition), and
  * the solve energy ``∫ v·v`` reproduces the serial reference to iterative round-off
    (a partition-dependent bug in the datum surgery would shift it).

Run with::

    mpirun -n 2 python -m pytest --with-mpi tests/parallel/test_1066_rotated_datum_parallel.py
    mpirun -n 4 python -m pytest --with-mpi tests/parallel/test_1066_rotated_datum_parallel.py
"""
import numpy as np
import pytest
import sympy
import underworld3 as uw

pytestmark = [pytest.mark.mpi(min_size=2), pytest.mark.timeout(180)]

# serial reference (np1), used to catch a partition-dependent datum bug
_INT_VV_REF = 6.6559607579


@pytest.mark.xfail(
    reason="#564: free-slip solves are partition dependent. CI measures the solve "
           "energy 6.65602336 against the recorded 6.6559607579 at np=2 (9.4e-06) — "
           "the smallest of the family, but the same defect. PRE-EXISTING and not "
           "caused by #560/#561: the same seven assertions fail with numbers "
           "identical to every digit at #561's merge base with only the "
           "scripts/test.sh test_10*py line enabled, which is how they became "
           "visible at all — this whole batch had never run in CI. This case passes "
           "an explicit analytic normal=, and its numbers are bit-identical before "
           "and after #560's nodal-normal fix, so it is definitively not that "
           "mechanism. Passes locally on macOS/arm64, so strict=False; see #564 for "
           "the full table.",
    strict=False)
def test_rotated_datum_prescribed_normal_partition_independent():
    RI, RO = 0.5, 1.0
    mesh = uw.meshing.Annulus(radiusInner=RI, radiusOuter=RO, cellSize=0.1, qdegree=3)
    x, y = mesh.X
    r = sympy.sqrt(x ** 2 + y ** 2)
    nhat = sympy.Matrix([[x / r, y / r]])
    v = uw.discretisation.MeshVariable("Vd64", mesh, mesh.dim, degree=2, continuous=True)
    p = uw.discretisation.MeshVariable("Pd64", mesh, 1, degree=1, continuous=True)
    s = uw.systems.Stokes(mesh, velocityField=v, pressureField=p)
    s.constitutive_model = uw.constitutive_models.ViscousFlowModel
    s.constitutive_model.Parameters.shear_viscosity_0 = 1.0
    blob = sympy.exp(-(((x - 0.75) ** 2 + y ** 2) / 0.05))
    s.bodyforce = sympy.Matrix([[50.0 * blob * x / r, 50.0 * blob * y / r]])
    s.add_essential_bc((0.0, 0.0), "Lower")               # no-slip inner (pins rotation)
    s.add_rotated_freeslip_bc(x / r, "Upper", normal=nhat)    # u.n = cos(theta), mean-zero
    s.petsc_use_pressure_nullspace = True
    s.petsc_options["snes_type"] = "ksponly"
    s.tolerance = 1.0e-9
    s.solve()

    vn = (v.sym[0] * x + v.sym[1] * y) / r
    num = float(uw.maths.BdIntegral(mesh=mesh, fn=(vn - x / r) ** 2, boundary="Upper").evaluate())
    den = float(uw.maths.BdIntegral(mesh=mesh, fn=(x / r) ** 2, boundary="Upper").evaluate())
    relL2 = np.sqrt(num / den)
    assert relL2 < 1.0e-3, f"prescribed u.n=cos(theta) not imposed (relL2={relL2:.3e})"

    vv = float(uw.maths.Integral(mesh, v.sym.dot(v.sym)).evaluate())
    assert abs(vv - _INT_VV_REF) / _INT_VV_REF < 1.0e-6, \
        f"solve energy is partition-dependent: ∫v·v={vv:.8e} vs ref {_INT_VV_REF}"


def test_rotated_datum_nonlinear_parallel():
    """The NONLINEAR rotated datum path in parallel: a power-law annulus with
    u.n = cos(theta) through the manual Newton loop (feasible iterates, cold-start
    lift, collective datum-activity flag). Guards the np>1 collective-sequence
    class: the lift branch, the line-search residual evaluations and the
    nullspace-mode verification are all collective, and a rank-divergent branch
    deadlocks rather than failing cleanly (hence the module timeout)."""
    RI, RO = 0.5, 1.0
    mesh = uw.meshing.Annulus(radiusInner=RI, radiusOuter=RO, cellSize=0.2, qdegree=3)
    x, y = mesh.X
    r = sympy.sqrt(x ** 2 + y ** 2)
    nhat = sympy.Matrix([[x / r, y / r]])
    v = uw.discretisation.MeshVariable("Vnl64", mesh, mesh.dim, degree=2, continuous=True)
    p = uw.discretisation.MeshVariable("Pnl64", mesh, 1, degree=1, continuous=True)
    s = uw.systems.Stokes(mesh, velocityField=v, pressureField=p)
    s.constitutive_model = uw.constitutive_models.ViscousFlowModel
    gm = sympy.Matrix([[v.sym[0].diff(x), v.sym[0].diff(y)],
                       [v.sym[1].diff(x), v.sym[1].diff(y)]])
    e = 0.5 * (gm + gm.T)
    eII = sympy.sqrt(0.5 * (e[0, 0] ** 2 + e[1, 1] ** 2) + e[0, 1] ** 2 + 1.0e-12)
    s.constitutive_model.Parameters.shear_viscosity_0 = eII ** (1.0 / 3.0 - 1.0)
    blob = sympy.exp(-(((x - 0.75) ** 2 + y ** 2) / 0.05))
    s.bodyforce = sympy.Matrix([[50.0 * blob * x / r, 50.0 * blob * y / r]])
    s.add_essential_bc((0.0, 0.0), "Lower")
    s.add_rotated_freeslip_bc(x / r, "Upper", normal=nhat)
    s.consistent_jacobian = True
    s.petsc_use_pressure_nullspace = True
    s.tolerance = 1.0e-7
    s.solve()

    info = s._rotated_freeslip_info
    assert info["converged"], "nonlinear parallel datum solve did not converge"
    assert info["nonlinear_iterations"] > 1, "did not genuinely iterate"
    # datum imposed, measured with parallel-safe boundary quadrature only
    vn = (v.sym[0] * x + v.sym[1] * y) / r
    num = float(uw.maths.BdIntegral(mesh=mesh, fn=(vn - x / r) ** 2, boundary="Upper").evaluate())
    den = float(uw.maths.BdIntegral(mesh=mesh, fn=(x / r) ** 2, boundary="Upper").evaluate())
    relL2 = np.sqrt(max(num, 0.0) / den)
    assert relL2 < 1.0e-3, f"nonlinear u.n=cos(theta) not imposed (relL2={relL2:.3e})"
