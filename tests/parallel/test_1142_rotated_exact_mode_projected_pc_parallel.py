"""The exact-rotation projection in the rotated velocity preconditioner, in parallel (#963).

At np 2 the inner velocity solve of test_1064's analytic-normal annulus stalled under
some OpenBLAS kernels until the job was killed; the projection fixes that (the serial
file, tests/test_1142_rotated_exact_mode_projected_pc.py, carries the forced-kernel
check). Here, on every rank:

- one mode is projected, the inner velocity solve converges in a few cycles, and the
  outer solve converges;
- destroying the solve's workspace destroys the multigrid the projection wraps. In
  parallel petsc4py defers the destruction of a dropped reference, and before the fix
  the whole hierarchy outlived every solve: 533 to 1106 MB over 31 adjoint solves at
  np 2, against 535 to 551 MB without the projection (hyperion, on 108f9dee).

    mpirun -n 2 python -m pytest --with-mpi tests/parallel/test_1142_rotated_exact_mode_projected_pc_parallel.py
"""
import numpy as np
import pytest
import sympy

import underworld3 as uw
from underworld3.utilities import custom_mg, rotated_bc

pytestmark = [pytest.mark.mpi(min_size=2), pytest.mark.level_2, pytest.mark.tier_b,
              pytest.mark.timeout(300)]

RI, RO = 0.5, 1.0


def _wrap(dm, m0):
    return uw.discretisation.Mesh(dm.clone(), simplex=True, qdegree=3,
                                  coordinate_system_type=m0.CoordinateSystem.coordinate_type,
                                  boundaries=m0.boundaries)


def test_the_projection_holds_and_is_released_on_every_rank():
    m0 = uw.meshing.Annulus(radiusInner=RI, radiusOuter=RO, cellSize=0.2, qdegree=3)
    dm1 = m0.dm.refine()
    dm2 = dm1.refine()
    coarse = [m0, _wrap(dm1, m0)]
    fine = _wrap(dm2, m0)
    x, y = fine.X
    r = sympy.sqrt(x ** 2 + y ** 2)
    th = sympy.atan2(y, x)
    v = uw.discretisation.MeshVariable("Vp", fine, 2, degree=2, continuous=True)
    p = uw.discretisation.MeshVariable("Pp", fine, 1, degree=1, continuous=True)
    s = uw.systems.Stokes(fine, velocityField=v, pressureField=p)
    s.constitutive_model = uw.constitutive_models.ViscousFlowModel
    s.constitutive_model.Parameters.shear_viscosity_0 = 1.0
    s.bodyforce = sympy.Matrix([[x / r * sympy.cos(4 * th) * (r - RI) * (RO - r) * 40.0,
                                 y / r * sympy.cos(4 * th) * (r - RI) * (RO - r) * 40.0]])
    nhat = sympy.Matrix([[x / r, y / r]])
    s.add_rotated_freeslip_bc(0, "Lower", normal=nhat)
    s.add_rotated_freeslip_bc(0, "Upper", normal=nhat)
    s.tolerance = 1e-9
    s.saddle_preconditioner = 1.0
    s.petsc_use_pressure_nullspace = True
    s.petsc_options["snes_type"] = "ksponly"
    custom_mg.set_custom_fmg(s, coarse, builder="barycentric", field_id=0)
    s.solve()
    info = s._rotated_freeslip_info
    solved = uw.mpi.comm.allgather((info["ksp_reason"] > 0, info["projected_null_modes"],
                                    max(info["vel_its_last"]) <= 10))

    ctx = s._rotated_linear_cache["ctx"]
    inner = ctx["pc"].getFieldSplitSubKSP()[0].getPC().getPythonContext().inner
    rotated_bc._destroy_rotated_ksp_ctx(ctx)
    released = uw.mpi.comm.allgather(not inner.handle)

    assert set(solved) == {(True, 1, True)}, (solved, info["vel_its_last"])
    assert all(released), released
