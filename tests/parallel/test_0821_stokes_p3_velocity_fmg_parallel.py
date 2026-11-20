"""A Stokes solve with a P3 velocity on a refined mesh takes the custom-P route (#821).

With a mesh hierarchy, ``preconditioner="auto"`` sends the Stokes velocity block
to native geometric FMG, which needs PETSc's DMCreateInjection between the
refined levels. PETSc cannot build that injection for a Lagrange space of degree
3 or more (error 62, "Could not locate matching functional for injection"), on
simplices and quads alike, so a P3 velocity failed at its first solve. The velocity
block of such a solver now takes the custom-P transfers over the mesh hierarchy,
which need no injection, and the decision is recorded in ``pc_fallbacks``. The
answer equals the one from ``set_custom_fmg`` on the same levels, the issue's
workaround. A P2 velocity keeps the native route.

    python scripts/mpi_supervisor.py --silence 300 -- mpirun -n 2 python -m pytest --with-mpi tests/parallel/test_0821_stokes_p3_velocity_fmg_parallel.py
"""
import numpy as np
import pytest
import sympy
from mpi4py import MPI

import underworld3 as uw

pytestmark = [pytest.mark.level_2, pytest.mark.tier_b, pytest.mark.timeout(300)]

_count = [0]


def _stokes(simplex, velocity_degree):
    _count[0] += 1
    if simplex:
        mesh = uw.meshing.UnstructuredSimplexBox(cellSize=0.25, refinement=1,
                                                 qdegree=velocity_degree + 2)
    else:
        mesh = uw.meshing.StructuredQuadBox(elementRes=(4, 4), refinement=1,
                                            qdegree=velocity_degree + 2)
    v = uw.discretisation.MeshVariable(f"V0821_{_count[0]}", mesh, 2, degree=velocity_degree)
    p = uw.discretisation.MeshVariable(f"P0821_{_count[0]}", mesh, 1,
                                       degree=velocity_degree - 1)
    stokes = uw.systems.Stokes(mesh, velocityField=v, pressureField=p)
    stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
    stokes.constitutive_model.Parameters.shear_viscosity_0 = 1.0
    x, y = mesh.X
    stokes.bodyforce = sympy.Matrix([0, sympy.sin(sympy.pi * x)])
    for wall in ("Left", "Right", "Top", "Bottom"):
        stokes.add_dirichlet_bc((0.0, 0.0), wall)
    stokes.tolerance = 1.0e-8
    stokes.petsc_options.delValue("ksp_monitor")
    return stokes, v


def _speed(v):
    return uw.mpi.comm.allreduce(float(np.abs(np.asarray(v.array)).max(initial=0.0)),
                                 op=MPI.MAX)


@pytest.mark.parametrize("simplex", [True, False], ids=["simplices", "quads"])
def test_a_p3_velocity_solves_by_the_custom_p_route(simplex):
    stokes, v = _stokes(simplex, 3)
    stokes.solve()
    assert stokes.snes.getConvergedReason() > 0
    assert "high_degree_velocity_gate" in stokes.pc_fallbacks
    reference, v_ref = _stokes(simplex, 3)
    reference.set_custom_fmg(reference.mesh._coarse_level_meshes()[:-1], field_id=0)
    reference.solve()
    assert _speed(v) == pytest.approx(_speed(v_ref), rel=1e-6)


def test_a_p2_velocity_keeps_the_native_route():
    stokes, _ = _stokes(True, 2)
    stokes.solve()
    assert stokes.snes.getConvergedReason() > 0
    assert "high_degree_velocity_gate" not in stokes.pc_fallbacks
    assert stokes.petsc_options.getString("fieldsplit_velocity_pc_type", "") == "mg"
