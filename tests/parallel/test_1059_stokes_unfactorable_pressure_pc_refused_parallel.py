"""A Stokes solve whose pressure preconditioner cannot be factored refuses (#1059).

That matrix is the pressure mass matrix divided by the viscosity, and PETSc's ILU
treats a pivot below 2.2e-14 as zero. With the viscosity large in the solver's
frame (a model in raw SI units; on this box from about 5.5e10, a limit that also
depends on the element size) the factorisation failed, the solve stopped at
iteration 0, and the fields held u = 0 with only a warning. It now raises on every
rank and names the remedy: reference quantities when one rescaling brings the
model into range, the contrast when it does not. A viscosity of 1e10 still solves,
and its answer scales as 1/viscosity.

    python scripts/mpi_supervisor.py --silence 300 -- mpirun -n 2 python -m pytest --with-mpi tests/parallel/test_1059_stokes_unfactorable_pressure_pc_refused_parallel.py
"""
import numpy as np
import pytest
import sympy
from mpi4py import MPI

import underworld3 as uw

pytestmark = [pytest.mark.level_2, pytest.mark.tier_b, pytest.mark.timeout(300)]

_count = [0]


def _box(viscosity):
    _count[0] += 1
    mesh = uw.meshing.UnstructuredSimplexBox(minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0),
                                             cellSize=0.1, qdegree=3)
    v = uw.discretisation.MeshVariable(f"V1059_{_count[0]}", mesh, 2, degree=2)
    p = uw.discretisation.MeshVariable(f"P1059_{_count[0]}", mesh, 1, degree=1)
    stokes = uw.systems.Stokes(mesh, velocityField=v, pressureField=p)
    stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
    stokes.constitutive_model.Parameters.shear_viscosity_0 = viscosity
    x, y = mesh.X
    stokes.bodyforce = sympy.Matrix([0, sympy.sin(sympy.pi * x)])
    for wall in ("Left", "Right", "Top", "Bottom"):
        stokes.add_dirichlet_bc((0.0, 0.0), wall)
    stokes.petsc_options.delValue("ksp_monitor")
    return stokes, v


def _speed(v):
    return uw.mpi.comm.allreduce(float(np.abs(np.asarray(v.array)).max(initial=0.0)),
                                 op=MPI.MAX)


@pytest.mark.parametrize("viscosity", [1.0e12, 1.0e21])
def test_an_unfactorable_pressure_preconditioner_is_refused_on_every_rank(viscosity):
    stokes, _ = _box(viscosity)
    message = ""
    try:
        stokes.solve()
    except RuntimeError as error:
        message = str(error)
    messages = uw.mpi.comm.allgather(message)
    assert all(messages), f"some rank did not refuse: {messages}"
    assert all("set_reference_quantities" in m and "#1059" in m for m in messages), messages[0]
    assert not any("contrast" in m for m in messages), messages[0]


def test_a_viscosity_contrast_is_refused_with_the_contrast_remedy():
    """1 below y = 0.8 and 1e12 above: only some entries are tiny. Reference
    quantities rescale every viscosity together and cannot help, so the message
    must not offer them."""
    import sympy as sp
    stokes, _ = _box(1.0)
    y = stokes.mesh.X[1]
    stokes.constitutive_model.Parameters.shear_viscosity_0 = sp.Piecewise(
        (1.0e12, y > 0.8), (1.0, True))
    message = ""
    try:
        stokes.solve()
    except RuntimeError as error:
        message = str(error)
    messages = uw.mpi.comm.allgather(message)
    assert all("contrast" in m and "zeropivot" in m for m in messages), messages[0]
    assert not any("set_reference_quantities" in m for m in messages), messages[0]


def test_a_viscous_model_with_a_moderate_contrast_gets_the_scale_remedy():
    """1e9 below y = 0.8 and 1e12 above: a contrast of 1e3, too viscous as a whole.
    The same model divided by 1e9 solves, so the remedy is reference quantities,
    not the contrast."""
    import sympy as sp
    stokes, _ = _box(1.0)
    y = stokes.mesh.X[1]
    stokes.constitutive_model.Parameters.shear_viscosity_0 = sp.Piecewise(
        (1.0e12, y > 0.8), (1.0e9, True))
    message = ""
    try:
        stokes.solve()
    except RuntimeError as error:
        message = str(error)
    messages = uw.mpi.comm.allgather(message)
    assert all("set_reference_quantities" in m for m in messages), messages[0]
    assert not any("contrast" in m for m in messages), messages[0]
    rescaled, _ = _box(1.0)
    rescaled.constitutive_model.Parameters.shear_viscosity_0 = sp.Piecewise(
        (1.0e3, rescaled.mesh.X[1] > 0.8), (1.0, True))
    rescaled.solve()
    assert rescaled.snes.getConvergedReason() > 0


def test_the_same_solver_solves_once_the_viscosity_is_fixed():
    """A preconditioner that failed kept its failed state, so the next solve on the
    same solver failed again at iteration 0 with a healthy matrix."""
    stokes, v = _box(1.0e12)
    with pytest.raises(RuntimeError):
        stokes.solve()
    stokes.constitutive_model.Parameters.shear_viscosity_0 = 1.0
    stokes.solve()
    assert stokes.snes.getConvergedReason() > 0
    unit, v_unit = _box(1.0)
    unit.solve()
    assert _speed(v) == pytest.approx(_speed(v_unit), rel=1e-6)


def test_a_viscosity_of_1e10_still_solves():
    stokes, v = _box(1.0e10)
    stokes.solve()
    assert stokes.snes.getConvergedReason() > 0
    unit, v_unit = _box(1.0)
    unit.solve()
    assert _speed(v) == pytest.approx(_speed(v_unit) / 1.0e10, rel=1e-6)
