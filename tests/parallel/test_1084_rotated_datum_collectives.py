"""Prescribed FE wall data must be evaluated in the same order on empty ranks.

Run serial or with mpiexec. The first two tests exercise the datum schedule and
evaluator alone; the third drives ``build_rotation`` through a Stokes solve.
"""

import numpy as np
import pytest
import sympy
import underworld3 as uw

from underworld3.utilities import rotated_bc
from underworld3.utilities.rotated_bc import _datum_batches, _eval_datum


pytestmark = [pytest.mark.level_1, pytest.mark.tier_b]


@pytest.mark.parametrize("empty_rank", [1, 2])
def test_registered_boundaries_determine_collective_schedule_not_local_nodes(empty_rank):
    rank = uw.mpi.rank
    boundaries = ["Upper", "Lower"]
    # All ranks have the same registration, but zero, one or two local groups.
    owned = {0: [(10, 1, "Upper", 1.0), (11, 2, "Lower", -1.0)],
             1: [(20, 2, "Lower", 1.0)]}.get(rank, [])
    if rank == empty_rank:
        owned = []
    coordinates = {1: [2.0, 0.0], 2: [1.0, 0.0]}
    batches = list(_datum_batches(boundaries, {"Upper": 2.0, "Lower": 3.0}, owned, coordinates, 2))
    # Gather before asserting: a rank-local assertion ahead of a collective turns a
    # schedule mismatch into a hang instead of a failure.
    schedules = uw.mpi.comm.allgather(
        [(datum, coords.shape, len(items)) for datum, coords, items in batches])
    local_groups = {0: [1, 1], 1: [0, 1]}
    for r, schedule in enumerate(schedules):
        counts = [0, 0] if r == empty_rank else local_groups.get(r, [0, 0])
        assert schedule == [(2.0, (counts[0], 2), counts[0]), (3.0, (counts[1], 2), counts[1])]


def test_fe_datum_evaluation_accepts_empty_coordinate_batches_on_other_ranks():
    mesh = uw.meshing.Annulus(radiusInner=1.22, radiusOuter=2.22, cellSize=1/32, qdegree=2)
    field = uw.discretisation.MeshVariable("DatumCollectiveP2", mesh, 1, degree=2, continuous=True)
    field.array[:, 0, 0] = 0.13*field.coords[:, 0]-0.2*field.coords[:, 1]
    # Only one rank queries nodes; every rank must still enter FE evaluation.
    coords = np.asarray(field.coords[:3]) if uw.mpi.rank == 0 else np.empty((0, 2))
    for _ in range(2):
        actual = _eval_datum(field.sym[0], coords, None)
        expected = 0.13*coords[:, 0]-0.2*coords[:, 1]
        np.testing.assert_allclose(actual, expected, atol=1e-12)


def test_build_rotation_evaluates_fe_data_on_every_rank_in_registration_order(monkeypatch):
    """FE data on two walls, through ``build_rotation`` in a real solve.

    Ranks owning nodes of one wall only used to evaluate one field while their
    peers evaluated two, and the solve hung before its first Stokes iteration
    (three ranks, this mesh). The data prescribe the uniform flow v = (0, 1/10)
    on the unit box, whose energy is exactly 1/100.
    """
    mesh = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), cellSize=1 / 16, qdegree=3)
    v = uw.discretisation.MeshVariable("DatumRotV", mesh, 2, degree=2, continuous=True)
    p = uw.discretisation.MeshVariable("DatumRotP", mesh, 1, degree=1, continuous=True)
    top = uw.discretisation.MeshVariable("DatumRotTop", mesh, 1, degree=2, continuous=True)
    bottom = uw.discretisation.MeshVariable("DatumRotBottom", mesh, 1, degree=2, continuous=True)
    top.array[:, 0, 0] = 0.1
    bottom.array[:, 0, 0] = -0.1

    schedule = []
    evaluate = rotated_bc._eval_datum

    def recording_evaluate(dspec, coords, solver):
        schedule.append(str(dspec))
        return evaluate(dspec, coords, solver)

    monkeypatch.setattr(rotated_bc, "_eval_datum", recording_evaluate)

    stokes = uw.systems.Stokes(mesh, velocityField=v, pressureField=p)
    stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
    stokes.constitutive_model.Parameters.shear_viscosity_0 = 1
    stokes.add_essential_bc((0.0, None), "Left")
    stokes.add_essential_bc((0.0, None), "Right")
    stokes.add_rotated_freeslip_bc(top.sym[0], "Top", normal=sympy.Matrix([[0, 1]]))
    stokes.add_rotated_freeslip_bc(bottom.sym[0], "Bottom", normal=sympy.Matrix([[0, -1]]))
    stokes.petsc_use_pressure_nullspace = True
    stokes.tolerance = 1.0e-9
    stokes.solve()

    expected = [str(top.sym[0]), str(bottom.sym[0])]
    for rank_schedule in uw.mpi.comm.allgather(schedule):
        assert rank_schedule == expected
    energy = float(uw.maths.Integral(mesh, v.sym.dot(v.sym)).evaluate())
    assert energy == pytest.approx(0.01, rel=1.0e-8)
