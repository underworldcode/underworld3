"""A second 3-D ``add_fault`` in parallel (#1099).

The second call repartitions the mesh the first call split, so that the new
sheet's neighbourhood stands on one rank. With the 3-D split interpolating its
cells on each rank before any star-forest was attached, the placement refused
that mesh for mixed cell orientation at 2 and at 4 ranks (385 cells of one
hand and 27013 of the other at 2). The split now attaches the vertex
star-forest before it interpolates, as the sheet placement does.

Two calls are compared with one call that names both sheets, at the same rank
count. In serial the two give the same mesh. In parallel they do not, because
the second sheet is placed after a repartition (27,398 cells against 27,396 at
2 ranks), and the peak slip on the second sheet differs by 1.3e-3 relative at
2 ranks and 2.0e-3 at 4; on the first sheet it is the same to six figures.
Measured on 2026-11-13 on a 96-thread Linux workstation (Intel Xeon Gold
6240R, MPICH). Each sheet has 17 node pairs and no normal jump either way,
at 1, 2 and 4 ranks; those are the hard checks.

    mpirun -n 2 python -m pytest --with-mpi tests/parallel/test_1174_second_add_fault_3d_parallel.py
    mpirun -n 4 python -m pytest --with-mpi tests/parallel/test_1174_second_add_fault_3d_parallel.py
"""
import warnings

import numpy as np
import pytest

import underworld3 as uw
from fault_fixtures import disc_sheet
from underworld3.utilities.fault_contact import fault_pair_jumps
from underworld3.utilities.place_surface import _cell_volumes_signed6

pytestmark = [pytest.mark.mpi(min_size=2), pytest.mark.level_2, pytest.mark.tier_b,
              pytest.mark.timeout(1500)]

# the two sheets of tests/test_1174: discs of radius 0.15, clear of each other
SHEET_A = ("FA",) + disc_sheet(
    normal=np.array([0.3, 0.5, 0.81]) / np.linalg.norm([0.3, 0.5, 0.81]),
    n_rim=16, centre=np.array([0.5, 0.5, 0.30]), radius=0.15)
SHEET_B = ("FB",) + disc_sheet(
    normal=np.array([-0.2, 0.4, 0.9]) / np.linalg.norm([-0.2, 0.4, 0.9]),
    n_rim=16, centre=np.array([0.5, 0.5, 0.70]), radius=0.15)


def _box():
    return uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0,) * 3, maxCoords=(1.0,) * 3, cellSize=0.12,
        regular=False, qdegree=2, refinement=1)


def _global_pairs(mesh, name):
    return uw.mpi.comm.allreduce(len(mesh._fault_point_pairs[name]))


def _slips_and_node_pairs(mesh):
    """The peak slip and the number of node pairs on each sheet, over every
    rank, with a frictionless law on both. The solve is silent and neither
    fault opens."""
    comm = uw.mpi.comm
    stokes = uw.systems.Stokes(mesh)
    stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
    stokes.constitutive_model.Parameters.shear_viscosity_0 = 1.0
    drive = (-(mesh.X[0] - 0.5), mesh.X[1] - 0.5, 0.0)
    for wall in ("Bottom", "Top", "Right", "Left", "Front", "Back"):
        stokes.add_dirichlet_bc(drive, wall)
    stokes.petsc_use_pressure_nullspace = True
    stokes.tolerance = 1.0e-6
    for name in ("FA", "FB"):
        stokes.add_fault_bc(0, name)
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        stokes.solve()
    assert stokes.snes.getConvergedReason() > 0
    slips, node_pairs = {}, {}
    for name in ("FA", "FB"):
        _x, jump, normal = fault_pair_jumps(stokes, name,
                                            stokes._rotated_freeslip_info)
        opening = np.abs(np.einsum("ij,ij->i", jump, normal)).max(initial=0.0)
        assert comm.allreduce(opening, op=max) < 1e-12, name
        slips[name] = comm.allreduce(
            np.linalg.norm(jump, axis=1).max(initial=0.0), op=max)
        node_pairs[name] = comm.allreduce(len(jump))
    return slips, node_pairs


def test_a_second_3d_add_fault_gives_what_one_call_gives():
    comm = uw.mpi.comm
    first = _box().add_fault([SHEET_A])
    second = first.add_fault([SHEET_B])
    assert sorted(second._fault_point_pairs) == ["FA", "FB"]
    assert _global_pairs(second, "FA") == _global_pairs(first, "FA") > 0
    assert _global_pairs(second, "FB") > 0

    # the mesh is whole and of one hand
    v6 = _cell_volumes_signed6(second.dm)
    assert abs(comm.allreduce(float(np.abs(v6).sum()) / 6.0) - 1.0) < 1e-9
    assert comm.allreduce(int((v6 >= 0).sum())) == 0 \
        or comm.allreduce(int((v6 <= 0).sum())) == 0

    one_slip, one_pairs = _slips_and_node_pairs(_box().add_fault([SHEET_A, SHEET_B]))
    two_slip, two_pairs = _slips_and_node_pairs(second)
    for name in ("FA", "FB"):
        assert two_pairs[name] == one_pairs[name] == 17, name
        assert one_slip[name] > 0.05, name
        assert abs(two_slip[name] - one_slip[name]) < 3e-3 * one_slip[name], name
