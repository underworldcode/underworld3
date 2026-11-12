"""A second 3-D ``add_fault`` in parallel (#1099).

The second call repartitions the mesh the first call split, so that the new
sheet's neighbourhood stands on one rank. The 3-D split interpolated its cells
on each rank with no star-forest attached, which leaves a face on a partition
seam with one cone order on one rank and another on the next. A repartition
takes such a face from one rank and the cells on it from another, and the
placement then refused the mesh for mixed cell orientation, at 2 and at 4
ranks. The split now attaches the vertex star-forest before it interpolates.

    mpirun -n 2 python -m pytest --with-mpi tests/parallel/test_1174_second_add_fault_3d_parallel.py
    mpirun -n 4 python -m pytest --with-mpi tests/parallel/test_1174_second_add_fault_3d_parallel.py
"""
import warnings

import numpy as np
import pytest
from petsc4py import PETSc

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


@pytest.fixture(scope="module")
def first():
    return uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0,) * 3, maxCoords=(1.0,) * 3, cellSize=0.12,
        regular=False, qdegree=2, refinement=1).add_fault([SHEET_A])


def _seam_cone_failure(dm):
    """What PETSc's check of the cone order on partition seams says of ``dm``,
    the same answer on every rank; None where the ranks agree."""
    checked = dm.clone()
    checked.setOptionsPrefix("uw_seam_cones_")
    options = PETSc.Options()
    options["uw_seam_cones_dm_plex_check_interface_cones"] = ""
    failure = None
    try:
        checked.setFromOptions()
    except PETSc.Error as error:
        failure = str(error)
    finally:
        del options["uw_seam_cones_dm_plex_check_interface_cones"]
    said = [f for f in uw.mpi.comm.allgather(failure) if f]
    return said[0] if said else None


def _global_pairs(mesh, name):
    return uw.mpi.comm.allreduce(len(mesh._fault_point_pairs[name]))


def test_a_3d_split_orders_the_seam_cones_alike_on_every_rank(first):
    assert _seam_cone_failure(first.dm) is None


def test_a_second_3d_add_fault_keeps_the_first(first):
    comm = uw.mpi.comm
    second = first.add_fault([SHEET_B])
    assert sorted(second._fault_point_pairs) == ["FA", "FB"]
    assert _global_pairs(second, "FA") == _global_pairs(first, "FA") > 0
    assert _global_pairs(second, "FB") > 0

    # the mesh is whole and of one hand, and its seams are ordered alike
    v6 = _cell_volumes_signed6(second.dm)
    assert abs(comm.allreduce(float(np.abs(v6).sum()) / 6.0) - 1.0) < 1e-9
    assert comm.allreduce(int((v6 >= 0).sum())) == 0 \
        or comm.allreduce(int((v6 <= 0).sum())) == 0
    assert _seam_cone_failure(second.dm) is None

    # both faults slip and neither opens, with no warning from the solve
    stokes = uw.systems.Stokes(second)
    stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
    stokes.constitutive_model.Parameters.shear_viscosity_0 = 1.0
    drive = (-(second.X[0] - 0.5), second.X[1] - 0.5, 0.0)
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
    for name in ("FA", "FB"):
        _x, jump, normal = fault_pair_jumps(stokes, name,
                                            stokes._rotated_freeslip_info)
        opening = np.abs(np.einsum("ij,ij->i", jump, normal)).max(initial=0.0)
        slip = np.linalg.norm(jump, axis=1).max(initial=0.0)
        assert comm.allreduce(opening, op=max) < 1e-12, name
        assert comm.allreduce(slip, op=max) > 0.05, name
