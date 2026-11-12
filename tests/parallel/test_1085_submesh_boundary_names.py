"""Submesh boundary names must survive on ranks owning no boundary points (#856).

An extracted region built each rank's boundary enum from the label strata it
held locally, so a rank with no points on a boundary left that name out. A
boundary condition on it then raised ``KeyError`` on that rank while the
others went into collective solver setup and waited (on development: a natural
condition on ``Top`` hung at np 2, 4 and 8). A checkpoint written by such a
rank also lost the names. The enum now holds every name that survives on any
rank.

The region-on-one-rank partition comes from the #858 tests: the outer slab of
a ``BoxInternalBoundary`` on rank 0, so every other rank holds none of the
child's cells, and none of its ``Top``.
"""

import importlib.util
from pathlib import Path

import numpy as np
import pytest
from mpi4py import MPI
from petsc4py import PETSc

import underworld3 as uw

pytestmark = [pytest.mark.level_2, pytest.mark.mpi(min_size=2), pytest.mark.timeout(120)]

# The boundary enum of the extracted outer slab (measured; H1's review of #856).
OUTER_SLAB_BOUNDARIES = [("Top", 12), ("Right", 13), ("Left", 14), ("Internal", 15),
                         ("Outer", 102), ("All_Boundaries", 1001)]


def _region_on_rank_0():
    """The #858 tests' slab parent with the outer slab on rank 0."""
    path = Path(__file__).resolve().parent / "test_1087_submesh_prolongation_ownership.py"
    spec = importlib.util.spec_from_file_location("_submesh_ownership", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module._thin_slab_parent("single_rank_region")


def _agree(values):
    """Every rank holds the same ``values`` (checked on every rank, so a
    disagreement fails everywhere rather than leaving peers waiting)."""
    return all(other == values for other in uw.mpi.comm.allgather(values))


@pytest.mark.tier_b
def test_surviving_labels_use_global_presence():
    # The reachable state: the label exists on every rank (a filtered DM
    # carries the parent's label list), with points only on rank 0.
    dm = PETSc.DMPlex().createBoxMesh([2, 2], simplex=False, comm=uw.mpi.comm)
    try:
        dm.createLabel("OnlyOnZero")
        if uw.mpi.rank == 0:
            dm.setLabelValue("OnlyOnZero", dm.getChart()[0], 42)
        surviving = uw.discretisation.Mesh._surviving_labels(
            None, dm, {"OnlyOnZero": 42, "AbsentEverywhere": 99})
        assert _agree(surviving)
        assert surviving == {"OnlyOnZero": 42}
        assert not dm.hasLabel("AbsentEverywhere")
    finally:
        # Reached on every rank, pass or fail, so a failure cannot leave the
        # other ranks waiting in the next test's collectives.
        dm.destroy()


@pytest.mark.tier_b
@pytest.mark.parametrize("partition", ["default", "region_on_rank_0"])
def test_extracted_region_boundary_enums_are_identical_on_every_rank(partition):
    if partition == "default":
        mesh = uw.meshing.BoxInternalBoundary(
            elementRes=(12, 12), zelementRes=(10, 2), zintCoord=0.85,
            minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), simplex=True, qdegree=2)
    else:
        mesh = _region_on_rank_0()
    child = mesh.extract_region("Outer")
    names = [(b.name, b.value) for b in child.boundaries]
    assert _agree(names)
    assert names == OUTER_SLAB_BOUNDARIES


@pytest.mark.tier_b
def test_a_natural_condition_on_a_boundary_some_ranks_lack():
    # The symptom: a solver on the extracted slab with a natural condition on
    # Top, which only rank 0 holds. The enums are checked on every rank first,
    # so the old code fails here instead of hanging in solver setup.
    child = _region_on_rank_0().extract_region("Outer")
    has_top = (child.dm.getLabel("Top").getStratumSize(12) > 0
               if child.dm.hasLabel("Top") else False)
    assert uw.mpi.comm.allgather(bool(has_top)) == [True] + [False] * (uw.mpi.size - 1)
    assert _agree([b.name for b in child.boundaries])

    # u = y on the slab 0.85 < y < 1: u = 0.85 on Internal, du/dn = 1 on Top
    # and 0 on the sides. A P1 field reproduces it exactly. The natural
    # condition carries the flux -k du/dn, hence -1 (with +1 the solution is
    # 1.7 - y).
    u = uw.discretisation.MeshVariable("u", child, 1, degree=1)
    poisson = uw.systems.Poisson(child, u_Field=u)
    poisson.constitutive_model = uw.constitutive_models.DiffusionModel
    poisson.constitutive_model.Parameters.diffusivity = 1.0
    poisson.f = 0.0
    poisson.add_dirichlet_bc(0.85, "Internal")
    poisson.add_natural_bc(-1.0, "Top")
    poisson.tolerance = 1.0e-11      # the default (1e-6 or so) shows as a 4e-6 error
    poisson.solve()
    y = np.asarray(u.coords)[:, 1]
    err = float(np.abs(np.asarray(u.data)[:, 0] - y).max()) if len(y) else 0.0
    assert uw.mpi.comm.allreduce(err, op=MPI.MAX) < 1e-8


@pytest.mark.tier_b
def test_surface_extraction_on_a_region_held_by_one_rank():
    # extract_surface goes through the same name selection (with #858's
    # collective guards in front of it).
    child = _region_on_rank_0().extract_region("Outer")
    # Every rank must know the name before any rank asks for the surface:
    # otherwise the ranks without it raise and the others wait in the
    # extraction's collectives.
    assert _agree([b.name for b in child.boundaries])
    assert "Top" in [b.name for b in child.boundaries]
    surface = child.extract_surface("Top")
    start, end = surface.dm.getHeightStratum(0)
    counts = uw.mpi.comm.allgather(end - start)
    assert counts[0] > 0 and all(count == 0 for count in counts[1:])
    assert _agree([(b.name, b.value) for b in surface.boundaries])
