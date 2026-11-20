"""A refined base whose cells all sit on one rank keeps its partition through
the coarse-level wraps, so ``add_fault`` partitions its add-on (UNFILED 44).

``Mesh._wrap_coarse_level`` wraps each coarse level of a refined hierarchy as a
Mesh. With every cell on one rank, PETSc reports the level's DM as not
distributed (``DMPlexIsDistributed``), and the Mesh constructor distributed it
afresh: the wrap no longer nested under the finer level, and at np 2
``add_fault`` refused with "84 fine cell(s) have no local parent cell on the
level below; the add-on levels are not co-partitioned". np 3 with one rank
empty worked, since two ranks then held cells. A wrap keeps the partition the
level was given.

The base here is a gmsh box (cellSize 0.25, refinement 1) whose cells all go
to rank 0 (a shell partition set on the loaded DM).
"""
import numpy as np
import pytest

import underworld3 as uw

pytestmark = [pytest.mark.mpi(min_size=2), pytest.mark.level_1, pytest.mark.tier_b,
              pytest.mark.timeout(300)]

TRACE = np.array([[0.3, 0.45], [0.7, 0.55]])


@pytest.fixture
def every_cell_on_rank_0(monkeypatch):
    from petsc4py import PETSc
    from underworld3.discretisation import discretisation_mesh as dmod
    original = dmod._from_plexh5

    def loaded(*a, **k):
        out = original(*a, **k)
        dm = out[1] if isinstance(out, tuple) else out
        cS, cE = dm.getHeightStratum(0)
        sizes = np.zeros(uw.mpi.size, dtype=np.int32)
        sizes[0] = cE - cS
        part = dm.getPartitioner()
        part.setType(PETSc.Partitioner.Type.SHELL)
        part.setShellPartition(uw.mpi.size, sizes=sizes,
                               points=np.arange(cE - cS, dtype=np.int32))
        return out

    monkeypatch.setattr(dmod, "_from_plexh5", loaded)


def _cells(dm):
    return uw.mpi.comm.allgather(dm.getHeightStratum(0)[1] - dm.getHeightStratum(0)[0])


def _base():
    base = uw.meshing.UnstructuredSimplexBox(minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0),
                                             cellSize=0.25, refinement=1, qdegree=3)
    held = _cells(base.dm)
    assert held[0] > 0 and not any(held[1:]), held
    return base


def test_a_coarse_wrap_keeps_the_level_partition(every_cell_on_rank_0):
    base = _base()
    for level in base.dm_hierarchy[:-1]:
        assert _cells(base._wrap_coarse_level(level).dm) == _cells(level)


def test_add_fault_partitions_the_add_on(every_cell_on_rank_0):
    child = _base().add_fault([("Flt", TRACE)])
    pairs = uw.mpi.comm.allreduce(sum(len(v) for v in child._fault_point_pairs.values()))
    assert pairs > 0


def test_the_faulted_mesh_solves(every_cell_on_rank_0):
    mesh = _base().add_fault([("Flt", TRACE)])
    x, y = mesh.X
    stokes = uw.systems.Stokes(mesh)
    stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
    stokes.constitutive_model.Parameters.shear_viscosity_0 = 1
    stokes.tolerance = 1e-6
    stokes.petsc_use_pressure_nullspace = True
    for wall in ("Top", "Bottom", "Left", "Right"):
        stokes.add_dirichlet_bc((y - 0.5, 0.0), wall)
    stokes.add_fault_bc(0, "Flt")
    stokes.solve()
    u = np.asarray(stokes.Unknowns.u.array)
    peak = uw.mpi.comm.allreduce(float(np.abs(u).max()) if u.size else 0.0, op=uw.MPI.MAX)
    finite = uw.mpi.comm.allreduce(int(np.isfinite(u).all()), op=uw.MPI.MIN)
    assert finite and peak > 0.1, peak             # the walls shear at up to 0.5
