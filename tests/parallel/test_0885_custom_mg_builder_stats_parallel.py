"""custom_mg's builder statistics belong to the call, and count only misplaced nodes (#885).

Two defects. The barycentric builder wrote its statistics to a module global,
which the RBF builder never wrote, so after an RBF build the "auto" check read the
previous barycentric build's count (``outside_hull`` 2 where the RBF build had
none). And ``outside_hull`` counted every fine node outside its rank's LOCAL coarse
hull, including nodes a curved boundary's snapping put beyond the coarse chords:
on an annulus every native pair was rebuilt across partitions (outside_hull 96 and
192 summed over ranks at np 2 and at np 4, the same at both, so geometry and not
the partition), an all-gather and a Delaunay per rank per level that cannot fix
those rows anyway.

The statistics are now filled into a dict the caller passes, and the rebuild
counts only fine nodes outside the local hull but inside the global one: the
nodes a cross-partition build can place.

    python scripts/mpi_supervisor.py --silence 300 -- mpirun -n 2 python -m pytest --with-mpi tests/parallel/test_0885_custom_mg_builder_stats_parallel.py
"""
import numpy as np
import pytest
import sympy

import underworld3 as uw
from underworld3.utilities import custom_mg

pytestmark = [pytest.mark.level_2, pytest.mark.tier_b, pytest.mark.timeout(300)]


def test_each_build_reports_its_own_statistics():
    coarse = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    outside = np.array([[2.0, 2.0], [3.0, 3.0], [0.5, 0.5]])
    inside = np.array([[0.25, 0.25], [0.5, 0.5], [0.75, 0.6]])
    first = {}
    custom_mg.barycentric_prolongation(coarse, outside, stats=first)
    assert first["outside_hull"] == 2
    second = {}
    custom_mg._node_transfer_no_repair(custom_mg.rbf_prolongation, coarse, inside,
                                       stats=second)
    assert second["outside_hull"] == 0
    assert not hasattr(custom_mg, "last_builder_stats")


def test_boundary_snapping_alone_does_not_send_a_level_across_partitions(monkeypatch):
    if uw.mpi.size < 2:
        pytest.skip("the cross-partition rebuild is a parallel path")
    rebuilds = []
    original = custom_mg._build_crosspart_transfer

    def counted(*args, **kwargs):
        rebuilds.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(custom_mg, "_build_crosspart_transfer", counted)
    mesh = uw.meshing.Annulus(radiusOuter=1.0, radiusInner=0.5, cellSize=0.2, refinement=2)
    T = uw.discretisation.MeshVariable("T0885", mesh, 1, degree=2)
    poisson = uw.systems.Poisson(mesh, u_Field=T)
    poisson.constitutive_model = uw.constitutive_models.DiffusionModel
    poisson.constitutive_model.Parameters.diffusivity = 1.0
    x, y = mesh.X
    poisson.f = sympy.sin(3 * x) * sympy.cos(2 * y)
    poisson.add_dirichlet_bc(0.0, "Upper")
    poisson.add_dirichlet_bc(0.0, "Lower")
    poisson.petsc_options.delValue("ksp_monitor")
    custom_mg.set_custom_fmg(poisson, mesh._coarse_level_meshes()[:-1])
    poisson.solve()
    assert poisson.snes.getConvergedReason() > 0
    assert uw.mpi.comm.allreduce(len(rebuilds)) == 0, (
        f"{uw.mpi.comm.allreduce(len(rebuilds))} cross-partition rebuild(s) on the annulus")


def test_a_node_another_rank_can_place_is_still_counted():
    """The positive case: a fine node outside its rank's coarse hull but inside the
    whole level's is one a cross-partition build places, so it counts. Each rank's
    coarse cloud is a vertical strip of the unit square; its fine nodes are a few of
    its own, a few from the next rank's strip (counted) and a few beyond the square
    (outside every hull, not counted)."""
    if uw.mpi.size < 2:
        pytest.skip("needs a rank whose fine nodes lie in another rank's strip")
    size, rank = uw.mpi.size, uw.mpi.rank
    xs = np.linspace(rank / size, (rank + 1) / size, 4)
    coarse = np.array([[x, y] for x in xs for y in np.linspace(0.0, 1.0, 4)])
    middle = (rank + 0.5) / size
    there = ((rank + 1) % size + 0.5) / size
    fine = np.array([[middle, 0.3], [middle, 0.6],          # its own strip
                     [there, 0.3], [there, 0.6],            # the next rank's strip
                     [1.4, 0.5], [0.5, -0.3]])              # beyond the square
    stats = {}
    custom_mg.barycentric_prolongation(coarse, fine, repair_orphans=False, stats=stats)
    assert stats["outside_hull"] == 4
    from petsc4py import PETSc
    counted = custom_mg._misplaced_outside_global_hull(coarse, fine, stats, PETSc.COMM_WORLD)
    assert counted == 2 * size


def test_no_hull_check_when_cross_partition_is_off(monkeypatch):
    if uw.mpi.size < 2:
        pytest.skip("the hull check is a parallel path")
    calls = []
    original = custom_mg._misplaced_outside_global_hull

    def counted(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(custom_mg, "_misplaced_outside_global_hull", counted)
    mesh = uw.meshing.Annulus(radiusOuter=1.0, radiusInner=0.5, cellSize=0.2, refinement=2)
    T = uw.discretisation.MeshVariable("T0885off", mesh, 1, degree=2)
    poisson = uw.systems.Poisson(mesh, u_Field=T)
    poisson.constitutive_model = uw.constitutive_models.DiffusionModel
    poisson.constitutive_model.Parameters.diffusivity = 1.0
    x, y = mesh.X
    poisson.f = sympy.sin(3 * x) * sympy.cos(2 * y)
    poisson.add_dirichlet_bc(0.0, "Upper")
    poisson.add_dirichlet_bc(0.0, "Lower")
    poisson.petsc_options.delValue("ksp_monitor")
    custom_mg.set_custom_fmg(poisson, mesh._coarse_level_meshes()[:-1], cross_partition=False)
    poisson.solve()
    assert poisson.snes.getConvergedReason() > 0
    assert uw.mpi.comm.allreduce(len(calls)) == 0
