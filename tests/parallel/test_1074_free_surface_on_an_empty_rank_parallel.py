"""A free surface builds and solves on a rank that holds no cells.

``FreeSurface._build_interior_diffuser`` chose between a direct nodal read of
the mesh displacement and a fallback by ``np.max`` of each rank's node
distances. On a rank that holds no nodes the array is empty and ``np.max``
raised, while the other ranks went on into the collective solve and waited
(measured at np 3 on a 96-thread Linux workstation, MPICH, by H1, the session
that reviews pull requests and runs the parallel tests). The choice was also
made rank by rank. The ranks now agree on it, and a rank with no nodes has
nothing against the direct read. The verbose report's maxima are reduced over
the ranks in the same way.

The box's cells go to every rank but the last, in bands, through a shell
partition, so from np 3 one rank is empty. The free-surface velocity's
integral of |v|^2 must equal that of the same model on the default partition,
in the same run.
"""

import numpy as np
import pytest
import sympy
from petsc4py import PETSc

import underworld3 as uw
from underworld3.discretisation import discretisation_mesh

pytestmark = [pytest.mark.level_2, pytest.mark.tier_b, pytest.mark.timeout(300)]


def _box(monkeypatch, empty_last):
    read_plex = discretisation_mesh._from_plexh5

    def read_and_leave_the_last_rank_empty(*args, **kwargs):
        result = read_plex(*args, **kwargs)
        dm = result[1] if isinstance(result, tuple) else result
        c_start, c_end = dm.getHeightStratum(0)
        ranks = max(uw.mpi.size - 1, 1)
        assign = (np.arange(c_end - c_start) * ranks) // max(c_end - c_start, 1)
        part = dm.getPartitioner()
        part.setType(PETSc.Partitioner.Type.SHELL)
        part.setShellPartition(
            uw.mpi.size,
            sizes=np.bincount(assign, minlength=uw.mpi.size).astype(PETSc.IntType),
            points=np.argsort(assign, kind="stable").astype(PETSc.IntType))
        return result

    if empty_last and uw.mpi.size > 2:
        monkeypatch.setattr(discretisation_mesh, "_from_plexh5", read_and_leave_the_last_rank_empty)
    mesh = uw.meshing.UnstructuredSimplexBox(minCoords=(0.0, 0.0), maxCoords=(1.0, 0.5),
                                             cellSize=1.0 / 16, qdegree=3)
    monkeypatch.undo()
    return mesh


def _free_surface_energy(mesh):
    x = mesh.X
    stokes = uw.systems.Stokes(mesh)
    stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
    stokes.constitutive_model.Parameters.shear_viscosity_0 = 1.0
    stokes.bodyforce = 50.0 * sympy.exp(-(((x[0] - 0.5) ** 2 + (x[1] - 0.25) ** 2) / 0.02)) \
        * sympy.Matrix([[0, 1]])
    stokes.add_essential_bc((0.0, 0.0), "Bottom")
    stokes.add_rotated_freeslip_bc(0.0, "Left")
    stokes.add_rotated_freeslip_bc(0.0, "Right")
    stokes.tolerance = 1.0e-8
    fs = uw.systems.FreeSurface(stokes, "Top", buoyancy_scale=50.0)
    fs.solve()
    v = stokes.u.sym
    return float(uw.maths.Integral(mesh, v.dot(v)).evaluate())


def test_a_free_surface_solves_with_a_rank_that_holds_no_cells(monkeypatch):
    mesh = _box(monkeypatch, empty_last=True)
    cs, ce = mesh.dm.getHeightStratum(0)
    counts = uw.mpi.comm.allgather(ce - cs)
    if uw.mpi.size > 2:
        assert 0 in counts
    with_empty = _free_surface_energy(mesh)
    reference = _free_surface_energy(_box(monkeypatch, empty_last=False))
    assert with_empty > 0.0
    assert with_empty == pytest.approx(reference, rel=1e-6)
