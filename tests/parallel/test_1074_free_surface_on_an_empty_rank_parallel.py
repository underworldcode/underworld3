"""A free surface builds and solves on a rank that holds no cells.

``FreeSurface._build_interior_diffuser`` chose between a direct nodal read of
the mesh displacement and a fallback by ``np.max`` of each rank's node
distances. On a rank that holds no nodes the array is empty and ``np.max``
raised, while the other ranks went on into the collective solve and waited.
The choice was also made rank by rank. The ranks now agree on it, and a rank
with no nodes has nothing against the direct read. The verbose report's maxima
are reduced over the ranks in the same way.

The box's cells go to every rank but the last, in bands, through a shell
partition, so from np 3 one rank is empty; below np 3 the test skips. The free
surface is built (verbose), solved and advanced one step, which reads the
displacement through the agreed choice and deforms the mesh. What it produced
must equal the same model's on the default partition, in the same run: the
integral of h_inf^2 over the surface, and of |v|^2 over the domain.
"""

import numpy as np
import pytest
import sympy
from petsc4py import PETSc

import underworld3 as uw
from underworld3.discretisation import discretisation_mesh

pytestmark = [pytest.mark.level_2, pytest.mark.tier_b, pytest.mark.timeout(300)]

# Two partitions of one model solve to the solver tolerance (1e-8) through
# different preconditioners; the bound is set from the measured difference.
SAME = 1.0e-6


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


def _free_surface_outputs(mesh):
    """The integral of h_inf^2 over the surface and of |v|^2 over the domain,
    after one solve and one step."""
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
    fs = uw.systems.FreeSurface(stokes, "Top", buoyancy_scale=50.0, verbose=True)
    fs.solve()
    fs.advance(fs.estimate_dt())
    fs.solve()
    h = fs._hinf_field.sym[0]
    v = stokes.u.sym
    return (float(uw.maths.BdIntegral(mesh, h * h, "Top").evaluate()),
            float(uw.maths.Integral(mesh, v.dot(v)).evaluate()))


def test_a_free_surface_steps_with_a_rank_that_holds_no_cells(monkeypatch):
    if uw.mpi.size < 3:
        pytest.skip("a rank is left empty from np 3; at np 2 one rank would hold every cell")
    mesh = _box(monkeypatch, empty_last=True)
    cs, ce = mesh.dm.getHeightStratum(0)
    assert 0 in uw.mpi.comm.allgather(ce - cs)
    with_empty = _free_surface_outputs(mesh)
    reference = _free_surface_outputs(_box(monkeypatch, empty_last=False))
    assert with_empty[0] > 0.0 and with_empty[1] > 0.0
    assert with_empty[0] == pytest.approx(reference[0], rel=SAME)
    assert with_empty[1] == pytest.approx(reference[1], rel=SAME)
