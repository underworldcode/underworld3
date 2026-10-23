# Reproducer kept out of the suite until the np 8 solve divergence is fixed.
# Run: mpirun -np 8 python -m pytest --with-mpi this_file
"""Custom-P multigrid on more ranks than its coarsest level has cells.

A coarse level is small by construction, so on a large run most ranks own none
of its cells, and the fine mesh of a placed fault is gathered onto one rank, so
most ranks own none of the finest cells either. Found on kaiju at np 104 (a
124-cell coarsest level, the faulted annulus): ``_colocate_level`` shaped its
centroid arrays from the cells it found, an empty rank raised IndexError, and
the ranks that had cells waited in the next all-gather for ever.

Run with at least four ranks; the 1 x 1 box at cell size 1 has four cells.
"""
import numpy as np
import pytest

import underworld3 as uw
from underworld3.utilities import custom_mg

pytestmark = [pytest.mark.mpi(min_size=4), pytest.mark.level_2, pytest.mark.tier_b,
              pytest.mark.timeout(300)]


def _box(cellSize, refinement=None):
    return uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), cellSize=cellSize,
        regular=True, refinement=refinement, qdegree=2)


def _cells(mesh):
    start, end = mesh.dm.getHeightStratum(0)
    return end - start


def _wrap(dm, like):
    return uw.discretisation.Mesh(
        dm.clone(), simplex=True,
        coordinate_system_type=like.CoordinateSystem.coordinate_type,
        qdegree=2, boundaries=like.boundaries)


def _poisson(mesh):
    p = uw.systems.Poisson(mesh)
    p.constitutive_model = uw.constitutive_models.DiffusionModel
    p.constitutive_model.Parameters.diffusivity = 1
    p.f = 0.0
    p.add_dirichlet_bc(0.0, "Bottom")
    p.add_dirichlet_bc(1.0, "Top")
    p.petsc_options["ksp_rtol"] = 1e-8
    p.petsc_options["ksp_type"] = "cg"
    return p


def test_faulted_annulus_with_empty_coarse_ranks():
    """A placed fault on a coarse annulus: the velocity multigrid is custom-P
    on every rank, with nothing degraded. At np 8 two ranks hold none of the
    coarse level the provenance transfer pairs with the placed host; they
    raised, retried alone with another builder, and the job died in
    mismatched collectives."""
    import sympy
    from underworld3.utilities.place_surface import place_fault_ribbon_2d
    base = uw.meshing.Annulus(radiusOuter=1.0, radiusInner=0.5, cellSize=0.8,
                              refinement=4, qdegree=3)
    width = 0.03
    r = np.arange(0.62, 0.88 + 1e-12, width / 2)
    trace = np.column_stack([r * np.cos(np.pi / 6), r * np.sin(np.pi / 6)])
    mesh, _info = place_fault_ribbon_2d(base, [("F", trace)], width)
    v = uw.discretisation.MeshVariable("V", mesh, 2, degree=2)
    p = uw.discretisation.MeshVariable("P", mesh, 1, degree=1)
    stokes = uw.systems.Stokes(mesh, velocityField=v, pressureField=p)
    stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
    stokes.constitutive_model.Parameters.shear_viscosity_0 = 1.0
    x, y = mesh.X
    stokes.bodyforce = sympy.Matrix([-y, x])
    stokes.add_dirichlet_bc((0.0, 0.0), "Lower")
    stokes.add_dirichlet_bc((0.0, 0.0), "Upper")
    stokes.add_fault_bc(0.0, "F")
    stokes.tolerance = 1.0e-5
    stokes.solve(zero_init_guess=True)
    assert stokes._rotated_freeslip_info["velocity_pc"] == "custom-FMG"
    assert stokes.snes.getConvergedReason() > 0
    assert not stokes.pc_fallbacks, stokes.pc_fallbacks
