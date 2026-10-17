"""The 2-D ``add_fault`` child of a refined base carries the base's coarse levels.

A mesh built with ``refinement >= 1`` is its own geometric-multigrid hierarchy:
its coarse levels are the tail, the finest level is the mesh itself. A cut
re-represents that finest grid with the fault conformed, so the cut-and-split
child takes the coarse levels as its tail with itself as the finest level —
the rule ``split_fault`` and ``Mesh._adopt_cut_child`` already apply. The 2-D
cut path constructed its child bare, so every solver on an ``add_fault`` child
of a refined box fell back to algebraic multigrid silently (measured
2026-10-17: refinement=2, tail 0, velocity KSP at its 200-iteration cap).

The contract asserted: the tail has exactly one level per base coarsening and
the velocity preconditioner of a fault solve on the child is the custom FMG.
"""
import numpy as np
import pytest

import underworld3 as uw

pytestmark = [pytest.mark.level_2, pytest.mark.tier_b]

TRACE = np.array([[0.30, 0.45], [0.70, 0.55]])


def _refined_box(refinement):
    return uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), cellSize=0.1,
        refinement=refinement, qdegree=2)


def _cells(mesh):
    cS, cE = mesh.dm.getHeightStratum(0)
    return cE - cS


@pytest.mark.parametrize("refinement", [1, 2])
def test_cut_child_takes_the_base_coarse_levels(refinement):
    base = _refined_box(refinement)
    child = base.add_fault([("Flt", TRACE)])
    tail = child._custom_mg_coarse_meshes
    assert tail is not None and len(tail) == refinement, tail
    # the levels are the base's own static wraps, coarsest first: each one
    # about a quarter of the next (2-D uniform refinement), none of them the
    # cut finest
    sizes = [_cells(m) for m in tail] + [_cells(child)]
    for coarse, fine in zip(sizes[:-1], sizes[1:]):
        assert 3 * coarse <= fine <= 5 * coarse + 64, sizes
    assert all(getattr(m, "_mg_static", False) for m in tail)


def test_fault_solve_on_the_cut_child_runs_on_the_geometric_tail():
    base = _refined_box(1)
    child = base.add_fault([("Flt", TRACE)])
    x, y = child.X
    stokes = uw.systems.Stokes(child)
    stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
    stokes.constitutive_model.Parameters.shear_viscosity_0 = 1
    stokes.tolerance = 1.0e-6
    stokes.petsc_use_pressure_nullspace = True
    for side in ("Top", "Bottom", "Left", "Right"):
        stokes.add_dirichlet_bc((y - 0.5, 0.0), side)
    stokes.add_fault_bc(0, "Flt")
    stokes.solve()
    info = stokes._rotated_freeslip_info
    assert info["velocity_pc"] == "custom-FMG" and info["converged"], info
