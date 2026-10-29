"""``build_rotation`` reads the solver's Jacobian for the operator's row layout.
``_build()`` sets the SNES up only on some paths (with essential boundary
conditions registered it ends up set up; without them the Jacobian is never
created), so ``build_rotation`` must set the SNES up itself rather than
dereference a null matrix. Before the guard: a segmentation fault (found
2026-10-13 building a rotation for a drag-only faulted layer)."""

import numpy as np
import pytest

import underworld3 as uw

pytestmark = [pytest.mark.level_1, pytest.mark.tier_b]


def test_build_rotation_on_a_built_solver_without_essential_bcs():
    from underworld3.utilities.rotated_bc import build_rotation

    mesh = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), cellSize=0.1, qdegree=2)
    split = mesh.add_fault([("Flt", np.array([[0.25, 0.5], [0.75, 0.5]]))])
    stokes = uw.systems.Stokes(split)
    stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
    stokes.constitutive_model.Parameters.shear_viscosity_0 = 1
    stokes.add_basal_drag(1, "Bottom")          # natural condition only
    stokes.add_fault_bc(0, "Flt")
    stokes._build()
    Q, Qt, normal_rows, _ = build_rotation(stokes, stokes._rotated_freeslip_bcs)
    n = stokes.snes.getJacobian()[0].getSize()[0]
    assert Q.getSize() == (n, n)
    assert len(normal_rows) > 0
