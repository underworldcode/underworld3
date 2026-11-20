"""A semi-Lagrangian velocity given as a column vector is read as the row it
means.

Mesh vectors are (1, dim) rows. ``V_fn=sympy.Matrix([vx, vy])`` is a (dim, 1)
column, and the trace read only its first row (``v[:, 0, :]`` of a
(N, dim, 1) evaluation): the x component alone, then a reshape failure
(docs/examples/convection/advanced/Ex_AdvectionDiffusion_1dBlock.py, the
example sweep for #944, W1 2026-11-21). The column is now transposed to the
row, as EulerianSUPG already does, and the solve matches the row form.
"""
import numpy as np
import pytest
import sympy

import underworld3 as uw

pytestmark = [pytest.mark.level_1, pytest.mark.tier_b]


def _advect(V_fn):
    mesh = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), cellSize=0.2, regular=True)
    T = uw.discretisation.MeshVariable("T", mesh, 1, degree=2)
    x, y = mesh.X
    T.array[:, 0, 0] = uw.function.evaluate(
        sympy.sin(sympy.pi * x) * sympy.sin(sympy.pi * y), T.coords)[:, 0, 0]
    adv = uw.systems.AdvDiffusionSLCN(mesh, T, V_fn=V_fn, order=1)
    adv.constitutive_model = uw.constitutive_models.DiffusionModel
    adv.constitutive_model.Parameters.diffusivity = 1.0e-3
    adv.add_dirichlet_bc(0.0, "Left")
    adv.add_dirichlet_bc(0.0, "Bottom")
    adv.solve(timestep=0.05)
    return T.array[:, 0, 0].copy()


def test_a_column_velocity_advects_as_its_row():
    row = _advect(sympy.Matrix([[0.5, 0.25]]))
    column = _advect(sympy.Matrix([0.5, 0.25]))
    np.testing.assert_allclose(column, row, rtol=0, atol=1.0e-12)
