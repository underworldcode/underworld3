"""A sum of two expressions keeps both as live atoms.

The sum used to be formed eagerly from the operands' current values, so the
compiled kernel froze the number: changing one operand afterwards changed nothing
(and in a units model a quantity entered the sum unscaled, #758). Checked on a
Poisson solve with diffusivity a + b: the solution is inversely proportional to
the diffusivity, so changing a must rescale it exactly.
"""
import numpy as np
import pytest

import underworld3 as uw

pytestmark = [pytest.mark.level_1, pytest.mark.tier_b]


def test_changing_an_operand_of_a_sum_reaches_the_solve():
    uw.reset_default_model()
    mesh = uw.meshing.UnstructuredSimplexBox(minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), cellSize=0.2)
    u = uw.discretisation.MeshVariable("u_sum", mesh, 1, degree=2)
    poisson = uw.systems.Poisson(mesh, u_Field=u)
    a = uw.expression(r"a_{\mathrm{sum}}", 2.0, "first operand")
    b = uw.expression(r"b_{\mathrm{sum}}", 1.0, "second operand")
    poisson.constitutive_model = uw.constitutive_models.DiffusionModel
    poisson.constitutive_model.Parameters.diffusivity = a + b
    poisson.f = 1.0
    poisson.tolerance = 1.0e-12
    for boundary in ("Left", "Right", "Top", "Bottom"):
        poisson.add_dirichlet_bc(0.0, boundary)
    poisson.solve()
    u3 = np.asarray(u.array).copy()
    a.sym = 7.0                                     # diffusivity 3 -> 8
    poisson.solve()
    u8 = np.asarray(u.array).copy()
    # a frozen sum leaves u unchanged (an error of order one); the live sum rescales
    # it to the solver's tolerance
    assert np.abs(u8 * 8.0 - u3 * 3.0).max() <= 1e-8 * np.abs(u3 * 3.0).max()
