#!/usr/bin/env python3
"""Regression: Projection.linear_solver() opt-in lightweight SPD solve (#156).

An L2 projection is a linear SPD problem; the default newtonls/gmres/gamg stack
is heavy (GAMG setup/repartition is a memory/communication bottleneck for
repeated post-processing projections). ``linear_solver()`` switches a projector
to ``ksponly + CG + cheap PC`` without changing the global default.
"""
import numpy as np
import sympy
import pytest

import underworld3 as uw

pytestmark = [pytest.mark.level_1, pytest.mark.tier_a]


@pytest.fixture
def mesh():
    return uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), cellSize=0.1, qdegree=3
    )


def _project(mesh, expr, use_linear):
    target = uw.discretisation.MeshVariable(
        f"t_{'lin' if use_linear else 'def'}", mesh, 1, degree=2
    )
    proj = uw.systems.Projection(mesh, target)
    proj.uw_function = expr
    proj.smoothing = 0.0
    if use_linear:
        proj.linear_solver()
    proj.solve()
    return proj, np.array(target.data[:, 0]).copy()


def test_linear_solver_sets_lightweight_options(mesh):
    """linear_solver() sets the ksponly/CG/jacobi stack and is chainable."""
    target = uw.discretisation.MeshVariable("t_opts", mesh, 1, degree=2)
    proj = uw.systems.Projection(mesh, target)
    ret = proj.linear_solver()
    assert ret is proj, "linear_solver() should return self for chaining"
    assert proj.petsc_options.getString("snes_type", "") == "ksponly"
    assert proj.petsc_options.getString("ksp_type", "") == "cg"
    assert proj.petsc_options.getString("pc_type", "") == "jacobi"
    # GAMG options removed
    assert not proj.petsc_options.hasName("pc_gamg_type")


def test_linear_solver_matches_default_projection(mesh):
    """The lightweight solve produces the same projection as the default."""
    x, y = mesh.X
    expr = sympy.sin(3 * x) * sympy.cos(2 * y) + x ** 2
    _, T_default = _project(mesh, expr, use_linear=False)
    _, T_linear = _project(mesh, expr, use_linear=True)
    rel = np.linalg.norm(T_default - T_linear) / np.linalg.norm(T_default)
    assert rel < 1.0e-4, f"linear_solver projection differs from default: rel L2 = {rel:.3e}"


def test_linear_solver_is_opt_in(mesh):
    """A projector that does NOT call linear_solver() keeps the default stack."""
    target = uw.discretisation.MeshVariable("t_optin", mesh, 1, degree=2)
    proj = uw.systems.Projection(mesh, target)
    # default SNES_Scalar stack untouched
    assert proj.petsc_options.getString("snes_type", "") == "newtonls"
    assert proj.petsc_options.getString("pc_type", "") == "gamg"
