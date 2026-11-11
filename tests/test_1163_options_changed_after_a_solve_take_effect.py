"""Options and a tolerance changed after a solve take effect at the next one (#1078).

The scalar, vector and multi-component solvers read their PETSc options once, at
the first build, so a change to ``solver.petsc_options`` or ``solver.tolerance``
made after the first solve never reached PETSc: a solve asked for 1e-9 after a
first solve at 1e-4 stopped at 1e-4. Stokes re-read them on every solve.

- after a first solve, a new ``tolerance``, ``snes_max_it``, line search and
  ``ksp_rtol`` are what the SNES and KSP hold at the next solve, in the scalar,
  vector and multi-component solvers and in Stokes;
- a Poisson solve tightened from 1e-4 to 1e-9 after its first solve meets 1e-9.
"""
import pytest
import sympy

import underworld3 as uw

pytestmark = [pytest.mark.level_1, pytest.mark.tier_b]


def _box():
    return uw.meshing.UnstructuredSimplexBox(minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0),
                                             cellSize=0.25)


def _poisson(tag):
    mesh = _box()
    T = uw.discretisation.MeshVariable(f"T1163{tag}", mesh, 1, degree=2)
    s = uw.systems.Poisson(mesh, u_Field=T)
    s.constitutive_model = uw.constitutive_models.DiffusionModel
    s.constitutive_model.Parameters.diffusivity = 1.0 + T.sym[0] ** 2     # nonlinear
    s.f = 1.0
    s.add_dirichlet_bc(0.0, "Bottom")
    return s


def _vector(tag):
    mesh = _box()
    u = uw.discretisation.MeshVariable(f"U1163{tag}", mesh, 2, degree=2)
    s = uw.systems.Vector_Projection(mesh, u)
    s.uw_function = sympy.Matrix([[1.0, 0.5]])
    s.smoothing = 1.0e-2
    return s


def _multicomponent(tag):
    s = uw.systems.MultiComponent_Projection(_box(), n_components=2, degree=2)
    s.uw_function = sympy.Matrix([[1.0, 0.5]])
    s.smoothing = 1.0e-2
    return s


def _stokes(tag):
    mesh = _box()
    v = uw.discretisation.MeshVariable(f"V1163{tag}", mesh, 2, degree=2)
    p = uw.discretisation.MeshVariable(f"P1163{tag}", mesh, 1, degree=1, continuous=True)
    s = uw.systems.Stokes(mesh, velocityField=v, pressureField=p)
    s.constitutive_model = uw.constitutive_models.ViscousFlowModel
    s.constitutive_model.Parameters.shear_viscosity_0 = 1.0
    s.bodyforce = sympy.Matrix([[0.0, sympy.cos(sympy.pi * mesh.X[0])]])
    for wall in ("Bottom", "Top"):
        s.add_dirichlet_bc((0.0, 0.0), wall)
    for wall in ("Left", "Right"):
        s.add_dirichlet_bc((0.0, None), wall)
    return s


_SOLVERS = {"scalar": _poisson, "vector": _vector, "multicomponent": _multicomponent,
            "stokes": _stokes}


@pytest.mark.parametrize("family", sorted(_SOLVERS))
def test_options_set_after_a_solve_reach_the_next_one(family):
    s = _SOLVERS[family](family)
    s.tolerance = 1.0e-4
    s.solve()
    s.tolerance = 1.0e-7
    s.petsc_options["snes_max_it"] = 7
    s.petsc_options["snes_linesearch_type"] = "l2"
    s.petsc_options["ksp_rtol"] = 0.0123
    s.solve(zero_init_guess=True)
    # development: the scalar, vector and multi-component solvers still held 1e-4,
    # 50, bt and their first ksp_rtol
    rtol, _atol, _stol, max_it = s.snes.getTolerances()
    assert rtol == pytest.approx(1.0e-7), rtol
    assert max_it == 7, max_it
    assert s.snes.getLineSearch().getType() == "l2"
    assert s.snes.getKSP().getTolerances()[0] == pytest.approx(0.0123)


def test_a_tolerance_tightened_after_a_solve_is_met():
    s = _poisson("tight")
    s.tolerance = 1.0e-4
    s.solve()
    loose = s.solve_report.history
    s.tolerance = 1.0e-9
    s.solve(zero_init_guess=True)
    tight = s.solve_report.history
    assert s.snes.getConvergedReason() > 0, s.snes.getConvergedReason()
    # development: the second solve stopped where the first did (2 Newton steps,
    # 5.75e-07 from 1.22e-01), well short of 1e-9 relative
    assert tight[-1] <= 1.0e-9 * tight[0], tight
    assert len(tight) > len(loose), (loose, tight)
