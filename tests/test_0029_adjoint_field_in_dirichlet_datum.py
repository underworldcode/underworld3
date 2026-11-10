"""A field that enters only through an essential datum, ``u|_G = f``, refuses (#787).

The field route differentiates the residual, and a datum is not in it: the
dual came back exactly zero (finite difference: 0.31), which reads as "no
influence". Until the reaction term is mapped onto the field's nodes, asking
for that dual raises, on the one-solve route and on the transcript walk. A
field read by the residual is unaffected.
"""
import numpy as np
import pytest
import sympy

import underworld3 as uw

pytestmark = [pytest.mark.level_1]


def _poisson(tag, datum_field=True):
    mesh = uw.meshing.UnstructuredSimplexBox(minCoords=(0, 0), maxCoords=(1, 1),
                                             cellSize=1 / 6, qdegree=3)
    T = uw.discretisation.MeshVariable(f"T_{tag}", mesh, 1, degree=2)
    f = uw.discretisation.MeshVariable(f"f_{tag}", mesh, 1, degree=2)
    f.array[:, 0, 0] = 1 + np.asarray(f.coords)[:, 0] ** 2
    poisson = uw.systems.Poisson(mesh, u_Field=T)
    poisson.constitutive_model = uw.constitutive_models.DiffusionModel
    poisson.constitutive_model.Parameters.diffusivity = 1.0
    if datum_field:
        poisson.f = 1.0
        poisson.add_dirichlet_bc(sympy.Matrix([f.sym[0]]), "Top")
    else:
        poisson.f = f.sym[0]
        poisson.add_dirichlet_bc(1.0, "Top")
    poisson.add_dirichlet_bc(0.0, "Bottom")
    poisson.petsc_options.delValue("ksp_monitor")
    poisson.tolerance = 1e-10
    return mesh, T, f, poisson


@pytest.mark.tier_a
def test_a_field_in_a_dirichlet_datum_refuses_rather_than_return_zero():
    mesh, T, f, poisson = _poisson("datum")
    poisson.solve(zero_init_guess=True)
    with pytest.raises(NotImplementedError, match=r"'f_datum'.*essential"):
        poisson.gradient(T.sym[0] ** 2 / 2, fields=[f])
    # the parameter-only call is unaffected
    out = poisson.gradient(T.sym[0] ** 2 / 2)
    assert out["J"] > 0


@pytest.mark.tier_a
def test_a_field_read_by_the_residual_still_gets_its_dual():
    mesh, T, f, poisson = _poisson("source", datum_field=False)
    misfit = T.sym[0] ** 2 / 2

    def J(F):
        f.array[:, 0, 0] = F
        poisson.solve(zero_init_guess=True)
        return float(uw.maths.Integral(mesh, misfit).evaluate())

    F0 = np.array(np.asarray(f.array)[:, 0, 0], copy=True)
    J(F0)
    dual = poisson.gradient(misfit, fields=[f])["fields"][f]
    direction = np.sin(np.pi * np.asarray(f.coords)[:, 0])
    adjoint = uw.adjoint.inner(f, dual, direction)
    h = 1e-4
    fd = (J(F0 + h * direction) - J(F0 - h * direction)) / (2 * h)
    assert adjoint == pytest.approx(fd, rel=1e-5), (adjoint, fd)


@pytest.mark.level_2
@pytest.mark.tier_a
def test_the_transcript_walk_refuses_a_field_in_a_datum():
    uw.reset_default_model()
    model = uw.get_default_model()
    mesh, T, f, poisson = _poisson("walk")
    model.clear_transcript()
    model.record_every = 1
    with model.step(0.1):
        poisson.solve(zero_init_guess=True)
    final = model.save_state()
    with pytest.raises(NotImplementedError, match=r"'f_walk'"):
        uw.adjoint.TranscriptAdjoint(model, final).gradient(T.sym[0] ** 2 / 2, fields=[f])


def _poisson_on(mesh, T, source, top):
    poisson = uw.systems.Poisson(mesh, u_Field=T)
    poisson.constitutive_model = uw.constitutive_models.DiffusionModel
    poisson.constitutive_model.Parameters.diffusivity = 1.0
    poisson.f = source
    poisson.add_dirichlet_bc(top, "Top")
    poisson.add_dirichlet_bc(0.0, "Bottom")
    poisson.petsc_options.delValue("ksp_monitor")
    poisson.tolerance = 1e-10
    return poisson


@pytest.mark.tier_a
def test_a_field_behind_a_nested_expression_in_the_datum_still_refuses():
    mesh, T, f, _ = _poisson("nested", datum_field=False)
    a = uw.expression(r"a_{787}", f.sym[0], "wraps the field")
    b = uw.expression(r"b_{787}", 2 * a, "wraps the wrapper")
    poisson = _poisson_on(mesh, T, 1.0, sympy.Matrix([b]))
    poisson.solve(zero_init_guess=True)
    with pytest.raises(NotImplementedError, match="f_nested"):
        poisson.gradient(T.sym[0] ** 2 / 2, fields=[f])


@pytest.mark.tier_a
def test_a_field_whose_name_another_contains_is_not_refused():
    """``\\bar{f}`` in the datum is another variable: the field ``f`` the
    residual reads still gets its dual, and it agrees with FD."""
    mesh, T, f, _ = _poisson("near", datum_field=False)
    fbar = uw.discretisation.MeshVariable("fbar_near", mesh, 1, degree=2, varsymbol=r"\bar{f}")
    fbar.array[:, 0, 0] = 1.0
    poisson = _poisson_on(mesh, T, f.sym[0], sympy.Matrix([fbar.sym[0]]))
    misfit = T.sym[0] ** 2 / 2

    def J(F):
        f.array[:, 0, 0] = F
        poisson.solve(zero_init_guess=True)
        return float(uw.maths.Integral(mesh, misfit).evaluate())

    F0 = np.array(np.asarray(f.array)[:, 0, 0], copy=True)
    J(F0)
    dual = poisson.gradient(misfit, fields=[f])["fields"][f]
    direction = np.sin(np.pi * np.asarray(f.coords)[:, 0])
    h = 1e-4
    fd = (J(F0 + h * direction) - J(F0 - h * direction)) / (2 * h)
    assert uw.adjoint.inner(f, dual, direction) == pytest.approx(fd, rel=1e-5)


@pytest.mark.tier_a
def test_a_field_in_a_natural_condition_refuses():
    mesh, T, f, _ = _poisson("neumann", datum_field=False)
    poisson = _poisson_on(mesh, T, 1.0, 0.0)
    poisson.add_natural_bc(sympy.Matrix([f.sym[0]]), "Left")
    poisson.solve(zero_init_guess=True)
    with pytest.raises(NotImplementedError, match=r"'f_neumann' \(natural"):
        poisson.gradient(T.sym[0] ** 2 / 2, fields=[f])


def _chain(tag, p_in_upstream):
    """Solve A gives g; solve B reads g in its datum. A parameter p is in B's
    source, and in A's too when ``p_in_upstream``."""
    uw.reset_default_model()
    model = uw.get_default_model()
    mesh = uw.meshing.UnstructuredSimplexBox(minCoords=(0, 0), maxCoords=(1, 1),
                                             cellSize=1 / 6, qdegree=3)
    g = uw.discretisation.MeshVariable(f"g_{tag}", mesh, 1, degree=2)
    T = uw.discretisation.MeshVariable(f"T_{tag}", mesh, 1, degree=2)
    p = uw.expression(rf"p_{{{tag}}}", 1.0, "source amplitude")
    A = _poisson_on(mesh, g, p if p_in_upstream else 1.0, 0.0)
    B = _poisson_on(mesh, T, p, sympy.Matrix([1 + g.sym[0]]))
    misfit = T.sym[0] ** 2 / 2

    def run(value):
        p.sym = float(value)
        model.clear_transcript()
        model.record_every = 1
        with model.step(0.1):
            A.solve(zero_init_guess=True)
            B.solve(zero_init_guess=True)
        return model.save_state(), float(uw.maths.Integral(mesh, misfit).evaluate())

    return model, p, misfit, run


@pytest.mark.tier_b      # two recorded solves and a walk
def test_the_walk_takes_a_parameter_gradient_through_a_datum_on_an_uncontrolled_output():
    model, p, misfit, run = _chain("free", p_in_upstream=False)
    final, _ = run(1.0)
    adjoint = uw.adjoint.TranscriptAdjoint(model, final).gradient(misfit, parameters=[p])
    h = 1e-4
    fd = (run(1.0 + h)[1] - run(1.0 - h)[1]) / (2 * h)
    assert adjoint["parameters"][p] == pytest.approx(fd, rel=1e-5)


@pytest.mark.tier_b
def test_the_walk_refuses_a_datum_on_an_output_that_depends_on_the_control():
    model, p, misfit, run = _chain("ctl", p_in_upstream=True)
    final, _ = run(1.0)
    with pytest.raises(NotImplementedError, match=r"'g_ctl'.*depends on the controls"):
        uw.adjoint.TranscriptAdjoint(model, final).gradient(misfit, parameters=[p])
