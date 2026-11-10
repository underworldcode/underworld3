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

pytestmark = [pytest.mark.level_1, pytest.mark.tier_a]


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


def test_a_field_in_a_dirichlet_datum_refuses_rather_than_return_zero():
    mesh, T, f, poisson = _poisson("datum")
    poisson.solve(zero_init_guess=True)
    with pytest.raises(NotImplementedError, match=r"'f_datum'.*essential"):
        poisson.gradient(T.sym[0] ** 2 / 2, fields=[f])
    # the parameter-only call is unaffected
    out = poisson.gradient(T.sym[0] ** 2 / 2)
    assert out["J"] > 0


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
