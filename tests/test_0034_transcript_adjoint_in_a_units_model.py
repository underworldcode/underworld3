"""The backward pass over a recorded run, in a model with reference quantities.

The rig is ``test_0020``'s: a level set carried by an advection-diffusion
solve in the velocity of a Stokes solve whose body force it drives, with a
misfit on the final velocity. Here the model has reference quantities and the
level set carries units, so the history the walk restores holds a unit-aware
field. Before the fix the walk saved it with ``np.array``, which drops the
units, and the write back raised "Cannot assign plain array ... when units are
active" (the ``sinker_transcript`` example failed the same way).

The baseline is a central difference of the same forward run, with ``J`` read
in the frame the adjoint reads it in (``uw.adjoint.integral``).
"""
import numpy as np
import pytest
import sympy
import underworld3 as uw

pytestmark = [pytest.mark.level_2, pytest.mark.tier_b]

DT, NSTEPS = 0.02, 2


def _build():
    model = uw.get_default_model()
    model.set_reference_quantities(
        length=uw.quantity(1000, "km"), time=uw.quantity(1, "Myr"),
        mass=uw.quantity(1e24, "kg"), temperature=uw.quantity(1000, "K"))
    mesh = uw.meshing.UnstructuredSimplexBox(minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0),
                                             cellSize=1.0 / 8, qdegree=3)
    x, y = mesh.X
    beta = uw.discretisation.MeshVariable("beta_u", mesh, 1, degree=2, units="K")
    v = uw.discretisation.MeshVariable("v_u", mesh, 2, degree=2)
    p = uw.discretisation.MeshVariable("p_u", mesh, 1, degree=1)
    eta0 = uw.expression(r"\eta_0", 1.0, "viscosity")

    stokes = uw.systems.Stokes(mesh, velocityField=v, pressureField=p)
    stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
    stokes.constitutive_model.Parameters.shear_viscosity_0 = eta0
    stokes.bodyforce = sympy.Matrix([0.0, -50.0 * beta.sym[0]])
    for b in ("Top", "Bottom"):
        stokes.add_dirichlet_bc((0.0, 0.0), b)
    for b in ("Left", "Right"):
        stokes.add_dirichlet_bc((0.0, sympy.oo), b)
    stokes.tolerance = 1.0e-12

    adv = uw.systems.AdvDiffusion(mesh, u_Field=beta, V_fn=v.sym, theta=1.0)
    adv.constitutive_model = uw.constitutive_models.DiffusionModel
    adv.constitutive_model.Parameters.diffusivity = 1.0e-3
    adv.tolerance = 1.0e-12

    X = np.asarray(beta.coords_nd)
    beta0 = np.sqrt((X[:, 0] - 0.5) ** 2 + (X[:, 1] - 0.6) ** 2) - 0.2
    return dict(model=model, mesh=mesh, beta=beta, v=v, p=p, eta0=eta0,
                stokes=stokes, adv=adv, beta0=beta0)


def _misfit_expr(m):
    return sympy.Rational(1, 2) * m["v"].sym.dot(m["v"].sym)


def _forward(m):
    model = m["model"]
    model.clear_transcript()
    model.tracker.time, model.tracker.step = 0.0, 0
    model.record_every = 1
    m["beta"].data[:, 0] = m["beta0"]
    m["v"].data[...] = 0.0
    m["p"].data[...] = 0.0
    m["adv"].DuDt.initialise_history()
    for _ in range(NSTEPS):
        with model.step(DT):
            m["stokes"].solve(zero_init_guess=True)
            m["adv"].solve(timestep=DT, zero_init_guess=False)
    return model.save_state(), uw.adjoint.integral(m["mesh"], _misfit_expr(m))


def test_parameter_gradient_matches_finite_differences():
    m = _build()
    eta0 = m["eta0"]

    final, J = _forward(m)
    result = uw.adjoint.TranscriptAdjoint(m["model"], final).gradient(
        _misfit_expr(m), parameters=[eta0])
    assert result["J"] == pytest.approx(J, rel=1e-10)

    h = 1.0e-4
    eta0.sym = sympy.Float(1.0 + h)
    _, Jp = _forward(m)
    eta0.sym = sympy.Float(1.0 - h)
    _, Jm = _forward(m)
    eta0.sym = sympy.Float(1.0)
    assert result["parameters"][eta0] == pytest.approx((Jp - Jm) / (2 * h), rel=1.0e-4)
