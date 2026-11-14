"""sensitivity() reads every boundary term of a natural or Nitsche condition (#1143).

It read only the value term fn_f. A Nitsche condition also has a flux term
fn_F, tested against the gradient of the test function, and its part of the
gradient was left out. A lid datum imposed by Nitsche then came out with the
wrong sign (-7.33e-2 against a central difference of +4.52e-2), and the
viscosity beside a Nitsche wall was off too. A normal datum also enters the
continuity-row term fn_p (assembled on the pressure field since #1072), which
is tested against the pressure adjoint: without it the gradient was 2.85e-5
against a central difference of 1.68e-3.
"""
import pytest
import sympy

import underworld3 as uw

pytestmark = [pytest.mark.level_2]


def _box():
    mesh = uw.meshing.UnstructuredSimplexBox(minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0),
                                             cellSize=1.0 / 8, qdegree=3)
    v = uw.discretisation.MeshVariable("U", mesh, 2, degree=2, vtype=uw.VarType.VECTOR)
    p = uw.discretisation.MeshVariable("P", mesh, 1, degree=1)
    stokes = uw.systems.Stokes(mesh, velocityField=v, pressureField=p)
    stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
    stokes.saddle_preconditioner = 1.0
    stokes.tolerance = 1e-12
    stokes.petsc_options["ksp_type"] = "fgmres"
    stokes.petsc_use_pressure_nullspace = True
    stokes.petsc_options.delValue("ksp_monitor")
    return mesh, v, p, stokes


def _adjoint_against_fd(mesh, v, p, stokes, parameter, value):
    u_adj = uw.discretisation.MeshVariable("UA", mesh, 2, degree=2)
    p_adj = uw.discretisation.MeshVariable("PA", mesh, 1, degree=1)

    def J(at):
        parameter.sym = sympy.Float(at)
        stokes.solve(zero_init_guess=True)
        assert stokes.snes.getConvergedReason() > 0
        return float(uw.maths.Integral(mesh, v.sym.dot(v.sym) / 2).evaluate())

    J(value)
    stokes.adjoint_solve(-stokes.dual_of(v.sym), target=(u_adj, p_adj))
    adjoint = stokes.sensitivity(u_adj, parameter, lam=p_adj)
    h = 1.0e-4 * value
    fd = (J(value + h) - J(value - h)) / (2 * h)
    return adjoint, fd


@pytest.mark.tier_b
def test_a_lid_datum_imposed_by_nitsche():
    mesh, v, p, stokes = _box()
    x, y = mesh.X
    m = uw.expression(r"m", 1.0, "lid amplitude")
    stokes.constitutive_model.Parameters.shear_viscosity_0 = 1.0
    stokes.bodyforce = sympy.Matrix([0, sympy.cos(sympy.pi * x)])
    for wall in ("Left", "Right", "Bottom"):
        stokes.add_dirichlet_bc((0.0, 0.0), wall)
    stokes.add_dirichlet_bc((None, 0.0), "Top")
    stokes.add_nitsche_bc(m * sympy.sin(sympy.pi * x), "Top", direction=(1, 0))
    adjoint, fd = _adjoint_against_fd(mesh, v, p, stokes, m, 1.0)
    # development: -7.33e-2 against +4.52e-2
    assert adjoint == pytest.approx(fd, rel=1e-6), (adjoint, fd)


@pytest.mark.tier_b
def test_the_viscosity_beside_a_nitsche_wall():
    """The viscosity enters the Nitsche flux term (the symmetric term carries the
    stress) as well as the volume."""
    mesh, v, p, stokes = _box()
    x, y = mesh.X
    eta = uw.expression(r"\eta_{1143}", 1.0, "viscosity")
    stokes.constitutive_model.Parameters.shear_viscosity_0 = eta
    stokes.bodyforce = sympy.Matrix([0, sympy.cos(sympy.pi * x)])
    for wall in ("Left", "Right", "Bottom"):
        stokes.add_dirichlet_bc((0.0, 0.0), wall)
    stokes.add_dirichlet_bc((None, 0.0), "Top")
    stokes.add_nitsche_bc(sympy.Rational(1, 2) * sympy.sin(sympy.pi * x), "Top", direction=(1, 0))
    adjoint, fd = _adjoint_against_fd(mesh, v, p, stokes, eta, 1.0)
    assert adjoint == pytest.approx(fd, rel=1e-6), (adjoint, fd)


@pytest.mark.tier_b
def test_a_normal_datum_imposed_by_nitsche():
    """A normal datum puts the parameter in the continuity-row term fn_p as well,
    -(n.d)(u.d - g), tested against the pressure adjoint. The datum carries no
    net flux, so the closed box stays compatible."""
    mesh, v, p, stokes = _box()
    x, y = mesh.X
    m = uw.expression(r"m_n", 1.0, "normal datum amplitude")
    stokes.constitutive_model.Parameters.shear_viscosity_0 = 1.0
    stokes.bodyforce = sympy.Matrix([0, sympy.cos(sympy.pi * x)])
    for wall in ("Left", "Top", "Bottom"):
        stokes.add_dirichlet_bc((0.0, 0.0), wall)
    stokes.add_dirichlet_bc((None, 0.0), "Right")
    stokes.add_nitsche_bc(m * sympy.Rational(1, 10) * sympy.sin(2 * sympy.pi * y), "Right", direction=(1, 0))
    adjoint, fd = _adjoint_against_fd(mesh, v, p, stokes, m, 1.0)
    assert adjoint == pytest.approx(fd, rel=1e-6), (adjoint, fd)


@pytest.mark.tier_b
def test_a_normal_datum_needs_the_pressure_adjoint():
    """Without lam the continuity-row part cannot be formed: refused, not dropped."""
    mesh, v, p, stokes = _box()
    x, y = mesh.X
    m = uw.expression(r"m_q", 1.0, "normal datum amplitude")
    stokes.constitutive_model.Parameters.shear_viscosity_0 = 1.0
    for wall in ("Left", "Top", "Bottom"):
        stokes.add_dirichlet_bc((0.0, 0.0), wall)
    stokes.add_dirichlet_bc((None, 0.0), "Right")
    stokes.add_nitsche_bc(m * sympy.Rational(1, 10) * sympy.sin(2 * sympy.pi * y), "Right", direction=(1, 0))
    stokes.solve(zero_init_guess=True)
    u_adj = uw.discretisation.MeshVariable("UQ", mesh, 2, degree=2)
    p_adj = uw.discretisation.MeshVariable("PQ", mesh, 1, degree=1)
    stokes.adjoint_solve(-stokes.dual_of(v.sym), target=(u_adj, p_adj))
    with pytest.raises(ValueError, match=r"m_q enters the continuity-row term of the Nitsche condition on 'Right'"):
        stokes.sensitivity(u_adj, m)
    # A parameter the continuity-row term does not read is not refused for it:
    # the viscosity enters the flux term only.
    eta = uw.expression(r"\eta_q", 1.0, "viscosity")
    stokes.constitutive_model.Parameters.shear_viscosity_0 = eta
    stokes.solve(zero_init_guess=True)
    stokes.adjoint_solve(-stokes.dual_of(v.sym), target=(u_adj, p_adj))
    stokes.sensitivity(u_adj, eta)
