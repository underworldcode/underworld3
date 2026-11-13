"""sensitivity() reads every boundary term of a natural or Nitsche condition (#1143).

It read only the value term fn_f. A Nitsche condition also has a flux term
fn_F, tested against the gradient of the test function, and its part of the
gradient was left out. A lid datum imposed by Nitsche then came out with the
wrong sign (-7.33e-2 against a central difference of +4.52e-2), and the
viscosity beside a Nitsche wall was off too. The continuity-row term fn_p of
a Stokes Nitsche condition is not assembled by development's forward operator
(#1072), so its row is not tested here.
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
    stress) as well as the volume. The wall condition is tangential: a Stokes
    Nitsche condition along the normal does not converge on development until
    its continuity term is assembled (#1072)."""
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
