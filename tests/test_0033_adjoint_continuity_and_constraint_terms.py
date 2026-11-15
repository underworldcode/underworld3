"""sensitivity() no longer returns 0 for a parameter in the continuity term or in an
add_constraint_bc datum (#1161).

A parameter in Stokes' volume continuity term (``constraints``) was dropped:
0 against a central difference of +4.015e-02. It is now tested against the pressure
adjoint. A parameter in an ``add_constraint_bc`` datum was dropped too (0 against
-6.29e-03) while ``adjoint_support()`` said supported; its part needs the
multiplier's adjoint, so it is refused by name, and ``adjoint_support()`` says so.
"""
import pytest
import sympy

import underworld3 as uw

pytestmark = [pytest.mark.level_2]


def _setup(solver_class):
    mesh = uw.meshing.UnstructuredSimplexBox(minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0),
                                             cellSize=1.0 / 6, qdegree=3)
    x, y = mesh.X
    v = uw.discretisation.MeshVariable("U", mesh, 2, degree=2)
    p = uw.discretisation.MeshVariable("P", mesh, 1, degree=1)
    stokes = solver_class(mesh, velocityField=v, pressureField=p)
    stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
    stokes.constitutive_model.Parameters.shear_viscosity_0 = 1.0
    stokes.tolerance = 1e-12
    stokes.petsc_options.delValue("ksp_monitor")
    stokes.bodyforce = sympy.Matrix([0, sympy.cos(sympy.pi * x)])
    return mesh, x, y, v, p, stokes


@pytest.mark.tier_b
def test_a_parameter_in_the_continuity_term():
    mesh, x, y, v, p, stokes = _setup(uw.systems.Stokes)
    m = uw.expression(r"m_{1161}", 1.0, "continuity source amplitude")
    for wall in ("Top", "Bottom", "Left"):
        stokes.add_dirichlet_bc((0.0, 0.0), wall)
    stokes.constraints = sympy.Matrix(
        [mesh.vector.divergence(v.sym) - m * sympy.sin(sympy.pi * y) * x])
    u_adj = uw.discretisation.MeshVariable("UA", mesh, 2, degree=2)
    p_adj = uw.discretisation.MeshVariable("PA", mesh, 1, degree=1)
    integrand = x * y * v.sym[0] + v.sym[1] ** 2
    dJdu = sympy.Matrix([[x * y, 2 * v.sym[1]]])

    def J(at):
        m.sym = sympy.Float(at)
        stokes.solve(zero_init_guess=True)
        assert stokes.snes.getConvergedReason() > 0
        return float(uw.maths.Integral(mesh, integrand).evaluate())

    J(1.0)
    _, reason = stokes.adjoint_solve(-stokes.dual_of(dJdu), target=(u_adj, p_adj))
    assert reason > 0
    adjoint = stokes.sensitivity(u_adj, m, lam=p_adj)
    h = 1.0e-4
    fd = (J(1.0 + h) - J(1.0 - h)) / (2 * h)
    # development: 0 against +4.015e-02
    assert adjoint == pytest.approx(fd, rel=1e-6), (adjoint, fd)
    J(1.0)
    with pytest.raises(ValueError, match=r"m_\{1161\} enters the continuity term"):
        stokes.sensitivity(u_adj, m)


@pytest.mark.tier_b
def test_a_parameter_in_a_constraint_datum_is_refused_by_name():
    mesh, x, y, v, p, stokes = _setup(uw.systems.Stokes_Constrained)
    m = uw.expression(r"m_{c1161}", 1.0, "constraint datum amplitude")
    stokes.add_dirichlet_bc((0.0, 0.0), "Bottom")
    stokes.add_dirichlet_bc((4 * y * (1 - y), 0.0), "Left")
    stokes.add_constraint_bc(m * sympy.sin(sympy.pi * x) / 10, "Top")
    supported, why = stokes.adjoint_support()
    assert supported and "add_constraint_bc datum is refused" in why, why
    u_adj = uw.discretisation.MeshVariable("UA", mesh, 2, degree=2)
    p_adj = uw.discretisation.MeshVariable("PA", mesh, 1, degree=1)
    stokes.solve(zero_init_guess=True)
    stokes.adjoint_solve(-stokes.dual_of(sympy.Matrix([[x * y, 2 * v.sym[1]]])),
                         target=(u_adj, p_adj))
    # development: 0 against a central difference of -6.29e-03, no refusal
    with pytest.raises(NotImplementedError, match=r"m_\{c1161\} enters the datum .* of the add_constraint_bc"):
        stokes.sensitivity(u_adj, m, lam=p_adj)
