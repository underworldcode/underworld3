"""A natural condition's tangent sees through the named expressions in it (#1162).

With ``consistent_jacobian = True`` the volume flux, and a Nitsche condition's
gradient term ``fn_F``, are unwrapped before they are differentiated, so the
tangent carries d(viscosity)/d(grad u). The boundary term ``fn_f`` was
differentiated as written: a named expression in it (the viscosity in the Nitsche
penalty, a user's coefficient in a natural flux) was a constant to the derivative.
The forward solve still converges, but the matrix is not the residual's
derivative, and an adjoint gradient through it was off by 4.1e-5 relative where
the strong wall agrees with a central difference to 6e-9.

Each solver family builds its boundary tangent in its own loop; each is checked
against a central difference of its own boundary residual, read in PETSc's
[fc*Nc + gc, df] layout.
"""
import numpy as np
import pytest
import sympy

import underworld3 as uw
from underworld3.function.expressions import unwrap_expression

pytestmark = [pytest.mark.level_1, pytest.mark.tier_b]

_count = [0]


def _name(stem):
    _count[0] += 1
    return f"{stem}1162_{_count[0]}"


def _boundary_tangent_gap(solver, bc, Nc, dim):
    """Largest gap between the assembled boundary blocks uu_G0, uu_G1 and a
    central difference of the boundary residual u_f0, relative to the largest
    derivative, at one point."""
    U = [solver.Unknowns.u.sym[c] for c in range(Nc)]
    L = sympy.Matrix(solver.Unknowns.L)
    if L.shape != (Nc, dim):
        L = L.reshape(Nc, dim)
    N = solver.mesh.CoordinateSystem.N
    u_s = [sympy.Symbol(f"u{c}") for c in range(Nc)]
    l_s = [[sympy.Symbol(f"l{c}{d}") for d in range(dim)] for c in range(Nc)]
    flat = u_s + [l_s[c][d] for c in range(Nc) for d in range(dim)]
    rep = {U[c]: u_s[c] for c in range(Nc)}
    rep.update({L[c, d]: l_s[c][d] for c in range(Nc) for d in range(dim)})
    rep.update({N[0]: sympy.Float(0.0), N[1]: sympy.Float(0.37)})
    rng = np.random.default_rng(1162)
    point = np.concatenate([0.3 * rng.standard_normal(Nc), rng.standard_normal(Nc * dim)])

    def numeric(e):
        e = sympy.sympify(unwrap_expression(e, mode="nondimensional")).xreplace(rep)
        # any other field (the Stokes pressure, say) is held at a value
        e = e.xreplace({f: sympy.Float(0.7)
                        for f in e.atoms(sympy.core.function.AppliedUndef)})
        left = [a for a in e.free_symbols if a not in flat]
        assert not left, f"unresolved symbols {left}"
        return sympy.lambdify(flat, e, "numpy")

    f0 = [numeric(e) for e in list(sympy.Matrix(bc.fns["u_f0"]))]
    G0 = list(sympy.Matrix(bc.fns["uu_G0"]))
    G1 = list(sympy.Matrix(bc.fns["uu_G1"]))
    h = 1.0e-6
    gaps, scale = [], 0.0
    for k in range(len(flat)):
        up, dn = point.copy(), point.copy()
        up[k] += h
        dn[k] -= h
        for fc in range(Nc):
            fd = (f0[fc](*up) - f0[fc](*dn)) / (2 * h)
            if k < Nc:                       # d f0[fc] / dU[gc]
                assembled = numeric(G0[fc * Nc + k])(*point)
            else:                            # d f0[fc] / dL[gc, df]
                gc, df = divmod(k - Nc, dim)
                assembled = numeric(G1[(fc * Nc + gc) * dim + df])(*point)
            gaps.append(abs(assembled - fd))
            scale = max(scale, abs(fd))
    assert scale > 0
    return max(gaps) / scale


def _stokes_nitsche():
    mesh = uw.meshing.UnstructuredSimplexBox(minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0),
                                             cellSize=1.0 / 4, qdegree=3)
    v = uw.discretisation.MeshVariable(_name("V"), mesh, 2, degree=2)
    p = uw.discretisation.MeshVariable(_name("P"), mesh, 1, degree=1)
    stokes = uw.systems.Stokes(mesh, velocityField=v, pressureField=p)
    stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
    stokes.constitutive_model.Parameters.shear_viscosity_0 = (
        1 + 5 * stokes.Unknowns.Einv2 ** 2) ** sympy.Rational(-1, 4)
    stokes.consistent_jacobian = True
    x, y = mesh.X
    stokes.add_nitsche_bc(-4 * y * (1 - y), "Left", normal=sympy.Matrix([-1.0, 0.0]),
                          local_h=False)
    return stokes


def test_stokes_nitsche_tangent_with_a_strain_rate_viscosity():
    stokes = _stokes_nitsche()
    stokes._setup_pointwise_functions()
    bc = stokes.natural_bcs[0]
    assert _boundary_tangent_gap(stokes, bc, 2, 2) < 1.0e-6


def _coefficient(mesh, grad):
    """A named coefficient that depends on the unknown's gradient."""
    return uw.expression(_name("k"), 1 + sum(g ** 2 for g in grad), "a gradient-dependent coefficient")


def test_poisson_natural_flux_with_a_named_coefficient():
    mesh = uw.meshing.UnstructuredSimplexBox(cellSize=1.0 / 4)
    T = uw.discretisation.MeshVariable(_name("T"), mesh, 1, degree=2)
    poisson = uw.systems.Poisson(mesh, u_Field=T)
    poisson.constitutive_model = uw.constitutive_models.DiffusionModel
    poisson.constitutive_model.Parameters.diffusivity = 1.0
    poisson.consistent_jacobian = True
    k = _coefficient(mesh, list(poisson.Unknowns.L))
    poisson.add_natural_bc([k * T.sym[0]], "Top")
    poisson._setup_pointwise_functions()
    assert _boundary_tangent_gap(poisson, poisson.natural_bcs[0], 1, 2) < 1.0e-6


def test_vector_projection_natural_flux_with_a_named_coefficient():
    mesh = uw.meshing.UnstructuredSimplexBox(cellSize=1.0 / 4)
    V = uw.discretisation.MeshVariable(_name("W"), mesh, 2, degree=2)
    projection = uw.systems.Vector_Projection(mesh, V)
    projection.uw_function = sympy.Matrix([[1.0, 0.0]])
    projection.consistent_jacobian = True
    k = _coefficient(mesh, list(projection.Unknowns.L))
    projection.add_natural_bc((k * V.sym[0], k * V.sym[1]), "Top")
    projection._setup_pointwise_functions()
    assert _boundary_tangent_gap(projection, projection.natural_bcs[0], 2, 2) < 1.0e-6


def test_the_adjoint_gradient_through_a_nitsche_wall_matches_a_difference():
    """#1162's measurement: d J / d b by the adjoint against a central difference,
    the inflow wall imposed by Nitsche. 4.1e-5 apart before; the strong wall
    gives 6e-9."""
    mesh = uw.meshing.UnstructuredSimplexBox(minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0),
                                             cellSize=1.0 / 6, qdegree=3)
    x, y = mesh.X
    v = uw.discretisation.MeshVariable(_name("U"), mesh, 2, degree=2)
    p = uw.discretisation.MeshVariable(_name("P"), mesh, 1, degree=1)
    u_adj = uw.discretisation.MeshVariable(_name("UA"), mesh, 2, degree=2)
    p_adj = uw.discretisation.MeshVariable(_name("PA"), mesh, 1, degree=1)
    stokes = uw.systems.Stokes(mesh, velocityField=v, pressureField=p)
    stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
    stokes.constitutive_model.Parameters.shear_viscosity_0 = (
        1 + 5 * stokes.Unknowns.Einv2 ** 2) ** sympy.Rational(-1, 4)
    stokes.consistent_jacobian = True
    stokes.tolerance = 1e-12
    stokes.petsc_options.delValue("ksp_monitor")
    b = uw.expression(_name("b"), 1.0, "body force amplitude")
    stokes.bodyforce = sympy.Matrix([0, b * sympy.cos(sympy.pi * x)])
    for wall in ("Top", "Bottom"):
        stokes.add_dirichlet_bc((0.0, 0.0), wall)
    stokes.add_dirichlet_bc((None, 0.0), "Left")
    stokes.add_nitsche_bc(-4 * y * (1 - y), "Left")
    integrand = x * y * v.sym[0] + v.sym[1] ** 2
    dJdu = sympy.Matrix([[x * y, 2 * v.sym[1]]])

    def J(at):
        b.sym = sympy.Float(at)
        stokes.solve(zero_init_guess=True)
        assert stokes.snes.getConvergedReason() > 0
        return float(uw.maths.Integral(mesh, integrand).evaluate())

    J(1.0)
    _, reason = stokes.adjoint_solve(-stokes.dual_of(dJdu), target=(u_adj, p_adj))
    assert reason > 0
    gradient = stokes.sensitivity(u_adj, b)
    h = 1.0e-5
    difference = (J(1.0 + h) - J(1.0 - h)) / (2 * h)
    assert abs(gradient - difference) < 1.0e-7 * abs(difference), (gradient, difference)
