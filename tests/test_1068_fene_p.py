"""FENE-P (relaxation='fene_p'): the Peterlin closure in the log-conformation
history, checked against the closed-form steady simple shear and the Oldroyd-B
limit of infinite extensibility."""
import numpy as np
import pytest
import sympy
import underworld3 as uw

pytestmark = [pytest.mark.level_2, pytest.mark.tier_b]


def steady_shear_fene_p(Wi, L2, G=1.0, d=2):
    """Steady homogeneous shear of a FENE-P fluid in d dimensions: from
    0 = L c + c L^T - (f c - I)/lambda with L = [[0, gdot], [0, 0]]: c_yy = 1/f,
    c_xy = Wi/f^2, c_xx = (1 + 2 Wi^2/f^2)/f, and f = (L^2 - d)/(L^2 - tr c) closes
    to f^2 (f - 1) = 2 Wi^2 / L^2 (d = 2). Returns (tau_xy / (eta gdot), N1 / G)."""
    roots = np.roots([1.0, -1.0, 0.0, -2.0 * Wi ** 2 / L2])
    f = float(max(r.real for r in roots if abs(r.imag) < 1e-12 and r.real > 0))
    return 1.0 / f, 2.0 * Wi ** 2 / f ** 2


def shear_box(relaxation, L2, dt, steps, tag):
    """The Maxwell shear box, eta = G = 1 (lambda = 1), wall speed 0.5 over a unit
    gap (gdot = 1, Wi = 1), ETD-1 with the log-conformation store, run to a steady
    state; returns the polymer stress (xy, N1) at the centre, by projection."""
    mesh = uw.meshing.UnstructuredSimplexBox(minCoords=(-1.0, -0.5), maxCoords=(1.0, 0.5), cellSize=0.25, qdegree=3)
    v = uw.discretisation.MeshVariable(f"U_{tag}", mesh, 2, degree=2)
    p = uw.discretisation.MeshVariable(f"P_{tag}", mesh, 1, degree=1)
    stokes = uw.systems.Stokes(mesh, velocityField=v, pressureField=p)
    stokes.stress_transport = "backward_nodes"
    stokes.constitutive_model = uw.constitutive_models.ViscoElasticPlasticFlowModel(
        stokes.Unknowns, order=1, integrator="etd", objective_rate="upper_convected",
        stress_history="log_conformation", relaxation=relaxation, element="jeffreys")
    cm = stokes.constitutive_model
    cm.Parameters.shear_viscosity_0 = 1.0
    cm.Parameters.shear_modulus = 1.0
    cm.Parameters.dt_elastic = dt
    if L2 is not None:
        cm.Parameters.extensibility = L2
    stokes.add_dirichlet_bc((0.5, 0.0), "Top")
    stokes.add_dirichlet_bc((-0.5, 0.0), "Bottom")
    stokes.add_dirichlet_bc((sympy.oo, 0.0), "Left")
    stokes.add_dirichlet_bc((sympy.oo, 0.0), "Right")
    stokes.tolerance = 1.0e-8
    for _ in range(steps):
        stokes.solve(timestep=dt, zero_init_guess=False)
    sigma = cm._carried_stress_sym(0)
    point = np.array([[0.0, 0.0]])
    xy = float(np.asarray(uw.function.evaluate(sigma[0, 1], point)).reshape(-1)[0])
    n1 = float(np.asarray(uw.function.evaluate(sigma[0, 0] - sigma[1, 1], point)).reshape(-1)[0])
    return xy, n1


def test_fene_p_steady_shear_converges_to_the_closed_form_at_first_order():
    """Wi = 1, L^2 = 10: f = 1.1510 (f^2 (f - 1) = 0.2), tau_xy = 0.8688 eta gdot,
    N1 = 1.5097 G. The spring factor is read from the record before the step, so
    the scheme's steady state is first order in dt/lambda: measured errors
    +0.0065 / +0.0095 at dt = 0.1 and +0.0032 / +0.0052 at dt = 0.05 (an earlier
    form of the step kept one 1/f too many on the stretching source and sat
    0.03 / 0.13 off at every dt)."""
    xy_ref, n1_ref = steady_shear_fene_p(Wi=1.0, L2=10.0)
    assert abs(xy_ref - 0.8688) < 1e-3 and abs(n1_ref - 1.5097) < 1e-3
    xy1, n11 = shear_box("fene_p", 10.0, 0.1, 80, "fene_a")
    xy2, n12 = shear_box("fene_p", 10.0, 0.05, 160, "fene_b")
    e1 = (abs(xy1 - xy_ref), abs(n11 - n1_ref))
    e2 = (abs(xy2 - xy_ref), abs(n12 - n1_ref))
    assert e1[0] < 0.010 and e1[1] < 0.015, (xy1, n11)
    assert e2[0] < 0.6 * e1[0] and e2[1] < 0.6 * e1[1], (e1, e2)
    # the linear spring at the same dt is the Oldroyd-B steady state, and the
    # FENE-P reduction (13% in shear stress, 25% in N1) is far outside these errors
    xy_ob, n1_ob = shear_box("linear", None, 0.1, 80, "ob_a")
    assert abs(xy_ob - 1.0) < 0.03 and abs(n1_ob - 2.0) < 0.08, (xy_ob, n1_ob)
    assert xy1 < 0.92 * xy_ob and n11 < 0.85 * n1_ob


def test_fene_p_with_infinite_extensibility_is_oldroyd_b():
    xy, n1 = shear_box("fene_p", 1.0e12, 0.1, 10, "fene_inf")
    xy_ob, n1_ob = shear_box("linear", None, 0.1, 10, "ob_10")
    assert abs(xy - xy_ob) < 1e-8 and abs(n1 - n1_ob) < 1e-8


def test_the_element_is_what_was_declared_and_a_maxwell_element_refuses_a_parallel_dashpot():
    mesh = uw.meshing.UnstructuredSimplexBox(minCoords=(-1.0, -0.5), maxCoords=(1.0, 0.5), cellSize=0.5, qdegree=3)
    v = uw.discretisation.MeshVariable("U_elem", mesh, 2, degree=2)
    p = uw.discretisation.MeshVariable("P_elem", mesh, 1, degree=1)
    stokes = uw.systems.Stokes(mesh, velocityField=v, pressureField=p)
    stokes.stress_transport = "backward_nodes"
    stokes.constitutive_model = uw.constitutive_models.ViscoElasticPlasticFlowModel(
        stokes.Unknowns, order=1, integrator="etd", objective_rate="upper_convected", element="jeffreys")
    cm = stokes.constitutive_model
    assert cm.element == "jeffreys" and cm.relaxation == "linear"
    with pytest.raises(ValueError, match="element"):
        uw.constitutive_models.ViscoElasticPlasticFlowModel(stokes.Unknowns, element="burgers")
    # undeclared, the element follows the solvent viscosity
    cm = uw.constitutive_models.ViscoElasticPlasticFlowModel(stokes.Unknowns, order=1, integrator="etd",
                                                              objective_rate="upper_convected")
    assert cm.element == "maxwell"
    cm.Parameters.solvent_viscosity = 0.5
    assert cm.element == "jeffreys"
    # a declared Maxwell element given a parallel dashpot is refused at the first solve
    stokes.constitutive_model = uw.constitutive_models.ViscoElasticPlasticFlowModel(
        stokes.Unknowns, order=1, integrator="etd", objective_rate="upper_convected", element="maxwell")
    cm = stokes.constitutive_model
    cm.Parameters.shear_viscosity_0 = 1.0
    cm.Parameters.shear_modulus = 1.0
    cm.Parameters.solvent_viscosity = 0.5
    stokes.add_dirichlet_bc((0.5, 0.0), "Top")
    stokes.add_dirichlet_bc((-0.5, 0.0), "Bottom")
    with pytest.raises(ValueError, match="parallel dashpot"):
        stokes.solve(timestep=0.1)


def test_fene_p_refuses_an_infinite_extensibility():
    mesh = uw.meshing.UnstructuredSimplexBox(minCoords=(-1.0, -0.5), maxCoords=(1.0, 0.5), cellSize=0.5, qdegree=3)
    v = uw.discretisation.MeshVariable("U_inf", mesh, 2, degree=2)
    p = uw.discretisation.MeshVariable("P_inf", mesh, 1, degree=1)
    stokes = uw.systems.Stokes(mesh, velocityField=v, pressureField=p)
    stokes.stress_transport = "backward_nodes"
    stokes.constitutive_model = uw.constitutive_models.ViscoElasticPlasticFlowModel(
        stokes.Unknowns, order=1, integrator="etd", objective_rate="upper_convected",
        stress_history="log_conformation", element="jeffreys", relaxation="fene_p")
    cm = stokes.constitutive_model
    cm.Parameters.shear_viscosity_0 = 1.0
    cm.Parameters.shear_modulus = 1.0
    stokes.add_dirichlet_bc((0.5, 0.0), "Top")
    stokes.add_dirichlet_bc((-0.5, 0.0), "Bottom")
    with pytest.raises(ValueError, match="extensibility"):
        stokes.solve(timestep=0.1)


def test_fene_p_needs_the_log_conformation_etd_path():
    mesh = uw.meshing.UnstructuredSimplexBox(minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), cellSize=0.5, qdegree=3)
    v = uw.discretisation.MeshVariable("U_fene_bad", mesh, 2, degree=2)
    p = uw.discretisation.MeshVariable("P_fene_bad", mesh, 1, degree=1)
    stokes = uw.systems.Stokes(mesh, velocityField=v, pressureField=p)
    with pytest.raises(NotImplementedError, match="fene_p"):
        uw.constitutive_models.ViscoElasticPlasticFlowModel(stokes.Unknowns, order=1, integrator="bdf",
                                                             relaxation="fene_p")


def test_the_fene_p_encoding_inverts_the_decoding():
    """encode_history(sigma) then the decode gives sigma back, at a non-trivial stress."""
    mesh = uw.meshing.UnstructuredSimplexBox(minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), cellSize=0.5, qdegree=3)
    v = uw.discretisation.MeshVariable("U_fene_enc", mesh, 2, degree=2)
    p = uw.discretisation.MeshVariable("P_fene_enc", mesh, 1, degree=1)
    stokes = uw.systems.Stokes(mesh, velocityField=v, pressureField=p)
    stokes.stress_transport = "backward_nodes"
    cm = uw.constitutive_models.ViscoElasticPlasticFlowModel(
        stokes.Unknowns, order=1, integrator="etd", objective_rate="upper_convected",
        stress_history="log_conformation", relaxation="fene_p")
    cm.Parameters.shear_modulus = 2.0
    cm.Parameters.extensibility = 10.0
    sigma = sympy.Matrix([[3.0, 0.7], [0.7, 0.4]])
    point = np.array([[0.5, 0.5]])
    from underworld3.constitutive_models import _expm_sym2

    def at_point(m):
        return np.array([[float(np.asarray(uw.function.evaluate(m[i, j], point)).reshape(-1)[0]) for j in range(2)]
                         for i in range(2)])

    # the exact inverse: f c = sigma/G + I with f = (L^2 - 2)/(L^2 - tr c)
    c_exact = cm._fene_exact_conformation(sigma)
    f_exact = cm._peterlin_sym(c_exact)
    assert np.abs(at_point(f_exact * c_exact) - (np.array(sigma, dtype=float) / 2.0 + np.eye(2))).max() < 1e-10
    # the history's encoding uses the spring factor of the step as a field: with
    # that field holding the exact f, encode then decode gives sigma back
    cm._fene_f.data[:, 0] = float(np.asarray(uw.function.evaluate(f_exact, point)).reshape(-1)[0])
    c = _expm_sym2(cm.encode_history(sigma))
    back = (cm._fene_f.sym[0] * c - sympy.eye(2)) * cm.Parameters.shear_modulus
    assert np.abs(at_point(back) - np.array(sigma, dtype=float)).max() < 1e-8
