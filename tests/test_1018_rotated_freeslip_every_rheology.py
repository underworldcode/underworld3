"""Rotated strong free-slip with every rheology and both momentum solvers.

Rotated free-slip is the free-slip condition we recommend, so it must hold whatever
the constitutive model: viscous, viscoplastic (constant and pressure-dependent
yield), viscoelastic-plastic (BDF, ETD, with yield, the deformation step, the
log-conformation history) and transversely isotropic VEP, in Stokes and in the
Eulerian Navier-Stokes solver.

Checked on a closed cell driven by a body force:
- every step converges;
- on a box the normal velocity vanishes to round-off (the wall normals are exact);
- on an annulus the constraint holds at the nodes along their nodal normals, so
  v . Gamma against the facet normal is the geometric error of a polygonal boundary
  (first order in h, measured 8.2e-2, 4.6e-2, 2.4e-2 at h = 1/4, 1/8, 1/16): it
  depends on the mesh, not the rheology, so each model is held to the viscous
  model's value on the same mesh.
"""
import numpy as np
import pytest
import sympy

import underworld3 as uw

pytestmark = [pytest.mark.level_2]

DT = 0.05
STEPS = 2
RHEOLOGIES = ["viscous", "vp", "vp_dp", "vep_bdf", "vep_etd", "vep_yield", "vep_def", "vep_log", "ti_vep"]


@pytest.fixture(autouse=True)
def _reset():
    uw.reset_default_model()
    yield
    uw.reset_default_model()


def _rheology(solver, kind):
    if kind == "viscous":
        solver.constitutive_model = uw.constitutive_models.ViscousFlowModel
        solver.constitutive_model.Parameters.shear_viscosity_0 = 1.0
        return
    if kind in ("vp", "vp_dp"):
        solver.constitutive_model = uw.constitutive_models.ViscoPlasticFlowModel
        cm = solver.constitutive_model
        cm.Parameters.shear_viscosity_0 = 1.0
        if kind == "vp":
            cm.Parameters.yield_stress = 0.2
        else:
            cm.Parameters.yield_stress = 0.2 + 0.3 * solver.p.sym[0]
            cm.Parameters.yield_stress_min = 0.02          # residual strength in tension
        cm.Parameters.strainrate_inv_II_min = 1.0e-6
        return
    if kind == "ti_vep":
        solver.constitutive_model = uw.constitutive_models.TransverseIsotropicVEPFlowModel(
            solver.Unknowns, order=1)
        cm = solver.constitutive_model
        cm.Parameters.shear_viscosity_0 = 1.0
        cm.Parameters.shear_viscosity_1 = 0.1
        cm.Parameters.shear_modulus = 1.0
        cm.Parameters.dt_elastic = DT
        cm.Parameters.director = sympy.Matrix([1, 1]) / sympy.sqrt(2)
        return
    options = {
        "vep_bdf": dict(integrator="bdf"),
        "vep_etd": dict(integrator="etd"),
        "vep_yield": dict(integrator="bdf"),
        "vep_def": dict(integrator="etd", objective_rate="upper_convected", convected_step="deformation"),
        "vep_log": dict(integrator="etd", objective_rate="upper_convected", stress_history="log_conformation"),
    }[kind]
    solver.constitutive_model = uw.constitutive_models.ViscoElasticPlasticFlowModel(
        solver.Unknowns, order=1, **options)
    cm = solver.constitutive_model
    cm.Parameters.shear_viscosity_0 = 1.0
    cm.Parameters.shear_modulus = 1.0
    cm.Parameters.dt_elastic = DT
    if kind == "vep_yield":
        cm.Parameters.yield_stress = 0.2
        cm.Parameters.strainrate_inv_II_min = 1.0e-6


def _normal_leak(geometry, solver_kind, kind):
    """Run STEPS steps; return rms(v . Gamma) / rms(v) over the free-slip walls."""
    if geometry == "box":
        mesh = uw.meshing.UnstructuredSimplexBox(minCoords=(0, 0), maxCoords=(1, 1), cellSize=0.125, qdegree=3)
        walls = ("Left", "Right", "Top", "Bottom")
    else:
        mesh = uw.meshing.Annulus(radiusOuter=1.0, radiusInner=0.5, cellSize=0.125, qdegree=3)
        walls = ("Upper", "Lower")
    v = uw.discretisation.MeshVariable("U", mesh, 2, degree=2)
    p = uw.discretisation.MeshVariable("P", mesh, 1, degree=1, continuous=True)
    if solver_kind == "stokes":
        solver = uw.systems.Stokes(mesh, velocityField=v, pressureField=p)
    else:
        solver = uw.systems.NavierStokes(mesh, v, p, rho=1.0, order=1)
        solver.stress_transport = "eulerian"
    _rheology(solver, kind)
    x, y = mesh.X
    if geometry == "box":
        solver.bodyforce = sympy.Matrix([[0, sympy.sin(sympy.pi * x)]])
    else:
        r = sympy.sqrt(x**2 + y**2)
        solver.bodyforce = sympy.Matrix([[x / r, y / r]]) * sympy.cos(3 * sympy.atan2(y, x))
    for wall in walls:
        solver.add_rotated_freeslip_bc(0, wall)
    solver.tolerance = 1.0e-6
    for k in range(STEPS):
        solver.solve(timestep=DT, zero_init_guess=(k == 0))
        assert solver.snes.getConvergedReason() > 0, (geometry, solver_kind, kind, k)
    G = mesh.Gamma
    vn = v.sym[0, 0] * G[0] + v.sym[0, 1] * G[1]
    num = sum(float(uw.maths.BdIntegral(mesh=mesh, fn=vn**2, boundary=w).evaluate()) for w in walls)
    den = sum(float(uw.maths.BdIntegral(mesh=mesh, fn=v.sym.dot(v.sym), boundary=w).evaluate()) for w in walls)
    assert den > 0.0
    return (num / den) ** 0.5


@pytest.mark.tier_b
@pytest.mark.parametrize("solver_kind", ["stokes", "ns"])
@pytest.mark.parametrize("kind", RHEOLOGIES)
def test_a_box_holds_free_slip_to_round_off(kind, solver_kind):
    assert _normal_leak("box", solver_kind, kind) < 1.0e-12


@pytest.mark.tier_b
@pytest.mark.parametrize("solver_kind", ["stokes", "ns"])
@pytest.mark.parametrize("kind", ["vp_dp", "vep_yield", "vep_log", "ti_vep"])
def test_an_annulus_holds_free_slip_as_the_viscous_model_does(kind, solver_kind):
    viscous = _normal_leak("annulus", solver_kind, "viscous")
    uw.reset_default_model()
    assert _normal_leak("annulus", solver_kind, kind) <= 1.25 * viscous
