"""Stokes fixtures for the JIT measurements of #823 (tier 1 and tier 2).

Each builder returns ``(stokes, admissible)``: the solver with its constitutive law set
and the Newton tangent selected, and the bounds of any field that is bounded by
construction (used when random states are drawn).
"""
import os
import sys

import sympy
import underworld3 as uw


def _box_stokes():
    mesh = uw.meshing.UnstructuredSimplexBox(cellSize=0.25)
    v = uw.discretisation.MeshVariable("V", mesh, 2, degree=2)
    p = uw.discretisation.MeshVariable("P", mesh, 1, degree=1)
    stokes = uw.systems.Stokes(mesh, velocityField=v, pressureField=p)
    stokes.bodyforce = sympy.Matrix([[0.0, -1.0]])
    return mesh, p, stokes


def build_box():
    """ViscoPlastic: Drucker-Prager yield C + mu p, temperature-dependent viscosity,
    yield-stress and viscosity floors."""
    mesh, p, stokes = _box_stokes()
    T = uw.discretisation.MeshVariable("T", mesh, 1, degree=1)
    stokes.constitutive_model = uw.constitutive_models.ViscoPlasticFlowModel
    P = stokes.constitutive_model.Parameters
    P.shear_viscosity_0 = uw.expression(r"\eta_0", 1.0) * sympy.exp(
        -uw.expression(r"\theta", 3.0) * T.sym[0])
    P.yield_stress = uw.expression(r"C", 0.5) + uw.expression(r"\mu", 0.6) * p.sym[0]
    P.yield_stress_min = uw.expression(r"\tau_{\min}", 0.01)
    P.shear_viscosity_min = uw.expression(r"\eta_{\min}", 1.0e-3)
    return stokes, {"T": (0.0, 1.0)}


def build_powerlaw(numeric_n=False):
    """A power-law viscosity on a NAMED strain-rate invariant, no viscosity cap:
    eta = eta_0 (edot_II / edot_ref)^(1/n - 1), n = 3, either as a constant atom or as
    the number 3."""
    mesh, p, stokes = _box_stokes()
    stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
    edot = uw.expression(r"\dot\varepsilon_{II}", stokes.Unknowns.Einv2, "strain-rate invariant")
    n = sympy.Integer(3) if numeric_n else uw.expression(r"n", 3, "stress exponent")
    stokes.constitutive_model.Parameters.shear_viscosity_0 = (
        uw.expression(r"\eta_0", 1.0) * (edot / uw.expression(r"\dot\varepsilon_0", 1.0)) ** (1 / n - 1))
    return stokes, {}


def build_linear():
    """Constant-viscosity Stokes: the small kernels on which any overhead shows."""
    mesh, p, stokes = _box_stokes()
    stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
    stokes.constitutive_model.Parameters.shear_viscosity_0 = uw.expression(r"\eta", 1.0)
    return stokes, {}


def build_vep():
    """Visco-elasto-plastic, order 1, with a Drucker-Prager yield and floors."""
    mesh, p, stokes = _box_stokes()
    cm = uw.constitutive_models.ViscoElasticPlasticFlowModel(stokes.Unknowns, order=1)
    stokes.constitutive_model = cm
    cm.Parameters.shear_viscosity_0 = 1.0
    cm.Parameters.shear_modulus = 1.0
    cm.Parameters.dt_elastic = sympy.Rational(1, 10)
    cm.Parameters.yield_stress = uw.expression(r"C", 0.5) + uw.expression(r"\mu", 0.3) * p.sym[0]
    cm.Parameters.yield_stress_min = uw.expression(r"\tau_{\min}", 0.01)
    return stokes, {}


def build_ti():
    """Transversely isotropic VEP with a yield on the weak direction (test_1066's law)."""
    mesh, p, stokes = _box_stokes()
    cm = uw.constitutive_models.TransverseIsotropicVEPFlowModel(stokes.Unknowns)
    stokes.constitutive_model = cm
    cm.Parameters.shear_viscosity_0 = 1.0
    cm.Parameters.shear_viscosity_1 = 0.01
    cm.Parameters.shear_modulus = 100.0
    cm.Parameters.dt_elastic = 0.1
    cm.Parameters.yield_stress = 0.5
    cm.Parameters.director = sympy.Matrix([0.0, 1.0])
    cm.Parameters.strainrate_inv_II_min = 1.0e-6
    return stokes, {}


def build_notch():
    """The Spiegelman notch campaign law (power-mean soft minimum)."""
    sys.path.insert(0, os.path.expanduser("~/+Simulations/spiegelman_hardcase/drivers"))
    import notch_model as nm
    S = nm.build(os.path.expanduser("~/+Simulations/spiegelman_hardcase/meshes/notch_mesh1.msh"),
                 1.0e24, 2.5, 30.0, 1, xi=0.0, seed=False, p_degree=0, p_continuous=False,
                 floor_scalar=1.0e-3, unique_params=True)
    # mat is P0 holding 0 or 1; xicap is a non-negative P0 cap
    return S["stokes"], {"mat": (0.0, 1.0), "xicap": (0.0, 3.0)}


BUILDERS = {"box": build_box, "powerlaw": build_powerlaw, "linear": build_linear,
            "vep": build_vep, "ti": build_ti, "notch": build_notch}


def build(name, **kwargs):
    stokes, admissible = BUILDERS[name](**kwargs)
    stokes.consistent_jacobian = True
    return stokes, admissible


def prepare_solve(stokes, name):
    """Boundary conditions and the initial guess for a solve of fixture ``name``;
    returns the keyword arguments for ``stokes.solve``.

    The box fixtures become a lid-driven box (shear everywhere, yielding at the lid
    corners) started from simple shear, the lid's own profile: an uncapped power law
    is singular at rest. The notch carries its own conditions and starts from rest.
    """
    import numpy as np

    if name == "notch":
        return {"zero_init_guess": True}
    stokes.add_dirichlet_bc((1.0, 0.0), "Top")
    stokes.add_dirichlet_bc((0.0, 0.0), "Bottom")
    stokes.add_dirichlet_bc((0.0, 0.0), "Left")
    stokes.add_dirichlet_bc((0.0, 0.0), "Right")
    v0 = np.zeros_like(stokes.u.array)
    v0[:, 0, 0] = stokes.u.coords[:, 1]
    stokes.u.array[...] = v0
    kwargs = {"zero_init_guess": False}
    if name in ("vep", "ti"):
        kwargs["timestep"] = 0.1
    return kwargs
