"""FreeSurface runs in a model with reference quantities, and moves the surface
exactly as it does without them (#998).

It read the mesh nodes from ``mesh.X.coords``, which is in metres in a units
model, and matched them against DM and ``coords_nd`` positions in model units.
No key matched, and the first solve raised "surface trace-mass weights lost
6.281e+00 of 6.281e+00 in coordinate-key matching". It now works in model units
throughout.

The two arms set the same non-dimensional problem: the temperature is written
from model coordinates into ``.data``, and every parameter is one model unit
(``model_units``: a bare number is refused in a units model). So the DM
coordinates after two steps must agree.
"""
import numpy as np
import pytest
import sympy

import underworld3 as uw
from _units_helpers import model_units

pytestmark = [pytest.mark.level_2, pytest.mark.tier_b]

R_IN, R_OUT = 0.5, 1.0


def _two_steps(advective_dt_only=False):
    mesh = uw.meshing.Annulus(radiusInner=R_IN, radiusOuter=R_OUT, cellSize=0.1, qdegree=3)
    x, y = mesh.X
    r = sympy.sqrt(x ** 2 + y ** 2)
    rhat = sympy.Matrix([[x / r, y / r]])
    T = uw.discretisation.MeshVariable("Tfs_998", mesh, 1, degree=2, continuous=True)
    c = np.asarray(T.coords_nd)
    rr, th = np.hypot(c[:, 0], c[:, 1]), np.arctan2(c[:, 1], c[:, 0])
    T.data[:, 0] = np.clip((R_OUT - rr) / (R_OUT - R_IN)
                           + 0.1 * np.sin(4 * th) * np.sin(np.pi * (rr - R_IN) / (R_OUT - R_IN)),
                           0.0, 1.0)
    stokes = uw.systems.Stokes(mesh)
    stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
    stokes.constitutive_model.Parameters.shear_viscosity_0 = model_units(1.0, "Pa*s")
    stokes.bodyforce = (3.0e4 / (R_OUT - R_IN) ** 3) * T.sym[0] * rhat
    stokes.add_essential_bc((0.0, 0.0), "Lower")
    stokes.tolerance = 1.0e-5
    fs = uw.systems.FreeSurface(stokes, "Upper", buoyancy_scale=5.0e5, normal=rhat,
                                composition=T, tangent_advect="shape", surface_filter=20,
                                smooth_length=0.2, consistent_constraint="strong")
    fs._comp_adv.constitutive_model.Parameters.diffusivity = model_units(1.0, "m**2/s")
    fs._comp_adv.add_dirichlet_bc(1.0, "Lower")
    fs._comp_adv.add_dirichlet_bc(0.0, "Upper")
    if advective_dt_only:
        # Lift the surface limit, so the composition solver's step decides.
        fs.solve()
        fs.max_surface_cfl = 1.0e30
        return fs.estimate_dt(advect_scale=1.0)
    for _ in range(2):
        fs.solve()
        fs.advance(fs.estimate_dt(advect_scale=10.0))
    return np.array(mesh.dm.getCoordinatesLocal().array)


def test_the_free_surface_moves_the_same_in_a_units_model():
    plain = _two_steps()
    uw.reset_default_model()
    uw.get_default_model().set_reference_quantities(
        length=uw.quantity(1000, "km"), time=uw.quantity(1, "Myr"),
        mass=uw.quantity(1e24, "kg"), temperature=uw.quantity(1000, "K"),
    )
    try:
        scaled = _two_steps()
    finally:
        uw.reset_default_model()
    # The surface did move, and moved the same way.
    assert np.abs(plain).max() < 1.0 - 1.0e-5
    assert np.abs(scaled - plain).max() < 1.0e-8


def test_the_advective_step_is_in_model_time():
    """The composition solver returns a dimensional time in a units model, and
    ``float`` of it was seconds."""
    plain = _two_steps(advective_dt_only=True)
    uw.reset_default_model()
    uw.get_default_model().set_reference_quantities(
        length=uw.quantity(1000, "km"), time=uw.quantity(1, "Myr"),
        mass=uw.quantity(1e24, "kg"), temperature=uw.quantity(1000, "K"),
    )
    try:
        scaled = _two_steps(advective_dt_only=True)
    finally:
        uw.reset_default_model()
    assert np.isfinite(plain) and scaled == pytest.approx(plain, rel=1e-9)
