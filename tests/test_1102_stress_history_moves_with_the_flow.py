"""A stored viscoelastic stress is carried by the flow.

A stress is made in the first step by a small perturbation of a rigid rotation
(velocity pinned by a stiff penalty body force), then the perturbation is
switched off and the rotation carries it. The viscosity is large against
G dt, so the stress does not relax, and the rotation has no strain rate, so it
adds none; with ``objective_rate="none"`` the components are carried as they
are. After the rotation turns through an angle theta, the stored stress at
R(theta) p must be the first step's stress at p.

The nodal semi-Lagrangian history (the default) froze its commit projection's
input before the trace-back, so its stored stress was rebuilt every step from
the field that had not moved: the momentum solve saw the carried stress, the
record stayed where it was made. Measured 2026-10-18, h = 1/16, eight steps of
1/16 (half a radian), max error relative to the peak stress:

=================  ================  =================
history            against rotated   against unmoved
=================  ================  =================
semi_lagrangian    0.29              1.45  (5e-5 before the fix)
integration_point  0.29              1.55
eulerian           0.086             1.55
=================  ================  =================

The semi-Lagrangian errors are interpolation of a carried peak at this
resolution; both semi-Lagrangian histories fit best a little beyond the
expected angle (0.55 rad here, 0.65-0.7 at four steps of 1/4), a separate
question not tested here.
"""
import numpy as np
import pytest
import sympy

import underworld3 as uw

pytestmark = [pytest.mark.level_2, pytest.mark.tier_a]

DT = 0.0625
STEPS = 8           # theta = STEPS * DT = 0.5, angular velocity 1


@pytest.fixture(autouse=True)
def _reset():
    uw.reset_default_model()
    yield
    uw.reset_default_model()


def _carried_error(transport):
    mesh = uw.meshing.UnstructuredSimplexBox(minCoords=(-1.0, -1.0), maxCoords=(1.0, 1.0),
                                             cellSize=1.0 / 16, qdegree=3)
    x, y = mesh.X
    v = uw.discretisation.MeshVariable("U", mesh, 2, degree=2)
    p = uw.discretisation.MeshVariable("P", mesh, 1, degree=1)
    stokes = uw.systems.Stokes(mesh, velocityField=v, pressureField=p)
    stokes.stress_transport = transport
    stokes.constitutive_model = uw.constitutive_models.ViscoElasticPlasticFlowModel(
        stokes.Unknowns, order=1, objective_rate="none")
    cm = stokes.constitutive_model
    cm.Parameters.shear_viscosity_0 = 1.0e6
    cm.Parameters.shear_modulus = 1.0
    cm.Parameters.dt_elastic = DT
    switch = uw.function.expression(r"A_{\mathrm{p}}", 1.0, "perturbation switch")
    psi = 0.05 * sympy.exp(-((x - 0.5) ** 2 + y ** 2) / 0.08)
    target = sympy.Matrix([[-y + switch * psi.diff(y), x - switch * psi.diff(x)]])
    stokes.bodyforce = -1.0e6 * (v.sym - target)
    for wall in ("Top", "Bottom", "Left", "Right"):
        stokes.add_dirichlet_bc(target, wall)
    stokes.tolerance = 1.0e-8

    r, phi = np.meshgrid(np.linspace(0.3, 0.7, 9), np.linspace(-0.6, 0.6, 13))
    points = np.c_[(r * np.cos(phi)).ravel(), (r * np.sin(phi)).ravel()]
    theta = STEPS * DT
    rotated = points @ np.array([[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]])

    def stored_sxy(at):
        sigma = stokes.Unknowns.DFDt.psi_star[0].sym
        return np.asarray(uw.function.global_evaluate(sigma[0, 1], at)).reshape(-1)

    stokes.solve(timestep=DT)
    made = stored_sxy(points)
    switch.sym = 0.0
    for _ in range(STEPS):
        stokes.solve(timestep=DT, zero_init_guess=False)
    carried = stored_sxy(rotated)
    unmoved = stored_sxy(points)
    scale = np.abs(made).max()
    assert scale > 1.0e-2                       # a stress was made
    return np.abs(carried - made).max() / scale, np.abs(unmoved - made).max() / scale


@pytest.mark.parametrize("transport", ["semi_lagrangian", "integration_point", "eulerian"])
def test_the_stored_stress_rotates_with_the_flow(transport):
    rotated, unmoved = _carried_error(transport)
    assert unmoved > 1.0, (transport, unmoved)     # it left where it was made
    assert rotated < 0.4, (transport, rotated)     # and arrived where the flow took it
