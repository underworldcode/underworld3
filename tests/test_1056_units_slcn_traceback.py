#!/usr/bin/env python3
"""Regression: semi-Lagrangian trace-back works with the units system active (#267).

The SL trace-back (`SemiLagrangian.update_pre_solve`) is performed in the mesh's
NON-DIMENSIONAL (DM) coordinate space. Previously the ``has_units`` branch kept
dimensional coords/velocity and left ``dt`` unitless, so a unit-aware model
crashed in the time loop with::

    ValueError: Cannot subtract arrays with incompatible units:
                'meter' and 'meter / second'

(see ``docs/examples/Tutorial_Thermal_Convection_Units.py``). The fix routes the
whole trace-back through ND/DM space (``coords_nd`` + non-dimensionalised
velocity + non-dimensional ``dt``), so a units-active SLCN solve runs and tracks
the equivalent non-dimensional run.
"""
import numpy as np
import sympy
import pytest

import underworld3 as uw

pytestmark = [pytest.mark.level_2, pytest.mark.tier_b]


def _advect_blob(use_units, vy=20.0, nsteps=5, dt=2.0):
    uw.reset_default_model()
    model = uw.get_default_model()
    if use_units:
        model.set_reference_quantities(
            domain_depth=uw.quantity(1000, "km"),
            plate_velocity=uw.quantity(5, "cm/year"),
            mantle_viscosity=uw.quantity(1e21, "Pa*s"),
            temperature_difference=uw.quantity(1000, "K"),
        )
        mesh = uw.meshing.StructuredQuadBox(
            elementRes=(12, 12), minCoords=(0.0, 0.0), maxCoords=(1000.0, 1000.0), units="km"
        )
        T = uw.discretisation.MeshVariable("T", mesh, 1, degree=2, units="K")
        V = uw.discretisation.MeshVariable("V", mesh, mesh.dim, degree=2, units="m/s")
    else:
        mesh = uw.meshing.StructuredQuadBox(
            elementRes=(12, 12), minCoords=(0.0, 0.0), maxCoords=(1000.0, 1000.0)
        )
        T = uw.discretisation.MeshVariable("T", mesh, 1, degree=2)
        V = uw.discretisation.MeshVariable("V", mesh, mesh.dim, degree=2)

    with uw.synchronised_array_update():
        V.data[...] = 0.0
        V.data[:, 1] = vy  # same ND value in both runs
        c = T.coords_nd  # DM-space coords (identical units vs nondim)
        T.data[:, 0] = np.exp(-(((c[:, 0] - 500) / 120) ** 2 + ((c[:, 1] - 300) / 120) ** 2))

    adv = uw.systems.AdvDiffusionSLCN(mesh, u_Field=T, V_fn=V.sym)
    adv.constitutive_model = uw.constitutive_models.DiffusionModel
    adv.constitutive_model.Parameters.diffusivity = 1.0e-6
    adv.add_dirichlet_bc([0.0], "Bottom")
    adv.add_dirichlet_bc([0.0], "Top")
    for _ in range(nsteps):
        adv.solve(timestep=dt)
    return np.array(T.data[:, 0]).copy()


def test_units_slcn_traceback_runs():
    """A units-active SLCN solve must complete the time loop (was crashing, #267)."""
    Tu = _advect_blob(use_units=True, nsteps=3)
    assert np.all(np.isfinite(Tu)), "units-active SLCN produced non-finite values"
    # blob stays bounded in [0, 1] (small under/overshoot from FE/advection)
    assert Tu.max() < 1.05 and Tu.min() > -0.05


def test_units_slcn_matches_nondimensional():
    """A units-active advection must track the equivalent non-dimensional run.

    They share identical ND values, so the trace-back (done in ND space) gives the
    same transport. A small residual (~1e-3) remains from the constitutive
    diffusivity scaling under units — a separate concern from the trace-back."""
    Tu = _advect_blob(use_units=True)
    Tn = _advect_blob(use_units=False)
    rel = np.linalg.norm(Tu - Tn) / np.linalg.norm(Tn)
    assert rel < 5.0e-3, f"units-active SLCN diverges from nondimensional: rel L2 = {rel:.3e}"
