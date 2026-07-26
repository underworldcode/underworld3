"""SLCN transport on a per-step-moving mesh must stay bounded (issue #423 guard).

The free-surface T blow-up was localised to an exponential instability of the SL
record -> trace -> solve loop that needs only (a) per-step mesh motion, (b) a sharp
front in squeezed cells, and (c) ``old_frame_traceback=True``. No free surface, no
Stokes: a prescribed 4-cell velocity plus a ±0.1%-of-radius mesh wobble reproduces it
(T -> 160 in 60 steps with old-frame ON; bounded with it OFF at any theta).

This test pins the configuration the FreeSurface manager relies on — the standard ALE
path with the monotone clamp and deform-aware foot restore — staying bounded under
motion. The full characterisation and the reproducer with all its ablation flags live in
``~/+Simulations/FreeSurface/annulus_fs_convection/teaching/slcn_minimal_repro.py``.
"""
import numpy as np
import pytest
import sympy

import underworld3 as uw

pytestmark = [pytest.mark.level_2, pytest.mark.tier_b]


def test_slcn_bounded_on_wobbling_squeezed_annulus():
    r_in, r_out = 0.5, 1.0
    squeeze, delta, dt = 0.09, 0.03, 7.0e-6
    mesh = uw.meshing.Annulus(radiusOuter=r_out, radiusInner=r_in, cellSize=0.06, qdegree=3)
    x, y = mesh.X
    r = sympy.sqrt(x ** 2 + y ** 2)
    th = sympy.atan2(y, x)

    c0 = np.array(mesh.X.coords, copy=True)

    def squeezed(amp):
        rr = np.linalg.norm(c0, axis=1)
        tt = np.arctan2(c0[:, 1], c0[:, 0])
        ramp = np.clip((rr - r_in) / (r_out - r_in), 0.0, 1.0)
        return c0 * ((rr + amp * r_out * np.cos(4.0 * tt) * ramp) / rr)[:, None]

    T = uw.discretisation.MeshVariable("Tsl", mesh, 1, degree=2, continuous=True)
    # 4-cell streamfunction velocity, |v| ~ 800 (the convective regime of the FS runs)
    d = r_out - r_in
    psi_sf = 800.0 * (d / np.pi) * sympy.sin(sympy.pi * (r - r_in) / d) * sympy.cos(4 * th)
    V = uw.discretisation.MeshVariable("Vsl", mesh, 2, degree=2, continuous=True)
    vx_e, vy_e = sympy.diff(psi_sf, y), -sympy.diff(psi_sf, x)

    adv = uw.systems.AdvDiffusionSLCN(
        mesh, u_Field=T, V_fn=V.sym, order=1,
        monotone_mode="clamp", old_frame_traceback=False,   # the manager's configuration
    )
    adv.constitutive_model = uw.constitutive_models.DiffusionModel
    adv.constitutive_model.Parameters.diffusivity = 1.0
    adv.tolerance = 1.0e-4
    adv.add_dirichlet_bc(1.0, "Lower")
    adv.add_dirichlet_bc(0.0, "Upper")

    mesh.deform(squeezed(squeeze), dt=dt)

    # sharp cold layer (1-2 cells) against the local deformed outer boundary
    tc = np.asarray(T.coords)
    rr = np.linalg.norm(tc, axis=1)
    tt = np.arctan2(tc[:, 1], tc[:, 0])
    R_local = r_out * (1.0 + squeeze * np.cos(4.0 * tt))
    T.array[:, 0, 0] = np.clip(
        1.0 - np.exp(-np.maximum(R_local - rr, 0.0) / delta), 0.0, 1.0)

    def set_velocity():
        V.array[...] = np.column_stack([
            np.asarray(uw.function.evaluate(vx_e, V.coords)).flatten(),
            np.asarray(uw.function.evaluate(vy_e, V.coords)).flatten(),
        ]).reshape(V.array.shape)

    set_velocity()

    # 12 wobble steps: with old_frame_traceback=True this reaches T > 1.7 by step 10;
    # the standard ALE path must stay next to [0, 1].
    for step in range(1, 13):
        mesh.deform(squeezed(squeeze * (1.0 + 0.01 * (-1) ** step)), dt=dt)
        set_velocity()
        adv.solve(timestep=dt, zero_init_guess=False)

    Tv = np.asarray(T.array[:, 0, 0])
    assert Tv.max() < 1.15 and Tv.min() > -0.15, (
        f"SLCN on a moving mesh lost boundedness: T in [{Tv.min():.3f}, {Tv.max():.3f}] "
        "(issue #423 regression — old-frame-style amplification)")
