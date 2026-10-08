"""The store smoothing of the integration-point stress history (#737).

Waters and King start-up below Courant one on a pure Maxwell element: the case
that rings. Minutes, so level 2.
"""
import numpy as np
import pytest
import sympy

import underworld3 as uw

pytestmark = [pytest.mark.level_2, pytest.mark.tier_c]   # 15 min: a slow characterisation with a hard baseline, reported not gated (as test_1064 is)



def _cell_scale_content(history, mesh):
    """RMS of the part of the carried point values no per-cell P1 function
    can represent, relative to the field: the mode the store cycle grows."""
    values, pts = history.carried_tensors()
    nq = int(history.psi_star[0].num_points_per_cell)
    ncell = pts.shape[0] // nq
    w = np.asarray(mesh.integration_rule.getData()[1]).reshape(-1); w = w / w.sum()
    A = np.concatenate([np.ones((ncell, nq, 1)), pts.reshape(ncell, nq, mesh.cdim)], axis=2)
    M = np.einsum("cqi,q,cqj->cij", A, w, A)
    raw = values.reshape(values.shape[0], -1)
    S = raw.reshape(ncell, nq, -1)
    beta = np.linalg.solve(M, np.einsum("cqi,q,cqk->cik", A, w, S))
    fit = np.einsum("cqi,cik->cqk", A, beta).reshape(raw.shape)
    rms = lambda a: float(np.sqrt((a ** 2).mean()))
    return rms(raw - fit) / max(rms(raw), 1.0e-300)


def waters_king_start_up(store_smoothing, res=16, dt=0.0125, t_end=2.0, transport="integration_point",
                         return_kind=False):
    """Waters and King start-up on the integration-point history, pure Maxwell,
    below Courant one. Returns u at the centre at t 1, the cell-scale content
    of the carried stress at t 1 and at t_end, and u at t_end."""
    h, Lx, eta, lam, G = 1.0, 1.0, 1.0, 1.0, 1.0
    mesh = uw.meshing.UnstructuredSimplexBox(minCoords=(-Lx, -h), maxCoords=(Lx, h),
                                             cellSize=h / res, qdegree=3, regular=True)
    v = uw.discretisation.MeshVariable(f"U_wk{store_smoothing}", mesh, 2, degree=2)
    p = uw.discretisation.MeshVariable(f"P_wk{store_smoothing}", mesh, 1, degree=1)
    ns = uw.systems.NavierStokes(mesh, v, p, rho=1.0, order=1)
    ns.stress_transport = transport
    ns.constitutive_model = uw.constitutive_models.ViscoElasticPlasticFlowModel(
        ns.Unknowns, order=1, integrator="bdf")
    ns.constitutive_model.Parameters.shear_viscosity_0 = eta
    ns.constitutive_model.Parameters.shear_modulus = eta / lam
    ns.constitutive_model.Parameters.dt_elastic = dt
    ns.add_dirichlet_bc((0.0, 0.0), "Top"); ns.add_dirichlet_bc((0.0, 0.0), "Bottom")
    ns.add_dirichlet_bc((sympy.oo, 0.0), "Left"); ns.add_dirichlet_bc((sympy.oo, 0.0), "Right")
    ns.bodyforce = sympy.Matrix([[G, 0.0]]); ns.tolerance = 1e-6
    if transport == "integration_point":
        ns.DFDt.store_smoothing = store_smoothing
    elif transport == "forward":
        ns.DFDt.flux_smoothing = store_smoothing * mesh.cell_size() ** 2
    # The content has to be read after the trace-back and before the solve: after
    # the store the point values are a P1 field sampled at the points and the
    # cell-scale part is zero by construction, whatever the run is doing.
    latest = {"content": float("nan")}
    if transport == "integration_point":
        carry = ns.DFDt.update_pre_solve
        def carry_and_measure(*args, **kwargs):
            out = carry(*args, **kwargs)
            latest["content"] = _cell_scale_content(ns.DFDt, mesh)
            return out
        ns.DFDt.update_pre_solve = carry_and_measure
    centre = np.array([[0.0, 0.0]])
    content = {}
    u1 = None
    for step in range(int(round(t_end / dt))):
        ns.solve(timestep=dt, zero_init_guess=False)
        t = (step + 1) * dt
        if abs(t - 1.0) < dt / 2:
            u1 = float(np.asarray(uw.function.evaluate(v.sym[0], centre)).reshape(-1)[0])
            content[1.0] = latest["content"]
    content[t_end] = latest["content"]
    u_end = float(np.asarray(uw.function.evaluate(v.sym[0], centre)).reshape(-1)[0])
    if return_kind:
        return u1, content, u_end, type(ns.DFDt).__name__
    return u1, content, u_end


def test_the_store_smoothing_holds_the_cell_scale_mode_of_the_integration_point_history():
    """Waters and King, 1/16, dt 0.0125 (Courant 0.2), pure Maxwell.

    The mode grows from round-off, so its AMPLITUDE is the platform's (a
    checkerboard seeded into the store is projected away within a few steps
    and does not set it); its GROWTH RATE is the scheme's, about 2.2 per unit
    time here: measured without smoothing, the cell-scale content of the
    carried stress goes from 5.8e-6 at t 1 to 5.1e-5 at t 2 (a factor 8.8)
    and rings by t 5. With c = 0.07 in the store it decays instead
    (1.3e-6 at t 1, 2.3e-7 at t 2). The cost on the centre velocity at t 1
    (0.9617 plain) is two percent here and scales with h^2.
    """
    u_plain, plain, _, kind = waters_king_start_up(0.0, return_kind=True)
    u_smooth, smooth, _ = waters_king_start_up(0.07)
    assert kind == "IntegrationPointSemiLagrangian"
    growth = plain[2.0] / plain[1.0]
    assert 4.0 < growth < 20.0, growth                  # e^{gamma}, gamma between 1.4 and 3 per unit time
    assert smooth[2.0] < smooth[1.0], smooth            # held: decaying, not growing
    assert smooth[2.0] < 1.0e-6, smooth                 # and at the round-off floor
    assert abs(u_plain - 0.9617) < 0.002, u_plain
    assert abs(u_smooth - 0.9429) < 0.002, u_smooth
