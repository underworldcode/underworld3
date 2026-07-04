"""Continent isostasy with TRUE FREE SURFACE (no sticky air).

Mesh: rock-only annulus from r_inner=0.5 to r_o=1.0. The Upper
boundary at r=r_o is the free surface (deformable, FSSA-friendly);
Lower at r_inner is no-slip.

Continent block: Lagrangian sector |θ|<θ_b ∩ r∈(r_min, r_o]
inside the rock, density anomaly β.

Body force = −(1 − β·B) · r̂, full gravity in ambient rock minus
buoyancy of block. No air layer, no Heaviside subtraction.

Schemes (RK2, RK4, curvS, midpoint) run with adaptive Δt; halfway
and final mesh + B are captured to UW3-native HDF5 + pyvista VTU
+ profile npz for later visualisation.
"""

import os
import argparse
import glob
import numpy as np
import sympy

import pyvista as pv

import nest_asyncio
nest_asyncio.apply()


OUT_DIR = "output"
SNAP_DIR = os.path.join(OUT_DIR, "continent_fs_snapshots")


def _build(res=20, beta=0.2, theta_block_half=0.4, r_min=0.7,
           structured=False, v_degree=2, p_degree=1, stokes_tol=1.0e-5):
    import underworld3 as uw
    r_inner, r_o = 0.5, 1.0
    cellsize = 1.0 / res
    # qdegree must support the velocity space for isoparametric mapping
    qdeg = max(3, v_degree + 1)

    if structured:
        from _structured_annulus import AnnulusStructured
        nA = max(8, int(round(2 * np.pi * r_o / cellsize)))
        nA = nA + (nA % 2)
        nR = max(2, int(round((r_o - r_inner) / cellsize)))
        mesh = AnnulusStructured(
            radiusOuter=r_o, radiusInner=r_inner,
            nRadial=nR, nAngular=nA, qdegree=qdeg,
        )
    else:
        mesh = uw.meshing.Annulus(
            radiusOuter=r_o, radiusInner=r_inner,
            cellSize=cellsize, qdegree=qdeg,
        )
    r, th = mesh.CoordinateSystem.R

    # Variable name carries the element-pair so we never collide
    # with a previously-built field at a different degree
    pair_tag = f"v{v_degree}p{p_degree}"
    Vr = uw.discretisation.MeshVariable(
        f"Vr_cont_fs_snap_{pair_tag}", mesh, vtype=uw.VarType.SCALAR,
        degree=1, continuous=True, varsymbol=r"v_r")
    v = uw.discretisation.MeshVariable(
        f"V_cont_fs_snap_{pair_tag}", mesh, vtype=uw.VarType.VECTOR,
        degree=v_degree, continuous=True, varsymbol=r"\mathbf{v}")
    p = uw.discretisation.MeshVariable(
        f"P_cont_fs_snap_{pair_tag}", mesh, vtype=uw.VarType.SCALAR,
        degree=p_degree, continuous=True, varsymbol="p")
    B = uw.discretisation.MeshVariable(
        f"B_cont_fs_snap_{pair_tag}", mesh, vtype=uw.VarType.SCALAR,
        degree=0, continuous=False, varsymbol=r"\beta")

    centroids = mesh._centroids
    cR = np.sqrt(centroids[:, 0] ** 2 + centroids[:, 1] ** 2)
    cTH = np.arctan2(centroids[:, 1], centroids[:, 0])
    in_block = ((np.abs(cTH) < theta_block_half)
                & (cR > r_min) & (cR <= r_o))
    B.data[:, 0] = in_block.astype(float)

    X_initial = mesh.X.coords.copy()
    R_initial = np.sqrt(X_initial[:, 0] ** 2 + X_initial[:, 1] ** 2)
    THETA_initial = np.arctan2(X_initial[:, 1], X_initial[:, 0])
    # In free-surface mesh, the deforming boundary is "Upper" (r=r_o)
    is_surface = np.abs(R_initial - r_o) < 0.5 * cellsize / r_o
    internal_idx = np.where(is_surface)[0]
    sort_order = np.argsort(THETA_initial[internal_idx])
    internal_idx = internal_idx[sort_order]
    internal_th = THETA_initial[internal_idx]

    diffuser = uw.systems.Poisson(mesh, Vr)
    diffuser.constitutive_model = uw.constitutive_models.DiffusionModel
    diffuser.constitutive_model.Parameters.diffusivity = 1.0
    diffuser.add_essential_bc(sympy.Matrix([0.0]),
                              mesh.boundaries.Upper.name)
    diffuser.add_essential_bc(sympy.Matrix([0.0]),
                              mesh.boundaries.Lower.name)
    diffuser.tolerance = 1.0e-3
    diffuser.solve()

    # Surface-load variable for the rk*_load_sl schemes: a scalar
    # field whose surface DOFs encode predicted Δh, used as a
    # natural-BC traction on the Upper boundary. Initialised to zero;
    # zero-valued surface load = original free-surface BC, so this
    # has no effect on schemes that don't write to it.
    delta_h_load = uw.discretisation.MeshVariable(
        f"dh_load_{pair_tag}", mesh, vtype=uw.VarType.SCALAR,
        degree=2, continuous=True, varsymbol=r"\delta h_{\rm load}")
    delta_h_load.data[:, 0] = 0.0

    stokes = uw.systems.Stokes(mesh, velocityField=v, pressureField=p)
    stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
    stokes.constitutive_model.Parameters.shear_viscosity_0 = 1.0
    stokes.penalty = 0.0
    # Full buoyancy: ambient rock has density 1, block has (1−β),
    # no air layer. Free surface at r_o is now "Upper" boundary.
    stokes.bodyforce = (
        -(1.0 - beta * B.sym[0]) * mesh.CoordinateSystem.unit_e_0
    )
    stokes.tolerance = stokes_tol
    stokes.add_essential_bc((0.0, 0.0), mesh.boundaries.Lower.name)
    # Natural BC on Upper: surface traction = -ρ·g·Δh r̂ (linearised
    # FSSA-style load). With ρ·g = 1 (nondim), traction is just -Δh
    # times the outward normal. Zero-valued by default → free surface.
    stokes.add_natural_bc(
        -delta_h_load.sym[0] * mesh.CoordinateSystem.unit_e_0,
        mesh.boundaries.Upper.name)
    stokes.solve()

    # Identify surface DOFs of the load variable (its DOF coordinates
    # may differ from the mesh-vertex coordinates because it's degree=2).
    dh_coords = delta_h_load.coords
    dh_R = np.sqrt(dh_coords[:, 0] ** 2 + dh_coords[:, 1] ** 2)
    dh_TH = np.arctan2(dh_coords[:, 1], dh_coords[:, 0])
    dh_surface_mask = np.abs(dh_R - r_o) < 0.5 * cellsize / r_o
    dh_surface_idx = np.where(dh_surface_mask)[0]
    dh_surface_th = dh_TH[dh_surface_idx]
    sort = np.argsort(dh_surface_th)
    dh_surface_idx = dh_surface_idx[sort]
    dh_surface_th = dh_surface_th[sort]

    # Curved-element initial area for ΔA reference
    area_uw_initial = abs(float(uw.maths.Integral(mesh, 1.0).evaluate()))

    return {
        'mesh': mesh, 'r_o': r_o,
        'th_sym': th, 'r_sym': r,
        'Vr': Vr, 'v': v, 'p': p, 'B': B,
        'delta_h_load': delta_h_load,
        'dh_surface_idx': dh_surface_idx,
        'dh_surface_th': dh_surface_th,
        'diffuser': diffuser, 'stokes': stokes,
        'internal_idx': internal_idx, 'internal_th': internal_th,
        'area_uw_initial': area_uw_initial,
    }


_GAMMA_HISTORY_MAX = [0.0]  # module-level retained estimate
_GAMMA_HISTORY = []         # raw γ_eff per step for diagnostics

# History buffers for multistep schemes (AB2-SL etc.). Reset between runs.
_U_N_EFF_PREV = [None]      # u_n_eff^{n-1} on the surface θ-grid
_DT_PREV = [None]           # Δt^{n-1} (used to derive variable-step AB coeffs)
# History for BDF schemes: h^n and h^{n-1} as Fourier coefficient arrays.
_H_PREV_FOURIER = [None]    # (a, b) Fourier coeffs of h^{n-1} on the start-of-step mesh
_BDF_BOOTSTRAPPED = [False] # True after one BDF1 step has produced h^n→h^{n+1}


def _reset_gamma_history():
    _GAMMA_HISTORY_MAX[0] = 0.0
    del _GAMMA_HISTORY[:]
    _U_N_EFF_PREV[0] = None
    _DT_PREV[0] = None
    _H_PREV_FOURIER[0] = None
    _BDF_BOOTSTRAPPED[0] = False


def _step(state, scheme, dt_factor, dt_cap=None,
          dt_cap_mode='fixed', dt_cap_c=0.5):
    """Take one adaptive-Δt step using `scheme` (FE/RK2/RK4/curvS/midpoint).

    Δt selection:
      dt_cap_mode='fixed'  → Δt = min(dt_factor·estimate_dt, dt_cap)
      dt_cap_mode='relax'  → Δt = min(dt_factor·estimate_dt,
                                       dt_cap_c / γ_eff)
        where γ_eff = -⟨u_n, h⟩ / ⟨h, h⟩ is the dominant surface
        relaxation rate from the current state. Amplitude-invariant.
      dt_cap_mode='none'   → Δt = dt_factor·estimate_dt (no cap)
    Returns (h_pole, dt_used)."""
    import underworld3 as uw
    mesh = state['mesh']; v = state['v']
    Vr = state['Vr']; diffuser = state['diffuser']
    stokes = state['stokes']
    internal_idx = state['internal_idx']
    internal_th = state['internal_th']
    th = state['th_sym']; r_o = state['r_o']
    unit_r = mesh.CoordinateSystem.unit_e_0

    dt = dt_factor * stokes.estimate_dt()
    if dt_cap_mode == 'fixed' and dt_cap is not None:
        dt = min(dt, dt_cap)
    elif dt_cap_mode == 'relax':
        # γ_eff from monotone history of |⟨u_n, h⟩| / ⟨h, h⟩.
        # Once a fast mode reveals itself in the surface state we
        # retain its γ as the binding constraint (damping
        # requirement). Otherwise an L² estimator will report the
        # slow dominant mode and let Δt grow large enough to
        # destabilise the fast modes — observed as wave-like step-
        # to-step oscillations in h_pole.
        upper_pos = mesh.X.coords[internal_idx]
        upper_r = np.sqrt(upper_pos[:, 0] ** 2 + upper_pos[:, 1] ** 2)
        h = upper_r - r_o
        u_n_now = np.asarray(uw.function.evaluate(
            v.sym.dot(unit_r), upper_pos)).flatten()
        num = abs(float(np.sum(u_n_now * h)))
        den = float(np.sum(h * h))
        if den > 1.0e-30 and num > 1.0e-30:
            gamma_eff = num / den
            _GAMMA_HISTORY.append(gamma_eff)
            if gamma_eff > _GAMMA_HISTORY_MAX[0]:
                _GAMMA_HISTORY_MAX[0] = gamma_eff
            gamma_used = _GAMMA_HISTORY_MAX[0]
            dt_relax = dt_cap_c / gamma_used
            dt = min(dt, dt_relax)
    n_modes = max(2, len(internal_th) // 3)

    # Trapezoid weights for Fourier on irregular θ
    def trap_w(thw):
        n = len(thw); d = np.empty(n)
        d[1:-1] = 0.5 * (thw[2:] - thw[:-2])
        d[0] = 0.5 * (thw[1] - (thw[-1] - 2 * np.pi))
        d[-1] = 0.5 * ((thw[0] + 2 * np.pi) - thw[-2])
        return d

    def fourier_decomp(values, n_modes):
        dthw = trap_w(internal_th)
        a = np.zeros(n_modes + 1); b = np.zeros(n_modes + 1)
        a[0] = float(np.sum(values * dthw) / (2 * np.pi))
        for m in range(1, n_modes + 1):
            a[m] = float(np.sum(values * np.cos(m * internal_th)
                                 * dthw) / np.pi)
            b[m] = float(np.sum(values * np.sin(m * internal_th)
                                 * dthw) / np.pi)
        return a, b

    def fourier_to_sympy(a, b, theta_sym):
        expr = sympy.Float(a[0])
        for m in range(1, len(a)):
            if abs(a[m]) > 1e-12:
                expr = expr + a[m] * sympy.cos(m * theta_sym)
            if abs(b[m]) > 1e-12:
                expr = expr + b[m] * sympy.sin(m * theta_sym)
        return expr

    def deform_by_inc(disp_internal):
        a, b = fourier_decomp(disp_internal, n_modes)
        inc_fn = fourier_to_sympy(a, b, th)
        diffuser._reset()
        diffuser.add_essential_bc(sympy.Matrix([inc_fn]),
                                  mesh.boundaries.Upper.name)
        diffuser.add_essential_bc(sympy.Matrix([0.0]),
                                  mesh.boundaries.Lower.name)
        diffuser.solve(zero_init_guess=False)
        f = np.asarray(uw.function.evaluate(
            Vr.sym * unit_r, mesh.X.coords)).reshape(-1, 2)
        mesh._deform_mesh(mesh.X.coords + f)

    def sample_un():
        return np.asarray(uw.function.evaluate(
            v.sym.dot(unit_r),
            mesh.X.coords[internal_idx])).flatten()

    unit_t = mesh.CoordinateSystem.unit_e_1

    def sample_ut():
        return np.asarray(uw.function.evaluate(
            v.sym.dot(unit_t),
            mesh.X.coords[internal_idx])).flatten()

    def fourier_eval_numpy(a, b, theta_array):
        """Evaluate a Fourier series (a_m, b_m) at arbitrary thetas.
        Periodic by construction, so trace-back angles need no wrapping."""
        out = np.full_like(theta_array, a[0], dtype=float)
        for m in range(1, len(a)):
            if abs(a[m]) > 1e-12:
                out += a[m] * np.cos(m * theta_array)
            if abs(b[m]) > 1e-12:
                out += b[m] * np.sin(m * theta_array)
        return out

    saved_X = mesh.X.coords.copy()

    if scheme == 'fe':
        # Diffuse v·r̂; FE deformation
        diffuser._reset()
        diffuser.add_essential_bc(
            sympy.Matrix([v.sym.dot(unit_r)]),
            mesh.boundaries.Upper.name)
        diffuser.add_essential_bc(sympy.Matrix([0.0]),
                                  mesh.boundaries.Lower.name)
        diffuser.solve(zero_init_guess=False)
        disp = dt * np.asarray(uw.function.evaluate(
            Vr.sym * unit_r, mesh.X.coords)).reshape(-1, 2)
        mesh._deform_mesh(mesh.X.coords + disp)

    elif scheme == 'rk2':
        k1 = sample_un()
        deform_by_inc((dt * 0.5) * k1)
        stokes.solve(zero_init_guess=False)
        k2 = sample_un()
        mesh._deform_mesh(saved_X)
        deform_by_inc(dt * k2)

    elif scheme == 'fe_sl':
        # Forward-Euler with semi-Lagrangian-horizontal surface
        # advection — single-stage stability baseline.
        upper_pos0 = mesh.X.coords[internal_idx]
        h0 = np.sqrt(upper_pos0[:, 0] ** 2 + upper_pos0[:, 1] ** 2) - r_o
        a_h, b_h = fourier_decomp(h0, n_modes)
        u_n = sample_un()
        u_t = sample_ut()
        theta_back = internal_th - dt * u_t / r_o
        h_back = fourier_eval_numpy(a_h, b_h, theta_back)
        u_n_eff = u_n + (h_back - h0) / dt
        deform_by_inc(dt * u_n_eff)

    elif scheme == 'ab2_sl':
        # Adams–Bashforth-2 with SL-horizontal surface advection.
        # One Stokes solve per step; one history term carries u_n_eff
        # from the previous step. In the Δt → 0 limit u_n_eff is the
        # material derivative Dh/Dt, so caching it and extrapolating
        # linearly via AB2 is consistent with the underlying ODE.
        # Variable-step coefficients with ω = Δt^n / Δt^{n-1} keep
        # 2nd-order accuracy under the relaxation-CFL adaptive step.
        #
        # Robustness clamp: when ω > AB2_OMEGA_MAX (Δt grew too fast,
        # typically while γ_eff is still initialising), fall back to
        # FE-SL. The AB2 predictor coefficient (1 + ω/2) explodes
        # for large ω and has no physical justification when the
        # u_n_eff history was sampled under very different conditions.
        # Honest degradation > extrapolating into invalid regimes.
        AB2_OMEGA_MAX = 1.5
        upper_pos0 = mesh.X.coords[internal_idx]
        h0 = np.sqrt(upper_pos0[:, 0] ** 2 + upper_pos0[:, 1] ** 2) - r_o
        a_h, b_h = fourier_decomp(h0, n_modes)
        u_n = sample_un()
        u_t = sample_ut()
        theta_back = internal_th - dt * u_t / r_o
        h_back = fourier_eval_numpy(a_h, b_h, theta_back)
        u_n_eff = u_n + (h_back - h0) / dt
        history_ok = (_U_N_EFF_PREV[0] is not None
                      and _DT_PREV[0] is not None
                      and dt / _DT_PREV[0] <= AB2_OMEGA_MAX)
        if not history_ok:
            inc = dt * u_n_eff   # bootstrap or fallback
        else:
            omega = dt / _DT_PREV[0]
            inc = dt * ((1.0 + 0.5 * omega) * u_n_eff
                        - 0.5 * omega * _U_N_EFF_PREV[0])
        deform_by_inc(inc)
        _U_N_EFF_PREV[0] = u_n_eff.copy()
        _DT_PREV[0] = dt

    elif scheme == 'rk4_sl':
        # RK4 with semi-Lagrangian-horizontal surface advection.
        # Each stage produces an effective u_n that, applied over its
        # stage interval, captures both radial uplift and tangential
        # surface transport. Combined with RK4 weights for u_n.
        upper_pos0 = mesh.X.coords[internal_idx]
        h0 = np.sqrt(upper_pos0[:, 0] ** 2 + upper_pos0[:, 1] ** 2) - r_o
        a_h, b_h = fourier_decomp(h0, n_modes)

        def stage_eff(stage_dt):
            u_n_s = sample_un()
            u_t_s = sample_ut()
            theta_back_s = internal_th - stage_dt * u_t_s / r_o
            h_back_s = fourier_eval_numpy(a_h, b_h, theta_back_s)
            return u_n_s + (h_back_s - h0) / stage_dt

        # Stage 1
        k1 = stage_eff(dt * 0.5)
        mesh._deform_mesh(saved_X); deform_by_inc((dt * 0.5) * k1)
        stokes.solve(zero_init_guess=False)
        # Stage 2
        k2 = stage_eff(dt * 0.5)
        mesh._deform_mesh(saved_X); deform_by_inc((dt * 0.5) * k2)
        stokes.solve(zero_init_guess=False)
        # Stage 3
        k3 = stage_eff(dt)
        mesh._deform_mesh(saved_X); deform_by_inc(dt * k3)
        stokes.solve(zero_init_guess=False)
        # Stage 4
        k4 = stage_eff(dt)
        mesh._deform_mesh(saved_X)
        deform_by_inc((dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4))

    elif scheme == 'bdf2_sl':
        # BDF2 with semi-Lagrangian-horizontal surface advection.
        # First step: BDF1 (= backward Euler), itself implicit.
        # Step n+1:  3/2 h^{n+1} − 2 h^n_back + 1/2 h^{n-1}_back = Δt u_n^{n+1}
        # where h^n_back, h^{n-1}_back are h^n, h^{n-1} traced back along
        # u_t^{n+1} (current tangential velocity, scaled by Δt^n and
        # (Δt^n + Δt^{n-1}) respectively — single-velocity approximation,
        # adequate while u_t × (2Δt) / r_o remains a small angular shift).
        #
        # Implicit solve via damped Picard iteration:
        #   h^{(k+1)} = h^{(k)} + α · (G(h^{(k)}) − h^{(k)})
        # where G(h) is the BDF formula evaluated at h. The damping factor
        #   α = 1 / (1 + γ_used · Δt)
        # under-relaxes when γΔt ≈ 1 (the regime where pure Picard would
        # oscillate). Recovers full Newton-step rate when γΔt → 0.
        BDF_PICARD_TOL = 1.0e-4
        BDF_PICARD_MAX_ITER = 12
        # Damped-Picard (= Newton with the linearised Jacobian dG/dh ≈ -γΔt).
        # γ-floor at γΔt ≥ 1 forces α ≤ 0.5 so the bootstrap step (where
        # γ_used is still 0) doesn't oscillate.
        gamma_used_local = max(_GAMMA_HISTORY_MAX[0], 1.0e-12)
        gdt = max(gamma_used_local * dt, 1.0)
        alpha_picard = 1.0 / (1.0 + gdt)

        upper_pos0 = mesh.X.coords[internal_idx]
        h0 = np.sqrt(upper_pos0[:, 0] ** 2 + upper_pos0[:, 1] ** 2) - r_o
        a_h0, b_h0 = fourier_decomp(h0, n_modes)
        # Initial guess: h_pred = h0 (zero displacement). An explicit
        # FE-SL predictor was tried and rejected — at large Δt where
        # BDF2's A-stability matters, FE overshoots equilibrium and
        # the first Picard residual is *worse* than h0. The undamped
        # h0 predictor is well-conditioned: damped Picard converges
        # geometrically with rate ~0.3 from |G−h|/|G| = 1.
        h_pred = h0.copy()
        n_iter = 0
        residual_history = []
        while True:
            n_iter += 1
            mesh._deform_mesh(saved_X)
            deform_by_inc(h_pred - h0)
            stokes.solve(zero_init_guess=False)
            u_n_new = sample_un()
            u_t_new = sample_ut()
            theta_back_n = internal_th - dt * u_t_new / r_o
            h_n_back = fourier_eval_numpy(a_h0, b_h0, theta_back_n)
            if (not _BDF_BOOTSTRAPPED[0]) or _H_PREV_FOURIER[0] is None:
                G_h = h_n_back + dt * u_n_new
            else:
                a_h_prev, b_h_prev = _H_PREV_FOURIER[0]
                theta_back_nm1 = internal_th - (dt + _DT_PREV[0]) * u_t_new / r_o
                h_nm1_back = fourier_eval_numpy(
                    a_h_prev, b_h_prev, theta_back_nm1)
                G_h = (2.0 / 3.0) * (
                    2.0 * h_n_back - 0.5 * h_nm1_back + dt * u_n_new)
            # Damped Picard update
            h_next = h_pred + alpha_picard * (G_h - h_pred)
            denom = max(np.max(np.abs(G_h)), 1.0e-12)
            residual = float(np.max(np.abs(G_h - h_pred)) / denom)
            residual_history.append(residual)
            print(f"    bdf2 iter {n_iter}: |G-h|/|G|={residual:.3e}, "
                  f"α={alpha_picard:.3f}, γΔt={gamma_used_local*dt:.3f}",
                  flush=True)
            h_pred = h_next
            if residual < BDF_PICARD_TOL or n_iter >= BDF_PICARD_MAX_ITER:
                break
        # Final deformation to converged h_pred
        mesh._deform_mesh(saved_X)
        deform_by_inc(h_pred - h0)
        # Update history
        _H_PREV_FOURIER[0] = (a_h0.copy(), b_h0.copy())
        _DT_PREV[0] = dt
        _BDF_BOOTSTRAPPED[0] = True
        state['_bdf_n_iter'] = n_iter

    elif scheme == 'rk2_sl':
        # Semi-Lagrangian-horizontal kinematic update.
        # Surface evolves by Dh/Dt = u_n: trace each surface point
        # back along the surface by u_t·dt/r_o, sample h there via
        # Fourier interpolation, then add u_n·dt. Interior remains
        # radial-only via the existing diffuser path — mesh quality
        # is preserved exactly as in vanilla rk2.
        upper_pos0 = mesh.X.coords[internal_idx]
        upper_r0 = np.sqrt(upper_pos0[:, 0] ** 2 + upper_pos0[:, 1] ** 2)
        h0 = upper_r0 - r_o
        a_h, b_h = fourier_decomp(h0, n_modes)

        # Stage 1: half-step trace-back using current u_t
        u_n_1 = sample_un()
        u_t_1 = sample_ut()
        theta_back_h = internal_th - (dt * 0.5) * u_t_1 / r_o
        h_back_h = fourier_eval_numpy(a_h, b_h, theta_back_h)
        # Effective normal velocity such that
        # h^{n+½}(θ) - h^n(θ) = (dt/2) · u_n_eff = h_back_h - h0 + (dt/2)·u_n_1
        u_n_eff_h = u_n_1 + (h_back_h - h0) / (dt * 0.5)
        deform_by_inc((dt * 0.5) * u_n_eff_h)
        stokes.solve(zero_init_guess=False)

        # Stage 2: full-step trace-back using midpoint u_t
        u_n_2 = sample_un()
        u_t_2 = sample_ut()
        theta_back = internal_th - dt * u_t_2 / r_o
        h_back = fourier_eval_numpy(a_h, b_h, theta_back)
        u_n_eff = u_n_2 + (h_back - h0) / dt
        mesh._deform_mesh(saved_X)
        deform_by_inc(dt * u_n_eff)

    elif scheme == 'bdf2_load_sl':
        # BDF2-load: same as bdf2_sl but each Picard iteration uses
        # a surface load instead of an actual mesh deformation. Mesh
        # deforms ONCE per step at the end, after Picard converges.
        # Saves 4–11 diffuser+mesh-deform operations per step
        # (BDF2-SL uses 5–12 Picard iterations at c=0.5..4).
        BDF_PICARD_TOL = 1.0e-4
        BDF_PICARD_MAX_ITER = 12
        gamma_used_local = max(_GAMMA_HISTORY_MAX[0], 1.0e-12)
        gdt = max(gamma_used_local * dt, 1.0)
        alpha_picard = 1.0 / (1.0 + gdt)

        delta_h_load = state['delta_h_load']
        dh_surface_idx = state['dh_surface_idx']
        dh_surface_th  = state['dh_surface_th']

        upper_pos0 = mesh.X.coords[internal_idx]
        h0 = np.sqrt(upper_pos0[:, 0] ** 2 + upper_pos0[:, 1] ** 2) - r_o
        a_h0, b_h0 = fourier_decomp(h0, n_modes)
        h_pred = h0.copy()
        n_iter = 0

        def set_load_from_internal(dh_array):
            a_d, b_d = fourier_decomp(dh_array, n_modes)
            delta_h_load.data[dh_surface_idx, 0] = fourier_eval_numpy(
                a_d, b_d, dh_surface_th)

        while True:
            n_iter += 1
            # Apply (predicted h^{n+1,(k)} - h^n) as surface load.
            # Mesh stays at saved_X — no deform/restore.
            set_load_from_internal(h_pred - h0)
            stokes.solve(zero_init_guess=False)
            u_n_new = sample_un()
            u_t_new = sample_ut()
            theta_back_n = internal_th - dt * u_t_new / r_o
            h_n_back = fourier_eval_numpy(a_h0, b_h0, theta_back_n)
            if (not _BDF_BOOTSTRAPPED[0]) or _H_PREV_FOURIER[0] is None:
                G_h = h_n_back + dt * u_n_new
            else:
                a_h_prev, b_h_prev = _H_PREV_FOURIER[0]
                theta_back_nm1 = internal_th - (dt + _DT_PREV[0]) * u_t_new / r_o
                h_nm1_back = fourier_eval_numpy(
                    a_h_prev, b_h_prev, theta_back_nm1)
                G_h = (2.0 / 3.0) * (
                    2.0 * h_n_back - 0.5 * h_nm1_back + dt * u_n_new)
            h_next = h_pred + alpha_picard * (G_h - h_pred)
            denom = max(np.max(np.abs(G_h)), 1.0e-12)
            residual = float(np.max(np.abs(G_h - h_pred)) / denom)
            print(f"    bdf2-load iter {n_iter}: |G-h|/|G|={residual:.3e}, "
                  f"α={alpha_picard:.3f}, γΔt={gamma_used_local*dt:.3f}",
                  flush=True)
            h_pred = h_next
            if residual < BDF_PICARD_TOL or n_iter >= BDF_PICARD_MAX_ITER:
                break
        # Mesh deformation: drop the surface load, then deform mesh once
        delta_h_load.data[:, 0] = 0.0
        deform_by_inc(h_pred - h0)
        _H_PREV_FOURIER[0] = (a_h0.copy(), b_h0.copy())
        _DT_PREV[0] = dt
        _BDF_BOOTSTRAPPED[0] = True
        state['_bdf_n_iter'] = n_iter

    elif scheme in ('rk2_load_sl', 'rk4_load_sl'):
        # RK2/RK4 with surface-load approximation: instead of deforming
        # the mesh between stages, treat the predicted Δh increment
        # as a Neumann surface traction on the upper boundary
        # (linearised FSSA-style: σ·n = -ρ·g·Δh r̂). This skips the
        # diffuser Poisson solve and the mesh deform/restore at every
        # intermediate stage. The mesh deforms ONCE per step at the
        # end, using the standard diffuser path.
        delta_h_load = state['delta_h_load']
        dh_surface_idx = state['dh_surface_idx']
        dh_surface_th  = state['dh_surface_th']

        upper_pos0 = mesh.X.coords[internal_idx]
        h0 = np.sqrt(upper_pos0[:, 0] ** 2 + upper_pos0[:, 1] ** 2) - r_o

        # Solution-only load-validity limiter: max|Δh|/max|h| < ratio_cap.
        # See relax_sl_zoo.py for derivation. The linearised surface
        # load is faithful only when the predicted increment per step
        # is much smaller than the surface displacement itself; this
        # check uses only the current u_n and h to enforce that.
        LOAD_VALIDITY_CAP = 0.25
        u_n_now = sample_un()
        # Floor at 0.01·r_o (1% of domain radius) so the limiter doesn't
        # trap Δt → 0 at startup when h ≈ 0 (e.g. convection from rest).
        # For cases with finite initial h (relaxation, continent isostasy),
        # h_inf is naturally larger than the floor so the floor is dormant.
        h_inf = max(float(np.max(np.abs(h0))), 0.01 * r_o)
        u_n_inf = float(np.max(np.abs(u_n_now)))
        ratio = dt * u_n_inf / h_inf
        if ratio > LOAD_VALIDITY_CAP and u_n_inf > 1e-10:
            dt_safe = LOAD_VALIDITY_CAP * h_inf / u_n_inf
            print(f"    load-validity limiter: dt {dt:.3e} → {dt_safe:.3e} "
                  f"(max|Δh|/max|h| = {ratio:.3f} > {LOAD_VALIDITY_CAP})",
                  flush=True)
            dt = dt_safe

        a_h, b_h = fourier_decomp(h0, n_modes)

        def stage_eff(stage_dt):
            u_n_s = sample_un()
            u_t_s = sample_ut()
            theta_back_s = internal_th - stage_dt * u_t_s / r_o
            h_back_s = fourier_eval_numpy(a_h, b_h, theta_back_s)
            return u_n_s + (h_back_s - h0) / stage_dt

        def set_load(dh_array_internal):
            # dh_array_internal is on internal_th; resample at
            # delta_h_load's surface DOF angles via Fourier.
            a_d, b_d = fourier_decomp(dh_array_internal, n_modes)
            delta_h_load.data[dh_surface_idx, 0] = fourier_eval_numpy(
                a_d, b_d, dh_surface_th)

        if scheme == 'rk2_load_sl':
            # Stage 1: half-step prediction with current u_n
            k1 = stage_eff(dt * 0.5)
            set_load((dt * 0.5) * k1)
            stokes.solve(zero_init_guess=False)
            # Stage 2: full-step rate at the loaded mesh
            k2 = stage_eff(dt)
            final_inc = dt * k2
        else:  # rk4_load_sl
            k1 = stage_eff(dt * 0.5)
            set_load((dt * 0.5) * k1)
            stokes.solve(zero_init_guess=False)
            k2 = stage_eff(dt * 0.5)
            set_load((dt * 0.5) * k2)
            stokes.solve(zero_init_guess=False)
            k3 = stage_eff(dt)
            set_load(dt * k3)
            stokes.solve(zero_init_guess=False)
            k4 = stage_eff(dt)
            final_inc = (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

        # Reset surface load to zero before the actual mesh deformation
        # (mesh will be at its true position; no load needed).
        delta_h_load.data[:, 0] = 0.0
        deform_by_inc(final_inc)

    elif scheme == 'rk2_full':
        # Full-velocity RK2: deform every node by dt·v_node (both
        # components). Stokes velocity is discretely div-free, so
        # this should preserve area to discretisation order — the
        # radial-only diffuser-Poisson smoothing is bypassed.
        # Mesh quality may degrade after enough steps; for the
        # benchmark we just want to see the volume-conservation
        # effect, regridding is a separate concern.
        def sample_v_full():
            return np.asarray(uw.function.evaluate(
                v.sym, mesh.X.coords)).reshape(-1, 2)
        v_full_0 = sample_v_full()
        mesh._deform_mesh(saved_X + (dt * 0.5) * v_full_0)
        stokes.solve(zero_init_guess=False)
        v_full_h = sample_v_full()
        mesh._deform_mesh(saved_X + dt * v_full_h)

    elif scheme == 'rk4':
        k1 = sample_un()
        mesh._deform_mesh(saved_X); deform_by_inc((dt * 0.5) * k1)
        stokes.solve(zero_init_guess=False); k2 = sample_un()
        mesh._deform_mesh(saved_X); deform_by_inc((dt * 0.5) * k2)
        stokes.solve(zero_init_guess=False); k3 = sample_un()
        mesh._deform_mesh(saved_X); deform_by_inc(dt * k3)
        stokes.solve(zero_init_guess=False); k4 = sample_un()
        mesh._deform_mesh(saved_X)
        deform_by_inc((dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4))

    elif scheme in ('curvS', 'midpoint'):
        # Curvature-γ ETD (with optional midpoint sampling)
        eta_eff = 1.0
        upper_pos = mesh.X.coords[internal_idx]
        upper_r = np.sqrt(upper_pos[:, 0] ** 2 + upper_pos[:, 1] ** 2)
        upper_dr = upper_r - r_o
        # Windowed γ
        n = len(upper_dr); dthw = 2 * np.pi / n
        h2 = np.empty(n)
        for i in range(n):
            ip = (i + 1) % n; im = (i - 1) % n
            h2[i] = (upper_dr[ip] - 2*upper_dr[i] + upper_dr[im]) / dthw**2
        ks_sq = np.empty(n)
        W = 4
        for i in range(n):
            num = 0.0; den = 0.0
            for j in range(-W, W + 1):
                k = (i + j) % n
                num += -h2[k] * upper_dr[k]
                den += upper_dr[k] ** 2
            ks_sq[i] = num / den if den > 1e-30 else 1.0
        ks_sq = ks_sq / r_o ** 2
        ks = np.sqrt(np.maximum(np.abs(ks_sq), (1.0 / r_o) ** 2))
        gamma = 1.0 / (2.0 * eta_eff * ks)

        # Diffuse v·r̂ for u_n
        diffuser._reset()
        diffuser.add_essential_bc(
            sympy.Matrix([v.sym.dot(unit_r)]),
            mesh.boundaries.Upper.name)
        diffuser.add_essential_bc(sympy.Matrix([0.0]),
                                  mesh.boundaries.Lower.name)
        diffuser.solve(zero_init_guess=False)
        Vr_full = np.asarray(uw.function.evaluate(
            Vr.sym, mesh.X.coords)).flatten()
        u_n = Vr_full[internal_idx]

        if scheme == 'midpoint':
            half = 0.5 * dt
            alpha_h = np.exp(-half * gamma)
            phi_h = np.where(half * gamma > 1e-6,
                             (1 - alpha_h) / np.maximum(gamma, 1e-12),
                             half * (1 - 0.5 * half * gamma))
            deform_by_inc(phi_h * u_n)
            stokes.solve(zero_init_guess=False)
            diffuser._reset()
            diffuser.add_essential_bc(
                sympy.Matrix([v.sym.dot(unit_r)]),
                mesh.boundaries.Upper.name)
            diffuser.add_essential_bc(sympy.Matrix([0.0]),
                                      mesh.boundaries.Lower.name)
            diffuser.solve(zero_init_guess=False)
            upper_pos_h = mesh.X.coords[internal_idx]
            upper_r_h = np.sqrt(upper_pos_h[:, 0] ** 2
                                + upper_pos_h[:, 1] ** 2)
            upper_dr_h = upper_r_h - r_o
            for i in range(n):
                ip = (i + 1) % n; im = (i - 1) % n
                h2[i] = (upper_dr_h[ip] - 2*upper_dr_h[i] + upper_dr_h[im]) / dthw**2
            for i in range(n):
                num = 0.0; den = 0.0
                for j in range(-W, W + 1):
                    k = (i + j) % n
                    num += -h2[k] * upper_dr_h[k]
                    den += upper_dr_h[k] ** 2
                ks_sq[i] = num / den if den > 1e-30 else 1.0
            ks_sq = ks_sq / r_o ** 2
            ks_h = np.sqrt(np.maximum(np.abs(ks_sq), (1.0 / r_o) ** 2))
            gamma_h = 1.0 / (2.0 * eta_eff * ks_h)
            Vr_full_h = np.asarray(uw.function.evaluate(
                Vr.sym, mesh.X.coords)).flatten()
            u_n_h = Vr_full_h[internal_idx]
            mesh._deform_mesh(saved_X)
            alpha = np.exp(-dt * gamma_h)
            phi = np.where(dt * gamma_h > 1e-6,
                           (1 - alpha) / np.maximum(gamma_h, 1e-12),
                           dt * (1 - 0.5 * dt * gamma_h))
            deform_by_inc(phi * u_n_h)
        else:  # curvS
            alpha = np.exp(-dt * gamma)
            phi = np.where(dt * gamma > 1e-6,
                           (1 - alpha) / np.maximum(gamma, 1e-12),
                           dt * (1 - 0.5 * dt * gamma))
            deform_by_inc(phi * u_n)
    else:
        raise ValueError(scheme)

    stokes.solve(zero_init_guess=False)

    upper_pos = mesh.X.coords[internal_idx]
    upper_r = np.sqrt(upper_pos[:, 0] ** 2 + upper_pos[:, 1] ** 2)
    pole_idx_in = int(np.argmin(np.abs(internal_th)))
    h_pole = float(upper_r[pole_idx_in] - r_o)
    return h_pole, dt


def _make_circle(radius, n=240):
    """A thin closed polyline at given radius (in xy-plane)."""
    theta = np.linspace(0, 2 * np.pi, n + 1)
    pts = np.column_stack([
        radius * np.cos(theta),
        radius * np.sin(theta),
        np.zeros_like(theta)])
    return pv.lines_from_points(pts)


def _capture(state, scheme_name, label_short, h_pole_value,
             beta=0.2):
    """Capture the simulation state at this point for the continent
    geometry.

    Writes:
      1. UW3-native checkpoint via mesh.write_timestep (H5 + XDMF) —
         the deformed mesh + B variable.
      2. A pyvista VTU of the rock pvmesh with cell_data B and
         eff = 1 − β·B (effective density).
      3. A profile npz with surface (θ, δr) AND the UW3
         Integral-of-1 area (curved-element-aware) so the volume
         check is independent of pyvista's straight-edge cell
         representation.
    """
    import underworld3 as uw
    import underworld3.visualisation as vis
    mesh = state['mesh']; B = state['B']
    os.makedirs(SNAP_DIR, exist_ok=True)

    # Curved-element area: ∫ 1 dV over the deformed mesh, using the
    # mesh's own quadrature degree → respects high-order edge nodes
    # that VTU export collapses to straight edges. abs() because the
    # structured-annulus half-mesh produces orientation-signed cells.
    area_uw = abs(float(uw.maths.Integral(mesh, 1.0).evaluate()))
    area_uw_initial = state.get('area_uw_initial', float('nan'))

    uw_filename = f"uw_{scheme_name}_{label_short}"
    mesh.write_timestep(
        filename=uw_filename,
        index=0,
        outputPath=SNAP_DIR,
        meshVars=[B],
        meshUpdates=True,
        create_xdmf=True,
    )

    pvmesh = vis.mesh_to_pv_mesh(mesh)
    B_cell = B.data[:, 0].astype(float)
    eff_cell = 1.0 - beta * B_cell

    pvmesh.cell_data["B"] = B_cell
    pvmesh.cell_data["eff"] = eff_cell
    # No air to threshold out — the entire mesh is rock.
    heavy = pvmesh

    # Also store h_pole as field data so the plot can label it
    heavy.field_data["h_pole"] = np.array([h_pole_value])

    pv_path = os.path.join(SNAP_DIR,
                           f"pv_{scheme_name}_{label_short}.vtu")
    heavy.save(pv_path)

    # 3) Surface profile (θ, δr) at internal boundary
    internal_idx = state['internal_idx']
    internal_th = state['internal_th']
    upper_pos = mesh.X.coords[internal_idx]
    upper_r = np.sqrt(upper_pos[:, 0] ** 2 + upper_pos[:, 1] ** 2)
    upper_dr = upper_r - state['r_o']
    profile_path = os.path.join(
        SNAP_DIR, f"profile_{scheme_name}_{label_short}.npz")
    np.savez(profile_path, theta=internal_th, dr=upper_dr,
             h_pole=h_pole_value, area_uw=area_uw,
             area_uw_initial=area_uw_initial)
    dA = area_uw - area_uw_initial
    pct = 100.0 * dA / area_uw_initial if area_uw_initial else float('nan')
    print(f"  [{scheme_name}/{label_short}] curved area_uw={area_uw:.6f} "
          f"ΔA={dA:+.4e} ({pct:+.4f}%)", flush=True)
    return pv_path


def _render_panel(plotter, idx, vtu_path, scheme_name, full_label,
                  show_bar=False, r_o=1.0):
    heavy = pv.read(vtu_path)
    plotter.subplot(*idx)
    plotter.add_mesh(
        heavy, scalars="eff", show_edges=True,
        edge_color='#888888', line_width=0.3,
        cmap='viridis', clim=[0.78, 1.02],
        scalar_bar_args={'title': 'M − β·B', 'fmt': '%.2f',
                         'vertical': False},
        show_scalar_bar=show_bar,
    )
    ref_circle = _make_circle(r_o, n=400)
    plotter.add_mesh(ref_circle, color='black', line_width=2.0,
                     render_lines_as_tubes=False)
    plotter.add_text(
        f"{scheme_name}\n{full_label}",
        font_size=11, position='upper_left', color='black')
    plotter.view_xy()
    plotter.camera.zoom(1.4)


def _plot_grid(schemes, labels, h_eq, suffix=""):
    """Render the (scheme × label) grid of pyvista heavy-mesh panels
    from saved VTU files."""
    pl = pv.Plotter(shape=(len(schemes), len(labels)),
                    window_size=(2200, 500 * len(schemes)),
                    off_screen=True)
    pl.set_background('white')

    for row, scheme in enumerate(schemes):
        for col, label_short in enumerate(labels):
            vtu = os.path.join(
                SNAP_DIR, f"pv_{scheme}_{label_short}.vtu")
            prof = os.path.join(
                SNAP_DIR, f"profile_{scheme}_{label_short}.npz")
            if not os.path.isfile(vtu):
                continue
            h_pole = (float(np.load(prof)['h_pole'])
                      if os.path.isfile(prof) else None)
            label = (f"{label_short}: h_pole={h_pole:.4e}"
                     if h_pole is not None else label_short)
            _render_panel(pl, (row, col), vtu, scheme, label,
                          show_bar=(row == 0 and col == 0))

    out = os.path.join(
        OUT_DIR, f"phase_i2d_fs_continent_fs_snapshots{suffix}.png")
    pl.screenshot(out)
    pl.close()
    print(f"  wrote {out}", flush=True)


def _plot_profiles(schemes, labels, suffix=""):
    """Surface-profile line plot: dr(θ) for each scheme at each label."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {'rk2': '#1f77b4', 'rk4': '#2ca02c',
              'curvS': '#d62728', 'midpoint': '#ff7f0e'}

    fig, axes = plt.subplots(1, len(labels), figsize=(7 * len(labels), 4),
                             sharey=True)
    if len(labels) == 1:
        axes = [axes]
    for col, label_short in enumerate(labels):
        ax = axes[col]
        for scheme in schemes:
            prof = os.path.join(
                SNAP_DIR, f"profile_{scheme}_{label_short}.npz")
            if not os.path.isfile(prof):
                continue
            d = np.load(prof)
            ax.plot(d['theta'], d['dr'], '-', color=colors.get(scheme),
                    lw=1.4, label=f"{scheme} (h_pole={float(d['h_pole']):.3e})")
        ax.axhline(0.0, color='grey', lw=0.6, alpha=0.5)
        ax.axvline(0.0, color='grey', lw=0.6, alpha=0.5,
                   ls=':')
        ax.set_xlabel("θ (rad)")
        ax.set_title(f"surface profile — {label_short}")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc='best')
        if col == 0:
            ax.set_ylabel("δr = r − r_o")
    fig.suptitle("Free-surface continent: surface profile by scheme",
                 fontsize=12)
    fig.tight_layout()
    out = os.path.join(
        OUT_DIR, f"phase_i2d_fs_continent_fs_profiles{suffix}.png")
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--res', type=int, default=20)
    p.add_argument('--n-steps', type=int, default=16)
    p.add_argument('--dt-factor', type=float, default=1.0)
    p.add_argument('--h-eq', type=float, default=0.060,
                   help="Approx equilibrium peak h_pole; halfway "
                   "snapshot triggers when h_pole crosses h_eq/2.")
    p.add_argument('--beta', type=float, default=0.2)
    p.add_argument('--theta-block-half', type=float, default=0.4)
    p.add_argument('--r-min', type=float, default=0.7)
    p.add_argument('--plot-only', action='store_true',
                   help="Skip simulation; render from existing VTU "
                   "snapshots in output/continent_snapshots/.")
    p.add_argument('--structured', action='store_true',
                   help="Use the polar-quad structured annulus mesh.")
    p.add_argument('--dt-cap', type=float, default=None,
                   help="Hard cap on Δt: each step uses "
                   "min(dt-factor·estimate_dt, dt-cap).")
    p.add_argument('--snap-suffix', type=str, default="",
                   help="Suffix for snapshot dir + output png "
                   "(use to keep multiple capped/uncapped runs).")
    p.add_argument('--schemes', type=str, default=None,
                   help="Comma-separated subset of schemes to run "
                   "(default: rk2,rk4,curvS,midpoint).")
    p.add_argument('--v-degree', type=int, default=2,
                   help="Velocity polynomial degree (default 2 → P2/Q2).")
    p.add_argument('--p-degree', type=int, default=1,
                   help="Pressure polynomial degree (default 1 → P1/Q1). "
                   "Use --v-degree 3 --p-degree 2 for V3/P2 Taylor–Hood.")
    p.add_argument('--stokes-tol', type=float, default=1.0e-5,
                   help="Stokes solver tolerance (default 1e-5).")
    p.add_argument('--dt-cap-mode', type=str, default='fixed',
                   choices=['none', 'fixed', 'relax'],
                   help="Δt cap mode. 'fixed' uses --dt-cap as the "
                   "scalar cap. 'relax' replaces the fixed cap with "
                   "dt_cap_c · 1/γ_eff where γ_eff is the empirical "
                   "surface relaxation rate (amplitude-invariant). "
                   "'none' disables capping.")
    p.add_argument('--dt-cap-c', type=float, default=0.5,
                   help="Safety factor c on the relaxation CFL "
                   "Δt ≤ c/γ_eff (default 0.5; RK4 stable to ≈ 2.78).")
    p.add_argument('--per-step-profile', action='store_true',
                   help="Save the surface profile (theta, dr, h_pole) "
                   "and area_uw at every step into the snapshot dir "
                   "as step_NNNN.npz. Used for convergence sweeps.")
    p.add_argument('--work-log', type=str, default=None,
                   help="Path to a CSV file to append per-step work "
                   "log entries: step,scheme,t,dt,h_pole,area_uw,"
                   "wall_step_s.")
    p.add_argument('--target-time', type=float, default=None,
                   help="If set, stop the run when cumulative simulated "
                   "time exceeds this value (in addition to --n-steps).")
    args = p.parse_args()

    if args.p_degree >= args.v_degree:
        raise SystemExit(
            f"Inf–sup violation: need v_degree > p_degree, "
            f"got v={args.v_degree}, p={args.p_degree}.")

    if args.schemes is not None:
        schemes = [s.strip() for s in args.schemes.split(',') if s.strip()]
    else:
        schemes = ['rk2', 'rk4', 'curvS', 'midpoint']
    labels = ['halfway', 'final']
    h_half = args.h_eq / 2.0

    # Re-target SNAP_DIR for structured runs so we don't clobber
    # unstructured snapshots
    global SNAP_DIR
    if args.structured:
        SNAP_DIR = os.path.join(OUT_DIR, "continent_fs_snapshots_struct")
    if args.snap_suffix:
        SNAP_DIR = SNAP_DIR + args.snap_suffix

    if not args.plot_only:
        import time as _time
        # Open work log once for this whole invocation (append mode)
        worklog_fh = None
        if args.work_log:
            os.makedirs(os.path.dirname(os.path.abspath(args.work_log))
                        or '.', exist_ok=True)
            new_file = not os.path.exists(args.work_log)
            worklog_fh = open(args.work_log, 'a')
            if new_file:
                worklog_fh.write(
                    "scheme,step,t,dt,h_pole,area_uw,wall_step_s\n")

        for scheme in schemes:
            print(f"\n=== {scheme} ===", flush=True)
            state = _build(res=args.res, beta=args.beta,
                           theta_block_half=args.theta_block_half,
                           r_min=args.r_min,
                           structured=args.structured,
                           v_degree=args.v_degree,
                           p_degree=args.p_degree,
                           stokes_tol=args.stokes_tol)
            _reset_gamma_history()
            os.makedirs(SNAP_DIR, exist_ok=True)

            captured_half = False
            h_pole = 0.0
            t_sim = 0.0
            for s in range(args.n_steps):
                t_step_start = _time.time()
                h_pole, dt = _step(state, scheme, args.dt_factor,
                                   dt_cap=args.dt_cap,
                                   dt_cap_mode=args.dt_cap_mode,
                                   dt_cap_c=args.dt_cap_c)
                wall_step = _time.time() - t_step_start
                t_sim += dt
                area_uw_step = float('nan')
                if args.per_step_profile or args.work_log:
                    import underworld3 as uw_in
                    area_uw_step = abs(float(
                        uw_in.maths.Integral(state['mesh'], 1.0).evaluate()))
                if args.per_step_profile:
                    mesh_p = state['mesh']
                    int_idx = state['internal_idx']
                    int_th = state['internal_th']
                    upos = mesh_p.X.coords[int_idx]
                    ur = np.sqrt(upos[:, 0] ** 2 + upos[:, 1] ** 2)
                    udr = ur - state['r_o']
                    np.savez(os.path.join(
                        SNAP_DIR, f"step_{scheme}_{s+1:04d}.npz"),
                        theta=int_th, dr=udr, h_pole=h_pole,
                        t=t_sim, dt=dt,
                        area_uw=area_uw_step,
                        area_uw_initial=state['area_uw_initial'])
                if worklog_fh is not None:
                    worklog_fh.write(
                        f"{scheme},{s+1},{t_sim:.6e},{dt:.6e},"
                        f"{h_pole:+.6e},{area_uw_step:.6e},"
                        f"{wall_step:.4f}\n")
                    worklog_fh.flush()
                print(f"  step {s+1}: h_pole={h_pole:+.4e} "
                      f"Δt={dt:.3e} wall={wall_step:.2f}s "
                      f"t={t_sim:.2f}", flush=True)
                if not captured_half and h_pole >= h_half:
                    _capture(state, scheme, 'halfway', h_pole,
                             beta=args.beta)
                    captured_half = True
                if args.target_time is not None and t_sim >= args.target_time:
                    print(f"  reached target_time={args.target_time}; "
                          f"stopping at step {s+1}", flush=True)
                    break
            _capture(state, scheme, 'final', h_pole,
                     beta=args.beta)

            if not captured_half:
                print(f"  WARNING: {scheme} never reached h_half="
                      f"{h_half:.4e}", flush=True)

        if worklog_fh is not None:
            worklog_fh.close()

    # Render plots from saved VTU + profile files
    suffix = "_struct" if args.structured else ""
    if args.snap_suffix:
        suffix = suffix + args.snap_suffix
    _plot_grid(schemes, labels, args.h_eq, suffix=suffix)
    _plot_profiles(schemes, labels, suffix=suffix)


if __name__ == "__main__":
    main()
