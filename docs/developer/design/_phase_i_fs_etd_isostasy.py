"""Phase I-2D isostatic-relaxation test: flat free surface, internal
buoyant blob drives surface uplift to isostatic equilibrium.

Mesh: same as `_phase_i_fs_etd_internal.py` —
  - r_inner = 0.5  (no-slip)
  - r_o     = 1.0  (internal boundary = "free surface", flat IC)
  - r_outer = 1.5  (no-slip)
  - heavy fluid in r ≤ r_o (η=1, ρ=1), sticky air outside (η=0.1, ρ=0)

Internal forcing: a buoyant Gaussian blob fixed in lab-frame
coordinates at (r_b, θ_b) = (0.85, 0):
    blob(x, y) = exp(-((x-x0)² + (y-y0)²)/(2σ²))
The blob is *Eulerian* (fixed in space) — it represents a continuously-
refreshed buoyant anomaly. Body force becomes
    f = -(M(x,y) - M_ref(r) - α·blob(x,y)) · r̂
so the buoyant region pushes upward radially. The surface bulges
above the blob until the bulge's gravitational restoring force
balances the blob's buoyancy — isostatic equilibrium.

This test stresses the kinematic-ETD scheme's curvature-γ estimator
in two ways:
  1. Flat IC means h ≈ 0 initially, so the curvature regression
     gets noise rather than the bulge wavenumber.
  2. The relaxation rate near the bulge is set by the local η near
     the surface above the blob, not by the global eta_eff=0.5
     baked into the curvS implementation.

We expect:
  - RK4 (no γ) → tracks equilibrium cleanly
  - RK2 (no γ) → tracks equilibrium cleanly at moderate dt
  - FE → undershoots (low-amplitude steady state) at moderate dt
  - curvS-FSSA → potentially overshoots / oscillates due to wrong γ
  - midpoint-FSSA → similar to curvS

Schemes available (same as internal runner):
  fe, rk2, rk4, curvS, midpoint × FSSA on/off.
"""

import os
import argparse

import numpy as np
import sympy

import petsc4py
import underworld3 as uw

import nest_asyncio
nest_asyncio.apply()


OUT_DIR = "output"


def run(scheme, dt_factor, n_steps, res=20, blob_amp=0.5,
        blob_eta_factor=1.0, adaptive_dt=False, cap_gdt=None,
        cap_gamma=None, target_t=None, verbose=True):
    """If cap_gdt is set and scheme.update ∈ {fe, rk2, rk4} (the
    explicit schemes), the per-step Δt is additionally capped at
    cap_gdt / γ_max where γ_max is estimated from the surface
    curvature at the start of each step.  The (1-α)/γ-prefactor
    schemes (curvS, midpoint) are not capped — they saturate at
    u_n/γ regardless.

    If target_t is set, loop runs until t ≥ target_t (or n_steps,
    whichever comes first)."""
    """One isostatic-relaxation run."""
    use_fssa, update = scheme
    label = (f"FSSA={int(use_fssa)}_UPD={update}_dtf{dt_factor:.2f}"
             f"_isostasy")
    if verbose:
        print(f"\n=== {label} ===", flush=True)

    r_inner = 0.5
    r_o = 1.0
    r_outer = 1.5
    cellsize = 1.0 / res

    # Buoyant blob parameters (Eulerian, fixed in lab frame)
    x_b, y_b = 0.85, 0.0
    sigma_b = 0.06
    blob_fn = sympy.exp(
        -((mesh_x_placeholder := None) or 0))  # built after mesh

    mesh = uw.meshing.AnnulusInternalBoundary(
        radiusOuter=r_outer,
        radiusInternal=r_o,
        radiusInner=r_inner,
        cellSize_Outer=3.0 * cellsize,
        cellSize=cellsize,
        qdegree=3,
    )

    r, th = mesh.CoordinateSystem.R

    if verbose:
        print(f"  mesh: {mesh.X.coords.shape[0]} nodes, "
              f"{mesh._centroids.shape[0]} cells", flush=True)

    # Build the Eulerian blob now that mesh.X is available
    blob_fn = sympy.exp(
        -((mesh.X[0] - x_b) ** 2 + (mesh.X[1] - y_b) ** 2)
        / (2.0 * sigma_b ** 2))

    # MeshVariables
    Vr = uw.discretisation.MeshVariable(
        f"Vr_{label}", mesh, vtype=uw.VarType.SCALAR, degree=1,
        continuous=True, varsymbol=r"v_r")
    v = uw.discretisation.MeshVariable(
        f"V_{label}", mesh, vtype=uw.VarType.VECTOR, degree=2,
        continuous=True, varsymbol=r"\mathbf{v}")
    p = uw.discretisation.MeshVariable(
        f"P_{label}", mesh, vtype=uw.VarType.SCALAR, degree=1,
        continuous=True, varsymbol="p")
    M = uw.discretisation.MeshVariable(
        f"M_{label}", mesh, vtype=uw.VarType.SCALAR, degree=0,
        continuous=False, varsymbol=r"\rho")

    # Initial layer indicator: 1 inside r_o (heavy), 0 outside (air)
    layer_fn = sympy.Piecewise((1.0, r <= r_o), (0.0, True))
    M.data[:, 0] = np.asarray(uw.function.evaluate(
        layer_fn, M.coords)).flatten()

    # Internal boundary node identification (flat IC — no perturbation)
    X_initial = mesh.X.coords.copy()
    R_initial = np.sqrt(X_initial[:, 0] ** 2 + X_initial[:, 1] ** 2)
    THETA_initial = np.arctan2(X_initial[:, 1], X_initial[:, 0])
    is_internal = np.abs(R_initial - r_o) < 0.5 * cellsize / r_o
    internal_idx = np.where(is_internal)[0]
    sort_order = np.argsort(THETA_initial[internal_idx])
    internal_idx = internal_idx[sort_order]
    internal_th = THETA_initial[internal_idx]
    if verbose:
        print(f"  {len(internal_idx)} nodes on internal boundary",
              flush=True)

    # Diffuser (Poisson) for boundary-deformation propagation
    diffuser = uw.systems.Poisson(mesh, Vr)
    diffuser.constitutive_model = uw.constitutive_models.DiffusionModel
    diffuser.constitutive_model.Parameters.diffusivity = 1.0

    # Trivial initial solve to initialize the diffuser's SNES — needed
    # because we don't apply any initial perturbation in this test.
    diffuser.add_essential_bc(sympy.Matrix([0.0]),
                              mesh.boundaries.Upper.name)
    diffuser.add_essential_bc(sympy.Matrix([0.0]),
                              mesh.boundaries.Internal.name)
    diffuser.add_essential_bc(sympy.Matrix([0.0]),
                              mesh.boundaries.Lower.name)
    diffuser.tolerance = 1.0e-3
    diffuser.solve()

    # Stokes setup
    stokes = uw.systems.Stokes(mesh, velocityField=v, pressureField=p)
    stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
    eta_layered = 0.1 + 0.9 * M.sym[0]
    if blob_eta_factor != 1.0:
        eta_layered = eta_layered * (1.0 + (blob_eta_factor - 1.0) * blob_fn)
    stokes.constitutive_model.Parameters.shear_viscosity_0 = eta_layered
    stokes.penalty = 0.0
    # Full buoyancy: heavy fluid (M=1) feels gravity, air (M=0) is
    # weightless. The Eulerian blob is an additional buoyant anomaly.
    # No (M - M_ref) subtraction: the previous formulation introduced
    # a non-physical Heaviside step at r=r_o (M Lagrangian rides with
    # the mesh, M_ref Eulerian fixed in space) that breaks RK4's
    # smooth-trial-state assumption.  The Stokes solver finds the
    # hydrostatic pressure self-consistently.
    stokes.bodyforce = (
        -(M.sym[0] - blob_amp * blob_fn)
        * mesh.CoordinateSystem.unit_e_0
    )
    stokes.tolerance = 1.0e-5
    stokes.add_essential_bc((0.0, 0.0), mesh.boundaries.Lower.name)
    stokes.add_essential_bc((0.0, 0.0), mesh.boundaries.Upper.name)

    stokes.solve()

    # FSSA Robin BC at the Internal boundary
    delta_t = uw.function.expression(
        R"\delta t", dt_factor * stokes.estimate_dt(), "Timestep")
    if use_fssa:
        Gamma = mesh.Gamma / sympy.sqrt(mesh.Gamma.dot(mesh.Gamma))
        FSSA_traction = (delta_t * Gamma.dot(v.sym) * Gamma / 2.0)
        stokes.add_natural_bc(FSSA_traction,
                              mesh.boundaries.Internal.name)
        stokes.solve()

    if verbose:
        print(f"  dt = {float(delta_t.value):.4e}", flush=True)

    # Diagnostics buffers
    times = [0.0]
    h_max = []          # max |dr| over internal boundary
    h_at_pole = []      # dr at the node closest to θ=0

    # Index of node closest to θ=0 (where bulge peaks)
    pole_idx_in_internal = int(np.argmin(np.abs(internal_th)))

    def _trap_weights(th_w):
        n = len(th_w)
        dth = np.empty(n)
        dth[1:-1] = 0.5 * (th_w[2:] - th_w[:-2])
        dth[0] = 0.5 * (th_w[1] - (th_w[-1] - 2 * np.pi))
        dth[-1] = 0.5 * ((th_w[0] + 2 * np.pi) - th_w[-2])
        return dth

    def windowed_ks_squared(h_vals, half_window=4):
        n = len(h_vals)
        dth_avg = 2 * np.pi / n
        h2 = np.empty(n)
        for i in range(n):
            ip = (i + 1) % n
            im = (i - 1) % n
            h2[i] = (h_vals[ip] - 2 * h_vals[i] + h_vals[im]) / dth_avg ** 2
        ks_sq = np.empty(n)
        for i in range(n):
            num = 0.0
            den = 0.0
            for j in range(-half_window, half_window + 1):
                k = (i + j) % n
                num += -h2[k] * h_vals[k]
                den += h_vals[k] ** 2
            ks_sq[i] = num / den if den > 1e-30 else 1.0
        return ks_sq

    def fourier_decompose(values, n_modes):
        dth = _trap_weights(internal_th)
        a = np.zeros(n_modes + 1)
        b = np.zeros(n_modes + 1)
        a[0] = float(np.sum(values * dth) / (2 * np.pi))
        for m in range(1, n_modes + 1):
            a[m] = float(np.sum(values * np.cos(m * internal_th) * dth) / np.pi)
            b[m] = float(np.sum(values * np.sin(m * internal_th) * dth) / np.pi)
        return a, b

    def fourier_to_sympy(a, b, theta_sym, tol=1e-12):
        expr = sympy.Float(a[0])
        for m in range(1, len(a)):
            if abs(a[m]) > tol:
                expr = expr + a[m] * sympy.cos(m * theta_sym)
            if abs(b[m]) > tol:
                expr = expr + b[m] * sympy.sin(m * theta_sym)
        return expr

    # Initial diagnostics (h ≡ 0 at flat IC)
    upper_pos = mesh.X.coords[internal_idx]
    upper_r = np.sqrt(upper_pos[:, 0] ** 2 + upper_pos[:, 1] ** 2)
    upper_dr = upper_r - r_o
    h_max.append(float(np.abs(upper_dr).max()))
    h_at_pole.append(float(upper_dr[pole_idx_in_internal]))
    if verbose:
        print(f"  step 0: h_max={h_max[-1]:.4e} "
              f"h_pole={h_at_pole[-1]:+.4e}", flush=True)

    blow_up = False
    dt_history = []
    is_explicit = update in ('fe', 'rk2', 'rk4')
    for step in range(n_steps):
        if target_t is not None and times[-1] >= target_t:
            break

        # Adaptive timestep: re-evaluate at the current Stokes solution
        # so Δt tracks the live CFL as velocities decay.
        if adaptive_dt:
            dt_bulk = dt_factor * stokes.estimate_dt()
            # Surface-stability cap (explicit schemes only):
            #   γ_max from current curvature; Δt ≤ cap/γ_max.
            if cap_gdt is not None and is_explicit:
                if cap_gamma is not None:
                    gamma_max = cap_gamma
                else:
                    upper_pos = mesh.X.coords[internal_idx]
                    upper_r = np.sqrt(upper_pos[:, 0] ** 2
                                      + upper_pos[:, 1] ** 2)
                    upper_dr_now = upper_r - r_o
                    ks_sq_now = (windowed_ks_squared(upper_dr_now,
                                                     half_window=4)
                                 / r_o ** 2)
                    ks_now = np.sqrt(np.maximum(
                        np.abs(ks_sq_now), (1.0 / r_o) ** 2))
                    gamma_max = float(np.max(
                        1.0 / (2.0 * 1.0 * ks_now)))
                dt_surf = cap_gdt / max(gamma_max, 1e-12)
                dt_eff = min(dt_bulk, dt_surf)
            else:
                dt_eff = dt_bulk
            delta_t.sym = dt_eff
        dt_history.append(float(delta_t.value))

        # Diffuse current radial velocity into the interior
        diffuser._reset()
        diffuser.add_essential_bc(sympy.Matrix([0.0]),
                                  mesh.boundaries.Upper.name)
        diffuser.add_essential_bc(
            sympy.Matrix([v.sym.dot(mesh.CoordinateSystem.unit_e_0)]),
            mesh.boundaries.Internal.name)
        diffuser.add_essential_bc(sympy.Matrix([0.0]),
                                  mesh.boundaries.Lower.name)
        diffuser.solve(zero_init_guess=False)

        if update == 'fe':
            displacement = float(delta_t.value) * np.asarray(
                uw.function.evaluate(
                    Vr.sym * mesh.CoordinateSystem.unit_e_0,
                    mesh.X.coords)).reshape(-1, 2)
            mesh._deform_mesh(mesh.X.coords + displacement)

        elif update == 'curvS':
            dt_val = float(delta_t.value)
            upper_pos = mesh.X.coords[internal_idx]
            upper_r = np.sqrt(upper_pos[:, 0] ** 2
                              + upper_pos[:, 1] ** 2)
            upper_dr = upper_r - r_o
            ks_sq = windowed_ks_squared(upper_dr, half_window=4) / r_o ** 2
            ks = np.sqrt(np.maximum(np.abs(ks_sq), (1.0 / r_o) ** 2))
            eta_eff = 1.0  # heavy-fluid viscosity
            gamma = 1.0 / (2.0 * eta_eff * ks)

            Vr_full = np.asarray(uw.function.evaluate(
                Vr.sym, mesh.X.coords)).flatten()
            u_n = Vr_full[internal_idx]

            alpha = np.exp(-dt_val * gamma)
            gd = dt_val * gamma
            phi1 = np.where(gd > 1e-6,
                            (1.0 - alpha) / np.maximum(gamma, 1e-12),
                            dt_val * (1.0 - 0.5 * gd))
            upper_dr_new = upper_dr + phi1 * u_n
            increment = upper_dr_new - upper_dr

            n_modes = max(2, len(internal_th) // 3)
            a_coef, b_coef = fourier_decompose(increment, n_modes)
            inc_fn = fourier_to_sympy(a_coef, b_coef, th)
            diffuser._reset()
            diffuser.add_essential_bc(sympy.Matrix([0.0]),
                                      mesh.boundaries.Upper.name)
            diffuser.add_essential_bc(sympy.Matrix([inc_fn]),
                                      mesh.boundaries.Internal.name)
            diffuser.add_essential_bc(sympy.Matrix([0.0]),
                                      mesh.boundaries.Lower.name)
            diffuser.solve(zero_init_guess=False)
            inc_field = np.asarray(uw.function.evaluate(
                Vr.sym * mesh.CoordinateSystem.unit_e_0,
                mesh.X.coords)).reshape(-1, 2)
            mesh._deform_mesh(mesh.X.coords + inc_field)

        elif update == 'midpoint':
            dt_val = float(delta_t.value)
            eta_eff = 1.0  # heavy-fluid viscosity
            saved_X = mesh.X.coords.copy()

            upper_pos = mesh.X.coords[internal_idx]
            upper_r = np.sqrt(upper_pos[:, 0] ** 2 + upper_pos[:, 1] ** 2)
            upper_dr_n = upper_r - r_o
            ks_sq_n = (windowed_ks_squared(upper_dr_n, half_window=4)
                       / r_o ** 2)
            ks_n = np.sqrt(np.maximum(np.abs(ks_sq_n), (1.0 / r_o) ** 2))
            gamma_n = 1.0 / (2.0 * eta_eff * ks_n)

            Vr_full_n = np.asarray(uw.function.evaluate(
                Vr.sym, mesh.X.coords)).flatten()
            u_n_at_n = Vr_full_n[internal_idx]

            half_dt = 0.5 * dt_val
            alpha_h = np.exp(-half_dt * gamma_n)
            gd_h = half_dt * gamma_n
            phi1_h = np.where(gd_h > 1e-6,
                              (1.0 - alpha_h)
                              / np.maximum(gamma_n, 1e-12),
                              half_dt * (1.0 - 0.5 * gd_h))
            increment_half = phi1_h * u_n_at_n

            n_modes = max(2, len(internal_th) // 3)
            a_h, b_h = fourier_decompose(increment_half, n_modes)
            inc_fn_half = fourier_to_sympy(a_h, b_h, th)
            diffuser._reset()
            diffuser.add_essential_bc(sympy.Matrix([0.0]),
                                      mesh.boundaries.Upper.name)
            diffuser.add_essential_bc(sympy.Matrix([inc_fn_half]),
                                      mesh.boundaries.Internal.name)
            diffuser.add_essential_bc(sympy.Matrix([0.0]),
                                      mesh.boundaries.Lower.name)
            diffuser.solve(zero_init_guess=False)
            inc_field_half = np.asarray(uw.function.evaluate(
                Vr.sym * mesh.CoordinateSystem.unit_e_0,
                mesh.X.coords)).reshape(-1, 2)
            mesh._deform_mesh(mesh.X.coords + inc_field_half)

            stokes.solve(zero_init_guess=False)
            diffuser._reset()
            diffuser.add_essential_bc(sympy.Matrix([0.0]),
                                      mesh.boundaries.Upper.name)
            diffuser.add_essential_bc(
                sympy.Matrix([v.sym.dot(mesh.CoordinateSystem.unit_e_0)]),
                mesh.boundaries.Internal.name)
            diffuser.add_essential_bc(sympy.Matrix([0.0]),
                                      mesh.boundaries.Lower.name)
            diffuser.solve(zero_init_guess=False)

            upper_pos_h = mesh.X.coords[internal_idx]
            upper_r_h = np.sqrt(upper_pos_h[:, 0] ** 2
                                + upper_pos_h[:, 1] ** 2)
            upper_dr_h = upper_r_h - r_o
            ks_sq_h = (windowed_ks_squared(upper_dr_h, half_window=4)
                       / r_o ** 2)
            ks_h = np.sqrt(np.maximum(np.abs(ks_sq_h), (1.0 / r_o) ** 2))
            gamma_h = 1.0 / (2.0 * eta_eff * ks_h)

            Vr_full_h = np.asarray(uw.function.evaluate(
                Vr.sym, mesh.X.coords)).flatten()
            u_n_at_half = Vr_full_h[internal_idx]

            mesh._deform_mesh(saved_X)

            alpha = np.exp(-dt_val * gamma_h)
            gd = dt_val * gamma_h
            phi1 = np.where(gd > 1e-6,
                            (1.0 - alpha) / np.maximum(gamma_h, 1e-12),
                            dt_val * (1.0 - 0.5 * gd))
            upper_dr_new = upper_dr_n + phi1 * u_n_at_half
            increment = upper_dr_new - upper_dr_n

            a_coef, b_coef = fourier_decompose(increment, n_modes)
            inc_fn = fourier_to_sympy(a_coef, b_coef, th)
            diffuser._reset()
            diffuser.add_essential_bc(sympy.Matrix([0.0]),
                                      mesh.boundaries.Upper.name)
            diffuser.add_essential_bc(sympy.Matrix([inc_fn]),
                                      mesh.boundaries.Internal.name)
            diffuser.add_essential_bc(sympy.Matrix([0.0]),
                                      mesh.boundaries.Lower.name)
            diffuser.solve(zero_init_guess=False)
            inc_field = np.asarray(uw.function.evaluate(
                Vr.sym * mesh.CoordinateSystem.unit_e_0,
                mesh.X.coords)).reshape(-1, 2)
            mesh._deform_mesh(mesh.X.coords + inc_field)

        elif update == 'rk2':
            dt_val = float(delta_t.value)
            saved_X = mesh.X.coords.copy()
            unit_r = mesh.CoordinateSystem.unit_e_0
            n_modes = max(2, len(internal_th) // 3)

            k1 = np.asarray(uw.function.evaluate(
                v.sym.dot(unit_r),
                mesh.X.coords[internal_idx])).flatten()

            inc_h = (dt_val * 0.5) * k1
            a_h, b_h = fourier_decompose(inc_h, n_modes)
            inc_fn_h = fourier_to_sympy(a_h, b_h, th)
            diffuser._reset()
            diffuser.add_essential_bc(sympy.Matrix([0.0]),
                                      mesh.boundaries.Upper.name)
            diffuser.add_essential_bc(sympy.Matrix([inc_fn_h]),
                                      mesh.boundaries.Internal.name)
            diffuser.add_essential_bc(sympy.Matrix([0.0]),
                                      mesh.boundaries.Lower.name)
            diffuser.solve(zero_init_guess=False)
            inc_field_h = np.asarray(uw.function.evaluate(
                Vr.sym * unit_r, mesh.X.coords)).reshape(-1, 2)
            mesh._deform_mesh(mesh.X.coords + inc_field_h)

            stokes.solve(zero_init_guess=False)
            k2 = np.asarray(uw.function.evaluate(
                v.sym.dot(unit_r),
                mesh.X.coords[internal_idx])).flatten()

            mesh._deform_mesh(saved_X)
            inc_full = dt_val * k2
            a_f, b_f = fourier_decompose(inc_full, n_modes)
            inc_fn = fourier_to_sympy(a_f, b_f, th)
            diffuser._reset()
            diffuser.add_essential_bc(sympy.Matrix([0.0]),
                                      mesh.boundaries.Upper.name)
            diffuser.add_essential_bc(sympy.Matrix([inc_fn]),
                                      mesh.boundaries.Internal.name)
            diffuser.add_essential_bc(sympy.Matrix([0.0]),
                                      mesh.boundaries.Lower.name)
            diffuser.solve(zero_init_guess=False)
            inc_field = np.asarray(uw.function.evaluate(
                Vr.sym * unit_r, mesh.X.coords)).reshape(-1, 2)
            mesh._deform_mesh(mesh.X.coords + inc_field)

        elif update == 'rk4':
            dt_val = float(delta_t.value)
            saved_X = mesh.X.coords.copy()
            unit_r = mesh.CoordinateSystem.unit_e_0
            n_modes_rk = max(2, len(internal_th) // 3)

            def _set_trial(disp_at_internal):
                mesh._deform_mesh(saved_X)
                ar, br = fourier_decompose(
                    disp_at_internal, n_modes_rk)
                inc_fn_local = fourier_to_sympy(ar, br, th)
                diffuser._reset()
                diffuser.add_essential_bc(
                    sympy.Matrix([0.0]),
                    mesh.boundaries.Upper.name)
                diffuser.add_essential_bc(
                    sympy.Matrix([inc_fn_local]),
                    mesh.boundaries.Internal.name)
                diffuser.add_essential_bc(
                    sympy.Matrix([0.0]),
                    mesh.boundaries.Lower.name)
                diffuser.solve(zero_init_guess=False)
                f = np.asarray(uw.function.evaluate(
                    Vr.sym * unit_r, mesh.X.coords)).reshape(-1, 2)
                mesh._deform_mesh(mesh.X.coords + f)

            def _sample_un():
                return np.asarray(uw.function.evaluate(
                    v.sym.dot(unit_r),
                    mesh.X.coords[internal_idx])).flatten()

            k1 = _sample_un()
            _set_trial((dt_val * 0.5) * k1)
            stokes.solve(zero_init_guess=False)
            k2 = _sample_un()
            _set_trial((dt_val * 0.5) * k2)
            stokes.solve(zero_init_guess=False)
            k3 = _sample_un()
            _set_trial(dt_val * k3)
            stokes.solve(zero_init_guess=False)
            k4 = _sample_un()
            _set_trial((dt_val / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4))

        else:
            raise ValueError(update)

        stokes.solve(zero_init_guess=False)

        upper_pos = mesh.X.coords[internal_idx]
        upper_r = np.sqrt(upper_pos[:, 0] ** 2 + upper_pos[:, 1] ** 2)
        upper_dr = upper_r - r_o
        h_max.append(float(np.abs(upper_dr).max()))
        h_at_pole.append(float(upper_dr[pole_idx_in_internal]))
        times.append(times[-1] + float(delta_t.value))

        if verbose and (step % max(1, n_steps // 10) == 0
                        or step == n_steps - 1):
            extra = (f"  Δt={dt_history[-1]:.3e}"
                     if adaptive_dt else "")
            print(f"  step {step + 1}: t={times[-1]:.4e} "
                  f"h_max={h_max[-1]:.4e} "
                  f"h_pole={h_at_pole[-1]:+.4e}{extra}",
                  flush=True)

        if h_max[-1] > 0.5:
            print(f"  *** BLOW-UP at step {step + 1} ***", flush=True)
            blow_up = True
            break

    upper_pos = mesh.X.coords[internal_idx]
    upper_r = np.sqrt(upper_pos[:, 0] ** 2 + upper_pos[:, 1] ** 2)
    final_dr = upper_r - r_o

    return {
        'label': label,
        'scheme': scheme,
        'dt': float(delta_t.value),
        'times': np.asarray(times, dtype=float),
        'dt_history': np.asarray(dt_history),
        'h_max': np.asarray(h_max),
        'h_pole': np.asarray(h_at_pole),
        'blow_up': blow_up,
        'final_dr': final_dr,
        'final_th': internal_th,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--res', type=int, default=20)
    parser.add_argument('--n-steps', type=int, default=32)
    parser.add_argument('--scheme', type=str, default='integrators',
                        choices=['all', 'fe-fssa', 'curvS-fssa',
                                 'fe-nofssa', 'curvS-nofssa',
                                 'fssa-vs-nofssa',
                                 'midpoint-fssa', 'midpoint-nofssa',
                                 'midpoint-vs-curvS',
                                 'rk2-nofssa', 'rk4-nofssa',
                                 'rk2-fssa', 'rk4-fssa',
                                 'integrators'])
    parser.add_argument('--dt-factor', type=float, default=1.0)
    parser.add_argument('--blob-amp', type=float, default=0.5,
                        help="Buoyant blob density anomaly amplitude.")
    parser.add_argument('--blob-eta', type=float, default=1.0,
                        help="Multiplier on η inside the blob (1 = "
                        "uniform; >1 = high-η contrast).")
    parser.add_argument('--adaptive-dt', action='store_true',
                        help="Re-evaluate stokes.estimate_dt() each "
                        "step (Δt grows as velocities decay).")
    parser.add_argument('--cap-gdt', action='store_true',
                        help="Apply scheme-specific γΔt cap: 1.0 "
                        "for FE, 2.0 for RK2, 2.78 for RK4. curvS / "
                        "midpoint are unaffected (saturation).")
    parser.add_argument('--cap-gamma', type=float, default=None,
                        help="Override the curvature-derived γ_max "
                        "with a fixed value (physical γ for the "
                        "problem). Use when curvature estimator is "
                        "unreliable, e.g. flat-IC isostasy.")
    parser.add_argument('--target-t', type=float, default=None,
                        help="Stop when times[-1] ≥ target_t "
                        "(use with --n-steps as a max-step cap).")
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)

    schemes = {
        'fe-fssa':         (True,  'fe'),
        'curvS-fssa':      (True,  'curvS'),
        'fe-nofssa':       (False, 'fe'),
        'curvS-nofssa':    (False, 'curvS'),
        'midpoint-fssa':   (True,  'midpoint'),
        'midpoint-nofssa': (False, 'midpoint'),
        'rk2-fssa':        (True,  'rk2'),
        'rk2-nofssa':      (False, 'rk2'),
        'rk4-fssa':        (True,  'rk4'),
        'rk4-nofssa':      (False, 'rk4'),
    }
    if args.scheme == 'all':
        run_list = list(schemes.values())
    elif args.scheme == 'fssa-vs-nofssa':
        run_list = [schemes['fe-fssa'], schemes['fe-nofssa'],
                    schemes['curvS-fssa'], schemes['curvS-nofssa']]
    elif args.scheme == 'midpoint-vs-curvS':
        run_list = [schemes['curvS-fssa'], schemes['midpoint-fssa']]
    elif args.scheme == 'integrators':
        run_list = [schemes['fe-nofssa'], schemes['rk2-nofssa'],
                    schemes['rk4-nofssa'], schemes['curvS-fssa'],
                    schemes['midpoint-fssa']]
    else:
        run_list = [schemes[args.scheme]]

    results = []
    for s in run_list:
        try:
            cap = None
            if args.cap_gdt:
                _, upd = s
                cap = {'fe': 1.0, 'rk2': 2.0, 'rk4': 2.78}.get(upd)
            r = run(s, args.dt_factor, args.n_steps, res=args.res,
                    blob_amp=args.blob_amp,
                    blob_eta_factor=args.blob_eta,
                    adaptive_dt=args.adaptive_dt,
                    cap_gdt=cap,
                    cap_gamma=args.cap_gamma,
                    target_t=args.target_t)
            results.append(r)
        except Exception as e:
            print(f"  *** EXCEPTION in {s}: {e!r} ***", flush=True)
            import traceback; traceback.print_exc()

    if not results:
        return

    eta_tag = f"_eta{args.blob_eta:.0f}" if args.blob_eta != 1.0 else ""
    adapt_tag = "_adt" if args.adaptive_dt else ""
    cap_tag = "_cap" if args.cap_gdt else ""
    targ_tag = (f"_T{int(args.target_t)}"
                if args.target_t is not None else "")
    tag = (f"dtf{args.dt_factor:.2f}_n{args.n_steps}"
           f"_isostasy_res{args.res}{eta_tag}{adapt_tag}"
           f"{cap_tag}{targ_tag}")
    out = os.path.join(OUT_DIR, f"phase_i2d_fs_etd_{tag}.npz")
    np.savez(out, **{
        f"{r['label']}_t":      r['times'] for r in results
    }, **{
        f"{r['label']}_hmax":   r['h_max']  for r in results
    }, **{
        f"{r['label']}_hpole":  r['h_pole'] for r in results
    }, **{
        f"{r['label']}_finalDr":  r['final_dr'] for r in results
    }, **{
        f"{r['label']}_finalTh":  r['final_th'] for r in results
    }, **{
        f"{r['label']}_dthistory": r['dt_history'] for r in results
    })
    print(f"\nWrote {out}", flush=True)

    print("\nFinal-step summary:")
    for r in results:
        print(f"  {r['label']:<55} "
              f"h_max={r['h_max'][-1]:.4e} "
              f"h_pole={r['h_pole'][-1]:+.4e}", flush=True)


if __name__ == "__main__":
    main()
