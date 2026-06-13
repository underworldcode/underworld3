"""Phase I (2-D) — free-surface relaxation: FSSA × ETD on the annulus.

Built on `_phase_i_freesurface_relaxation_0d.py` (in the vep-two-stokes
worktree): same exponential-vs-Euler comparison, but now coupled to
Stokes flow on a deforming annular mesh, using AnnulusND_FS.py as the
template for the Stokes + diffuser + mesh-deformation workflow.

Question: is ETD an alternative to (or a useful pair for) the
half-timestep FSSA stabilization that we currently apply via
`stokes.add_natural_bc`?

Setup
-----
  mesh:    annulus, r_i = 0.5, r_o = 1, viscous Stokes, ρg = 1
  pert:    upper surface deformed by sin(10·θ)/20 at t=0
  driver:  body force = -r̂ (gravity towards origin in 2-D)

Schemes (sweep both axes):
  FSSA on/off     — Robin BC `½·ρg·Δt·(u·n̂)·n̂` at upper surface
  Update on/off   — FE: x ← x + Δt·v_diffused
                  — ETD: project boundary δr onto sin(10θ), apply exact
                    exponential to the mode amplitude, reconstruct,
                    diffuse into interior.

Diagnostic: A(t) = mode-10 amplitude of upper-surface radial
displacement. The whole problem is essentially scalar in this mode at
this resolution, so the mode amplitude is a clean reference for
comparing schemes.

Validation:
  - small-Δt FE+FSSA gives the "true" relaxation curve
  - large-Δt FE-no-FSSA blows up (drunken sailor)
  - large-Δt FE+FSSA stays stable but under-relaxes
  - large-Δt ETD-only / ETD+FSSA: under what conditions does the
    exponential update recover the correct decay rate?
"""

import os
import sys
import json
import argparse

import numpy as np
import sympy

import petsc4py
import underworld3 as uw

import nest_asyncio
nest_asyncio.apply()


OUT_DIR = "output"


# -----------------------------------------------------------------
# Single relaxation run with the chosen (fssa, update) pair.
# Returns dict of arrays: t, A_mode10, |A|_max
# -----------------------------------------------------------------

def run(scheme, dt_factor, n_steps, res=20, mode=10, amp0=0.05,
        ic='single', visc_contrast=False, buoyancy=False, verbose=True):
    """Run one annulus free-surface relaxation.

    scheme: tuple (use_fssa: bool, update: 'fe'|'etd'|'curv'|'bdf2'|'etd2')
    dt_factor: dt = dt_factor * estimate_dt()
    n_steps: number of relaxation steps
    res, mode, amp0: mesh + perturbation parameters
    ic: 'single' (sin(mode·θ)) or 'multi' (sin(10θ)/2 + sin(25θ)/4)
    visc_contrast: if True, η spatially varying with a low-η window
    buoyancy: if True, add internal density anomaly (forced free
        surface — no initial surface perturbation, surface responds
        to the bulk-driven flow). Forces ic to 'flat'.
    """
    use_fssa, update = scheme
    if buoyancy:
        ic = 'flat'  # override: surface starts flat, rises in response
    tag_ic = {'multi': 'm', 'single': 's', 'flat': 'f'}.get(ic, 's')
    tag_v = 'V' if visc_contrast else 'v'
    tag_b = 'B' if buoyancy else 'b'
    label = (f"FSSA={int(use_fssa)}_UPD={update}_dtf{dt_factor:.2f}"
             f"_ic{tag_ic}_{tag_v}{tag_b}")
    if verbose:
        print(f"\n=== {label} ===", flush=True)

    r_o = 1.0
    r_i = 0.5
    cellsize = 1.0 / res

    mesh = uw.meshing.Annulus(
        radiusOuter=r_o, radiusInner=r_i, cellSize=cellsize, qdegree=3
    )
    r, th = mesh.CoordinateSystem.R

    # Save the initial undeformed coordinates for ETD δr reconstruction
    X_initial = mesh.X.coords.copy()
    R_initial = np.sqrt(X_initial[:, 0] ** 2 + X_initial[:, 1] ** 2)
    THETA_initial = np.arctan2(X_initial[:, 1], X_initial[:, 0])

    # Identify upper-boundary nodes
    is_upper = R_initial > (r_o - 0.5 * cellsize / r_o)
    upper_idx = np.where(is_upper)[0]
    upper_th = THETA_initial[upper_idx]
    # Sort by theta so projections are stable
    sort_order = np.argsort(upper_th)
    upper_idx = upper_idx[sort_order]
    upper_th = upper_th[sort_order]
    if verbose:
        print(f"  mesh: {mesh.X.coords.shape[0]} nodes, "
              f"{len(upper_idx)} on upper boundary", flush=True)

    # MeshVariables
    M = uw.discretisation.MeshVariable(
        f"M_{label}", mesh, vtype=uw.VarType.SCALAR, degree=1,
        continuous=True, varsymbol=r"{\cal{M}}")
    v = uw.discretisation.MeshVariable(
        f"V_{label}", mesh, vtype=uw.VarType.VECTOR, degree=2,
        continuous=True, varsymbol=r"\mathbf{v}")
    p = uw.discretisation.MeshVariable(
        f"P_{label}", mesh, vtype=uw.VarType.SCALAR, degree=1,
        continuous=True, varsymbol=r"p")
    # Discontinuous P0 density per element.  The element density
    # rides with the deforming mesh: it's set once at the reference
    # configuration and not updated as the mesh moves.  The element
    # carries its initial density into whatever new spatial position
    # it ends up at.
    rho_var = uw.discretisation.MeshVariable(
        f"rho_{label}", mesh, vtype=uw.VarType.SCALAR, degree=0,
        continuous=False, varsymbol=r"\rho")

    # Apply the initial perturbation to the upper surface, smoothed
    # inwards via a Poisson "diffuser".
    if ic == 'multi':
        # Two modes: 10 (dominant) and 25 (faster-relaxing for
        # half-space dispersion γ = ρg/(2η|k|)).
        deform_fn = ((r / r_o) * amp0 *
                     (0.5 * sympy.sin(10 * th)
                      + 0.25 * sympy.sin(25 * th)))
    elif ic == 'flat':
        # No initial perturbation — driven by buoyancy
        deform_fn = sympy.Float(0)
    else:
        deform_fn = (r / r_o) * sympy.sin(mode * th) * amp0
    diffuser = uw.systems.Poisson(mesh, M)
    diffuser.constitutive_model = uw.constitutive_models.DiffusionModel
    diffuser.constitutive_model.Parameters.diffusivity = 100
    diffuser.add_essential_bc(sympy.Matrix([deform_fn]),
                              mesh.boundaries.Upper.name)
    diffuser.add_essential_bc(sympy.Matrix([0.0]),
                              mesh.boundaries.Lower.name)
    diffuser.solve()

    displacement = np.asarray(uw.function.evaluate(
        M.sym * mesh.CoordinateSystem.unit_e_0, mesh.X.coords)).reshape(-1, 2)
    mesh._deform_mesh(mesh.X.coords + displacement)

    # Stokes setup
    stokes = uw.systems.Stokes(mesh, velocityField=v, pressureField=p)
    stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
    if visc_contrast:
        # Lateral viscosity contrast: weak window centred at θ=0 with
        # half-width ~0.4 rad. Drops viscosity by 20× there.
        eta_fn = 1.0 / (1.0 + 19.0 * sympy.exp(-(th * th) / 0.16))
        stokes.constitutive_model.Parameters.shear_viscosity_0 = eta_fn
    else:
        stokes.constitutive_model.Parameters.shear_viscosity_0 = 1.0
    stokes.penalty = 1.0
    # Body force = -(ρ - ρ_ref(r))·r̂.
    #   ρ comes from the per-element discontinuous P0 mesh variable.
    #     Set to 1 (fluid) for all elements at initialization.  As
    #     the mesh deforms the elements ride with their density.
    #   ρ_ref(r) is a sharp piecewise step at the unperturbed
    #     boundary radius r_o: 1 inside, 0 outside.  Evaluated at
    #     the integration point's current radial position.
    # The anomaly (and hence the force) is non-zero only at IPs
    # that have crossed r_o due to mesh deformation.
    r_node = sympy.sqrt(mesh.X[0] ** 2 + mesh.X[1] ** 2)
    # Sharp piecewise reference (per user's spec):
    rho_ref = sympy.Piecewise((1.0, r_node < r_o), (0.0, True))
    # NOTE: a smooth-tanh alternative reference can be activated for
    # debugging the integration of discontinuous body force.
    # Initialise rho element-by-element.  At reference (rest)
    # configuration, all elements are inside r_o, so ρ = 1.
    # For the buoyancy case, also mark the buoyant-blob elements
    # (lighter) at this initialization.
    rho_var.data[:] = 1.0
    if buoyancy:
        # Locate elements whose centroid is in the buoyant blob and
        # reduce their density.
        x0, y0 = 0.7, 0.0
        sig = 0.08
        # Element centroids (one per element for P0)
        centroids = mesh._centroids
        dx = centroids[:, 0] - x0
        dy = centroids[:, 1] - y0
        blob_mag = 0.6 * np.exp(-(dx ** 2 + dy ** 2)
                                / (2.0 * sig ** 2))
        rho_var.data[:, 0] = 1.0 - blob_mag
    rho_scalar = rho_var.sym[0]
    anomaly = rho_scalar - rho_ref
    stokes.bodyforce = -anomaly * mesh.CoordinateSystem.unit_e_0
    stokes.add_essential_bc((0.0, 0.0), mesh.boundaries.Lower.name)

    stokes.solve()  # initial solve to seed estimate_dt()

    # FSSA Robin BC
    delta_t = uw.function.expression(
        R"\delta t", dt_factor * stokes.estimate_dt(), "Timestep")
    if use_fssa:
        Gamma = mesh.Gamma / sympy.sqrt(mesh.Gamma.dot(mesh.Gamma))
        FSSA_traction = delta_t * Gamma.dot(v.sym) * Gamma / 2.0
        stokes.add_natural_bc(FSSA_traction, "Upper")
        stokes.solve()  # re-solve with FSSA

    # Diagnostics buffer
    times = [0.0]
    A_mode = []  # mode-`mode` amplitude (signed)
    A_max = []   # max |radial displacement| anywhere on upper boundary

    def _trap_weights(th_w):
        """Periodic trapezoidal widths for samples sorted by θ."""
        n = len(th_w)
        dth = np.empty(n)
        dth[1:-1] = 0.5 * (th_w[2:] - th_w[:-2])
        dth[0] = 0.5 * (th_w[1] - (th_w[-1] - 2 * np.pi))
        dth[-1] = 0.5 * ((th_w[0] + 2 * np.pi) - th_w[-2])
        return dth

    def project_mode_amp(boundary_disp, m=mode):
        """Recover sin(m·θ) coefficient from boundary δr samples
        ordered by θ. Trapezoidal integration."""
        s = np.sin(m * upper_th)
        dth = _trap_weights(upper_th)
        return float(np.sum(boundary_disp * s * dth) / np.pi)

    def fourier_decompose(values, n_modes):
        """Compute (a_0, a_1..a_M, b_1..b_M) for sum a_0 + Σ a_m·cos
        + b_m·sin reconstruction."""
        dth = _trap_weights(upper_th)
        a = np.zeros(n_modes + 1)
        b = np.zeros(n_modes + 1)
        a[0] = float(np.sum(values * dth) / (2 * np.pi))
        for m in range(1, n_modes + 1):
            a[m] = float(np.sum(values * np.cos(m * upper_th) * dth)
                         / np.pi)
            b[m] = float(np.sum(values * np.sin(m * upper_th) * dth)
                         / np.pi)
        return a, b

    def fourier_to_sympy(a, b, theta_sym, tol=1e-12):
        expr = sympy.Float(a[0])
        for m in range(1, len(a)):
            if abs(a[m]) > tol:
                expr = expr + a[m] * sympy.cos(m * theta_sym)
            if abs(b[m]) > tol:
                expr = expr + b[m] * sympy.sin(m * theta_sym)
        return expr

    def second_derivative_periodic(values):
        """∂²values/∂θ² via central FD on uniformly-spaced periodic
        samples (good enough for the regular annulus mesh)."""
        n = len(values)
        dth_avg = 2 * np.pi / n
        d2 = np.empty(n)
        for i in range(n):
            ip = (i + 1) % n
            im = (i - 1) % n
            d2[i] = (values[ip] - 2 * values[i] + values[im]) / dth_avg ** 2
        return d2

    def windowed_ks_squared(h_vals, half_window=4):
        """Local k_s² estimate via windowed regression
            k² ≈ -⟨h'', h⟩_w / ⟨h, h⟩_w
        where the window has ±half_window neighbours (periodic).
        Robust at zero-crossings of h: when h is nearly zero in a
        window, h'' is also nearly zero (for smooth h with definite
        wavelength), and the ratio remains well-defined.
        Returns k_s² in radians² (multiply by 1/r_o² for arclength)."""
        n = len(h_vals)
        h2 = second_derivative_periodic(h_vals)
        ks_sq_loc = np.empty(n)
        for i in range(n):
            num = 0.0
            den = 0.0
            for j in range(-half_window, half_window + 1):
                k = (i + j) % n
                num += -h2[k] * h_vals[k]
                den += h_vals[k] ** 2
            if den > 1e-30:
                ks_sq_loc[i] = num / den
            else:
                ks_sq_loc[i] = 1.0  # safe default
        return ks_sq_loc

    # Initial displacement
    upper_pos = mesh.X.coords[upper_idx]
    upper_r = np.sqrt(upper_pos[:, 0] ** 2 + upper_pos[:, 1] ** 2)
    upper_dr = upper_r - r_o
    A_mode.append(project_mode_amp(upper_dr))
    A_max.append(float(np.abs(upper_dr).max()))
    if verbose:
        print(f"  step 0: A_mode={A_mode[-1]:+.4e}  A_max={A_max[-1]:.4e}",
              flush=True)

    # Per-step state buffer (for BDF-2 needing h^{n-1})
    step_state = {}

    # Relaxation loop
    blow_up = False
    for step in range(n_steps):
        # Diffuse current radial velocity into the interior
        diffuser._reset()
        diffuser.add_essential_bc(
            sympy.Matrix([v.sym.dot(mesh.CoordinateSystem.unit_e_0)]),
            mesh.boundaries.Upper.name)
        diffuser.add_essential_bc(sympy.Matrix([0.0]),
                                  mesh.boundaries.Lower.name)
        diffuser.solve(zero_init_guess=False)

        if update == 'fe':
            displacement = delta_t.value * np.asarray(uw.function.evaluate(
                M.sym * mesh.CoordinateSystem.unit_e_0,
                mesh.X.coords)).reshape(-1, 2)
            mesh._deform_mesh(mesh.X.coords + displacement)

        elif update == 'etd':
            # Sample radial velocity at upper-boundary nodes from M
            # (the diffused field, smooth on the boundary).
            v_field = np.asarray(uw.function.evaluate(
                M.sym * mesh.CoordinateSystem.unit_e_0,
                mesh.X.coords)).reshape(-1, 2)
            v_radial = (v_field[:, 0] * mesh.X.coords[:, 0]
                        + v_field[:, 1] * mesh.X.coords[:, 1])
            # normalise: this gives v · r̂_now where |r_now| ~ r_o
            current_r = np.sqrt(mesh.X.coords[:, 0] ** 2
                                + mesh.X.coords[:, 1] ** 2)
            v_radial = v_radial / np.maximum(current_r, 1e-12)

            v_radial_upper = v_radial[upper_idx]
            upper_pos = mesh.X.coords[upper_idx]
            upper_r = np.sqrt(upper_pos[:, 0] ** 2
                              + upper_pos[:, 1] ** 2)
            upper_dr = upper_r - r_o

            # Project current displacement and dh/dt onto sin(mode·θ)
            A_now = project_mode_amp(upper_dr)
            dAdt = project_mode_amp(v_radial_upper)

            # Local τ from scalar mode: τ = -A / dA/dt (assuming
            # pure relaxation). Guard against zero-crossing of A —
            # the τ estimator becomes 0/0 and the FE fallback would
            # take a misleading constant-magnitude step. Freeze
            # instead (no surface motion) when |A| is below 0.1% of
            # the initial amplitude.
            dt_val = float(delta_t.value)
            A_threshold = 1e-3 * amp0
            if abs(A_now) < A_threshold:
                A_new = A_now  # freeze
            elif (-dAdt / A_now) > 0.0:
                gamma = -dAdt / A_now  # positive = relaxation rate
                alpha = float(np.exp(-dt_val * gamma))
                A_new = alpha * A_now
            else:
                # Sign mismatch (driving rather than relaxing). Use
                # FE step on the mode amplitude — accept that this
                # may be inaccurate but at least it isn't reset
                # by the freeze guard.
                A_new = A_now + dt_val * dAdt

            # Reconstruct the new boundary δr from the updated mode
            # amplitude. Use the diffuser to propagate the displacement
            # increment into the interior.
            scale = (A_new - A_now)  # δr increment as mode amplitude
            if abs(scale) < 1e-14:
                # No mesh motion needed; skip the diffuser solve.
                pass
            else:
                inc_fn = scale * sympy.sin(mode * th)
                diffuser._reset()
                diffuser.add_essential_bc(sympy.Matrix([inc_fn]),
                                          mesh.boundaries.Upper.name)
                diffuser.add_essential_bc(sympy.Matrix([0.0]),
                                          mesh.boundaries.Lower.name)
                diffuser.solve(zero_init_guess=False)
                inc_field = np.asarray(uw.function.evaluate(
                    M.sym * mesh.CoordinateSystem.unit_e_0,
                    mesh.X.coords)).reshape(-1, 2)
                mesh._deform_mesh(mesh.X.coords + inc_field)

        elif update == 'empE':
            # Empirical ETD: γ from per-node SPATIAL windowed
            # regression of u vs h at the current timestep.
            # Locally fit u = -γ·h + s over a small window of
            # neighbouring boundary nodes; γ is minus the slope.
            # No history required: the spatial samples themselves
            # provide the data points for the linear-response
            # regression.  Robust signal-to-noise as long as h
            # varies appreciably across the window.
            dt_val = float(delta_t.value)

            upper_pos = mesh.X.coords[upper_idx]
            upper_r = np.sqrt(upper_pos[:, 0] ** 2
                              + upper_pos[:, 1] ** 2)
            upper_dr = upper_r - r_o

            # Use M (diffused radial velocity) — matches curvS's
            # extraction so we can compare apples to apples.
            v_field_full = np.asarray(uw.function.evaluate(
                M.sym * mesh.CoordinateSystem.unit_e_0,
                mesh.X.coords)).reshape(-1, 2)
            v_radial_full = (
                v_field_full[:, 0] * mesh.X.coords[:, 0]
                + v_field_full[:, 1] * mesh.X.coords[:, 1])
            current_r_full = np.sqrt(
                mesh.X.coords[:, 0] ** 2
                + mesh.X.coords[:, 1] ** 2)
            v_radial_full = (v_radial_full
                             / np.maximum(current_r_full, 1e-12))
            u_n_now = v_radial_full[upper_idx]

            # Floor only to avoid division by zero / runaway from
            # numerical noise.  Real γ values for the test problems
            # are ~0.04 (mode-10) to ~0.5 (mode-1), so γ_min must
            # be much smaller than these.
            gamma_min = 1e-4

            # Per-node least-squares regression of u vs h in a
            # local window: u = a·h + b, γ = -a.
            n_pts = len(upper_dr)
            half_window = 4
            gamma = np.empty(n_pts)
            for i in range(n_pts):
                idx_w = [(i + j) % n_pts
                         for j in range(-half_window,
                                         half_window + 1)]
                h_w = upper_dr[idx_w]
                u_w = u_n_now[idx_w]
                h_mean = h_w.mean()
                u_mean = u_w.mean()
                hc = h_w - h_mean
                uc = u_w - u_mean
                den = float(np.sum(hc * hc))
                if den > 1e-30:
                    slope = float(np.sum(hc * uc)) / den
                else:
                    slope = -gamma_min
                gamma[i] = max(-slope, gamma_min)

            # Kinematic ETD update — same form as curvS, just with
            # empirical γ.
            alpha = np.exp(-dt_val * gamma)
            gd = dt_val * gamma
            phi1 = np.where(gd > 1e-6,
                            (1.0 - alpha) / np.maximum(gamma, 1e-12),
                            dt_val * (1.0 - 0.5 * gd))
            upper_dr_new = upper_dr + phi1 * u_n_now

            increment = upper_dr_new - upper_dr

            # Decompose & propagate (same as curv/curvS)
            n_modes = max(2, len(upper_th) // 3)
            a_coef, b_coef = fourier_decompose(increment, n_modes)
            inc_fn = fourier_to_sympy(a_coef, b_coef, th)
            diffuser._reset()
            diffuser.add_essential_bc(sympy.Matrix([inc_fn]),
                                      mesh.boundaries.Upper.name)
            diffuser.add_essential_bc(sympy.Matrix([0.0]),
                                      mesh.boundaries.Lower.name)
            diffuser.solve(zero_init_guess=False)
            inc_field = np.asarray(uw.function.evaluate(
                M.sym * mesh.CoordinateSystem.unit_e_0,
                mesh.X.coords)).reshape(-1, 2)
            mesh._deform_mesh(mesh.X.coords + inc_field)

        elif update in ('curv', 'bdf2', 'etd2', 'curvS'):
            # Curvature-derived local τ on the boundary, applied with
            # the chosen integrator: 'curv' = ETD-1 (exact along
            # frozen γ), 'etd2' = predictor-corrector ETD-2
            # (γ averaged across the step), 'bdf2' = backward
            # second-order, with γ from current curvature.
            # In all three: increment is Fourier-decomposed and
            # propagated into the interior via the diffuser.
            dt_val = float(delta_t.value)

            upper_pos = mesh.X.coords[upper_idx]
            upper_r = np.sqrt(upper_pos[:, 0] ** 2
                              + upper_pos[:, 1] ** 2)
            upper_dr = upper_r - r_o

            # Local k_s²(θ) via windowed regression (robust at
            # zero-crossings of δr); convert to arclength wavenumber
            # by dividing by r_o²
            def _gamma_from_dr(dr):
                ks_sq = (windowed_ks_squared(dr, half_window=4)
                         / r_o ** 2)
                ks_loc = np.sqrt(
                    np.maximum(np.abs(ks_sq), (1.0 / r_o) ** 2))
                return rho_g / (2.0 * eta_local * ks_loc)

            # Local η at upper boundary nodes
            if visc_contrast:
                eta_local = 1.0 / (1.0 + 19.0 *
                                   np.exp(-(upper_th ** 2) / 0.16))
            else:
                eta_local = np.ones_like(upper_th)

            rho_g = 1.0
            gamma = _gamma_from_dr(upper_dr)

            if update == 'curv':
                alpha = np.exp(-dt_val * gamma)
                upper_dr_new = alpha * upper_dr
            elif update == 'curvS':
                # Source-aware ETD-1: h^{n+1} = h^n + (1-α)/γ · u_n
                # where u_n is the current Stokes radial surface velocity
                # at upper boundary nodes. Reduces to FE for γΔt → 0,
                # and to steady-state h_eq = u_n/γ for γΔt → ∞.
                #
                # CRITICAL: mode-0 of u_n at the upper boundary should
                # be exactly zero by mass conservation in an
                # incompressible closed annulus.  In practice the
                # pressure-penalty regularisation injects a small DC
                # bias (~1e-4) that the kinematic update integrates
                # linearly into a uniform inward drift of the
                # boundary.  Subtract the spatial mean of u_n to
                # enforce the physical constraint.  For driven cases
                # where mode-0 is physically non-zero (e.g. closed
                # buoyancy), this filter would need to be skipped or
                # replaced with a known mode-0 target.
                v_field_full = np.asarray(uw.function.evaluate(
                    M.sym * mesh.CoordinateSystem.unit_e_0,
                    mesh.X.coords)).reshape(-1, 2)
                v_radial_full = (
                    v_field_full[:, 0] * mesh.X.coords[:, 0]
                    + v_field_full[:, 1] * mesh.X.coords[:, 1])
                current_r_full = np.sqrt(
                    mesh.X.coords[:, 0] ** 2
                    + mesh.X.coords[:, 1] ** 2)
                v_radial_full = (v_radial_full
                                 / np.maximum(current_r_full, 1e-12))
                u_n = v_radial_full[upper_idx]
                alpha = np.exp(-dt_val * gamma)
                # (1-α)/γ — careful at small γΔt where (1-α)/γ → Δt
                gd = dt_val * gamma
                phi1 = np.where(gd > 1e-6,
                                (1.0 - alpha) / np.maximum(gamma, 1e-12),
                                dt_val * (1.0 - 0.5 * gd))
                upper_dr_new = upper_dr + phi1 * u_n
            elif update == 'etd2':
                # Predictor-corrector ETD: predict with γ at current
                # state; recompute γ at predicted state; average and
                # apply exponential.
                alpha_pred = np.exp(-dt_val * gamma)
                upper_dr_pred = alpha_pred * upper_dr
                gamma_pred = _gamma_from_dr(upper_dr_pred)
                gamma_avg = 0.5 * (gamma + gamma_pred)
                alpha = np.exp(-dt_val * gamma_avg)
                upper_dr_new = alpha * upper_dr
            elif update == 'bdf2':
                # BDF-2 needs h^{n-1}. Use first-step BE bootstrap.
                upper_dr_prev = step_state.get('dr_prev')
                if upper_dr_prev is None:
                    # First step: backward Euler
                    upper_dr_new = upper_dr / (1.0 + dt_val * gamma)
                else:
                    # h^{n+1} = (4 h^n - h^{n-1}) / (3 + 2Δt·γ_n)
                    upper_dr_new = ((4.0 * upper_dr - upper_dr_prev)
                                    / (3.0 + 2.0 * dt_val * gamma))
                step_state['dr_prev'] = upper_dr.copy()
            else:
                raise ValueError(update)

            increment = upper_dr_new - upper_dr

            # Decompose the increment as a Fourier series on θ
            # (n_modes = up to half the boundary node count)
            n_modes = max(2, len(upper_th) // 3)
            a_coef, b_coef = fourier_decompose(increment, n_modes)
            inc_fn = fourier_to_sympy(a_coef, b_coef, th)

            diffuser._reset()
            diffuser.add_essential_bc(sympy.Matrix([inc_fn]),
                                      mesh.boundaries.Upper.name)
            diffuser.add_essential_bc(sympy.Matrix([0.0]),
                                      mesh.boundaries.Lower.name)
            diffuser.solve(zero_init_guess=False)
            inc_field = np.asarray(uw.function.evaluate(
                M.sym * mesh.CoordinateSystem.unit_e_0,
                mesh.X.coords)).reshape(-1, 2)
            mesh._deform_mesh(mesh.X.coords + inc_field)

        else:
            raise ValueError(update)

        # Re-solve Stokes on deformed mesh (FSSA term re-evaluates
        # automatically since it references v.sym).
        stokes.solve(zero_init_guess=False)

        # Record diagnostics
        upper_pos = mesh.X.coords[upper_idx]
        upper_r = np.sqrt(upper_pos[:, 0] ** 2 + upper_pos[:, 1] ** 2)
        upper_dr = upper_r - r_o
        A_mode.append(project_mode_amp(upper_dr))
        A_max.append(float(np.abs(upper_dr).max()))
        times.append(times[-1] + float(delta_t.value))

        if verbose and (step % max(1, n_steps // 20) == 0
                        or step == n_steps - 1):
            print(f"  step {step + 1}: t={times[-1]:.4e} "
                  f"A_mode={A_mode[-1]:+.4e}  A_max={A_max[-1]:.4e}",
                  flush=True)

        # Blow-up guard
        if A_max[-1] > 10 * amp0:
            print(f"  *** BLOW-UP at step {step + 1} ***", flush=True)
            blow_up = True
            break

    return {
        'label': label,
        'scheme': scheme,
        'dt': float(delta_t.value),
        'dt_factor': dt_factor,
        'res': res,
        'mode': mode,
        'amp0': amp0,
        'times': np.asarray(times, dtype=float),
        'A_mode': np.asarray(A_mode),
        'A_max': np.asarray(A_max),
        'blow_up': blow_up,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--res', type=int, default=20)
    parser.add_argument('--n-steps', type=int, default=40)
    parser.add_argument('--scheme', type=str, default='all',
                        choices=['all', 'fe-nofssa', 'fe-fssa',
                                 'etd-nofssa', 'etd-fssa',
                                 'curv-nofssa', 'curv-fssa',
                                 'curv-only',
                                 'bdf2-fssa', 'etd2-fssa',
                                 'order2', 'all2',
                                 'curvS-fssa', 'curvS-nofssa',
                                 'empE-fssa', 'empE-nofssa',
                                 'buoyancy-set',
                                 'compare-empE'])
    parser.add_argument('--dt-factor', type=float, default=1.0)
    parser.add_argument('--quick', action='store_true',
                        help='Reduce mesh+steps for smoke test')
    parser.add_argument('--ic', type=str, default='single',
                        choices=['single', 'multi'])
    parser.add_argument('--mode', type=int, default=10)
    parser.add_argument('--visc-contrast', action='store_true')
    parser.add_argument('--buoyancy', action='store_true',
                        help='Internal density anomaly drives surface'
                             ' (the forced free-surface case).')
    args = parser.parse_args()

    if args.quick:
        args.res = 12
        args.n_steps = 8

    os.makedirs(OUT_DIR, exist_ok=True)

    schemes = {
        'fe-nofssa':   (False, 'fe'),
        'fe-fssa':     (True,  'fe'),
        'etd-nofssa':  (False, 'etd'),
        'etd-fssa':    (True,  'etd'),
        'curv-nofssa': (False, 'curv'),
        'curv-fssa':   (True,  'curv'),
        'bdf2-fssa':   (True,  'bdf2'),
        'etd2-fssa':   (True,  'etd2'),
        'curvS-fssa':  (True,  'curvS'),
        'curvS-nofssa':(False, 'curvS'),
        'empE-fssa':   (True,  'empE'),
        'empE-nofssa': (False, 'empE'),
    }
    if args.scheme == 'all':
        # Original 6 (no order-2 yet)
        run_list = [schemes[k] for k in
                    ['fe-nofssa', 'fe-fssa', 'etd-nofssa', 'etd-fssa',
                     'curv-nofssa', 'curv-fssa']]
    elif args.scheme == 'all2':
        # All 8 + empE
        run_list = list(schemes.values())
    elif args.scheme == 'order2':
        # Only the 2nd-order curvature variants + curv-fssa baseline
        run_list = [schemes[k] for k in
                    ['curv-fssa', 'bdf2-fssa', 'etd2-fssa']]
    elif args.scheme == 'curv-only':
        run_list = [schemes['curv-nofssa'], schemes['curv-fssa']]
    elif args.scheme == 'buoyancy-set':
        # Schemes appropriate for the forced (buoyancy) test:
        # FE+FSSA, FE only, curv (homogeneous form — should miss
        # the forcing), curvS (source-aware ETD-1), bdf2/etd2
        # (homogeneous order-2 — should also miss), empE (empirical
        # γ from history).
        run_list = [schemes[k] for k in
                    ['fe-fssa', 'fe-nofssa',
                     'curv-fssa', 'curvS-fssa',
                     'bdf2-fssa', 'etd2-fssa',
                     'empE-fssa']]
    elif args.scheme == 'compare-empE':
        # Curvature-γ vs empirical-γ comparison
        run_list = [schemes[k] for k in
                    ['fe-fssa', 'curvS-fssa', 'empE-fssa']]
    else:
        run_list = [schemes[args.scheme]]

    results = []
    for s in run_list:
        try:
            r = run(s, args.dt_factor, args.n_steps, res=args.res,
                    mode=args.mode, ic=args.ic,
                    visc_contrast=args.visc_contrast,
                    buoyancy=args.buoyancy)
            results.append(r)
        except Exception as e:
            print(f"  *** EXCEPTION in {s}: {e!r} ***", flush=True)
            import traceback; traceback.print_exc()

    if not results:
        print("No successful runs.")
        return

    # Save raw results — tag with ic / visc-contrast / buoyancy
    if args.buoyancy:
        tag = f"dtf{args.dt_factor:.2f}_n{args.n_steps}_buoyancy"
    else:
        tag = f"dtf{args.dt_factor:.2f}_n{args.n_steps}_ic{args.ic}"
        if args.visc_contrast:
            tag = tag + "_V"
    out = os.path.join(OUT_DIR, f"phase_i2d_fs_etd_{tag}.npz")
    np.savez(out, **{
        f"{r['label']}_t":    r['times'] for r in results
    }, **{
        f"{r['label']}_A":    r['A_mode'] for r in results
    }, **{
        f"{r['label']}_Amax": r['A_max']  for r in results
    })
    print(f"\nWrote {out}", flush=True)

    # Console summary
    print("\nFinal-step summary:")
    print(f"  {'scheme':<40} {'A_mode_final':>14} {'A_max_final':>14} "
          f"{'blow_up':>8}")
    for r in results:
        print(f"  {r['label']:<40} {r['A_mode'][-1]:+14.4e} "
              f"{r['A_max'][-1]:14.4e} {str(r['blow_up']):>8}")


if __name__ == "__main__":
    main()
