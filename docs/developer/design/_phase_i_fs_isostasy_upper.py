"""Phase I-2D isostatic relaxation on the EXTERNAL UPPER free surface.

Upper-surface counterpart of `_phase_i_fs_etd_isostasy.py`. The free
surface is the *external* outer boundary, where a held-lid free-slip
stress is consistent and well-posed — so the stress-equilibrium
integrator `etd_topo` can measure the dynamic topography h_eq cleanly.

Validation (probe `_probe_heq_upper.py` + curvS reference): the held-lid
projected n·σ·n equals the kinematic equilibrium (≈0.0227) to ~1%, and is
penalty/γ-independent on the external boundary. (On the *internal* sticky-
air interface the held-lid stress can't be measured cleanly — the pure
penalty has no consistency term and only the penalty-dependent reaction
carries the load; that case is deferred.)

Mesh: plain annulus —
  - r_i = 0.5  (no-slip inner)
  - r_o = 1.0  (free upper surface, flat IC, to vacuum)
  - whole annulus heavy fluid (η=1, ρ=1)
  - buoyant blob (P0) at (0.7, 0), σ=0.08, mag 0.6 drives surface uplift
  - body force -(ρ - ρ_ref)·r̂ with ρ_ref the sharp step at r_o (removes
    the uniform hydrostatic part; only anomalies + surface crossings drive)

Schemes: fe, rk2, rk4, curvS, etd_topo (× FSSA on/off). etd_topo is the
stress-equilibrium integrator:
  1. held-lid free-slip (Nitsche) solve on the current mesh → project
     n·σ·n → h_eq = -σ_rr/(Δρg)  (infinite-time dynamic topography)
  2. free solve → u_n (kinematic rate)
  3. per-node γ = u_n/(h_eq - h); step h <- h_eq + (h - h_eq)e^{-γΔt}
"""

import os
import argparse

import numpy as np
import sympy

import underworld3 as uw

import nest_asyncio
nest_asyncio.apply()


OUT_DIR = "output"

# h_eq = ETD_TOPO_SIGN · (projected n·σ·n) / (Δρ g).  On the upper surface
# the held-lid constitutive σ_rr is NEGATIVE for an upward bulge (probe:
# -0.0227 for equilibrium +0.0227), so SIGN = -1.  Verified by the
# step-0 [etd_topo] debug print (h_eq_pole must be +, ≈ kinematic eq).
ETD_TOPO_SIGN = -1.0


def run(scheme, dt_factor, n_steps, res=20, blob_amp=0.6,
        nitsche_gamma=10.0, adaptive_dt=False, cap_gdt=None,
        cap_gamma=None, target_t=None, velocity_align=False,
        vel_smooth=None, topo_smooth=None, sl_correction=False,
        tangent_slip=False, use_mover=False, mover_iters=5,
        verbose=True):
    """One upper-surface isostatic-relaxation run.

    cap_gdt: explicit-scheme (fe/rk2/rk4) γΔt cap (curvS/etd_topo unaffected).
    target_t: stop when t ≥ target_t (with n_steps as a max-step cap).
    """
    use_fssa, update = scheme
    label = (f"FSSA={int(use_fssa)}_UPD={update}_dtf{dt_factor:.2f}"
             f"_upper")
    if verbose:
        print(f"\n=== {label} ===", flush=True)

    r_i = 0.5
    r_o = 1.0
    cellsize = 1.0 / res

    # Buoyant blob (Eulerian, lab-frame), matches _phase_i_fs_etd_annulus
    x_b, y_b = 0.7, 0.0
    sigma_b = 0.08

    mesh = uw.meshing.Annulus(
        radiusOuter=r_o, radiusInner=r_i, cellSize=cellsize, qdegree=3)

    r, th = mesh.CoordinateSystem.R
    unit_r = mesh.CoordinateSystem.unit_e_0

    if verbose:
        print(f"  mesh: {mesh.X.coords.shape[0]} nodes, "
              f"{mesh._centroids.shape[0]} cells", flush=True)

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
    rho_var = uw.discretisation.MeshVariable(
        f"rho_{label}", mesh, vtype=uw.VarType.SCALAR, degree=0,
        continuous=False, varsymbol=r"\rho")
    # held-lid stress projection target (etd_topo only)
    topo = uw.discretisation.MeshVariable(
        f"topo_{label}", mesh, vtype=uw.VarType.SCALAR, degree=1,
        continuous=True, varsymbol=r"h_{eq}")

    # Buoyant blob (per-element P0).  ρ_ref sharp step at r_o; the uniform
    # hydrostatic part cancels so only the blob + surface crossings drive.
    cen = mesh._centroids
    blob = blob_amp * np.exp(-((cen[:, 0] - x_b) ** 2
                               + (cen[:, 1] - y_b) ** 2)
                             / (2.0 * sigma_b ** 2))
    rho_var.data[:, 0] = 1.0 - blob
    r_node = sympy.sqrt(mesh.X[0] ** 2 + mesh.X[1] ** 2)
    rho_ref = sympy.Piecewise((1.0, r_node < r_o), (0.0, True))
    anomaly = rho_var.sym[0] - rho_ref
    bodyforce = -anomaly * unit_r

    # Eulerian blob-only driving for the held-lid (dynamic-topography)
    # solve. The held-lid measures the topography the DRIVING wants, so it
    # must EXCLUDE the Lagrangian surface-crossing (ρ_ref) restoring term —
    # on a deformed mesh that term injects a spurious restoring load that
    # flips the held-lid stress negative. At flat IC the full anomaly
    # reduces to exactly this blob-only force (which is why the probe and
    # step 0 agree with the equilibrium).
    blob_E = blob_amp * sympy.exp(
        -((mesh.X[0] - x_b) ** 2 + (mesh.X[1] - y_b) ** 2)
        / (2.0 * sigma_b ** 2))
    bodyforce_held = blob_E * unit_r

    # Upper-boundary node identification (flat IC)
    X_initial = mesh.X.coords.copy()
    R_initial = np.sqrt(X_initial[:, 0] ** 2 + X_initial[:, 1] ** 2)
    THETA_initial = np.arctan2(X_initial[:, 1], X_initial[:, 0])
    is_upper = R_initial > (r_o - 0.5 * cellsize / r_o)
    upper_idx = np.where(is_upper)[0]
    upper_idx = upper_idx[np.argsort(THETA_initial[upper_idx])]
    upper_th = THETA_initial[upper_idx]
    pole_idx = int(np.argmin(np.abs(upper_th)))      # θ≈0, above blob
    if verbose:
        print(f"  {len(upper_idx)} nodes on upper (free) boundary",
              flush=True)

    # Diffuser (Poisson) to propagate the surface increment inward
    diffuser = uw.systems.Poisson(mesh, Vr)
    diffuser.constitutive_model = uw.constitutive_models.DiffusionModel
    diffuser.constitutive_model.Parameters.diffusivity = 1.0
    diffuser.add_essential_bc(sympy.Matrix([0.0]), mesh.boundaries.Upper.name)
    diffuser.add_essential_bc(sympy.Matrix([0.0]), mesh.boundaries.Lower.name)
    diffuser.tolerance = 1.0e-3
    diffuser.solve()

    # Stokes (FREE upper surface) — gives the kinematic velocity u_n
    stokes = uw.systems.Stokes(mesh, velocityField=v, pressureField=p)
    stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
    stokes.constitutive_model.Parameters.shear_viscosity_0 = 1.0
    stokes.penalty = 1.0
    stokes.bodyforce = bodyforce
    stokes.tolerance = 1.0e-6
    stokes.add_essential_bc((0.0, 0.0), mesh.boundaries.Lower.name)
    stokes.solve()

    # Held-lid Stokes (free-slip Upper via Nitsche) — gives the dynamic
    # topography via projected n·σ·n.  Built only for etd_topo.
    stokes_held = None
    if update == 'etd_topo':
        v_h = uw.discretisation.MeshVariable(
            f"Vh_{label}", mesh, vtype=uw.VarType.VECTOR, degree=2,
            continuous=True)
        p_h = uw.discretisation.MeshVariable(
            f"Ph_{label}", mesh, vtype=uw.VarType.SCALAR, degree=1,
            continuous=True)
        stokes_held = uw.systems.Stokes(mesh, velocityField=v_h,
                                        pressureField=p_h)
        stokes_held.constitutive_model = uw.constitutive_models.ViscousFlowModel
        stokes_held.constitutive_model.Parameters.shear_viscosity_0 = 1.0
        stokes_held.penalty = 1.0
        stokes_held.bodyforce = bodyforce_held
        stokes_held.tolerance = 1.0e-6
        stokes_held.add_essential_bc((0.0, 0.0), mesh.boundaries.Lower.name)
        stokes_held.add_nitsche_bc(mesh.boundaries.Upper.name,
                                   gamma=nitsche_gamma)
        # Optional: estimate u_n from a P1-projected (optionally screened-
        # Poisson-smoothed) copy of the P2 velocity, to suppress the small-
        # scale modes the P2 field carries into the γ estimator / topography.
        v_est = v
        vel_proj = None
        if vel_smooth is not None:
            v_p1 = uw.discretisation.MeshVariable(
                f"Vp1_{label}", mesh, vtype=uw.VarType.VECTOR, degree=1,
                continuous=True, varsymbol=r"\hat{v}")
            vel_proj = uw.systems.Vector_Projection(mesh, v_p1)
            vel_proj.uw_function = v.sym
            vel_proj.smoothing_length = vel_smooth     # 0 = P1 only
            v_est = v_p1
        topo_proj = uw.systems.Projection(mesh, topo)
        if velocity_align:
            # Project the held stress onto the FREE-velocity direction û
            # (the direction the surface actually moves) instead of the
            # geometric radial normal. (Empirically worse than radial — the
            # off-pole velocity is dominated by tangential flow.)
            eps = 1.0e-6
            uhat = v_est.sym / sympy.sqrt(v_est.sym.dot(v_est.sym) + eps ** 2)
            topo_proj.uw_function = (uhat * stokes_held.stress * uhat.T)[0, 0]
        else:
            topo_proj.uw_function = (unit_r * stokes_held.stress * unit_r.T)[0, 0]
        topo_proj.smoothing = 0.0
        if topo_smooth is not None:
            # Screened-Poisson smoothing of the held-stress projection. Stress
            # ~ ∇v amplifies high-k modes, so this is where the small-scale
            # topography fluctuations originate (more than the velocity).
            topo_proj.smoothing_length = topo_smooth
        stokes_held.solve()        # initialise held-solver state

    delta_t = uw.function.expression(
        R"\delta t", dt_factor * stokes.estimate_dt(), "Timestep")
    if use_fssa:
        Gamma = mesh.Gamma / sympy.sqrt(mesh.Gamma.dot(mesh.Gamma))
        FSSA_traction = (delta_t * Gamma.dot(v.sym) * Gamma / 2.0)
        stokes.add_natural_bc(FSSA_traction, mesh.boundaries.Upper.name)
        stokes.solve()

    if verbose:
        print(f"  dt = {float(delta_t.value):.4e}", flush=True)

    # Diagnostics buffers
    times = [0.0]
    h_max = []
    h_at_pole = []

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
        dth = _trap_weights(upper_th)
        a = np.zeros(n_modes + 1)
        b = np.zeros(n_modes + 1)
        a[0] = float(np.sum(values * dth) / (2 * np.pi))
        for m in range(1, n_modes + 1):
            a[m] = float(np.sum(values * np.cos(m * upper_th) * dth) / np.pi)
            b[m] = float(np.sum(values * np.sin(m * upper_th) * dth) / np.pi)
        return a, b

    def fourier_to_sympy(a, b, theta_sym, tol=1e-12):
        expr = sympy.Float(a[0])
        for m in range(1, len(a)):
            if abs(a[m]) > tol:
                expr = expr + a[m] * sympy.cos(m * theta_sym)
            if abs(b[m]) > tol:
                expr = expr + b[m] * sympy.sin(m * theta_sym)
        return expr

    def surface_dr():
        pos = mesh.X.coords[upper_idx]
        return np.sqrt(pos[:, 0] ** 2 + pos[:, 1] ** 2) - r_o

    def surface_area():
        # enclosed area A = ½∮ r² dθ ; ΔA/A measures volume conservation
        r = r_o + surface_dr()
        return 0.5 * float(np.sum(r ** 2 * _trap_weights(upper_th)))

    def diffuse_increment(increment):
        """Propagate the per-node surface radial increment into the mesh.

        Two mechanisms:
          - use_mover=False (legacy): Poisson DIFFUSER — Fourier-filter the
            increment, set it as the Upper BC, solve ∇²Vr=0, deform by the
            smooth radial field. Moves ALL interior nodes every step.
          - use_mover=True (MMPDE): move ONLY the surface ring radially by the
            increment, then relax the interior with smooth_mesh_interior
            (surface + inner pinned). The mover does the accommodation the
            diffuser did, but in 2D and decoupled from the surface step.
        """
        if use_mover:
            new_coords = mesh.X.coords.copy()
            pos = new_coords[upper_idx]
            r = np.sqrt(pos[:, 0] ** 2 + pos[:, 1] ** 2)
            rhat = pos / r[:, None]
            new_coords[upper_idx] = pos + increment[:, None] * rhat
            mesh._deform_mesh(new_coords)
            uw.meshing.smooth_mesh_interior(
                mesh, pinned_labels=[mesh.boundaries.Upper.name,
                                     mesh.boundaries.Lower.name],
                method='spring', n_iters=mover_iters, alpha=0.5)
            return
        # --- legacy Poisson diffuser ---
        n_modes = max(2, len(upper_th) // 3)
        a_c, b_c = fourier_decompose(increment, n_modes)
        inc_fn = fourier_to_sympy(a_c, b_c, th)
        diffuser._reset()
        diffuser.add_essential_bc(sympy.Matrix([inc_fn]),
                                  mesh.boundaries.Upper.name)
        diffuser.add_essential_bc(sympy.Matrix([0.0]),
                                  mesh.boundaries.Lower.name)
        diffuser.solve(zero_init_guess=False)
        inc_field = np.asarray(uw.function.evaluate(
            Vr.sym * unit_r, mesh.X.coords)).reshape(-1, 2)
        mesh._deform_mesh(mesh.X.coords + inc_field)

    def sample_un_at_upper():
        return np.asarray(uw.function.evaluate(
            v.sym.dot(unit_r), mesh.X.coords[upper_idx])).flatten()

    upper_dr = surface_dr()
    h_max.append(float(np.abs(upper_dr).max()))
    h_at_pole.append(float(upper_dr[pole_idx]))
    dr_snaps = [upper_dr.copy()]          # surface profile h(θ) per step
    areas = [surface_area()]              # enclosed area (volume conservation)
    if verbose:
        print(f"  step 0: h_max={h_max[-1]:.4e} "
              f"h_pole={h_at_pole[-1]:+.4e}", flush=True)

    blow_up = False
    dt_history = []
    is_explicit = update in ('fe', 'rk2', 'rk4')
    for step in range(n_steps):
        if target_t is not None and times[-1] >= target_t:
            break

        if adaptive_dt:
            dt_bulk = dt_factor * stokes.estimate_dt()
            if cap_gdt is not None and is_explicit:
                if cap_gamma is not None:
                    gamma_max = cap_gamma
                else:
                    dr_now = surface_dr()
                    ks_sq_now = (windowed_ks_squared(dr_now, 4) / r_o ** 2)
                    ks_now = np.sqrt(np.maximum(np.abs(ks_sq_now),
                                                (1.0 / r_o) ** 2))
                    gamma_max = float(np.max(1.0 / (2.0 * 1.0 * ks_now)))
                dt_eff = min(dt_bulk, cap_gdt / max(gamma_max, 1e-12))
            else:
                dt_eff = dt_bulk
            delta_t.sym = dt_eff
        dt_history.append(float(delta_t.value))
        dt_val = float(delta_t.value)

        if update == 'fe':
            # Diffuse the current radial velocity, then FE-advect the mesh.
            diffuser._reset()
            diffuser.add_essential_bc(
                sympy.Matrix([v.sym.dot(unit_r)]),
                mesh.boundaries.Upper.name)
            diffuser.add_essential_bc(sympy.Matrix([0.0]),
                                      mesh.boundaries.Lower.name)
            diffuser.solve(zero_init_guess=False)
            disp = dt_val * np.asarray(uw.function.evaluate(
                Vr.sym * unit_r, mesh.X.coords)).reshape(-1, 2)
            mesh._deform_mesh(mesh.X.coords + disp)

        elif update == 'curvS':
            upper_dr = surface_dr()
            ks_sq = windowed_ks_squared(upper_dr, 4) / r_o ** 2
            ks = np.sqrt(np.maximum(np.abs(ks_sq), (1.0 / r_o) ** 2))
            gamma = 1.0 / (2.0 * 1.0 * ks)
            u_n = sample_un_at_upper()
            gd = dt_val * gamma
            phi1 = np.where(gd > 1e-6,
                            (1.0 - np.exp(-gd)) / np.maximum(gamma, 1e-12),
                            dt_val * (1.0 - 0.5 * gd))
            diffuse_increment(phi1 * u_n)

        elif update == 'etd_topo':
            if tangent_slip:
                # Nodes may have slid tangentially last step — re-sort the
                # surface index by current angle so the θ-grid stays ordered.
                _c = mesh.X.coords[upper_idx]
                _o = np.argsort(np.arctan2(_c[:, 1], _c[:, 0]))
                upper_idx = upper_idx[_o]
                _cc = mesh.X.coords[upper_idx]
                upper_th = np.arctan2(_cc[:, 1], _cc[:, 0])
                pole_idx = int(np.argmin(np.abs(upper_th)))
            # 1. h_eq: per-step held-lid dynamic topography on the CURRENT
            #    (deformed) surface. Two essentials:
            #    - blob-only DRIVING body force (set on stokes_held): on a
            #      deformed mesh the topography is geometric, so a ρ_ref load
            #      shell would DOUBLE-COUNT the load.
            #    - MEAN-RELATIVE σ_nn: the held free-slip solve (no-slip inner
            #      + free-slip outer) has a pressure null space, so the
            #      absolute σ_nn carries an arbitrary constant. Subtracting the
            #      surface mean removes it and is the correct isostatic datum.
            stokes_held.solve(zero_init_guess=False)
            if vel_proj is not None:
                vel_proj.solve()           # smoothed P1 velocity for u_n
            topo_proj.solve()
            sig_nn = np.asarray(uw.function.evaluate(
                topo.sym[0], mesh.X.coords[upper_idx])).flatten()
            h_eq = -(sig_nn - sig_nn.mean())              # Δρ g = 1
            upper_dr = surface_dr()
            shape = upper_dr - upper_dr.mean()
            u_n = np.asarray(uw.function.evaluate(
                v_est.sym.dot(unit_r), mesh.X.coords[upper_idx])).flatten()
            # 2. SL correction (radial nodes don't follow tangential flow, so
            #    the lateral transport -u_t·∇_s h must be added). Trace the
            #    current shape back along u_t to its departure angle. h_eq is
            #    predicted RADIALLY (lateral stress adds no vertical force);
            #    only the EVOLUTION needs the lateral velocity.
            if sl_correction:
                u_t = np.asarray(uw.function.evaluate(
                    v_est.sym.dot(mesh.CoordinateSystem.unit_e_1),
                    mesh.X.coords[upper_idx])).flatten()
                r_surf = r_o + upper_dr
                theta_back = upper_th - dt_val * u_t / r_surf
                h_back = np.interp(np.mod(theta_back + np.pi, 2 * np.pi) - np.pi,
                                   upper_th, shape, period=2 * np.pi)
            else:
                h_back = shape
            # 3. per-node γ (mean-relative velocity), relax the ADVECTED shape
            #    toward h_eq, then add the lateral-transport increment.
            disp = h_eq - h_back
            u_n_rel = u_n - u_n.mean()
            with np.errstate(divide='ignore', invalid='ignore'):
                gamma = np.where(np.abs(disp) > 1e-12, u_n_rel / disp, 0.0)
            gamma = np.maximum(gamma, 0.0)
            if verbose:
                print(f"    [etd_topo] h_eq_pole={h_eq[pole_idx]:+.4e} "
                      f"shape_pole={shape[pole_idx]:+.4e} "
                      f"u_n_pole={u_n[pole_idx]:+.4e} "
                      f"γ_pole={gamma[pole_idx]:.4e}", flush=True)
            relax_inc = disp * (1.0 - np.exp(-dt_val * gamma))
            increment = (h_back - shape) + relax_inc   # advect + relax
            diffuse_increment(increment)               # radial topography

            if tangent_slip:
                # Lateral transport WITHOUT an SL trace-back: let the surface
                # nodes slide tangentially with the flow via the tangent-slip
                # mechanism (FREE label → slide, no restore to the circle).
                # project_to_slip_surface keeps the tangential part of the
                # displacement and drops the radial part (so the topography
                # just set isn't disturbed); the nodes carry their height
                # around the annulus, so the SL correction is unnecessary.
                vx = np.asarray(uw.function.evaluate(
                    v_est.sym[0], mesh.X.coords[upper_idx])).flatten()
                vy = np.asarray(uw.function.evaluate(
                    v_est.sym[1], mesh.X.coords[upper_idx])).flatten()
                Y = mesh.X.coords.copy()
                Y[upper_idx, 0] += dt_val * vx
                Y[upper_idx, 1] += dt_val * vy
                Y = mesh.project_to_slip_surface(
                    Y, slip_spec={mesh.boundaries.Upper.name: False},
                    reference_coords=mesh.X.coords)
                mesh._deform_mesh(Y)

        elif update == 'rk2':
            saved_X = mesh.X.coords.copy()
            k1 = sample_un_at_upper()
            diffuse_increment((dt_val * 0.5) * k1)
            stokes.solve(zero_init_guess=False)
            k2 = sample_un_at_upper()
            mesh._deform_mesh(saved_X)
            diffuse_increment(dt_val * k2)

        elif update == 'rk4':
            saved_X = mesh.X.coords.copy()

            def _set_trial(disp_at_upper):
                mesh._deform_mesh(saved_X)
                diffuse_increment(disp_at_upper)

            k1 = sample_un_at_upper()
            _set_trial((dt_val * 0.5) * k1)
            stokes.solve(zero_init_guess=False)
            k2 = sample_un_at_upper()
            _set_trial((dt_val * 0.5) * k2)
            stokes.solve(zero_init_guess=False)
            k3 = sample_un_at_upper()
            _set_trial(dt_val * k3)
            stokes.solve(zero_init_guess=False)
            k4 = sample_un_at_upper()
            _set_trial((dt_val / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4))

        else:
            raise ValueError(update)

        stokes.solve(zero_init_guess=False)

        upper_dr = surface_dr()
        h_max.append(float(np.abs(upper_dr).max()))
        h_at_pole.append(float(upper_dr[pole_idx]))
        dr_snaps.append(upper_dr.copy())
        areas.append(surface_area())
        times.append(times[-1] + dt_val)

        if verbose and (step % max(1, n_steps // 10) == 0
                        or step == n_steps - 1):
            extra = (f"  Δt={dt_history[-1]:.3e}" if adaptive_dt else "")
            print(f"  step {step + 1}: t={times[-1]:.4e} "
                  f"h_max={h_max[-1]:.4e} "
                  f"h_pole={h_at_pole[-1]:+.4e}{extra}", flush=True)

        if h_max[-1] > 0.5:
            print(f"  *** BLOW-UP at step {step + 1} ***", flush=True)
            blow_up = True
            break

    return {
        'label': label,
        'scheme': scheme,
        'dt': float(delta_t.value),
        'times': np.asarray(times, dtype=float),
        'dt_history': np.asarray(dt_history),
        'h_max': np.asarray(h_max),
        'h_pole': np.asarray(h_at_pole),
        'blow_up': blow_up,
        'final_dr': surface_dr(),
        'final_th': upper_th,
        'dr_snaps': np.asarray(dr_snaps),
        'areas': np.asarray(areas),
        'dA_rel': (areas[-1] - areas[0]) / areas[0],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--res', type=int, default=20)
    parser.add_argument('--n-steps', type=int, default=32)
    parser.add_argument('--scheme', type=str, default='etd-vs-rk4',
                        choices=['fe-fssa', 'fe-nofssa',
                                 'curvS-fssa', 'curvS-nofssa',
                                 'rk2-nofssa', 'rk4-nofssa',
                                 'rk2-fssa', 'rk4-fssa',
                                 'etd_topo-fssa', 'etd_topo-nofssa',
                                 'etd-vs-rk4', 'integrators'])
    parser.add_argument('--dt-factor', type=float, default=1.0)
    parser.add_argument('--blob-amp', type=float, default=0.6)
    parser.add_argument('--nitsche-gamma', type=float, default=10.0)
    parser.add_argument('--adaptive-dt', action='store_true')
    parser.add_argument('--cap-gdt', action='store_true')
    parser.add_argument('--cap-gamma', type=float, default=None)
    parser.add_argument('--target-t', type=float, default=None)
    parser.add_argument('--velocity-align', action='store_true',
                        help="etd_topo: project held stress onto the free-"
                        "velocity direction û instead of radial r̂.")
    parser.add_argument('--vel-smooth', type=float, default=None,
                        help="etd_topo: estimate u_n from a P1-projected "
                        "velocity; value is the screened-Poisson smoothing "
                        "length (0 = P1 projection only, no extra smoothing).")
    parser.add_argument('--topo-smooth', type=float, default=None,
                        help="etd_topo: screened-Poisson smoothing length on "
                        "the held-stress (h_eq) projection — where high-k "
                        "topography noise originates (stress ~ ∇v).")
    parser.add_argument('--sl', action='store_true',
                        help="etd_topo: semi-Lagrange trace-back along u_t to "
                        "carry lateral transport of the topography (needed "
                        "with radial node motion).")
    parser.add_argument('--tangent-slip', action='store_true',
                        help="etd_topo: let surface nodes slide tangentially "
                        "with the flow (tangent-slip mechanism) — lateral "
                        "transport without an SL trace-back.")
    parser.add_argument('--mover', action='store_true',
                        help="Replace the Poisson diffuser with the MMPDE "
                        "mover (smooth_mesh_interior) for interior "
                        "accommodation — moves only the surface ring per step.")
    parser.add_argument('--mover-iters', type=int, default=5,
                        help="smooth_mesh_interior sweeps per step (--mover).")
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)

    schemes = {
        'fe-fssa':         (True,  'fe'),
        'fe-nofssa':       (False, 'fe'),
        'curvS-fssa':      (True,  'curvS'),
        'curvS-nofssa':    (False, 'curvS'),
        'rk2-fssa':        (True,  'rk2'),
        'rk2-nofssa':      (False, 'rk2'),
        'rk4-fssa':        (True,  'rk4'),
        'rk4-nofssa':      (False, 'rk4'),
        'etd_topo-fssa':   (True,  'etd_topo'),
        'etd_topo-nofssa': (False, 'etd_topo'),
    }
    if args.scheme == 'integrators':
        run_list = [schemes[k] for k in
                    ('fe-nofssa', 'rk2-nofssa', 'rk4-nofssa',
                     'curvS-fssa', 'etd_topo-nofssa')]
    elif args.scheme == 'etd-vs-rk4':
        run_list = [schemes[k] for k in
                    ('rk4-nofssa', 'curvS-fssa',
                     'etd_topo-nofssa', 'etd_topo-fssa')]
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
                    nitsche_gamma=args.nitsche_gamma,
                    adaptive_dt=args.adaptive_dt, cap_gdt=cap,
                    cap_gamma=args.cap_gamma, target_t=args.target_t,
                    velocity_align=args.velocity_align,
                    vel_smooth=args.vel_smooth, topo_smooth=args.topo_smooth,
                    sl_correction=args.sl,
                    tangent_slip=args.tangent_slip,
                    use_mover=args.mover, mover_iters=args.mover_iters)
            results.append(r)
        except Exception as e:
            print(f"  *** EXCEPTION in {s}: {e!r} ***", flush=True)
            import traceback
            traceback.print_exc()

    if not results:
        return

    adapt_tag = "_adt" if args.adaptive_dt else ""
    vel_tag = "_vel" if args.velocity_align else ""
    sm_tag = f"_sm{args.vel_smooth:g}" if args.vel_smooth is not None else ""
    ts_tag = f"_ts{args.topo_smooth:g}" if args.topo_smooth is not None else ""
    sl_tag = "_sl" if args.sl else ("_tslip" if args.tangent_slip else "")
    mv_tag = "_mover" if args.mover else ""
    tag = (f"dtf{args.dt_factor:.2f}_n{args.n_steps}_upper_res{args.res}"
           f"{adapt_tag}{vel_tag}{sm_tag}{ts_tag}{sl_tag}{mv_tag}")
    out = os.path.join(OUT_DIR, f"phase_i2d_fs_upper_{tag}.npz")
    np.savez(out,
             **{f"{r['label']}_t": r['times'] for r in results},
             **{f"{r['label']}_hmax": r['h_max'] for r in results},
             **{f"{r['label']}_hpole": r['h_pole'] for r in results},
             **{f"{r['label']}_finalDr": r['final_dr'] for r in results},
             **{f"{r['label']}_finalTh": r['final_th'] for r in results},
             **{f"{r['label']}_drsnaps": r['dr_snaps'] for r in results},
             **{f"{r['label']}_areas": r['areas'] for r in results},
             **{f"{r['label']}_dthistory": r['dt_history'] for r in results})
    print(f"\nWrote {out}", flush=True)

    print("\nFinal-step summary:")
    for r in results:
        print(f"  {r['label']:<48} h_max={r['h_max'][-1]:.4e} "
              f"h_pole={r['h_pole'][-1]:+.4e} dA/A={r['dA_rel']:+.3e} "
              f"blow_up={r['blow_up']}", flush=True)


if __name__ == "__main__":
    main()
