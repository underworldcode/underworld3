"""Phase I-2D internal-boundary variant: free-surface relaxation with
two layers.

Structure adapted from `docs/examples/free_surface/advanced/AnnulusND_FS-OuterSphere.py`.

Three radii:
  - r_inner = 0.5  (inner mesh boundary, no-slip)
  - r_o     = 1.0  (internal boundary — the "free surface")
  - r_outer = 1.5  (outer mesh boundary, no-slip)

The mesh fills annular space between r_inner and r_outer.  A discontinuous
P0 layer indicator M is set to 1 on elements initially below r_o (heavy
fluid) and 0 on elements above r_o (sticky air).  Body force = -r̂·M
(only the heavy layer feels gravity).  Viscosity = 0.01 + M (η ≈ 1
in fluid, 0.01 in air).

Because the mesh contains BOTH heavy and light layers, the
mean-density-subtracted body force `-(ρ - ρ_ref)·r̂` works correctly:
ρ_ref(r) = M_initial(r) (1 inside r_o, 0 outside).  The anomaly is
non-zero ONLY at elements that have moved across r_o due to mesh
deformation:
  - Heavy element above r_o (initially M=1, now in the air zone):
    anomaly = +1, force inward → restoring
  - Air element below r_o (initially M=0, now in the fluid zone):
    anomaly = -1, force outward → restoring
Both directions are restoring symmetrically.

Schemes compared:
  - FE+FSSA: forward Euler on the kinematic update + FSSA Robin BC
  - kinematic ETD with curvature-derived γ: bounded saturation
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


def run(scheme, dt_factor, n_steps, res=20, mode=10, amp0=0.05,
        verbose=True):
    """One internal-boundary free-surface run."""
    use_fssa, update = scheme
    label = (f"FSSA={int(use_fssa)}_UPD={update}_dtf{dt_factor:.2f}"
             f"_internal")
    if verbose:
        print(f"\n=== {label} ===", flush=True)

    r_inner = 0.5
    r_o = 1.0  # Internal-boundary radius (rest position of "free surface")
    r_outer = 1.5
    cellsize = 1.0 / res

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
    # Layer indicator: P0 discontinuous, 1 in heavy layer, 0 in air
    M = uw.discretisation.MeshVariable(
        f"M_{label}", mesh, vtype=uw.VarType.SCALAR, degree=0,
        continuous=False, varsymbol=r"\rho")

    # Initial layer indicator: 1 inside r_o (heavy), 0 outside (air)
    layer_fn = sympy.Piecewise((1.0, r <= r_o), (0.0, True))
    M.data[:, 0] = np.asarray(uw.function.evaluate(
        layer_fn, M.coords)).flatten()

    # Internal boundary node identification (after initial perturbation)
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

    # Apply the initial perturbation amp0·sin(mode·θ) to the
    # internal boundary, smooth into the interior with the diffuser.
    deform_fn = (r / r_o) * sympy.sin(mode * th) * amp0
    diffuser.add_essential_bc(sympy.Matrix([0.0]),
                              mesh.boundaries.Upper.name)
    diffuser.add_essential_bc(sympy.Matrix([deform_fn]),
                              mesh.boundaries.Internal.name)
    diffuser.add_essential_bc(sympy.Matrix([0.0]),
                              mesh.boundaries.Lower.name)
    diffuser.tolerance = 1.0e-3
    diffuser.solve()

    displacement = np.asarray(uw.function.evaluate(
        Vr.sym * mesh.CoordinateSystem.unit_e_0,
        mesh.X.coords)).reshape(-1, 2)
    mesh._deform_mesh(mesh.X.coords + displacement)

    # Stokes setup
    stokes = uw.systems.Stokes(mesh, velocityField=v, pressureField=p)
    stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
    # Viscosity contrast 10× (was 100× = 0.01 + M.sym[0]): air η=0.1,
    # heavy fluid η=1.0.  Test whether the residual drift floor is
    # from the layer-interface velocity gradient.
    stokes.constitutive_model.Parameters.shear_viscosity_0 = 0.1 + 0.9 * M.sym[0]
    stokes.penalty = 0.0
    # Body force: -r̂ · (M - M_ref(r)).  M is the per-element
    # discontinuous P0 density (rides with the deforming mesh).
    # M_ref is the rest-configuration angular mean: 1 inside r_o
    # (heavy layer), 0 outside (air).  Force is non-zero only at
    # elements that have moved across r_o due to deformation:
    #   - heavy-element above r_o: anomaly = +1, force inward
    #   - air-element below r_o:  anomaly = -1, force outward
    # Both directions restore symmetrically.
    # M_ref(r): sharp Piecewise (smooth tanh tested and was worse).
    M_ref = sympy.Piecewise((1.0, r <= r_o), (0.0, True))
    stokes.bodyforce = (
        -(M.sym[0] - M_ref) * mesh.CoordinateSystem.unit_e_0
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
    A_mode = []
    A_max = []

    def _trap_weights(th_w):
        n = len(th_w)
        dth = np.empty(n)
        dth[1:-1] = 0.5 * (th_w[2:] - th_w[:-2])
        dth[0] = 0.5 * (th_w[1] - (th_w[-1] - 2 * np.pi))
        dth[-1] = 0.5 * ((th_w[0] + 2 * np.pi) - th_w[-2])
        return dth

    def project_mode_amp(boundary_disp, m=mode):
        s = np.sin(m * internal_th)
        dth = _trap_weights(internal_th)
        return float(np.sum(boundary_disp * s * dth) / np.pi)

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

    # Initial diagnostics
    upper_pos = mesh.X.coords[internal_idx]
    upper_r = np.sqrt(upper_pos[:, 0] ** 2 + upper_pos[:, 1] ** 2)
    upper_dr = upper_r - r_o
    A_mode.append(project_mode_amp(upper_dr))
    A_max.append(float(np.abs(upper_dr).max()))
    if verbose:
        print(f"  step 0: A_mode={A_mode[-1]:+.4e} "
              f"A_max={A_max[-1]:.4e}", flush=True)

    # Relaxation loop
    blow_up = False
    for step in range(n_steps):
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
            # Kinematic ETD: γ from local curvature, update with
            # the saturated factor (1-α)/γ · u_n.
            dt_val = float(delta_t.value)
            upper_pos = mesh.X.coords[internal_idx]
            upper_r = np.sqrt(upper_pos[:, 0] ** 2
                              + upper_pos[:, 1] ** 2)
            upper_dr = upper_r - r_o
            ks_sq = windowed_ks_squared(upper_dr, half_window=4) / r_o ** 2
            ks = np.sqrt(np.maximum(np.abs(ks_sq), (1.0 / r_o) ** 2))
            # γ = ρg/(2η|k|) — η = 1 in heavy layer, 0.01 in air;
            # at internal boundary the relevant η is the average.
            eta_eff = 0.5  # rough average — needed to scale saturation
            gamma = 1.0 / (2.0 * eta_eff * ks)

            # Sample u_n via Vr (the diffused field) at internal nodes
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

        else:
            raise ValueError(update)

        stokes.solve(zero_init_guess=False)

        # Record diagnostics
        upper_pos = mesh.X.coords[internal_idx]
        upper_r = np.sqrt(upper_pos[:, 0] ** 2 + upper_pos[:, 1] ** 2)
        upper_dr = upper_r - r_o
        A_mode.append(project_mode_amp(upper_dr))
        A_max.append(float(np.abs(upper_dr).max()))
        times.append(times[-1] + float(delta_t.value))

        if verbose and (step % max(1, n_steps // 10) == 0
                        or step == n_steps - 1):
            print(f"  step {step + 1}: t={times[-1]:.4e} "
                  f"A_mode={A_mode[-1]:+.4e} "
                  f"A_max={A_max[-1]:.4e}", flush=True)

        if A_max[-1] > 10 * amp0:
            print(f"  *** BLOW-UP at step {step + 1} ***", flush=True)
            blow_up = True
            break

    # Final spectral diagnostic: Fourier decomposition of the
    # final boundary deformation.  Used to identify which modes
    # dominate the residual drift.
    upper_pos = mesh.X.coords[internal_idx]
    upper_r = np.sqrt(upper_pos[:, 0] ** 2 + upper_pos[:, 1] ** 2)
    final_dr = upper_r - r_o
    n_modes_diag = min(20, len(internal_th) // 3)
    a_final, b_final = fourier_decompose(final_dr, n_modes_diag)
    if verbose:
        print("\n  Fourier decomposition of final boundary δr:",
              flush=True)
        print(f"  {'mode':>4} {'a (cos)':>12} {'b (sin)':>12} "
              f"{'|amplitude|':>12}", flush=True)
        for m in range(n_modes_diag + 1):
            amag = np.sqrt(a_final[m] ** 2 + b_final[m] ** 2)
            print(f"  {m:>4d} {a_final[m]:>+12.4e} "
                  f"{b_final[m]:>+12.4e} {amag:>12.4e}",
                  flush=True)

    return {
        'label': label,
        'scheme': scheme,
        'dt': float(delta_t.value),
        'times': np.asarray(times, dtype=float),
        'A_mode': np.asarray(A_mode),
        'A_max': np.asarray(A_max),
        'blow_up': blow_up,
        'final_dr': final_dr,
        'final_th': internal_th,
        'a_final': a_final,
        'b_final': b_final,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--res', type=int, default=20)
    parser.add_argument('--n-steps', type=int, default=32)
    parser.add_argument('--scheme', type=str, default='all',
                        choices=['all', 'fe-fssa', 'curvS-fssa',
                                 'fe-nofssa', 'curvS-nofssa',
                                 'fssa-vs-nofssa'])
    parser.add_argument('--dt-factor', type=float, default=1.0)
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)

    schemes = {
        'fe-fssa':      (True,  'fe'),
        'curvS-fssa':   (True,  'curvS'),
        'fe-nofssa':    (False, 'fe'),
        'curvS-nofssa': (False, 'curvS'),
    }
    if args.scheme == 'all':
        run_list = list(schemes.values())
    elif args.scheme == 'fssa-vs-nofssa':
        run_list = [schemes['fe-fssa'], schemes['fe-nofssa'],
                    schemes['curvS-fssa'], schemes['curvS-nofssa']]
    else:
        run_list = [schemes[args.scheme]]

    results = []
    for s in run_list:
        try:
            r = run(s, args.dt_factor, args.n_steps, res=args.res)
            results.append(r)
        except Exception as e:
            print(f"  *** EXCEPTION in {s}: {e!r} ***", flush=True)
            import traceback; traceback.print_exc()

    if not results:
        return

    tag = f"dtf{args.dt_factor:.2f}_n{args.n_steps}_internal_res{args.res}"
    out = os.path.join(OUT_DIR, f"phase_i2d_fs_etd_{tag}.npz")
    np.savez(out, **{
        f"{r['label']}_t":    r['times'] for r in results
    }, **{
        f"{r['label']}_A":    r['A_mode'] for r in results
    }, **{
        f"{r['label']}_Amax": r['A_max']  for r in results
    })
    print(f"\nWrote {out}", flush=True)

    print("\nFinal-step summary:")
    for r in results:
        print(f"  {r['label']:<50} A_mode={r['A_mode'][-1]:+.4e} "
              f"A_max={r['A_max'][-1]:.4e}", flush=True)


if __name__ == "__main__":
    main()
