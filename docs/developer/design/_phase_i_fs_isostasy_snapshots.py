"""Visual comparison of mesh + density at two milestones (halfway to
equilibrium and 'at' equilibrium) for the four working integrator
schemes (RK2, RK4, curvS, midpoint) on the full-buoyancy isostasy
problem.

For each scheme, runs the simulation; when h_pole first crosses ½·h_eq
the mesh/density is captured; the run continues to the end where a
second snapshot is captured.  Plots are tiled into a 4×2 figure.

Reference h_eq from the small-dt FE-noFSSA reference run.
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
SNAP_DIR = os.path.join(OUT_DIR, "isostasy_snapshots")


def _build(res=20, blob_amp=0.5):
    import underworld3 as uw
    r_inner, r_o, r_outer = 0.5, 1.0, 1.5
    cellsize = 1.0 / res
    x_b, y_b, sigma_b = 0.85, 0.0, 0.06

    mesh = uw.meshing.AnnulusInternalBoundary(
        radiusOuter=r_outer, radiusInternal=r_o, radiusInner=r_inner,
        cellSize_Outer=3.0 * cellsize, cellSize=cellsize, qdegree=3,
    )
    r, th = mesh.CoordinateSystem.R
    blob_fn = sympy.exp(
        -((mesh.X[0] - x_b) ** 2 + (mesh.X[1] - y_b) ** 2)
        / (2.0 * sigma_b ** 2))

    Vr = uw.discretisation.MeshVariable(
        f"Vr_iso_snap", mesh, vtype=uw.VarType.SCALAR, degree=1,
        continuous=True, varsymbol=r"v_r")
    v = uw.discretisation.MeshVariable(
        f"V_iso_snap", mesh, vtype=uw.VarType.VECTOR, degree=2,
        continuous=True, varsymbol=r"\mathbf{v}")
    p = uw.discretisation.MeshVariable(
        f"P_iso_snap", mesh, vtype=uw.VarType.SCALAR, degree=1,
        continuous=True, varsymbol="p")
    M = uw.discretisation.MeshVariable(
        f"M_iso_snap", mesh, vtype=uw.VarType.SCALAR, degree=0,
        continuous=False, varsymbol=r"\rho")

    layer_fn = sympy.Piecewise((1.0, r <= r_o), (0.0, True))
    M.data[:, 0] = np.asarray(uw.function.evaluate(
        layer_fn, M.coords)).flatten()

    X_initial = mesh.X.coords.copy()
    R_initial = np.sqrt(X_initial[:, 0] ** 2 + X_initial[:, 1] ** 2)
    THETA_initial = np.arctan2(X_initial[:, 1], X_initial[:, 0])
    is_internal = np.abs(R_initial - r_o) < 0.5 * cellsize / r_o
    internal_idx = np.where(is_internal)[0]
    sort_order = np.argsort(THETA_initial[internal_idx])
    internal_idx = internal_idx[sort_order]
    internal_th = THETA_initial[internal_idx]

    diffuser = uw.systems.Poisson(mesh, Vr)
    diffuser.constitutive_model = uw.constitutive_models.DiffusionModel
    diffuser.constitutive_model.Parameters.diffusivity = 1.0
    diffuser.add_essential_bc(sympy.Matrix([0.0]),
                              mesh.boundaries.Upper.name)
    diffuser.add_essential_bc(sympy.Matrix([0.0]),
                              mesh.boundaries.Internal.name)
    diffuser.add_essential_bc(sympy.Matrix([0.0]),
                              mesh.boundaries.Lower.name)
    diffuser.tolerance = 1.0e-3
    diffuser.solve()

    stokes = uw.systems.Stokes(mesh, velocityField=v, pressureField=p)
    stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
    stokes.constitutive_model.Parameters.shear_viscosity_0 = 0.1 + 0.9 * M.sym[0]
    stokes.penalty = 0.0
    # Full buoyancy (no Heaviside subtraction)
    stokes.bodyforce = (
        -(M.sym[0] - blob_amp * blob_fn)
        * mesh.CoordinateSystem.unit_e_0
    )
    stokes.tolerance = 1.0e-5
    stokes.add_essential_bc((0.0, 0.0), mesh.boundaries.Lower.name)
    stokes.add_essential_bc((0.0, 0.0), mesh.boundaries.Upper.name)
    stokes.solve()

    return {
        'mesh': mesh, 'r_o': r_o,
        'th_sym': th, 'r_sym': r,
        'Vr': Vr, 'v': v, 'p': p, 'M': M,
        'diffuser': diffuser, 'stokes': stokes,
        'internal_idx': internal_idx, 'internal_th': internal_th,
    }


def _step(state, scheme, dt_factor):
    """Take one adaptive-Δt step using `scheme` (FE/RK2/RK4/curvS/midpoint).
    Returns (h_pole, dt_used)."""
    import underworld3 as uw
    mesh = state['mesh']; v = state['v']; M = state['M']
    Vr = state['Vr']; diffuser = state['diffuser']
    stokes = state['stokes']
    internal_idx = state['internal_idx']
    internal_th = state['internal_th']
    th = state['th_sym']; r_o = state['r_o']
    unit_r = mesh.CoordinateSystem.unit_e_0

    dt = dt_factor * stokes.estimate_dt()
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
        diffuser.add_essential_bc(sympy.Matrix([0.0]),
                                  mesh.boundaries.Upper.name)
        diffuser.add_essential_bc(sympy.Matrix([inc_fn]),
                                  mesh.boundaries.Internal.name)
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

    saved_X = mesh.X.coords.copy()

    if scheme == 'fe':
        # Diffuse v·r̂; FE deformation
        diffuser._reset()
        diffuser.add_essential_bc(sympy.Matrix([0.0]),
                                  mesh.boundaries.Upper.name)
        diffuser.add_essential_bc(
            sympy.Matrix([v.sym.dot(unit_r)]),
            mesh.boundaries.Internal.name)
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
        diffuser.add_essential_bc(sympy.Matrix([0.0]),
                                  mesh.boundaries.Upper.name)
        diffuser.add_essential_bc(
            sympy.Matrix([v.sym.dot(unit_r)]),
            mesh.boundaries.Internal.name)
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
            diffuser.add_essential_bc(sympy.Matrix([0.0]),
                                      mesh.boundaries.Upper.name)
            diffuser.add_essential_bc(
                sympy.Matrix([v.sym.dot(unit_r)]),
                mesh.boundaries.Internal.name)
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
             blob_amp=0.5, x_b=0.85, y_b=0.0, sigma_b=0.06):
    """Capture the simulation state at this point.

    Writes:
      1. UW3-native checkpoint via mesh.write_timestep (H5 + XDMF) —
         the mesh + M variable, suitable for re-visualising with
         uw or paraview. Filename pattern:
           output/isostasy_snapshots/uw_<scheme>_<label>.{h5,xmf}
      2. A pyvista VTU of the thresholded heavy-layer pvmesh with
         cell-data eff = M − α·blob, for fast pyvista plotting.
         Filename: output/isostasy_snapshots/pv_<scheme>_<label>.vtu
    """
    import underworld3.visualisation as vis
    mesh = state['mesh']; M = state['M']
    os.makedirs(SNAP_DIR, exist_ok=True)

    # 1) UW3-native checkpoint
    uw_filename = f"uw_{scheme_name}_{label_short}"
    mesh.write_timestep(
        filename=uw_filename,
        index=0,
        outputPath=SNAP_DIR,
        meshVars=[M],
        meshUpdates=True,
        create_xdmf=True,
    )

    # 2) PyVista VTU for fast plotting
    pvmesh = vis.mesh_to_pv_mesh(mesh)
    M_cell = M.data[:, 0].astype(float)
    centroids = mesh._centroids
    blob_cell = blob_amp * np.exp(
        -((centroids[:, 0] - x_b) ** 2
          + (centroids[:, 1] - y_b) ** 2)
        / (2.0 * sigma_b ** 2))
    eff_cell = M_cell - blob_cell

    pvmesh.cell_data["M"] = M_cell
    pvmesh.cell_data["eff"] = eff_cell
    heavy = pvmesh.threshold(0.5, scalars="M", method="upper")

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
             h_pole=h_pole_value)
    return pv_path


def _render_panel(plotter, idx, vtu_path, scheme_name, full_label,
                  show_bar=False, r_o=1.0):
    heavy = pv.read(vtu_path)
    plotter.subplot(*idx)
    plotter.add_mesh(
        heavy, scalars="eff", show_edges=True,
        edge_color='#888888', line_width=0.3,
        cmap='RdYlBu_r', clim=[0.4, 1.0],
        scalar_bar_args={'title': 'M − α·blob', 'fmt': '%.2f',
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


def _plot_grid(schemes, labels, h_eq):
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

    out = os.path.join(OUT_DIR, "phase_i2d_fs_isostasy_snapshots.png")
    pl.screenshot(out)
    pl.close()
    print(f"  wrote {out}", flush=True)


def _plot_profiles(schemes, labels):
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
    fig.suptitle("Internal-boundary surface profile by scheme",
                 fontsize=12)
    fig.tight_layout()
    out = os.path.join(OUT_DIR,
                       "phase_i2d_fs_isostasy_profiles.png")
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--res', type=int, default=20)
    p.add_argument('--n-steps', type=int, default=16)
    p.add_argument('--dt-factor', type=float, default=1.0)
    p.add_argument('--h-eq', type=float, default=0.0298,
                   help="Equilibrium pole-height from small-dt FE ref")
    p.add_argument('--plot-only', action='store_true',
                   help="Skip simulation; render from existing VTU "
                   "snapshots in output/isostasy_snapshots/.")
    args = p.parse_args()

    schemes = ['rk2', 'rk4', 'curvS', 'midpoint']
    labels = ['halfway', 'final']
    h_half = args.h_eq / 2.0

    if not args.plot_only:
        for scheme in schemes:
            print(f"\n=== {scheme} ===", flush=True)
            state = _build(res=args.res)

            captured_half = False
            h_pole = 0.0
            for s in range(args.n_steps):
                h_pole, dt = _step(state, scheme, args.dt_factor)
                print(f"  step {s+1}: h_pole={h_pole:+.4e} "
                      f"Δt={dt:.3e}", flush=True)
                if not captured_half and h_pole >= h_half:
                    _capture(state, scheme, 'halfway', h_pole)
                    captured_half = True
            _capture(state, scheme, 'final', h_pole)

            if not captured_half:
                print(f"  WARNING: {scheme} never reached h_half="
                      f"{h_half:.4e}", flush=True)

    # Render plots from saved VTU + profile files
    _plot_grid(schemes, labels, args.h_eq)
    _plot_profiles(schemes, labels)


if __name__ == "__main__":
    main()
