"""Phase E PyVista field plot — hybrid BDF/ETD integrator.

Same structure as ``_plot_phase_b_pyvista.py`` but uses
``TransverseIsotropicVEPFlowModel(integrator='hybrid', fault_weight=...)``.
Captures the yield-active step at θ=+15°, τ_y ∈ {0.05, 0.15} and
renders the 4-panel field figure (u_y, |ε̇|_II, |σ|_II, yield_ratio).
"""

import os
import time

import numpy as np
import sympy

import underworld3 as uw
from underworld3 import VarType
from underworld3.function import expression
import underworld3.visualisation as vis


V0 = 0.5
OMEGA = np.pi / 2.0
DT = 0.05
H = 1.0; W = 1.0
FAULT_LENGTH = 0.6
FAULT_WIDTH = 0.06
ETA_0 = 1.0; ETA_1 = 1.0; MU = 1.0
TAU_Y_BULK = 200.0
RES = 32

OUT_DIR = "output"


def _key(theta_deg, tau_y_at_fault):
    return f"phase_e_hybrid_th{theta_deg:+.0f}_ty{tau_y_at_fault:.2f}".replace(".", "p")


def _meta_path(key):
    return os.path.join(OUT_DIR, key + ".meta.npz")


def build_model(theta_deg, tau_y_at_fault, label_suffix=""):
    label = _key(theta_deg, tau_y_at_fault) + label_suffix

    mesh = uw.meshing.StructuredQuadBox(
        elementRes=(RES, RES),
        minCoords=(0.0, 0.0), maxCoords=(W, H),
        qdegree=3,
    )

    u = uw.discretisation.MeshVariable(
        f"U_{label}", mesh, 2, degree=2, vtype=VarType.VECTOR,
    )
    p_sol = uw.discretisation.MeshVariable(
        f"P_{label}", mesh, 1, degree=1, continuous=True, vtype=VarType.SCALAR,
    )

    theta = np.radians(theta_deg)
    cx, cy = 0.5 * W, 0.5 * H
    dx = 0.5 * FAULT_LENGTH * np.cos(theta)
    dy = 0.5 * FAULT_LENGTH * np.sin(theta)
    fault = uw.meshing.Surface(
        f"fault_{label}", mesh,
        np.array([[cx - dx, cy - dy], [cx + dx, cy + dy]]),
        symbol=f"F{label}",
    )
    fault.discretize()

    n_x = -np.sin(theta); n_y = np.cos(theta)
    director = sympy.Matrix([n_x, n_y])
    weakness = fault.influence_function(
        width=FAULT_WIDTH,
        value_near=1.0 / tau_y_at_fault, value_far=1.0 / TAU_Y_BULK,
        profile="gaussian",
    )
    tau_y_field = 1.0 / weakness

    weakness_min = 1.0 / TAU_Y_BULK
    weakness_max = 1.0 / tau_y_at_fault
    fault_weight = (weakness - weakness_min) / (weakness_max - weakness_min)

    edot_II_var = uw.discretisation.MeshVariable(
        f"edotII_{label}", mesh, 1, degree=1, continuous=True, vtype=VarType.SCALAR,
    )
    tau_y_var = uw.discretisation.MeshVariable(
        f"tauy_{label}", mesh, 1, degree=1, continuous=True, vtype=VarType.SCALAR,
    )
    sigma_II_var = uw.discretisation.MeshVariable(
        f"sigmaII_{label}", mesh, 1, degree=1, continuous=True, vtype=VarType.SCALAR,
    )
    yield_ratio_var = uw.discretisation.MeshVariable(
        f"yieldRatio_{label}", mesh, 1, degree=1, continuous=True, vtype=VarType.SCALAR,
    )

    stokes = uw.systems.Stokes(mesh, velocityField=u, pressureField=p_sol)
    stokes.constitutive_model = uw.constitutive_models.TransverseIsotropicVEPFlowModel(
        stokes.Unknowns, integrator="hybrid", fault_weight=fault_weight,
    )
    cm = stokes.constitutive_model
    cm.Parameters.shear_viscosity_0 = ETA_0
    cm.Parameters.shear_viscosity_1 = ETA_1
    cm.Parameters.shear_modulus = MU
    cm.Parameters.yield_stress = tau_y_field
    cm.Parameters.director = director
    cm.Parameters.shear_viscosity_min = ETA_0 * 1.0e-3
    cm.Parameters.strainrate_inv_II_min = 1.0e-6
    cm.yield_mode = "softmin"

    stokes.saddle_preconditioner = 1.0 / cm.K
    stokes.tolerance = 1.0e-4
    stokes.petsc_options["ksp_type"] = "fgmres"
    stokes.petsc_options["snes_force_iteration"] = True

    V_top = expression(rf"V_{{top,{label}}}", sympy.Float(0.0), "Top BC")
    stokes.add_essential_bc(sympy.Matrix([V_top, 0.0]), "Top")
    stokes.add_essential_bc(sympy.Matrix([0.0, 0.0]), "Bottom")
    stokes.add_essential_bc((sympy.oo, 0.0), "Left")
    stokes.add_essential_bc((sympy.oo, 0.0), "Right")
    stokes.bodyforce = sympy.Matrix([0.0, 0.0])

    return dict(
        mesh=mesh, stokes=stokes,
        u=u, V_top=V_top,
        edot_II_var=edot_II_var, tau_y_var=tau_y_var,
        sigma_II_var=sigma_II_var, yield_ratio_var=yield_ratio_var,
        n_vec=np.array([n_x, n_y]),
    )


def capture(theta_deg, tau_y_at_fault, n_periods=1.5):
    obj = build_model(theta_deg, tau_y_at_fault, label_suffix="_cap")
    mesh = obj["mesh"]; stokes = obj["stokes"]
    u = obj["u"]; V_top = obj["V_top"]
    edot_II_var = obj["edot_II_var"]
    tau_y_var = obj["tau_y_var"]
    sigma_II_var = obj["sigma_II_var"]
    yield_ratio_var = obj["yield_ratio_var"]
    cm = stokes.constitutive_model
    DFDt = stokes.Unknowns.DFDt

    sigma_coords = DFDt.psi_star[0].coords
    n_x, n_y = obj["n_vec"]; cx, cy = 0.5 * W, 0.5 * H
    sd = np.abs((sigma_coords[:, 0] - cx) * n_x + (sigma_coords[:, 1] - cy) * n_y)
    fault_mask = sd < 1.5 * FAULT_WIDTH
    E_sym = stokes.Unknowns.E

    T_END = n_periods * 2.0 * np.pi / OMEGA
    best = None
    saved = []
    iters = []; reasons = []
    t_cur = 0.0
    t0 = time.time()
    while t_cur < T_END - 1e-9:
        dt = min(DT, T_END - t_cur)
        t_end_step = t_cur + dt
        v_now = V0 * float(np.cos(OMEGA * t_end_step))
        V_top.sym = sympy.Float(v_now)
        cm.Parameters.dt_elastic = dt
        stokes.solve(zero_init_guess=False, timestep=dt, divergence_retries=2)
        iters.append(int(stokes.snes.getIterationNumber()))
        reasons.append(int(stokes.snes.getConvergedReason()))

        sigma_arr = np.asarray(DFDt.psi_star[0].array)
        sigma_II = np.sqrt(0.5 * (sigma_arr ** 2).sum(axis=(1, 2)))
        recordable = t_end_step > 0.5 * 2.0 * np.pi / OMEGA
        in_fault_max = float(sigma_II[fault_mask].max()) if fault_mask.any() else 0.0
        if recordable and (best is None or in_fault_max > best[0]):
            best = (in_fault_max, len(saved))
        saved.append(dict(
            t=t_end_step, v_top=v_now,
            u_arr=np.asarray(u.array).copy(),
            sigma_arr=sigma_arr.copy(),
            sigma_II=sigma_II.copy(),
        ))
        t_cur = t_end_step

    if best is None:
        best = (saved[-1]["sigma_II"].max(), len(saved) - 1)
    chosen = saved[best[1]]
    u.array[...] = chosen["u_arr"]
    DFDt.psi_star[0].array[...] = chosen["sigma_arr"]
    u._sync_lvec_to_gvec()

    plot_coords = edot_II_var.coords
    edot_xx = np.asarray(uw.function.evaluate(E_sym[0, 0], plot_coords)).flatten()
    edot_xy = np.asarray(uw.function.evaluate(E_sym[0, 1], plot_coords)).flatten()
    edot_yy = np.asarray(uw.function.evaluate(E_sym[1, 1], plot_coords)).flatten()
    edot_II_at_plot = np.sqrt(0.5 * (edot_xx ** 2 + edot_yy ** 2 + 2 * edot_xy ** 2))
    edot_II_var.array[:, 0, 0] = edot_II_at_plot

    ty_at_plot = np.asarray(
        uw.function.evaluate(cm.Parameters.yield_stress.sym, plot_coords)
    ).flatten()
    tau_y_var.array[:, 0, 0] = ty_at_plot

    sigma_sym_II = sympy.sqrt(
        (DFDt.psi_star[0].sym * DFDt.psi_star[0].sym).trace() / 2
    )
    try:
        sigma_II_at_plot = np.asarray(
            uw.function.evaluate(sigma_sym_II, plot_coords)
        ).flatten()
    except Exception:
        from underworld3.kdtree import KDTree
        tree = KDTree(np.asarray(sigma_coords))
        sigma_II_at_plot = tree.rbf_interpolator_local(
            plot_coords, chosen["sigma_II"][:, None], 4, 2,
        ).flatten()
    sigma_II_var.array[:, 0, 0] = sigma_II_at_plot
    yield_ratio_var.array[:, 0, 0] = sigma_II_at_plot / np.maximum(ty_at_plot, 1e-30)

    key = _key(theta_deg, tau_y_at_fault)
    os.makedirs(OUT_DIR, exist_ok=True)
    mesh.write_timestep(
        key, index=0, outputPath=OUT_DIR,
        meshVars=[u, edot_II_var, tau_y_var, sigma_II_var, yield_ratio_var],
        create_xdmf=True,
    )
    DFDt.psi_star[0].write(
        os.path.join(OUT_DIR, key + ".mesh.sigma.00000.h5")
    )

    iters_arr = np.array(iters); reasons_arr = np.array(reasons)
    metadata = dict(
        theta_deg=theta_deg,
        tau_y_at_fault=tau_y_at_fault,
        n_x=float(n_x), n_y=float(n_y),
        t=float(chosen["t"]), v_top=float(chosen["v_top"]),
        T_END=float(T_END), RES=int(RES),
        wall_seconds=float(time.time() - t0),
        max_in_fault_sigma_II=float(best[0]),
        n_steps=len(saved),
        iters=iters_arr, reasons=reasons_arr,
    )
    np.savez(os.path.join(OUT_DIR, key + ".meta.npz"), **metadata)
    n_diverged = int((reasons_arr < 0).sum())
    print(
        f"  ran {len(saved)} steps in {metadata['wall_seconds']:.1f}s; "
        f"chose t={metadata['t']:.3f}, V_top={metadata['v_top']:+.4f}; "
        f"max in-fault σ_II = {metadata['max_in_fault_sigma_II']:.4f} "
        f"({metadata['max_in_fault_sigma_II']/tau_y_at_fault:.3f}·τ_y_centre); "
        f"checkpointed → {OUT_DIR}/{key}.*",
        flush=True,
    )
    print(
        f"  SNES iters per step: mean={iters_arr.mean():.1f} "
        f"median={int(np.median(iters_arr))} max={iters_arr.max()} "
        f"diverged_steps={n_diverged}/{len(reasons_arr)}",
        flush=True,
    )


def load_into_fresh_model(theta_deg, tau_y_at_fault):
    obj = build_model(theta_deg, tau_y_at_fault, label_suffix="_load")
    key = _key(theta_deg, tau_y_at_fault)
    obj["edot_II_var"].read_timestep(
        key, obj["edot_II_var"].clean_name.replace("_load", "_cap"),
        index=0, outputPath=OUT_DIR,
    )
    obj["tau_y_var"].read_timestep(
        key, obj["tau_y_var"].clean_name.replace("_load", "_cap"),
        index=0, outputPath=OUT_DIR,
    )
    obj["sigma_II_var"].read_timestep(
        key, obj["sigma_II_var"].clean_name.replace("_load", "_cap"),
        index=0, outputPath=OUT_DIR,
    )
    obj["yield_ratio_var"].read_timestep(
        key, obj["yield_ratio_var"].clean_name.replace("_load", "_cap"),
        index=0, outputPath=OUT_DIR,
    )
    obj["u"].read_timestep(
        key, obj["u"].clean_name.replace("_load", "_cap"),
        index=0, outputPath=OUT_DIR,
    )
    meta = dict(np.load(os.path.join(OUT_DIR, key + ".meta.npz")))
    obj["meta"] = {k: (v.item() if v.ndim == 0 else v) for k, v in meta.items()}
    return obj


def plot_panels(obj, out_path, off_screen=True):
    import pyvista as pv

    pv.global_theme.background = "white"
    pv.global_theme.anti_aliasing = "ssaa"

    mesh = obj["mesh"]
    u = obj["u"]
    sII = obj["sigma_II_var"]
    eII = obj["edot_II_var"]
    yr = obj["yield_ratio_var"]
    ty = obj["tau_y_var"]
    meta = obj["meta"]

    pvmesh = vis.mesh_to_pv_mesh(mesh)
    pvmesh.point_data["sigma_II"] = vis.scalar_fn_to_pv_points(pvmesh, sII.sym)
    pvmesh.point_data["edot_II"] = vis.scalar_fn_to_pv_points(pvmesh, eII.sym)
    pvmesh.point_data["yield_ratio"] = np.clip(
        vis.scalar_fn_to_pv_points(pvmesh, yr.sym), 0.0, 1.5,
    )
    pvmesh.point_data["tau_y"] = vis.scalar_fn_to_pv_points(pvmesh, ty.sym)

    u_cloud = vis.meshVariable_to_pv_cloud(u)
    u_cloud.point_data["u"] = vis.vector_fn_to_pv_points(u_cloud, u.sym)
    u_speed = np.linalg.norm(u_cloud.point_data["u"][:, :2], axis=1)
    u_cloud.point_data["|u|"] = u_speed
    pvmesh.point_data["u_y"] = vis.scalar_fn_to_pv_points(pvmesh, u.sym[1])
    pvmesh.point_data["|u|"] = vis.scalar_fn_to_pv_points(
        pvmesh, sympy.sqrt(u.sym.dot(u.sym))
    )

    n_x = float(meta["n_x"]); n_y = float(meta["n_y"])
    cx, cy = 0.5 * W, 0.5 * H
    L = FAULT_LENGTH
    t_x, t_y = n_y, -n_x
    fault_line = pv.Line(
        (cx - 0.5 * L * t_x, cy - 0.5 * L * t_y, 0.0),
        (cx + 0.5 * L * t_x, cy + 0.5 * L * t_y, 0.0),
    )

    pl = pv.Plotter(off_screen=off_screen, shape=(2, 2),
                    window_size=(1500, 1400), border=True)

    def _common(p):
        p.view_xy()
        p.camera.parallel_projection = True
        p.add_mesh(fault_line, color="red", line_width=4)

    pl.subplot(0, 0)
    uy_max = float(np.max(np.abs(pvmesh.point_data["u_y"])))
    pl.add_mesh(
        pvmesh, scalars="u_y", cmap="seismic",
        clim=(-uy_max, uy_max),
        show_scalar_bar=True, scalar_bar_args={"title": "u_y"},
    )
    sub = max(1, len(u_cloud.points) // 250)
    pl.add_arrows(u_cloud.points[::sub], u_cloud.point_data["u"][::sub],
                  mag=0.35, color="#333333")
    pl.add_text(
        "velocity: u_y heatmap (+arrows show full u)",
        position="upper_edge", font_size=11, color="black",
    )
    _common(pl)

    pl.subplot(0, 1)
    pl.add_mesh(pvmesh, scalars="edot_II", cmap="viridis",
                show_scalar_bar=True, scalar_bar_args={"title": "|ε̇|_II"})
    pl.add_text("|ε̇|_II", position="upper_edge", font_size=12, color="black")
    _common(pl)

    pl.subplot(1, 0)
    pl.add_mesh(pvmesh, scalars="sigma_II", cmap="magma",
                show_scalar_bar=True, scalar_bar_args={"title": "|σ|_II"})
    ty_levels = [meta["tau_y_at_fault"] * f for f in (4.0, 20.0, 100.0)]
    contours = pvmesh.contour(isosurfaces=ty_levels, scalars="tau_y")
    if contours.n_points > 0:
        pl.add_mesh(contours, color="cyan", line_width=1.2)
    pl.add_text("|σ|_II — cyan: τ_y(x) contours",
                position="upper_edge", font_size=12, color="black")
    _common(pl)

    pl.subplot(1, 1)
    pl.add_mesh(pvmesh, scalars="yield_ratio", cmap="RdYlGn_r",
                clim=(0.0, 1.2),
                show_scalar_bar=True, scalar_bar_args={"title": "|σ|_II / τ_y(x)"})
    yc = pvmesh.contour(isosurfaces=[1.0], scalars="yield_ratio")
    if yc.n_points > 0:
        pl.add_mesh(yc, color="black", line_width=2.0)
    pl.add_text("yield activation",
                position="upper_edge", font_size=12, color="black")
    _common(pl)

    pl.add_text(
        f"Phase E hybrid BDF/ETD, RES={int(meta['RES'])}, "
        f"θ={meta['theta_deg']:+.0f}°, τ_y_fault={meta['tau_y_at_fault']} "
        f"(t={meta['t']:.2f}, V_top={meta['v_top']:+.3f})",
        position="lower_edge", font_size=10, color="black",
    )

    pl.screenshot(out_path, scale=1.5)
    pl.close()
    print(f"  wrote {out_path}", flush=True)


def capture_or_load(theta_deg, tau_y_at_fault, n_periods=1.5):
    if os.path.exists(_meta_path(_key(theta_deg, tau_y_at_fault))):
        print(f"  cache hit: {_key(theta_deg, tau_y_at_fault)}.* — skipping run",
              flush=True)
    else:
        print(f"  cache miss: running capture", flush=True)
        capture(theta_deg, tau_y_at_fault, n_periods=n_periods)
    return load_into_fresh_model(theta_deg, tau_y_at_fault)


def main():
    cases = [(15.0, 0.05), (15.0, 0.15)]
    for theta, ty in cases:
        print(f"\n=== θ={theta:+.0f}°, τ_y={ty:.2f} ===", flush=True)
        obj = capture_or_load(theta, ty, n_periods=1.5)
        out_path = os.path.join(
            OUT_DIR,
            f"exp_integrator_phase_e_pyvista_hybrid_th{theta:+.0f}_ty{ty:.2f}".replace(".", "p") + ".png",
        )
        plot_panels(obj, out_path)


if __name__ == "__main__":
    main()
