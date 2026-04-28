"""Phase B PyVista field plots — high-resolution snapshot at yield-active step.

Runs the bench_ti_vep_harmonic geometry with ETD-2 at **RES=32** for one
yielding cycle and **checkpoints** the snapshot via ``mesh.write_timestep``
(HDF5 + XDMF, ParaView-compatible) so we can replot without re-running.
Renders 4-panel PyVista figures using the UW3 ``visualisation`` API.

Capture-or-load pattern: each case checkpoints to
``output/phase_b_<key>.{mesh, U, sigma, edot_II, ty, sigma_II, yield_ratio}.00000.h5``.
If those files exist, skip the simulation and read back from disk.

Run::

    pixi run -e amr-dev python -u docs/developer/design/_plot_phase_b_pyvista.py

Force re-capture::

    rm output/phase_b_*.h5 output/phase_b_*.xdmf
    pixi run -e amr-dev python -u docs/developer/design/_plot_phase_b_pyvista.py
"""

import os
import sys
import time

import numpy as np
import sympy

import underworld3 as uw
from underworld3 import VarType
from underworld3.function import expression
import underworld3.visualisation as vis


# Geometric parameters (kept aligned with the killer test, but at RES=32)
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
    return f"phase_b_th{theta_deg:+.0f}_ty{tau_y_at_fault:.2f}".replace(".", "p")


def _meta_path(key):
    return os.path.join(OUT_DIR, key + ".meta.npz")


# ---------------------------------------------------------------------------
# Build a fresh model + plotting variables for a given (θ, τ_y).
# Used by both the capture path and the load path so the mesh+var
# discretisation is byte-identical.
# ---------------------------------------------------------------------------

def build_model(theta_deg, tau_y_at_fault, label_suffix=""):
    label = _key(theta_deg, tau_y_at_fault) + label_suffix

    mesh = uw.meshing.StructuredQuadBox(
        elementRes=(RES, RES),
        minCoords=(0.0, 0.0), maxCoords=(W, H),
        qdegree=3,
    )

    # Solver variables (degree=2 / degree=1)
    u = uw.discretisation.MeshVariable(
        f"U_{label}", mesh, 2, degree=2, vtype=VarType.VECTOR,
    )
    p_sol = uw.discretisation.MeshVariable(
        f"P_{label}", mesh, 1, degree=1,
        continuous=True, vtype=VarType.SCALAR,
    )

    # Fault geometry / spatial yield_stress field
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

    # Scalar mesh variables for the four post-solve plottable fields.
    # Same degree=1 / continuous so they share the canonical mesh nodes.
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

    # Solver (only built when capturing)
    stokes = uw.systems.Stokes(mesh, velocityField=u, pressureField=p_sol)
    stokes.constitutive_model = uw.constitutive_models.TransverseIsotropicVEPFlowModel(
        stokes.Unknowns, integrator="etd",
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


# ---------------------------------------------------------------------------
# Capture: run the sim, project plottable fields, write_timestep
# ---------------------------------------------------------------------------

def capture(theta_deg, tau_y_at_fault, n_periods=1.5):
    """Run + checkpoint a yield-active snapshot via mesh.write_timestep."""
    obj = build_model(theta_deg, tau_y_at_fault, label_suffix="_cap")
    mesh = obj["mesh"]; stokes = obj["stokes"]
    u = obj["u"]
    V_top = obj["V_top"]
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
    ty_at_psi_coords = np.asarray(
        uw.function.evaluate(cm.Parameters.yield_stress.sym, sigma_coords)
    ).flatten()

    T_END = n_periods * 2.0 * np.pi / OMEGA
    best = None     # (in_fault_max, step_index)
    saved = []      # full state history so we can rewind to the chosen step
    t_cur = 0.0
    t0 = time.time()
    while t_cur < T_END - 1e-9:
        dt = min(DT, T_END - t_cur)
        t_end_step = t_cur + dt
        v_now = V0 * float(np.cos(OMEGA * t_end_step))
        V_top.sym = sympy.Float(v_now)
        cm.Parameters.dt_elastic = dt
        stokes.solve(zero_init_guess=False, timestep=dt, divergence_retries=2)

        sigma_arr = np.asarray(DFDt.psi_star[0].array)
        sigma_II = np.sqrt(0.5 * (sigma_arr ** 2).sum(axis=(1, 2)))
        # post-transient window
        recordable = t_end_step > 0.5 * 2.0 * np.pi / OMEGA
        in_fault_max = float(sigma_II[fault_mask].max()) if fault_mask.any() else 0.0
        if recordable and (best is None or in_fault_max > best[0]):
            # Snapshot current state
            best = (in_fault_max, len(saved))
        # Snapshot of solver+history state (cheap: arrays)
        saved.append(dict(
            t=t_end_step, v_top=v_now,
            u_arr=np.asarray(u.array).copy(),
            sigma_arr=sigma_arr.copy(),
            sigma_II=sigma_II.copy(),
        ))
        t_cur = t_end_step

    if best is None:
        # Edge case: didn't reach the post-transient window
        best = (saved[-1]["sigma_II"].max(), len(saved) - 1)
    chosen = saved[best[1]]

    # Replant chosen state (so subsequent eval calls see the right u for ε̇)
    u.array[...] = chosen["u_arr"]
    DFDt.psi_star[0].array[...] = chosen["sigma_arr"]
    u._sync_lvec_to_gvec()

    # Project the four scalar fields onto the plotting mesh variables.
    # Use direct nodal evaluation (degree=1 continuous nodes).
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

    # σ_II at plot nodes — interpolate from psi_star[0] coords via kd-tree.
    # psi_star[0] degree = u.degree-1 = 1 → typically same nodes as plot
    # mesh, but in general we use uw.function.evaluate on psi_star[0].sym
    # for safety.
    sigma_sym_II = sympy.sqrt((DFDt.psi_star[0].sym * DFDt.psi_star[0].sym).trace() / 2)
    try:
        sigma_II_at_plot = np.asarray(
            uw.function.evaluate(sigma_sym_II, plot_coords)
        ).flatten()
    except Exception:
        # Fallback — kd-tree interpolation from psi_star coords
        from underworld3.kdtree import KDTree
        tree = KDTree(np.asarray(sigma_coords))
        sigma_II_at_plot = tree.rbf_interpolator_local(
            plot_coords, chosen["sigma_II"][:, None], 4, 2,
        ).flatten()
    sigma_II_var.array[:, 0, 0] = sigma_II_at_plot

    yield_ratio_var.array[:, 0, 0] = sigma_II_at_plot / np.maximum(ty_at_plot, 1e-30)

    # Write the checkpoint
    key = _key(theta_deg, tau_y_at_fault)
    os.makedirs(OUT_DIR, exist_ok=True)
    mesh.write_timestep(
        key, index=0, outputPath=OUT_DIR,
        meshVars=[u, edot_II_var, tau_y_var, sigma_II_var, yield_ratio_var],
        create_xdmf=True,
    )
    # Also the raw stress (rank-2 sym tensor) — psi_star[0] is on the
    # solver's DDt, save by writing its underlying mesh-variable
    DFDt.psi_star[0].write(
        os.path.join(OUT_DIR, key + ".mesh.sigma.00000.h5")
    )

    metadata = dict(
        theta_deg=theta_deg,
        tau_y_at_fault=tau_y_at_fault,
        n_x=float(n_x), n_y=float(n_y),
        t=float(chosen["t"]), v_top=float(chosen["v_top"]),
        T_END=float(T_END), RES=int(RES),
        wall_seconds=float(time.time() - t0),
        max_in_fault_sigma_II=float(best[0]),
        n_steps=len(saved),
    )
    np.savez(os.path.join(OUT_DIR, key + ".meta.npz"), **metadata)

    print(
        f"  ran {len(saved)} steps in {metadata['wall_seconds']:.1f}s; "
        f"chose t={metadata['t']:.3f}, V_top={metadata['v_top']:+.4f}; "
        f"max in-fault σ_II = {metadata['max_in_fault_sigma_II']:.4f} "
        f"({metadata['max_in_fault_sigma_II']/tau_y_at_fault:.3f}·τ_y_centre); "
        f"checkpointed → {OUT_DIR}/{key}.*",
        flush=True,
    )


# ---------------------------------------------------------------------------
# Load: rebuild mesh + variables, read_timestep into them, return them
# ---------------------------------------------------------------------------

def load_into_fresh_model(theta_deg, tau_y_at_fault):
    obj = build_model(theta_deg, tau_y_at_fault, label_suffix="_load")
    key = _key(theta_deg, tau_y_at_fault)
    # The plotting mesh variables — ``read_timestep`` interpolates from the
    # checkpointed coords to the current mesh's coords (kd-tree RBF).
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


# ---------------------------------------------------------------------------
# Plot via UW3 visualisation (PyVista)
# ---------------------------------------------------------------------------

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

    # Build PV mesh once + add scalar fields as point_data
    pvmesh = vis.mesh_to_pv_mesh(mesh)
    pvmesh.point_data["sigma_II"] = vis.scalar_fn_to_pv_points(pvmesh, sII.sym)
    pvmesh.point_data["edot_II"] = vis.scalar_fn_to_pv_points(pvmesh, eII.sym)
    pvmesh.point_data["yield_ratio"] = np.clip(
        vis.scalar_fn_to_pv_points(pvmesh, yr.sym), 0.0, 1.5,
    )
    pvmesh.point_data["tau_y"] = vis.scalar_fn_to_pv_points(pvmesh, ty.sym)

    # Velocity arrows from the velocity-degree variable
    u_cloud = vis.meshVariable_to_pv_cloud(u)
    u_cloud.point_data["u"] = vis.vector_fn_to_pv_points(u_cloud, u.sym)
    u_speed = np.linalg.norm(u_cloud.point_data["u"][:, :2], axis=1)
    u_cloud.point_data["|u|"] = u_speed

    # Fault line for overlay
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

    # Velocity
    pl.subplot(0, 0)
    pl.add_mesh(pvmesh, scalars=None, color="#eeeeee", show_edges=False)
    sub = max(1, len(u_cloud.points) // 250)
    pl.add_arrows(u_cloud.points[::sub], u_cloud.point_data["u"][::sub],
                  mag=0.35, color="#333333")
    pl.add_text("velocity field", position="upper_edge", font_size=12, color="black")
    _common(pl)

    # |ε̇|_II
    pl.subplot(0, 1)
    pl.add_mesh(pvmesh, scalars="edot_II", cmap="viridis",
                show_scalar_bar=True, scalar_bar_args={"title": "|ε̇|_II"})
    pl.add_text("|ε̇|_II", position="upper_edge", font_size=12, color="black")
    _common(pl)

    # |σ|_II
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

    # σ/τ_y ratio with active surface contour
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
        f"ETD-2, RES={int(meta['RES'])}, θ={meta['theta_deg']:+.0f}°, "
        f"τ_y_fault={meta['tau_y_at_fault']} "
        f"(t={meta['t']:.2f}, V_top={meta['v_top']:+.3f})",
        position="lower_edge", font_size=10, color="black",
    )

    pl.screenshot(out_path, scale=1.5)
    pl.close()
    print(f"  wrote {out_path}", flush=True)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def capture_or_load(theta_deg, tau_y_at_fault, n_periods=1.5):
    if os.path.exists(_meta_path(_key(theta_deg, tau_y_at_fault))):
        print(f"  cache hit: {_key(theta_deg, tau_y_at_fault)}.* — skipping run",
              flush=True)
    else:
        print(f"  cache miss: running capture", flush=True)
        capture(theta_deg, tau_y_at_fault, n_periods=n_periods)
    return load_into_fresh_model(theta_deg, tau_y_at_fault)


def main():
    cases = [(0.0, 0.15), (15.0, 0.15)]
    for theta, ty in cases:
        print(f"\n=== θ={theta:+.0f}°, τ_y={ty:.2f} ===", flush=True)
        obj = capture_or_load(theta, ty, n_periods=1.5)
        out = os.path.join(
            OUT_DIR,
            f"exp_integrator_phase_b_pyvista_th{theta:+.0f}_ty{ty:.2f}".replace(".", "p")
            + ".png",
        )
        plot_panels(obj, out)


if __name__ == "__main__":
    main()
