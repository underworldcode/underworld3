"""Side-by-side publication figure for a fault/ridge convection run:

  LEFT  : temperature + velocity streamlines (no mesh, no scale-bar)
  RIGHT : adapted mesh + strain-rate II

The strain-rate is built with a SMOOTHED L2 projection (uw.systems.Projection
with a small smoothing term) onto a continuous field — point-sampling the raw
invariant (derivatives of the P2 velocity are discontinuous per element) looks
blocky/noisy; the projection gives a clean continuous field. Fault trace (red /
lime) and the surface ridge marker (magenta) are overlaid from the manifest.

Usage:
  python figure_compare.py --run <dir> --index 380
  python figure_compare.py --run <dir> --index 380 --smoothing 3e-3 --strm-width 0.7
"""
import os
import argparse

import numpy as np
import sympy
import underworld3 as uw
import underworld3.visualisation as vis
import pyvista as pv

from render import (fault_from_manifest, _add_fault,
                    ridge_from_manifest, _add_ridge_marker, _per_step_theta)

pv.OFF_SCREEN = True
RUN_NAME = "run"


def smoothed_strainrate(mesh, v, smoothing, degree=2):
    """ε̇_II projected onto a continuous field with a smoothing term."""
    X = mesh.CoordinateSystem.X
    E = sympy.Matrix(2, 2, lambda i, j:
                     sympy.Rational(1, 2) * (v.sym[i].diff(X[j]) + v.sym[j].diff(X[i])))
    Einv2 = sympy.sqrt(sympy.Rational(1, 2) *
                       (E[0, 0] ** 2 + E[1, 1] ** 2 + 2 * E[0, 1] ** 2))
    sr = uw.discretisation.MeshVariable("Edot", mesh, 1, degree=degree,
                                        continuous=True)
    proj = uw.systems.Projection(mesh, sr)
    proj.uw_function = Einv2
    proj.smoothing = smoothing
    proj.solve()
    return sr


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--index", type=int, required=True)
    ap.add_argument("--smoothing", type=float, default=2.0e-3,
                    help="Projection smoothing for the strain-rate.")
    ap.add_argument("--strm-width", type=float, default=0.7,
                    help="Streamline line width (default half of render.py's 1.4).")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    D = os.path.expanduser(args.run)
    idx = args.index
    mesh = uw.discretisation.Mesh(os.path.join(D, f"{RUN_NAME}.mesh.{idx:05d}.h5"))
    T = uw.discretisation.MeshVariable("T", mesh, 1, degree=3, continuous=True)
    T.read_timestep(data_filename=RUN_NAME, data_name="T", index=idx, outputPath=D)
    v = uw.discretisation.MeshVariable("v", mesh, mesh.dim, degree=2, continuous=True)
    v.read_timestep(data_filename=RUN_NAME, data_name="v", index=idx, outputPath=D)
    # Per-step feature azimuths (moving features) — fall back to manifest initial.
    tf, tm = _per_step_theta(D).get(idx, (None, None))
    fault_xyz, _ = fault_from_manifest(D, theta_deg=tf)
    ridge = ridge_from_manifest(D, theta_deg=tm)
    edges = vis.mesh_to_pv_mesh(mesh).extract_all_edges()

    # left: T + streamlines
    pvT = vis.meshVariable_to_pv_mesh_object(T)
    pvT.point_data["T"] = np.asarray(T.data[:, 0])
    pvV = vis.meshVariable_to_pv_mesh_object(v)
    Vd = np.asarray(v.data)
    vec = np.zeros((pvV.n_points, 3)); vec[:, 0], vec[:, 1] = Vd[:, 0], Vd[:, 1]
    pvV["V"] = vec; pvV.set_active_vectors("V")
    vmax = float(np.linalg.norm(Vd, axis=1).max())
    rng = np.linspace(0.55, 0.95, 4); th = np.linspace(0, 2 * np.pi, 14, endpoint=False)
    R, TH = np.meshgrid(rng, th)
    seed = pv.PolyData(np.c_[(R * np.cos(TH)).ravel(),
                             (R * np.sin(TH)).ravel(), np.zeros(R.size)])
    strm = pvV.streamlines_from_source(
        seed, vectors="V", integration_direction="both", max_step_length=0.5,
        initial_step_length=0.02, max_steps=300, terminal_speed=max(vmax * 1e-4, 1e-9))

    # right: smoothed strain-rate + mesh
    sr = smoothed_strainrate(mesh, v, args.smoothing)
    pvS = vis.meshVariable_to_pv_mesh_object(sr)
    srd = np.asarray(sr.data[:, 0])
    pvS.point_data["lsr"] = np.log10(np.clip(srd, 1e-6, None))
    clim = (float(np.percentile(pvS.point_data["lsr"], 45)),
            float(np.percentile(pvS.point_data["lsr"], 99)))

    pl = pv.Plotter(off_screen=True, window_size=(1760, 900), shape=(1, 2))
    pl.subplot(0, 0); pl.set_background("white")
    pl.add_mesh(pvT, scalars="T", cmap="RdBu_r", clim=(0, 1), show_edges=False,
                lighting=False, show_scalar_bar=False)        # no scale-bar, no mesh
    if strm.n_points > 0:
        pl.add_mesh(strm, color="black", line_width=args.strm_width, lighting=False)
    _add_fault(pl, fault_xyz)
    _add_ridge_marker(pl, ridge[0], ridge[1])
    pl.add_text(f"step {idx}: T + streamlines", font_size=11, color="black",
                position="upper_left")
    pl.view_xy(); pl.camera.zoom(1.3)

    pl.subplot(0, 1); pl.set_background("white")
    pl.add_mesh(pvS.copy(), scalars="lsr", cmap="YlOrRd", clim=clim,
                show_edges=False, lighting=False,
                scalar_bar_args={"title": "log10 strain-rate II", "color": "black"})
    pl.add_mesh(edges, color="black", line_width=0.4, opacity=0.5, lighting=False)
    _add_fault(pl, fault_xyz, color="lime")
    _add_ridge_marker(pl, ridge[0], ridge[1])
    pl.add_text(f"step {idx}: mesh + strain-rate", font_size=11, color="black",
                position="upper_left")
    pl.view_xy(); pl.camera.zoom(1.3)

    out = args.out or os.path.join(D, f"COMPARE_{idx:05d}.png")
    pl.screenshot(out); pl.close()
    print("->", out, flush=True)


if __name__ == "__main__":
    main()
