"""Render reconstructed strain-rate and viscosity for a fault-convection run.

The fault run checkpoints ``[v, T, p, gfac]`` (gfac = the fault influence
field ``f`` ∈ [0,1], 1 on the fault). The viscosity and strain rate are
NOT saved, but both reconstruct from those fields:

  * strain-rate 2nd invariant  ``ε̇_II = sqrt(½ ε̇:ε̇)``  from ``v``
  * effective viscosity        ``η = η_FK·(1−f) + floor·f``  from ``T``, ``f``
    where ``η_FK = exp(ln Δη · (1−T))`` is the Frank-Kamenetskii field. This
    is ``shear_viscosity_1`` (the fault-parallel shear viscosity) for the TI
    model — the channel through which the weak fault acts.

A weak fault shows as a LOW-viscosity stripe cutting the cold lid and (if it
is actively localizing) a strain-rate CONCENTRATION along the trace. Both
panels are drawn log-scaled with the fault trace overlaid.

Usage:
  python render_fields.py --run <dir> --index 250 --fault
  python render_fields.py --run <dir> --indices 50 100 150 200 250 --fault
  python render_fields.py --run <dir> --all --fault
"""
import os
import re
import glob
import argparse

import numpy as np
import sympy
import underworld3 as uw
import underworld3.visualisation as vis
import pyvista as pv

from render import fault_from_manifest, _add_fault  # reuse the trace overlay

pv.OFF_SCREEN = True
RUN_NAME = "run"


def _manifest_params(D):
    """Read Δη and the fault floor from the run manifest (defaults if absent)."""
    import yaml
    p = os.path.join(D, "manifest.yaml")
    snap = {}
    if os.path.exists(p):
        snap = (yaml.safe_load(open(p)) or {}).get("config_snapshot", {})
    delta_eta = float(snap.get("delta_eta", 1.0e3))
    floor = float(snap.get("fault_floor", 1.0))
    return delta_eta, floor


def _eval_fields(D, index, delta_eta, floor):
    """Load a checkpoint and evaluate (ε̇_II, η) on a P1 field's nodes.

    Returns (pv_mesh, sr, eta, edges, vmax) ready for plotting."""
    mesh = uw.discretisation.Mesh(os.path.join(D, f"{RUN_NAME}.mesh.{index:05d}.h5"))
    v = uw.discretisation.MeshVariable("v", mesh, mesh.dim, degree=2, continuous=True)
    v.read_timestep(data_filename=RUN_NAME, data_name="v", index=index, outputPath=D)
    T = uw.discretisation.MeshVariable("T", mesh, 1, degree=3, continuous=True)
    T.read_timestep(data_filename=RUN_NAME, data_name="T", index=index, outputPath=D)
    gfac = uw.discretisation.MeshVariable("eta_fac", mesh, 1, degree=1, continuous=True)
    gfac.read_timestep(data_filename=RUN_NAME, data_name="eta_fac", index=index, outputPath=D)

    X = mesh.CoordinateSystem.X
    # symmetric velocity gradient and its 2nd invariant sqrt(1/2 e:e)
    E = sympy.Matrix(2, 2, lambda i, j:
                     sympy.Rational(1, 2) * (v.sym[i].diff(X[j]) + v.sym[j].diff(X[i])))
    Einv2 = sympy.sqrt(sympy.Rational(1, 2) *
                       (E[0, 0] ** 2 + E[1, 1] ** 2 + 2 * E[0, 1] ** 2))
    eta_FK = sympy.exp(float(np.log(delta_eta)) * (1 - T.sym[0]))
    # geometric blend — matches fault_config.create_solvers (η_1=η_FK^(1−f)·floor^f)
    eta_weak = eta_FK ** (1.0 - gfac.sym[0]) * floor ** gfac.sym[0]
    # background-removed: the multiplicative weakening (=1 off-fault, dips in
    # the band) — isolates the fault from the radial T-viscosity gradient.
    eta_ratio = eta_weak / eta_FK

    # evaluate onto a P1 carrier so we get a clean DOF-faithful pv mesh
    out = uw.discretisation.MeshVariable("o", mesh, 1, degree=1, continuous=True)
    coords = np.asarray(out.coords)
    sr = np.asarray(uw.function.evaluate(Einv2, coords)).reshape(-1)
    ratio = np.asarray(uw.function.evaluate(eta_ratio, coords)).reshape(-1)
    vmax = float(np.linalg.norm(np.asarray(v.data), axis=1).max())

    pv_mesh = vis.meshVariable_to_pv_mesh_object(out)
    edges = vis.mesh_to_pv_mesh(mesh).extract_all_edges()
    return pv_mesh, sr, ratio, edges, vmax


def render(D, index, *, delta_eta, floor, fault_xyz=None, sr_clim=None,
           eta_clim=None):
    pv_mesh, sr, ratio, edges, vmax = _eval_fields(D, index, delta_eta, floor)
    log_sr = np.log10(np.clip(sr, 1e-6, None))
    log_ratio = np.log10(np.clip(ratio, 1e-6, None))   # ≤0; 0 = no fault
    pv_mesh.point_data["log_sr"] = log_sr
    pv_mesh.point_data["log_ratio"] = log_ratio

    # strain-rate clim focused on the LID band (plumes saturate) so the fault
    # localization is visible rather than washed out by the hot boundary layers.
    sr_clim = sr_clim or (float(np.percentile(log_sr, 45)),
                          float(np.percentile(log_sr, 98)))
    eta_clim = eta_clim or (float(np.percentile(log_ratio, 0.5)), 0.0)

    pl = pv.Plotter(off_screen=True, window_size=(1700, 900), shape=(1, 2))
    for col, (scal, clim, cmap, title) in enumerate([
            ("log_sr", sr_clim, "inferno", "log10 strain-rate II (lid-scaled)"),
            ("log_ratio", eta_clim, "viridis_r", "log10 η_weak/η_FK (fault weakening)")]):
        pl.subplot(0, col)
        pl.set_background("white")
        pl.add_mesh(pv_mesh.copy(), scalars=scal, cmap=cmap, clim=clim,
                    show_edges=False, lighting=False,
                    scalar_bar_args={"title": title, "color": "black"})
        pl.add_mesh(edges, color="grey", line_width=0.3, lighting=False, opacity=0.3)
        if fault_xyz is not None:
            _add_fault(pl, fault_xyz)
        pl.add_text(f"step {index}", font_size=9, color="black")
        pl.view_xy()
        pl.camera.zoom(1.3)
    out = os.path.join(D, f"FIELDS_{index:05d}.png")
    pl.screenshot(out)
    pl.close()
    print("->", out, flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--index", type=int, default=None)
    ap.add_argument("--indices", type=int, nargs="+", default=None)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--fault", action="store_true")
    ap.add_argument("--sr-clim", type=float, nargs=2, default=None,
                    help="Fixed log10 strain-rate colour limits (else per-frame).")
    ap.add_argument("--eta-clim", type=float, nargs=2, default=None,
                    help="Fixed log10 viscosity colour limits (else per-frame).")
    args = ap.parse_args(argv)

    D = os.path.expanduser(args.run)
    delta_eta, floor = _manifest_params(D)
    fault_xyz, _ = fault_from_manifest(D) if args.fault else (None, None)

    if args.all:
        xdmfs = sorted(glob.glob(os.path.join(D, f"{RUN_NAME}.mesh.[0-9]*.xdmf")),
                       key=lambda c: int(re.search(r"mesh\.(\d+)\.xdmf", c).group(1)))
        sel = [int(re.search(r"mesh\.(\d+)\.xdmf", os.path.basename(c)).group(1))
               for c in xdmfs]
    elif args.indices is not None:
        sel = args.indices
    elif args.index is not None:
        sel = [args.index]
    else:
        sel = []

    for idx in sel:
        render(D, idx, delta_eta=delta_eta, floor=floor,
               fault_xyz=fault_xyz,
               sr_clim=tuple(args.sr_clim) if args.sr_clim else None,
               eta_clim=tuple(args.eta_clim) if args.eta_clim else None)


if __name__ == "__main__":
    main()
