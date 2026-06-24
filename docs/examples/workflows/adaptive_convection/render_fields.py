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
    """Read Δη, the fault floor and the lid-blob params from the run manifest."""
    import yaml
    p = os.path.join(D, "manifest.yaml")
    snap = {}
    if os.path.exists(p):
        snap = (yaml.safe_load(open(p)) or {}).get("config_snapshot", {})
    delta_eta = float(snap.get("delta_eta", 1.0e3))
    floor = float(snap.get("fault_floor", 1.0))
    blob = None
    if snap.get("blob_enable"):
        blob = dict(theta=float(snap.get("blob_theta_deg", 0.0)),
                    radius=float(snap.get("blob_radius", 0.92)),
                    size=float(snap.get("blob_size", 0.05)),
                    edge=float(snap.get("blob_edge", 0.012)),
                    floor=float(snap.get("blob_floor", 1.0)))
    return delta_eta, floor, blob


def _blob_sym(X, blob):
    """Smooth analytic (x,y) box ≈1 inside the blob — mirrors fault_config."""
    th0 = float(np.deg2rad(blob["theta"]))
    x0, y0 = blob["radius"] * np.cos(th0), blob["radius"] * np.sin(th0)
    hw, e = blob["size"], blob["edge"]
    half = sympy.Rational(1, 2)
    bx = (half * (1 + sympy.tanh((X[0] - (x0 - hw)) / e))
          * half * (1 + sympy.tanh(((x0 + hw) - X[0]) / e)))
    by = (half * (1 + sympy.tanh((X[1] - (y0 - hw)) / e))
          * half * (1 + sympy.tanh(((y0 + hw) - X[1]) / e)))
    return bx * by


def _eval_fields(D, index, delta_eta, floor, blob=None):
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
    # isotropic lid blob (if present): weaken eta_weak only (NOT the eta_FK
    # background denominator, else the ratio cancels it) so the blob shows up
    # in the weakening panel alongside the fault.
    if blob is not None:
        b = _blob_sym(X, blob)
        eta_weak = eta_weak ** (1.0 - b) * float(blob["floor"]) ** b
    # background-removed: the multiplicative weakening (=1 off-fault, dips in
    # the band) — isolates the weak zones from the radial T-viscosity gradient.
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


def render(D, index, *, delta_eta, floor, blob=None, fault_xyz=None, sr_clim=None,
           eta_clim=None, focus=None):
    pv_mesh, sr, ratio, edges, vmax = _eval_fields(D, index, delta_eta, floor, blob)
    log_sr = np.log10(np.clip(sr, 1e-6, None))
    # weakening as a POSITIVE quantity: log10(η_FK/η_weak) ≥ 0, 0 = no fault,
    # large = strongly weakened — so "more = darker" on a light sequential map,
    # consistent with the strain-rate panel.
    log_weak = -np.log10(np.clip(ratio, 1e-12, None))
    pv_mesh.point_data["log_sr"] = log_sr
    pv_mesh.point_data["log_weak"] = log_weak

    # strain-rate clim focused on the LID band (plumes saturate) so the fault
    # localization is visible rather than washed out by the hot boundary layers.
    sr_clim = sr_clim or (float(np.percentile(log_sr, 45)),
                          float(np.percentile(log_sr, 98)))
    eta_clim = eta_clim or (0.0, float(np.percentile(log_weak, 99.5)))

    pl = pv.Plotter(off_screen=True, window_size=(1700, 900), shape=(1, 2))
    # light-background sequential maps (white at the low end) so black text and
    # the fault overlay read cleanly; a bright fault line contrasts on both.
    for col, (scal, clim, cmap, title) in enumerate([
            ("log_sr", sr_clim, "YlOrRd", "log10 strain-rate II (lid-scaled)"),
            ("log_weak", eta_clim, "Blues", "log10 η_FK/η_weak (fault weakening)")]):
        pl.subplot(0, col)
        pl.set_background("white")
        pl.add_mesh(pv_mesh.copy(), scalars=scal, cmap=cmap, clim=clim,
                    show_edges=False, lighting=False,
                    scalar_bar_args={"title": title, "color": "black",
                                     "title_font_size": 16, "label_font_size": 13})
        pl.add_mesh(edges, color="grey", line_width=0.3, lighting=False, opacity=0.25)
        if fault_xyz is not None:
            _add_fault(pl, fault_xyz, color="lime", line_width=2.5)
        pl.add_text(f"step {index}  ({title.split(' ',1)[1]})",
                    font_size=11, color="black", position="upper_left")
        pl.view_xy()
        if focus is not None:
            cx, cy, hw = focus
            pl.camera.parallel_projection = True
            pl.camera.parallel_scale = hw
            pl.camera.focal_point = (cx, cy, 0)
            pl.camera.position = (cx, cy, 1)
        else:
            pl.camera.zoom(1.3)
    tag = "FIELDSfocus" if focus is not None else "FIELDS"
    out = os.path.join(D, f"{tag}_{index:05d}.png")
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
    ap.add_argument("--focus-fault", type=float, default=None, metavar="HALFWIDTH",
                    help="Crop centred on the fault midpoint at this half-width.")
    ap.add_argument("--focus", type=float, nargs=3, default=None,
                    metavar=("CX", "CY", "HALFWIDTH"), help="Crop on a point.")
    ap.add_argument("--sr-clim", type=float, nargs=2, default=None,
                    help="Fixed log10 strain-rate colour limits (else per-frame).")
    ap.add_argument("--eta-clim", type=float, nargs=2, default=None,
                    help="Fixed log10 viscosity colour limits (else per-frame).")
    args = ap.parse_args(argv)

    D = os.path.expanduser(args.run)
    delta_eta, floor, blob = _manifest_params(D)
    fault_xyz, fault_mid = (fault_from_manifest(D)
                            if (args.fault or args.focus_fault) else (None, None))
    focus = ((fault_mid[0], fault_mid[1], args.focus_fault)
             if (args.focus_fault and fault_mid is not None)
             else (tuple(args.focus) if args.focus else None))

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
        render(D, idx, delta_eta=delta_eta, floor=floor, blob=blob,
               fault_xyz=fault_xyz, focus=focus,
               sr_clim=tuple(args.sr_clim) if args.sr_clim else None,
               eta_clim=tuple(args.eta_clim) if args.eta_clim else None)


if __name__ == "__main__":
    main()
