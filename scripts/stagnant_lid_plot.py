"""Plot T-field snapshots + V arrows + Nu/vrms time series for a
stagnant-lid run produced by stagnant_lid_uniform.py.

Rendering: UW3 pyvista helpers (the project's standard for
high-order field viz) — high-order T on its own DOF cloud +
deformed-mesh edges overlay + add_arrows from V's DOF cloud,
white background, lighting off (per repo memory:
feedback_pyvista_viz_pattern.md).

Time-series scalars (Nu, vrms, η range) stay in matplotlib —
no benefit from pyvista there.
"""
from __future__ import annotations
import os
import glob
import re
import math
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import underworld3 as uw
import underworld3.visualisation as vis
import pyvista as pv


p = argparse.ArgumentParser()
p.add_argument('--run-dir', type=str,
               default=os.path.expanduser(
                   '~/+Simulations/StagnantLid/'
                   'uniform_res32_Ra1e6_dEta1e4'))
p.add_argument('--max-snapshots', type=int, default=8)
args = p.parse_args()

pv.OFF_SCREEN = True

run_dir = args.run_dir
tag = os.path.basename(run_dir.rstrip('/'))

hist_path = os.path.join(run_dir, f"sl_{tag}_history.npz")
if not os.path.exists(hist_path):
    raise SystemExit(f"history not found: {hist_path}")
H = np.load(hist_path)
print(f"history: {len(H['step'])} log entries, "
      f"step range {H['step'].min()}..{H['step'].max()}, "
      f"t_sim {H['t_sim'].min():.3f}..{H['t_sim'].max():.3f}")


# ---- Nu, vrms time series + η range (matplotlib) ---------------

fig, ax = plt.subplots(3, 1, figsize=(8, 9), sharex=True)
ax[0].plot(H['step'], H['Nu'], '-o', ms=3, lw=1.0)
ax[0].axhline(1.0, color='gray', ls=':', lw=0.7,
              label='pure conduction')
ax[0].set_ylabel('Nu  (mid-shell total flux)')
ax[0].grid(alpha=0.3)
ax[0].legend(loc='best', fontsize=9)

ax[1].semilogy(H['step'], H['vrms'], '-o', ms=3, lw=1.0)
ax[1].set_ylabel(r'$v_\mathrm{rms}$')
ax[1].grid(alpha=0.3, which='both')

ax[2].semilogy(H['step'], H['eta_max'], '-o', ms=3, lw=1.0,
               label=r'$\eta_\max$ (cold, lid)')
ax[2].semilogy(H['step'], H['eta_min'], '-s', ms=3, lw=1.0,
               label=r'$\eta_\min$ (hot, base)')
ax[2].set_ylabel(r'realised $\eta$ range')
ax[2].set_xlabel('step')
ax[2].grid(alpha=0.3, which='both')
ax[2].legend(loc='best', fontsize=9)
fig.suptitle(f"{tag}  —  t={H['t_sim'][-1]:.4f}", fontsize=11)
fig.tight_layout()
out_ts = os.path.join(run_dir, f"plot_{tag}_timeseries.png")
fig.savefig(out_ts, dpi=130, bbox_inches='tight')
print(f"  wrote {out_ts}")
plt.close(fig)


# ---- T snapshots + V arrows (pyvista) --------------------------

mesh_files = sorted(glob.glob(os.path.join(
    run_dir, f"sl_{tag}_step*.mesh.00000.h5")))
init_file = sorted(glob.glob(os.path.join(
    run_dir, f"sl_{tag}_init.mesh.00000.h5")))
pat = re.compile(r"sl_.+_step(\d+)\.mesh\.00000\.h5$")
entries = []
for f in init_file:
    entries.append((0, f))
for f in mesh_files:
    m = pat.search(os.path.basename(f))
    if m:
        entries.append((int(m.group(1)), f))
entries.sort(key=lambda e: e[0])

if not entries:
    print("no snapshots found, skipping field plots")
    raise SystemExit(0)

if len(entries) > args.max_snapshots:
    idx = np.linspace(0, len(entries) - 1,
                      args.max_snapshots).round().astype(int)
    entries = [entries[i] for i in idx]


# Pre-pass: find global |v|max so streamline tube width is
# comparable across panels — tubes scale with local |v| (radius
# proportional to magnitude), so the lid shows as thin lines and
# the active layer as fat ones.
print("  scanning |v|max across snapshots...", flush=True)
global_Vmax = 0.0
for (step, mfile) in entries:
    m_ = uw.discretisation.Mesh(mfile)
    V_ = uw.discretisation.MeshVariable(
        f"V_scan_{step}", m_, vtype=uw.VarType.VECTOR,
        degree=2, continuous=True)
    stem_ = os.path.basename(mfile)[:-len(".mesh.00000.h5")]
    V_.read_timestep(stem_, "V_v2p1", 0, outputPath=run_dir)
    vm_ = float(np.sqrt(V_.data[:, 0] ** 2
                        + V_.data[:, 1] ** 2).max())
    if vm_ > global_Vmax:
        global_Vmax = vm_
print(f"  global |v|max = {global_Vmax:.3e}", flush=True)
if global_Vmax <= 0:
    global_Vmax = 1.0

# Plot grid
n = len(entries)
ncol = min(3, n)
nrow = math.ceil(n / ncol)
pl = pv.Plotter(shape=(nrow, ncol), off_screen=True,
                window_size=(900 * ncol, 900 * nrow),
                border=False)
pl.set_background("white")

for k, (step, mfile) in enumerate(entries):
    rr, cc = divmod(k, ncol)
    m = uw.discretisation.Mesh(mfile)
    T = uw.discretisation.MeshVariable(
        f"T_view_{step}", m, vtype=uw.VarType.SCALAR,
        degree=3, continuous=True)
    V = uw.discretisation.MeshVariable(
        f"V_view_{step}", m, vtype=uw.VarType.VECTOR,
        degree=2, continuous=True)
    stem = os.path.basename(mfile)[:-len(".mesh.00000.h5")]
    T.read_timestep(stem, "T_v2p1", 0, outputPath=run_dir)
    V.read_timestep(stem, "V_v2p1", 0, outputPath=run_dir)

    # T on its DOF cloud (P3) — Delaunay-triangulated viz mesh
    pv_T = vis.meshVariable_to_pv_mesh_object(T)
    pv_T.point_data["T"] = np.asarray(T.data[:, 0])

    # Mesh edges for context
    edges = vis.mesh_to_pv_mesh(m).extract_all_edges()

    # Streamlines on a 2-D Cartesian ImageData (uniform grid) +
    # masked to the annulus. Pyvista's `.streamlines()` does the
    # integration; we get tube radius proportional to local |v|
    # so the lid shows as thin and the active layer as bold.
    n_cart = 220
    extent = 1.05
    xs = np.linspace(-extent, extent, n_cart)
    ys = np.linspace(-extent, extent, n_cart)
    Xc, Yc = np.meshgrid(xs, ys, indexing='xy')
    Rc = np.sqrt(Xc ** 2 + Yc ** 2)
    in_ann = (Rc > 0.51) & (Rc < 0.99)
    pts_eval = np.column_stack([Xc.ravel(), Yc.ravel()])
    bad = ~in_ann.ravel()
    if bad.any():
        th_proj = np.arctan2(pts_eval[bad, 1], pts_eval[bad, 0])
        pts_eval[bad, 0] = 0.75 * np.cos(th_proj)
        pts_eval[bad, 1] = 0.75 * np.sin(th_proj)
    Vx = np.asarray(uw.function.evaluate(
        V.sym[0], pts_eval)).reshape(-1)
    Vy = np.asarray(uw.function.evaluate(
        V.sym[1], pts_eval)).reshape(-1)
    Vx[bad] = 0.0
    Vy[bad] = 0.0
    # Build pyvista ImageData carrying the velocity
    img = pv.ImageData(
        dimensions=(n_cart, n_cart, 1),
        spacing=((2 * extent) / (n_cart - 1),
                 (2 * extent) / (n_cart - 1), 1.0),
        origin=(-extent, -extent, 0.0))
    Vvec3 = np.zeros((n_cart * n_cart, 3))
    Vvec3[:, 0] = Vx
    Vvec3[:, 1] = Vy
    img.point_data["V"] = Vvec3
    img.point_data["Vmag"] = np.sqrt(Vx ** 2 + Vy ** 2)
    img.set_active_vectors("V")
    # Seed points: a polar tile inside the annulus so every
    # azimuth gets coverage but seeds avoid the lid (where v≈0
    # gives stranded short streamlines).
    seed_r = np.linspace(0.55, 0.78, 4)
    seed_th = np.linspace(0, 2 * np.pi, 60, endpoint=False)
    sR, sT = np.meshgrid(seed_r, seed_th, indexing='ij')
    seed_pts = np.column_stack([
        (sR * np.cos(sT)).ravel(),
        (sR * np.sin(sT)).ravel(),
        np.zeros(sR.size)])
    seeds = pv.PolyData(seed_pts)
    streams = img.streamlines_from_source(
        seeds, vectors="V",
        integration_direction="both",
        max_step_length=0.02,
        compute_vorticity=False)
    Vmax_step = float(np.sqrt(V.data[:, 0] ** 2
                              + V.data[:, 1] ** 2).max())

    pl.subplot(rr, cc)
    pl.add_text(f"step {step}    "
                f"|v|max = {Vmax_step:.2e}",
                font_size=12, color="black")
    pl.add_mesh(pv_T, scalars="T", cmap="RdBu_r",
                clim=(0.0, 1.0), show_edges=False,
                lighting=False,
                show_scalar_bar=(k == n - 1),
                scalar_bar_args=dict(title="T", color="black"))
    pl.add_mesh(edges, color="#202020", line_width=0.6,
                lighting=False, opacity=0.4)
    if streams is not None and streams.n_points > 0:
        pl.add_mesh(streams, color="black",
                    line_width=2.0, opacity=0.5,
                    lighting=False)
    pl.view_xy()
    pl.camera.zoom(1.25)

out_T = os.path.join(run_dir, f"plot_{tag}_T_snapshots.png")
pl.screenshot(out_T)
pl.close()
print(f"  wrote {out_T}")
