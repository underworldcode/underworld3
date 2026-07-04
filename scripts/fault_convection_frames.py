"""Render EVERY snapshot of a fault-convection run as a convection-style
frame (temperature in red-blue + the adapted mesh edges + fault trace), and
assemble them into a gif so the adaptation can be watched step by step.

Writes frame_stepNNNN.png for each snapshot and anim.gif into the run dir.
"""
from __future__ import annotations
import os, glob, re, argparse
import numpy as np
import underworld3 as uw
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation

ap = argparse.ArgumentParser()
ap.add_argument('--tag', type=str, default='fault_mmpde_res24')
ap.add_argument('--fault-dip-deg', type=float, default=30.0)
ap.add_argument('--fault-theta-deg', type=float, default=90.0)
ap.add_argument('--fault-depth', type=float, default=0.225)
ap.add_argument('--fps', type=float, default=3.0)
args = ap.parse_args()
DIR = os.path.expanduser(f'~/+Simulations/StagnantLid/{args.tag}')


def tris_of(mesh):
    dm = mesh.dm
    pS, pE = dm.getDepthStratum(0); cS, cE = dm.getHeightStratum(0)
    return np.asarray([[p - pS for p in dm.getTransitiveClosure(c)[0]
                        if pS <= p < pE] for c in range(cS, cE)])


def fault_trace():
    delta = np.deg2rad(args.fault_dip_deg); th0 = np.deg2rad(args.fault_theta_deg)
    P0 = np.array([np.cos(th0), np.sin(th0)])
    e_hat = np.array([np.cos(th0), np.sin(th0)]); t_hat = np.array([-np.sin(th0), np.cos(th0)])
    dhat = np.cos(delta) * t_hat - np.sin(delta) * e_hat
    s = np.linspace(0.0, args.fault_depth / np.sin(delta), 25)[:, None]
    return P0[None, :] + s * dhat[None, :]


xy = fault_trace()
cands = sorted(glob.glob(os.path.join(DIR, "step*.mesh.00000.h5")),
               key=lambda c: int(re.search(r"step(\d+)\.mesh", c).group(1)))
if not cands:
    raise SystemExit(f"no snapshots in {DIR}")

frames = []
for path in cands:
    label = re.search(r"(step\d+)\.mesh", os.path.basename(path)).group(1)
    mesh = uw.discretisation.Mesh(path)
    T = uw.discretisation.MeshVariable("T_v2p1", mesh, 1, degree=3, varsymbol="T")
    T.read_timestep(label, "T_v2p1", 0, outputPath=DIR)
    C, tris = np.asarray(mesh.X.coords), tris_of(mesh)
    tri = Triangulation(C[:, 0], C[:, 1], tris)
    Tv = np.asarray(uw.function.evaluate(T.sym[0], C)).reshape(-1)

    fig, ax = plt.subplots(1, 2, figsize=(11, 5.6))
    tpc = ax[0].tripcolor(tri, Tv, shading="gouraud", cmap="RdBu_r", vmin=0, vmax=1)
    ax[0].plot(xy[:, 0], xy[:, 1], "k-", lw=1.3)
    ax[0].set_title(f"T  {label}")
    ax[1].triplot(tri, color="0.35", lw=0.3)
    ax[1].plot(xy[:, 0], xy[:, 1], "r-", lw=1.6)
    ax[1].set_title("adapted mesh")
    for a in ax:
        a.set_aspect("equal"); a.axis("off")
    fig.tight_layout()
    fpath = os.path.join(DIR, f"frame_{label}.png")
    fig.savefig(fpath, dpi=110)
    plt.close(fig)
    frames.append(fpath)
    print(f"  {label}: T[{Tv.min():.2f},{Tv.max():.2f}] n={len(C)}", flush=True)

# assemble gif
gif = os.path.join(DIR, "anim.gif")
try:
    import imageio.v2 as imageio
    imgs = [imageio.imread(f) for f in frames]
    imageio.mimsave(gif, imgs, duration=1.0 / args.fps, loop=0)
    print(f"→ {gif} ({len(frames)} frames)")
except Exception as e:
    try:
        from PIL import Image
        imgs = [Image.open(f) for f in frames]
        imgs[0].save(gif, save_all=True, append_images=imgs[1:],
                     duration=int(1000 / args.fps), loop=0)
        print(f"→ {gif} ({len(frames)} frames, via PIL)")
    except Exception as e2:
        print(f"  gif assembly failed ({e}; {e2}); frames are in {DIR}")
print(f"frames: {len(frames)}  dir: {DIR}")
