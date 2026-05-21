"""Take the step-125 settled state of the Ra=1e7 Δη=1e4 stagnant
lid uniform-mesh run and adapt the mesh to the |∇T| metric — the
catalogue's equidist mover at resolution_ratio=1.5 (validated
production setting). Remap T/V/P onto the adapted nodes via FE
evaluate (the validated local-FE remap from adaptive_saturation
.py:adapt_local_fe_interp). Save the adapted state so we can use
it as side (b) of the Stokes-solver preset sweep.

Output: ~/+Simulations/StagnantLid/adapted_R15_Ra1e7_dEta1e4/
  - adapted.mesh.{T,V,P}.00000.h5   — fields on the graded mesh
  - plot_adapt_compare.png          — uniform vs adapted side-by-side
"""
from __future__ import annotations
import os
import sys
import time
import argparse
import numpy as np
import sympy

import underworld3 as uw
import underworld3.visualisation as vis
import pyvista as pv

pv.OFF_SCREEN = True


_t0 = time.time()
_tprev = _t0


def _stage(label):
    global _tprev
    t = time.time()
    dt = t - _tprev
    tot = t - _t0
    print(f"  [{tot:6.2f}s | +{dt:5.2f}s]  {label}", flush=True)
    _tprev = t


p = argparse.ArgumentParser()
p.add_argument('--src-dir', type=str,
               default=os.path.expanduser(
                   '~/+Simulations/StagnantLid/'
                   'uniform_res16_Ra1e7_dEta1e4'))
p.add_argument('--src-step', type=int, default=125)
p.add_argument('--out-dir', type=str,
               default=os.path.expanduser(
                   '~/+Simulations/StagnantLid/'
                   'adapted_R15_Ra1e7_dEta1e4'))
p.add_argument('--resolution-ratio', type=float, default=1.5)
p.add_argument('--amp', type=float, default=8.0)
p.add_argument('--lo-pct', type=float, default=50.0)
p.add_argument('--hi-pct', type=float, default=97.0)
p.add_argument('--n-outer', type=int, default=12)
p.add_argument('--relax', type=float, default=0.2)
args = p.parse_args()


# ---------------------- locate the source snapshot --------------

src_dir = args.src_dir
src_tag = os.path.basename(src_dir.rstrip('/'))
stem = f"sl_{src_tag}_step{args.src_step:05d}"
mesh_path = os.path.join(src_dir, f"{stem}.mesh.00000.h5")
if not os.path.exists(mesh_path):
    sys.exit(f"missing snapshot mesh: {mesh_path}")
print(f"loading {stem} from {src_dir}")

mesh = uw.discretisation.Mesh(mesh_path)
_stage("mesh load")

# Re-create MeshVariables matching the saved layout
# (T degree=3 scalar; V degree=2 vector; P degree=1 scalar)
T = uw.discretisation.MeshVariable(
    "T_v2p1", mesh, vtype=uw.VarType.SCALAR,
    degree=3, continuous=True, varsymbol="T")
V = uw.discretisation.MeshVariable(
    "V_v2p1", mesh, vtype=uw.VarType.VECTOR,
    degree=2, continuous=True, varsymbol=r"\mathbf{v}")
P = uw.discretisation.MeshVariable(
    "P_v2p1", mesh, vtype=uw.VarType.SCALAR,
    degree=1, continuous=True, varsymbol="p")

T.read_timestep(stem, "T_v2p1", 0, outputPath=src_dir)
V.read_timestep(stem, "V_v2p1", 0, outputPath=src_dir)
P.read_timestep(stem, "P_v2p1", 0, outputPath=src_dir)
_stage("field load (T,V,P)")

print(f"  loaded: T=[{T.data.min():.3f},{T.data.max():.3f}], "
      f"|v|max={float(np.sqrt(V.data[:,0]**2 + V.data[:,1]**2).max()):.2e}")


# ---------------------- capture uniform mesh for the compare ----

uniform_X = np.asarray(mesh.X.coords).copy()
uniform_T = np.asarray(T.data).copy()
uniform_V = np.asarray(V.data).copy()
uniform_P = np.asarray(P.data).copy()


# ---------------------- build metric + adapt --------------------

print(f"building ρ ∝ |∇T| metric "
      f"(amp={args.amp}, pct=[{args.lo_pct:.0f},{args.hi_pct:.0f}])")
rho = uw.meshing.metric_density_from_gradient(
    mesh, T, amp=args.amp,
    lo_percentile=args.lo_pct, hi_percentile=args.hi_pct,
    name="sl_adapt")
_stage("metric build (|∇T| projection + percentile-normalise)")

print(f"adapting (anisotropic, resolution_ratio={args.resolution_ratio}, "
      f"n_outer={args.n_outer}, relax={args.relax})")
old_X = uniform_X.copy()
old_T = uniform_T.copy()
old_V = uniform_V.copy()
old_P = uniform_P.copy()

uw.meshing.smooth_mesh_interior(
    mesh, metric=rho, method="anisotropic",
    method_kwargs=dict(
        resolution_ratio=args.resolution_ratio,
        relax=args.relax, n_outer=args.n_outer))
_stage("mover (Winslow anisotropic MMPDE)")

new_X = np.asarray(mesh.X.coords).copy()
new_Tx = np.asarray(T.coords).copy()
new_Vx = np.asarray(V.coords).copy()
new_Px = np.asarray(P.coords).copy()


# ---------------------- local-FE remap T, V, P ------------------
# Per the catalogue's adapt_local_fe_interp: restore field on the
# OLD geometry, FE-evaluate at NEW dof positions, write onto the
# NEW geometry. (Topology-preserving ⇒ FE basis is the same; we
# just sample the old solution at the new physical points.)

mesh._deform_mesh(old_X)
T.data[...] = old_T
V.data[...] = old_V
P.data[...] = old_P

remap_T = np.asarray(uw.function.evaluate(
    T.sym[0], new_Tx)).reshape(-1)
remap_V = np.asarray(uw.function.evaluate(V.sym, new_Vx))
remap_P = np.asarray(uw.function.evaluate(
    P.sym[0], new_Px)).reshape(-1)

mesh._deform_mesh(new_X)
T.data[:, 0] = remap_T
V.data[...] = remap_V.reshape(V.data.shape)
P.data[:, 0] = remap_P
_stage("FE remap T,V,P → new DOF coords")

print(f"  after remap: T=[{T.data.min():.3f},{T.data.max():.3f}], "
      f"|v|max={float(np.sqrt(V.data[:,0]**2 + V.data[:,1]**2).max()):.2e}")


# ---------------------- save adapted snapshot -------------------

os.makedirs(args.out_dir, exist_ok=True)
mesh.write_timestep(
    filename="adapted", index=0, outputPath=args.out_dir,
    meshVars=[T, V, P], meshUpdates=True, create_xdmf=True)
_stage("write adapted snapshot to disk")
print(f"saved adapted snapshot to {args.out_dir}")


# ---------------------- side-by-side comparison plot ------------

print("rendering uniform vs adapted comparison...")
pl = pv.Plotter(shape=(1, 2), off_screen=True,
                window_size=(1800, 900), border=False)
pl.set_background("white")

# Compute |v|max once (same field on both panels)
Vmax = float(np.sqrt(V.data[:, 0] ** 2
                     + V.data[:, 1] ** 2).max())

# Subplot 0: uniform mesh + T (restore uniform geometry to render)
mesh._deform_mesh(old_X)
T.data[...] = old_T
V.data[...] = old_V

pv_T_unif = vis.meshVariable_to_pv_mesh_object(T)
pv_T_unif.point_data["T"] = np.asarray(T.data[:, 0])
edges_unif = vis.mesh_to_pv_mesh(mesh).extract_all_edges()

pl.subplot(0, 0)
pl.add_text(f"uniform res-16  (step {args.src_step})",
            font_size=14, color="black")
pl.add_mesh(pv_T_unif, scalars="T", cmap="RdBu_r",
            clim=(0.0, 1.0), show_edges=False, lighting=False,
            show_scalar_bar=False)
pl.add_mesh(edges_unif, color="#202020", line_width=0.7,
            lighting=False, opacity=0.55)
pl.view_xy()
pl.camera.zoom(1.25)

# Subplot 1: adapted mesh + remapped T
mesh._deform_mesh(new_X)
T.data[:, 0] = remap_T
V.data[...] = remap_V.reshape(V.data.shape)

pv_T_adapt = vis.meshVariable_to_pv_mesh_object(T)
pv_T_adapt.point_data["T"] = np.asarray(T.data[:, 0])
edges_adapt = vis.mesh_to_pv_mesh(mesh).extract_all_edges()

pl.subplot(0, 1)
pl.add_text(f"adapted (R={args.resolution_ratio}, "
            f"|∇T| metric, amp={args.amp}, "
            f"pct={args.lo_pct:.0f}/{args.hi_pct:.0f})",
            font_size=14, color="black")
pl.add_mesh(pv_T_adapt, scalars="T", cmap="RdBu_r",
            clim=(0.0, 1.0), show_edges=False, lighting=False,
            show_scalar_bar=True,
            scalar_bar_args=dict(title="T", color="black"))
pl.add_mesh(edges_adapt, color="#202020", line_width=0.7,
            lighting=False, opacity=0.55)
pl.view_xy()
pl.camera.zoom(1.25)

out_png = os.path.join(args.out_dir, "plot_adapt_compare.png")
pl.screenshot(out_png)
pl.close()
print(f"saved {out_png}")


# ---------------------- mesh stats ------------------------------

# Quick mesh-quality numbers
from underworld3.meshing.smoothing import _tri_cells, _signed_areas
tris = _tri_cells(mesh.dm)
A = np.abs(_signed_areas(np.asarray(mesh.X.coords), tris))
A_uniform = np.abs(_signed_areas(uniform_X, _tri_cells(mesh.dm)))
print(f"mesh stats:")
print(f"  uniform : minA/meanA = {A_uniform.min()/A_uniform.mean():.3f}, "
      f"A range [{A_uniform.min():.2e}, {A_uniform.max():.2e}]")
print(f"  adapted : minA/meanA = {A.min()/A.mean():.3f}, "
      f"A range [{A.min():.2e}, {A.max():.2e}], "
      f"max/min = {A.max()/A.min():.2f}")
