"""Render the passive-fault convection run: T (RdBu_r) + adaptive mesh + fault
trace, a montage across timesteps so the mesh tracking the developing convection
(and holding the inert fault) is visible.
"""
import os, glob, re, argparse, numpy as np
import underworld3 as uw, underworld3.visualisation as vis, pyvista as pv

pv.OFF_SCREEN = True
ap = argparse.ArgumentParser()
ap.add_argument("--tag", default="passive_run1")
ap.add_argument("--sim-dir", default="~/+Simulations/StagnantLid+Fault")
ap.add_argument("--dip", type=float, default=30.0)
ap.add_argument("--depth", type=float, default=0.15)
ap.add_argument("--theta", type=float, default=90.0)
ap.add_argument("--n", type=int, default=4, help="number of montage frames")
args = ap.parse_args()
D = os.path.expanduser(os.path.join(args.sim_dir, args.tag))

def fault_trace():
    d, th = np.deg2rad(args.dip), np.deg2rad(args.theta)
    P0 = np.array([np.cos(th), np.sin(th)]); e=np.array([np.cos(th),np.sin(th)]); t=np.array([-np.sin(th),np.cos(th)])
    dh = np.cos(d)*t - np.sin(d)*e
    return P0[None,:] + np.linspace(0, args.depth/np.sin(d), 30)[:,None]*dh[None,:]
XY = fault_trace()

cands = sorted(glob.glob(os.path.join(D, "step*.mesh.00000.h5")),
               key=lambda c: int(re.search(r"step(\d+)\.mesh", c).group(1)))
if not cands:
    raise SystemExit("no step snapshots yet in %s" % D)
labels = [re.search(r"(step\d+)\.mesh", os.path.basename(c)).group(1) for c in cands]
if len(labels) > args.n:
    idx = np.linspace(0, len(labels)-1, args.n).round().astype(int)
    labels = [labels[i] for i in idx]
print("frames:", labels, flush=True)

def load(label):
    mesh = uw.discretisation.Mesh(os.path.join(D, f"{label}.mesh.00000.h5"))
    T = uw.discretisation.MeshVariable("T_v2p1", mesh, 1, degree=3, continuous=True, varsymbol="T")
    T.read_timestep(label, "T_v2p1", 0, outputPath=D)
    pvT = vis.meshVariable_to_pv_mesh_object(T); pvT.point_data["T"]=np.asarray(T.data[:,0])
    return pvT, vis.mesh_to_pv_mesh(mesh).extract_all_edges(), mesh._coords.shape[0]

pl = pv.Plotter(off_screen=True, shape=(1,len(labels)), window_size=(900*len(labels),950), border=False)
for i,lab in enumerate(labels):
    pvT, edges, n = load(lab)
    pl.subplot(0,i); pl.set_background('white')
    pl.add_mesh(pvT, scalars="T", cmap="RdBu_r", clim=(0,1), lighting=False, show_scalar_bar=False, opacity=0.55)
    pl.add_mesh(edges, color="#111111", line_width=0.45, lighting=False)
    pl.add_mesh(pv.lines_from_points(np.column_stack([XY, np.zeros(len(XY))])), color="#00c000", line_width=4, lighting=False)
    pl.add_text("%s (%d nodes)" % (lab, n), font_size=11, color='black')
    pl.view_xy(); pl.enable_parallel_projection(); pl.camera.focal_point=(0,0,0); pl.camera.position=(0,0,1); pl.camera.zoom(0.95)
out = os.path.join(D, "passive_montage.png"); pl.screenshot(out); pl.close()
print("->", out, flush=True)
