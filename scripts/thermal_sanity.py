"""SANITY CHECK: real convection T, NO faults. Does the thermal metric refine the
actual thermal structure (boundary layers + up/downwelling sheets)?

Three panels, same real Ra=1e7 developed-convection T (refinement=5 metric,
front-following — the driver settings):
  1. uniform mesh (as loaded)
  2. mmpde (FIXED topology) — caps ~2.4x (the create-from-uniform ceiling)
  3. OT-reset (mesh.OT_adapt, re-mesh) — the PRODUCTION mover, reaches full grading
Measured dynamic range (p95/p05 of nearest-neighbour spacing) printed per panel.
"""
import os, numpy as np, underworld3 as uw, underworld3.visualisation as vis, pyvista as pv
from scipy.spatial import cKDTree

pv.OFF_SCREEN = True
SRC = os.path.expanduser('~/+Simulations/StagnantLid/uniform_res16_Ra1e7_dEta1e4')
OUT = os.path.expanduser('~/+Simulations/StagnantLid')
LABEL = "sl_uniform_res16_Ra1e7_dEta1e4_step00100"

def load():
    mesh = uw.discretisation.Mesh(os.path.join(SRC, LABEL + ".mesh.00000.h5"))
    T = uw.discretisation.MeshVariable("T_v2p1", mesh, 1, degree=3, continuous=True, varsymbol="T")
    T.read_timestep(LABEL, "T_v2p1", 0, outputPath=SRC)
    return mesh, T

def dyn_range(mesh):
    C = np.asarray(mesh._coords)[:,:2]; tr = cKDTree(C); dd,_ = tr.query(C,k=2); nn = dd[:,1]
    return np.percentile(nn,95)/np.percentile(nn,5)

def snap(mesh, T, title):
    pvT = vis.meshVariable_to_pv_mesh_object(T); pvT.point_data["T"] = np.asarray(T.data[:,0])
    return (pvT, vis.mesh_to_pv_mesh(mesh).extract_all_edges(),
            "%s  (range %.1fx, %d nodes)" % (title, dyn_range(mesh), mesh._coords.shape[0]))

panels = []
# 1. uniform
mesh, T = load(); panels.append(snap(mesh, T, "uniform (as loaded)"))
print("uniform dyn-range %.2f" % dyn_range(mesh), flush=True)

# 2. mmpde fixed-topology
rho = uw.meshing.metric_density_from_gradient(mesh, T, refinement=5.0, coarsening="auto",
                                              metric_choice="front-following", name="mmpde")
for _ in range(8):
    uw.meshing.smooth_mesh_interior(mesh, metric=rho, method="mmpde", skip_threshold=None,
        slip_surfaces=True, method_kwargs=dict(step_frac=0.2, accel="cg", momentum=0.0), verbose=False)
panels.append(snap(mesh, T, "mmpde (fixed topology)"))
print("mmpde  dyn-range %.2f" % dyn_range(mesh), flush=True)

# 3. OT-reset (production mover, re-mesh)
mesh2, T2 = load()
moved = mesh2.OT_adapt(T2, refinement=5.0, coarsening="auto", metric_choice="front-following",
                       skip_threshold=None, verbose=False)
panels.append(snap(mesh2, T2, "OT-reset (production)"))
print("OT     dyn-range %.2f (moved=%s)" % (dyn_range(mesh2), moved), flush=True)

pl = pv.Plotter(off_screen=True, shape=(1,3), window_size=(3300,1150), border=False)
for i,(pvT,edges,title) in enumerate(panels):
    pl.subplot(0,i); pl.set_background('white')
    pl.add_mesh(pvT, scalars="T", cmap="RdBu_r", clim=(0,1), lighting=False, show_scalar_bar=False, opacity=0.55)
    pl.add_mesh(edges, color="#111111", line_width=0.6, lighting=False)
    pl.add_text(title, font_size=11, color='black')
    pl.view_xy(); pl.enable_parallel_projection(); pl.camera.focal_point=(0,0,0); pl.camera.position=(0,0,1); pl.camera.zoom(0.97)
out = os.path.join(OUT, "thermal_sanity.png"); pl.screenshot(out); pl.close(); print("->", out, flush=True)
print("DONE", flush=True)
