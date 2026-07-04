"""Is mmpde's thermal saturation a NODE-BUDGET limit (Louis' claim), not an R
limit? Same real convection T, same metric (refinement=5), mmpde — at res16 vs a
DOUBLE-resolution mesh. If budget-bound, res32 grades harder (finer BL cells,
larger dynamic range) AND stays clean (no slivers).
"""
import os, numpy as np, underworld3 as uw, underworld3.visualisation as vis, pyvista as pv
from scipy.spatial import cKDTree

pv.OFF_SCREEN = True
SRC = os.path.expanduser('~/+Simulations/StagnantLid/uniform_res16_Ra1e7_dEta1e4')
OUT = os.path.expanduser('~/+Simulations/StagnantLid')
LABEL = "sl_uniform_res16_Ra1e7_dEta1e4_step00100"

# --- load the real convection T (res16) ---
m16 = uw.discretisation.Mesh(os.path.join(SRC, LABEL + ".mesh.00000.h5"))
T16 = uw.discretisation.MeshVariable("T_v2p1", m16, 1, degree=3, continuous=True, varsymbol="T")
T16.read_timestep(LABEL, "T_v2p1", 0, outputPath=SRC)
C = np.asarray(m16._coords)[:,:2]; rad = np.hypot(C[:,0], C[:,1])
R_o, R_i = float(rad.max()), float(rad.min())
tr = cKDTree(C); dd,_ = tr.query(C,k=2); h16 = float(np.median(dd[:,1]))
print("loaded res16: %d nodes, R=[%.3f,%.3f], median spacing %.4f" % (C.shape[0], R_i, R_o, h16), flush=True)

# --- res32: fresh annulus at HALF the spacing, same T interpolated on ---
m32 = uw.meshing.Annulus(radiusOuter=R_o, radiusInner=R_i, cellSize=h16/2, qdegree=3)
T32 = uw.discretisation.MeshVariable("T32", m32, 1, degree=3, continuous=True, varsymbol="T")
T32.data[:,0] = np.asarray(uw.function.evaluate(T16.sym, np.asarray(T32.coords))).reshape(-1)
print("built res32: %d nodes" % (m32._coords.shape[0],), flush=True)

def spacings(mesh):
    C = np.asarray(mesh._coords)[:,:2]; tr = cKDTree(C); dd,_ = tr.query(C,k=2); return dd[:,1]

def quality(mesh):       # min interior angle proxy via shortest/longest edge per cell -> sliver flag
    pvm = vis.mesh_to_pv_mesh(mesh); c = pvm.cells.reshape(-1,4)[:,1:]; p = pvm.points[:,:2]; v = p[c]
    e = np.stack([v[:,1]-v[:,0], v[:,2]-v[:,1], v[:,0]-v[:,2]], axis=1)
    L = np.linalg.norm(e, axis=2); ar = L.max(1)/np.maximum(L.min(1), 1e-12)   # aspect ratio
    a = 0.5*((v[:,1,0]-v[:,0,0])*(v[:,2,1]-v[:,0,1]) - (v[:,2,0]-v[:,0,0])*(v[:,1,1]-v[:,0,1]))
    return float(np.percentile(ar,99)), int((np.sign(a)!=np.sign(np.median(a))).sum())

def adapt(mesh, T):
    rho = uw.meshing.metric_density_from_gradient(mesh, T, refinement=5.0, coarsening="auto",
                                                  metric_choice="front-following", name=T.name)
    for _ in range(8):
        uw.meshing.smooth_mesh_interior(mesh, metric=rho, method="mmpde", skip_threshold=None,
            slip_surfaces=True, method_kwargs=dict(step_frac=0.2, accel="cg", momentum=0.0), verbose=False)

def report(mesh, tag):
    s = spacings(mesh); ar99, fold = quality(mesh)
    print("%s: %d nodes | finest p02=%.4f  median=%.4f  coarsest p98=%.4f | dyn-range p98/p02=%.2f | "
          "aspect p99=%.2f folded=%d" % (tag, mesh._coords.shape[0], np.percentile(s,2), np.median(s),
          np.percentile(s,98), np.percentile(s,98)/np.percentile(s,2), ar99, fold), flush=True)
    return np.percentile(s,2)

print("--- BEFORE adapt ---", flush=True); report(m16,"res16"); report(m32,"res32")
adapt(m16, T16); adapt(m32, T32)
print("--- AFTER mmpde (refinement=5) ---", flush=True)
f16 = report(m16,"res16"); f32 = report(m32,"res32")
print("finest-cell ratio res16/res32 = %.2f (budget-bound => res32 finer)" % (f16/f32), flush=True)

def snap(mesh,T):
    pvT = vis.meshVariable_to_pv_mesh_object(T); pvT.point_data["T"]=np.asarray(T.data[:,0])
    return pvT, vis.mesh_to_pv_mesh(mesh).extract_all_edges()
pl = pv.Plotter(off_screen=True, shape=(1,2), window_size=(2300,1200), border=False)
for i,(mesh,T,title) in enumerate([(m16,T16,"res16 + mmpde (%d nodes)"%m16._coords.shape[0]),
                                   (m32,T32,"res32 + mmpde (%d nodes)"%m32._coords.shape[0])]):
    pvT,edges = snap(mesh,T); pl.subplot(0,i); pl.set_background('white')
    pl.add_mesh(pvT, scalars="T", cmap="RdBu_r", clim=(0,1), lighting=False, show_scalar_bar=False, opacity=0.5)
    pl.add_mesh(edges, color="#111111", line_width=0.5, lighting=False)
    pl.add_text(title, font_size=12, color='black')
    pl.view_xy(); pl.enable_parallel_projection(); pl.camera.focal_point=(0,0,0); pl.camera.position=(0,0,1); pl.camera.zoom(0.97)
out = os.path.join(OUT,"thermal_budget.png"); pl.screenshot(out); pl.close(); print("->",out,flush=True)
print("DONE", flush=True)
