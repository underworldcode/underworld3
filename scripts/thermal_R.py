"""res32, same real convection T, mmpde — dial R: refinement=5 vs refinement=10.
Does doubling the metric's target grading change what mmpde achieves, or does it
saturate the same (and does it stay clean)?
"""
import os, numpy as np, underworld3 as uw, underworld3.visualisation as vis, pyvista as pv
from scipy.spatial import cKDTree

pv.OFF_SCREEN = True
SRC = os.path.expanduser('~/+Simulations/StagnantLid/uniform_res16_Ra1e7_dEta1e4')
OUT = os.path.expanduser('~/+Simulations/StagnantLid')
LABEL = "sl_uniform_res16_Ra1e7_dEta1e4_step00100"

m16 = uw.discretisation.Mesh(os.path.join(SRC, LABEL + ".mesh.00000.h5"))
T16 = uw.discretisation.MeshVariable("T_v2p1", m16, 1, degree=3, continuous=True, varsymbol="T")
T16.read_timestep(LABEL, "T_v2p1", 0, outputPath=SRC)
C = np.asarray(m16._coords)[:,:2]; rad = np.hypot(C[:,0],C[:,1]); R_o,R_i = float(rad.max()),float(rad.min())
tr = cKDTree(C); dd,_ = tr.query(C,k=2); h16 = float(np.median(dd[:,1]))

def build_res32():
    m = uw.meshing.Annulus(radiusOuter=R_o, radiusInner=R_i, cellSize=h16/2, qdegree=3)
    T = uw.discretisation.MeshVariable("Tr"+str(id(m))[-4:], m, 1, degree=3, continuous=True, varsymbol="T")
    T.data[:,0] = np.asarray(uw.function.evaluate(T16.sym, np.asarray(T.coords))).reshape(-1)
    return m, T

def adapt(mesh, T, R):
    rho = uw.meshing.metric_density_from_gradient(mesh, T, refinement=float(R), coarsening="auto",
                                                  metric_choice="front-following", name=T.name)
    print("  R=%g rho range [%.2f, %.2f]" % (R,
        float(uw.function.evaluate(rho, np.asarray(mesh._coords)).min()),
        float(uw.function.evaluate(rho, np.asarray(mesh._coords)).max())), flush=True)
    for _ in range(10):
        uw.meshing.smooth_mesh_interior(mesh, metric=rho, method="mmpde", skip_threshold=None,
            slip_surfaces=True, method_kwargs=dict(step_frac=0.2, accel="cg", momentum=0.0), verbose=False)

def report(mesh, tag):
    C = np.asarray(mesh._coords)[:,:2]; tr = cKDTree(C); dd,_ = tr.query(C,k=2); s = dd[:,1]
    pvm = vis.mesh_to_pv_mesh(mesh); c = pvm.cells.reshape(-1,4)[:,1:]; p = pvm.points[:,:2]; v = p[c]
    e = np.stack([v[:,1]-v[:,0], v[:,2]-v[:,1], v[:,0]-v[:,2]], axis=1); L = np.linalg.norm(e,axis=2)
    ar = L.max(1)/np.maximum(L.min(1),1e-12)
    a = 0.5*((v[:,1,0]-v[:,0,0])*(v[:,2,1]-v[:,0,1]) - (v[:,2,0]-v[:,0,0])*(v[:,1,1]-v[:,0,1]))
    print("%s: finest p02=%.4f median=%.4f coarsest p98=%.4f | grading=%.2f | aspect p99=%.2f p999=%.2f folded=%d"
          % (tag, np.percentile(s,2), np.median(s), np.percentile(s,98), np.percentile(s,98)/np.percentile(s,2),
             np.percentile(ar,99), np.percentile(ar,99.9), int((np.sign(a)!=np.sign(np.median(a))).sum())), flush=True)

results = []
for R in (5, 10):
    m, T = build_res32(); print("R=%g:" % R, flush=True); adapt(m, T, R); report(m, "  res32 R=%g"%R)
    results.append((m, T, R))

pl = pv.Plotter(off_screen=True, shape=(1,2), window_size=(2300,1200), border=False)
for i,(mesh,T,R) in enumerate(results):
    pvT = vis.meshVariable_to_pv_mesh_object(T); pvT.point_data["T"]=np.asarray(T.data[:,0])
    edges = vis.mesh_to_pv_mesh(mesh).extract_all_edges()
    pl.subplot(0,i); pl.set_background('white')
    pl.add_mesh(pvT, scalars="T", cmap="RdBu_r", clim=(0,1), lighting=False, show_scalar_bar=False, opacity=0.5)
    pl.add_mesh(edges, color="#111111", line_width=0.5, lighting=False)
    pl.add_text("res32  refinement=%g" % R, font_size=12, color='black')
    pl.view_xy(); pl.enable_parallel_projection(); pl.camera.focal_point=(0,0,0); pl.camera.position=(0,0,1); pl.camera.zoom(0.97)
out = os.path.join(OUT,"thermal_R.png"); pl.screenshot(out); pl.close(); print("->",out,flush=True)
print("DONE", flush=True)
