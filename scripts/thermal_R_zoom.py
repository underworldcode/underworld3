"""Settle the plume-interior question by direct cell COUNT + a zoom. res32, same T,
mmpde, R=5 vs R=10 in ONE run. Pick a hot-lobe interior box, count cells whose
centroid is inside it for each R, and render that box zoomed for both.
"""
import os, numpy as np, underworld3 as uw, underworld3.visualisation as vis, pyvista as pv

pv.OFF_SCREEN = True
SRC = os.path.expanduser('~/+Simulations/StagnantLid/uniform_res16_Ra1e7_dEta1e4')
OUT = os.path.expanduser('~/+Simulations/StagnantLid')
LABEL = "sl_uniform_res16_Ra1e7_dEta1e4_step00100"
m16 = uw.discretisation.Mesh(os.path.join(SRC, LABEL + ".mesh.00000.h5"))
T16 = uw.discretisation.MeshVariable("T_v2p1", m16, 1, degree=3, continuous=True, varsymbol="T")
T16.read_timestep(LABEL, "T_v2p1", 0, outputPath=SRC)
C0 = np.asarray(m16._coords)[:,:2]; rad=np.hypot(C0[:,0],C0[:,1]); R_o,R_i=float(rad.max()),float(rad.min())
from scipy.spatial import cKDTree
tr=cKDTree(C0); dd,_=tr.query(C0,k=2); h16=float(np.median(dd[:,1]))

# pick a hot-lobe interior centre from the res16 field (high T, mid-depth)
rmid = (C0[:,0]**2+C0[:,1]**2)**0.5
midband = (rmid>R_i+0.12)&(rmid<R_o-0.18)
Tvals = np.asarray(T16.data[:,0])[ :C0.shape[0] ] if T16.data.shape[0]==C0.shape[0] else None
# robust: evaluate T at vertices
Tv = np.asarray(uw.function.evaluate(T16.sym, C0)).reshape(-1)
ci = np.argmax(np.where(midband, Tv, -1)); cx, cy = C0[ci]
HALF = 0.14
box = (cx-HALF, cx+HALF, cy-HALF, cy+HALF)
print("plume-interior box centre (%.2f,%.2f) T=%.2f, half=%.2f" % (cx,cy,Tv[ci],HALF), flush=True)

def run(R):
    m = uw.meshing.Annulus(radiusOuter=R_o, radiusInner=R_i, cellSize=h16/2, qdegree=3)
    T = uw.discretisation.MeshVariable("Tr%g"%R, m, 1, degree=3, continuous=True, varsymbol="T")
    T.data[:,0] = np.asarray(uw.function.evaluate(T16.sym, np.asarray(T.coords))).reshape(-1)
    rho = uw.meshing.metric_density_from_gradient(m, T, refinement=float(R), coarsening="auto",
                                                  metric_choice="front-following", name=T.name)
    for _ in range(10):
        uw.meshing.smooth_mesh_interior(m, metric=rho, method="mmpde", skip_threshold=None,
            slip_surfaces=True, method_kwargs=dict(step_frac=0.2, accel="cg", momentum=0.0), verbose=False)
    pvm = vis.mesh_to_pv_mesh(m); c = pvm.cells.reshape(-1,4)[:,1:]; p = pvm.points[:,:2]
    cen = p[c].mean(1)
    inbox = (cen[:,0]>box[0])&(cen[:,0]<box[1])&(cen[:,1]>box[2])&(cen[:,1]<box[3])
    return m, T, int(inbox.sum())

m5,T5,n5 = run(5); m10,T10,n10 = run(10)
print("cells in plume-interior box:  R=5 -> %d   R=10 -> %d   (%+.0f%%)" % (n5, n10, 100*(n10/n5-1)), flush=True)

pl = pv.Plotter(off_screen=True, shape=(1,2), window_size=(2200,1150), border=False)
for i,(m,T,R,n) in enumerate([(m5,T5,5,n5),(m10,T10,10,n10)]):
    pvT = vis.meshVariable_to_pv_mesh_object(T); pvT.point_data["T"]=np.asarray(T.data[:,0])
    edges = vis.mesh_to_pv_mesh(m).extract_all_edges()
    pl.subplot(0,i); pl.set_background('white')
    pl.add_mesh(pvT, scalars="T", cmap="RdBu_r", clim=(0,1), lighting=False, show_scalar_bar=False, opacity=0.45)
    pl.add_mesh(edges, color="#111111", line_width=1.0, lighting=False)
    pl.add_mesh(pv.Box((box[0],box[1],box[2],box[3],-0.01,0.01)).extract_all_edges(), color="#0044cc", line_width=3, lighting=False)
    pl.add_text("R=%g : %d cells in box" % (R,n), font_size=13, color='black')
    pl.view_xy(); pl.enable_parallel_projection(); pl.camera.focal_point=(cx,cy,0); pl.camera.position=(cx,cy,1); pl.camera.zoom(3.0)
out=os.path.join(OUT,"thermal_R_zoom.png"); pl.screenshot(out); pl.close(); print("->",out,flush=True)
print("DONE", flush=True)
