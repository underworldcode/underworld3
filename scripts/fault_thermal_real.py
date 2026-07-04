"""THE COMBINED PICTURE on a REAL field: developed Ra=1e7 convection T + two faults
in the cold lid, adapted by one mmpde pass with the production combined metric:
  rho = rho_T(|grad T|, refinement=R)  *  fault_draw      (isotropic)
  M   = rho*I + sum_f (Rf^2-1)*gauss_sharp_f * n_f n_f^T   (anisotropic, per fault)
Faults get a gmsh base (refine_lines) + the tensor maintains/aligns; thermal BL +
plume roots refined by rho_T. Render full annulus (T + mesh + fault traces) + a
zoom on one fault sitting in the thermal mesh.
"""
import os, numpy as np, sympy, underworld3 as uw, underworld3.visualisation as vis, pyvista as pv
from underworld3.utilities.geometry_tools import signed_distance_pointcloud_polyline_2d
from scipy.spatial import cKDTree

pv.OFF_SCREEN = True
SRC = os.path.expanduser('~/+Simulations/StagnantLid/uniform_res16_Ra1e7_dEta1e4')
OUT = os.path.expanduser('~/+Simulations/StagnantLid+Fault')
LABEL = "sl_uniform_res16_Ra1e7_dEta1e4_step00100"
R_THERM, DIP, DEPTH = 5.0, np.deg2rad(30.0), 0.15

m16 = uw.discretisation.Mesh(os.path.join(SRC, LABEL + ".mesh.00000.h5"))
T16 = uw.discretisation.MeshVariable("T_v2p1", m16, 1, degree=3, continuous=True, varsymbol="T")
T16.read_timestep(LABEL, "T_v2p1", 0, outputPath=SRC)
C0 = np.asarray(m16._coords)[:,:2]; rad=np.hypot(C0[:,0],C0[:,1]); R_o,R_i=float(rad.max()),float(rad.min())
tr=cKDTree(C0); dd,_=tr.query(C0,k=2); h16=float(np.median(dd[:,1]))
BULK = h16/2; SMIN = BULK/4; RF = BULK/SMIN

# place faults at the COLDEST lid spots (downwellings): sample T on a near-surface ring
th = np.linspace(0, 2*np.pi, 360, endpoint=False)
ring = np.column_stack([0.9*np.cos(th), 0.9*np.sin(th)])
Tr = np.asarray(uw.function.evaluate(T16.sym, ring)).reshape(-1)
order = np.argsort(Tr)                      # coldest first
picks = [th[order[0]]]
for o in order:                             # second pick, >60deg from the first
    if abs((th[o]-picks[0]+np.pi) % (2*np.pi) - np.pi) > np.deg2rad(60): picks.append(th[o]); break
print("fault lid angles (deg, cold): %s" % [round(np.rad2deg(p)) for p in picks], flush=True)

def polyline(theta0):
    P0 = np.array([np.cos(theta0), np.sin(theta0)])
    e = np.array([np.cos(theta0), np.sin(theta0)]); t = np.array([-np.sin(theta0), np.cos(theta0)])
    dhat = np.cos(DIP)*t - np.sin(DIP)*e
    return P0[None,:] + np.linspace(0, DEPTH/np.sin(DIP), 40)[:,None]*dhat[None,:], dhat
XY = [polyline(p)[0] for p in picks]; DH = [polyline(p)[1] for p in picks]

# fault-refined base, interpolate the real T on
mesh = uw.meshing.Annulus(radiusOuter=R_o, radiusInner=R_i, cellSize=BULK, qdegree=3,
                          refine_lines=XY, refine_size_min=SMIN, refine_dist_min=0.02, refine_dist_max=0.12)
T = uw.discretisation.MeshVariable("T", mesh, 1, degree=3, continuous=True, varsymbol="T")
T.data[:,0] = np.asarray(uw.function.evaluate(T16.sym, np.asarray(T.coords))).reshape(-1)
dfac = uw.discretisation.MeshVariable("dist_fac", mesh, 1, degree=1, continuous=True)
GATES = [uw.discretisation.MeshVariable("gate_%d"%i, mesh, 1, degree=1, continuous=True) for i in range(len(XY))]
print("base: %d nodes (bulk %.4f, fault smin %.4f, RF %.1f)" % (mesh._coords.shape[0], BULK, SMIN, RF), flush=True)

# combined metric (driver pattern): thermal rho_T * fault draw + per-fault anisotropy
rho_T = uw.meshing.metric_density_from_gradient(mesh, T, refinement=R_THERM, coarsening="auto",
                                                metric_choice="front-following", name="therm")
dmin = np.full(np.asarray(dfac.coords).shape[0], 1e9)
for xy in XY: dmin = np.minimum(dmin, np.abs(signed_distance_pointcloud_polyline_2d(np.asarray(dfac.coords)[:,:2], xy)))
dfac.data[:,0] = dmin
refine_w, draw_w = 2.0*SMIN, 6.0*SMIN
gw = sympy.exp(-(dfac.sym[0]/draw_w)**2)
rho = rho_T * (1.0 + 3.0*gw)
M = rho*sympy.eye(2)
for xy, dh, gate in zip(XY, DH, GATES):
    nrm = np.array([-dh[1], dh[0]]); nrm /= np.linalg.norm(nrm)
    nnT = sympy.Matrix(nrm.reshape(2,1))*sympy.Matrix(nrm.reshape(2,1)).T
    gate.data[:,0] = np.exp(-(np.abs(signed_distance_pointcloud_polyline_2d(np.asarray(gate.coords)[:,:2], xy))/refine_w)**2)
    M = M + (RF**2 - 1.0)*gate.sym[0]*nnT
for _ in range(8):
    uw.meshing.smooth_mesh_interior(mesh, metric=M, method="mmpde", skip_threshold=None,
        slip_surfaces=True, method_kwargs=dict(step_frac=0.2, accel="cg", momentum=0.0), verbose=False)

# diagnostics
Cf = np.asarray(mesh._coords)[:,:2]; tr=cKDTree(Cf); ddf,_=tr.query(Cf,k=2); s=ddf[:,1]
bulk = float(np.median(s))
fr = [float(np.median(s[np.abs(signed_distance_pointcloud_polyline_2d(Cf,xy))<0.03]))/bulk for xy in XY]
pvm=vis.mesh_to_pv_mesh(mesh); c=pvm.cells.reshape(-1,4)[:,1:]; p=pvm.points[:,:2]; v=p[c]
a=0.5*((v[:,1,0]-v[:,0,0])*(v[:,2,1]-v[:,0,1])-(v[:,2,0]-v[:,0,0])*(v[:,1,1]-v[:,0,1]))
print("fault/bulk ratios %s | folded %d" % ([round(x,2) for x in fr], int((np.sign(a)!=np.sign(np.median(a))).sum())), flush=True)

# render: full annulus + zoom on fault 0
def draw(pl, zoom_xy=None):
    pvT=vis.meshVariable_to_pv_mesh_object(T); pvT.point_data["T"]=np.asarray(T.data[:,0])
    pl.set_background('white')
    pl.add_mesh(pvT, scalars="T", cmap="RdBu_r", clim=(0,1), lighting=False, show_scalar_bar=False, opacity=0.55)
    pl.add_mesh(vis.mesh_to_pv_mesh(mesh).extract_all_edges(), color="#111111",
                line_width=(0.9 if zoom_xy is not None else 0.45), lighting=False)
    for xy in XY:
        pl.add_mesh(pv.lines_from_points(np.column_stack([xy,np.zeros(len(xy))])), color="#00c000", line_width=5, lighting=False)
    pl.view_xy(); pl.enable_parallel_projection()
    if zoom_xy is None: pl.camera.focal_point=(0,0,0); pl.camera.position=(0,0,1); pl.camera.zoom(0.95)
    else: pl.camera.focal_point=(zoom_xy[0],zoom_xy[1],0); pl.camera.position=(zoom_xy[0],zoom_xy[1],1); pl.camera.zoom(3.2)

pl = pv.Plotter(off_screen=True, shape=(1,2), window_size=(2400,1200), border=False)
pl.subplot(0,0); draw(pl); pl.add_text("real Ra1e7 T + 2 lid faults (mmpde)", font_size=12, color='black')
pl.subplot(0,1); fc = XY[0][len(XY[0])//2]; draw(pl, zoom_xy=fc); pl.add_text("zoom: fault in the thermal mesh", font_size=12, color='black')
out=os.path.join(OUT,"fault_thermal_real.png"); pl.screenshot(out); pl.close(); print("->",out,flush=True)
print("DONE", flush=True)
