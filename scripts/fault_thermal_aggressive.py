"""Aggressive thermal refinement + fault refinement, together.

Fix for the weak (1.2x) thermal: give the thermal boundary layer its OWN gmsh base
via a CIRCULAR refine_line near the surface (same refine_lines machinery as the
faults), so it is MAINTAINED (~strong) not created-from-uniform (~2x cap). Plus a
SHARP synthetic T (thin cold surface BL + punchy mode-4 plumes) so |grad T| is
concentrated and the thermal refinement reads as a crisp band, not a smear.

Coarse interior base (1/11) + gmsh-refined: surface BL ring + 2 dipping faults.
Combined metric: thermal |grad T| density (maintains BL + concentrates on plumes)
+ fault anisotropy gates. Faults migrate (carrier); BL ring is static. Free mmpde.
Render mesh-PROMINENT: dense surface band + dense fault clusters cutting through it,
coarse interior, faint T for context.
"""
import os, numpy as np, sympy, underworld3 as uw, underworld3.visualisation as vis, pyvista as pv
from underworld3.utilities.geometry_tools import signed_distance_pointcloud_polyline_2d
from scipy.spatial import cKDTree

pv.OFF_SCREEN = True
D = os.path.expanduser('~/+Simulations/StagnantLid+Fault'); os.makedirs(D, exist_ok=True)
R_o, R_i = 1.0, 0.5
BULK, SMIN = 1/13, 1/52
RF = BULK/SMIN
DIP, DEPTH = np.deg2rad(30.0), 0.26       # DEEP faults -> distinct fingers below the BL ring
DTHETA = SMIN/R_o
NSTEP, SUBITERS = 4, 3
CORE, WIDTH = 0.14, 0.10
MODE, AMPT = 4, 0.40          # mode-4 plumes
RB, DBL = 0.90, 0.06          # cold surface BL: transition radius + thickness
FAULTS = [dict(theta=np.pi/2, rate=+1.0), dict(theta=np.deg2rad(150.0), rate=-1.0)]

def polyline(theta0):
    P0 = np.array([np.cos(theta0), np.sin(theta0)])
    e = np.array([np.cos(theta0), np.sin(theta0)]); t = np.array([-np.sin(theta0), np.cos(theta0)])
    dhat = np.cos(DIP)*t - np.sin(DIP)*e
    return P0[None,:] + np.linspace(0, DEPTH/np.sin(DIP), 40)[:,None]*dhat[None,:], dhat
XY0 = [polyline(f['theta'])[0] for f in FAULTS]
TC = np.linspace(0, 2*np.pi, 260); SURF_RING = np.column_stack([0.94*np.cos(TC), 0.94*np.sin(TC)])
mesh = uw.meshing.Annulus(radiusOuter=R_o, radiusInner=R_i, cellSize=BULK, qdegree=3,
                          refine_lines=[SURF_RING]+XY0, refine_size_min=SMIN,
                          refine_dist_min=0.02, refine_dist_max=0.07)   # thin band -> BL + faults, no budget hog
n_cells0 = mesh.dm.getHeightStratum(0); n_cells0 = n_cells0[1]-n_cells0[0]
dfac = uw.discretisation.MeshVariable("dist_fac", mesh, 1, degree=1, continuous=True)
rhoT = uw.discretisation.MeshVariable("rhoT", mesh, 1, degree=1, continuous=True)
GATES = [uw.discretisation.MeshVariable("gate_%d"%i, mesh, 1, degree=1, continuous=True) for i in range(len(FAULTS))]
Tv = uw.discretisation.MeshVariable("Tfield", mesh, 1, degree=2, continuous=True, varsymbol="T")
print("base: %d nodes, %d cells" % (mesh._coords.shape[0], n_cells0), flush=True)

def _env(r):     # radial envelope: plumes span the layer, peak mid-depth
    return np.sin(np.pi*(r - R_i)/(R_o - R_i))
def Tfield(xy):
    r = np.hypot(xy[:,0], xy[:,1]); th = np.arctan2(xy[:,1], xy[:,0])
    bl = 0.5*(1.0 + np.tanh((RB - r)/DBL))                       # hot interior -> cold surface BL
    return bl + AMPT*np.sin(MODE*th)*_env(r)                     # + mode-4 plumes

def gradT_mag(xy):
    r = np.hypot(xy[:,0], xy[:,1]); th = np.arctan2(xy[:,1], xy[:,0])
    L = R_o - R_i
    dbl_dr = -0.5/DBL*(1.0 - np.tanh((RB - r)/DBL)**2)           # BL radial gradient
    denv = np.cos(np.pi*(r - R_i)/L)*(np.pi/L)
    dTdr = dbl_dr + AMPT*np.sin(MODE*th)*denv
    dTdth = AMPT*MODE*np.cos(MODE*th)*_env(r)                    # angular plume walls
    return np.sqrt(dTdr**2 + (dTdth/r)**2)

def build_metric(thetas):
    g = gradT_mag(np.asarray(rhoT.coords)[:,:2])
    lo, hi = np.percentile(g, 35), np.percentile(g, 92)
    t = np.clip((g - lo)/(hi - lo + 1e-12), 0, 1)
    rhoT.data[:,0] = (1.0 + 7.0*t)**1.3                          # aggressive thermal density (BL+plumes)
    coords = np.asarray(dfac.coords)[:,:2]; dmin = np.full(coords.shape[0], 1e9)
    for th in thetas:
        dmin = np.minimum(dmin, np.abs(signed_distance_pointcloud_polyline_2d(coords, polyline(th)[0])))
    dfac.data[:,0] = dmin
    refine_w, draw_w = 2.0*SMIN, 6.0*SMIN
    gw = sympy.exp(-(dfac.sym[0]/draw_w)**2)
    T = (rhoT.sym[0] + 2.0*gw)*sympy.eye(2)
    for th, gate in zip(thetas, GATES):
        xy, dhat = polyline(th)
        nrm = np.array([-dhat[1], dhat[0]]); nrm /= np.linalg.norm(nrm)
        nnT = sympy.Matrix(nrm.reshape(2,1))*sympy.Matrix(nrm.reshape(2,1)).T
        dd = np.abs(signed_distance_pointcloud_polyline_2d(np.asarray(gate.coords)[:,:2], xy))
        gate.data[:,0] = np.exp(-(dd/refine_w)**2)
        T = T + (RF**2 - 1.0)*gate.sym[0]*nnT
    return T

def carrier(tp, dth):
    C = np.asarray(mesh._coords).copy(); x0,y0 = C[:,0].copy(), C[:,1].copy(); ang = np.zeros(C.shape[0])
    for thp, d in zip(tp, dth):
        dd = np.abs(signed_distance_pointcloud_polyline_2d(C[:,:2], polyline(thp)[0]))
        ang = ang + d*np.exp(-(np.maximum(dd - CORE, 0.0)/WIDTH)**2)
    ca, sa = np.cos(ang), np.sin(ang); C[:,0] = ca*x0 - sa*y0; C[:,1] = sa*x0 + ca*y0
    mesh._deform_mesh(C)

def metrics():
    C = np.asarray(mesh._coords)[:,:2]; tr = cKDTree(C); dd,_ = tr.query(C,k=2); nn = dd[:,1]
    g = gradT_mag(C); bulk = float(np.median(nn[g < np.percentile(g,40)]))
    fr = lambda xy: (lambda mm: float(np.median(nn[mm]))/bulk if mm.any() else np.nan)(np.abs(signed_distance_pointcloud_polyline_2d(C,xy))<0.04)
    therm = float(np.median(nn[g > np.percentile(g,85)]))/bulk
    return fr, therm

def folded():
    pvm = vis.mesh_to_pv_mesh(mesh); c = pvm.cells.reshape(-1,4)[:,1:]; p = pvm.points[:,:2]; v = p[c]
    a = 0.5*((v[:,1,0]-v[:,0,0])*(v[:,2,1]-v[:,0,1]) - (v[:,2,0]-v[:,0,0])*(v[:,1,1]-v[:,0,1]))
    return int((np.sign(a) != np.sign(np.median(a))).sum())

SNAPS = []
def snapshot(thetas, step):
    if step not in (0, NSTEP): return
    Tv.data[:,0] = Tfield(np.asarray(Tv.coords)[:,:2])
    pvT = vis.meshVariable_to_pv_mesh_object(Tv); pvT.point_data["T"] = np.asarray(Tv.data[:,0])
    SNAPS.append((pvT, vis.mesh_to_pv_mesh(mesh).extract_all_edges(), list(thetas), step))

def render():
    pl = pv.Plotter(off_screen=True, shape=(1,len(SNAPS)), window_size=(1300*len(SNAPS),1350), border=False)
    for i,(pvT, edges, thetas, step) in enumerate(SNAPS):
        pl.subplot(0,i); pl.set_background('white')
        pl.add_mesh(pvT, scalars="T", cmap="RdBu_r", lighting=False, show_scalar_bar=False, opacity=0.28)
        pl.add_mesh(edges, color="#111111", line_width=0.7, lighting=False)
        for xy in [polyline(t)[0] for t in thetas]:
            pl.add_mesh(pv.lines_from_points(np.column_stack([xy, np.zeros(len(xy))])), color="#00c000", line_width=5, lighting=False)
        pl.add_text("step %d"%step, font_size=11, color='black')
        pl.view_xy(); pl.enable_parallel_projection(); pl.camera.focal_point=(0,0,0); pl.camera.position=(0,0,1); pl.camera.zoom(0.97)
    out = os.path.join(D, "thermal_aggressive.png"); pl.screenshot(out); pl.close(); print("->", out, flush=True)

thetas = [f['theta'] for f in FAULTS]
build_metric(thetas); fr, therm = metrics()
print("step 0: A r=%.2f B r=%.2f thermal r=%.2f folded=%d" % (fr(polyline(thetas[0])[0]), fr(polyline(thetas[1])[0]), therm, folded()), flush=True)
snapshot(thetas, 0)
for s in range(1, NSTEP+1):
    tp = list(thetas); dth = [f['rate']*DTHETA for f in FAULTS]; thetas = [a+b for a,b in zip(tp,dth)]
    carrier(tp, dth); T = build_metric(thetas)
    for _ in range(SUBITERS):
        uw.meshing.smooth_mesh_interior(mesh, metric=T, method="mmpde", skip_threshold=None,
            slip_surfaces=True, method_kwargs=dict(step_frac=0.2, accel="cg", momentum=0.0), verbose=False)
    nc = mesh.dm.getHeightStratum(0); nc = nc[1]-nc[0]; fr, therm = metrics()
    print("step %d: A@%.0f r=%.2f B@%.0f r=%.2f thermal r=%.2f folded=%d %s"
          % (s, np.rad2deg(thetas[0]), fr(polyline(thetas[0])[0]), np.rad2deg(thetas[1]), fr(polyline(thetas[1])[0]),
             therm, folded(), 'TOPO OK' if nc==n_cells0 else 'TOPO CHANGED'), flush=True)
    snapshot(thetas, s)
render(); print("DONE", flush=True)
