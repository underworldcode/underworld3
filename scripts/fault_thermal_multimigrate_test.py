"""Combined THERMAL + FAULT metric at a higher node budget. Two migrating faults
(carrier transport) AND a synthetic developed-convection T (mode-4 plumes + cold
top) whose |grad T| drives an isotropic thermal density. Question: at sufficient
budget, do BOTH demands coexist — thermal structure refined AND faults refined +
followed — with a free mmpde maintenance pass, no folding, fixed topology?

T is a fixed SPATIAL field here (static convection snapshot): re-sampled Eulerian
at current node positions each step so the thermal metric stays pinned in space
while the faults migrate through it. Renders full annulus: T (RdBu_r) + mesh edges
+ fault traces.
"""
import os, numpy as np, sympy, underworld3 as uw, underworld3.visualisation as vis, pyvista as pv
from underworld3.utilities.geometry_tools import signed_distance_pointcloud_polyline_2d
from scipy.spatial import cKDTree

pv.OFF_SCREEN = True
D = os.path.expanduser('~/+Simulations/StagnantLid+Fault'); os.makedirs(D, exist_ok=True)
R_o, R_i = 1.0, 0.5
BULK, SMIN = 1/16, 1/64
RF = BULK/SMIN                          # fault anisotropy matched to gmsh band (~4)
AMP_T = 3.0                            # thermal isotropic density amplitude
DIP, DEPTH = np.deg2rad(30.0), 0.12
DTHETA = SMIN/R_o
NSTEP, SUBITERS = 6, 2
CORE, WIDTH = 0.12, 0.09
FAULTS = [dict(theta=np.pi/2, rate=+1.0), dict(theta=np.deg2rad(150.0), rate=-1.0)]

def polyline(theta0):
    P0 = np.array([np.cos(theta0), np.sin(theta0)])
    e = np.array([np.cos(theta0), np.sin(theta0)]); t = np.array([-np.sin(theta0), np.cos(theta0)])
    dhat = np.cos(DIP)*t - np.sin(DIP)*e
    return P0[None,:] + np.linspace(0, DEPTH/np.sin(DIP), 40)[:,None]*dhat[None,:], dhat
XY0 = [polyline(f['theta'])[0] for f in FAULTS]

mesh = uw.meshing.Annulus(radiusOuter=R_o, radiusInner=R_i, cellSize=BULK, qdegree=3,
                          refine_lines=XY0, refine_size_min=SMIN,
                          refine_dist_min=0.03, refine_dist_max=0.16)
n_cells0 = mesh.dm.getHeightStratum(0); n_cells0 = n_cells0[1]-n_cells0[0]
dfac = uw.discretisation.MeshVariable("dist_fac", mesh, 1, degree=1, continuous=True)
rhoT = uw.discretisation.MeshVariable("rhoT", mesh, 1, degree=1, continuous=True)
GATES = [uw.discretisation.MeshVariable("gate_%d"%i, mesh, 1, degree=1, continuous=True) for i in range(len(FAULTS))]
Tv = uw.discretisation.MeshVariable("Tfield", mesh, 1, degree=2, continuous=True, varsymbol="T")
print("base: %d nodes, %d cells" % (mesh._coords.shape[0], n_cells0), flush=True)

def Tfield(xy):                 # synthetic developed convection: cold top + mode-4 plumes
    r = np.hypot(xy[:,0], xy[:,1]); th = np.arctan2(xy[:,1], xy[:,0])
    return (1.0 - r)/0.5 + 0.25*np.sin(4*th)*np.sin(np.pi*(r-0.5)/0.5)

def gradT_mag(xy):              # analytic |grad T| (Eulerian, fixed in space)
    r = np.hypot(xy[:,0], xy[:,1]); th = np.arctan2(xy[:,1], xy[:,0])
    s = np.pi/0.5
    dTdr = -2.0 + 0.25*np.sin(4*th)*s*np.cos(s*(r-0.5))
    dTdth = 0.25*4*np.cos(4*th)*np.sin(s*(r-0.5))
    return np.sqrt(dTdr**2 + (dTdth/r)**2)

def build_metric(thetas):
    # thermal isotropic density from |grad T| (re-sampled Eulerian)
    g = gradT_mag(np.asarray(rhoT.coords)[:,:2])
    lo, hi = np.percentile(g, 20), np.percentile(g, 90)
    t = np.clip((g - lo)/(hi - lo + 1e-12), 0, 1)
    rhoT.data[:,0] = 1.0 + AMP_T*t
    # fault min-distance isotropic draw + per-fault anisotropy gates
    coords = np.asarray(dfac.coords)[:,:2]
    dmin = np.full(coords.shape[0], 1e9)
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

def carrier(thetas_prev, dthetas):
    C = np.asarray(mesh._coords).copy(); x0,y0 = C[:,0].copy(), C[:,1].copy(); ang = np.zeros(C.shape[0])
    for thp, dth in zip(thetas_prev, dthetas):
        d = np.abs(signed_distance_pointcloud_polyline_2d(C[:,:2], polyline(thp)[0]))
        ang = ang + dth*np.exp(-(np.maximum(d - CORE, 0.0)/WIDTH)**2)
    ca, sa = np.cos(ang), np.sin(ang)
    C[:,0] = ca*x0 - sa*y0; C[:,1] = sa*x0 + ca*y0
    mesh._deform_mesh(C)

def metrics():
    C = np.asarray(mesh._coords)[:,:2]; tr = cKDTree(C); dd,_ = tr.query(C, k=2); nn = dd[:,1]
    g = gradT_mag(C); bulk = float(np.median(nn[(g < np.percentile(g,40))]))
    def fr(xy):
        d = np.abs(signed_distance_pointcloud_polyline_2d(C, xy)); m = d<0.04
        return float(np.median(nn[m]))/bulk if m.any() else np.nan
    therm = float(np.median(nn[g > np.percentile(g,85)]))/bulk
    return fr, therm

def folded():
    pvm = vis.mesh_to_pv_mesh(mesh); c = pvm.cells.reshape(-1,4)[:,1:]; p = pvm.points[:,:2]; v = p[c]
    a = 0.5*((v[:,1,0]-v[:,0,0])*(v[:,2,1]-v[:,0,1]) - (v[:,2,0]-v[:,0,0])*(v[:,1,1]-v[:,0,1]))
    return int((np.sign(a) != np.sign(np.median(a))).sum())

SNAPS = []
def snapshot(thetas, step):
    if step not in (0, 3, 6): return
    Tv.data[:,0] = Tfield(np.asarray(Tv.coords)[:,:2])     # current-geom T for colouring
    pvT = vis.meshVariable_to_pv_mesh_object(Tv); pvT.point_data["T"] = np.asarray(Tv.data[:,0])
    edges = vis.mesh_to_pv_mesh(mesh).extract_all_edges()
    SNAPS.append((pvT, edges, list(thetas), step))

def montage():
    # mesh PROMINENT (so refinement is visible), T faint for context. Full annulus.
    pl = pv.Plotter(off_screen=True, shape=(1,len(SNAPS)), window_size=(1300*len(SNAPS),1350), border=False)
    for i,(pvT, edges, thetas, step) in enumerate(SNAPS):
        pl.subplot(0,i); pl.set_background('white')
        pl.add_mesh(pvT, scalars="T", cmap="RdBu_r", show_edges=False, lighting=False, show_scalar_bar=False, opacity=0.30)
        pl.add_mesh(edges, color="#111111", line_width=0.8, lighting=False)        # mesh in front, dark
        for xy in [polyline(t)[0] for t in thetas]:
            pl.add_mesh(pv.lines_from_points(np.column_stack([xy, np.zeros(len(xy))])), color="#00c000", line_width=5, lighting=False)
        pl.add_text("step %d"%step, font_size=11, color='black')
        pl.view_xy(); pl.enable_parallel_projection(); pl.camera.focal_point=(0,0,0); pl.camera.position=(0,0,1); pl.camera.zoom(0.97)
    out = os.path.join(D, "thermal_multimigrate.png"); pl.screenshot(out); pl.close(); print("->", out, flush=True)
    # plus a zoom on the upper region (the two fault clusters) at the final step, mesh only
    pvT, edges, thetas, step = SNAPS[-1]
    pl = pv.Plotter(off_screen=True, window_size=(1500,900), border=False); pl.set_background('white')
    pl.add_mesh(pvT, scalars="T", cmap="RdBu_r", show_edges=False, lighting=False, show_scalar_bar=False, opacity=0.30)
    pl.add_mesh(edges, color="#111111", line_width=1.0, lighting=False)
    for xy in [polyline(t)[0] for t in thetas]:
        pl.add_mesh(pv.lines_from_points(np.column_stack([xy, np.zeros(len(xy))])), color="#00c000", line_width=6, lighting=False)
    pl.view_xy(); pl.enable_parallel_projection(); pl.camera.focal_point=(-0.3,0.62,0); pl.camera.position=(-0.3,0.62,1); pl.camera.zoom(2.2)
    out2 = os.path.join(D, "thermal_multimigrate_zoom.png"); pl.screenshot(out2); pl.close(); print("->", out2, flush=True)

thetas = [f['theta'] for f in FAULTS]
build_metric(thetas)
fr, therm = metrics()
print("step 0: A r=%.2f  B r=%.2f  thermal r=%.2f  folded=%d" % (fr(polyline(thetas[0])[0]), fr(polyline(thetas[1])[0]), therm, folded()), flush=True)
snapshot(thetas, 0)
for s in range(1, NSTEP+1):
    tp = list(thetas); dth = [f['rate']*DTHETA for f in FAULTS]; thetas = [a+b for a,b in zip(tp,dth)]
    carrier(tp, dth)
    T = build_metric(thetas)
    for _ in range(SUBITERS):
        uw.meshing.smooth_mesh_interior(mesh, metric=T, method="mmpde", skip_threshold=None,
            slip_surfaces=True, method_kwargs=dict(step_frac=0.2, accel="cg", momentum=0.0), verbose=False)
    nc = mesh.dm.getHeightStratum(0); nc = nc[1]-nc[0]; fr, therm = metrics()
    print("step %d: A@%.0f r=%.2f  B@%.0f r=%.2f  thermal r=%.2f  folded=%d %s"
          % (s, np.rad2deg(thetas[0]), fr(polyline(thetas[0])[0]), np.rad2deg(thetas[1]), fr(polyline(thetas[1])[0]),
             therm, folded(), 'TOPO OK' if nc==n_cells0 else 'TOPO CHANGED'), flush=True)
    snapshot(thetas, s)
montage()
print("DONE", flush=True)
