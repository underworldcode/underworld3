"""Region-resolved: does R=10 coarsen the PLUME INTERIORS (low |grad T|) to feed
the BL (high |grad T|)? Global percentiles miss it; measure each region directly.
res32, same T, mmpde, R=5 vs R=10. Report median spacing in low/high-gradient
zones + element COUNT in the plume interiors (Louis counts ~20% fewer at R=10).
"""
import os, numpy as np, underworld3 as uw
from scipy.spatial import cKDTree

SRC = os.path.expanduser('~/+Simulations/StagnantLid/uniform_res16_Ra1e7_dEta1e4')
LABEL = "sl_uniform_res16_Ra1e7_dEta1e4_step00100"
m16 = uw.discretisation.Mesh(os.path.join(SRC, LABEL + ".mesh.00000.h5"))
T16 = uw.discretisation.MeshVariable("T_v2p1", m16, 1, degree=3, continuous=True, varsymbol="T")
T16.read_timestep(LABEL, "T_v2p1", 0, outputPath=SRC)
C = np.asarray(m16._coords)[:,:2]; rad=np.hypot(C[:,0],C[:,1]); R_o,R_i=float(rad.max()),float(rad.min())
tr=cKDTree(C); dd,_=tr.query(C,k=2); h16=float(np.median(dd[:,1]))

def run(R):
    m = uw.meshing.Annulus(radiusOuter=R_o, radiusInner=R_i, cellSize=h16/2, qdegree=3)
    T = uw.discretisation.MeshVariable("Tr%g"%R, m, 1, degree=3, continuous=True, varsymbol="T")
    T.data[:,0] = np.asarray(uw.function.evaluate(T16.sym, np.asarray(T.coords))).reshape(-1)
    rho = uw.meshing.metric_density_from_gradient(m, T, refinement=float(R), coarsening="auto",
                                                  metric_choice="front-following", name=T.name)
    for _ in range(10):
        uw.meshing.smooth_mesh_interior(m, metric=rho, method="mmpde", skip_threshold=None,
            slip_surfaces=True, method_kwargs=dict(step_frac=0.2, accel="cg", momentum=0.0), verbose=False)
    Cf = np.asarray(m._coords)[:,:2]; tr=cKDTree(Cf); dd,_=tr.query(Cf,k=2); s=dd[:,1]
    g = np.asarray(uw.function.evaluate(rho, Cf)).reshape(-1)          # rho ~ |grad T| proxy
    lo = g < np.percentile(g,33); hi = g > np.percentile(g,67)         # plume interior / BL+margin
    # count nodes in the geometric plume-interior shell (low-grad AND mid-depth) as a direct tally
    interior = lo & (rad_at(Cf) > R_i+0.12) & (rad_at(Cf) < R_o-0.12)
    return dict(R=R, n=Cf.shape[0],
                lo=float(np.median(s[lo])), hi=float(np.median(s[hi])),
                interior_n=int(interior.sum()), interior_h=float(np.median(s[interior])))

def rad_at(P): return np.hypot(P[:,0],P[:,1])

a = run(5); b = run(10)
print("region          R=5        R=10       change", flush=True)
print("plume-interior  %.4f     %.4f     %+.0f%% spacing  (%d -> %d nodes, %+.0f%%)" % (
    a['lo'], b['lo'], 100*(b['lo']/a['lo']-1), a['interior_n'], b['interior_n'],
    100*(b['interior_n']/a['interior_n']-1)), flush=True)
print("BL + margins    %.4f     %.4f     %+.0f%% spacing" % (a['hi'], b['hi'], 100*(b['hi']/a['hi']-1)), flush=True)
print("interior median spacing %.4f -> %.4f (%+.0f%%)" % (a['interior_h'], b['interior_h'],
    100*(b['interior_h']/a['interior_h']-1)), flush=True)
print("DONE", flush=True)
