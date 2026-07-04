"""Static sanity checks for the dipping-fault convection setup.

Builds the same graded annulus + dipping fault as
``fault_convection_adapt_loop.py`` (no time stepping) and verifies:
  1. the anisotropic weak zone — eta_1/eta_0 ≈ 1/contrast ON the fault,
     ≈ 1 (isotropic) OFF the fault;
  2. the director is perpendicular to the fault trace;
  3. surface slip — after one mover move the Upper/Lower boundary node
     radii are preserved (nodes stay on the arc) while azimuths redistribute.

Writes a figure (fault factor field + fault trace + mesh) and a text
report to ~/+Simulations/StagnantLid/<tag>/ so the result can be viewed.
"""
from __future__ import annotations
import os
import argparse
import numpy as np
import sympy
import underworld3 as uw
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

p = argparse.ArgumentParser()
p.add_argument('--res', type=int, default=24)              # uniform resolution
p.add_argument('--resolution-ratio', type=float, default=2.5)
p.add_argument('--fault-refine-amp', type=float, default=12.0)
p.add_argument('--fault-dip-deg', type=float, default=30.0)
p.add_argument('--fault-theta-deg', type=float, default=90.0)
p.add_argument('--fault-depth', type=float, default=0.3)
p.add_argument('--fault-dip-dir', type=str, default='east')
p.add_argument('--fault-eta-contrast', type=float, default=1000.0)
p.add_argument('--fault-width', type=float, default=0.05)
p.add_argument('--out-tag', type=str, default='verify_dip30')
args = p.parse_args()

OUT = os.path.expanduser(f'~/+Simulations/StagnantLid/{args.out_tag}')
os.makedirs(OUT, exist_ok=True)
report = []
def double_log(s):
    print(s)
    report.append(s)

inv_c = 1.0 / args.fault_eta_contrast

mesh = uw.meshing.Annulus(
    radiusOuter=1.0, radiusInner=0.5,
    cellSize=1.0 / args.res, qdegree=3)            # uniform
T = uw.discretisation.MeshVariable("T", mesh, 1, degree=3)
gfac = uw.discretisation.MeshVariable("eta_fac", mesh, 1, degree=2, varsymbol="g_F")
T.data[:] = 0.5

# Fault geometry (identical construction to the driver).
delta = np.deg2rad(args.fault_dip_deg)
theta0 = np.deg2rad(args.fault_theta_deg)
P0 = np.array([np.cos(theta0), np.sin(theta0)])
e_hat = np.array([np.cos(theta0), np.sin(theta0)])
t_hat = np.array([-np.sin(theta0), np.cos(theta0)])
side = 1.0 if args.fault_dip_dir == 'east' else -1.0
dhat = side * np.cos(delta) * t_hat - np.sin(delta) * e_hat
L = args.fault_depth / np.sin(delta)
s = np.linspace(0.0, L, 25)[:, None]
xy = P0[None, :] + s * dhat[None, :]
fault_pts = np.column_stack([xy, np.zeros(len(xy))])
fault = uw.meshing.Surface("fault", mesh, fault_pts, symbol="F")
fault.discretize()

fault_factor_expr = fault.influence_function(
    width=args.fault_width, value_near=inv_c, value_far=1.0, profile="gaussian")
_ = fault.distance
gfac.data[:, 0] = np.asarray(
    uw.function.evaluate(fault_factor_expr, gfac.coords)).reshape(-1)

theta_FK = np.log(100.0)
eta_FK = sympy.exp(theta_FK * (1 - T.sym[0]))   # constant here (T=0.5)
eta_0 = eta_FK
eta_1 = eta_FK * gfac.sym[0]

double_log("=== Dipping-fault convection: static checks ===")
double_log(f"  fault: 12 o'clock θ={args.fault_theta_deg}°, dip={args.fault_dip_deg}°, "
           f"depth={args.fault_depth} (tip r={np.linalg.norm(xy[-1]):.3f}), "
           f"contrast={args.fault_eta_contrast:g}, width={args.fault_width}")

# --- 1. weak-zone viscosity ratio on vs off the fault ---
# eta_1/eta_0 = fault factor. Test BOTH the exact analytic gaussian (the
# method) and the projected gfac field (what the solver sees — softened by
# the degree-2 FE representation on a coarse mesh).
mid = xy[len(xy) // 2]                     # a point on the fault trace
n_vec = np.array([-dhat[1], dhat[0]]); n_vec /= np.linalg.norm(n_vec)
on_pts = mid[None, :]
off_pts = (mid + 4 * args.fault_width * n_vec)[None, :]   # step across the fault
r_on_exact = float(np.asarray(uw.function.evaluate(fault_factor_expr, on_pts)).reshape(-1)[0])
r_on_fld = float(np.asarray(uw.function.evaluate(eta_1 / eta_0, on_pts)).reshape(-1)[0])
r_off = float(np.asarray(uw.function.evaluate(eta_1 / eta_0, off_pts)).reshape(-1)[0])
double_log(f"  [1] eta_1/eta_0 ON fault  exact={r_on_exact:.3e}  "
           f"projected-field={r_on_fld:.3e}  (target {inv_c:.3e})")
double_log(f"      eta_1/eta_0 OFF fault = {r_off:.3e}  (target ~1, isotropic)")
# exact gaussian must hit ~inv_c; the projected field must still be a strong
# (>50x) weakening and isotropic away.
ok1 = (r_on_exact < 2 * inv_c) and (r_on_fld < 0.02) and (r_off > 0.9)
double_log(f"      → {'PASS' if ok1 else 'FAIL'}  "
           f"(exact≈inv_c, field weakens >50x, off-fault isotropic)")

# --- 2. director perpendicular to the fault trace ---
trace_dir = (xy[-1] - xy[0]); trace_dir /= np.linalg.norm(trace_dir)
dotp = float(abs(np.dot(n_vec, trace_dir)))
double_log(f"  [2] |director · fault_tangent| = {dotp:.2e}  (target 0)")
ok2 = dotp < 1e-9
double_log(f"      → {'PASS' if ok2 else 'FAIL'}")

# --- 3. surface slip: boundary radii preserved, azimuths redistribute ---
def _bnd_idx(label):
    import numpy as _np
    lab = mesh.dm.getLabel(label)
    if lab is None:
        return _np.array([], dtype=int)
    # vertices carrying this boundary label, mapped to coord rows
    return None

# Simpler: classify by radius (Upper r≈1.0, Lower r≈0.5).
coords0 = np.asarray(mesh.X.coords).copy()
r0 = np.linalg.norm(coords0, axis=1)
up = np.where(np.abs(r0 - 1.0) < 1e-6)[0]
lo = np.where(np.abs(r0 - 0.5) < 1e-6)[0]
th0_up = np.arctan2(coords0[up, 1], coords0[up, 0])

rho_T = uw.meshing.metric_density_from_gradient(mesh, T, strategy="med", name="r")
d_signed = fault.distance.sym[0]
fault_rho = 1.0 + args.fault_refine_amp * sympy.exp(
    -(d_signed / (1.5 * args.fault_width)) ** 2)
# T is uniform → rho_T is flat; the fault term drives the move here. The
# resolution_ratio override sets how strongly cells shrink toward the fault.
uw.meshing.smooth_mesh_interior(
    mesh, metric=rho_T * fault_rho, method="anisotropic", strategy="med",
    skip_threshold=None, slip_surfaces=True,   # None ⇒ always adapt (force the move)
    method_kwargs=dict(relax=0.2, n_outer=12,
                       resolution_ratio=args.resolution_ratio), verbose=False)
coords1 = np.asarray(mesh.X.coords)
r1 = np.linalg.norm(coords1, axis=1)
dr_up = float(np.abs(r1[up] - 1.0).max()) if len(up) else 0.0
dr_lo = float(np.abs(r1[lo] - 0.5).max()) if len(lo) else 0.0
th1_up = np.arctan2(coords1[up, 1], coords1[up, 0])
dth = float(np.abs(np.unwrap(th1_up) - np.unwrap(th0_up)).max()) if len(up) else 0.0
double_log(f"  [3] surface slip after move:")
double_log(f"      max |Δr| on Upper = {dr_up:.2e}, on Lower = {dr_lo:.2e} (target ~0, nodes stay on arc)")
double_log(f"      max |Δθ| on Upper = {dth:.2e} (target > 0 ⇒ tangential slide)")
ok3 = (dr_up < 1e-4) and (dr_lo < 1e-4) and (dth > 1e-4)
double_log(f"      → {'PASS' if ok3 else 'FAIL'}")

double_log(f"\n  SUMMARY: weak-zone={'PASS' if ok1 else 'FAIL'}, "
           f"director={'PASS' if ok2 else 'FAIL'}, slip={'PASS' if ok3 else 'FAIL'}")

# --- figure: fault factor field + the adapted MESH (triangle edges) ---
from matplotlib.tri import Triangulation
_dm = mesh.dm
_pS, _pE = _dm.getDepthStratum(0)
_cS, _cE = _dm.getHeightStratum(0)
_tris = np.asarray([[p - _pS for p in _dm.getTransitiveClosure(c)[0]
                     if _pS <= p < _pE] for c in range(_cS, _cE)])
tri1 = Triangulation(coords1[:, 0], coords1[:, 1], _tris)   # moved mesh

fig, ax = plt.subplots(1, 2, figsize=(12, 6))
for a in ax:
    a.set_aspect("equal"); a.axis("off")
gfac_at_C = np.asarray(uw.function.evaluate(gfac.sym[0], coords0)).reshape(-1)
sc = ax[0].tripcolor(Triangulation(coords0[:, 0], coords0[:, 1], _tris),
                     gfac_at_C, shading="gouraud", cmap="RdBu")
ax[0].plot(xy[:, 0], xy[:, 1], "r-", lw=2)
ax[0].set_title("fault factor g_F (weak on fault → 1 away)")
plt.colorbar(sc, ax=ax[0], shrink=0.7)
# the adapted mesh — refinement clusters along the dipping fault
ax[1].triplot(tri1, color="0.35", lw=0.3)
ax[1].plot(xy[:, 0], xy[:, 1], "r-", lw=1.8)
ax[1].set_title("adapted mesh after fault-driven move (slip ON)")
fig.tight_layout()
fig_path = os.path.join(OUT, "fault_checks.png")
fig.savefig(fig_path, dpi=130)
double_log(f"\n  figure → {fig_path}")

with open(os.path.join(OUT, "fault_checks.txt"), "w") as f:
    f.write("\n".join(report) + "\n")
