"""Probe the pieces of the Zhong & Gurnis (1995) analogue on a small annulus:
a dipping trace running past the outer wall placed as a ladder band on the
base's finest, spine split blind below the surface, a TI bridge painted on the
band's top, rotated free-slip on both walls, temperature-dependent viscosity
from a base-mesh T evaluated onto the add-on, custom FMG with the base tail,
topography and surface velocity read back.

python zg95_probe.py <outdir>
"""
import os
import sys
import time

import numpy as np
import sympy
import underworld3 as uw
from underworld3.utilities.custom_mg import set_custom_fmg

out = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser("~/+Simulations/zg95")
os.makedirs(out, exist_ok=True)
RI, RO = 0.55, 1.0
H = 0.16               # coarsest base cell; REF halvings below it
REF = 3                # finest base cell H / 2**REF = 0.02; the coarsest annulus
                       # stays small enough for the SVD coarse solve the free-slip
                       # gauge needs (14k unknowns at H = 0.08 took over 30 min)
DIP = np.deg2rad(45.0)
DEPTH = 0.15           # fault reaches this far below the surface
RA, DETA = 1e6, 1e3

t0 = time.perf_counter()
base = uw.meshing.Annulus(radiusOuter=RO, radiusInner=RI, cellSize=H, refinement=REF, qdegree=2)
print(f"base: {base.dm.getHeightStratum(0)[1]} cells, levels {len(base._coarse_level_meshes())}",
      flush=True)
tail = base._coarse_level_meshes()[:-1] + [base]
S = 0.005

# temperature on the BASE: conductive profile + a lid, perturbed
T_b = uw.discretisation.MeshVariable("T_b", base, 1, degree=2)
r_b, th_b = base.CoordinateSystem.R
xb = np.asarray(T_b.coords)
rr = np.hypot(xb[:, 0], xb[:, 1])
tt = np.arctan2(xb[:, 1], xb[:, 0])
depth = (RO - rr) / (RO - RI)
T_b.array[:, 0, 0] = np.clip(1.0 - np.exp(-depth / 0.15) * 0.0 - depth * 0.0, 0, 1)   # placeholder
T_b.array[:, 0, 0] = 0.5 * (1 - np.cos(np.pi * np.minimum(depth / 0.3, 1.0))) * 0.5 + 0.5 * np.minimum(depth / 0.3, 1.0)
T_b.array[:, 0, 0] = np.clip(T_b.array[:, 0, 0] + 0.05 * np.cos(3 * tt) * np.sin(np.pi * depth), 0, 1)

# the band: the structured ladder (one element either side of the spine, width 2S)
# along a line from past the outer wall down at DIP toward +theta, sampled at S;
# the spine is split on its own rung vertices for the samples at least three
# spacings inside the surface; the ladder's outcrop cap carries the band to the wall
from underworld3.utilities.place_surface import place_fault_ribbon_2d
W = 2.0 * S
th0 = 0.0
t_hat = np.array([np.cos(th0), np.sin(th0)])            # outward radial at th0
s_hat = np.array([-np.sin(th0), np.cos(th0)])           # +theta direction
d_hat = -np.cos(DIP) * t_hat + np.sin(DIP) * s_hat      # down-dip, toward +theta
n_hat = np.array([-d_hat[1], d_hat[0]])                 # fault normal
top = t_hat * (RO + 0.02)
L_line = (DEPTH + 0.02) / np.cos(DIP)
n_s = int(round(L_line / S))
spine = np.array([top + d_hat * (L_line * i / n_s) for i in range(n_s + 1)])
r_sp = np.hypot(spine[:, 0], spine[:, 1])
trace = spine[r_sp <= RO - 3.0 * S]
t1 = time.perf_counter()
mesh, rinfo = place_fault_ribbon_2d(base, [("Trench", trace)], W, spines=[("Trench", spine)],
                                    margin_rings=1, clearance=0.7, split=True, mesher="ladder")
print(f"finest: {mesh.dm.getHeightStratum(0)[1]} cells, placed + split {time.perf_counter() - t1:.1f} s, "
      f"pairs {len(mesh._fault_point_pairs['Trench'])}, band {int(rinfo['band'].sum())} cells, "
      f"tail levels {len(mesh._custom_mg_coarse_meshes)}", flush=True)
print("boundaries:", [b.name for b in mesh.boundaries], flush=True)
foot = np.asarray(rinfo["band"], dtype=bool)
BAND_LABEL, BAND_VALUE = "Band", int(mesh.boundaries["Band"].value) if "Band" in [b.name for b in mesh.boundaries] else None

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection, PolyCollection


def mesh_art(m, fault, offset, n_hat):
    dm = m.dm
    eS, eE = dm.getDepthStratum(1)
    vS_, vE_ = dm.getDepthStratum(0)
    cS_, cE_ = dm.getHeightStratum(0)
    Xm = np.asarray(dm.getCoordinatesLocal().array).reshape(-1, 2)
    sides = {}
    for side, sgn in (("Plus", 1.0), ("Minus", -1.0)):
        name = f"{fault}{side}"
        if name in [bd.name for bd in m.boundaries]:
            lab = dm.getLabel(name)
            value = m.boundaries[name].value
            if lab and lab.getStratumSize(value) > 0:
                sides[side] = (set(int(e) for e in lab.getStratumIS(value).getIndices()
                                   if eS <= int(e) < eE), sgn)
    segs, fsegs = [], {"Plus": [], "Minus": []}
    for e in range(eS, eE):
        a_, b_ = (int(q) - vS_ for q in dm.getCone(e))
        seg = np.array([Xm[a_], Xm[b_]])
        placed = False
        for side, (es, sgn) in sides.items():
            if e in es:
                fsegs[side].append(seg + sgn * offset * n_hat)
                placed = True
        if not placed:
            segs.append(seg)
    polys = []
    try:
        mask = foot
        for c in np.nonzero(mask)[0]:
            cl = dm.getTransitiveClosure(int(c) + cS_)[0]
            polys.append(Xm[[int(q) - vS_ for q in cl if vS_ <= q < vE_]])
    except Exception:
        pass
    return segs, fsegs, polys


segs, fsegs, polys = mesh_art(mesh, "Trench", 0.0015, n_hat)
fig, axes = plt.subplots(1, 2, figsize=(13, 6.2), gridspec_kw={"width_ratios": [1, 1]})
for ax, (x0, x1, y0, y1), lw in ((axes[0], (-1.05, 1.05, -1.05, 1.05), 0.25),
                                 (axes[1], (0.80, 1.02, -0.06, 0.20), 0.5)):
    if polys:
        ax.add_collection(PolyCollection(polys, facecolors="#dff0d8", edgecolors="none"))
    ax.add_collection(LineCollection(segs, colors="#777", linewidths=lw))
    for side, col in (("Plus", "#c0392b"), ("Minus", "#1a5276")):
        if fsegs[side]:
            ax.add_collection(LineCollection(fsegs[side], colors=col, linewidths=1.6))
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
axes[0].set_title(f"the add-on finest: base annulus h = {H / 2**REF} ({REF} halvings of {H}) with the band placed, "
                  f"{mesh.dm.getHeightStratum(0)[1]} cells", fontsize=9, loc="left")
axes[1].set_title(f"the trench: ladder width {W}, spacing {S}, outcropping through the cap; spine split (red Plus, blue Minus) "
                  f"to three spacings below the surface", fontsize=9, loc="left")
fig.tight_layout()
fig.savefig(os.path.join(out, "probe_mesh.png"), dpi=170)
plt.close(fig)
print("wrote probe_mesh.png", flush=True)

# T on the add-on, by evaluation from the base
T_f = uw.discretisation.MeshVariable("T_f", mesh, 1, degree=2)
xf = np.asarray(T_f.coords)
rf = np.hypot(xf[:, 0], xf[:, 1])
xq = xf * np.minimum(1.0, (RO - 1e-10) / np.maximum(rf, 1e-12))[:, None]
xq = xq * np.maximum(1.0, (RI + 1e-10) / np.maximum(np.hypot(xq[:, 0], xq[:, 1]), 1e-12))[:, None]
t2 = time.perf_counter()
T_f.array[:, 0, 0] = uw.function.evaluate(T_b.sym[0], xq).ravel()
print(f"T transfer {time.perf_counter() - t2:.1f} s, range {T_f.array.min():.3f} {T_f.array.max():.3f}",
      flush=True)

# Stokes on the add-on
v = uw.discretisation.MeshVariable("U", mesh, 2, degree=2)
p = uw.discretisation.MeshVariable("P", mesh, 1, degree=1, continuous=True)
stokes = uw.systems.Stokes(mesh, velocityField=v, pressureField=p)
unit_r = mesh.CoordinateSystem.unit_e_0
stokes.bodyforce = RA * T_f.sym[0] * unit_r
eta_T = sympy.exp(sympy.log(DETA) * (1 - T_f.sym[0]))          # 1 at T=1, DETA at T=0
if os.environ.get("ZG95_CONSTETA"):
    eta_T = sympy.Integer(1)
    print("constant viscosity", flush=True)
# TI bridge: band cells whose centroid lies above the split's top tip
pairs = mesh._fault_point_pairs
if "Trench" in pairs:
    pairs = pairs["Trench"]
print("pairs:", type(pairs).__name__, len(pairs), flush=True)
vS, vE = mesh.dm.getDepthStratum(0)
X = np.asarray(mesh.dm.getCoordinatesLocal().array).reshape(-1, 2)
if isinstance(pairs, dict):
    pair_pts = np.array([int(k) for k in pairs] + [int(q) for q in pairs.values()])
else:
    pair_pts = np.asarray(pairs, dtype=int).ravel()
pv = pair_pts[(pair_pts >= vS) & (pair_pts < vE)] - vS
r_tip = np.hypot(X[pv, 0], X[pv, 1]).max()
print(f"split top tip at r = {r_tip:.4f} (surface {RO})", flush=True)
cS, cE = mesh.dm.getHeightStratum(0)
cent = np.array([X[[int(q) - vS for q in mesh.dm.getTransitiveClosure(c)[0] if vS <= q < vE]].mean(axis=0)
                 for c in range(cS, cE)])
bridge = foot & (np.hypot(cent[:, 0], cent[:, 1]) > r_tip - 0.5 * W)
zone = uw.discretisation.MeshVariable("zone", mesh, 1, degree=0, continuous=False)
zone.array[:, 0, 0] = bridge.astype(float)
print(f"footprint cells {int(foot.sum())}, bridge cells {int(bridge.sum())}", flush=True)
if os.environ.get("ZG95_NOTI"):
    stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
    stokes.constitutive_model.Parameters.shear_viscosity_0 = eta_T
    print("no TI bridge (isotropic)", flush=True)
else:
    stokes.constitutive_model = uw.constitutive_models.TransverseIsotropicFlowModel
    stokes.constitutive_model.Parameters.shear_viscosity_0 = eta_T
    stokes.constitutive_model.Parameters.shear_viscosity_1 = eta_T * (1 - 0.999 * zone.sym[0])
    stokes.constitutive_model.Parameters.director = sympy.Matrix([[float(n_hat[0]), float(n_hat[1])]])
if os.environ.get("ZG95_DIRICHLET"):
    stokes.add_dirichlet_bc((0.0, 0.0), "Upper")
    stokes.add_dirichlet_bc((0.0, 0.0), "Lower")
    print("no-slip walls (Dirichlet)", flush=True)
else:
    # the exact radial normal: with facet-averaged normals on an irregular
    # polygon the rigid rotation is not exactly tangential, the gauge mode is
    # rejected, and the coarse Galerkin operator is nearly singular for LU
    stokes.add_rotated_freeslip_bc(0, "Upper", normal=unit_r)
    stokes.add_rotated_freeslip_bc(0, "Lower", normal=unit_r)
if os.environ.get("ZG95_NOFAULT"):
    print("no fault bc (welded)", flush=True)
else:
    stokes.add_fault_bc(0, boundary="Trench")
stokes.petsc_use_pressure_nullspace = True
stokes.tolerance = 1e-5
print(f"T_f nan {int(np.isnan(T_f.array).sum())}, zone nan {int(np.isnan(zone.array).sum())}", flush=True)
if os.environ.get("ZG95_REL"):
    from underworld3.utilities import fault_mg
    fault_mg.UNREACHED_REL = float(os.environ["ZG95_REL"])
    print("unreached threshold", fault_mg.UNREACHED_REL, flush=True)
if not os.environ.get("ZG95_NOFMG"):
    set_custom_fmg(stokes, tail, field_id=0,
                   fac_zone=None if os.environ.get("ZG95_NOFAC") else foot)
    print("FMG with" + ("out" if os.environ.get("ZG95_NOFAC") else "") + " band patch", flush=True)
else:
    print("FMG disabled (GAMG)", flush=True)
t3 = time.perf_counter()
if os.environ.get("ZG95_TRAP"):
    from petsc4py import PETSc
    for pre in ("rotfs1_fieldsplit_vel_mg_levels_", "rotfs1_fieldsplit_vel_mg_levels_3_",
                "rotfs1_fieldsplit_vel_", "rotfs1_"):
        PETSc.Options()[pre + "ksp_error_if_not_converged"] = True
    print("trapping KSP failures", flush=True)
stokes.solve()
info = stokes._rotated_freeslip_info
try:
    ctx = (getattr(stokes, "_rotated_linear_cache", None) or {})["ctx"]
    velpc = ctx["ksp"].getPC().getFieldSplitSubKSP()[0].getPC()
    print(f"  null modes verified: {getattr(stokes, '_rotated_velocity_null_modes', None)}, "
          f"gauge removed: {info.get('rotation_gauge_removed')}, nsp: {ctx.get('nsp') is not None}, "
          f"coarse pc: {velpc.getMGCoarseSolve().getPC().getType()}", flush=True)
    velksp = ctx["ksp"].getPC().getFieldSplitSubKSP()[0]
    print(f"  prefixes: outer {ctx['ksp'].getOptionsPrefix()!r} velocity {velksp.getOptionsPrefix()!r}", flush=True)
    if os.environ.get("ZG95_PCAPPLY"):
        from petsc4py import PETSc
        A_vv = velpc.getOperators()[0]
        xx, yy = A_vv.createVecs()
        xx.setRandom()
        try:
            velpc.apply(xx, yy)
            print(f"  whole velocity PC apply: |y| {yy.norm():.3e}", flush=True)
        except Exception as e:
            print("  whole velocity PC apply raised:", str(e)[:200], flush=True)
        for l in range(velpc.getMGLevels()):
            sm = velpc.getMGSmoother(l)
            A = sm.getOperators()[0]
            Pm = sm.getOperators()[1]
            dA = A.getDiagonal().array; dP = Pm.getDiagonal().array
            print(f"  level {l}: Amat is Pmat {A.handle == Pm.handle}; zero diag A {int((dA == 0).sum())} P {int((dP == 0).sum())}; "
                  f"|dP| min {np.abs(dP).min():.3e}; nan P {int(np.isnan(dP).sum())} types {A.getType()}/{Pm.getType()}", flush=True)
            b, x = A.createVecs()
            b.setRandom(); x.set(0.0)
            try:
                sm.solve(b, x)
                print(f"  level {l} smoother ({sm.getType()}/{sm.getPC().getType()}): |x| {x.norm():.3e} its {sm.getIterationNumber()} reason {sm.getConvergedReason()}", flush=True)
            except Exception as e:
                print(f"  level {l} smoother raised:", str(e)[:200], flush=True)
            if l > 0:
                P = velpc.getMGInterpolation(l)
                xc = P.createVecRight(); xf = P.createVecLeft()
                xc.setRandom()
                P.mult(xc, xf)
                print(f"    P_{l} apply |xf| {xf.norm():.3e}; A_{l} apply |Ax| {A.createVecs()[0].norm():.3e}", flush=True)
        cs = velpc.getMGCoarseSolve()
        A0 = cs.getOperators()[0]
        b, x = A0.createVecs(); b.setRandom(); x.set(0.0)
        try:
            cs.solve(b, x)
            print(f"  coarse solve ({cs.getType()}/{cs.getPC().getType()}): |x| {x.norm():.3e} reason {cs.getConvergedReason()}", flush=True)
        except Exception as e:
            print("  coarse solve raised:", str(e)[:300], flush=True)
    for l in range(velpc.getMGLevels()):
        A = velpc.getMGSmoother(l).getOperators()[0]
        d = A.getDiagonal().array
        line = f"  level {l}: n {A.getSize()[0]} |A| {A.norm():.3e} diag min|.| {np.abs(d).min():.3e} nan {int(np.isnan(d).sum())}"
        if l > 0:
            P = velpc.getMGInterpolation(l)
            Pa = P.getValuesCSR()[2]
            line += f" | P {P.getSize()} nnz {len(Pa)} max {np.abs(Pa).max():.3e} nan {int(np.isnan(Pa).sum())} inf {int(np.isinf(Pa).sum())}"
        print(line, flush=True)
        if l > 0:
            bad = np.flatnonzero(np.abs(d) < 1e-8)
            print(f"    tiny-diagonal unknowns on level {l}: {bad[:10]} ({len(bad)})", flush=True)
            if len(bad):
                ia, ja, va = P.getValuesCSR()
                for b in bad[:4]:
                    rows = np.flatnonzero(np.diff(ia))[np.isin(ja[ia[np.flatnonzero(np.diff(ia))]], [b])] if False else None
                    hit = np.flatnonzero(ja == b)
                    rr = np.searchsorted(ia, hit, side="right") - 1
                    print(f"      col {b}: {len(hit)} entries, weights {va[hit][:8]}, fine rows {rr[:8]}", flush=True)
                    # coarse row b of the Galerkin operator: its own entries
                    cols_b, vals_b = A.getRow(int(b))
                    print(f"      A row {b}: {len(cols_b)} entries, max |.| {np.abs(vals_b).max() if len(vals_b) else 0:.3e}", flush=True)
            # the coarse levels' own layout: which node is unknown b?
            if l == 1 and len(bad):
                h_ = getattr(stokes, "_custom_mg_hierarchy", None)
                print("      hierarchy object:", type(h_).__name__ if h_ is not None else None,
                      [k for k in dir(h_) if "layout" in k or "reduced" in k or "map" in k][:12] if h_ is not None else "", flush=True)
except Exception as e:
    import traceback; traceback.print_exc()
    print("hierarchy inspection failed:", e, flush=True)
print(f"solve {time.perf_counter() - t3:.1f} s: pc {info.get('velocity_pc')}, vel its {info.get('vel_its_last')}, "
      f"pres its {info.get('pres_its_last')}, converged {info.get('converged')}", flush=True)
from underworld3.utilities.fault_contact import fault_slip
_s, Vslip, leak = fault_slip(stokes, "Trench", info)
print(f"slip max {np.abs(Vslip).max():.4f}, leak {np.abs(leak).max():.2e}", flush=True)

h = uw.discretisation.MeshVariable("topo", mesh, 1, degree=1)
stokes.dynamic_topography("Upper", h, buoyancy_scale=RA)
xs = np.asarray(h.coords)
on = np.hypot(xs[:, 0], xs[:, 1]) > RO - 1e-6
print(f"topography on {on.sum()} nodes, range {h.array[on].min():.3e} {h.array[on].max():.3e}", flush=True)
vmag = np.linalg.norm(v.array[:, 0, :], axis=1)
print(f"vmax {vmag.max():.3f}; total {time.perf_counter() - t0:.1f} s", flush=True)
np.savez(os.path.join(out, "probe.npz"), topo_x=xs[on], topo=h.array[on, 0, 0])
# solution figure: T with velocity arrows, surface velocity and topography against theta
import matplotlib.tri as mtri
Tx = np.asarray(T_f.coords)
tri_cells = np.array([[int(q) - vS for q in mesh.dm.getTransitiveClosure(c)[0] if vS <= q < vE]
                      for c in range(cS, cE)])
fig, axes = plt.subplots(1, 3, figsize=(17, 5.2), gridspec_kw={"width_ratios": [1.2, 1, 1]})
ax = axes[0]
Tv = uw.function.evaluate(T_f.sym[0], X * (1 - 1e-12)).ravel()
tc = ax.tripcolor(mtri.Triangulation(X[:, 0], X[:, 1], tri_cells), Tv, shading="gouraud",
                  cmap="RdBu_r", vmin=0, vmax=1)
vx = v.array[:, 0, :]
vc = np.asarray(v.coords)
pick = np.arange(0, len(vc), max(1, len(vc) // 900))
ax.quiver(vc[pick, 0], vc[pick, 1], vx[pick, 0], vx[pick, 1], color="k", scale=None, width=0.0025)
for side, col in (("Plus", "#c0392b"), ("Minus", "#1a5276")):
    if fsegs[side]:
        ax.add_collection(LineCollection(fsegs[side], colors=col, linewidths=1.2))
ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
ax.set_title(f"T (base field evaluated onto the add-on) and velocity; vmax {vmag.max():.1f}", fontsize=9, loc="left")
fig.colorbar(tc, ax=ax, shrink=0.8)
ax = axes[1]
surf = np.hypot(vc[:, 0], vc[:, 1]) > RO - 1e-6
th_s = np.arctan2(vc[surf, 1], vc[surf, 0])
vt = -vx[surf, 0] * np.sin(th_s) + vx[surf, 1] * np.cos(th_s)
o = np.argsort(th_s)
ax.plot(np.degrees(th_s[o]), vt[o], "-", lw=1)
ax.axvline(0, color="#c0392b", lw=0.8, ls="--")
ax.set_xlabel("theta (deg); trench at 0, dipping toward +theta"); ax.set_ylabel("surface tangential velocity")
ax.grid(alpha=0.3); ax.set_title("surface velocity", fontsize=9, loc="left")
ax = axes[2]
th_t = np.arctan2(xs[on, 1], xs[on, 0]); o = np.argsort(th_t)
ax.plot(np.degrees(th_t[o]), h.array[on, 0, 0][o], "-", lw=1)
ax.axvline(0, color="#c0392b", lw=0.8, ls="--")
ax.set_xlabel("theta (deg)"); ax.set_ylabel("dynamic topography  (-(s_nn - mean) / Ra)")
ax.grid(alpha=0.3); ax.set_title("topography from the free-slip reaction", fontsize=9, loc="left")
fig.tight_layout()
fig.savefig(os.path.join(out, "probe_solution.png"), dpi=150)
print("wrote probe_solution.png", flush=True)

# mesh + solution: the fault and its cut in the elements
from scipy.spatial import cKDTree
tri_obj = mtri.Triangulation(X[:, 0], X[:, 1], tri_cells)
bridge_polys = [X[tri_cells[c]] for c in np.nonzero(bridge)[0]]
_d, _k = cKDTree(vc).query(X)
vmag_v = vmag[_k]                                   # |v| at the mesh vertices
tip_xy = trace[0]                                  # the sample nearest the surface
fig, axes = plt.subplots(1, 3, figsize=(19, 6.4), gridspec_kw={"width_ratios": [1.15, 1, 1]})
ax = axes[0]
ax.tripcolor(tri_obj, Tv, shading="gouraud", cmap="RdBu_r", vmin=0, vmax=1, alpha=0.85)
ax.add_collection(LineCollection(segs, colors="#555", linewidths=0.18))
for side, col in (("Plus", "#c0392b"), ("Minus", "#1a5276")):
    if fsegs[side]:
        ax.add_collection(LineCollection(fsegs[side], colors=col, linewidths=1.4))
ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
ax.set_xlim(-1.03, 1.03); ax.set_ylim(-1.03, 1.03)
ax.set_title(f"T on the add-on finest ({mesh.dm.getHeightStratum(0)[1]} cells) with the mesh and the trench",
             fontsize=9, loc="left")
for ax, win, lw, off, title in (
        (axes[1], (0.84, 1.01, -0.04, 0.17), 0.45, 0.0015,
         "the trench: |v| under the mesh; band (green rim), TI bridge (orange rim), split spine (red Plus, blue Minus)"),
        (axes[2], (tip_xy[0] - 0.03, tip_xy[0] + 0.03, tip_xy[1] - 0.03, tip_xy[1] + 0.03), 0.7, 0.0008,
         f"the upper tip: the cut stops at r = {r_tip:.3f} inside the band; the bridge carries it to the outcrop")):
    tc = ax.tripcolor(tri_obj, vmag_v, shading="gouraud", cmap="viridis", vmin=0, vmax=np.percentile(vmag_v, 99))
    if polys:
        ax.add_collection(PolyCollection(polys, facecolors="none", edgecolors="#2e7d32", linewidths=0.6))
    if bridge_polys:
        ax.add_collection(PolyCollection(bridge_polys, facecolors="none", edgecolors="#e67e22", linewidths=0.9))
    ax.add_collection(LineCollection(segs, colors="#ddd", linewidths=lw))
    for side, col in (("Plus", "#c0392b"), ("Minus", "#1a5276")):
        if fsegs[side]:
            ax.add_collection(LineCollection(fsegs[side], colors=col, linewidths=2.0))
    ax.set_xlim(win[0], win[1]); ax.set_ylim(win[2], win[3])
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title, fontsize=8.5, loc="left")
fig.colorbar(tc, ax=axes[2], shrink=0.7, label="|v|")
fig.tight_layout()
fig.savefig(os.path.join(out, "probe_mesh_solution.png"), dpi=160)
print("wrote probe_mesh_solution.png", flush=True)
print("DONE")
