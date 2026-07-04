"""OT + spring-polish chain on the step-25 snapshot.

Compares three variants on the same T-field:
  (a) original
  (b) OT × 5  (the plain-OT baseline from _ot_snapshot_compare.py)
  (c) OT × 4 → spring(size_w=2) × 1 → OT × 1  (the cooperative
      recipe from the synthetic-shapes sweep)

Each variant re-solves Stokes on the same T and reports
(vrms, Nu_surface). Renders panels incrementally and writes
PNG + npz to ~/+Simulations/StagnantLid/ot_test/diagnostics/.
"""
from __future__ import annotations
import os
import time
import argparse
import numpy as np
import sympy
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.tri import Triangulation

import underworld3 as uw


p = argparse.ArgumentParser()
p.add_argument("--snapshot-dir", type=str,
               default=os.path.expanduser(
                   "~/+Simulations/StagnantLid/ot_test"))
p.add_argument("--step", type=int, default=25)
p.add_argument("--Ra", type=float, default=1.0e7)
p.add_argument("--delta-eta", type=float, default=1.0e2)
p.add_argument("--refinement", type=float, default=3.0)
p.add_argument("--slip", type=str, default="ring",
               choices=["ring", "box", "false"],
               help="boundary_slip mode (default 'ring' — correct "
                    "for annulus; 'box' was wrong and distorted "
                    "the circular boundary in earlier runs).")
args = p.parse_args()

SNAPSHOT_LABEL = f"step{args.step:04d}"
SRC = args.snapshot_dir
DIAG = os.path.join(SRC, "diagnostics")
os.makedirs(DIAG, exist_ok=True)
SLIP = False if args.slip == "false" else args.slip
SLIP_TAG = f"slip-{args.slip}"
PNG_PATH = os.path.join(
    DIAG,
    f"ot_chain_{SNAPSHOT_LABEL}_r{int(args.refinement)}_{SLIP_TAG}.png")

print(f"=== OT + spring-polish chain on {SNAPSHOT_LABEL} ===",
      flush=True)
print(f"  Ra={args.Ra:.1e} Δη={args.delta_eta:.0f} "
      f"refinement={args.refinement}", flush=True)
print(f"  panel PNG: {PNG_PATH}", flush=True)

R_INNER, R_OUTER = 0.5, 1.0
Q_COND = 2.0 * np.pi / np.log(R_OUTER / R_INNER)
theta_FK = float(np.log(args.delta_eta))


def load_state():
    mesh = uw.discretisation.Mesh(
        os.path.join(SRC, f"{SNAPSHOT_LABEL}.mesh.00000.h5"))
    T = uw.discretisation.MeshVariable(
        "T_v2p1", mesh, vtype=uw.VarType.SCALAR, degree=3,
        continuous=True, varsymbol="T")
    V = uw.discretisation.MeshVariable(
        "V_v2p1", mesh, vtype=uw.VarType.VECTOR, degree=2,
        continuous=True, varsymbol=r"\mathbf{v}")
    P = uw.discretisation.MeshVariable(
        "P_v2p1", mesh, vtype=uw.VarType.SCALAR, degree=1,
        continuous=True, varsymbol="p")
    T.read_timestep(SNAPSHOT_LABEL, "T_v2p1", 0, outputPath=SRC)
    V.read_timestep(SNAPSHOT_LABEL, "V_v2p1", 0, outputPath=SRC)
    P.read_timestep(SNAPSHOT_LABEL, "P_v2p1", 0, outputPath=SRC)
    return mesh, T, V, P


def fe_remap_T(mesh, T, old_X, old_T_data, new_X):
    new_Tx = np.asarray(T.coords).copy()
    mesh._deform_mesh(old_X)
    T.data[...] = old_T_data
    remapped = np.asarray(uw.function.evaluate(
        T.sym[0], new_Tx)).reshape(-1)
    mesh._deform_mesh(new_X)
    T.data[:, 0] = remapped


def stokes_solve(mesh, T, V, P, label):
    X = mesh.CoordinateSystem.X
    unit_r = mesh.CoordinateSystem.unit_e_0
    stokes = uw.systems.Stokes(
        mesh, velocityField=V, pressureField=P)
    stokes.constitutive_model = (
        uw.constitutive_models.ViscousFlowModel)
    stokes.constitutive_model.Parameters.shear_viscosity_0 = (
        sympy.exp(theta_FK * (1 - T.sym[0])))
    stokes.tolerance = 1.0e-5
    stokes.bodyforce = args.Ra * T.sym[0] * unit_r
    stokes.add_essential_bc((0.0, 0.0),
                             mesh.boundaries.Lower.name)
    stokes.add_essential_bc((0.0, 0.0),
                             mesh.boundaries.Upper.name)
    V.data[...] = 0.0
    P.data[...] = 0.0
    t0 = time.time()
    stokes.solve(zero_init_guess=True)
    elapsed = time.time() - t0

    vol = float(uw.maths.Integral(mesh=mesh, fn=1.0).evaluate())
    v2_int = float(uw.maths.Integral(
        mesh=mesh, fn=V.sym.dot(V.sym)).evaluate())
    vrms = float(np.sqrt(max(v2_int / vol, 0.0)))

    n = mesh.Gamma_N
    qn = -(T.sym[0].diff(X[0]) * n[0]
           + T.sym[0].diff(X[1]) * n[1])
    bd = uw.maths.BdIntegral(
        mesh=mesh, fn=qn, boundary=mesh.boundaries.Upper.name)
    nu = float(bd.evaluate()) / Q_COND
    print(f"  [{label}] Stokes {elapsed:.1f}s  "
          f"vrms={vrms:.3e}  Nu={nu:.3f}", flush=True)
    return vrms, nu


def mesh_edges_xy(mesh):
    dm = mesh.dm
    coords = np.asarray(mesh.X.coords)
    pStart, _ = dm.getDepthStratum(0)
    eStart, eEnd = dm.getDepthStratum(1)
    segs = np.empty((eEnd - eStart, 2, 2), dtype=float)
    for k, e in enumerate(range(eStart, eEnd)):
        cone = dm.getCone(e)
        segs[k, 0] = coords[cone[0] - pStart]
        segs[k, 1] = coords[cone[1] - pStart]
    return segs


fig, axes = plt.subplots(1, 3, figsize=(18, 6.5),
                          constrained_layout=True)


def render_panel(ax, mesh, T_field, title):
    ax.clear()
    Xc = np.asarray(T_field.coords)
    Tv = np.asarray(T_field.data[:, 0])
    tri = Triangulation(Xc[:, 0], Xc[:, 1])
    cx = Xc[tri.triangles, 0].mean(axis=1)
    cy = Xc[tri.triangles, 1].mean(axis=1)
    rcen = np.sqrt(cx**2 + cy**2)
    mask = (rcen > R_OUTER + 1e-6) | (rcen < R_INNER - 1e-6)
    tri.set_mask(mask)
    ax.tripcolor(tri, Tv, cmap="inferno", shading="gouraud",
                 vmin=0, vmax=1)
    ax.tricontour(tri, Tv, levels=[0.2, 0.4, 0.6, 0.8],
                  colors="cyan", linewidths=0.4, alpha=0.55)
    segs = mesh_edges_xy(mesh)
    lc = LineCollection(segs, colors="white", linewidths=0.3,
                        alpha=0.75)
    ax.add_collection(lc)
    ax.set_aspect("equal")
    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(-1.05, 1.05)
    ax.set_title(title, fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])


def flush_fig():
    fig.suptitle(
        f"OT + spring-polish chain on {SNAPSHOT_LABEL}  "
        f"(Ra={args.Ra:.0e}, Δη={args.delta_eta:.0f}, "
        f"refinement={args.refinement})")
    fig.savefig(PNG_PATH, dpi=160)
    print(f"  → wrote {PNG_PATH}", flush=True)


def mesh_quality_summary(mesh, label):
    """Mean / min normalised cell area as a quick sliver indicator."""
    coords = np.asarray(mesh.X.coords)
    dm = mesh.dm
    cStart, cEnd = dm.getHeightStratum(0)
    pStart, _ = dm.getDepthStratum(0)
    # Each triangle: closure → 3 vertices
    areas = np.empty(cEnd - cStart, dtype=float)
    for k, c in enumerate(range(cStart, cEnd)):
        closure, _ = dm.getTransitiveClosure(c, useCone=True)
        verts = [p for p in closure if pStart <= p < pStart + coords.shape[0]]
        v = [coords[p - pStart] for p in verts[:3]]
        v0, v1, v2 = v
        areas[k] = 0.5 * abs((v1[0] - v0[0]) * (v2[1] - v0[1])
                              - (v2[0] - v0[0]) * (v1[1] - v0[1]))
    mean_a = areas.mean()
    min_a = areas.min()
    print(f"  [{label}] mesh.area mean={mean_a:.4e}  "
          f"min={min_a:.4e}  min/mean={min_a/mean_a:.3f}",
          flush=True)
    return min_a / mean_a


# --- (a) original ---
print("\n--- (a) original ---", flush=True)
mesh_a, T_a, V_a, P_a = load_state()
qual_a = mesh_quality_summary(mesh_a, "original")
vrms_a, nu_a = stokes_solve(mesh_a, T_a, V_a, P_a, "original")
render_panel(axes[0], mesh_a, T_a,
             f"(a) original\nvrms={vrms_a:.3e}  Nu={nu_a:.3f}  "
             f"min/mean={qual_a:.3f}")
flush_fig()

# --- (b) OT × 5 (plain) ---
print("\n--- (b) OT × 5 ---", flush=True)
mesh_b, T_b, V_b, P_b = load_state()
old_X_b = np.asarray(mesh_b.X.coords).copy()
old_T_b = np.asarray(T_b.data).copy()
rho_b = uw.meshing.metric_density_from_gradient(
    mesh_b, T_b, refinement=args.refinement, name="chain_b")
t0 = time.time()
uw.meshing.smooth_mesh_interior(
    mesh_b, metric=rho_b, method="ot",
    boundary_slip=SLIP,
    method_kwargs=dict(n_outer=5, relax=0.1, step_frac=0.3),
    verbose=True)
print(f"  OT×5 wall {time.time() - t0:.1f}s", flush=True)
new_X_b = np.asarray(mesh_b.X.coords).copy()
fe_remap_T(mesh_b, T_b, old_X_b, old_T_b, new_X_b)
qual_b = mesh_quality_summary(mesh_b, "OT×5")
vrms_b, nu_b = stokes_solve(mesh_b, T_b, V_b, P_b, "OT×5")
render_panel(axes[1], mesh_b, T_b,
             f"(b) OT × 5\nvrms={vrms_b:.3e}  Nu={nu_b:.3f}  "
             f"min/mean={qual_b:.3f}")
flush_fig()

# --- (c) OT × 4  →  spring(size_w=2)  →  OT × 1 ---
print("\n--- (c) OT × 4 → spring polish → OT × 1 ---",
      flush=True)
mesh_c, T_c, V_c, P_c = load_state()
old_X_c = np.asarray(mesh_c.X.coords).copy()
old_T_c = np.asarray(T_c.data).copy()
rho_c = uw.meshing.metric_density_from_gradient(
    mesh_c, T_c, refinement=args.refinement, name="chain_c")
# Note: re-evaluate metric source after each move? rho_c is a
# Lagrangian frozen field so the metric tracks the original
# nodes; metric value at each node is fixed when we evaluate D
# on the deformed mesh. That's the same semantics as the
# anisotropic mover's per-outer-step evaluation. Pass the
# *same* rho_c handle into all three steps.
t0 = time.time()
print("  step 1/3: OT × 4", flush=True)
uw.meshing.smooth_mesh_interior(
    mesh_c, metric=rho_c, method="ot",
    boundary_slip=SLIP,
    method_kwargs=dict(n_outer=4, relax=0.1, step_frac=0.3),
    verbose=True)
print("  step 2/3: spring polish (size_w=2)", flush=True)
uw.meshing.smooth_mesh_interior(
    mesh_c, metric=rho_c, method="spring",
    boundary_slip=SLIP,
    method_kwargs=dict(size_w=2.0),
    verbose=True)
print("  step 3/3: OT × 1 (end on OT)", flush=True)
uw.meshing.smooth_mesh_interior(
    mesh_c, metric=rho_c, method="ot",
    boundary_slip=SLIP,
    method_kwargs=dict(n_outer=1, relax=0.1, step_frac=0.3),
    verbose=True)
print(f"  chain wall {time.time() - t0:.1f}s", flush=True)
new_X_c = np.asarray(mesh_c.X.coords).copy()
fe_remap_T(mesh_c, T_c, old_X_c, old_T_c, new_X_c)
qual_c = mesh_quality_summary(mesh_c, "chain")
vrms_c, nu_c = stokes_solve(mesh_c, T_c, V_c, P_c, "chain")
render_panel(
    axes[2], mesh_c, T_c,
    f"(c) OT×4 → spring(size_w=2) → OT×1\n"
    f"vrms={vrms_c:.3e}  Nu={nu_c:.3f}  "
    f"min/mean={qual_c:.3f}")
flush_fig()

print("\n=== SUMMARY ===", flush=True)
print(f"  original  vrms={vrms_a:.3e}  Nu={nu_a:.3f}  "
      f"q={qual_a:.3f}", flush=True)
print(f"  OT×5      vrms={vrms_b:.3e}  Nu={nu_b:.3f}  "
      f"q={qual_b:.3f}", flush=True)
print(f"  chain     vrms={vrms_c:.3e}  Nu={nu_c:.3f}  "
      f"q={qual_c:.3f}", flush=True)

np.savez(
    os.path.join(DIAG,
                 f"ot_chain_{SNAPSHOT_LABEL}"
                 f"_r{int(args.refinement)}_{SLIP_TAG}.npz"),
    original=np.array([vrms_a, nu_a, qual_a]),
    ot5=np.array([vrms_b, nu_b, qual_b]),
    chain=np.array([vrms_c, nu_c, qual_c]),
    refinement=args.refinement,
)
print("done", flush=True)
