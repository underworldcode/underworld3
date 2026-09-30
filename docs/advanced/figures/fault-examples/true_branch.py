"""A true Y-branch, bracketed: does the near-miss tributary capture it?

A genuine branch (three arms meeting at a point) cannot be split — but
it can be DECOMPOSED two ways, and the pair brackets the truth:

- A: the TRUNK is continuous (west + east arms as one fault); the
  splay abuts, stopping a ligament short.
- B: the BENT fault is continuous (west arm + splay as one deliberately
  kinked polyline — the kink response is the physics, so no smoothed
  normal); the east arm abuts.

Each decomposition welds a different pair of arms through the junction
and feeds the third across a ligament. If both give the same slip on
every arm at a small gap, the offset representation reproduces the true
branch; sweeping the gap in decomposition A shows how close "close by"
has to be. All three arms slip freely under the same drive.
"""
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv

import underworld3 as uw
from underworld3.meshing.surfaces import prepare_fault_network
from underworld3.utilities import fault_contact

import common

pv.OFF_SCREEN = True
D = os.path.dirname(os.path.abspath(__file__))
H = 0.012
MU_P = 0.4
TREND = np.degrees(np.arctan2(0.10, 0.70))

J = np.array([0.50, 0.50])                       # the branch point
W_END = np.array([0.15, 0.45])
E_END = np.array([0.85, 0.55])
S_END = np.array([0.80, 0.76])
ARMS = {"west": (W_END - J) / np.linalg.norm(W_END - J),
        "east": (E_END - J) / np.linalg.norm(E_END - J),
        "splay": (S_END - J) / np.linalg.norm(S_END - J)}


def decomposition(kind, lig):
    if kind == "A":                              # trunk continuous
        faults = [("Trunk", np.array([W_END, E_END])),
                  ("Splay", np.array([J, S_END]))]
        through = ["Trunk"]
    else:                                        # bent fault continuous
        faults = [("Bent", np.array([W_END, J, S_END])),
                  ("East", np.array([J, E_END]))]
        through = ["Bent"]
    return prepare_fault_network(faults, spacing=H, ligament=lig,
                                 through=through, verbose=False)


ETA_WELD = 200.0 * common.ETA / 0.2


def _stress_p0(child, stokes, tagc):
    x, y = child.X
    v, p = stokes.Unknowns.u, stokes.Unknowns.p
    comps = {}
    for cname, expr in (
            ("sxx", -p.sym[0] + 2 * common.ETA * v.sym[0].diff(x)),
            ("syy", -p.sym[0] + 2 * common.ETA * v.sym[1].diff(y)),
            ("sxy", common.ETA * (v.sym[0].diff(y) + v.sym[1].diff(x)))):
        s_var = uw.discretisation.MeshVariable(f"{cname}_{tagc}", child, 1,
                                               degree=0, continuous=False)
        proj = uw.systems.Projection(child, s_var)
        proj.uw_function = expr
        proj.smoothing = 0.0
        proj.solve()
        row = common.split_mesh_cell_rows(child, s_var)
        comps[cname] = np.asarray(s_var.data[:, 0])[row].copy()
    return comps


def run(tag, prepared, want_field=False):
    child = common.base_mesh(H).add_fault(prepared)
    stokes = common.stokes_on(
        child, common.boundary_simple_shear(child, TREND))
    for n, _p in prepared:
        stokes.add_fault_bc(0, boundary=n)
    t0 = time.perf_counter()
    fault_contact.solve_with_fault(stokes, picard=2)

    field = None
    if want_field:
        c1 = _stress_p0(child, stokes, f"a{run.n}")
        s0 = common.stokes_on(
            child, common.boundary_simple_shear(child, TREND))
        for n, _p in prepared:
            s0.add_fault_bc(ETA_WELD, boundary=n)
        fault_contact.solve_with_fault(s0, picard=2)
        c0 = _stress_p0(child, s0, f"b{run.n}")
        run.n += 1
        beta = np.radians(TREND)
        nx, ny = -np.sin(beta), np.cos(beta)
        tx, ty = np.cos(beta), np.sin(beta)

        def resolve(c):
            s_nn = (c["sxx"] * nx * nx + 2 * c["sxy"] * nx * ny
                    + c["syy"] * ny * ny)
            s_t = (c["sxx"] * tx * nx + c["sxy"] * (tx * ny + ty * nx)
                   + c["syy"] * ty * ny)
            return s_nn, s_t

        nn0, t_0 = resolve(c0)
        nn1, t_1 = resolve(c1)
        dcff = np.sign(np.median(t_0)) * (t_1 - t_0) + MU_P * (nn1 - nn0)
        pts, faces = common.split_mesh_cell_render(child)
        fc = np.asarray(faces).reshape(-1, 4)[:, 1:]
        cent = np.asarray(pts)[fc].mean(axis=1)
        dcff, gauge = common.far_field_anchor(
            cent, dcff, [p for _n, p in prepared], cut=0.18)
        print(f"[{tag}] gauge {gauge:+.4f}", flush=True)
        field = (pts, faces, dcff)
    arms = {k: ([], []) for k in ARMS}
    for n, _p in prepared:
        coords, jumps, normals = fault_contact.fault_pair_jumps(
            stokes, n, stokes._rotated_freeslip_info)
        if not len(coords):
            continue
        tang = np.column_stack([-normals[:, 1], normals[:, 0]])
        V = np.abs(np.einsum("ij,ij->i", jumps, tang))
        R = coords - J
        # classify each pair node by the arm it lies along
        proj = {k: R @ t for k, t in ARMS.items()}
        dist2 = {k: np.einsum("ij,ij->i", R - np.outer(
            np.clip(proj[k], 0, None), ARMS[k]),
            R - np.outer(np.clip(proj[k], 0, None), ARMS[k]))
            for k in ARMS}
        best = np.argmin(np.vstack([dist2[k] for k in ARMS]), axis=0)
        for i, k in enumerate(ARMS):
            sel = best == i
            arms[k][0].extend(proj[k][sel])
            arms[k][1].extend(V[sel])
    out = {}
    for k in ARMS:
        s = np.asarray(arms[k][0])
        v = np.asarray(arms[k][1])
        order = np.argsort(s)
        out[k] = (s[order], v[order])
    print(f"[{tag}] peaks " + "  ".join(
        f"{k} {out[k][1].max():.4f}" for k in ARMS)
        + f"  ({time.perf_counter() - t0:.1f} s)", flush=True)
    return out, field, prepared


run.n = 0

cases = [("A, gap 1h", "A", 1.0, "#1565c0", "-"),
         ("A, gap 2h", "A", 2.0, "#64b5f6", "--"),
         ("A, gap 4h", "A", 4.0, "#b3d7f7", ":"),
         ("B, gap 1h", "B", 1.0, "#c62828", "-")]
profiles, fields = {}, {}
for tag, kind, lig, _c, _ls in cases:
    prepared, _rep = decomposition(kind, lig)
    want = tag in ("A, gap 1h", "B, gap 1h")
    profiles[tag], field, prep = run(tag, prepared, want_field=want)
    if field is not None:
        fields[tag] = (field, prep)

fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.3), sharey=True)
for ax, arm in zip(axes, ("west", "east", "splay")):
    for tag, _k, _l, col, ls in cases:
        s, v = profiles[tag][arm]
        ax.plot(s, v, ls, lw=1.4, color=col, label=tag)
    ax.set_title(f"the {arm} arm", fontsize=10.5)
    ax.set_xlabel("distance from the branch point")
axes[0].set_ylabel("|slip|")
axes[0].legend(fontsize=8, title="decomposition, ligament")
fig.suptitle(
    "One true Y-branch, two decompositions: continuous-trunk (A) vs "
    "continuous-bend (B).\nAgreement between A and B at small gap = the "
    "near-miss tributary reproduces the true branch", fontsize=10.5)
fig.tight_layout()
out = os.path.join(D, "true-branch.png")
fig.savefig(out, dpi=200)
print("wrote", out, flush=True)

# ---- the stress maps: the kink-lock made visible ---------------------------


def render(field, prepared, png, scale, focal):
    pts, faces, dcff = field
    pvm = pv.PolyData(np.asarray(pts, dtype=float),
                      faces=np.asarray(faces, dtype=np.int64))
    pvm.cell_data["dcff"] = dcff
    pl = pv.Plotter(off_screen=True, window_size=(850, 800))
    pl.set_background("white")
    pl.add_mesh(pvm, scalars="dcff", cmap="RdBu_r", clim=(-1.0, 1.0),
                lighting=False, show_scalar_bar=False)
    for n, p in prepared:
        line = pv.lines_from_points(
            np.column_stack([p, np.full(len(p), 1e-3)]))
        pl.add_mesh(line, color="black", line_width=4, lighting=False)
    pl.view_xy()
    pl.camera.parallel_projection = True
    pl.camera.parallel_scale = scale
    pl.camera.focal_point = (focal[0], focal[1], 0.0)
    pl.screenshot(png)
    pl.close()
    return png


panels = []
for tag, sub in (("A, gap 1h", "a"), ("B, gap 1h", "b")):
    field, prep = fields[tag]
    panels.append((render(field, prep, os.path.join(
        D, f"_tb_map_{sub}.png"), 0.40, (0.5, 0.55)),
        f"{tag.split(',')[0]}: "
        + ("trunk continuous" if sub == "a" else "bend continuous")))
    panels.append((render(field, prep, os.path.join(
        D, f"_tb_zoom_{sub}.png"), 0.09, J),
        "the branch point"))

fig = plt.figure(figsize=(12.8, 11.6))
gs = fig.add_gridspec(2, 3, width_ratios=[1.25, 1.0, 0.05])
for k, (png, title) in enumerate(panels):
    ax = fig.add_subplot(gs[k // 2, k % 2])
    ax.imshow(plt.imread(png))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=10.5)
from matplotlib import cm, colors as mcolors
sm = cm.ScalarMappable(norm=mcolors.Normalize(-1, 1), cmap="RdBu_r")
cax = fig.add_subplot(gs[:, 2])
fig.colorbar(sm, cax=cax, label=r"$\Delta$CFF (per unit stress drop)")
fig.suptitle("The same Y-branch, two continuity choices: "
             r"$\Delta$CFF on trunk-parallel planes", fontsize=12)
fig.tight_layout()
out2 = os.path.join(D, "true-branch-stress.png")
fig.savefig(out2, dpi=200)
print("wrote", out2, flush=True)
