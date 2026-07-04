"""Visualise the Winslow smoother before/after on the step-30 sanity mesh.

Loads the deformed mesh from the sanity_baseline snapshot (taken just
before catastrophic ringing started), applies the same Winslow sweep
as the live runner (n_iters=5, alpha=0.5), and renders before/after
side-by-side with matplotlib triplot.

Run:
  pixi run -e amr-dev python -u docs/developer/design/_winslow_viz_step30.py
"""
import numpy as np
import matplotlib.pyplot as plt
import underworld3 as uw
from scipy.sparse import csr_matrix

MESH_FILE = ("output/convection_zoo_snapshots_sanity_baseline/"
             "uw_bdf2_sl_step0030.mesh.00000.h5")
OUT_PNG = "output/winslow_smoother_step30_before_after.png"

# Match live-run Winslow hyperparameters
N_ITERS = 5
ALPHA = 0.5


def winslow_adjacency(mesh, boundary_labels=("Upper", "Lower")):
    """Build row-normalised vertex-vertex adjacency + boundary mask."""
    dm = mesh.dm
    pStart, pEnd = dm.getDepthStratum(0)
    eStart, eEnd = dm.getDepthStratum(1)
    n_verts = pEnd - pStart
    rows, cols = [], []
    for e in range(eStart, eEnd):
        cone = dm.getCone(e)
        if len(cone) != 2:
            continue
        v0, v1 = cone[0] - pStart, cone[1] - pStart
        if 0 <= v0 < n_verts and 0 <= v1 < n_verts:
            rows += [v0, v1]
            cols += [v1, v0]
    rows = np.asarray(rows, dtype=np.int64)
    cols = np.asarray(cols, dtype=np.int64)
    data = np.ones_like(rows, dtype=np.float64)
    A_pat = csr_matrix((data, (rows, cols)), shape=(n_verts, n_verts))
    n_nbr = np.asarray(A_pat.sum(axis=1)).ravel()
    n_nbr_safe = np.where(n_nbr > 0, n_nbr, 1.0)
    inv = 1.0 / n_nbr_safe
    A = csr_matrix((data * inv[rows], (rows, cols)),
                   shape=(n_verts, n_verts))
    is_bd = np.zeros(n_verts, dtype=bool)
    for lname in boundary_labels:
        try:
            label = dm.getLabel(lname)
            if label is None:
                continue
            for val in label.getValueIS().getIndices():
                iset = label.getStratumIS(val)
                if iset is None:
                    continue
                for idx in iset.getIndices():
                    if pStart <= idx < pEnd:
                        is_bd[idx - pStart] = True
        except Exception:
            pass
    return A, is_bd


def get_tri_cells(mesh):
    """Return (n_cells, 3) array of vertex indices per triangle."""
    dm = mesh.dm
    pStart, pEnd = dm.getDepthStratum(0)
    cStart, cEnd = dm.getHeightStratum(0)
    tris = []
    for c in range(cStart, cEnd):
        closure, _ = dm.getTransitiveClosure(c, useCone=True)
        verts = [p - pStart for p in closure
                 if pStart <= p < pEnd]
        if len(verts) == 3:
            tris.append(verts)
    return np.asarray(tris, dtype=np.int64)


def main():
    print(f"loading {MESH_FILE}")
    mesh = uw.discretisation.Mesh(MESH_FILE)
    coords_before = mesh.X.coords.copy()
    tris = get_tri_cells(mesh)
    print(f"mesh: {coords_before.shape[0]} vertices, "
          f"{tris.shape[0]} triangles")

    A, is_bd = winslow_adjacency(mesh)
    is_int = ~is_bd
    print(f"boundary verts: {int(is_bd.sum())}, "
          f"interior verts: {int(is_int.sum())}")

    coords = coords_before.copy()
    for k in range(N_ITERS):
        avg = A @ coords
        coords[is_int] = ((1.0 - ALPHA) * coords[is_int]
                          + ALPHA * avg[is_int])
    coords_after = coords

    delta = np.linalg.norm(coords_after - coords_before, axis=1)
    print(f"max |Δx| at any interior vertex: {delta.max():.3e}")
    print(f"mean |Δx| over interior vertices: "
          f"{delta[is_int].mean():.3e}")

    def tri_areas(coords_, tris_):
        x = coords_[:, 0][tris_]
        y = coords_[:, 1][tris_]
        return 0.5 * np.abs(
            (x[:, 1] - x[:, 0]) * (y[:, 2] - y[:, 0])
            - (x[:, 2] - x[:, 0]) * (y[:, 1] - y[:, 0]))

    area_before = tri_areas(coords_before, tris)
    area_after = tri_areas(coords_after, tris)
    rel_change = (area_after - area_before) / np.maximum(area_before,
                                                          1.0e-12)
    print(f"max |Δarea/area|: {np.abs(rel_change).max():.3f}")
    print(f"mean |Δarea/area|: {np.abs(rel_change).mean():.3f}")

    # Pick the worst surface lobe for the zoom (largest area change)
    centroids = coords_before[tris].mean(axis=1)
    worst_cell = int(np.argmax(np.abs(rel_change)))
    zx, zy = centroids[worst_cell]
    zoom_r = 0.18

    fig = plt.figure(figsize=(16, 9), constrained_layout=True)
    gs = fig.add_gridspec(2, 3,
                          width_ratios=[1, 1, 1],
                          height_ratios=[2, 1])

    ax_b = fig.add_subplot(gs[:, 0])
    ax_a = fig.add_subplot(gs[:, 1])
    ax_zb = fig.add_subplot(gs[0, 2])
    ax_za = fig.add_subplot(gs[1, 2])

    for ax, c, title in (
            (ax_b, coords_before, "Before  (step 30)"),
            (ax_a, coords_after,
             f"After  ({N_ITERS} sweeps, α={ALPHA})")):
        ax.triplot(c[:, 0], c[:, 1], tris,
                   color="0.25", linewidth=0.3)
        ax.set_aspect("equal")
        ax.set_title(title, fontsize=12)
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ("top", "right", "bottom", "left"):
            ax.spines[s].set_visible(False)
        # Highlight the zoom window
        ax.add_patch(plt.Rectangle((zx - zoom_r, zy - zoom_r),
                                   2 * zoom_r, 2 * zoom_r,
                                   fill=False, ec="C3", lw=1.2))

    for ax, c, title in (
            (ax_zb, coords_before, "Zoom — before"),
            (ax_za, coords_after, "Zoom — after")):
        ax.triplot(c[:, 0], c[:, 1], tris,
                   color="0.15", linewidth=0.6)
        ax.set_aspect("equal")
        ax.set_xlim(zx - zoom_r, zx + zoom_r)
        ax.set_ylim(zy - zoom_r, zy + zoom_r)
        ax.set_title(title, fontsize=10, color="C3")
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle(
        "Winslow smoother on step-30 mesh from sanity_baseline run\n"
        f"max |Δx|={delta.max():.2e}, max |Δarea/area|"
        f"={np.abs(rel_change).max():.2f}",
        fontsize=13)
    fig.savefig(OUT_PNG, dpi=140, bbox_inches="tight",
                facecolor="white")
    print(f"wrote {OUT_PNG}")


if __name__ == "__main__":
    main()
