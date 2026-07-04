"""Compare meshes between sanity_baseline (no Winslow) and
launchclip_winslow_n5 (Winslow every 5 steps), at the same step numbers.

Renders a grid (rows = step, cols = run) of triplot wireframes so the
effect of the periodic Winslow on the in-flight mesh is visible.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import underworld3 as uw

BASE_DIR = "output/convection_zoo_snapshots_sanity_baseline"
WINSLOW_DIR = "output/convection_zoo_snapshots_launchclip_winslow_n5"
STEPS = (5, 15, 20, 25)
OUT_PNG = "output/winslow_run_vs_baseline_meshes.png"


def get_tri_cells(mesh):
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


def mesh_path(folder, step):
    return os.path.join(folder,
                        f"uw_bdf2_sl_step{step:04d}.mesh.00000.h5")


def load_mesh_and_tris(path):
    print(f"  loading {path}")
    m = uw.discretisation.Mesh(path)
    return m.X.coords.copy(), get_tri_cells(m)


def tri_areas(coords, tris):
    x = coords[:, 0][tris]
    y = coords[:, 1][tris]
    return 0.5 * np.abs(
        (x[:, 1] - x[:, 0]) * (y[:, 2] - y[:, 0])
        - (x[:, 2] - x[:, 0]) * (y[:, 1] - y[:, 0]))


def aspect_ratio(coords, tris):
    """min-altitude / longest-edge — closer to 0.577 (=√3/2) is best."""
    pts = coords[tris]               # (n_tri, 3, 2)
    e01 = pts[:, 1] - pts[:, 0]
    e12 = pts[:, 2] - pts[:, 1]
    e20 = pts[:, 0] - pts[:, 2]
    L = np.linalg.norm(np.stack([e01, e12, e20], axis=1), axis=2)
    Lmax = L.max(axis=1)
    area = tri_areas(coords, tris)
    # min altitude = 2*area / longest edge
    h_min = 2.0 * area / np.maximum(Lmax, 1.0e-12)
    return h_min / np.maximum(Lmax, 1.0e-12)


def main():
    fig, axes = plt.subplots(len(STEPS), 2,
                             figsize=(12, 5.5 * len(STEPS)),
                             constrained_layout=True)
    for i, s in enumerate(STEPS):
        print(f"step {s}:")
        coords_b, tris_b = load_mesh_and_tris(mesh_path(BASE_DIR, s))
        coords_w, tris_w = load_mesh_and_tris(mesh_path(WINSLOW_DIR, s))

        ar_b = aspect_ratio(coords_b, tris_b)
        ar_w = aspect_ratio(coords_w, tris_w)
        print(f"  baseline aspect ratio: "
              f"min={ar_b.min():.3f} median={np.median(ar_b):.3f}")
        print(f"  winslow  aspect ratio: "
              f"min={ar_w.min():.3f} median={np.median(ar_w):.3f}")

        for ax, c, t, title in (
                (axes[i, 0], coords_b, tris_b,
                 f"baseline  step {s}"
                 f"\n(min AR={ar_b.min():.3f})"),
                (axes[i, 1], coords_w, tris_w,
                 f"launch-clip + Winslow  step {s}"
                 f"\n(min AR={ar_w.min():.3f})")):
            ax.triplot(c[:, 0], c[:, 1], t, color="0.2",
                       linewidth=0.35)
            ax.set_aspect("equal")
            ax.set_title(title, fontsize=11)
            ax.set_xticks([])
            ax.set_yticks([])
            for sp in ("top", "right", "bottom", "left"):
                ax.spines[sp].set_visible(False)

    fig.suptitle(
        "Mesh evolution: sanity_baseline (no Winslow) vs "
        "launchclip_winslow_n5 (Winslow every 5 steps)\n"
        "(triangle min altitude / longest edge, larger = better shape)",
        fontsize=13)
    fig.savefig(OUT_PNG, dpi=130, bbox_inches="tight",
                facecolor="white")
    print(f"wrote {OUT_PNG}")


if __name__ == "__main__":
    main()
