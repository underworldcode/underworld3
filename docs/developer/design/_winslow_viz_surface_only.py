"""Plot ONLY the surface (Upper-boundary) vertices for both runs.

If the Winslow run's surface vertices form a smooth curve, then the
'spikes' visible in the triplot view are a triangulation artefact
(Winslow has receded the near-surface interior vertices, so the
boundary-adjacent triangles look thin/spiky) — not a corruption of
the surface itself.

If the Winslow surface vertices are themselves displaced (off-curve,
zig-zag, etc.), the bug is in the Winslow boundary-pinning.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import underworld3 as uw

BASE_DIR = "output/convection_zoo_snapshots_sanity_baseline"
WINSLOW_DIR = "output/convection_zoo_snapshots_launchclip_winslow_n5"
STEPS = (5, 15, 20, 25)
OUT_PNG = "output/winslow_surface_vertices_check.png"


def surface_vertices(mesh):
    """Return (n, 2) array of Upper-boundary vertex coordinates,
    sorted by polar angle."""
    dm = mesh.dm
    pStart, pEnd = dm.getDepthStratum(0)
    label = dm.getLabel("Upper")
    ids_all = []
    for v in label.getValueIS().getIndices():
        iset = label.getStratumIS(int(v))
        if iset is not None:
            ids_all.append(iset.getIndices())
    ids = np.concatenate(ids_all) if ids_all else np.array([], int)
    vert_ids = ids[(ids >= pStart) & (ids < pEnd)] - pStart
    coords = mesh.X.coords[vert_ids]
    theta = np.arctan2(coords[:, 1], coords[:, 0])
    r = np.hypot(coords[:, 0], coords[:, 1])
    order = np.argsort(theta)
    return theta[order], r[order], coords[order]


def main():
    fig, axes = plt.subplots(len(STEPS), 2,
                             figsize=(15, 4 * len(STEPS)),
                             constrained_layout=True)
    for i, s in enumerate(STEPS):
        for j, folder, tag in (
                (0, BASE_DIR, "baseline (no Winslow)"),
                (1, WINSLOW_DIR, "launch-clip + Winslow")):
            path = os.path.join(folder,
                                f"uw_bdf2_sl_step{s:04d}."
                                "mesh.00000.h5")
            mesh = uw.discretisation.Mesh(path)
            theta, r, coords = surface_vertices(mesh)
            print(f"step {s:2d}  {tag:24s}: "
                  f"r_min={r.min():.5f} r_max={r.max():.5f} "
                  f"range={r.max()-r.min():.5f} "
                  f"n_verts={len(theta)}")
            ax = axes[i, j]
            ax.plot(theta, r, "o-", color="C0",
                    markersize=2.5, linewidth=0.9)
            ax.axhline(1.0, color="0.6", linewidth=0.6,
                       linestyle="--")
            ax.set_xlim(-np.pi, np.pi)
            ax.set_ylim(0.9, 1.1)
            ax.grid(alpha=0.3)
            ax.set_title(f"{tag} — step {s}", fontsize=11)
            ax.set_xlabel(r"$\theta$ (rad)")
            ax.set_ylabel(r"$r$")

    fig.suptitle("Upper-boundary vertex radius vs angle\n"
                 "(if both columns are smooth curves, the 'spikes' "
                 "in the triplot view are triangulation artefacts)",
                 fontsize=12)
    fig.savefig(OUT_PNG, dpi=130, bbox_inches="tight",
                facecolor="white")
    print(f"wrote {OUT_PNG}")


if __name__ == "__main__":
    main()
