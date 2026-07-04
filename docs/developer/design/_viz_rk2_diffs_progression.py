"""Compare T-diffs across several consecutive RK2 step pairs to see
if the 'pepper' is growing gradually or appears suddenly at 32→33.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import underworld3 as uw

SNAP_DIR = "output/convection_zoo_snapshots_rk2_full"
OUT_PNG = "output/rk2_diffs_progression.png"

PAIRS = [(29, 30), (30, 31), (31, 32), (32, 33)]
R_INNER, R_OUTER = 0.5, 1.0


def load_T(snap_dir, step):
    root = f"uw_rk2_step{step:04d}"
    mesh = uw.discretisation.Mesh(
        os.path.join(snap_dir, f"{root}.mesh.00000.h5"))
    T = uw.discretisation.MeshVariable(
        "T_conv_v2p1", mesh, vtype=uw.VarType.SCALAR,
        degree=3, continuous=True)
    T.read_timestep(root, "T_conv_v2p1", 0, outputPath=snap_dir)
    return mesh, T


fig, axes = plt.subplots(3, len(PAIRS),
                         figsize=(4 * len(PAIRS), 12),
                         constrained_layout=True)

theta_plot = np.linspace(0, 2 * np.pi, 360)

for j, (s0, s1) in enumerate(PAIRS):
    _, T0 = load_T(SNAP_DIR, s0)
    m1, T1 = load_T(SNAP_DIR, s1)
    T0v = np.asarray(T0.data[:, 0])
    T1v = np.asarray(T1.data[:, 0])
    dT = T1v - T0v
    coords = T1.coords
    abs_dT = np.abs(dT)

    n_001 = int((abs_dT > 0.01).sum())
    n_005 = int((abs_dT > 0.05).sum())
    n_010 = int((abs_dT > 0.10).sum())
    pct_001 = 100 * n_001 / len(dT)
    pct_005 = 100 * n_005 / len(dT)
    print(f"{s0:2d}→{s1:2d}: max|ΔT|={abs_dT.max():.4f}  "
          f"|ΔT|>0.01: {n_001} ({pct_001:.2f}%)  "
          f"|ΔT|>0.05: {n_005} ({pct_005:.2f}%)  "
          f"|ΔT|>0.10: {n_010}")

    # Row 0: ΔT colormap at tight ±0.02 clim
    ax = axes[0, j]
    sc = ax.scatter(coords[:, 0], coords[:, 1], c=dT,
                    cmap="RdBu_r", vmin=-0.02, vmax=0.02, s=2)
    fig.colorbar(sc, ax=ax, orientation="vertical", shrink=0.7,
                 label=r"$\Delta T$ (clim ±0.02)")
    for rr in (R_INNER, R_OUTER):
        ax.plot(rr * np.cos(theta_plot), rr * np.sin(theta_plot),
                color="gray", linewidth=0.4)
    ax.set_aspect("equal")
    ax.set_title(f"step {s0}→{s1}   ΔT (clim ±0.02)",
                 fontsize=11)
    ax.set_xticks([]); ax.set_yticks([])

    # Row 1: |ΔT|>0.05 highlight
    ax = axes[1, j]
    ax.scatter(coords[:, 0], coords[:, 1], c="0.88", s=1)
    big = abs_dT > 0.05
    if big.sum() > 0:
        sc = ax.scatter(coords[big, 0], coords[big, 1], c=dT[big],
                        cmap="RdBu_r", vmin=-0.2, vmax=0.2,
                        s=30, edgecolor="black", linewidth=0.4)
    for rr in (R_INNER, R_OUTER):
        ax.plot(rr * np.cos(theta_plot), rr * np.sin(theta_plot),
                color="gray", linewidth=0.4)
    ax.set_aspect("equal")
    ax.set_title(f"|ΔT|>0.05 DOFs ({int(big.sum())})  ",
                 fontsize=11)
    ax.set_xticks([]); ax.set_yticks([])

    # Row 2: histogram (log y)
    ax = axes[2, j]
    ax.hist(dT, bins=100, range=(-0.15, 0.15),
            color="C2", edgecolor="none")
    ax.set_yscale("log")
    ax.set_xlabel(r"$\Delta T$ per DOF")
    ax.set_ylabel("# DOFs (log)")
    ax.set_title(
        f"max|ΔT|={abs_dT.max():.3f}  >0.05: {n_005}",
        fontsize=11)
    ax.axvline(0, color="black", linewidth=0.5)
    for c in (-0.05, +0.05):
        ax.axvline(c, color="C1", linewidth=0.5, linestyle="--")

fig.suptitle(
    "RK2: per-DOF ΔT across 4 consecutive step pairs — "
    "is the 'pepper' building or appearing discretely?",
    fontsize=13)
fig.savefig(OUT_PNG, dpi=130, bbox_inches="tight",
            facecolor="white")
print(f"wrote {OUT_PNG}")
