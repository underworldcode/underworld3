"""Quantify the 'random T~0.5 patches' observed in steps 32, 33, 34 of
the RK2 run.

Two views per step:
  (a) histogram of T DOF values, log-y, with the [0.3, 0.7] mid-band
      shaded. Anomalous mid-range DOFs that aren't physically explained
      by BL transition will pile up there.
  (b) spatial mask: DOFs with T in [0.3, 0.7] coloured by their T value
      AND a separate mask showing DOFs outside the BL band (radial
      distance > 0.05 from either boundary). Helps see if the mid-T
      DOFs are in plume cores or just in the BL transition zone.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import underworld3 as uw

SNAP_DIR = "output/convection_zoo_snapshots_rk2_full"
STEPS = [31, 32, 33, 34]
OUT_PNG = "output/rk2_onset_anomaly.png"

R_INNER = 0.5
R_OUTER = 1.0
BL_PAD = 0.08   # exclude inner/outer this far from boundary
MID_BAND = (0.3, 0.7)


def load_T(snap_dir, step):
    root = f"uw_rk2_step{step:04d}"
    mesh = uw.discretisation.Mesh(
        os.path.join(snap_dir, f"{root}.mesh.00000.h5"))
    T = uw.discretisation.MeshVariable(
        "T_conv_v2p1", mesh, vtype=uw.VarType.SCALAR,
        degree=3, continuous=True)
    T.read_timestep(root, "T_conv_v2p1", 0, outputPath=snap_dir)
    return mesh, T


fig, axes = plt.subplots(2, len(STEPS),
                         figsize=(4.0 * len(STEPS), 8),
                         constrained_layout=True)

for j, step in enumerate(STEPS):
    mesh, T = load_T(SNAP_DIR, step)
    coords = T.coords
    r = np.hypot(coords[:, 0], coords[:, 1])
    Tv = np.asarray(T.data[:, 0])

    # Histogram, full and zoom on mid-band
    ax = axes[0, j]
    ax.hist(Tv, bins=100, range=(-0.2, 1.2),
            color="C0", edgecolor="none")
    ax.axvspan(MID_BAND[0], MID_BAND[1], color="C1", alpha=0.2)
    ax.set_yscale("log")
    ax.set_xlabel("T value")
    ax.set_ylabel("# DOFs")
    ax.set_title(f"step {step}   "
                 f"T=[{Tv.min():+.3f}, {Tv.max():+.3f}]",
                 fontsize=11)

    # Spatial: highlight mid-band DOFs sitting AWAY from boundary layers
    ax = axes[1, j]
    away = (r > R_INNER + BL_PAD) & (r < R_OUTER - BL_PAD)
    mid = (Tv > MID_BAND[0]) & (Tv < MID_BAND[1])
    plume_core_mid = away & mid

    # Plot full DOF cloud lightly, then plume-core-mid hot
    ax.scatter(coords[:, 0], coords[:, 1], c=Tv, cmap="RdBu_r",
               vmin=0, vmax=1, s=1.5, alpha=0.35)
    if plume_core_mid.sum() > 0:
        ax.scatter(coords[plume_core_mid, 0],
                   coords[plume_core_mid, 1],
                   c=Tv[plume_core_mid], cmap="RdBu_r",
                   vmin=0, vmax=1, s=18, edgecolor="black",
                   linewidth=0.5)

    # Annulus outlines
    theta_plot = np.linspace(0, 2 * np.pi, 360)
    for rr in (R_INNER, R_OUTER):
        ax.plot(rr * np.cos(theta_plot), rr * np.sin(theta_plot),
                color="gray", linewidth=0.5)
    for rr in (R_INNER + BL_PAD, R_OUTER - BL_PAD):
        ax.plot(rr * np.cos(theta_plot), rr * np.sin(theta_plot),
                color="C1", linewidth=0.4, linestyle=":")

    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(
        f"mid-band DOFs in plume cores: "
        f"{int(plume_core_mid.sum())}  / "
        f"{int(away.sum())} core DOFs   "
        f"({100 * plume_core_mid.sum() / max(1, away.sum()):.1f}%)",
        fontsize=10)
    print(f"step {step}: T∈{MID_BAND} in plume cores: "
          f"{int(plume_core_mid.sum())}")

fig.suptitle(
    "RK2 onset: T histogram (top) and spatial location of "
    f"T∈[{MID_BAND[0]}, {MID_BAND[1]}] DOFs in plume cores "
    f"(r more than {BL_PAD} from boundary)",
    fontsize=13)
fig.savefig(OUT_PNG, dpi=130, bbox_inches="tight",
            facecolor="white")
print(f"wrote {OUT_PNG}")
