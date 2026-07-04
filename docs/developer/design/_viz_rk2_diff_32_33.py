"""Diff T(step 33) - T(step 32) for the RK2 run.

Plots:
  Row 1: T at step 32 and step 33 (reference) on their own meshes.
  Row 2: ΔT = T(33) - T(32) per DOF index, plotted at step-33 DOF
         positions. Two color ranges: wide (|ΔT|≤0.1) and tight
         (|ΔT|≤0.02) to expose structure at different scales.
  Row 3: histogram of ΔT, log-y; spatial location of |ΔT|>0.05 DOFs
         marked.

Caveat: positions of DOF i differ slightly between step 32 and 33
(mesh has deformed between steps). So part of the diff is "physical
evolution at moved DOF" plus "interpolation across the mesh motion".
Random patches *unexplained* by either would be the corruption signal.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import underworld3 as uw

SNAP_DIR = "output/convection_zoo_snapshots_rk2_full"
OUT_PNG = "output/rk2_diff_32_33.png"

R_INNER = 0.5
R_OUTER = 1.0


def load_T(snap_dir, step):
    root = f"uw_rk2_step{step:04d}"
    mesh = uw.discretisation.Mesh(
        os.path.join(snap_dir, f"{root}.mesh.00000.h5"))
    T = uw.discretisation.MeshVariable(
        "T_conv_v2p1", mesh, vtype=uw.VarType.SCALAR,
        degree=3, continuous=True)
    T.read_timestep(root, "T_conv_v2p1", 0, outputPath=snap_dir)
    return mesh, T


m32, T32 = load_T(SNAP_DIR, 32)
m33, T33 = load_T(SNAP_DIR, 33)
T32v = np.asarray(T32.data[:, 0])
T33v = np.asarray(T33.data[:, 0])
dT = T33v - T32v
coords33 = T33.coords

print(f"T32 range: [{T32v.min():.4f}, {T32v.max():.4f}]")
print(f"T33 range: [{T33v.min():.4f}, {T33v.max():.4f}]")
print(f"|ΔT| stats: max={np.abs(dT).max():.4f}, "
      f"mean={np.abs(dT).mean():.4e}, "
      f"99-percentile={np.quantile(np.abs(dT), 0.99):.4f}")
print(f"DOFs with |ΔT|>0.01: {int((np.abs(dT) > 0.01).sum())}")
print(f"DOFs with |ΔT|>0.05: {int((np.abs(dT) > 0.05).sum())}")
print(f"DOFs with |ΔT|>0.1:  {int((np.abs(dT) > 0.1).sum())}")
print(f"DOFs with ΔT<-0.05:  {int((dT < -0.05).sum())}")
print(f"DOFs with ΔT>+0.05:  {int((dT > +0.05).sum())}")

fig, axes = plt.subplots(2, 3, figsize=(15, 10),
                         constrained_layout=True)

# Row 1, col 0: T at step 32
ax = axes[0, 0]
sc = ax.scatter(T32.coords[:, 0], T32.coords[:, 1], c=T32v,
                cmap="RdBu_r", vmin=0, vmax=1, s=1.5)
fig.colorbar(sc, ax=ax, orientation="vertical", shrink=0.7,
             label="T")
ax.set_aspect("equal"); ax.set_title("T at step 32")
ax.set_xticks([]); ax.set_yticks([])

# Row 1, col 1: T at step 33
ax = axes[0, 1]
sc = ax.scatter(coords33[:, 0], coords33[:, 1], c=T33v,
                cmap="RdBu_r", vmin=0, vmax=1, s=1.5)
fig.colorbar(sc, ax=ax, orientation="vertical", shrink=0.7,
             label="T")
ax.set_aspect("equal"); ax.set_title("T at step 33")
ax.set_xticks([]); ax.set_yticks([])

# Row 1, col 2: histogram of ΔT
ax = axes[0, 2]
ax.hist(dT, bins=120, range=(-0.3, 0.3),
        color="C2", edgecolor="none")
ax.set_yscale("log")
ax.set_xlabel(r"$\Delta T = T_{33} - T_{32}$ per DOF")
ax.set_ylabel("# DOFs (log)")
ax.set_title(r"$\Delta T$ histogram")
ax.axvline(0, color="black", linewidth=0.5)
for c in (-0.05, +0.05):
    ax.axvline(c, color="C1", linewidth=0.5, linestyle="--")

# Row 2, col 0: ΔT wide range
ax = axes[1, 0]
sc = ax.scatter(coords33[:, 0], coords33[:, 1], c=dT,
                cmap="RdBu_r", vmin=-0.1, vmax=0.1, s=2)
fig.colorbar(sc, ax=ax, orientation="vertical", shrink=0.7,
             label=r"$\Delta T$ (clim ±0.1)")
ax.set_aspect("equal")
ax.set_title(r"$\Delta T$ per DOF (clim ±0.1)")
ax.set_xticks([]); ax.set_yticks([])

# Row 2, col 1: ΔT tight range to expose small structure
ax = axes[1, 1]
sc = ax.scatter(coords33[:, 0], coords33[:, 1], c=dT,
                cmap="RdBu_r", vmin=-0.02, vmax=0.02, s=2)
fig.colorbar(sc, ax=ax, orientation="vertical", shrink=0.7,
             label=r"$\Delta T$ (clim ±0.02)")
ax.set_aspect("equal")
ax.set_title(r"$\Delta T$ per DOF (clim ±0.02)")
ax.set_xticks([]); ax.set_yticks([])

# Row 2, col 2: highlight |ΔT|>0.05 DOFs
ax = axes[1, 2]
big = np.abs(dT) > 0.05
ax.scatter(coords33[:, 0], coords33[:, 1], c="0.85", s=1.5)
if big.sum() > 0:
    sc = ax.scatter(coords33[big, 0], coords33[big, 1], c=dT[big],
                    cmap="RdBu_r", vmin=-0.2, vmax=0.2, s=30,
                    edgecolor="black", linewidth=0.4)
    fig.colorbar(sc, ax=ax, orientation="vertical", shrink=0.7,
                 label=r"$\Delta T$ (highlighted)")
ax.set_aspect("equal")
ax.set_title(f"|ΔT|>0.05 DOFs ({int(big.sum())})")
ax.set_xticks([]); ax.set_yticks([])

# Boundary outlines
theta_plot = np.linspace(0, 2 * np.pi, 360)
for ax in axes.flat[3:]:
    for rr in (R_INNER, R_OUTER):
        ax.plot(rr * np.cos(theta_plot), rr * np.sin(theta_plot),
                color="gray", linewidth=0.4)

fig.suptitle(
    "RK2 step 32 → 33: T fields and per-DOF Δ"
    f"   (max |ΔT|={np.abs(dT).max():.3f}, "
    f"|ΔT|>0.05 DOFs: {int(big.sum())})",
    fontsize=13)
fig.savefig(OUT_PNG, dpi=130, bbox_inches="tight",
            facecolor="white")
print(f"wrote {OUT_PNG}")
