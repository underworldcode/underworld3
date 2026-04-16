# ---
# jupyter:
#   jupytext:
#     formats: py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # H2Ex Workflow — Inspection and Visualization
#
# Loads the VTK output from `Step5-RunPipeline` and provides diagnostic
# cross-section views of each pipeline stage.
#
# **Cross-section approach**: The mesh lives on a spherical shell in
# Cartesian coordinates.  We clip by radius to reveal a shallow depth
# slice (~5-10% of the box depth from the surface), which gives a clean
# plan view of each field without relying on opacity.

# %% [markdown]
# ## 1. Load Data and Setup

# %%
import numpy as np
import pyvista as pv
from pathlib import Path

OUTPUT_DIR = Path("output")

pvmesh = pv.read(str(OUTPUT_DIR / "h2ex_results.vtk"))
print(f"Mesh: {pvmesh.n_points} points, {pvmesh.n_cells} cells")

# Load fault surfaces
fault_files = sorted(OUTPUT_DIR.glob("fault_*.vtk"))
fault_meshes = {}
for f in fault_files:
    fid = f.stem.replace("fault_", "")
    fault_meshes[fid] = pv.read(str(f))
print(f"Loaded {len(fault_meshes)} fault surfaces")

# %%
# Field summary
for name in pvmesh.point_data:
    arr = pvmesh.point_data[name]
    if arr.ndim == 1:
        print(f"  {name:20s}  [{arr.min():.4e}, {arr.max():.4e}]")
    else:
        mag = np.linalg.norm(arr, axis=1)
        print(f"  {name:20s}  |max|={mag.max():.4e}  ({arr.shape[1]}D)")

# %% [markdown]
# ## 2. Cross-Section Setup
#
# The mesh is on a spherical shell (radius ~6320-6372 km in Cartesian).
# Clipping by radius gives plan-view cross-sections at a chosen depth.
# We compute the radius field once and use it for all slices.

# %%
# Compute radius for depth clipping
pts = pvmesh.points
r = np.sqrt(np.sum(pts**2, axis=1))
pvmesh.point_data["radius"] = r

r_surface = r.max()
r_bottom = r.min()
depth_range = r_surface - r_bottom

# Default clip: 7.5% depth from surface (~3.75 km for 50 km box)
CLIP_FRACTION = 0.075
r_clip = r_surface - CLIP_FRACTION * depth_range

print(f"Surface radius: {r_surface:.0f}")
print(f"Bottom radius:  {r_bottom:.0f}")
print(f"Depth range:    {depth_range:.0f} ({depth_range/1000:.1f} km)")
print(f"Clip at {CLIP_FRACTION*100:.1f}% depth: r = {r_clip:.0f} ({CLIP_FRACTION*depth_range/1000:.1f} km below surface)")

# %% [markdown]
# ## Helpers

# %%
FAULT_COLORS = [
    "Red", "Orange", "Blue", "Green", "Purple",
    "Cyan", "Yellow", "Magenta", "Lime", "Pink",
]

def add_faults(plotter, opacity=0.7, show_edges=True):
    """Add all fault surfaces to a plotter."""
    for i, (fid, fmesh) in enumerate(fault_meshes.items()):
        plotter.add_mesh(
            fmesh, style="surface",
            color=FAULT_COLORS[i % len(FAULT_COLORS)],
            opacity=opacity, show_edges=show_edges,
        )


def clip_at_depth(mesh, fraction=CLIP_FRACTION):
    """Clip mesh below a depth fraction, returning a surface cross-section."""
    r_cut = r_surface - fraction * depth_range
    return mesh.clip_scalar(scalars="radius", value=r_cut)

# %% [markdown]
# ## 3. Mesh and Fault Geometry
#
# Adapted mesh with fault distance.  Refinement should be concentrated
# near faults.

# %%
clipped = clip_at_depth(pvmesh)

pl = pv.Plotter(window_size=(1000, 800))
pl.add_mesh(clipped, scalars="fault_dist", cmap="viridis_r",
            show_edges=True, clim=[0,0.1],
            scalar_bar_args={"title": "Distance to nearest fault"})
add_faults(pl)
pl.add_title(f"Fault Distance (clipped at {CLIP_FRACTION*100:.0f}% depth)")
pl.show()

# %% [markdown]
# Fault influence (Gaussian weight): ~1 on faults, decaying to 0.

# %%
pl = pv.Plotter(window_size=(1000, 800))
pl.add_mesh(clipped, scalars="fault_weight", cmap="hot",
            show_edges=False,
            scalar_bar_args={"title": "Fault weight (Gaussian)"})
add_faults(pl, opacity=0.3)
pl.add_title("Fault Influence Weight")
pl.show()

# %% [markdown]
# ## 4. Stokes Velocity and Strain Rate
#
# The TI solve should show E-W shear with perturbation near faults.

# %%
pl = pv.Plotter(window_size=(1000, 800))
pl.add_mesh(clipped, scalars="Vmag", cmap="plasma",
            show_edges=False, clim=[1e-11, 4e-10],
            scalar_bar_args={"title": "|v| (nd)"})

V = clipped.point_data["V"]
v_max = np.linalg.norm(V, axis=1).max()
if v_max > 0:
    pl.add_arrows(clipped.points, V, mag=5000 / v_max,
                  color="White", opacity=0.6)

add_faults(pl, opacity=0.5)
pl.add_title("Stokes Velocity (TI Rheology)")
pl.show()

# %%
# Strain rate invariant — faulted vs reference
pl = pv.Plotter(shape=(1, 1), window_size=(1400, 700))

pl.subplot(0, 0)
pl.add_mesh(clipped, scalars="eps", cmap="RdBu_r",
            show_edges=False,
            clim=(2e3,1e4),
            scalar_bar_args={"title": "Strain rate (faulted)"})
add_faults(pl, opacity=1)
pl.add_title("Faulted")

# pl.subplot(0, 1)
# pl.add_mesh(clipped, scalars="eps_ref", cmap="inferno",
#             show_edges=False,
#             scalar_bar_args={"title": "Strain rate (reference)"})
# pl.add_title("Reference (isotropic)")

pl.link_views()
pl.show()

# %% [markdown]
# ## 5. Strain Rate Anomaly and Permeability
#
# The **delta** (faulted minus reference) shows where faults concentrate
# or shadow deformation.  Permeability is modified only near faults,
# weighted by the Gaussian influence function.

# %%
# Strain rate anomaly — symmetric colormap centred on zero
delta_clip = clipped.point_data["delta_eps"]
delta_absmax = np.percentile(np.abs(delta_clip), 95)  # robust range

pl = pv.Plotter(window_size=(1000, 800))
pl.add_mesh(clipped, scalars="delta_eps", cmap="RdBu_r",
            show_edges=False,
            clim=[-delta_absmax, delta_absmax],
            scalar_bar_args={"title": "Strain rate anomaly"})
add_faults(pl, opacity=0.5)
pl.add_title("Strain Rate Anomaly")
pl.show()

# %%
# Permeability (log10 scale) — should be localised near faults
pl = pv.Plotter(window_size=(1000, 800))
pl.add_mesh(clipped, scalars="log_perm", cmap="RdYlBu_r",
            show_edges=False,
            # clim=[-2, 2],
            scalar_bar_args={"title": "log10(k)"})
add_faults(pl, opacity=0.5)
pl.add_title("Permeability (log10) — fault-localised")
pl.show()

# %% [markdown]
# ## 6. Darcy Flow
#
# Hydraulic head and Darcy velocity.  Flow should be channelled
# upward through high-permeability fault zones.

# %%
pl = pv.Plotter(window_size=(1000, 800))
pl.add_mesh(clipped, scalars="h_darcy", cmap="coolwarm",
            show_edges=False,
            scalar_bar_args={"title": "Hydraulic head"})
add_faults(pl, opacity=0.5)
pl.add_title("Darcy Hydraulic Head")
pl.show()

# %%
# Darcy velocity magnitude + arrows
pl = pv.Plotter(window_size=(1000, 800))
pl.add_mesh(clipped, scalars="Vd_mag", cmap="coolwarm",
            show_edges=True,
            scalar_bar_args={"title": "|v_darcy| (nd)"})

Vd = clipped.point_data["Vd"]
vd_max = np.linalg.norm(Vd, axis=1).max()
if vd_max > 0:
    pl.add_arrows(clipped.points, Vd, mag=10000 / vd_max,
                  color="White", opacity=0.5)

add_faults(pl, opacity=0.5)
pl.add_title("Darcy Velocity")
pl.show()

# %% [markdown]
# ## 7. Depth Variation
#
# Same field at different depth fractions to check vertical structure.

# %%
fractions = [0.05, 0.25, 0.50, 0.75]
pl = pv.Plotter(shape=(2, 2), window_size=(1400, 1100))

for i, frac in enumerate(fractions):
    row, col = divmod(i, 2)
    pl.subplot(row, col)

    depth_km = frac * depth_range / 1000
    cut = clip_at_depth(pvmesh, fraction=frac)
    pl.add_mesh(cut, scalars="log_perm", cmap="RdYlBu_r",
                show_edges=False, clim=[-2, 2],
                scalar_bar_args={"title": f"log10(k)"})
    add_faults(pl, opacity=0.3)
    pl.add_title(f"Depth ~{depth_km:.1f} km ({frac*100:.0f}%)")

pl.link_views()
pl.show()

# %% [markdown]
# ## 8. Combined Overview
#
# Side-by-side: permeability and Darcy velocity at the same depth.

# %%
pl = pv.Plotter(shape=(1, 2), window_size=(1600, 800))

pl.subplot(0, 0)
pl.add_mesh(clipped, scalars="log_perm", cmap="RdYlBu_r",
            show_edges=False, clim=[-2, 2],
            scalar_bar_args={"title": "log10(k)"})
add_faults(pl, opacity=0.5)
pl.add_title("Permeability")

pl.subplot(0, 1)
pl.add_mesh(clipped, scalars="Vd_mag", cmap="viridis",
            show_edges=False,
            scalar_bar_args={"title": "|v_darcy|"})
if vd_max > 0:
    pl.add_arrows(clipped.points, Vd, mag=0.02 / vd_max,
                  color="White", opacity=0.4)
add_faults(pl, opacity=0.5)
pl.add_title("Darcy Velocity")

pl.link_views()
pl.show()

# %%
