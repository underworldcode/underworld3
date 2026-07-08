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
"""
# Rock-Only vs Air-Layer (dP1) Comparison

All solves use normalised Gamma_N, penalty=1e4, tol=1e-4, discontinuous pressure.

Three cases:
- Rock-only submesh (extracted via DMPlexFilter)
- Air-layer with eta_air=1e-3
- Air-layer with eta_air=1e-6
"""

# %%
import underworld3 as uw
import underworld3.visualisation as vis
from underworld3.cython.petsc_discretisation import petsc_dm_filter_by_label
from underworld3.discretisation import Mesh
from underworld3.coordinates import CoordinateSystemType
import numpy as np
import sympy
from enum import Enum
from scipy.spatial import cKDTree

if uw.mpi.size == 1:
    import pyvista as pv

# %%
r_inner = 0.5
r_internal = 1.0
r_outer_full = 1.5
cellsize = 1/16

# %% [markdown]
"""
## Create meshes and load checkpoints
"""

# %%
full_mesh = uw.meshing.AnnulusInternalBoundary(
    radiusOuter=r_outer_full, radiusInternal=r_internal,
    radiusInner=r_inner, cellSize=cellsize,
)

subdm = petsc_dm_filter_by_label(full_mesh.dm, "Inner", 101)
subdm.markBoundaryFaces("All_Boundaries", 1001)

class sub_bd(Enum):
    Lower = 1; Internal = 2

rock_mesh = Mesh(subdm, degree=1, qdegree=2, boundaries=sub_bd,
                 coordinate_system_type=CoordinateSystemType.CYLINDRICAL2D)

# Rock-only
v_rock = uw.discretisation.MeshVariable("V_rock", rock_mesh, rock_mesh.dim, degree=2)
p_rock = uw.discretisation.MeshVariable("P_rock", rock_mesh, 1, degree=1, continuous=True)
v_rock.read_timestep("rock", "V", 0, outputPath="../output/normalised_rock/")
p_rock.read_timestep("rock", "P", 0, outputPath="../output/normalised_rock/")
print(f"Rock submesh: {v_rock.data.shape[0]} v-nodes")

# Air-layer eta=1e-3 (dP1)
v_dg3 = uw.discretisation.MeshVariable("V_dg3", full_mesh, full_mesh.dim, degree=2)
p_dg3 = uw.discretisation.MeshVariable("P_dg3", full_mesh, 1, degree=1, continuous=False)
v_dg3.read_timestep("nitsche", "V", 0, outputPath="../output/normalised_nitsche/")
p_dg3.read_timestep("nitsche", "P", 0, outputPath="../output/normalised_nitsche/")
print(f"Air-layer eta=1e-3 (dP1): {v_dg3.data.shape[0]} v-nodes")

# Air-layer eta=1e-5 (dP1, from bootstrap)
v_dg5 = uw.discretisation.MeshVariable("V_dg5", full_mesh, full_mesh.dim, degree=2)
p_dg5 = uw.discretisation.MeshVariable("P_dg5", full_mesh, 1, degree=1, continuous=False)
v_dg5.read_timestep("eta1em05", "V", 0, outputPath="../output/bootstrap_eta1em05/")
p_dg5.read_timestep("eta1em05", "P", 0, outputPath="../output/bootstrap_eta1em05/")
print(f"Air-layer eta=1e-5 (dP1): {v_dg5.data.shape[0]} v-nodes")

# %% [markdown]
"""
## Rock-only: velocity and pressure
"""

# %%
if uw.mpi.size == 1:
    vmag = np.sqrt(v_rock.data[:, 0]**2 + v_rock.data[:, 1]**2)
    vis.plot_vector(rock_mesh, v_rock, vector_name="V_rock", vfreq=1, vmag=2e1,
                    clip_angle=0., cpos="xy", show_arrows=True,
                    clim=[0., float(vmag.max())], cmap="coolwarm")

# %%
if uw.mpi.size == 1:
    pvals = uw.function.evaluate(p_rock.sym[0, 0], p_rock.coords).flatten()
    plim = float(max(abs(pvals.min()), abs(pvals.max())))
    vis.plot_scalar(rock_mesh, p_rock.sym, "P_rock",
                    clip_angle=0., cpos="xy", cmap="RdBu",
                    clim=[-plim, plim])

# %% [markdown]
"""
## Air-layer eta=1e-3 (dP1): velocity and pressure
"""

# %%
if uw.mpi.size == 1:
    vmag3 = np.sqrt(v_dg3.data[:, 0]**2 + v_dg3.data[:, 1]**2)
    vis.plot_vector(full_mesh, v_dg3, vector_name="V_dg3", vfreq=1, vmag=2e1,
                    clip_angle=0., cpos="xy", show_arrows=True,
                    clim=[0., float(vmag3.max())], cmap="coolwarm")

# %%
if uw.mpi.size == 1:
    pvals3 = uw.function.evaluate(p_dg3.sym[0, 0], p_dg3.coords).flatten()
    plim3 = float(max(abs(pvals3.min()), abs(pvals3.max())))
    vis.plot_scalar(full_mesh, p_dg3.sym, "P_dg3",
                    clip_angle=0., cpos="xy", cmap="RdBu",
                    clim=[-plim3, plim3])

# %% [markdown]
"""
## Air-layer eta=1e-5 (dP1): velocity and pressure
"""

# %%
if uw.mpi.size == 1:
    vmag5 = np.sqrt(v_dg5.data[:, 0]**2 + v_dg5.data[:, 1]**2)
    vis.plot_vector(full_mesh, v_dg5, vector_name="V_dg5", vfreq=1, vmag=2e1,
                    clip_angle=0., cpos="xy", show_arrows=True,
                    clim=[0., float(vmag5.max())], cmap="coolwarm")

# %%
if uw.mpi.size == 1:
    pvals5 = uw.function.evaluate(p_dg5.sym[0, 0], p_dg5.coords).flatten()
    plim5 = float(max(abs(pvals5.min()), abs(pvals5.max())))
    vis.plot_scalar(full_mesh, p_dg5.sym, "P_dg5",
                    clip_angle=0., cpos="xy", cmap="RdBu",
                    clim=[-plim5, plim5])

# %% [markdown]
"""
## Overlay: rock-only (blue), eta=1e-3 (red), eta=1e-5 (green)
"""

# %%
if uw.mpi.size == 1:
    tree = cKDTree(v_rock.coords)

    dists3, idx3 = tree.query(v_dg3.coords)
    matched3 = dists3 < 1e-10

    dists5, idx5 = tree.query(v_dg5.coords)
    matched5 = dists5 < 1e-10

    # Rock submesh
    rock_pts = pv.PolyData(np.column_stack([v_rock.coords, np.zeros(len(v_rock.coords))]))
    rock_pts["vectors"] = np.column_stack([v_rock.data, np.zeros(len(v_rock.data))])

    # eta=1e-3 at matched nodes
    c3 = v_dg3.coords[matched3]
    d3 = v_dg3.data[matched3]
    pts3 = pv.PolyData(np.column_stack([c3, np.zeros(len(c3))]))
    pts3["vectors"] = np.column_stack([d3, np.zeros(len(d3))])

    # eta=1e-5 at matched nodes
    c5 = v_dg5.coords[matched5]
    d5 = v_dg5.data[matched5]
    pts5 = pv.PolyData(np.column_stack([c5, np.zeros(len(c5))]))
    pts5["vectors"] = np.column_stack([d5, np.zeros(len(d5))])

    vmax = max(np.sqrt(v_rock.data[:, 0]**2 + v_rock.data[:, 1]**2).max(),
               np.sqrt(d3[:, 0]**2 + d3[:, 1]**2).max(),
               np.sqrt(d5[:, 0]**2 + d5[:, 1]**2).max())
    factor = 0.1 / vmax if vmax > 0 else 1.0

    rock_arrows = rock_pts.glyph(orient="vectors", scale="vectors", factor=factor)
    arrows3 = pts3.glyph(orient="vectors", scale="vectors", factor=factor)
    arrows5 = pts5.glyph(orient="vectors", scale="vectors", factor=factor)

    pl = pv.Plotter()
    pl.add_mesh(rock_arrows, color="blue", opacity=0.7, label="Rock-only submesh")
    #pl.add_mesh(arrows3, color="red", opacity=0.7, label="Air-layer eta=1e-3")
    pl.add_mesh(arrows5, color="green", opacity=0.7, label="Air-layer eta=1e-5")

    theta = np.linspace(0, 2*np.pi, 200)
    circle = pv.lines_from_points(np.column_stack([
        1.0 * np.cos(theta), 1.0 * np.sin(theta), np.zeros(200)
    ]))
    pl.add_mesh(circle, color="black", line_width=2)

    pl.add_legend()
    pl.camera_position = "xy"
    pl.show()

# %% [markdown]
"""
## Norm comparison at matched nodes
"""

# %%
tree = cKDTree(v_rock.coords)

dists3, idx3 = tree.query(v_dg3.coords)
matched3 = dists3 < 1e-10

dists5, idx5 = tree.query(v_dg5.coords)
matched5 = dists5 < 1e-10

v_ref = v_rock.data
v3_m = v_dg3.data[matched3]
v5_m = v_dg5.data[matched5]
v_ref3 = v_ref[idx3[matched3]]
v_ref5 = v_ref[idx5[matched5]]

def l2(a, b):
    return np.sqrt(np.sum((a - b)**2)) / np.sqrt(np.sum(b**2))

print(f"Matched: eta=1e-3: {matched3.sum()} nodes, eta=1e-5: {matched5.sum()} nodes")
print()
print(f"{'Metric':<22} {'eta=1e-3':>12} {'eta=1e-5':>12}")
print("-" * 48)
print(f"{'Velocity L2 rel':<22} {l2(v3_m, v_ref3):>12.4e} {l2(v5_m, v_ref5):>12.4e}")

vmag_r3 = np.sqrt(v_ref3[:, 0]**2 + v_ref3[:, 1]**2)
vmag_3 = np.sqrt(v3_m[:, 0]**2 + v3_m[:, 1]**2)
vmag_r5 = np.sqrt(v_ref5[:, 0]**2 + v_ref5[:, 1]**2)
vmag_5 = np.sqrt(v5_m[:, 0]**2 + v5_m[:, 1]**2)
print(f"{'|v| ratio (air/rock)':<22} {vmag_3.mean()/vmag_r3.mean():>12.4f} {vmag_5.mean()/vmag_r5.mean():>12.4f}")
