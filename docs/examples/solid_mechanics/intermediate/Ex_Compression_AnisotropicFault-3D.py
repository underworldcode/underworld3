# %% [markdown]
"""
# 🔬 Compression AnisotropicFault-3D

**PHYSICS:** solid_mechanics  
**DIFFICULTY:** intermediate  
**MIGRATED:** From underworld3-documentation/Notebooks

## Description
This example has been migrated from the original UW3 documentation.
Additional documentation and parameter annotations will be added.

## Migration Notes
- Original complexity preserved
- Parameters to be extracted and annotated
- Claude hints to be added in future update
"""

# %% [markdown]
"""
## Original Code
The following is the migrated code with minimal modifications.
"""

# %%
# ---
# jupyter:
#   jupytext:
#     formats: py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.6
#   kernelspec:
#     display_name: Python (Pixi)
#     language: python
#     name: pixi-kernel-python3
# ---

# %% [markdown]
# ## Anisotropic, embedded-fault model 3D
#
# Describe faults by a point cloud - use pyvista to triangulate
#
#

# %%
import nest_asyncio

nest_asyncio.apply()


# %%
import os

os.environ["UW_TIMING_ENABLE"] = "1"

import numpy as np
import petsc4py
import pyvista as pv
import sympy
import underworld3 as uw
from underworld3 import timing

# %%
mesh = uw.meshing.UnstructuredSimplexBox(
    minCoords=(-2, 0, 0), maxCoords=(2, 1, 1), cellSize=0.1, qdegree=3, regular=False
)

v_soln = uw.discretisation.MeshVariable("U", mesh, mesh.dim, degree=2)
p_soln = uw.discretisation.MeshVariable("P", mesh, 1, degree=1)
strain_rate_inv2 = uw.discretisation.MeshVariable("eps", mesh, 1, degree=1)
fault_dist = uw.discretisation.MeshVariable("df", mesh, 1, degree=2, continuous=False)
fault_norm = uw.discretisation.MeshVariable(
    "nf", mesh, mesh.dim, degree=1, continuous=True, varsymbol=r"{\hat{n}}"
)


# %%
points_list = []

fxy = lambda x, y: (x+0.7*y**2)**2 

for y in np.linspace(0.0, 1.0, 20):
    for x in np.linspace(-0.7*y**2, 1.0-0.7*y**2, 20):
        points_list.append((x, y, fxy(x, y)))

points = np.array(points_list)

pv_fpoints = pv.PolyData(points+np.array((-0.5, 0.0,0.0)))
surf2d = pv_fpoints.delaunay_2d(tol=0.01)
surf2d.compute_normals(inplace=True)

pv_fpoints = pv.PolyData(points+np.array((0.5, 0.0,0.0)))
surf2d_1 = pv_fpoints.delaunay_2d(tol=0.01)
surf2d_1.compute_normals(inplace=True)

shallow_points = points[np.where(points[:,2] > 0.5)]

pv_fpoints = pv.PolyData(shallow_points+np.array((-1.25, 0.0,0.0)))
surf2d_2 = pv_fpoints.delaunay_2d(tol=0.01)
surf2d_2.compute_normals(inplace=True)

pv_fpoints = pv.PolyData(shallow_points+np.array((-2.0, 0.0,0.0)))
surf2d_3 = pv_fpoints.delaunay_2d(tol=0.01)
surf2d_3.compute_normals(inplace=True)

pass

# %%

# %%

# %%
## Fault distance function (using pyvista)

sample_points = pv.PolyData(fault_dist.coords)
pv_mesh_d = sample_points.compute_implicit_distance(surf2d)
sample_points.point_data["df"] = pv_mesh_d.point_data["implicit_distance"]

fault_dist.data[:, 0] = np.abs(sample_points.point_data["df"])

pv_mesh_d = sample_points.compute_implicit_distance(surf2d_1)
sample_points.point_data["df"] = pv_mesh_d.point_data["implicit_distance"]

fault_dist.data[:, 0] = np.minimum(np.abs(sample_points.point_data["df"]), fault_dist.data[:, 0])

pv_mesh_d = sample_points.compute_implicit_distance(surf2d_2)
sample_points.point_data["df"] = pv_mesh_d.point_data["implicit_distance"]

fault_dist.data[:, 0] = np.minimum(np.abs(sample_points.point_data["df"]), fault_dist.data[:, 0])

pv_mesh_d = sample_points.compute_implicit_distance(surf2d_3)
sample_points.point_data["df"] = pv_mesh_d.point_data["implicit_distance"]

fault_dist.data[:, 0] = np.minimum(np.abs(sample_points.point_data["df"]), fault_dist.data[:, 0])


# %%
## Map fault normals (computed by pyvista)

surf2D_tree = uw.kdtree.KDTree(surf2d.points)

closest_points, dist_sq, _ = surf2D_tree.find_closest_point(fault_norm.coords)
dist = np.sqrt(dist_sq)
mask = dist < mesh.get_min_radius() * 2.5
fault_norm.data[mask] = surf2d.point_data["Normals"][closest_points[mask]]

surf2D_tree = uw.kdtree.KDTree(surf2d_1.points)

closest_points, dist_sq, _ = surf2D_tree.find_closest_point(fault_norm.coords)
dist = np.sqrt(dist_sq)
mask = dist < mesh.get_min_radius() * 2.5
fault_norm.data[mask] = surf2d_1.point_data["Normals"][closest_points[mask]]

surf2D_tree = uw.kdtree.KDTree(surf2d_2.points)

closest_points, dist_sq, _ = surf2D_tree.find_closest_point(fault_norm.coords)
dist = np.sqrt(dist_sq)
mask = dist < mesh.get_min_radius() * 2.5
fault_norm.data[mask] = surf2d_2.point_data["Normals"][closest_points[mask]]


surf2D_tree = uw.kdtree.KDTree(surf2d_3.points)

closest_points, dist_sq, _ = surf2D_tree.find_closest_point(fault_norm.coords)
dist = np.sqrt(dist_sq)
mask = dist < mesh.get_min_radius() * 2.5
fault_norm.data[mask] = surf2d_3.point_data["Normals"][closest_points[mask]]



# %%
import underworld3.visualisation as vis

pv_mesh = vis.mesh_to_pv_mesh(mesh)
pv_mesh.point_data["norm"] = uw.function.evalf(fault_norm.sym, pv_mesh.points)
pv_mesh.point_data["dist"] = uw.function.evalf(fault_dist.sym, pv_mesh.points)

pv_mesh_clipped = pv_mesh.clip(normal="y", origin=(0,0.5,0))

pl = pv.Plotter(window_size=[1000, 1000])

pl.add_mesh(pv_mesh, style="wireframe")
pl.add_mesh(pv_mesh_clipped, scalars="dist", cmap="RdBu_r", clim=(0,0.25))


# pl.add_points(sample_points, scalars="df", cmap="RdBu", clim=(-0.5,0.5))
pl.add_arrows(pv_mesh.points, pv_mesh.point_data["norm"], mag=0.1)
pl.add_mesh(surf2d)
pl.add_mesh(surf2d_1, color="Red")
pl.add_mesh(surf2d_2, color="Blue")
pl.add_mesh(surf2d_3, color="Green")

pl.show()

# %%
## Solver

## Now determine how the problem will be set up: Stokes (viscous) solve to compute stresses
## in which faults will appear as weak zones (could be elastic / damage or viscous / damage)

stokes = uw.systems.Stokes(mesh, velocityField=v_soln, pressureField=p_soln)

stokes.constitutive_model = uw.constitutive_models.TransverseIsotropicFlowModel
stokes.constitutive_model.Parameters.eta_0 = 1
stokes.constitutive_model.Parameters.eta_1 = 0 + sympy.Piecewise(
    (0.001, fault_dist.sym[0] < mesh.get_min_radius() * 2),
    (1, True))

stokes.constitutive_model.Parameters.director = fault_norm.sym

stokes.penalty = 1.0
stokes.saddle_preconditioner = sympy.simplify(
    1 / (stokes.constitutive_model.viscosity + stokes.penalty)
)

stokes.add_essential_bc(
    [1, 0, 0],
    mesh.boundaries.Left.name)
stokes.add_essential_bc(
    [0, 0, 0],
    mesh.boundaries.Right.name)
stokes.add_essential_bc(
    [None, None, 0],
    mesh.boundaries.Bottom.name)
stokes.add_essential_bc(
    [None, 0, None],
    mesh.boundaries.Front.name)
stokes.add_essential_bc(
    [None, 0, None],
    mesh.boundaries.Back.name)

stokes.bodyforce = -1 * mesh.CoordinateSystem.unit_k

# %%
stokes.petsc_options["ksp_monitor"] = None
stokes.petsc_options["snes_monitor"] = None
stokes.petsc_options["snes_atol"] = 1.0e-4

stokes.petsc_options["fieldsplit_velocity_ksp_type"] = "cg"
stokes.petsc_options["fieldsplit_velocity_pc_type"] = "mg"

stokes.petsc_options["fieldsplit_pressure_ksp_type"] = "gmres"
stokes.petsc_options["fieldsplit_pressure_pc_type"] = "mg"

timing.reset()
timing.start()

stokes.solve(zero_init_guess=False)

timing.print_table(display_fraction=0.999)


# %%
nodal_strain_rate_inv2 = uw.systems.Projection(mesh, strain_rate_inv2)
nodal_strain_rate_inv2.uw_function = stokes.Unknowns.Einv2
nodal_strain_rate_inv2.smoothing = 1.0e-2
nodal_strain_rate_inv2.petsc_options["ksp_monitor"] = None
nodal_strain_rate_inv2.petsc_options["snes_monitor"] = None

nodal_strain_rate_inv2.solve()

# %%
import underworld3.visualisation as vis

pv_mesh = vis.mesh_to_pv_mesh(mesh)
pv_mesh.point_data["norm"] = uw.function.evalf(fault_norm.sym, pv_mesh.points)
pv_mesh.point_data["D"] = uw.function.evalf(fault_dist.sym, pv_mesh.points)
pv_mesh.point_data["V"] = uw.function.evalf(v_soln.sym, pv_mesh.points)
pv_mesh.point_data["Edot"] = uw.function.evalf(strain_rate_inv2.sym, pv_mesh.points)

pv_mesh_clip = pv_mesh.clip(normal="y", origin=(0, 0.5, 0))

surf2d.point_data["Edot"] = uw.function.evalf(strain_rate_inv2.sym, surf2d.points)
surf2d.point_data["V"] = uw.function.evalf(v_soln.sym, surf2d.points)

surf2d_1.point_data["Edot"] = uw.function.evalf(strain_rate_inv2.sym, surf2d_1.points)
surf2d_1.point_data["V"] = uw.function.evalf(v_soln.sym, surf2d_1.points)

surf2d_2.point_data["Edot"] = uw.function.evalf(strain_rate_inv2.sym, surf2d_2.points)
surf2d_2.point_data["V"] = uw.function.evalf(v_soln.sym, surf2d_2.points)

surf2d_3.point_data["Edot"] = uw.function.evalf(strain_rate_inv2.sym, surf2d_3.points)
surf2d_3.point_data["V"] = uw.function.evalf(v_soln.sym, surf2d_3.points)



pl = pv.Plotter(window_size=[1000, 1000])

pl.add_mesh(pv_mesh, style="wireframe", color="Grey", opacity=0.25)
# pl.add_mesh(pv_mesh_clip, style="surface", scalars="D", cmap="RdBu")

# pl.add_points(sample_points, scalars="df", cmap="RdBu", clim=(-0.5,0.5))
# pl.add_arrows(pv_mesh.points, pv_mesh.point_data["norm"], mag=0.1)

pl.add_arrows(pv_mesh.points, pv_mesh.point_data["V"], mag=0.2, opacity=0.3)


pl.add_mesh(surf2d, cmap="RdBu_r", scalars="Edot", clim=(0,0.75))
#pl.add_arrows(surf2d.points, surf2d.point_data["V"], mag=0.3)

pl.add_mesh(surf2d_1, cmap="RdBu_r", scalars="Edot", clim=(0,0.75))
#pl.add_arrows(surf2d_1.points, surf2d_1.point_data["V"], mag=0.3)

pl.add_mesh(surf2d_2, cmap="RdBu_r", scalars="Edot", clim=(0,0.75))
#pl.add_arrows(surf2d_2.points, surf2d_2.point_data["V"], mag=0.3)

pl.add_mesh(surf2d_3, cmap="RdBu_r", scalars="Edot", clim=(0,0.75))
#pl.add_arrows(surf2d_3.points, surf2d_3.point_data["V"], mag=0.3)


pl.show()

# %%
stokes.F1.sym

# %%
0 / 0

# %%
0 / 0

# %%
