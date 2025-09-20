# %% [markdown]
"""
# 🔬 Compression AnisotropicFault

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
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# ## Anisotropic, embedded-fault model

# %%
import nest_asyncio
nest_asyncio.apply()


# %%
import os

os.environ["UW_TIMING_ENABLE"] = "1"


import petsc4py
import underworld3 as uw
from underworld3 import timing


import numpy as np
import sympy

# %% jupyter={"source_hidden": true}
# def distance_pointcloud_linesegment(p, a, b):
#     """
#     p - numpy array of points in 3D
#     a, b - triangle points (numpy 1x3 arrays)

#     returns:
#         numpy array of distances from each of the points to the nearest point within the triangle (0 if in the plane, within the triangle)
#     """

#     ab = (b - a).reshape(1, 3)
#     ap = (p - a) # .reshape(1, 3)
#     bp = (p - b) # .reshape(1, 3)

#     def dot(v1, v2):
#         d = v1[:, 0] * v2[:, 0] + v1[:, 1] * v2[:, 1] + v1[:, 2] * v2[:, 2]
#         return d

#     ab_norm = np.sqrt(dot(ab, ab))
#     adotp = dot(ab, ap) / ab_norm

#     P = adotp / ab_norm

#     # Three different cases:
#     #     P < 0 return distance p to a
#     #     P > 1 return distance p to b
#     #     0 <=  P <= 1 return perpendicular distance

#     d = np.sqrt(dot(ap,ap) - adotp**2)

#     print(P, flush=True)

#     mask = P<0
#     d[mask] = np.sqrt(dot(ap[mask], ap[mask]))

#     mask = P>1
#     d[mask] = np.sqrt(dot(bp[mask], bp[mask]))

#     return d

# %%
a = np.array((0.0, 0.0, 0.0))
b = np.array((1.0, 1.0, 0.0))
p = np.array(((3, 3, 1.0), (0, 0, 0), (-1, 2, 0)))
p = np.array(((3, 3, 1.0)))

# %%
uw.utilities.distance_pointcloud_linesegment(p, a, b)

# %%
mesh = uw.meshing.UnstructuredSimplexBox(
    minCoords=(-2, 0), maxCoords=(2, 1), cellSize=0.025, qdegree=3, regular=False
)

v_soln = uw.discretisation.MeshVariable("U", mesh, mesh.dim, degree=2)
p_soln = uw.discretisation.MeshVariable("P", mesh, 1, degree=1)
strain_rate_inv2 = uw.discretisation.MeshVariable("eps", mesh, 1, degree=1)
fault_dist = uw.discretisation.MeshVariable("df", mesh, 1, degree=2, continuous=False)
fault_norm = uw.discretisation.MeshVariable("nf", mesh, mesh.dim, degree=2, continuous=False)


# fault_swarm = uw.swarm.Swarm(mesh)
# fault_swarm_idx = uw.swarm.SwarmVariable(
#     "IDX",
#     fault_swarm,
#     dtype=int,
#     vtype=uw.VarType.SCALAR,
#     proxy_degree=1,
#     proxy_continuous=False,
# )
# fault_swarm_dist = uw.swarm.SwarmVariable(
#     r"d_f",
#     fault_swarm,
#     dtype=float,
#     vtype=uw.VarType.SCALAR,
#     proxy_degree=1,
#     proxy_continuous=False,
# )
# fault_swarm_norm = uw.swarm.SwarmVariable(
#     r"n_f",
#     fault_swarm,
#     dtype=float,
#     vtype=uw.VarType.VECTOR,
#     proxy_degree=1,
#     proxy_continuous=False,
#     varsymbol=r"\hat{\mathbf{n}}_f",
# )

# fault_swarm.populate(fill_param=4)

# %%
# Map properties from fault(s) to swarm

# Faults:

segments = 30 * 3
fault_segments = np.zeros(shape=(segments, 5))
fault_segment_normals = np.zeros(shape=(segments, 2))
fault_segment_centroids = np.zeros(shape=(segments, 2))

x_locations = np.linspace(0, 1.0, 31)
for i in range(0,30):
    y = lambda x: x ** 2
    x_a = x_locations[i]
    x_b = x_locations[i + 1]
    y_a = y(x_a)
    y_b = y(x_b)

    fault_segments[i, 0] = x_a
    fault_segments[i, 1] = y_a
    fault_segments[i, 2] = x_b
    fault_segments[i, 3] = y_b
    fault_segments[i, 4] = 0  # id

    fault_segment_normals[i, 0] = y_b - y_a
    fault_segment_normals[i, 1] = x_a - x_b

    fault_segment_centroids[i, 0] = (x_a + x_b) / 2
    fault_segment_centroids[i, 1] = (y_a + y_b) / 2

## Repeat 

x_locations = np.linspace(0.8, 1.8,  31)
for i in range(0,30):
    ii = 30+i
    y = lambda x: (x-0.8) ** 2
    x_a = x_locations[i]
    x_b = x_locations[i + 1]
    y_a = y(x_a)
    y_b = y(x_b)

    fault_segments[ii, 0] = x_a
    fault_segments[ii, 1] = y_a
    fault_segments[ii, 2] = x_b
    fault_segments[ii, 3] = y_b
    fault_segments[ii, 4] = 1  # id

    fault_segment_normals[ii, 0] = y_b - y_a
    fault_segment_normals[ii, 1] = x_a - x_b

    fault_segment_centroids[ii, 0] = (x_a + x_b) / 2
    fault_segment_centroids[ii, 1] = (y_a + y_b) / 2
## Repeat 

x_locations = np.linspace(-0.8, 0.2,  31)
for i in range(0,30):
    ii = 60+i
    y = lambda x: (x+0.8) ** 2
    x_a = x_locations[i]
    x_b = x_locations[i + 1]
    y_a = y(x_a)
    y_b = y(x_b)

    fault_segments[ii, 0] = x_a
    fault_segments[ii, 1] = y_a
    fault_segments[ii, 2] = x_b
    fault_segments[ii, 3] = y_b
    fault_segments[ii, 4] = 1  # id

    fault_segment_normals[ii, 0] = y_b - y_a
    fault_segment_normals[ii, 1] = x_a - x_b

    fault_segment_centroids[ii, 0] = (x_a + x_b) / 2
    fault_segment_centroids[ii, 1] = (y_a + y_b) / 2

fault_segment_normals /= np.sqrt(
    fault_segment_normals[:, 0] ** 2 + fault_segment_normals[:, 1] ** 2
).reshape(-1, 1)



# %%
fault_centroid_index = uw.kdtree.KDTree(fault_segment_centroids)

with mesh.access():
    point_closest_seg, point_seg_sqdistance, _ = (
        fault_centroid_index.find_closest_point(fault_dist.coords)
    )

point_seg_c_distance = np.sqrt(point_seg_sqdistance)
point_closest_fault = fault_segments[point_closest_seg, 4]

with mesh.access(fault_dist, fault_norm):
    fault_norm.data[...] = fault_segment_normals[point_closest_seg, ...]
    fault_dist.data[:, 0] = point_seg_c_distance[...]

    # True distance ... takes time, so we only do the closest points

    # close_points = np.where(fault_dist.data[:, 0] < mesh.get_min_radius() * 8)[0]
    # close_segments = fault_segments[point_closest_seg[close_points]]

    # for i in range(close_points.shape[0]):
    #     pt = close_points[i]
    #     sg = close_segments[i]
    #     dt = uw.utilities.distance_pointcloud_linesegment(
    #         fault_dist.coords[pt].reshape(1, 2), sg[0:2], sg[2:4]
    #     )[0]

    #     fault_dist.data[pt, 0] = dt


# %%

# %%
## Similar but for a swarm variable to register distance to faults - can also record index properly

# with fault_swarm.access():
#     swarm_closest_seg, swarm_seg_sqdistance, _ = (
#         fault_centroid_index.find_closest_point(fault_swarm.data)
#     )

# swarm_seg_c_distance = np.sqrt(swarm_seg_sqdistance)
# swarm_closest_fault = fault_segments[swarm_closest_seg, 4]

# with fault_swarm.access(fault_swarm_dist, fault_swarm_idx, fault_swarm_norm):
#     fault_swarm_norm.data[...] = fault_segment_normals[swarm_closest_seg, ...]
#     fault_swarm_idx.data[:, 0] = fault_segments[swarm_closest_seg, 4]
#     fault_swarm_dist.data[:, 0] = swarm_seg_c_distance[...]

#     # True distance ... takes time, so we only do the closest points

#     close_points = np.where(fault_swarm_dist.data[:, 0] < mesh.get_min_radius() * 4)[0]
#     close_segments = fault_segments[swarm_closest_seg[close_points]]

#     for i in range(close_points.shape[0]):
#         pt = close_points[i]
#         sg = close_segments[i]
#         dt = uw.utilities.distance_pointcloud_linesegment(
#             fault_swarm._particle_coordinates.data[pt].reshape(1, 2), sg[0:2], sg[2:4]
#         )[0]

#         fault_swarm_dist.data[pt, 0] = dt

# %%
# ## Visualise this

# import pyvista as pv
# import underworld3.visualisation as vis

# with fault_swarm.access():
#     close_points = np.where(fault_swarm_dist.data < mesh.get_min_radius() * 2)[0]

#     points = np.zeros((close_points.shape[0], 3))
#     points[:, 0] = fault_swarm.data[close_points, 0]
#     points[:, 1] = fault_swarm.data[close_points, 1]
#     points[:, 2] = 0

#     arrows = np.zeros((close_points.shape[0], 3))
#     arrows[:, 0] = fault_swarm_norm.data[close_points, 0]
#     arrows[:, 1] = fault_swarm_norm.data[close_points, 1]
#     arrows[:, 2] = 0

#     close_swarm_cloud = pv.PolyData(points)
#     close_swarm_cloud.point_data["dist"] = fault_swarm_dist.data[close_points, 0]
#     close_swarm_cloud.point_data["norm"] = fault_swarm_norm.data[close_points]


# pv_mesh = vis.mesh_to_pv_mesh(mesh)


# pl = pv.Plotter(window_size=[1000, 1000])

# pl.add_points(
#     close_swarm_cloud,
#     scalars="dist",
#     cmap="rainbow",
#     # opacity="opacity",
#     style="points_gaussian",
#     point_size=5,
#     # clim=(0,1),
#     render_points_as_spheres=False,
#     show_scalar_bar=False,
# )

# pl.add_arrows(points, arrows, 0.05, color="Grey", opacity=0.3)

# pl.add_mesh(
#     pv_mesh,
#     style="wireframe",
#     use_transparency=False,
#     show_scalar_bar=False,
#     opacity=0.5,
# )


# pl.show(jupyter_backend="trame")


# %%
stokes = uw.systems.Stokes(
    mesh,
    velocityField=v_soln,
    pressureField=p_soln,
    verbose=False)

stokes.constitutive_model = uw.constitutive_models.TransverseIsotropicFlowModel
stokes.constitutive_model.Parameters.eta_0 = 1 
 
stokes.constitutive_model.Parameters.eta_1 = sympy.Piecewise(
    (0.0001, fault_dist.sym[0] < mesh.get_min_radius() * 5), (1.0, True)
)

stokes.constitutive_model.Parameters.director = fault_norm.sym

# stokes.constitutive_model.Parameters.director = sympy.Matrix((1/sympy.sqrt(2),1/sympy.sqrt(2)))
# th =  sympy.pi / 4
# stokes.constitutive_model.Parameters.director = sympy.Matrix((sympy.cos(th),sympy.sin(th)))

# stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
# stokes.constitutive_model.Parameters.shear_viscosity_0 = 1

stokes.penalty = 0.1

stokes.petsc_options["ksp_monitor"] = None
stokes.petsc_options["snes_monitor"] = None
stokes.petsc_options["snes_atol"] = 1.0e-4

stokes.petsc_options["fieldsplit_velocity_ksp_type"] = "cg"
stokes.petsc_options["fieldsplit_velocity_pc_type"] = "mg"

stokes.petsc_options["fieldsplit_pressure_ksp_type"] = "gmres"
stokes.petsc_options["fieldsplit_pressure_pc_type"] = "mg"

# stokes.bodyforce = -10 * mesh1.CoordinateSystem.unit_j

# Velocity boundary conditions

# stokes.add_dirichlet_bc((0.0, 0.0), "Hump", (0, 1))
# stokes.add_dirichlet_bc((vx_ps, vy_ps), ["top", "bottom", "left", "right"], (0, 1))

stokes.add_dirichlet_bc((1.0, 0.0), "Left")
stokes.add_dirichlet_bc((0.0, 0.0), "Right")
stokes.add_dirichlet_bc((None, 0.0), "Bottom")


# %%
nodal_strain_rate_inv2 = uw.systems.Projection(mesh, strain_rate_inv2)
nodal_strain_rate_inv2.uw_function = stokes.Unknowns.Einv2
nodal_strain_rate_inv2.smoothing = 0.0e-3
nodal_strain_rate_inv2.petsc_options["ksp_monitor"] = None
nodal_strain_rate_inv2.petsc_options["snes_monitor"] = None

# nodal_strain_rate_inv2.petsc_options.delValue("ksp_monitor")


# %%
stokes.solve()
nodal_strain_rate_inv2.solve()

# %%
# check the mesh if in a notebook / serial

if uw.mpi.size == 1:

    import pyvista as pv
    import underworld3.visualisation as vis

    pvmesh = vis.mesh_to_pv_mesh(mesh)
    pvmesh.point_data["P"] = vis.scalar_fn_to_pv_points(pvmesh, p_soln.sym)
    pvmesh.point_data["Edot"] = vis.scalar_fn_to_pv_points(pvmesh, strain_rate_inv2.sym)
    # pvmesh.point_data["Visc"] = vis.scalar_fn_to_pv_points(pvmesh, node_viscosity.sym)
    # pvmesh.point_data["Str"] = vis.scalar_fn_to_pv_points(pvmesh, dev_stress_inv2.sym)
    # pvmesh.point_data["D"] = vis.scalar_fn_to_pv_points(pvmesh, fault_swarm_dist._meshVar.sym)
    pvmesh.point_data["D"] = vis.scalar_fn_to_pv_points(pvmesh, fault_dist.sym)
    pvmesh.point_data["V"] = vis.vector_fn_to_pv_points(pvmesh, v_soln.sym)
    pvmesh.point_data["Nm"] = vis.vector_fn_to_pv_points(pvmesh, fault_norm.sym)
    pvmesh.point_data["Vmag"] = vis.scalar_fn_to_pv_points(
        pvmesh, v_soln.sym.dot(v_soln.sym)
    )

    velocity_points = vis.meshVariable_to_pv_cloud(v_soln)
    velocity_points.point_data["V"] = vis.vector_fn_to_pv_points(
        velocity_points, v_soln.sym
    )

    # point sources at cell centres
    points = np.zeros((mesh._centroids.shape[0], 3))
    points[:, 0] = mesh._centroids[:, 0]
    points[:, 1] = mesh._centroids[:, 1]
    point_cloud = pv.PolyData(points)

    pvstream = pvmesh.streamlines_from_source(
        point_cloud, vectors="V", integration_direction="both", max_steps=100
    )

    pl = pv.Plotter(window_size=(1000, 750))

    # pl.add_arrows(
    #     velocity_points.points, velocity_points.point_data["V"], mag=0.1, opacity=0.25
    # )

    # pl.add_arrows(
    #     pvmesh.points, pvmesh.point_data["Nm"], mag=0.1, opacity=0.75
    # )
    

    pl.add_mesh(
        pvmesh,
        cmap="coolwarm",
        edge_color="Black",
        show_edges=True,
        scalars="Edot",
        use_transparency=False,
        opacity=1.0,
        clim=[0.0, 6.0])


    pl.show()

# %%
0/0

# %%
