## pyvista helper routines


def mesh_to_pv_mesh(mesh):
    """Initialise pyvista engine from existing mesh"""

    # Required in notebooks
    import nest_asyncio

    nest_asyncio.apply()

    import os
    import shutil
    import tempfile
    import pyvista as pv

    pv.global_theme.background = "white"
    pv.global_theme.antialiasing = True
    pv.global_theme.jupyter_backend = "client"
    pv.global_theme.smooth_shading = True
    pv.global_theme.smooth_shading = True
    pv.global_theme.camera["viewup"] = [0.0, 1.0, 0.0]
    pv.global_theme.camera["position"] = [0.0, 0.0, 5.0]

    with tempfile.TemporaryDirectory() as tmp:
        
        if type(mesh)==str: # reading msh file directly
            vtk_filename = os.path.join(tmp, "tmpMsh.msh")
            shutil.copyfile(mesh, vtk_filename)
        else: # reading mesh by creating vtk 
            vtk_filename = os.path.join(tmp, "tmpMsh.vtk")
            mesh.vtk(vtk_filename)

        try:
            pv.global_theme.jupyter_backend = "trame"
            pvmesh = pv.read(vtk_filename)
        except RuntimeError:
            pv.global_theme.jupyter_backend = "panel"
            pvmesh = pv.read(vtk_filename)

    return pvmesh


def coords_to_pv_coords(coords):
    """pyvista requires 3D coordinates / vectors - fix if they are 2D"""

    return vector_to_pv_vector(coords)


def vector_to_pv_vector(vector):
    """pyvista requires 3D coordinates / vectors - fix if they are 2D"""

    import numpy as np

    if vector.shape[1] == 3:
        return vector
    else:
        vector3 = np.zeros((vector.shape[0], 3))
        vector3[:, 0:2] = vector[:]

        return vector3


def swarm_to_pv_cloud(swarm):
    """swarm points to pyvista PolyData  object"""

    import numpy as np
    import pyvista as pv

    with swarm.access():
        points = np.zeros((swarm.data.shape[0], 3))
        points[:, 0] = swarm.data[:, 0]
        points[:, 1] = swarm.data[:, 1]
        if swarm.mesh.dim == 2:
            points[:, 2] = 0.0
        else:
            points[:, 2] = swarm.data[:, 2]

    point_cloud = pv.PolyData(points)

    return point_cloud


def meshVariable_to_pv_cloud(meshVar):
    """meshVariable point locations to pyvista PolyData object"""

    import numpy as np
    import pyvista as pv

    points = np.zeros((meshVar.coords.shape[0], 3))
    points[:, 0] = meshVar.coords[:, 0]
    points[:, 1] = meshVar.coords[:, 1]

    if meshVar.mesh.dim == 2:
        points[:, 2] = 0.0
    else:
        points[:, 2] = meshVar.coords[:, 2]

    point_cloud = pv.PolyData(points)

    return point_cloud


def scalar_fn_to_pv_points(pv_mesh, uw_fn, dim=None):
    """evaluate uw scalar function at mesh/cloud points"""

    import underworld3 as uw

    if dim is None:
        if pv_mesh.points[:, 2].max() - pv_mesh.points[:, 2].min() < 1.0e-6:
            dim = 2
        else:
            dim = 3

    coords = pv_mesh.points[:, 0:dim]
    scalar_values = uw.function.evalf(uw_fn, coords)

    return scalar_values


def vector_fn_to_pv_points(pv_mesh, uw_fn, dim=None):
    """evaluate uw vector function at mesh/cloud points"""

    import numpy as np
    import underworld3 as uw

    dim = uw_fn.shape[1]
    if dim != 2 and dim != 3:
        print(f"UW vector function should have dimension 2 or 3")

    coords = pv_mesh.points[:, 0:dim]
    vector_values = np.zeros_like(pv_mesh.points)

    for i in range(0, dim):
        vector_values[:, i] = uw.function.evalf(uw_fn[i], coords)

    return vector_values


# def vector_fn_to_pv_arrows(coords, uw_fn, dim=None):
#     """evaluate uw vector function on point cloud"""

#     import numpy as np

#     dim = uw_fn.shape[1]
#     if dim != 2 and dim != 3:
#         print(f"UW vector function should have dimension 2 or 3")

#     coords = pv_mesh.points[:, 0 : dim - 1]
#     vector_values = np.zeros_like(coords)

#     for i in range(0, dim):
#         vector_values[:, i] = uw.function.evalf(uw_fn[i], coords)

#     return vector_values
